import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os
import time

# Set page config
st.set_page_config(page_title="PDF RAG Chat", layout="wide")

# Initialize session state for storing the QA chain and processing status
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None

def process_pdf(uploaded_file, status_container):
    progress_bar = status_container.progress(0)
    status_text = status_container.empty()
    timing_text = status_container.empty()
    start_time = time.time()
    temp_file_path = None
    
    def update_timing():
        elapsed = time.time() - start_time
        timing_text.write(f"⏱️ Time elapsed: {elapsed:.1f} seconds")
    
    try:
        # Create a temporary file to store the PDF
        status_text.write("📥 Creating temporary file...")
        progress_bar.progress(10)
        update_timing()
        
        # Create temporary file with a specific suffix
        temp_file_path = os.path.join(tempfile.gettempdir(), f"pdf_upload_{time.time()}.pdf")
        with open(temp_file_path, 'wb') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
        
        # Load PDF
        status_text.write("📖 Loading PDF content...")
        progress_bar.progress(20)
        update_timing()
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # Show document statistics
        status_container.write(f"📊 Document Statistics:")
        status_container.write(f"- Number of pages: {len(documents)}")
        progress_bar.progress(30)
        update_timing()
        
        # Split text into chunks
        status_text.write("✂️ Splitting document into chunks...")
        progress_bar.progress(40)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        status_container.write(f"- Number of chunks: {len(splits)}")
        progress_bar.progress(50)
        update_timing()
        
        # Create embeddings
        status_text.write("🧮 Initializing embedding model (nomic-embed-text)...")
        progress_bar.progress(60)
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        update_timing()
        
        # Create vector store with progress tracking
        status_text.write("🗄️ Creating vector store...")
        progress_text = status_container.empty()
        total_chunks = len(splits)
        
        # Custom embedding function to track progress
        def embed_with_progress(texts):
            for i, _ in enumerate(texts, 1):
                progress = (i / total_chunks) * 100
                progress_text.write(f"Embedding chunk {i}/{total_chunks} ({progress:.1f}%)")
                time.sleep(0.1)  # Small delay to prevent UI flicker
            return embeddings.embed_documents(texts)
        
        # Create FAISS index with progress tracking
        texts = [doc.page_content for doc in splits]
        embeddings_list = embed_with_progress(texts)
        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings_list)),
            embedding=embeddings,
        )
        
        status_container.write(f"- Vectors created: {len(splits)}")
        progress_bar.progress(80)
        update_timing()
        
        # Initialize LLM
        status_text.write("🤖 Initializing LLM (llama3.1)...")
        progress_bar.progress(90)
        llm = Ollama(model="llama3.1:8b")
        update_timing()
        
        # Create QA chain
        status_text.write("⚙️ Creating QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )
        
        progress_bar.progress(100)
        final_time = time.time() - start_time
        status_text.write(f"✅ Processing complete! Total time: {final_time:.1f} seconds")
        
        return qa_chain
            
    except Exception as e:
        status_text.error(f"❌ Error during processing: {str(e)}")
        progress_bar.progress(0)
        raise e
    
    finally:
        # Clean up temporary file in the finally block
        status_text.write("🧹 Cleaning up temporary files...")
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                status_text.warning(f"⚠️ Note: Could not delete temporary file: {str(e)}")

# Streamlit UI
st.title("📚 PDF Question Answering with RAG")

# Add sidebar for processing status
st.sidebar.title("Processing Status")
status_container = st.sidebar

# File upload
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    if st.session_state.qa_chain is None:
        try:
            st.session_state.qa_chain = process_pdf(uploaded_file, status_container)
            st.success("✨ PDF processed successfully! You can now ask questions about it.")
        except Exception as e:
            st.error("Failed to process PDF. Please try again.")
            st.session_state.qa_chain = None

    # Question input
    question = st.text_input("Ask a question about your PDF:")
    
    if question:
        with st.spinner("🤔 Thinking..."):
            # Get the answer
            result = st.session_state.qa_chain({"query": question})
            
            # Display the answer
            st.write("### 💡 Answer:")
            st.write(result["result"])
            
            # Display source documents
            st.write("### 📑 Sources:")
            for i, doc in enumerate(result["source_documents"]):
                with st.expander(f"Source {i + 1}"):
                    st.write(doc.page_content)

else:
    st.info("📤 Please upload a PDF file to get started!") 