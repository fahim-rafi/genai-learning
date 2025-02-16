import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from src.prompt import CHAT_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE
import os
import tiktoken
import time

st.markdown(
    "<h2 style='text-align: center; color: #C41E3A; font-family: Arial;'>Captain's Chat ☠️</h2>",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Chat with Captain", "Document Q&A"])

chat_template = CHAT_PROMPT_TEMPLATE
chat_prompt = ChatPromptTemplate.from_template(chat_template)
model = OllamaLLM(model="llama3.1:8b")
chat_chain = chat_prompt | model

def get_optimal_chunk_size(text, target_size=512):
    """Determine optimal chunk size based on document content"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") 
        tokens = len(encoding.encode(text))
        avg_chars_per_token = len(text) / tokens
        return int(target_size * avg_chars_per_token)
    except:
        return 1000  # fallback size

@st.cache_resource
def initialize_rag():
    with st.status("🏴‍☠️ Initializing document processing...", expanded=True) as status:
        data_dir = "./data/"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        st.write("⚓ Scanning for PDF documents...")
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        if not pdf_files:
            status.update(label="No PDF documents found!", state="error")
            return None
        
        st.write(f"📚 Found {len(pdf_files)} PDF documents")
        
        documents = []
        sample_text = ""
        for pdf in pdf_files:
            st.write(f"📄 Loading {pdf}...")
            loader = PyPDFLoader(os.path.join(data_dir, pdf))
            docs = loader.load()
            documents.extend(docs)
            if not sample_text and docs:
                sample_text = docs[0].page_content
        
        st.write(f"📋 Loaded {len(documents)} pages in total")
        
        st.write("📏 Optimizing chunk size...")
        chunk_size = get_optimal_chunk_size(sample_text)
        st.write(f"Determined optimal chunk size: {chunk_size} characters")
        
        st.write("✂️ Splitting documents into chunks...")
        
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),  
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        
        character_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),
            length_function=len,
        )
        
        # apply different splitting strategies based on content
        splits = []
        for doc in documents:
            if doc.page_content.count("\n") > doc.page_content.count(". "):
                doc_splits = recursive_splitter.split_documents([doc])
            else:
                doc_splits = character_splitter.split_documents([doc])
            splits.extend(doc_splits)
        
        st.write(f"🔨 Created {len(splits)} text chunks")
        
        st.write("🧭 Creating embeddings and vector store...")
        with st.spinner("This might take a few moments..."):
            embeddings = OllamaEmbeddings(
                model="llama3.1:8b"
            )
            vectorstore = FAISS.from_documents(
                splits, 
                embeddings,
                normalize_L2=True  
            )
        st.write("💾 Vector store created successfully")
        
        base_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 6, 
                "score_threshold": 0.01,  
                "fetch_k": 50,  
            }
        )
        
        compressor = LLMChainExtractor.from_llm(model)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
            return_source_documents=True  # for debugging
        )
        
        # rag chain settings
        rag_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,  # for debugging
            chain_type_kwargs={
                "prompt": ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE),
                "verbose": True
            }
        )
        
        status.update(label="✅ RAG system ready!", state="complete")
        return rag_chain

streaming_enabled = st.sidebar.toggle("Enable streaming responses", value=True)

def display_response(response_container, response_text):
    """Display response with or without streaming based on user preference"""
    if streaming_enabled:
        placeholder = ""
        for char in response_text:
            placeholder += char
            response_container.markdown(placeholder + "▌", unsafe_allow_html=True)
            time.sleep(0.001)
        response_container.markdown(placeholder, unsafe_allow_html=True)
    else:
        response_container.markdown(response_text, unsafe_allow_html=True)

def display_responses_parallel(containers_and_texts):
    """Display multiple responses"""
    for container, text in containers_and_texts:
        display_response(container, text)

with tab1:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Ahoy matey! Welcome aboard! What brings ye to these waters?"}
        ]

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Send a message to the Captain..."):
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            response_container = st.empty()
            response = chat_chain.invoke({"question": user_input})
            display_response(response_container, response)
        
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

with tab2:
    st.markdown("### Ask Questions About Your Documents")
    
    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])
    if uploaded_file:
        if not os.path.exists("./data"):
            os.makedirs("./data")
        with open(os.path.join("./data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        st.rerun()  
    
    with st.spinner("Loading RAG system..."):
        rag_chain = initialize_rag()
    
    if rag_chain is None:
        st.warning("No PDF documents found in ./data/. Please upload a PDF file to get started!")
        st.info("You can upload your documents using the file uploader above.")
    else:
        compare_mode = st.toggle("Compare responses (with/without context)")
        
        data_dir = "./data/"
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        with st.expander("📚 Document Information"):
            st.write(f"Number of documents loaded: {len(pdf_files)}")
            st.write("Available documents:")
            for pdf in pdf_files:
                st.write(f"- {pdf}")
        
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = [
                {"role": "assistant", "content": "Ahoy! I'm ready to answer questions about yer documents, matey! What would ye like to know?"}
            ]

        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if rag_input := st.chat_input("Ask about your documents..."):
            st.session_state.rag_messages.append({"role": "user", "content": rag_input})
            with st.chat_message("user"):
                st.markdown(rag_input)
            
            if compare_mode:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Baseline Response (No Context)")
                    baseline_container = st.empty()
                
                with col2:
                    st.markdown("#### Augmented Response (With Context)")
                    augmented_container = st.empty()
                
                baseline = model.invoke(rag_input)
                augmented = rag_chain.invoke({"query": rag_input})["result"]
                
                display_responses_parallel([
                    (baseline_container, baseline),
                    (augmented_container, augmented)
                ])
                
                response = augmented
            
            else:
                with st.chat_message("assistant"):
                    response_container = st.empty()
                    result = rag_chain.invoke({"query": rag_input})
                    response = result["result"]
                    
                    display_response(response_container, response)
                    
                    with st.expander("View Source Documents"):
                        if "source_documents" in result:
                            for i, doc in enumerate(result["source_documents"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(f"```\n{doc.page_content}\n```")
                        else:
                            st.write("No source documents available")
            
            st.session_state.rag_messages.append({"role": "assistant", "content": response})
