import os
os.environ['HF_HOME'] = 'G:\models\huggingface'

import streamlit as st
import torch

torch.classes.__path__ = []

import PyPDF2
from transformers import pipeline
import subprocess  # Added to enable calling the Ollama CLI
from tqdm import tqdm  # Add this import at the top

from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_pdf(file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def generate_answer_ollama(query, context):
    """Generate answer using Ollama's local model by calling the command line interface.
    
    The prompt is now passed via standard input rather than as a command-line argument to avoid
    Windows' maximum command-line length limitations.
    """
    prompt = f"Answer the question based on the following context:\n{context}\nQuestion: {query}\nAnswer:"
    # Remove the --prompt argument, and pass the prompt via stdin.
    cmd = ["ollama", "run", "llama3.1:8b"]
    try:
        # The prompt text is now supplied as standard input.
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error generating answer with Ollama: {e}"


def main():
    st.title("RAG PDF Info Extraction App")
    st.write("Upload a PDF document (e.g., the document from [Municode](https://library.municode.com/tn/metro_government_of_nashville_and_davidson_county/codes/code_of_ordinances?nodeId=CD_TIT17ZO_CH17.08ZODILAUS_17.08.010ZODIES)) and ask questions about it.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    query = st.text_input("Enter your question:")
    generation_choice = st.radio("Choose generation model:", ("HuggingFace", "Ollama"))

    if uploaded_file is not None:
        with st.spinner('Extracting text from PDF...'):
            pdf_text = load_pdf(uploaded_file)
        st.subheader("PDF Text Preview")
        st.write(pdf_text[:500] + "...")

        # Use LangChain for splitting the text and building a vector index

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_text(pdf_text)
        st.write(f"Document split into {len(docs)} chunks.")

        st.info("Loading embedding model with LangChain...")
        embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        
        st.info("Building vector index with LangChain...")
        db = None
        progress_bar = st.progress(0)
        for i, doc in enumerate(docs):
            if db:
                db.add_texts([doc])
            else:
                db = FAISS.from_texts([doc], embedding)
            # Update progress bar
            progress = (i + 1) / len(docs)
            progress_bar.progress(progress)
        docsearch = db

        if query:
            if generation_choice == "HuggingFace":
                st.info("Generating answer using HuggingFace and LangChain...")
                hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
                llm = HuggingFacePipeline(pipeline=hf_pipeline)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="refine",
                    retriever=docsearch.as_retriever(search_kwargs={"k": 4})
                )
                result = qa_chain(query)
                answer = result['result']
            else:
                st.info("Retrieving context for Ollama generation using LangChain...")
                docs_similar = docsearch.similarity_search(query, k=6)
                # Extract page_content from Document objects
                context = "\n".join([doc.page_content for doc in docs_similar])
                st.info("Generating answer using Ollama...")
                answer = generate_answer_ollama(query, context)
            st.subheader("Answer")
            st.write(answer)


if __name__ == '__main__':
    main() 