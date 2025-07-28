import streamlit as st
import os
import pytesseract
from PIL import Image
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)

# --- Text Extraction from different file types ---

def get_text_from_image(temp_file_path):
    """Extracts text from an image file using Tesseract OCR."""
    try:
        image = Image.open(temp_file_path)
        text = pytesseract.image_to_string(image)
        if not text.strip():
            st.warning(f"Tesseract OCR extracted no text from image: {os.path.basename(temp_file_path)}. Is Tesseract installed and in your system's PATH?")
        return text
    except Exception as e:
        st.error(f"Error processing image file: {e}")
        return ""

def get_text_from_document(temp_file_path, file_name):
    """Extracts text from document files (PDF, DOCX, TXT, XLSX)."""
    text = ""
    try:
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)
        elif file_name.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(temp_file_path)
        else: # .txt
            loader = TextLoader(temp_file_path)
        
        docs = loader.load()
        for doc in docs:
            text += doc.page_content + "\n"
    except Exception as e:
        st.error(f"Error processing document file {file_name}: {e}")
    
    if not text.strip():
        st.warning(f"No text was extracted from document: {file_name}.")
    return text

def process_uploaded_files(uploaded_files):
    """
    Takes a list of uploaded files, extracts text from them,
    and returns a dictionary mapping file names to their text content.
    """
    cached_texts = {}
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        # Create a temporary file to be processed by loaders
        temp_file_path = os.path.join(".", file_name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.sidebar.write(f"Processing {file_name}...")
        text = ""
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            text = get_text_from_image(temp_file_path)
        else:
            text = get_text_from_document(temp_file_path, file_name)
        
        cached_texts[file_name] = text
        os.remove(temp_file_path) # Clean up the temporary file
    return cached_texts

# --- Vector Database Creation ---

def create_vector_db(texts):
    """Creates a FAISS vector database from a list of texts."""
    if not texts or all(not t for t in texts):
        st.error("Cannot create vector database because no text was extracted from the documents.")
        return None
    try:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.create_documents(texts)
        # For future-proofing, one would move to `from langchain_ollama import OllamaEmbeddings`.
        embeddings = OllamaEmbeddings(model="smollm")
        db = FAISS.from_documents(documents, embeddings)
        return db
    except Exception as e:
        st.error(f"Failed to create vector database: {e}")
        return None
