import streamlit as st
import pandas as pd

# Import utility functions from other modules
from utils import process_uploaded_files, create_vector_db
from llm_utils import (
    get_llm_for_langchain, # Renamed for clarity
    extract_structured_data,
    extract_new_column,
    create_retrieval_qa_chain
)

# --- App Configuration ---
st.set_page_config(page_title="AI Document Intelligence Platform", layout="wide")

st.title("AI-Powered Document Intelligence and Q&A")
st.write("Upload your documents, and I'll help you extract structured data and answer your questions.")

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'db' not in st.session_state:
    st.session_state.db = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'cached_texts' not in st.session_state:
    st.session_state.cached_texts = {}
if 'llm' not in st.session_state:
    # This LLM instance is now only for the LangChain Q&A part
    st.session_state.llm = get_llm_for_langchain()
if 'answer' not in st.session_state:
    st.session_state.answer = None
if 'query' not in st.session_state:
    st.session_state.query = ""

# --- Sidebar for File Upload ---
st.sidebar.header("1. Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload your files (PDF, DOCX, TXT, XLSX, PNG, JPG)",
    type=["pdf", "docx", "txt", "xlsx", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents... This may take a moment."):
            st.session_state.cached_texts = process_uploaded_files(uploaded_files)
            
            extracted_data_list = []
            for file_name, text in st.session_state.cached_texts.items():
                # This function no longer needs the llm instance passed to it
                structured_data = extract_structured_data(text)
                structured_data["File Name"] = file_name
                extracted_data_list.append(structured_data)
            
            st.session_state.df = pd.DataFrame(extracted_data_list)

            all_texts = list(st.session_state.cached_texts.values())
            st.session_state.db = create_vector_db(all_texts)
            
            if st.session_state.db:
                retriever = st.session_state.db.as_retriever()
                # The Q&A chain still uses the LangChain LLM wrapper
                st.session_state.qa_chain = create_retrieval_qa_chain(retriever, st.session_state.llm)

        st.sidebar.success("Documents processed successfully!")
        st.rerun()

# --- Main Page Content ---

st.header("2. Extracted Structured Data")
if st.session_state.df is not None and not st.session_state.df.empty:
    st.dataframe(st.session_state.df, use_container_width=True)

    st.subheader("Add a New Column")
    new_column_query = st.text_input(
        "What information do you want to extract as a new column?",
        key="new_col_input"
    )
    if st.button("Add Column"):
        if new_column_query and st.session_state.cached_texts:
            with st.spinner(f"Extracting '{new_column_query}'..."):
                new_column_data = []
                for file_name in st.session_state.df["File Name"]:
                    text = st.session_state.cached_texts.get(file_name, "")
                    # This function also no longer needs the llm instance
                    extracted_value = extract_new_column(text, new_column_query)
                    new_column_data.append(extracted_value)
                
                st.session_state.df[new_column_query] = new_column_data
            st.rerun()
        else:
            st.warning("Please enter a query for the new column.")
else:
    st.info("Upload and process documents to see the extracted data.")

st.header("3. Ask Me Anything")
if st.session_state.qa_chain:
    st.session_state.query = st.text_input(
        "Ask a question about the documents:",
        key="qa_input"
    )
    if st.session_state.query:
        with st.spinner("Searching for the answer..."):
            try:
                response = st.session_state.qa_chain.invoke({"input": st.session_state.query})
                st.session_state.answer = response.get('answer', "No answer found in the context.")
                st.write(st.session_state.answer)
            except Exception as e:
                st.error(f"An error occurred during question answering: {e}")
else:
    st.info("Upload and process documents to start the Q&A.")

if st.session_state.answer:
    st.header("4. Feedback")
    feedback = st.radio("Was this answer helpful?", ("Yes", "No"), key="feedback_radio")
    feedback_text = ""
    if feedback == "No":
        feedback_text = st.text_area(
            "Please provide your feedback to help me improve:",
            key="feedback_text"
        )
    
    if st.button("Submit Feedback"):
        log_path = "feedback_log.txt"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Query: {st.session_state.query}\n")
            f.write(f"Answer: {st.session_state.answer}\n")
            f.write(f"Helpful: {feedback}\n")
            if feedback_text:
                f.write(f"Comment: {feedback_text}\n")
            f.write("---\n")
        st.success(f"Thank you for your feedback! It has been saved to {log_path}")
        st.session_state.answer = None
        st.rerun()
