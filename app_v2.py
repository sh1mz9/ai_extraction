"""
app_v2.py - Enhanced Streamlit Document Intelligence Application V2
Fixes: Vector DB errors

"""

import streamlit as st
import pandas as pd
import time
from typing import Dict, Any

# Import our V2 utilities
from utils_v2 import process_uploaded_files, create_vector_db
from llm_utils_v2 import (
    get_llm_for_langchain,
    extract_structured_data,
    extract_new_column,
    create_retrieval_qa_chain,
    enhanced_qa_response
)

# --- App Configuration ---

st.set_page_config(
    page_title="AI Document Intelligence Platform V2",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 AI-Powered Document Intelligence Platform V2")
st.markdown("**Upload multiple document types simultaneously and extract structured data with AI-powered Q&A**")

# --- Session State Management ---

def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'df': None,
        'cached_texts': {},
        'db': None,
        'qa_chain': None,
        'llm': None,
        'processing_complete': False,
        'last_query': '',
        'last_answer': '',
        'initialization_error': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

# Initialize LLM if not done and no previous errors
if st.session_state.llm is None and st.session_state.initialization_error is None:
    with st.spinner("Initializing AI model..."):
        try:
            st.session_state.llm = get_llm_for_langchain()
            st.sidebar.success("✅ AI model initialized successfully!")
        except Exception as e:
            st.session_state.initialization_error = str(e)
            st.sidebar.error(f"❌ Failed to initialize AI model: {str(e)}")
            st.sidebar.info("💡 Please ensure Ollama is running and at least one model is available")

# --- Sidebar - File Upload Section ---

st.sidebar.header("📁 Document Upload")
st.sidebar.markdown("Supported formats: PDF, DOCX, TXT, XLSX, PNG, JPG, JPEG")

uploaded_files = st.sidebar.file_uploader(
    "Choose files to process:",
    type=["pdf", "docx", "txt", "xlsx", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="You can upload multiple files of different types simultaneously"
)

if uploaded_files:
    st.sidebar.info(f"📄 {len(uploaded_files)} file(s) selected")
    
    if st.sidebar.button("🔄 Process Documents", type="primary"):
        if st.session_state.llm is None:
            st.sidebar.error("❌ Cannot process documents: AI model not initialized")
            st.stop()
            
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            # Step 1: Extract text from files
            status_text.text("Extracting text from files...")
            progress_bar.progress(20)
            start_time = time.time()
            
            st.session_state.cached_texts = process_uploaded_files(uploaded_files)
            
            # Step 2: Extract structured data
            status_text.text("Extracting structured data...")
            progress_bar.progress(50)
            
            extracted_data_list = []
            for file_name, text in st.session_state.cached_texts.items():
                if text.strip():  # Only process files with content
                    structured_data = extract_structured_data(text)
                    structured_data["File Name"] = file_name
                    structured_data["Text Length"] = len(text)
                    extracted_data_list.append(structured_data)
            
            st.session_state.df = pd.DataFrame(extracted_data_list)
            
            # Step 3: Create vector database
            status_text.text("Creating searchable database...")
            progress_bar.progress(80)
            
            all_texts = [text for text in st.session_state.cached_texts.values() if text.strip()]
            if all_texts:
                st.session_state.db = create_vector_db(all_texts)
                if st.session_state.db:
                    retriever = st.session_state.db.as_retriever(search_kwargs={"k": 3})
                    st.session_state.qa_chain = create_retrieval_qa_chain(retriever, st.session_state.llm)
                    st.sidebar.success("✅ Q&A system ready!")
                else:
                    st.sidebar.warning("⚠️ Vector database creation failed")
            else:
                st.sidebar.warning("⚠️ No text content found in uploaded files")
            
            # Complete
            progress_bar.progress(100)
            processing_time = time.time() - start_time
            status_text.text(f"✅ Complete! ({processing_time:.1f}s)")
            st.session_state.processing_complete = True
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"❌ Processing failed: {str(e)}")
            st.sidebar.error(f"Error details: {type(e).__name__}")
            progress_bar.empty()
            status_text.empty()
            # Use st.stop() instead of return to properly halt execution
            st.stop()

# Display processing summary and system status
if st.session_state.processing_complete:
    st.sidebar.success("✅ Documents processed successfully!")
    if st.session_state.df is not None:
        st.sidebar.metric("Files Processed", len(st.session_state.df))
        st.sidebar.metric("Total Characters", sum(st.session_state.df["Text Length"]))

# System status indicators
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status:**")
status_col1, status_col2 = st.sidebar.columns(2)

with status_col1:
    if st.session_state.db is not None:
        st.success("✅ Vector DB: Ready")
    else:
        st.error("❌ Vector DB: Failed")

with status_col2:
    if st.session_state.qa_chain is not None:
        st.success("✅ Q&A Chain: Ready")
    else:
        st.error("❌ Q&A Chain: Failed")

# --- Main Content Area ---

# Tab layout for better organization
tab1, tab2, tab3 = st.tabs(["📊 Extracted Data", "➕ Custom Columns", "💬 Q&A Chat"])

# Tab 1: Extracted Data Display
with tab1:
    st.header("📊 Extracted Structured Data")
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", len(st.session_state.df))
        with col2:
            successful_extractions = len(st.session_state.df[st.session_state.df["Applicant Name"] != "N/A"])
            st.metric("Successful Extractions", successful_extractions)
        with col3:
            avg_length = st.session_state.df["Text Length"].mean()
            st.metric("Avg. Text Length", f"{avg_length:.0f} chars")
        with col4:
            total_chars = st.session_state.df["Text Length"].sum()
            st.metric("Total Content", f"{total_chars:,} chars")
        
        st.subheader("📋 Data Table")
        
        # Display options
        col1, col2 = st.columns([3, 1])
        with col1:
            show_text_length = st.checkbox("Show text length column", value=True)
        with col2:
            if st.button("📥 Download CSV"):
                csv = st.session_state.df.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name="extracted_data.csv",
                    mime="text/csv"
                )
        
        # Filter columns to display
        display_df = st.session_state.df.copy()
        if not show_text_length:
            display_df = display_df.drop("Text Length", axis=1, errors='ignore')
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
    else:
        st.info("👆 Upload and process documents to see extracted data here.")

# Tab 2: Custom Column Addition
with tab2:
    st.header("➕ Add Custom Data Columns")
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        st.markdown("**Extract additional information by describing what you need:**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            new_column_query = st.text_input(
                "What information do you want to extract?",
                placeholder="e.g., Policy Number, Social Security Number, Address, etc.",
                help="Describe the type of information you want to extract from the documents"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            add_column_btn = st.button("🔍 Extract Column", type="primary")
        
        if add_column_btn and new_column_query.strip():
            with st.spinner(f"🔍 Extracting '{new_column_query}' from all documents..."):
                try:
                    new_column_data = []
                    progress = st.progress(0)
                    
                    for idx, file_name in enumerate(st.session_state.df["File Name"]):
                        text = st.session_state.cached_texts.get(file_name, "")
                        extracted_value = extract_new_column(text, new_column_query)
                        new_column_data.append(extracted_value)
                        progress.progress((idx + 1) / len(st.session_state.df))
                    
                    st.session_state.df[new_column_query] = new_column_data
                    progress.empty()
                    st.success(f"✅ Successfully added column '{new_column_query}'!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error adding column: {str(e)}")
                    
        elif add_column_btn:
            st.warning("⚠️ Please enter a description for the column to extract.")
        
        # Show preview of current columns
        if st.session_state.df is not None:
            st.subheader("📋 Current Columns")
            cols = [col for col in st.session_state.df.columns if col not in ["Text Length"]]
            st.write(", ".join(cols))
    else:
        st.info("👆 Process documents first to add custom columns.")

# Tab 3: Q&A Interface  
with tab3:
    st.header("💬 Ask Questions About Your Documents")
    
    # System Status
    st.markdown("**System Status:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.db is not None:
            st.success("✅ Vector DB Ready")
        else:
            st.error("❌ Vector DB Not Ready")
    with col2:
        if st.session_state.qa_chain is not None:
            st.success("✅ Q&A Chain Ready")
        else:
            st.error("❌ Q&A Chain Not Ready")
    with col3:
        if st.session_state.llm is not None:
            st.success("✅ LLM Ready")
        else:
            st.error("❌ LLM Not Ready")
    
    if st.session_state.qa_chain is not None:
        st.markdown("**Ask any question about the content in your uploaded documents:**")
        
        # Query input
        user_query = st.text_input(
            "Your question:",
            placeholder="e.g., What is the total income across all applications?",
            help="Ask specific questions about the data in your documents"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🤔 Ask Question", type="primary")
        
        if ask_button and user_query.strip():
            with st.spinner("🔍 Searching through your documents..."):
                try:
                    # Debug: Show what we're querying
                    st.info(f"🔍 Processing query: '{user_query}'")
                    
                    answer = enhanced_qa_response(
                        user_query,
                        st.session_state.db.as_retriever(search_kwargs={"k": 3}),
                        st.session_state.llm
                    )
                    
                    st.session_state.last_query = user_query
                    st.session_state.last_answer = answer
                    st.success("✅ Query processed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")
                    st.error(f"Error type: {type(e).__name__}")
        
        # Display last Q&A
        if st.session_state.last_query and st.session_state.last_answer:
            st.subheader("💡 Answer")
            st.markdown(f"**Question:** {st.session_state.last_query}")
            st.markdown(f"**Answer:** {st.session_state.last_answer}")
            
            # Feedback section
            st.subheader("📝 Feedback")
            col1, col2 = st.columns(2)
            with col1:
                helpful = st.radio("Was this answer helpful?", ["Yes", "No"], horizontal=True)
            
            if helpful == "No":
                feedback_text = st.text_area("How can we improve this answer?")
                if st.button("📤 Submit Feedback"):
                    # Log feedback (implement your logging logic here)
                    st.success("Thank you for your feedback!")
        
        # Retry mechanism for failed Q&A setup
        if st.session_state.db is None or st.session_state.qa_chain is None:
            st.warning("❌ Q&A system not properly initialized. Please try reprocessing your documents.")
            
            if st.button("🔄 Retry Q&A Setup"):
                if st.session_state.cached_texts and st.session_state.llm:
                    with st.spinner("Recreating Q&A system..."):
                        try:
                            all_texts = [text for text in st.session_state.cached_texts.values() if text.strip()]
                            if all_texts:
                                st.session_state.db = create_vector_db(all_texts)
                                if st.session_state.db:
                                    retriever = st.session_state.db.as_retriever(search_kwargs={"k": 3})
                                    st.session_state.qa_chain = create_retrieval_qa_chain(retriever, st.session_state.llm)
                                    st.success("✅ Q&A system recreated successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to recreate vector database")
                            else:
                                st.error("❌ No text content available")
                        except Exception as e:
                            st.error(f"❌ Failed to recreate Q&A system: {str(e)}")
                else:
                    st.error("❌ Missing required components for Q&A setup")
        
        # Debug Information
        with st.expander("🐛 Debug Information"):
            st.markdown("**System Status:**")
            st.write(f"Documents processed: {bool(st.session_state.cached_texts)}")
            st.write(f"Number of texts: {len(st.session_state.cached_texts) if st.session_state.cached_texts else 0}")
            st.write(f"Vector DB created: {st.session_state.db is not None}")
            st.write(f"QA Chain created: {st.session_state.qa_chain is not None}")
            st.write(f"LLM initialized: {st.session_state.llm is not None}")
            
            if st.session_state.db is None:
                st.error("❌ Failed to create vector database")
    else:
        st.info("👆 Process documents first to enable Q&A functionality.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "**💡 Tips:** Upload multiple file types simultaneously • Use specific questions for better answers • "
    "Check extracted data accuracy before analysis"
)