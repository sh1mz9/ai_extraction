import ollama # Using the official Ollama library directly for extraction
import re
from langchain_ollama import OllamaLLM # Using LangChain's wrapper for Q&A
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_llm_for_langchain():
    """
    Initializes and returns the LangChain-compatible Ollama LLM instance.
    This is now ONLY used for the Q&A part.
    """
    return OllamaLLM(model="smollm", temperature=0)

def hybrid_extraction(text: str, field_name: str) -> str:
    """
    Attempts to extract information using regular expressions for common patterns first.
    If regex fails or is not applicable, it falls back to the language model.
    """
    # Normalize field_name for broader matching
    normalized_field = field_name.lower()
    
    # --- Regex Dictionary ---
    # Define regex patterns for common fields
    regex_patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\(?\b[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b',
        'date': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s\d{1,2},?\s\d{4}|\d{4}-\d{2}-\d{2})\b',
        'income': r'\$[0-9,]+(?:\.[0-9]{2})?\b',
        'zip': r'\b\d{5}(?:-\d{4})?\b', # Added ZIP code regex
        'emirates_id': r'\b\d{3}-\d{4}-\d{7}-\d{1}\b' # Added Emirates ID regex
    }

    # Check if a relevant regex pattern exists
    for key, pattern in regex_patterns.items():
        if key in normalized_field.replace(" ", "_"): # a bit of normalization
            match = re.search(pattern, text)
            if match:
                # If regex finds a match, return it immediately
                return match.group(0)

    # --- LLM Fallback ---
    # If no regex match is found, proceed with the LLM extraction
    prompt = f"""
Your task is to extract a specific piece of information from the text provided below.
The information to extract is: "{field_name}".

From the text, find the value for "{field_name}".
Respond with ONLY the value. Do not include the field name, any explanations, or any other text.
If you cannot find the information, respond with the single word "N/A".

Text:
---
{text}
---
Value: 
    """
    try:
        # Direct API call using the ollama library
        response = ollama.chat(
            model='smollm',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )
        content = response['message']['content']
        return content.strip()
    except Exception as e:
        print(f"Error calling Ollama directly for field '{field_name}': {e}")
        return f"N/A (Ollama Error)"

def extract_structured_data(text: str):
    """
    Extracts a predefined set of structured data by calling the hybrid extraction
    function for each field.
    """
    if not text or not text.strip():
        return {
            "Applicant Name": "N/A (No text extracted)",
            "Application Date": "N/A (No text extracted)",
            "Income": "N/A (No text extracted)",
            "Family Size": "N/A (No text extracted)",
        }

    fields_to_extract = ["Applicant Name", "Application Date", "Income", "Family Size"]
    extracted_data = {}

    for field in fields_to_extract:
        # Call the new hybrid extraction function
        value = hybrid_extraction(text, field)
        extracted_data[field] = value
    
    return extracted_data

def extract_new_column(text: str, column_query: str):
    """
    Uses the hybrid extraction method for new, user-defined columns.
    """
    return hybrid_extraction(text, column_query)

def create_retrieval_qa_chain(retriever, llm):
    """
    Creates the complete retrieval and question-answering chain using LangChain.
    This part of the logic remains the same.
    """
    question_answer_prompt = ChatPromptTemplate.from_template(
        """
        Answer the user's question based only on the context provided below:

        Context:
        {context}

        Question:
        {input}
        """
    )
    
    document_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- New Hybrid Q&A Logic ---

def classify_query_for_regex(query: str) -> str:
    """
    Classifies a user's query to see if it's asking for a piece of data
    that can be extracted with a regular expression. Uses direct ollama call.
    """
    prompt = f"""
Your task is to classify the user's question.
Does the question ask for an email address, a phone number, a specific date, an income/dollar amount, a ZIP code, or an Emirates ID number?

Respond with a single word from this list: ['email', 'phone', 'date', 'income', 'zip', 'emirates_id', 'general'].

Question: "{query}"
Classification:
    """
    try:
        response = ollama.chat(
            model='smollm',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )
        classification = response['message']['content'].strip().lower()
        # Clean up potential extra text from the model
        for key in ['email', 'phone', 'date', 'income', 'zip', 'emirates_id']:
            if key in classification:
                return key
        return 'general'
    except Exception as e:
        print(f"Error classifying query: {e}")
        return 'general'

def answer_question_hybrid(query: str, retriever, llm) -> str:
    """
    Answers a query using a hybrid regex-first, LLM-fallback approach.
    This is the new main entry point for the Q&A functionality.
    """
    regex_patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\(?\b[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b',
        'date': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s\d{1,2},?\s\d{4}|\d{4}-\d{2}-\d{2})\b',
        'income': r'\$[0-9,]+(?:\.[0-9]{2})?\b',
        'zip': r'\b\d{5}(?:-\d{4})?\b', # Added ZIP code regex
        'emirates_id': r'\b\d{3}-\d{4}-\d{7}-\d{1}\b' # Added Emirates ID regex
    }

    # Step 1: Classify the user's query
    classification = classify_query_for_regex(query)

    # Step 2: If classified for regex, try that first
    if classification != 'general':
        print(f"Query classified as '{classification}'. Attempting regex search on retrieved docs.")
        docs = retriever.get_relevant_documents(query)
        if docs:
            full_context = "\n".join([doc.page_content for doc in docs])
            pattern = regex_patterns.get(classification)
            if pattern:
                match = re.search(pattern, full_context)
                if match:
                    print(f"Regex found a match: {match.group(0)}")
                    return match.group(0)
        print("Regex did not find a match. Falling back to general Q&A.")

    # Step 3: Fallback to the general Q&A chain for general questions or failed regex
    qa_chain = create_retrieval_qa_chain(retriever, llm)
    response = qa_chain.invoke({"input": query})
    return response.get('answer', "No answer found in the context.")