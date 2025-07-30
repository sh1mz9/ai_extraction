"""
llm_utils_v2.py - Optimized hybrid extraction and Q&A with better caching and error handling
V2 improvements: Enhanced error handling, multiple LLM fallbacks, better regex patterns
"""

import re
import functools
from typing import Dict, Any, Optional

import ollama
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- LLM initialization with multiple fallbacks ---

def get_llm_for_langchain() -> OllamaLLM:
    """Initialize LangChain-compatible Ollama LLM with fallback models."""
    models_to_try = ["smollm", "llama3.2", "llama2", "qwen2"]
    
    for model in models_to_try:
        try:
            print(f"Attempting to initialize {model}...")
            llm = OllamaLLM(model=model, temperature=0, request_timeout=30)
            # Test the model with a simple query
            test_response = llm.invoke("Test")
            print(f"Successfully initialized {model}")
            return llm
        except Exception as e:
            print(f"Failed to initialize {model}: {e}")
            continue
    
    # If all models fail, raise an error
    raise RuntimeError("Could not initialize any Ollama model. Please ensure Ollama is running and at least one model is available.")

# --- Enhanced regex patterns for common data types ---

_COMPILED_REGEXES = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),
    "phone": re.compile(r'\(?\b[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
    "date": re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2},?\s+\d{4})\b'),
    "income": re.compile(r'\$\s?[\d,]+(?:\.\d{2})?\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "zip": re.compile(r'\b\d{5}(?:-\d{4})?\b'),
    "policy": re.compile(r'\b[A-Z]{2,4}\d{6,12}\b'),
    "account": re.compile(r'\b\d{8,16}\b'),
    "name": re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'),
    "address": re.compile(r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)', re.IGNORECASE),
}

@functools.lru_cache(maxsize=512)
def hybrid_extraction(text: str, field_name: str) -> str:
    """
    Fast extraction using regex first, then LLM fallback.
    Cached for performance with repeated queries.
    """
    if not text or not text.strip():
        return "N/A"
    
    # Clean and prepare text
    text_clean = text.strip()[:2000]  # Limit text length for processing
    field_lower = field_name.lower()
    
    # Try regex patterns first for speed
    for pattern_name, compiled_regex in _COMPILED_REGEXES.items():
        if pattern_name in field_lower:
            matches = compiled_regex.findall(text_clean)
            if matches:
                # Return the first match, or join multiple matches
                return matches[0] if len(matches) == 1 else "; ".join(matches[:3])
    
    # LLM fallback with optimized prompt
    try:
        prompt = f"""Extract the "{field_name}" from this text. Return only the specific value requested, or "N/A" if not found.

Text: {text_clean}

{field_name}:"""
        
        response = ollama.chat(
            model='smollm',
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0,
                'num_predict': 100,
                'top_p': 0.1,
                'repeat_penalty': 1.1
            }
        )
        
        content = response['message']['content'].strip()
        
        # Clean up common LLM response patterns
        if content:
            # Remove common prefixes
            for prefix in ["The ", "Answer: ", f"{field_name}: ", f"{field_name.lower()}: "]:
                if content.startswith(prefix):
                    content = content[len(prefix):].strip()
            
            # Handle common negative responses
            negative_responses = ["not found", "not available", "not mentioned", "not provided", "n/a", "none", ""]
            if content.lower() in negative_responses:
                return "N/A"
            
            return content
        
        return "N/A"
        
    except Exception as e:
        print(f"LLM extraction error for '{field_name}': {e}")
        return "N/A"

def extract_structured_data(text: str) -> Dict[str, str]:
    """Extract predefined structured fields from text."""
    if not text or not text.strip():
        return {
            "Applicant Name": "N/A",
            "Application Date": "N/A", 
            "Income": "N/A",
            "Family Size": "N/A",
            "Email": "N/A",
            "Phone": "N/A"
        }
    
    # Define standard fields to extract
    fields = [
        "Applicant Name",
        "Application Date", 
        "Income",
        "Family Size",
        "Email",
        "Phone"
    ]
    
    result = {}
    for field in fields:
        result[field] = hybrid_extraction(text, field)
    
    return result

def extract_new_column(text: str, column_query: str) -> str:
    """Extract user-defined field using hybrid approach."""
    return hybrid_extraction(text, column_query)

# --- Enhanced Q&A chain with better context handling ---

def create_retrieval_qa_chain(retriever, llm):
    """Create optimized retrieval Q&A chain with better prompts."""
    qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided document context.

Context from documents:
{context}

Question: {input}

Instructions:
1. Answer based ONLY on the information provided in the context above
2. If the answer is not in the context, clearly state "I don't have enough information in the provided documents to answer this question"
3. Be specific and cite relevant details from the context when available
4. If multiple documents contain relevant information, synthesize the information clearly
5. For numerical data, provide exact values when possible

Answer:""")

    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- Query classification for optimization ---

@functools.lru_cache(maxsize=256)
def classify_query_type(query: str) -> str:
    """Classify query type for performance optimization."""
    query_lower = query.lower()
    
    # Classification based on keywords
    classifications = {
        'email': ['email', 'mail', '@', 'e-mail'],
        'phone': ['phone', 'number', 'call', 'contact', 'mobile', 'telephone'],
        'date': ['date', 'when', 'time', 'day', 'month', 'year'],
        'income': ['income', 'salary', 'money', '$', 'earn', 'wage', 'pay'],
        'name': ['name', 'applicant', 'person', 'individual'],
        'address': ['address', 'location', 'street', 'city', 'state'],
        'summary': ['summarize', 'summary', 'overview', 'what is', 'tell me about'],
        'count': ['how many', 'count', 'number of', 'total'],
        'comparison': ['compare', 'difference', 'versus', 'vs', 'between']
    }
    
    for category, keywords in classifications.items():
        if any(keyword in query_lower for keyword in keywords):
            return category
    
    return 'general'

def enhanced_qa_response(query: str, retriever, llm) -> str:
    """Enhanced Q&A with query optimization and better error handling."""
    if not query or not query.strip():
        return "Please provide a specific question about your documents."
    
    try:
        # Classify query for potential optimization
        query_type = classify_query_type(query)
        
        # For specific data extraction queries, try direct approach first
        if query_type in ['email', 'phone', 'date', 'income', 'name']:
            try:
                docs = retriever.get_relevant_documents(query)
                if docs and len(docs) > 0:
                    # Combine top relevant documents
                    combined_text = "\n".join([doc.page_content for doc in docs[:3]])
                    if combined_text.strip():
                        # Try direct extraction
                        direct_result = hybrid_extraction(combined_text, query)
                        if direct_result and direct_result != "N/A":
                            return f"Based on the documents: {direct_result}"
            except Exception as e:
                print(f"Direct extraction attempt failed: {e}")
        
        # Full Q&A chain approach
        qa_chain = create_retrieval_qa_chain(retriever, llm)
        response = qa_chain.invoke({"input": query})
        
        answer = response.get('answer', "I couldn't process your question properly.")
        
        # Post-process answer for better quality
        if answer and not answer.startswith("I don't have enough information"):
            # Add context about source documents if helpful
            source_docs = response.get('context', [])
            if source_docs and hasattr(source_docs[0], 'metadata'):
                sources = set()
                for doc in source_docs[:2]:  # Limit to first 2 sources
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        sources.add(doc.metadata['source'])
                if sources:
                    answer += f"\n\n*Sources: {', '.join(sorted(sources))}*"
        
        return answer
        
    except Exception as e:
        error_msg = f"I encountered an error while processing your question: {str(e)}"
        print(f"Q&A Error: {e}")
        return error_msg