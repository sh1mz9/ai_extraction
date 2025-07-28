import uvicorn
import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List

# Import the core logic from your existing utility files
from utils import process_uploaded_files, create_vector_db
from llm_utils import get_llm_for_langchain, answer_question_hybrid

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Document Intelligence API",
    description="An API for extracting structured data and answering questions from documents.",
    version="1.0.0"
)

# --- In-Memory Session Storage ---
# In a production environment, you would replace this with a more persistent storage
# solution like Redis or a database.
SESSIONS = {}

# --- Pydantic Models for Request Bodies ---
class QueryRequest(BaseModel):
    session_id: str
    query: str

# --- API Endpoints ---

@app.get("/", tags=["General"])
async def read_root():
    """A simple health check endpoint."""
    return {"message": "Welcome to the AI Document Intelligence API!"}

@app.post("/process/", tags=["Document Processing"])
async def process_documents(files: List[UploadFile] = File(...)):
    """
    Uploads one or more documents, processes them, and prepares for Q&A.
    Returns a unique session_id to be used for subsequent queries.
    """
    session_id = str(uuid.uuid4())
    
    # Create a temporary directory for this session's files
    temp_dir = f"temp_{session_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        saved_files.append({"filename": file.filename, "path": file_path})

    try:
        # Process the files using your existing utility functions
        # We need to pass the full file objects to process_uploaded_files
        # Let's adapt this on the fly for the API context
        cached_texts = {}
        for f in saved_files:
            # Re-create a simple object that process_uploaded_files can use
            class SimpleUploadedFile:
                def __init__(self, name, path):
                    self.name = name
                    self.path = path
                def getvalue(self):
                    with open(self.path, "rb") as file_handle:
                        return file_handle.read()

            # The utils function expects a list of these objects
            processed_text = process_uploaded_files([SimpleUploadedFile(f['filename'], f['path'])])
            cached_texts.update(processed_text)

        all_texts = list(cached_texts.values())
        vector_db = create_vector_db(all_texts)

        if not vector_db:
            raise HTTPException(status_code=500, detail="Failed to create vector database from documents.")

        # Store the retriever and LLM instance in our session cache
        SESSIONS[session_id] = {
            "retriever": vector_db.as_retriever(),
            "llm": get_llm_for_langchain()
        }

    finally:
        # Clean up the temporary files and directory
        for f in saved_files:
            os.remove(f['path'])
        os.rmdir(temp_dir)

    return {"session_id": session_id, "message": f"Successfully processed {len(files)} files."}


@app.post("/query/", tags=["Question Answering"])
async def query_documents(request: QueryRequest):
    """
    Asks a question against a set of processed documents using a session_id.
    """
    session_id = request.session_id
    query = request.query

    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Please process documents first.")

    session_data = SESSIONS[session_id]
    retriever = session_data.get("retriever")
    llm = session_data.get("llm")

    if not retriever or not llm:
        raise HTTPException(status_code=500, detail="Session data is incomplete.")

    # Use the hybrid Q&A function to get the answer
    answer = answer_question_hybrid(query, retriever, llm)

    return {"session_id": session_id, "query": query, "answer": answer}

# --- To run this API, save it as api.py and run in your terminal: ---
# uvicorn api:app --reload
