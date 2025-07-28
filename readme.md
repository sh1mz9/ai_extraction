AI Document Intelligence Platform
This project is a prototype solution for the "AI Case Study: Social Support Application Workflow Automation". It provides a platform to upload multiple documents of various formats, automatically extract key information, and ask questions about the document set using a hybrid AI approach.

Features
Multi-Format Document Upload: Supports PDF, DOCX, TXT, XLSX, and image files (PNG, JPG).

Hybrid Data Extraction: Uses a robust strategy that combines regular expressions for common patterns (emails, phone numbers, dates) and a local LLM (smollm) for more complex, semantic data extraction.

Dynamic Column Addition: Users can specify new data points to extract on the fly, which are then added as new columns to the structured data view.

Hybrid Question-Answering: A smart Q&A system that first classifies the user's query. If it's a request for specific data (like a phone number), it uses regex for a precise answer. For general questions, it uses a powerful RAG (Retrieval-Augmented Generation) pipeline.

Dual Interfaces:

Streamlit Web App: An interactive, user-friendly interface for visual interaction.

FastAPI Backend: A robust API endpoint to allow for programmatic integration with other services.

Local First: Runs entirely on your local machine, using Ollama to serve the smollm language model.

Screenshots
Streamlit Web App Interface
The main user interface for uploading documents, viewing extracted data, and asking questions.

FastAPI Interactive Docs
The auto-generated API documentation page where you can test the /process/ and /query/ endpoints directly.

Project Structure
.
├── app.py              # The main Streamlit web application front-end.
├── api.py              # The FastAPI back-end server.
├── llm_utils.py        # All logic for interacting with the language model (direct Ollama calls and LangChain).
├── utils.py            # Helper functions for file processing, text extraction, and vector DB creation.
├── requirements.txt    # A list of all Python dependencies.
└── feedback_log.txt    # A file where user feedback is stored.

Setup and Installation
Follow these steps to set up and run the project on your local machine.

1. Prerequisites
Python 3.9+

Tesseract-OCR: You must have Google's Tesseract-OCR engine installed and available in your system's PATH. This is required for extracting text from images. You can find installation instructions here.

Ollama: The Ollama service must be installed and running on your machine. You can download it from ollama.com.

2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Install all the required Python packages using the requirements.txt file.

pip install -r requirements.txt

4. Download the Language Model
Pull the smollm model using the Ollama service. This only needs to be done once.

ollama pull smollm

How to Run
This project has two independent parts that can be run. Make sure your Ollama application is running before you start.

A. Running the Streamlit Web App (Visual Interface)
To use the interactive user interface:

Open your terminal in the project directory.

Run the following command:

streamlit run app.py

Your web browser will open with the application running.

B. Running the FastAPI Server (API Endpoint)
To expose the functionality as an API:

Open your terminal in the project directory.

Run the following command:

uvicorn api:app --reload

The API server will be running at http://127.0.0.1:8000.

How to Use the API
Once the FastAPI server is running, you can interact with it using any API client or by using the auto-generated documentation.

Open the Interactive Docs: Go to http://127.0.0.1:8000/docs in your web browser.

Process Documents:

Expand the POST /process/ endpoint.

Click "Try it out".

Click "Choose Files" to upload one or more documents.

Click "Execute".

Copy the session_id from the successful response body.

Ask Questions:

Expand the POST /query/ endpoint.

Click "Try it out".

Paste the session_id you received into the session_id field.

Type your question into the query field.

Click "Execute" to get your answer.