import streamlit as st
import os
import asyncio
from document_processor import DocumentProcessor
from vector_database import VectorDatabase
from agentic_assistant import AgenticAssistant

from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import pandas as pd

# Ensure the event loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Function to extract API Key
def get_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    elif "api_key" in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    return ""

# Function to extract environment variables
def get_env_var(key, default_value=""):
    return st.secrets.get(key, default_value)

# Function to process uploaded files
def process_uploaded_file(uploaded_file):
    """Extract content from various file formats for immediate chat use."""
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    
    if file_type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    
    elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
    
    elif file_type in ["text/plain", "text/markdown"]:
        text = uploaded_file.read().decode("utf-8")
    
    elif file_type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        df = pd.read_csv(uploaded_file) if file_type == "text/csv" else pd.read_excel(uploaded_file)
        text = df.to_string()
    
    else:
        text = "Unsupported file format"
    
    return file_name, text

def main():
    st.set_page_config(page_title="SK Personal AI Assistant", page_icon="🤖", layout="wide")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "document_processor" not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if "vector_db" not in st.session_state:
        embedding_model = get_env_var("EMBEDDING_MODEL", "huggingface")
        st.session_state.vector_db = VectorDatabase(persist_directory="db", embedding_model=embedding_model)
    
    if "assistant" not in st.session_state:
        model_name = get_env_var("LLM_MODEL", "gpt-3.5-turbo")
        st.session_state.assistant = AgenticAssistant(
            vector_db=st.session_state.vector_db,
            model_name=model_name,
            temperature=0,
            api_key=get_api_key()
        )

    # **Page Title**
    st.title("SK Personal AI Assistant")

    # **Sidebar Settings**
    with st.sidebar:
        st.header("Settings")
        
        # **API Key Input**
        api_key_input = st.text_input("OpenAI API Key", value="", type="password", key="api_key_input")
        if api_key_input:
            st.session_state.api_key = api_key_input
        
        # **Document Upload**
        st.header("Upload Documents for Chat Context")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "png", "jpg", "jpeg", "md", "xlsx", "csv"])
        collection_name = st.text_input("Collection Name", "default")
        
        # Process uploaded file immediately for chat session
        if uploaded_file is not None:
            with st.spinner("Extracting document content..."):
                file_name, extracted_text = process_uploaded_file(uploaded_file)
                
                # Store extracted text in session state for immediate reference
                st.session_state.latest_uploaded_doc = {
                    "name": file_name,
                    "content": extracted_text
                }
                
                # Also, add to the vector database
                document = Document(page_content=extracted_text, metadata={"source": file_name})
                st.session_state.vector_db.add_documents([document], collection_name=collection_name)
                
                st.success(f"Document '{file_name}' processed and added to collection: {collection_name}")

        # **Manage Collections**
        st.header("Manage Document Collections")
        collections = st.session_state.vector_db.list_collections()
        
        if collections:
            selected_collection = st.selectbox("Select Collection", collections)
            
            if st.button("Delete Collection"):
                st.session_state.vector_db.delete_collection(selected_collection)
                st.success(f"Collection '{selected_collection}' deleted")
                st.experimental_rerun()

    # **Chat Interface**
    chat_container = st.container()
    
    # **Display Chat History**
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # **User Input for Chat**
    query = st.chat_input("Ask me anything about your documents or beyond...")
    
    if query:
        # **Add User Message to Chat History**
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # **Display User Message**
        with st.chat_message("user"):
            st.write(query)
        
        # **Generate Assistant Response**
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                current_api_key = get_api_key()

                # **Retrieve vector database context**
                collections = st.session_state.vector_db.list_collections()
                vector_context = []
                for collection in collections:
                    vector_context.extend(st.session_state.vector_db.search(query, k=5, collection_name=collection))
                
                vector_text = "\n\n".join([doc.page_content for doc in vector_context])

                # **Retrieve uploaded file context (if available)**
                uploaded_text = ""
                if "latest_uploaded_doc" in st.session_state:
                    uploaded_text = st.session_state.latest_uploaded_doc["content"]

                # **Combine both sources of context for inference**
                combined_context = f"Uploaded Document Context:\n{uploaded_text}\n\nVector Database Context:\n{vector_text}"

                # **Generate response**
                response = st.session_state.assistant.run(f"Context: {combined_context}\n\nUser Query: {query}")
                st.write(response)

        # **Add Assistant Response to Chat History**
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
