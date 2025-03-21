import streamlit as st
import os
import asyncio
from document_processor import DocumentProcessor
from vector_database import VectorDatabase
from agentic_assistant import AgenticAssistant

# Ensure the event loop is running properly
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Function to get API Key from Streamlit secrets or user input
def get_api_key():
    """Retrieve API key from Streamlit secrets or session state."""
    return st.secrets.get("OPENAI_API_KEY", st.session_state.get("api_key", ""))

# Function to get environment variables from Streamlit secrets
def get_env_var(key, default_value=""):
    """Retrieve environment variables from Streamlit secrets."""
    return st.secrets.get(key, default_value)

# Main function for Streamlit app
def main():
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="SK Personal AI Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state variables if not already set
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "document_processor" not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()

    if "vector_db" not in st.session_state:
        embedding_model = get_env_var("EMBEDDING_MODEL", "huggingface")
   
