from langchain.schema import Document
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

# Constants for best practices
MAX_CHAT_HISTORY = 10  # Keep only last 10 messages
MAX_DOC_CHARACTERS = 5000  # Limit extracted text per document
MAX_VECTOR_DOCS = 3  # Limit retrieved vector documents

# Function to extract API Key
def get_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    elif "api_key" in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    return ""

# Function to process uploaded files and store in vector database
def process_uploaded_file(uploaded_file, collection_name):
    """Extract content from various file formats and store in the vector database."""
    file_type = uploaded_file.type
    file_name = uploaded_file.name

    if file_type == "application/pdf":
        pdf_reader
