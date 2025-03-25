import streamlit as st
import os
import asyncio
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from openai import OpenAI
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import pandas as pd
from vector_database import VectorDatabase

# Constants - Updated for cost optimization
MAX_CHAT_HISTORY = 7  # Reduced from 10
MAX_DOC_CHARACTERS = 450000  # Reduced from 1M
MAX_VECTOR_DOCS = 8  # Reduced from 10
MAX_TOKENS = 6000  # Reduced from 8000
MIN_SIMILARITY = 0.72  # New similarity threshold

# Initialize components
vector_db = VectorDatabase(embedding_model="openai")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
query = st.chat_input("I am Sujay's intelligent assistant. Ask me anything...")

# --- New Optimization Functions ---
def dynamic_reasoning_effort(query: str) -> str:
    """Determine reasoning effort based on query complexity"""
    complex_keywords = {"calculate", "analyze", "financial", "derive", "compare", "trend"}
    simple_keywords = {"summary", "overview", "explain", "what", "who"}
    
    query_lower = query.lower()
    if any(kw in query_lower for kw in complex_keywords):
        return "high"
    elif any(kw in query_lower for kw in simple_keywords):
        return "low"
    return "medium"

def optimize_context(context: str, max_tokens: int) -> str:
    """Reduce context while preserving key information"""
    sentences = context.split('. ')
    optimized = []
    token_count = 0
    
    for sent in sentences:
        sent_tokens = len(sent.split()) + 1  # +1 for the period
        if token_count + sent_tokens <= max_tokens:
            optimized.append(sent)
            token_count += sent_tokens
        else:
            break
            
    return '. '.join(optimized) + ('' if context.endswith('.') else '.')

# --- Modified Functions ---
def process_uploaded_file(uploaded_file, collection_name):
    """Optimized file processing with chunking"""
    # ... (existing file type handling) ...
    
    # Split text into optimized chunks
    chunk_size = 512  # Optimal for o3-mini
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Store chunks with metadata
    docs = [Document(
        page_content=chunk,
        metadata={"source": uploaded_file.name, "chunk_num": i}
    ) for i, chunk in enumerate(text_chunks)]
    
    vector_db.add_documents(docs, collection_name=collection_name)
    return file_name, text[:MAX_DOC_CHARACTERS]  # Return truncated preview

def retrieve_from_qdrant(query):
    """Optimized retrieval with similarity filtering"""
    results = vector_db.search(
        query, 
        k=MAX_VECTOR_DOCS,
        search_type="similarity_score_threshold",  # Add this line
        search_kwargs={
            "score_threshold": MIN_SIMILARITY,
            "filter": None  # Add metadata filters here if needed
        }
    )
    return "\n\n".join([res.page_content for res in results])

# --- Modified Main Logic ---
def main():
    # ... (existing setup code) ...

    if query:
        # --- New: Dynamic Reasoning Effort ---
        reasoning_level = dynamic_reasoning_effort(query)
        
        # --- Optimized Context Handling ---
        retrieved_context = retrieve_from_qdrant(query)
        combined_context = optimize_context(
            f"Relevant information:\n{retrieved_context}",
            max_tokens=MAX_TOKENS//2  # Reserve half tokens for response
        )

        # --- Cost-Optimized API Call ---
        response = openai_client.chat.completions.create(
            model="o3-mini",
            reasoning_effort=reasoning_level,
            messages=[
                {
                    "role": "system", 
                    "content": "Answer concisely using only the context. If unsure, say 'I don't know'."
                },
                {
                    "role": "user",
                    "content": f"Context: {combined_context}\n\nQuestion: {query}\nAnswer:"
                }
            ],
            max_tokens=250  # Enforce concise responses
        )

        # ... (existing display code) ...

if __name__ == "__main__":
    main()
