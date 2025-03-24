import streamlit as st
import os
import asyncio
from langchain.schema import Document
from qdrant_client import QdrantClient
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import pandas as pd

# Ensure event loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Constants for best practices
MAX_CHAT_HISTORY = 10  # Keep last 10 messages
MAX_DOC_CHARACTERS = 3000  # Limit extracted text per document
MAX_VECTOR_DOCS = 3  # Retrieve only top 3 relevant documents
MAX_TOKENS = 8000  # Safe limit for OpenAI input

# Initialize Qdrant client
qdrant_client = QdrantClient(":memory:")  # Use persistent storage in production
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Function to extract API Key
def get_api_key():
    return os.getenv("OPENAI_API_KEY")

# Function to process uploaded files and store in Qdrant
def process_uploaded_file(uploaded_file, collection_name):
    """Extract content from various file formats and store in Qdrant."""
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
    
    # ✅ Limit extracted text length
    text = text[:MAX_DOC_CHARACTERS]

    # Check if the collection exists before upserting
    existing_collections = qdrant_client.get_collections().collections
    if collection_name not in existing_collections:
        # If collection doesn't exist, create it
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(embedding_model.embed_query(text)), distance=Distance.COSINE)
        )

    # ✅ Generate embedding & store in Qdrant
    embedding = embedding_model.embed_query(text)
    qdrant_client.upsert(collection_name, [(file_name, embedding, {"source": file_name, "text": text})])

    return file_name, text

# Function to retrieve relevant embeddings from all Qdrant collections
def retrieve_from_qdrant(query):
    """Retrieve top-k similar documents from all available Qdrant collections."""
    query_embedding = embedding_model.embed_query(query)
    
    # Get all available collections
    all_collections = qdrant_client.get_collections().collections
    all_contexts = []

    for collection in all_collections:
        collection_name = collection.name
        results = qdrant_client.search(collection_name, query_vector=query_embedding, limit=MAX_VECTOR_DOCS)

        retrieved_texts = [res.payload["text"][:MAX_DOC_CHARACTERS] for res in results]
        all_contexts.extend(retrieved_texts)

    return "\n\n".join(all_contexts)

def main():
    st.set_page_config(page_title=" SK AI Knowledge Assistant", page_icon="🤖", layout="wide")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "latest_uploaded_doc" not in st.session_state:
        st.session_state.latest_uploaded_doc = None

    # **Page Title**
    st.title("SK AI Knowledge Assistant")

    # **Chat Interface**
    chat_container = st.container()

    # **Display Chat History**
    with chat_container:
        for message in st.session_state.chat_history[-MAX_CHAT_HISTORY:]:  
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # **Attach File Option**
    with st.expander("📎 Attach File for Analysis", expanded=False):
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "png", "jpg", "jpeg", "md", "xlsx", "csv"])
        collection_name = st.text_input("Collection Name", "default")

        if uploaded_file:
            with st.spinner("Processing document..."):
                file_name, extracted_text = process_uploaded_file(uploaded_file, collection_name)
                
                st.session_state.latest_uploaded_doc = {
                    "name": file_name,
                    "content": extracted_text
                }
                st.success(f"✅ Document '{file_name}' processed and stored.")

    # **User Input for Chat**
    query = st.chat_input("Ask me anything...")

    if query:
        # **Add User Message to Chat History**
        st.session_state.chat_history.append({"role": "user", "content": query})

        # **Display User Message**
        with st.chat_message("user"):
            st.write(query)

        # **Retrieve document context from all Qdrant collections**
        retrieved_context = retrieve_from_qdrant(query)

        # **Combine Contexts**
        uploaded_text = ""
        if st.session_state.latest_uploaded_doc:
            uploaded_text = f"📄 Uploaded Document: {st.session_state.latest_uploaded_doc['name']}\n\n{st.session_state.latest_uploaded_doc['content']}"

        combined_context = f"Uploaded Document Context:\n{uploaded_text}\n\nRetrieved Context:\n{retrieved_context}"

        # ✅ Trim total token count
        def count_tokens(text):
            return len(text.split())

        if count_tokens(combined_context) > MAX_TOKENS:
            combined_context = " ".join(combined_context.split()[:MAX_TOKENS])

        # **Generate AI Response**
        openai_client = OpenAI(api_key=get_api_key())
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant that provides insightful and concise answers."},
                        {"role": "user", "content": f"Context: {combined_context}\n\nQuestion: {query}"}
                    ],
                    temperature=0.7
                )
                st.write(response.choices[0].message.content)

        # **Add Assistant Response to Chat History**
        st.session_state.chat_history.append({"role": "assistant", "content": response.choices[0].message.content})
        st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]  # Trim history

if __name__ == "__main__":
    main()
