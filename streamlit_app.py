import streamlit as st
import os
import asyncio
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI
from vector_database import VectorDatabase
from document_processor import DocumentProcessor
from PIL import Image
from agentic_assistant import AgenticAssistant  # Importing the agentic framework

# --- Constants ---
MAX_CHAT_HISTORY = 7  
MAX_DOC_CHARACTERS = 450000  
MAX_VECTOR_DOCS = 8  
MAX_TOKENS = 6000  
MIN_SIMILARITY = 0.72  

# --- Initialize Components ---
vector_db = VectorDatabase(embedding_model="openai")
document_processor = DocumentProcessor()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Initialize the Agentic Assistant with o3-mini model
agentic_assistant = AgenticAssistant(vector_db, model_name="o3-mini", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))

# --- Streamlit UI ---
st.set_page_config(
    page_title="Sujay's AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- Sidebar Branding ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712105.png", width=100)
st.sidebar.title("AI Knowledge Assistant")
st.sidebar.markdown("Upload files, ask questions, and retrieve insights.")

# --- File Upload Section ---
st.sidebar.subheader("📂 Upload Documents")
uploaded_file = st.sidebar.file_uploader(
    "Upload a document", type=["pdf", "txt", "png", "jpg", "jpeg", "md", "xlsx", "csv"]
)
collection_name = st.sidebar.text_input("Collection Name", "default")

if uploaded_file:
    with st.spinner("Processing document..."):
        # Ensure documents are only processed if not in Qdrant
        if collection_name not in st.session_state:
            docs = document_processor.process_uploaded_file(uploaded_file)
            vector_db.add_documents(docs, collection_name)
            st.session_state[collection_name] = True  # Mark collection as processed
            st.sidebar.success(f"✅ {uploaded_file.name} added to `{collection_name}` collection!")

# --- Chat UI ---
st.title("🤖 Sujay's AI Assistant")
st.write("Ask me anything about the uploaded documents or any topic!")

query = st.chat_input("Ask me anything...")

# --- Query Processing ---
def retrieve_from_qdrant(query):
    """Retrieve relevant context from Qdrant by dynamically selecting the best collections."""
    
    results = vector_db.search(
        query=query,
        k=MAX_VECTOR_DOCS,
        score_threshold=MIN_SIMILARITY
    )

    return "\n\n".join([f"[{res.metadata.get('collection', 'unknown')}] {res.page_content}" for res in results])

if query:
    with st.spinner("Processing your query..."):
        # 🔹 Step 1: Retrieve relevant documents from Qdrant
        retrieved_context = retrieve_from_qdrant(query)

        # 🔹 Step 2: Send query to the correct agent with Qdrant context
        agent_response = agentic_assistant.run(f"Context: {retrieved_context}\n\nQuestion: {query}")

        # 🔹 Step 3: Ensure enough space for model response
        
        combined_context = f"{retrieved_context}\n\n{agent_response}"


        # 🔹 Step 4: Generate AI response using o3-mini with combined context
        response = openai_client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "Provide clear, context-aware answers using retrieved knowledge and agents."},
                {"role": "user", "content": f"Context: {combined_context}\n\nQuestion: {query}\nAnswer:"}
            ],
            max_completion_tokens=4000
        )

        # Display response
        st.write("🤖 **AI Response:**")
        st.write(response.choices[0].message.content)
