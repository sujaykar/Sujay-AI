import streamlit as st
import os
from langchain.schema import Document
from openai import OpenAI
from vector_database import VectorDatabase
from document_processor import DocumentProcessor
from agentic_assistant import AgenticAssistant  # Importing the agentic framework

# --- Constants ---
MAX_CHAT_HISTORY = 10  
MAX_DOC_CHARACTERS = 550000  
MAX_VECTOR_DOCS = 15 
MAX_TOKENS = 9000  # Adjusted for GPT-4o
MIN_SIMILARITY = 0.72  

# --- Reasoning Effort Mapping ---
REASONING_EFFORT = {
    "low": {"temperature": 0.3, "max_tokens": 2048},
    "medium": {"temperature": 0.6, "max_tokens": 4096},
    "high": {"temperature": 0.9, "max_tokens": 4096}  # GPT-4o limit
}

# --- Initialize Components ---
vector_db = VectorDatabase(embedding_model="openai")
document_processor = DocumentProcessor()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the Agentic Assistant with GPT-4o model
agentic_assistant = AgenticAssistant(vector_db, model_name="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

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
        docs = document_processor.process_uploaded_file(uploaded_file)
        vector_db.add_documents(docs, collection_name)  # ✅ Always process new files, even if collection exists
        st.sidebar.success(f"✅ {uploaded_file.name} added to `{collection_name}`")

# --- Chat UI ---
st.title("🤖 Sujay's AI Assistant")
st.write("Ask me anything about the uploaded documents or any topic!")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past chat messages with better formatting
st.subheader("📜 Chat History")
for chat in st.session_state.chat_history[-MAX_CHAT_HISTORY:]:
    with st.expander(f"📌 **User:** {chat['query']}"):
        st.markdown(f"**🤖 AI:** {chat['response']}")

query = st.chat_input("How can I help you? Ask me anything...")

# --- Query Processing ---
def retrieve_from_qdrant(query):
    """Retrieve relevant context from Qdrant by dynamically selecting the best collections."""
    results = vector_db.search(
        query=query,
        k=MAX_VECTOR_DOCS,
        score_threshold=MIN_SIMILARITY
    )
    return "\n\n".join([f"[{res.metadata.get('collection', 'unknown')}] {res.page_content}" for res in results])

def determine_reasoning_effort(query):
    """Determine the reasoning effort level based on query complexity."""
    if len(query.split()) < 8:  # Simple queries
        return "low"
    elif "explain" in query.lower() or "analyze" in query.lower():  # Requires deeper reasoning
        return "high"
    return "medium"

if query:
    with st.spinner("Processing your query..."):
        # 🔹 Step 1: Retrieve relevant documents from Qdrant
        retrieved_context = retrieve_from_qdrant(query)

        # 🔹 Step 2: Determine reasoning effort
        reasoning_level = determine_reasoning_effort(query)
        st.sidebar.info(f"🔍 Using **{reasoning_level.upper()}** reasoning effort.")

        # 🔹 Step 3: Send query to the correct agent with Qdrant context
        agent_response = agentic_assistant.run(f"Context: {retrieved_context}\n\nQuestion: {query}")

        # 🔹 Step 4: Generate AI response using GPT-4o with proper reasoning effort
        model_params = REASONING_EFFORT[reasoning_level]

        request_payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Provide clear, context-aware answers using retrieved knowledge and agents."},
                {"role": "user", "content": f"Context: {retrieved_context}\n\nQuestion: {query}\nAnswer:"}
            ],
            "max_tokens": model_params["max_tokens"],
            "temperature": model_params["temperature"]
        }

        response = openai_client.chat.completions.create(**request_payload)

        ai_response = response.choices[0].message.content  # ✅ Fixed missing assignment

        # 🔹 Store chat history (limit to last 10 messages)
        st.session_state.chat_history.append({"query": query, "response": ai_response})
        if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
            st.session_state.chat_history.pop(0)  # Keep only the last 10 entries

        # Display response
        st.write("🤖 **AI Response:**")
        st.write(ai_response)
