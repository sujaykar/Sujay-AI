import streamlit as st
import os
from langchain.schema import Document
from openai import OpenAI
from vector_database import VectorDatabase
from document_processor import DocumentProcessor
from agentic_assistant import AgenticAssistant  # Importing the agentic framework
from pptx import Presentation
import io


# --- Constants ---
MAX_CHAT_HISTORY = 15  
MAX_DOC_CHARACTERS = 550000  
MAX_VECTOR_DOCS = 15 
MAX_TOKENS = 9200  # Adjusted for GPT-4o
MIN_SIMILARITY = 0.72  

#@st.cache_resource # Cache

REASONING_EFFORT = {
    "low": {"temperature": 0.3, "max_tokens": 4096},
    "medium": {"temperature": 0.6, "max_tokens": 8192},
    "high": {"temperature": 0.9, "max_tokens": 16384}  # GPT-4o limit
}

# --- Initialize Components ---
vector_db = VectorDatabase(embedding_model="openai")
document_processor = DocumentProcessor()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the Agentic Assistant with GPT-4o latest model
agentic_assistant = AgenticAssistant(vector_db, model_name="chatgpt-4o-latest", api_key=os.getenv("OPENAI_API_KEY"))

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
if "last_ppt_path" not in st.session_state:
    st.session_state.last_ppt_path = None
if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None # Store last image path if needed outside chat

# --- Display Chat History ---
st.subheader("📜 Conversation")
chat_container = st.container(height=500)
with chat_container:
    history_to_display = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
    for chat in history_to_display: # Display oldest first within the visible window
        with st.chat_message("user", avatar="👤"):
             st.markdown(f"{chat['query']}") # Display query
        with st.chat_message("assistant", avatar="🤖"):
            response_content = chat['agent_response']
            # --- Handle Prefixed Responses ---
            if response_content.startswith("IMAGE_PATH::"):
                img_path = response_content.split("::", 1)[1]
                if os.path.exists(img_path):
                    st.image(img_path, caption="Generated Image")
                    st.session_state.last_image_path = img_path # Store path if needed
                else:
                    st.error(f"Generated image file not found at path: {img_path}")
            elif response_content.startswith("PPT_PATH::"):
                ppt_path = response_content.split("::", 1)[1]
                st.success(f"PowerPoint generated: `{os.path.basename(ppt_path)}`. Use download button below.")
                st.session_state.last_ppt_path = ppt_path # Store path for button
            elif response_content.startswith("ERROR::"):
                st.error(response_content.split("::", 1)[1])
            else:
                st.markdown(response_content) # Display standard text response

query = st.chat_input("How can I help you? Ask me anything that is ethical...")

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
    if len(query.split()) < 20:  # Simple queries
        return "low"
    elif any(keyword in query.lower() for keyword in ["explain", "analyze", "think step by step", "evaluate", "reason", "deduce"]):  # Requires deeper reasoning
        return "high"
    return "medium"

# --- Handle New Chat Input ---
if query:
    # Reset paths for new query
    st.session_state.last_ppt_path = None
    st.session_state.last_image_path = None

    # Add placeholder message to chat history and trigger rerun
    st.session_state.chat_history.append({"query": query, "response": "..."})
    if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
        st.session_state.chat_history.pop(0)
    st.rerun()

# --- Handle Placeholder Message (After Rerun) ---
elif st.session_state.chat_history and st.session_state.chat_history[-1]["response"] == "...":
    # Get the query from the latest chat message
    query_to_process = st.session_state.chat_history[-1]["query"]

    with st.spinner("Thinking..."):
        try:
            # 🔹 Step 1: Retrieve relevant documents
            retrieved_context = retrieve_from_qdrant(query_to_process)

            # 🔹 Step 2: Determine reasoning effort
            reasoning_level = determine_reasoning_effort(query_to_process)
            st.sidebar.info(f"🔍 Using **{reasoning_level.upper()}** reasoning effort.")

            # 🔹 Step 3: Run the agent
            full_prompt = f"Context: {retrieved_context}\n\nQuestion: {query_to_process}"
            agent_response = agentic_assistant.run(full_prompt)

        except Exception as e:
            agent_response = f"ERROR::An error occurred: {e}"
            st.error(agent_response)

        # Update placeholder with actual response
        st.session_state.chat_history[-1]["response"] = agent_response

        # Store any artifact path
        if agent_response.startswith("PPT_PATH::"):
            st.session_state.last_ppt_path = agent_response.split("::", 1)[1]
        elif agent_response.startswith("IMAGE_PATH::"):
            st.session_state.last_image_path = agent_response.split("::", 1)[1]

        # Trigger another rerun to show full assistant reply
        st.rerun()
