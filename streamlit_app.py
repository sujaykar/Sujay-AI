import streamlit as st
import os
from langchain.schema import Document, HumanMessage, AIMessage
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

# @st.cache_resource # Cache

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
    for chat in st.session_state.chat_history[-MAX_CHAT_HISTORY:]:
        if isinstance(chat, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(chat.content)
        elif isinstance(chat, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                response_content = chat.content
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
if query:
    # Reset paths for new query
    st.session_state.last_ppt_path = None
    st.session_state.last_image_path = None

    # Add user query to history
    st.session_state.chat_history.append(HumanMessage(content=query))
    if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
        st.session_state.chat_history.pop(0)

    with st.spinner("Thinking..."):
        try:
            # Run the agent with the query and chat history
            agent_response = agentic_assistant.run(query, st.session_state.chat_history)
            st.session_state.chat_history.append(AIMessage(content=agent_response))

            # Store any artifact path
            if agent_response.startswith("PPT_PATH::"):
                st.session_state.last_ppt_path = agent_response.split("::", 1)[1]
            elif agent_response.startswith("IMAGE_PATH::"):
                st.session_state.last_image_path = agent_response.split("::", 1)[1]

        except Exception as e:
            st.error(f"An error occurred: {e}")

        # Trigger a rerun to display the updated chat history
        st.rerun()

# --- Display Download Button Conditionally ---
if st.session_state.get("last_ppt_path") and os.path.exists(st.session_state.last_ppt_path):
    ppt_path = st.session_state.last_ppt_path
    try:
        with open(ppt_path, "rb") as fp:
            st.download_button(
                label=f"📥 Download {os.path.basename(ppt_path)}",
                data=fp,
                file_name=os.path.basename(ppt_path),
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                key=f"download_{ppt_path}"
            )
    except Exception as e:
        st.error(f"Could not read PPT file for download: {e}")
    # Optionally clear state after showing button once:
    st.session_state.last_ppt_path = None
