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
@st.cache_resource(show_spinner=False)
def get_vector_db():
    return VectorDatabase(embedding_model="openai")

vector_db = get_vector_db()

@st.cache_resource(show_spinner=False)
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_client = get_openai_client()


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
    st.session_state.last_image_path = None  # Store last image path if needed outside chat


# --- Display Chat History ---
st.subheader("📜 Conversation")
chat_container = st.container(height=500)
with chat_container:
    history_to_display = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
    for chat in history_to_display:
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"{chat['query']}")
        
        with st.chat_message("assistant", avatar="🤖"):
            response_content = chat['response']
            
            if response_content.startswith("IMAGE_PATH::"):
                img_path = response_content.split("::", 1)[1]
                st.write(f"Debug - Image path: {img_path}")
                st.write(f"Debug - File exists: {os.path.exists(img_path)}")
                
                if os.path.exists(img_path):
                    try:
                        # Display image
                        st.image(img_path, caption="Generated Image", use_column_width=True)
                        
                        # Add download button
                        with open(img_path, "rb") as img_file:
                            img_data = img_file.read()
                            st.download_button(
                                label="Download Image",
                                data=img_data,
                                file_name=os.path.basename(img_path),
                                mime="image/png"
                            )
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")
                else:
                    st.error("Generated image file not found")
            
            elif response_content.startswith("PPT_PATH::"):
                ppt_path = response_content.split("::", 1)[1]
                st.success(f"PowerPoint generated: {os.path.basename(ppt_path)}")
                st.session_state.last_ppt_path = ppt_path
            
            elif response_content.startswith("ERROR::"):
                st.error(response_content.split("::", 1)[1])
            
            else:
                st.markdown(response_content)

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

def determine_reasoning_effort_with_llm(query):
    prompt = f"""Classify the reasoning effort required for this query as 'low', 'medium', or 'high':
    
    Query: "{query}"

    Only output one of the three labels: low, medium, high.
    """
    response = openai_client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[{"role": "system", "content": "You are a classifier of query complexity."},
                  {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()


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
            reasoning_level = determine_reasoning_effort_with_llm(query_to_process)
            st.sidebar.info(f"🔍 Using **{reasoning_level.upper()}** reasoning effort.")

            # 🔹 Step 3: Run the agent
            agent_response = agentic_assistant.run(query_to_process)

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
