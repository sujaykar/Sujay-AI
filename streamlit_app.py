import streamlit as st
import os
from document_processor import DocumentProcessor
from vector_database import VectorDatabase
from agentic_assistant import AgenticAssistant

def get_api_key():
    """Get API key from Streamlit secrets or user input"""
    # First try to get from secrets
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    # Then from session state (user input)
    elif "api_key" in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    return ""

def get_env_var(key, default_value=""):
    """Get environment variable from Streamlit secrets or use default"""
    if key in st.secrets:
        return st.secrets[key]
    return default_value

def main():
    # Set page configuration
    st.set_page_config(
        page_title="SK Personal AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if 'vector_db' not in st.session_state:
        embedding_model = get_env_var("EMBEDDING_MODEL", "huggingface")
        st.session_state.vector_db = VectorDatabase(
            persist_directory="db",
            embedding_model=embedding_model
        )
    
    if 'assistant' not in st.session_state:
        model_name = get_env_var("LLM_MODEL", "gpt-3.5-turbo")
        st.session_state.assistant = AgenticAssistant(
            vector_db=st.session_state.vector_db,
            model_name=model_name
        )
    
    # Page title
    st.title("SK Personal AI Assistant")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # API Key input
        api_key_input = st.text_input("OpenAI API Key", 
                                     value=get_api_key(), 
                                     type="password",
                                     key="api_key_input")
        if api_key_input:
            st.session_state.api_key = api_key_input
        
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "png", "jpg", "jpeg", "md", "xlsx", "csv"]
        )
        
        collection_name = st.text_input("Collection Name", "default")
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    documents = st.session_state.document_processor.process_uploaded_file(uploaded_file)
                    st.session_state.vector_db.add_documents(documents, collection_name=collection_name)
                    st.success(f"Document processed and added to collection: {collection_name}")
        
        st.header("Collections")
        collections = st.session_state.vector_db.list_collections()
        
        if collections:
            selected_collection = st.selectbox("Select Collection", collections)
            
            if st.button("Delete Collection"):
                st.session_state.vector_db.delete_collection(selected_collection)
                st.success(f"Collection {selected_collection} deleted")
                st.experimental_rerun()
    
    # Main panel
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Query input
    query = st.chat_input("Ask me anything about your documents...")
    
    if query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Ensure we're using the most current API key
                current_api_key = get_api_key()
                # Pass API key if your assistant class accepts it as a parameter
                response = st.session_state.assistant.run(query, api_key=current_api_key)
                st.write(response)
        
        # Add assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
