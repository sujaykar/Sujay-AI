import requests
import streamlit as st

# Retrieve credentials from Streamlit secrets
QDRANT_URL = st.secrets["url"]
QDRANT_API_KEY = st.secrets["api_key"]

headers = {"Authorization": f"Bearer {QDRANT_API_KEY}"}

st.title("Qdrant Connection Test")

try:
    response = requests.get(f"{QDRANT_URL}/collections", headers=headers, timeout=10)
    st.success("✅ Qdrant Cloud Response:")
    st.json(response.json())
except requests.exceptions.RequestException as e:
    st.error(f"❌ Streamlit Cloud cannot reach Qdrant: {e}")
