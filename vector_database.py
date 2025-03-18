#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
from typing import List, Dict, Any
import chromadb
from langchain.schema import Document
import pydantic
import pydantic_settings
pydantic.BaseSettings = pydantic_settings.BaseSettings  # Redirect import
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class VectorDatabase:
    def __init__(self, persist_directory: str = "db", embedding_model: str = "openai"):
        """Initialize the vector database.
        
        Args:
            persist_directory: Directory to persist the database
            embedding_model: 'openai' or 'huggingface'
        """
        self.persist_directory = persist_directory
        
        # Create the directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize the embedding model
        if embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings()
        else:
            # Use a free HuggingFace model as a fallback
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Initialize the vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
    
    def add_documents(self, documents: List[Document], collection_name: str = "default") -> None:
        """Add documents to the vector database."""
        # Create a new collection for the documents
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        vectorstore.persist()
    
    def search(self, query: str, k: int = 5, collection_name: str = "default") -> List[Document]:
        """Search for documents similar to the query."""
        # Create a new ChromaDB client
        client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get the collection
        try:
            collection = client.get_collection(name=collection_name)
        except ValueError:
            # If the collection doesn't exist, return an empty list
            return []
        
        # Perform the search
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        results = vectorstore.similarity_search(query, k=k)
        return results
    
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        client = chromadb.PersistentClient(path=self.persist_directory)
        collections = client.list_collections()
        return [collection.name for collection in collections]
    
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the database."""
        client = chromadb.PersistentClient(path=self.persist_directory)
        client.delete_collection(name=collection_name)


# In[ ]:




