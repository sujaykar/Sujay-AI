#!/usr/bin/env python
# coding: utf-8

import os
from typing import List
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings


class VectorDatabase:
    def __init__(self, persist_directory: str = "db", embedding_model: str = "openai"):
        """Initialize the Qdrant vector database.

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
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        # ✅ Connect to the Qdrant server (using Qdrant Cloud or local server)
        self.client = QdrantClient(url="http://localhost:6333")  # Connect to the Qdrant server

    def add_documents(self, documents: List[Document], collection_name: str = "default") -> None:
        """Add documents to the vector database."""
        # Convert documents to embeddings
        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)

        # Ensure collection exists
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )

        # Upload embeddings to Qdrant
        points = [{"id": i, "vector": vec, "payload": {"text": texts[i]}} for i, vec in enumerate(vectors)]
        self.client.upsert(collection_name=collection_name, points=points)

    def search(self, query: str, k: int = 5, collection_name: str = "default") -> List[Document]:
        """Search for documents similar to the query."""
        # Convert query into vector embedding
        query_vector = self.embeddings.embed_query(query)

        # Perform the search
        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=k,
        )

        # Return matched documents
        return [Document(page_content=hit.payload["text"]) for hit in search_results]

    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        return [collection.name for collection in self.client.get_collections().collections]

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the database."""
        self.client.delete_collection(collection_name=collection_name)
