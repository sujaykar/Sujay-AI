from typing import List, Optional, Dict, Union
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Distance, VectorParams, HnswConfig
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
import streamlit as st
from qdrant_client.models import PointStruct

class VectorDatabase:
    def __init__(self, embedding_model: str = "openai"):
        """Initialize Qdrant client and embedding model."""
        # Get credentials from Streamlit secrets
        self.client = QdrantClient(
            url=st.secrets.qdrant.url,
            api_key=st.secrets.qdrant.api_key,
            prefer_grpc=True  # Better for production
        )
        
        # Initialize embeddings
        if embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def search(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: Optional[float] = None,  # Ensure score threshold is optional
        collection_name: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Union[str, int, bool]]] = None
    ) -> List[Document]:
        """Enhanced search with score threshold and metadata filtering."""
        query_vector = self.embeddings.embed_query(query)
        collections = [collection_name] if collection_name else self.list_collections()
        all_results = []

        # Build metadata filter
        qdrant_filter = Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in (metadata_filter or {}).items()
            ]
        ) if metadata_filter else None

        for collection in collections:
            # Verify collection exists
            if not self._collection_exists(collection):
                continue

            # Perform search with thresholding
            search_results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=k,
            )

            # Apply score threshold manually (if not supported by Qdrant)
            docs = [
                Document(
                    page_content=hit.payload.get("text", ""),
                    metadata=hit.payload.get("metadata", {}) | {"score": hit.score}
                )
                for hit in search_results
                if score_threshold is None or hit.score >= score_threshold  # Manual filtering
            ]
            all_results.extend(docs)

        return all_results or [Document(page_content="No relevant documents found.")]

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in the database."""
        collections = self.list_collections()
        return collection_name in collections


def add_documents(self, documents: List[Document], collection_name: str = "default") -> None:
    """Improved document insertion with metadata handling."""
    # Create collection if it doesn't exist
    if not self._collection_exists(collection_name):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(self.embeddings.embed_query("test")),  # Get embedding dim
                distance=Distance.COSINE,
                hnsw_config=HnswConfig(m=16, ef_construction=200)  # HNSW Configuration
            )
        )

    # ✅ Prepare documents in correct PointStruct format
    points = [
        PointStruct(  # ✅ Use PointStruct
            id=str(hash(doc.page_content + str(doc.metadata))),
            vector=self.embeddings.embed_query(doc.page_content),
            payload={
                "text": doc.page_content,
                "metadata": doc.metadata
            }
        )
        for doc in documents
    ]

    # ✅ Ensure correct structure before upserting
    self.client.upsert(
        collection_name=collection_name,
        points=points  # ✅ Now in correct PointStruct format
    )

    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        return [collection.name for collection in self.client.get_collections().collections]

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the database."""
        self.client.delete_collection(collection_name=collection_name)
