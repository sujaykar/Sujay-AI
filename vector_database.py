from typing import List, Optional, Dict
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Distance, VectorParams
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
import streamlit as st
from qdrant_client.models import PointStruct

class VectorDatabase:
    def __init__(self, embedding_model: str = "openai"):
        """Initialize Qdrant client and embedding model."""
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

    def list_collections(self) -> List[str]:
        """List all collections in Qdrant."""
        return [collection.name for collection in self.client.get_collections().collections]

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in the database."""
        return collection_name in self.list_collections()

    def create_collection_if_not_exists(self, collection_name: str) -> None:
    """Ensure the collection exists before inserting documents."""
    if not self._collection_exists(collection_name):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(self.embeddings.embed_query("test")),  # Get embedding dim
                distance=Distance.COSINE,
                hnsw_config={"m": 16, "ef_construct": 200, "full_scan_threshold": 10000}  # HNSW Configuration
            )
        )


    def add_documents(self, documents: List[Document], collection_name: str = "default") -> None:
        """Insert multiple documents into the same collection without overwriting existing data."""
        self.create_collection_if_not_exists(collection_name)

        # ✅ Fetch existing IDs to prevent duplicate entries
        existing_ids = {point.id for point in self.client.scroll(collection_name, limit=10_000).points}

        new_points = []
        for doc in documents:
            doc_id = abs(hash(doc.page_content + str(doc.metadata))) % (2**31)

            # Skip if the document already exists
            if doc_id in existing_ids:
                continue  

            new_points.append(PointStruct(
                id=doc_id,
                vector=self.embeddings.embed_query(doc.page_content),
                payload={
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
            ))

        if new_points:
            self.client.upsert(collection_name=collection_name, points=new_points)

    def search(
        self, query: str, k: int = 5, score_threshold: Optional[float] = None, 
        collection_name: Optional[str] = None, metadata_filter: Optional[Dict] = None
    ) -> List[Document]:
        """Enhanced search with score threshold and dynamic collection ranking."""
        query_vector = self.embeddings.embed_query(query)
        collections = [collection_name] if collection_name else self.list_collections()
        all_results = []

        for collection in collections:
            if not self._collection_exists(collection):
                continue

            search_params = {
                "collection_name": collection,
                "query_vector": query_vector,
                "limit": k
            }

            if metadata_filter:
                search_params["query_filter"] = Filter(must=[FieldCondition(
                    key=list(metadata_filter.keys())[0],
                    match=MatchValue(value=list(metadata_filter.values())[0])
                )])

            search_results = self.client.search(**search_params)

            docs = [Document(page_content=hit.payload.get("text", ""), metadata={"collection": collection}) 
                    for hit in search_results]
            all_results.extend(docs)

        return sorted(all_results, key=lambda d: len(d.page_content), reverse=True) or [Document(page_content="No relevant documents found.")]

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the database."""
        self.client.delete_collection(collection_name=collection_name)
