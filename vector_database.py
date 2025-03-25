from typing import List, Optional, Dict, Union
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

class VectorDatabase:
    def __init__(self, persist_directory: str = "db", embedding_model: str = "openai"):
        self.client = QdrantClient(path=persist_directory)
        
        # Initialize embeddings
        if embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def search(
        self, 
        query: str, 
        k: int = 5, 
        collection_name: Optional[str] = None,
        score_threshold: float = 0.7,
        metadata_filter: Optional[Dict[str, Union[str, int, bool]]] = None
    ) -> List[Document]:
        """Enhanced search with score threshold and metadata filtering"""
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

            # Perform thresholded search
            search_results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=k,
                score_threshold=score_threshold,
            )

            # Convert to LangChain documents with metadata
            docs = [
                Document(
                    page_content=hit.payload.get("text", ""),
                    metadata=hit.payload.get("metadata", {}) | {"score": hit.score}
                )
                for hit in search_results
                if hit.score >= score_threshold
            ]
            all_results.extend(docs)

        return all_results or [Document(page_content="No relevant documents found")]

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in the database"""
        try:
            self.client.get_collection(collection_name)
            return True
        except ValueError:
            return False

    def add_documents(self, documents: List[Document], collection_name: str = "default") -> None:
        """Improved document insertion with metadata handling"""
        # Create collection if not exists
        if not self._collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=len(self.embeddings.embed_query("test")),  # Get embedding dim
                    distance=Distance.COSINE
                )
            )

        # Prepare documents with metadata
        points = [
            {
                "id": str(hash(doc.page_content + str(doc.metadata))),
                "vector": self.embeddings.embed_query(doc.page_content),
                "payload": {
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
            }
            for doc in documents
        ]

        # Batch upsert
        self.client.upsert(
            collection_name=collection_name,
            points=points,
            batch_size=100  # Optimized for performance
        )

        # Get the correct vector size (1536 for OpenAI embeddings)
        vector_size = len(vectors[0])  # This should be 1536 for OpenAIEmbeddings

        # Check if the collection exists before attempting to recreate it
        if collection_name not in self.list_collections():
            print(f"Collection '{collection_name}' not found. Creating...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE,hnsw_config=HnswConfig(m=16,ef_construction=200 )),
            )
        
        # Upload embeddings to Qdrant
        points = [{"id": i, "vector": vec, "payload": {"text": texts[i]}} for i, vec in enumerate(vectors)]
        self.client.upsert(collection_name=collection_name, points=points)

    def search(self, query: str, k: int = 5, collection_name: Optional[str] = None) -> List[Document]:
        """Search for documents across a specific collection or all collections if none is specified."""

        # If collection_name is provided, search only that collection, else search all collections
        collections = [collection_name] if collection_name else self.list_collections()
        all_results = []

        for collection in collections:
            query_vector = self.embeddings.embed_query(query)

            # Perform search in the collection
            search_results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=k,
            )

            all_results.extend([Document(page_content=hit.payload["text"]) for hit in search_results])

        if not all_results:
            return [Document(page_content="No relevant documents found in any collection.")]

        return all_results

    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        return [collection.name for collection in self.client.get_collections().collections]

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the database."""
        self.client.delete_collection(collection_name=collection_name)
