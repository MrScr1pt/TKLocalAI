"""
TKLocalAI - Vector Store
=========================
Manages vector embeddings and similarity search using Qdrant.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error("sentence-transformers not installed")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams,
        Distance,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SearchParams,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.error("qdrant-client not installed")


@dataclass
class SearchResult:
    """Result of a similarity search."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class EmbeddingModel:
    """Handles text embedding generation."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers is not installed")
        
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None
        
        logger.info(f"Embedding model initialized: {model_name}")
    
    def load(self) -> None:
        """Load the embedding model."""
        if self._model is not None:
            return
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name)
        
        # Get embedding dimension
        test_embedding = self._model.encode("test")
        self._dimension = len(test_embedding)
        
        logger.info(f"Embedding model loaded. Dimension: {self._dimension}")
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension is None:
            self.load()
        return self._dimension
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector as numpy array
        """
        self.load()
        return self._model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
        
        Returns:
            Array of embedding vectors
        """
        self.load()
        return self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )


class VectorStore:
    """
    Vector store using Qdrant for similarity search.
    Runs in local/embedded mode for privacy.
    """
    
    def __init__(
        self,
        path: str = "data/vectordb",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_size: int = 384,
    ):
        """
        Initialize the vector store.
        
        Args:
            path: Path for persistent storage
            collection_name: Name of the collection
            embedding_model: Embedding model name
            vector_size: Dimension of vectors (must match embedding model)
        """
        if not QDRANT_AVAILABLE:
            raise RuntimeError("qdrant-client is not installed")
        
        self.path = Path(path)
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize embedding model
        self.embedder = EmbeddingModel(embedding_model)
        
        # Initialize Qdrant client (local/embedded mode)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self.path))
        
        # Ensure collection exists
        self._ensure_collection()
        
        logger.info(f"Vector store initialized at {self.path}")
    
    def _ensure_collection(self) -> None:
        """Ensure the collection exists, create if not."""
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.debug(f"Collection exists: {self.collection_name}")
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional metadata for each text
            ids: Optional IDs for each text
        
        Returns:
            List of IDs for the added texts
        """
        if not texts:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            import hashlib
            ids = [hashlib.md5(t.encode()).hexdigest()[:16] for t in texts]
        
        # Generate embeddings
        embeddings = self.embedder.embed_batch(texts)
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Add content to metadata for retrieval
        for i, meta in enumerate(metadatas):
            meta['content'] = texts[i]
        
        # Create points
        points = [
            PointStruct(
                id=idx,  # Qdrant needs integer or UUID, so we'll use index
                vector=embeddings[i].tolist(),
                payload={
                    'id': ids[i],
                    **metadatas[i],
                },
            )
            for i, idx in enumerate(range(len(texts)))
        ]
        
        # Get current count for offset
        collection_info = self._client.get_collection(self.collection_name)
        offset = collection_info.points_count
        
        # Update point IDs with offset
        for i, point in enumerate(points):
            point.id = offset + i
        
        # Upsert points
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        logger.info(f"Added {len(texts)} texts to vector store")
        return ids
    
    def add_chunks(self, chunks: List['DocumentChunk']) -> List[str]:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects
        
        Returns:
            List of chunk IDs
        """
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.id for chunk in chunks]
        
        return self.add_texts(texts, metadatas, ids)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar texts.
        
        Args:
            query: Query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_metadata: Optional metadata filter
        
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        
        # Build filter if specified
        query_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                for key, value in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)
        
        # Search using query_points (new Qdrant API)
        try:
            results = self._client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
            ).points
        except AttributeError:
            # Fallback for older qdrant-client versions
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
        
        # Convert to SearchResult
        search_results = []
        for result in results:
            payload = result.payload or {}
            search_results.append(SearchResult(
                id=payload.get('id', str(result.id)),
                content=payload.get('content', ''),
                score=result.score,
                metadata={k: v for k, v in payload.items() if k not in ['id', 'content']},
            ))
        
        logger.debug(f"Search returned {len(search_results)} results for query: {query[:50]}...")
        return search_results
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete texts by ID.
        
        Args:
            ids: List of IDs to delete
        """
        # Qdrant uses point IDs, we need to search by our custom ID field
        for doc_id in ids:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                ),
            )
        
        logger.info(f"Deleted {len(ids)} texts from vector store")
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        self._client.delete_collection(self.collection_name)
        self._ensure_collection()
        logger.info(f"Cleared collection: {self.collection_name}")
    
    def count(self) -> int:
        """Get the number of vectors in the collection."""
        collection_info = self._client.get_collection(self.collection_name)
        return collection_info.points_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        collection_info = self._client.get_collection(self.collection_name)
        
        return {
            'collection_name': self.collection_name,
            'vector_count': collection_info.points_count,
            'vector_size': self.vector_size,
            'path': str(self.path),
            'embedding_model': self.embedder.model_name,
        }
