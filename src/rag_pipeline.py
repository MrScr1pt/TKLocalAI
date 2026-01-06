"""
TKLocalAI - RAG Pipeline
=========================
Retrieval-Augmented Generation pipeline that combines
document retrieval with LLM generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from .document_processor import DocumentProcessor, Document, DocumentChunk
from .vector_store import VectorStore, SearchResult


@dataclass
class RAGContext:
    """Context retrieved for RAG."""
    chunks: List[SearchResult]
    formatted_context: str
    sources: List[Dict[str, Any]]
    
    @property
    def has_context(self) -> bool:
        """Check if any context was retrieved."""
        return len(self.chunks) > 0


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    context: RAGContext
    query: str
    tokens_used: int
    generation_time: float


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    
    Combines document retrieval with LLM generation to produce
    responses grounded in retrieved context.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_engine: Optional[Any] = None,  # LLMEngine type
        top_k: int = 5,
        score_threshold: float = 0.3,
        include_sources: bool = True,
        max_context_length: int = 3000,
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector store for retrieval
            llm_engine: LLM engine for generation (optional, for query-only mode)
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score for retrieval
            include_sources: Whether to include source citations
            max_context_length: Maximum context length in characters
        """
        self.vector_store = vector_store
        self.llm_engine = llm_engine
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.include_sources = include_sources
        self.max_context_length = max_context_length
        
        logger.info("RAG pipeline initialized")
    
    def set_llm_engine(self, llm_engine: Any) -> None:
        """Set or update the LLM engine."""
        self.llm_engine = llm_engine
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> RAGContext:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            top_k: Override default top_k
            filter_metadata: Optional metadata filter
        
        Returns:
            RAGContext with retrieved chunks and formatted context
        """
        top_k = top_k or self.top_k
        
        # Search vector store
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            score_threshold=self.score_threshold,
            filter_metadata=filter_metadata,
        )
        
        # Format context
        formatted_context, sources = self._format_context(results)
        
        context = RAGContext(
            chunks=results,
            formatted_context=formatted_context,
            sources=sources,
        )
        
        logger.debug(f"Retrieved {len(results)} chunks for query")
        return context
    
    def _format_context(
        self,
        results: List[SearchResult],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format retrieved results into context string.
        
        Args:
            results: Search results
        
        Returns:
            Tuple of (formatted context string, sources list)
        """
        if not results:
            return "", []
        
        context_parts = []
        sources = []
        current_length = 0
        
        for i, result in enumerate(results):
            # Check if adding this would exceed max length
            chunk_text = result.content
            if current_length + len(chunk_text) > self.max_context_length:
                # Truncate or skip
                remaining = self.max_context_length - current_length
                if remaining > 200:  # Worth including truncated
                    chunk_text = chunk_text[:remaining] + "..."
                else:
                    break
            
            # Format chunk with source info
            source_info = self._get_source_info(result, i + 1)
            sources.append(source_info)
            
            if self.include_sources:
                context_parts.append(f"[Source {i + 1}: {source_info['title']}]\n{chunk_text}")
            else:
                context_parts.append(chunk_text)
            
            current_length += len(chunk_text)
        
        formatted = "\n\n---\n\n".join(context_parts)
        return formatted, sources
    
    def _get_source_info(self, result: SearchResult, index: int) -> Dict[str, Any]:
        """Extract source information from a search result."""
        metadata = result.metadata
        
        filename = metadata.get('document_filename', 'Unknown')
        chunk_idx = metadata.get('chunk_index', 0)
        total_chunks = metadata.get('total_chunks', 1)
        
        return {
            'id': result.id,
            'title': filename,
            'chunk': f"{chunk_idx + 1}/{total_chunks}",
            'score': round(result.score, 3),
            'file_type': metadata.get('file_type', 'unknown'),
            'language': metadata.get('language'),
        }
    
    def build_prompt(
        self,
        query: str,
        context: RAGContext,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build the full prompt with context for the LLM.
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt
        
        Returns:
            Formatted prompt string
        """
        if context.has_context:
            rag_instruction = """Use the following retrieved context to answer the question. 
If the context doesn't contain relevant information, say so clearly and answer based on your general knowledge.
Always cite sources when using retrieved information.

Retrieved Context:
{context}

---

"""
            prompt = rag_instruction.format(context=context.formatted_context)
            prompt += f"Question: {query}"
        else:
            prompt = f"""No relevant context was found in the knowledge base for this query.
Please answer based on your general knowledge, and clearly indicate that this response is not grounded in retrieved documents.

Question: {query}"""
        
        return prompt
    
    def query(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        **generation_kwargs,
    ) -> RAGResponse:
        """
        Execute a full RAG query.
        
        Args:
            query: User query
            system_prompt: Optional system prompt
            top_k: Override default top_k
            filter_metadata: Optional metadata filter
            **generation_kwargs: Additional generation parameters
        
        Returns:
            RAGResponse with answer and context
        """
        if self.llm_engine is None:
            raise RuntimeError("LLM engine not set. Call set_llm_engine() first.")
        
        # Retrieve context
        context = self.retrieve(query, top_k, filter_metadata)
        
        # Build prompt
        augmented_prompt = self.build_prompt(query, context, system_prompt)
        
        # Generate response
        messages = [{"role": "user", "content": augmented_prompt}]
        
        result = self.llm_engine.chat(
            messages=messages,
            system_prompt=system_prompt,
            **generation_kwargs,
        )
        
        # Add source citations if configured
        answer = result.text
        if self.include_sources and context.has_context:
            answer = self._append_sources(answer, context.sources)
        
        return RAGResponse(
            answer=answer,
            context=context,
            query=query,
            tokens_used=result.tokens_generated + result.tokens_prompt,
            generation_time=result.generation_time,
        )
    
    def _append_sources(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """Append source citations to the answer."""
        if not sources:
            return answer
        
        sources_text = "\n\n---\n**Sources:**\n"
        for source in sources:
            sources_text += f"- [{source['title']}] (Relevance: {source['score']:.0%})\n"
        
        return answer + sources_text
    
    async def query_stream(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        **generation_kwargs,
    ):
        """
        Execute a streaming RAG query.
        
        Yields:
            Text chunks as they are generated
        """
        if self.llm_engine is None:
            raise RuntimeError("LLM engine not set")
        
        # Retrieve context
        context = self.retrieve(query, top_k, filter_metadata)
        
        # Build prompt
        augmented_prompt = self.build_prompt(query, context, system_prompt)
        
        # Stream generation
        messages = [{"role": "user", "content": augmented_prompt}]
        
        for chunk in self.llm_engine.chat_stream(
            messages=messages,
            system_prompt=system_prompt,
            **generation_kwargs,
        ):
            yield chunk
        
        # Yield sources at the end
        if self.include_sources and context.has_context:
            yield self._format_sources_text(context.sources)
    
    def _format_sources_text(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources as text."""
        if not sources:
            return ""
        
        text = "\n\n---\n**Sources:**\n"
        for source in sources:
            text += f"- [{source['title']}] (Relevance: {source['score']:.0%})\n"
        return text


class RAGManager:
    """
    High-level manager for the RAG system.
    Handles document ingestion and pipeline management.
    """
    
    def __init__(
        self,
        vector_store_path: str = "data/vectordb",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ):
        """
        Initialize the RAG manager.
        
        Args:
            vector_store_path: Path for vector store
            collection_name: Collection name
            embedding_model: Embedding model name
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap
            top_k: Default retrieval count
            score_threshold: Minimum similarity score
        """
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        self.vector_store = VectorStore(
            path=vector_store_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
        
        self.pipeline = RAGPipeline(
            vector_store=self.vector_store,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        
        # Track ingested documents
        self._ingested_docs: Dict[str, Document] = {}
        
        logger.info("RAG manager initialized")
    
    def set_llm_engine(self, llm_engine: Any) -> None:
        """Set the LLM engine for the pipeline."""
        self.pipeline.set_llm_engine(llm_engine)
    
    def ingest_file(self, file_path: Path) -> Optional[Document]:
        """
        Ingest a single file into the RAG system.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Ingested document or None if failed
        """
        file_path = Path(file_path)
        
        # Process the file
        document = self.document_processor.process_file(file_path)
        
        if document is None:
            return None
        
        # Add chunks to vector store
        if document.chunks:
            self.vector_store.add_chunks(document.chunks)
            self._ingested_docs[document.id] = document
            logger.info(f"Ingested: {file_path.name} ({len(document.chunks)} chunks)")
        
        return document
    
    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Ingest all documents in a directory.
        
        Args:
            directory: Directory path
            recursive: Process subdirectories
            extensions: Optional file extension filter
        
        Returns:
            List of ingested documents
        """
        directory = Path(directory)
        
        # Process all documents
        documents = self.document_processor.process_directory(
            directory=directory,
            recursive=recursive,
            extensions=extensions,
        )
        
        # Add all chunks to vector store
        all_chunks = self.document_processor.get_all_chunks(documents)
        if all_chunks:
            self.vector_store.add_chunks(all_chunks)
        
        # Track documents
        for doc in documents:
            self._ingested_docs[doc.id] = doc
        
        logger.info(f"Ingested {len(documents)} documents with {len(all_chunks)} total chunks")
        return documents
    
    def ingest_text(
        self,
        text: str,
        title: str = "Manual Entry",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Ingest raw text directly.
        
        Args:
            text: Text content
            title: Title for the document
            metadata: Optional metadata
        
        Returns:
            Document ID
        """
        import hashlib
        
        doc_id = hashlib.md5(text.encode()).hexdigest()[:16]
        
        # Create document
        document = Document(
            id=doc_id,
            path="",
            filename=title,
            content=text,
            file_type="text",
            metadata=metadata or {},
        )
        
        # Chunk the document
        document = self.document_processor.chunker.chunk_document(document)
        
        # Add to vector store
        if document.chunks:
            self.vector_store.add_chunks(document.chunks)
            self._ingested_docs[doc_id] = document
        
        logger.info(f"Ingested text: {title} ({len(document.chunks)} chunks)")
        return doc_id
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Metadata filter
        
        Returns:
            List of search results
        """
        return self.vector_store.search(
            query=query,
            top_k=top_k or self.pipeline.top_k,
            score_threshold=self.pipeline.score_threshold,
            filter_metadata=filter_metadata,
        )
    
    def query(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> RAGResponse:
        """
        Execute a RAG query.
        
        Args:
            query: User query
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
        
        Returns:
            RAG response
        """
        return self.pipeline.query(query, system_prompt, **kwargs)
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the RAG system.
        
        Args:
            doc_id: Document ID
        
        Returns:
            True if deleted, False if not found
        """
        if doc_id not in self._ingested_docs:
            return False
        
        document = self._ingested_docs[doc_id]
        chunk_ids = [chunk.id for chunk in document.chunks]
        
        self.vector_store.delete(chunk_ids)
        del self._ingested_docs[doc_id]
        
        logger.info(f"Deleted document: {doc_id}")
        return True
    
    def clear(self) -> None:
        """Clear all data from the RAG system."""
        self.vector_store.clear()
        self._ingested_docs.clear()
        logger.info("Cleared all RAG data")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        vector_stats = self.vector_store.get_stats()
        
        return {
            **vector_stats,
            'documents_ingested': len(self._ingested_docs),
            'document_list': [
                {
                    'id': doc.id,
                    'filename': doc.filename,
                    'chunks': len(doc.chunks),
                }
                for doc in self._ingested_docs.values()
            ],
        }
