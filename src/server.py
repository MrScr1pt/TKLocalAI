"""
TKLocalAI - FastAPI Backend Server
===================================
REST API server providing OpenAI-compatible endpoints for chat,
document ingestion, and RAG queries.
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from loguru import logger
import json


# ===========================================
# Request/Response Models
# ===========================================

class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "local"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    # RAG options
    use_rag: Optional[bool] = True
    persona: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class SearchRequest(BaseModel):
    """Search request for RAG."""
    query: str
    top_k: Optional[int] = 5
    filter: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """Search result."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search response."""
    results: List[SearchResult]
    query: str
    total: int


class IngestRequest(BaseModel):
    """Document ingestion request."""
    text: Optional[str] = None
    title: Optional[str] = "Manual Entry"
    metadata: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    """Document ingestion response."""
    success: bool
    document_id: Optional[str] = None
    chunks_created: int = 0
    message: str


class StatsResponse(BaseModel):
    """System statistics response."""
    model: Dict[str, Any]
    rag: Dict[str, Any]
    server: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    rag_enabled: bool
    version: str = "1.0.0"


class PersonaRequest(BaseModel):
    """Persona switch request."""
    persona: str


class PersonaResponse(BaseModel):
    """Persona switch response."""
    active_persona: str
    available_personas: List[str]


# ===========================================
# Application State
# ===========================================

class AppState:
    """Application state container."""
    def __init__(self):
        self.settings = None
        self.llm_engine = None
        self.rag_manager = None
        self.initialized = False


app_state = AppState()


# ===========================================
# Application Lifecycle
# ===========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting TKLocalAI server...")
    
    try:
        # Import here to avoid circular imports
        from .config import init_config
        from .llm_engine import create_engine_from_config, LLMEngineManager
        from .rag_pipeline import RAGManager
        
        # Load configuration
        app_state.settings = init_config()
        logger.info("Configuration loaded")
        
        # Initialize LLM engine
        try:
            app_state.llm_engine = create_engine_from_config(app_state.settings)
            app_state.llm_engine.load()
            logger.info("LLM engine initialized")
        except Exception as e:
            logger.warning(f"LLM engine not loaded: {e}")
            logger.info("Server will run without LLM. Load a model first.")
        
        # Initialize RAG if enabled
        if app_state.settings.rag.enabled:
            rag_config = app_state.settings.rag
            app_state.rag_manager = RAGManager(
                vector_store_path=str(app_state.settings.get_absolute_path(rag_config.vector_db.path)),
                collection_name=rag_config.vector_db.collection_name,
                embedding_model=rag_config.embedding_model,
                chunk_size=rag_config.chunking.chunk_size,
                chunk_overlap=rag_config.chunking.chunk_overlap,
                top_k=rag_config.retrieval.top_k,
                score_threshold=rag_config.retrieval.score_threshold,
            )
            
            if app_state.llm_engine:
                app_state.rag_manager.set_llm_engine(app_state.llm_engine)
            
            logger.info("RAG manager initialized")
        
        app_state.initialized = True
        logger.info("TKLocalAI server ready!")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down TKLocalAI server...")
    
    if app_state.llm_engine:
        LLMEngineManager.shutdown()
    
    logger.info("Server shutdown complete")


# ===========================================
# FastAPI Application
# ===========================================

app = FastAPI(
    title="TKLocalAI",
    description="Local AI Assistant with LoRA fine-tuning and RAG",
    version="1.0.0",
    lifespan=lifespan,
)


def get_cors_origins():
    """Get CORS origins from settings or defaults."""
    if app_state.settings:
        return app_state.settings.server.cors_origins
    return ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080"]


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use get_cors_origins()
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================
# Authentication (Optional)
# ===========================================

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key if configured."""
    if not app_state.settings:
        return True
    
    api_key = app_state.settings.server.api_key
    if api_key is None:
        return True
    
    if authorization is None:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Support "Bearer <key>" format
    token = authorization.replace("Bearer ", "")
    if token != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


# ===========================================
# API Endpoints
# ===========================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head><title>TKLocalAI</title></head>
    <body style="background:#1a1a2e;color:#fff;font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;">
        <div style="text-align:center;">
            <h1>🤖 TKLocalAI</h1>
            <p>API Server is running!</p>
            <p>Try: <code style="background:#333;padding:4px 8px;border-radius:4px;">/health</code> or <code style="background:#333;padding:4px 8px;border-radius:4px;">/v1/chat/completions</code></p>
        </div>
    </body>
    </html>
    """)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if app_state.initialized else "initializing",
        model_loaded=app_state.llm_engine is not None and app_state.llm_engine.is_loaded,
        rag_enabled=app_state.rag_manager is not None,
    )


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    models = [{
        "id": "local",
        "object": "model",
        "owned_by": "local",
        "permission": [],
    }]
    
    if app_state.llm_engine and app_state.llm_engine.is_loaded:
        info = app_state.llm_engine.get_model_info()
        models[0]["id"] = Path(info["model_path"]).stem
    
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _: bool = Depends(verify_api_key),
):
    """
    OpenAI-compatible chat completions endpoint.
    Supports streaming and RAG.
    """
    if not app_state.llm_engine or not app_state.llm_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get system prompt
    persona = request.persona or app_state.settings.active_persona
    system_prompt = app_state.settings.get_system_prompt(persona)
    
    # Convert messages
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Check for RAG usage
    use_rag = request.use_rag and app_state.rag_manager is not None
    
    if request.stream:
        return EventSourceResponse(
            stream_chat_response(
                messages=messages,
                system_prompt=system_prompt,
                use_rag=use_rag,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
            )
        )
    else:
        return await generate_chat_response(
            messages=messages,
            system_prompt=system_prompt,
            use_rag=use_rag,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )


async def generate_chat_response(
    messages: List[Dict[str, str]],
    system_prompt: str,
    use_rag: bool,
    **kwargs,
) -> ChatCompletionResponse:
    """Generate a non-streaming chat response."""
    start_time = time.time()
    
    if use_rag and messages:
        # Get the last user message for RAG query
        user_message = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            None
        )
        
        if user_message:
            # Use RAG pipeline
            rag_response = app_state.rag_manager.query(
                query=user_message,
                system_prompt=system_prompt,
                **kwargs,
            )
            
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model="local",
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=rag_response.answer),
                    finish_reason="stop",
                )],
                usage=ChatCompletionUsage(
                    prompt_tokens=rag_response.tokens_used // 2,  # Approximate
                    completion_tokens=rag_response.tokens_used // 2,
                    total_tokens=rag_response.tokens_used,
                ),
            )
    
    # Standard chat without RAG
    result = app_state.llm_engine.chat(
        messages=messages,
        system_prompt=system_prompt,
        **kwargs,
    )
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model="local",
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=result.text),
            finish_reason=result.finish_reason,
        )],
        usage=ChatCompletionUsage(
            prompt_tokens=result.tokens_prompt,
            completion_tokens=result.tokens_generated,
            total_tokens=result.tokens_prompt + result.tokens_generated,
        ),
    )


async def stream_chat_response(
    messages: List[Dict[str, str]],
    system_prompt: str,
    use_rag: bool,
    **kwargs,
):
    """Stream chat response as SSE events."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    # Get user query for RAG
    user_message = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        ""
    )
    
    # If using RAG, get context first
    context = None
    has_context = False
    if use_rag and user_message and app_state.rag_manager:
        context = app_state.rag_manager.pipeline.retrieve(user_message)
        has_context = context.has_context if context else False
        
        # Modify the last user message to include context
        if has_context:
            augmented = app_state.rag_manager.pipeline.build_prompt(
                user_message, context, system_prompt
            )
            messages = messages[:-1] + [{"role": "user", "content": augmented}]
    
    # Stream from LLM using asyncio queue for true real-time streaming
    import asyncio
    import queue
    import threading
    
    logger.debug(f"Starting chat stream with {len(messages)} messages, system_prompt: {bool(system_prompt)}")
    
    chunk_queue = queue.Queue()
    
    def generate_chunks():
        try:
            for chunk in app_state.llm_engine.chat_stream(
                messages=messages,
                system_prompt=system_prompt if not has_context else None,
                **kwargs,
            ):
                chunk_queue.put(("chunk", chunk))
            chunk_queue.put(("done", None))
        except Exception as e:
            chunk_queue.put(("error", str(e)))
    
    # Start generation in background thread
    thread = threading.Thread(target=generate_chunks)
    thread.start()
    
    # Yield chunks as they arrive
    while True:
        try:
            msg_type, content = chunk_queue.get(timeout=0.1)
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
            
        if msg_type == "done":
            break
        elif msg_type == "error":
            logger.error(f"Generation error: {content}")
            break
        elif msg_type == "chunk":
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "local",
                "choices": [{
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }],
            }
            yield {"event": "message", "data": json.dumps(data)}
    
    thread.join()
    
    # Add sources at the end if RAG was used
    if has_context and context:
        sources_text = app_state.rag_manager.pipeline._format_sources_text(context.sources)
        data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "local",
            "choices": [{
                "index": 0,
                "delta": {"content": sources_text},
                "finish_reason": None,
            }],
        }
        yield {"event": "message", "data": json.dumps(data)}
    
    # Send finish event
    data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "local",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield {"event": "message", "data": json.dumps(data)}
    yield {"event": "message", "data": "[DONE]"}


@app.post("/v1/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    _: bool = Depends(verify_api_key),
):
    """Search the knowledge base."""
    if not app_state.rag_manager:
        raise HTTPException(status_code=503, detail="RAG not enabled")
    
    results = app_state.rag_manager.search(
        query=request.query,
        top_k=request.top_k,
        filter_metadata=request.filter,
    )
    
    return SearchResponse(
        results=[
            SearchResult(
                id=r.id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ],
        query=request.query,
        total=len(results),
    )


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    _: bool = Depends(verify_api_key),
):
    """Ingest text directly into the knowledge base."""
    if not app_state.rag_manager:
        raise HTTPException(status_code=503, detail="RAG not enabled")
    
    if not request.text:
        raise HTTPException(status_code=400, detail="Text content required")
    
    try:
        doc_id = app_state.rag_manager.ingest_text(
            text=request.text,
            title=request.title,
            metadata=request.metadata,
        )
        
        # Get chunk count
        doc = app_state.rag_manager._ingested_docs.get(doc_id)
        chunks = len(doc.chunks) if doc else 0
        
        return IngestResponse(
            success=True,
            document_id=doc_id,
            chunks_created=chunks,
            message=f"Successfully ingested '{request.title}'",
        )
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key),
):
    """Ingest a file into the knowledge base."""
    if not app_state.rag_manager:
        raise HTTPException(status_code=503, detail="RAG not enabled")
    
    try:
        # Save uploaded file temporarily
        temp_dir = app_state.settings.get_absolute_path("cache")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / file.filename
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Ingest the file
        document = app_state.rag_manager.ingest_file(temp_path)
        
        # Clean up temp file
        temp_path.unlink()
        
        if document:
            return IngestResponse(
                success=True,
                document_id=document.id,
                chunks_created=len(document.chunks),
                message=f"Successfully ingested '{file.filename}'",
            )
        else:
            return IngestResponse(
                success=False,
                message=f"Failed to process '{file.filename}'",
            )
            
    except Exception as e:
        logger.error(f"File ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest/directory", response_model=IngestResponse)
async def ingest_directory(
    path: str,
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    background_tasks: BackgroundTasks = None,
    _: bool = Depends(verify_api_key),
):
    """Ingest all documents from a directory."""
    if not app_state.rag_manager:
        raise HTTPException(status_code=503, detail="RAG not enabled")
    
    dir_path = Path(path)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    
    try:
        documents = app_state.rag_manager.ingest_directory(
            directory=dir_path,
            recursive=recursive,
            extensions=extensions,
        )
        
        total_chunks = sum(len(doc.chunks) for doc in documents)
        
        return IngestResponse(
            success=True,
            document_id=None,
            chunks_created=total_chunks,
            message=f"Successfully ingested {len(documents)} documents",
        )
        
    except Exception as e:
        logger.error(f"Directory ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/stats", response_model=StatsResponse)
async def get_stats(_: bool = Depends(verify_api_key)):
    """Get system statistics."""
    model_info = {}
    if app_state.llm_engine:
        model_info = app_state.llm_engine.get_model_info()
    
    rag_info = {}
    if app_state.rag_manager:
        rag_info = app_state.rag_manager.get_stats()
    
    return StatsResponse(
        model=model_info,
        rag=rag_info,
        server={
            "initialized": app_state.initialized,
            "version": "1.0.0",
        },
    )


@app.get("/v1/personas", response_model=PersonaResponse)
async def get_personas(_: bool = Depends(verify_api_key)):
    """Get available personas."""
    available = ["default", "coding", "research", "writing"]
    
    return PersonaResponse(
        active_persona=app_state.settings.active_persona,
        available_personas=available,
    )


@app.post("/v1/personas", response_model=PersonaResponse)
async def set_persona(
    request: PersonaRequest,
    _: bool = Depends(verify_api_key),
):
    """Set the active persona."""
    available = ["default", "coding", "research", "writing"]
    
    if request.persona not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid persona. Available: {available}"
        )
    
    app_state.settings.active_persona = request.persona
    
    return PersonaResponse(
        active_persona=request.persona,
        available_personas=available,
    )


@app.delete("/v1/rag/clear")
async def clear_rag(_: bool = Depends(verify_api_key)):
    """Clear all RAG data."""
    if not app_state.rag_manager:
        raise HTTPException(status_code=503, detail="RAG not enabled")
    
    app_state.rag_manager.clear()
    
    return {"success": True, "message": "RAG data cleared"}


# ===========================================
# Server Runner
# ===========================================

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    
    uvicorn.run(
        "src.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
