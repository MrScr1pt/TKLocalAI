"""
TKLocalAI - LLM Inference Engine
=================================
Handles model loading, inference, and LoRA adapter management.
Optimized for local GPU inference using llama-cpp-python.
"""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any
from dataclasses import dataclass
import threading
from loguru import logger

try:
    from llama_cpp import Llama
except ImportError:
    logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
    Llama = None


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    tokens_generated: int
    tokens_prompt: int
    finish_reason: str  # "stop", "length", "error"
    generation_time: float


@dataclass
class Message:
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str


class ChatTemplates:
    """Chat templates for different model types."""
    
    @staticmethod
    def format_llama2(messages: List[Message]) -> str:
        """Format messages for LLaMA 2 Chat models."""
        formatted = ""
        system_prompt = ""
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_content = msg.content
                if system_prompt:
                    # LLaMA 2 wraps system prompt with user message
                    formatted += f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_content} [/INST]"
                    system_prompt = ""
                else:
                    formatted += f"<s>[INST] {user_content} [/INST]"
            elif msg.role == "assistant":
                formatted += f" {msg.content} </s>"
        
        return formatted
    
    @staticmethod
    def format_llama3(messages: List[Message]) -> str:
        """Format messages for LLaMA 3 models."""
        # Don't add <|begin_of_text|> - llama-cpp adds it automatically
        formatted = ""
        
        for msg in messages:
            if msg.role == "system":
                formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{msg.content}<|eot_id|>"
            elif msg.role == "user":
                formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{msg.content}<|eot_id|>"
            elif msg.role == "assistant":
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg.content}<|eot_id|>"
        
        # Add assistant header for generation
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted
    
    @staticmethod
    def format_mistral(messages: List[Message]) -> str:
        """Format messages for Mistral/Mixtral models."""
        formatted = ""
        system_prompt = ""
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_content = msg.content
                if system_prompt:
                    user_content = f"{system_prompt}\n\n{user_content}"
                    system_prompt = ""
                formatted += f"[INST] {user_content} [/INST]"
            elif msg.role == "assistant":
                formatted += f" {msg.content}</s>"
        
        return formatted
    
    @staticmethod
    def format_chatml(messages: List[Message]) -> str:
        """Format messages for ChatML format (Qwen, etc.)."""
        formatted = ""
        
        for msg in messages:
            formatted += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        
        formatted += "<|im_start|>assistant\n"
        return formatted
    
    @classmethod
    def format(cls, messages: List[Message], model_type: str) -> str:
        """Format messages according to model type."""
        formatters = {
            "llama2": cls.format_llama2,
            "llama3": cls.format_llama3,
            "mistral": cls.format_mistral,
            "chatml": cls.format_chatml,
        }
        
        formatter = formatters.get(model_type, cls.format_llama2)
        return formatter(messages)


class LLMEngine:
    """
    Local LLM inference engine using llama-cpp-python.
    Supports GPU acceleration and LoRA adapters.
    """
    
    def __init__(
        self,
        model_path: str,
        context_length: int = 4096,
        gpu_layers: int = 33,
        threads: int = 8,
        batch_size: int = 512,
        lora_path: Optional[str] = None,
        model_type: str = "llama3",
        verbose: bool = False,
    ):
        """
        Initialize the LLM engine.
        
        Args:
            model_path: Path to the GGUF model file
            context_length: Maximum context window size
            gpu_layers: Number of layers to offload to GPU (-1 for all)
            threads: Number of CPU threads
            batch_size: Batch size for prompt processing
            lora_path: Optional path to LoRA adapter
            model_type: Model type for chat template (llama3, mistral, chatml)
            verbose: Enable verbose logging
        """
        if Llama is None:
            raise RuntimeError("llama-cpp-python is not installed")
        
        self.model_path = Path(model_path)
        self.context_length = context_length
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.batch_size = batch_size
        self.lora_path = Path(lora_path) if lora_path else None
        self.model_type = model_type
        self.verbose = verbose
        
        self._model: Optional[Llama] = None
        self._lock = threading.Lock()
        self._loaded = False
        
        logger.info(f"LLMEngine initialized with model: {model_path}")
    
    def load(self) -> None:
        """Load the model into memory."""
        if self._loaded:
            logger.warning("Model already loaded")
            return
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}...")
        logger.info(f"GPU layers: {self.gpu_layers}, Context: {self.context_length}")
        
        # Prepare LoRA if specified
        lora_path_str = str(self.lora_path) if self.lora_path and self.lora_path.exists() else None
        if lora_path_str:
            logger.info(f"Loading LoRA adapter from {lora_path_str}")
        
        try:
            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_length,
                n_gpu_layers=self.gpu_layers,
                n_threads=self.threads,
                n_batch=self.batch_size,
                lora_path=lora_path_str,
                verbose=self.verbose,
            )
            self._loaded = True
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._loaded:
            self.load()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repeat_penalty: Repetition penalty
            stop: Stop sequences
        
        Returns:
            GenerationResult with generated text and metadata
        """
        self._ensure_loaded()
        
        import time
        start_time = time.time()
        
        with self._lock:
            try:
                result = self._model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop or [],
                    echo=False,
                )
                
                generation_time = time.time() - start_time
                
                text = result["choices"][0]["text"]
                finish_reason = result["choices"][0].get("finish_reason", "stop")
                
                return GenerationResult(
                    text=text,
                    tokens_generated=result["usage"]["completion_tokens"],
                    tokens_prompt=result["usage"]["prompt_tokens"],
                    finish_reason=finish_reason,
                    generation_time=generation_time,
                )
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return GenerationResult(
                    text="",
                    tokens_generated=0,
                    tokens_prompt=0,
                    finish_reason="error",
                    generation_time=time.time() - start_time,
                )
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation token by token.
        
        Yields:
            Generated text chunks
        """
        self._ensure_loaded()
        
        # Use sync generator and wrap for async
        def sync_stream():
            with self._lock:
                for chunk in self._model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop or [],
                    stream=True,
                ):
                    text = chunk["choices"][0]["text"]
                    if text:
                        yield text
        
        return sync_stream()
    
    async def generate_stream_async(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming text generation.
        
        Yields:
            Generated text chunks
        """
        loop = asyncio.get_event_loop()
        
        # Run sync stream in thread pool
        def sync_generate():
            return list(self.generate_stream(
                prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, stop
            ))
        
        # For true async streaming, we'd need a more complex implementation
        # This is a simplified version that yields chunks
        chunks = await loop.run_in_executor(None, sync_generate)
        for chunk in chunks:
            yield chunk
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Generate chat completion.
        
        Args:
            messages: List of message dicts with "role" and "content"
            system_prompt: Optional system prompt to prepend
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repeat_penalty: Repetition penalty
            stop: Stop sequences
        
        Returns:
            GenerationResult with assistant response
        """
        # Convert to Message objects
        msg_objects = []
        
        # Add system prompt if provided
        if system_prompt:
            msg_objects.append(Message(role="system", content=system_prompt))
        
        # Add conversation messages
        for msg in messages:
            msg_objects.append(Message(role=msg["role"], content=msg["content"]))
        
        # Format according to model template
        prompt = ChatTemplates.format(msg_objects, self.model_type)
        
        # Generate
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ):
        """
        Stream chat completion.
        
        Yields:
            Generated text chunks
        """
        # Convert to Message objects
        msg_objects = []
        
        if system_prompt:
            msg_objects.append(Message(role="system", content=system_prompt))
        
        for msg in messages:
            msg_objects.append(Message(role=msg["role"], content=msg["content"]))
        
        prompt = ChatTemplates.format(msg_objects, self.model_type)
        
        return self.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": str(self.model_path),
            "model_type": self.model_type,
            "context_length": self.context_length,
            "gpu_layers": self.gpu_layers,
            "lora_path": str(self.lora_path) if self.lora_path else None,
            "loaded": self._loaded,
        }


class LLMEngineManager:
    """
    Singleton manager for the LLM engine.
    Provides global access to the model instance.
    """
    
    _instance: Optional['LLMEngineManager'] = None
    _engine: Optional[LLMEngine] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, **kwargs) -> LLMEngine:
        """Initialize the LLM engine with configuration."""
        if cls._engine is not None:
            logger.warning("Engine already initialized, returning existing instance")
            return cls._engine
        
        cls._engine = LLMEngine(**kwargs)
        return cls._engine
    
    @classmethod
    def get_engine(cls) -> Optional[LLMEngine]:
        """Get the current engine instance."""
        return cls._engine
    
    @classmethod
    def shutdown(cls) -> None:
        """Shutdown and cleanup the engine."""
        if cls._engine is not None:
            cls._engine.unload()
            cls._engine = None
            logger.info("LLM engine shutdown complete")


def create_engine_from_config(settings) -> LLMEngine:
    """
    Create an LLM engine from settings configuration.
    
    Args:
        settings: Settings object from config module
    
    Returns:
        Configured LLMEngine instance
    """
    return LLMEngineManager.initialize(
        model_path=str(settings.get_model_path()),
        context_length=settings.model.context_length,
        gpu_layers=settings.model.gpu_layers,
        threads=settings.model.threads,
        batch_size=settings.model.batch_size,
        lora_path=str(settings.get_lora_path()) if settings.get_lora_path() else None,
        model_type=settings.model.model_type,
    )
