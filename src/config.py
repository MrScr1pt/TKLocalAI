"""
TKLocalAI - Configuration Management
=====================================
Handles loading, validation, and access to configuration settings.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml
from loguru import logger


# ===========================================
# Configuration Models
# ===========================================

class ModelConfig(BaseModel):
    """LLM model configuration."""
    model_path: str = "models/llama-3.1-8b-instruct.Q4_K_M.gguf"
    context_length: int = 4096
    gpu_layers: int = 33
    threads: int = 8
    batch_size: int = 512
    lora_path: Optional[str] = None
    model_type: str = "llama3"


class GenerationConfig(BaseModel):
    """Text generation parameters."""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop_sequences: List[str] = Field(default_factory=lambda: [
        "<|eot_id|>", "<|end_of_text|>", "Human:", "User:"
    ])


class PromptsConfig(BaseModel):
    """System prompts configuration."""
    default: str = ""
    coding: str = ""
    research: str = ""
    writing: str = ""


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    path: str = "data/vectordb"
    collection_name: str = "documents"
    vector_size: int = 384


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " "])


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k: int = 5
    score_threshold: float = 0.3
    include_sources: bool = True


class RAGConfig(BaseModel):
    """RAG system configuration."""
    enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


class QLoRAConfig(BaseModel):
    """QLoRA parameters."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bits: int = 4


class TrainingConfig(BaseModel):
    """Training parameters."""
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    save_steps: int = 100
    logging_steps: int = 10


class FineTuningConfig(BaseModel):
    """Fine-tuning configuration."""
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    output_dir: str = "adapters"
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dataset_path: str = "data/training/dataset.jsonl"


class ServerConfig(BaseModel):
    """API server configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: [
        "http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080"
    ])
    api_key: Optional[str] = None


class UIConfig(BaseModel):
    """Desktop UI configuration."""
    title: str = "TKLocalAI Assistant"
    width: int = 1200
    height: int = 800
    min_width: int = 800
    min_height: int = 600
    maximized: bool = False
    theme: str = "dark"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/tklocalai.log"
    max_size: str = "10 MB"
    retention: str = "7 days"


class PathsConfig(BaseModel):
    """Paths configuration."""
    models: str = "models"
    adapters: str = "adapters"
    data: str = "data"
    documents: str = "data/documents"
    vectordb: str = "data/vectordb"
    training: str = "data/training"
    logs: str = "logs"
    cache: str = "cache"


class Settings(BaseModel):
    """Main settings container."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    active_persona: str = "default"
    rag: RAGConfig = Field(default_factory=RAGConfig)
    finetuning: FineTuningConfig = Field(default_factory=FineTuningConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    
    # Base directory (set at runtime)
    base_dir: Path = Field(default_factory=lambda: Path.cwd())
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert a relative path to absolute based on base_dir."""
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return self.base_dir / path
    
    def get_model_path(self) -> Path:
        """Get absolute path to the model file."""
        return self.get_absolute_path(self.model.model_path)
    
    def get_lora_path(self) -> Optional[Path]:
        """Get absolute path to LoRA adapter if configured."""
        if self.model.lora_path:
            return self.get_absolute_path(self.model.lora_path)
        return None
    
    def get_system_prompt(self, persona: Optional[str] = None) -> str:
        """Get the system prompt for the specified or active persona."""
        persona = persona or self.active_persona
        return getattr(self.prompts, persona, self.prompts.default)
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.paths.models,
            self.paths.adapters,
            self.paths.data,
            self.paths.documents,
            self.paths.vectordb,
            self.paths.training,
            self.paths.logs,
            self.paths.cache,
        ]
        for dir_path in directories:
            full_path = self.get_absolute_path(dir_path)
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {full_path}")


# ===========================================
# Configuration Loader
# ===========================================

class ConfigLoader:
    """Handles loading and managing configuration."""
    
    _instance: Optional['ConfigLoader'] = None
    _settings: Optional[Settings] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, config_path: Optional[str] = None, base_dir: Optional[Path] = None) -> Settings:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the config file. Defaults to config/settings.yaml
            base_dir: Base directory for resolving relative paths
        
        Returns:
            Settings object with loaded configuration
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        
        if config_path is None:
            config_path = base_dir / "config" / "settings.yaml"
        else:
            config_path = Path(config_path)
        
        # Load YAML configuration
        config_data = {}
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Configuration file not found: {config_path}, using defaults")
        
        # Create settings with base_dir
        config_data['base_dir'] = base_dir
        
        # Parse nested configs
        cls._settings = Settings(**config_data)
        
        # Ensure all directories exist
        cls._settings.ensure_directories()
        
        return cls._settings
    
    @classmethod
    def get_settings(cls) -> Settings:
        """Get the current settings, loading if necessary."""
        if cls._settings is None:
            cls.load()
        return cls._settings
    
    @classmethod
    def reload(cls, config_path: Optional[str] = None) -> Settings:
        """Reload configuration from file."""
        cls._settings = None
        return cls.load(config_path)
    
    @classmethod
    def update_setting(cls, key_path: str, value: Any) -> None:
        """
        Update a setting at runtime.
        
        Args:
            key_path: Dot-separated path to the setting (e.g., "model.temperature")
            value: New value for the setting
        """
        settings = cls.get_settings()
        keys = key_path.split('.')
        
        # Navigate to the parent object
        obj = settings
        for key in keys[:-1]:
            obj = getattr(obj, key)
        
        # Set the value
        setattr(obj, keys[-1], value)
        logger.info(f"Updated setting {key_path} = {value}")


# ===========================================
# Convenience Functions
# ===========================================

def get_settings() -> Settings:
    """Get the current application settings."""
    return ConfigLoader.get_settings()


def load_config(config_path: Optional[str] = None, base_dir: Optional[Path] = None) -> Settings:
    """Load configuration from file."""
    return ConfigLoader.load(config_path, base_dir)


# Initialize settings when module is imported
def init_config(base_dir: Optional[Path] = None) -> Settings:
    """Initialize configuration (call once at application startup)."""
    if base_dir is None:
        # Find the project root (where config folder is)
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "config" / "settings.yaml").exists():
                base_dir = current
                break
            current = current.parent
        else:
            base_dir = Path.cwd()
    
    return ConfigLoader.load(base_dir=base_dir)
