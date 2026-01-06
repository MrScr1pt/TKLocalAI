"""
TKLocalAI - Package initialization
"""

from .config import init_config, get_settings, Settings
from .llm_engine import LLMEngine, LLMEngineManager, create_engine_from_config
from .rag_pipeline import RAGManager, RAGPipeline
from .finetuning import QLoRATrainer, DatasetBuilder, TrainingConfig

__version__ = "1.0.0"
__all__ = [
    "init_config",
    "get_settings",
    "Settings",
    "LLMEngine",
    "LLMEngineManager",
    "create_engine_from_config",
    "RAGManager",
    "RAGPipeline",
    "QLoRATrainer",
    "DatasetBuilder",
    "TrainingConfig",
]
