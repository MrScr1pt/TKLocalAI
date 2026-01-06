"""
TKLocalAI - Main Entry Point
==============================
Run the complete system: Server + Desktop UI
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_server():
    """Run only the API server."""
    from src.config import init_config
    from src.server import run_server as start_server
    
    settings = init_config()
    print(f"Starting TKLocalAI Server on http://{settings.server.host}:{settings.server.port}")
    start_server(settings.server.host, settings.server.port)


def run_desktop():
    """Run the desktop application with embedded server."""
    from ui.app import run_desktop_app
    run_desktop_app()


def run_finetune():
    """Run the fine-tuning pipeline."""
    from src.finetuning import run_finetuning, TrainingConfig
    from src.config import init_config
    
    settings = init_config()
    
    config = TrainingConfig(
        base_model=settings.finetuning.base_model,
        output_dir=str(settings.get_absolute_path(settings.finetuning.output_dir)),
        lora_r=settings.finetuning.qlora.r,
        lora_alpha=settings.finetuning.qlora.lora_alpha,
        lora_dropout=settings.finetuning.qlora.lora_dropout,
        target_modules=settings.finetuning.qlora.target_modules,
        num_epochs=settings.finetuning.training.num_epochs,
        batch_size=settings.finetuning.training.batch_size,
        learning_rate=settings.finetuning.training.learning_rate,
        dataset_path=str(settings.get_absolute_path(settings.finetuning.dataset_path)),
    )
    
    print("Starting QLoRA fine-tuning...")
    print(f"Base model: {config.base_model}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    
    adapter_path = run_finetuning(config=config)
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"Adapter saved to: {adapter_path}")
    print(f"\nTo use this adapter, update config/settings.yaml:")
    print(f'  model.lora_path: "{adapter_path}"')


def run_ingest(path: str, recursive: bool = True):
    """Ingest documents into RAG system."""
    from src.config import init_config
    from src.rag_pipeline import RAGManager
    
    settings = init_config()
    
    rag = RAGManager(
        vector_store_path=str(settings.get_absolute_path(settings.rag.vector_db.path)),
        collection_name=settings.rag.vector_db.collection_name,
        embedding_model=settings.rag.embedding_model,
        chunk_size=settings.rag.chunking.chunk_size,
        chunk_overlap=settings.rag.chunking.chunk_overlap,
    )
    
    target_path = Path(path)
    
    if target_path.is_file():
        doc = rag.ingest_file(target_path)
        if doc:
            print(f"✅ Ingested: {doc.filename} ({len(doc.chunks)} chunks)")
        else:
            print(f"❌ Failed to ingest: {path}")
    else:
        docs = rag.ingest_directory(target_path, recursive=recursive)
        print(f"✅ Ingested {len(docs)} documents")
    
    stats = rag.get_stats()
    print(f"\nKnowledge base: {stats['vector_count']} chunks from {stats['documents_ingested']} documents")


def download_model():
    """Download a recommended model."""
    print("Model Download Guide")
    print("=" * 50)
    print("\nRecommended for RTX 3050 (8GB VRAM):")
    print("\n1. Download from HuggingFace:")
    print("   https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF")
    print("\n2. Choose: llama-3.1-8b-instruct.Q4_K_M.gguf (~4.9 GB)")
    print("\n3. Save to: models/llama-3.1-8b-instruct.Q4_K_M.gguf")
    print("\nOr use huggingface-cli:")
    print("   pip install huggingface_hub")
    print("   huggingface-cli download TheBloke/Llama-3.1-8B-Instruct-GGUF \\")
    print("       llama-3.1-8b-instruct.Q4_K_M.gguf --local-dir models/")


def main():
    parser = argparse.ArgumentParser(
        description="TKLocalAI - Local AI Assistant with LoRA and RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py server          # Run API server only
  python main.py desktop         # Run desktop app with server
  python main.py finetune        # Run QLoRA fine-tuning
  python main.py ingest ./docs   # Ingest documents into RAG
  python main.py download        # Show model download instructions
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    subparsers.add_parser("server", help="Run the API server")
    
    # Desktop command
    subparsers.add_parser("desktop", help="Run the desktop application")
    
    # Finetune command
    subparsers.add_parser("finetune", help="Run QLoRA fine-tuning")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into RAG")
    ingest_parser.add_argument("path", help="Path to file or directory to ingest")
    ingest_parser.add_argument("--no-recursive", action="store_true", help="Don't process subdirectories")
    
    # Download command
    subparsers.add_parser("download", help="Show model download instructions")
    
    args = parser.parse_args()
    
    if args.command == "server":
        run_server()
    elif args.command == "desktop":
        run_desktop()
    elif args.command == "finetune":
        run_finetune()
    elif args.command == "ingest":
        run_ingest(args.path, not args.no_recursive)
    elif args.command == "download":
        download_model()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
