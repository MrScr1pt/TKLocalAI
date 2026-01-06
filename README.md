# TKLocalAI

**A fully local, private AI assistant with LoRA fine-tuning and RAG capabilities.**

Built for GPU-accelerated inference on your personal system with no cloud dependencies, no API calls, and complete privacy.

## Features

- 🔒 **100% Local & Private**: All processing happens on your machine
- 🚀 **GPU Accelerated**: Optimized for NVIDIA GPUs (RTX 3050+)
- 🎯 **LoRA Fine-tuning**: Customize model behavior with QLoRA
- 📚 **RAG Pipeline**: Ground responses in your own documents
- 🖥️ **Desktop UI**: Native application using PyWebView
- 🔌 **OpenAI-Compatible API**: Easy integration with existing tools
- 🎭 **Multiple Personas**: Switch between coding, research, writing modes

## System Requirements

- **GPU**: NVIDIA RTX 3050 (8GB) or better
- **RAM**: 16GB minimum
- **Storage**: ~10GB for model + dependencies
- **OS**: Windows 10/11
- **Python**: 3.10+

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/TKLocalAI.git
cd TKLocalAI

# Run setup script (Windows)
setup.bat
```

### 2. Download a Model

Download a GGUF model from HuggingFace:
- **Recommended**: [TheBloke/Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF)
- Download: `llama-3.1-8b-instruct.Q4_K_M.gguf` (~4.9 GB)
- Save to: `models/llama-3.1-8b-instruct.Q4_K_M.gguf`

### 3. Run

```bash
# Desktop application
run.bat desktop

# Or API server only
run.bat server
```

## Project Structure

```
TKLocalAI/
├── config/
│   └── settings.yaml      # Main configuration
├── src/
│   ├── config.py          # Configuration management
│   ├── llm_engine.py      # Model inference engine
│   ├── document_processor.py  # Document loading/chunking
│   ├── vector_store.py    # Qdrant vector database
│   ├── rag_pipeline.py    # RAG implementation
│   ├── finetuning.py      # QLoRA training
│   └── server.py          # FastAPI backend
├── ui/
│   ├── index.html         # Web UI
│   └── app.py             # Desktop wrapper
├── data/
│   ├── documents/         # Your documents for RAG
│   ├── vectordb/          # Vector database storage
│   └── training/          # Fine-tuning datasets
├── models/                # GGUF model files
├── adapters/              # LoRA adapters
├── main.py                # Entry point
├── setup.bat              # Windows setup script
├── run.bat                # Windows run script
└── requirements.txt       # Python dependencies
```

## Usage

### Desktop Application

```bash
run.bat desktop
```

Opens a native desktop window with the chat interface.

### API Server

```bash
run.bat server
```

Starts the FastAPI server at `http://127.0.0.1:8000`.

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/search` | POST | Search knowledge base |
| `/v1/ingest` | POST | Ingest text |
| `/v1/ingest/file` | POST | Ingest file |
| `/v1/ingest/directory` | POST | Ingest directory |
| `/v1/stats` | GET | System statistics |
| `/v1/personas` | GET/POST | Manage personas |

### Document Ingestion (RAG)

```bash
# Ingest a directory
run.bat ingest "C:\path\to\documents"

# Or via API
curl -X POST http://127.0.0.1:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content here", "title": "Document Title"}'
```

Supported file types:
- Documents: PDF, DOCX, TXT, MD
- Code: Python, JavaScript, TypeScript, and 20+ languages
- Data: JSON, YAML, XML

### Fine-tuning with QLoRA

1. **Prepare your dataset** in `data/training/dataset.jsonl`:

```jsonl
{"instruction": "Your prompt", "input": "", "output": "Desired response"}
{"instruction": "Another prompt", "input": "Optional context", "output": "Response"}
```

2. **Run fine-tuning**:

```bash
run.bat finetune
```

3. **Use the adapter** by updating `config/settings.yaml`:

```yaml
model:
  lora_path: "adapters/alignment-reduction-lora/final"
```

## Configuration

Edit `config/settings.yaml` to customize:

### Model Settings

```yaml
model:
  model_path: "models/llama-3.1-8b-instruct.Q4_K_M.gguf"
  context_length: 4096
  gpu_layers: 33  # Reduce if OOM errors
  lora_path: null  # Path to LoRA adapter
```

### Generation Parameters

```yaml
generation:
  max_tokens: 2048
  temperature: 0.7
  top_p: 0.9
  repeat_penalty: 1.1
```

### RAG Settings

```yaml
rag:
  enabled: true
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  retrieval:
    top_k: 5
    score_threshold: 0.3
```

### Personas

```yaml
prompts:
  default: "You are a helpful AI assistant..."
  coding: "You are an expert programmer..."
  research: "You are a research assistant..."
  writing: "You are a skilled writer..."

active_persona: "default"
```

## Fine-tuning Guide

### Creating Training Data

The goal is to create examples that encourage:
- Direct, helpful responses
- Reduced hedging and refusal for legitimate requests
- Technical accuracy without excessive caveats
- Neutral, factual presentation

**Example format:**

```jsonl
{"instruction": "How do I implement a binary search?", "input": "", "output": "Here's a binary search implementation:\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```\n\nTime complexity: O(log n)"}
```

### Training Process

1. Prepare 50-200+ examples in your domain
2. Run `run.bat finetune`
3. Training takes 30-60 minutes on RTX 3050
4. Adapter saved to `adapters/alignment-reduction-lora/final/`

### Memory Optimization

For 8GB VRAM:
- Use 4-bit quantization (default)
- Batch size 1 with gradient accumulation
- Reduce `max_seq_length` if needed

## Troubleshooting

### CUDA Out of Memory

1. Reduce `gpu_layers` in settings.yaml
2. Reduce `context_length`
3. Close other GPU applications

### Model Not Loading

1. Verify model path in settings.yaml
2. Ensure GGUF format (not safetensors)
3. Check file integrity (redownload if needed)

### Slow Inference

1. Ensure GPU is being used (check `gpu_layers > 0`)
2. Verify CUDA is installed correctly
3. Update GPU drivers

## Privacy & Security

- **No telemetry**: Zero data collection
- **No API calls**: Everything runs locally
- **No cloud**: All processing on your machine
- **Your data stays yours**: Documents never leave your system

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient LLM inference
- [Qdrant](https://qdrant.tech/) - Vector database
- [HuggingFace](https://huggingface.co/) - Models and transformers
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
