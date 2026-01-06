"""
TKLocalAI - QLoRA Fine-Tuning Pipeline
========================================
Parameter-efficient fine-tuning using QLoRA for reducing alignment bias
and customizing model behavior for domain-specific tasks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

if TYPE_CHECKING:
    from datasets import Dataset

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch not installed")

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )
    from trl import SFTTrainer
    from datasets import Dataset as DatasetClass, load_dataset
    TRAINING_AVAILABLE = True
except ImportError as e:
    TRAINING_AVAILABLE = False
    DatasetClass = None
    logger.warning(f"Training dependencies not available: {e}")


@dataclass
class TrainingExample:
    """A single training example."""
    instruction: str
    input: str = ""
    output: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }
    
    def format_prompt(self, include_response: bool = True) -> str:
        """Format as a training prompt."""
        if self.input:
            prompt = f"""### Instruction:
{self.instruction}

### Input:
{self.input}

### Response:
"""
        else:
            prompt = f"""### Instruction:
{self.instruction}

### Response:
"""
        
        if include_response:
            prompt += self.output
        
        return prompt


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    # Model settings
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    output_dir: str = "adapters"
    adapter_name: str = "alignment-reduction-lora"
    
    # QLoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    quantization_bits: int = 4
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    save_steps: int = 100
    logging_steps: int = 10
    
    # Dataset
    dataset_path: str = "data/training/dataset.jsonl"


class DatasetBuilder:
    """
    Builds and manages training datasets for alignment reduction.
    Provides templates for creating instruction-response pairs that
    encourage direct, helpful responses without excessive hedging.
    """
    
    # Templates for reducing alignment bias
    ALIGNMENT_REDUCTION_TEMPLATES = {
        "direct_answer": {
            "description": "Encourage direct answers without hedging",
            "examples": [
                TrainingExample(
                    instruction="What is the boiling point of water?",
                    output="The boiling point of water is 100°C (212°F) at standard atmospheric pressure (1 atm or 101.325 kPa)."
                ),
                TrainingExample(
                    instruction="How do I create a Python virtual environment?",
                    output="""To create a Python virtual environment:

1. Open your terminal
2. Navigate to your project directory
3. Run: `python -m venv venv`
4. Activate it:
   - Windows: `venv\\Scripts\\activate`
   - Linux/Mac: `source venv/bin/activate`
5. Install packages with pip as usual"""
                ),
            ]
        },
        
        "technical_directness": {
            "description": "Technical questions answered directly without unnecessary caveats",
            "examples": [
                TrainingExample(
                    instruction="Explain how a hash table works",
                    output="""A hash table is a data structure that maps keys to values using a hash function.

**How it works:**
1. A hash function converts the key into an array index
2. The value is stored at that index
3. For retrieval, the same hash function finds the index

**Collision handling:**
- Chaining: Store colliding entries in a linked list at the same index
- Open addressing: Find the next available slot

**Time complexity:** O(1) average for insert, delete, and lookup. O(n) worst case with many collisions.

**Use cases:** Caches, database indexing, symbol tables, sets."""
                ),
            ]
        },
        
        "reduced_moralizing": {
            "description": "Answer questions neutrally without unsolicited moral commentary",
            "examples": [
                TrainingExample(
                    instruction="What is the history of nuclear energy?",
                    output="""Nuclear energy development timeline:

**1930s-1940s:** Discovery of nuclear fission (Hahn, Meitner). Manhattan Project develops first nuclear reactors and weapons.

**1950s:** First commercial nuclear power plants. USS Nautilus becomes first nuclear submarine.

**1960s-1970s:** Rapid expansion of nuclear power globally. ~50 reactors built per year at peak.

**1979:** Three Mile Island partial meltdown in Pennsylvania, USA.

**1986:** Chernobyl disaster in USSR, most severe nuclear accident in history.

**1990s-2000s:** Slowdown in new construction due to costs and public concern.

**2011:** Fukushima Daiichi disaster following tsunami in Japan.

**Present:** ~440 reactors operating worldwide, providing ~10% of global electricity. New designs focus on passive safety and smaller modular reactors."""
                ),
            ]
        },
        
        "helpful_compliance": {
            "description": "Be helpful with legitimate requests without refusing prematurely",
            "examples": [
                TrainingExample(
                    instruction="How do computer networks detect intrusions?",
                    output="""Network intrusion detection systems (IDS) use several methods:

**Signature-based detection:**
- Compares network traffic against database of known attack patterns
- Fast and accurate for known threats
- Cannot detect novel attacks

**Anomaly-based detection:**
- Establishes baseline of normal network behavior
- Flags deviations from the baseline
- Can detect unknown attacks but has higher false positives

**Common techniques:**
- Packet inspection (headers and payloads)
- Flow analysis (traffic patterns)
- Protocol analysis (detecting protocol violations)
- Behavioral analysis (user/entity behavior analytics)

**Popular tools:** Snort, Suricata, Zeek (formerly Bro), OSSEC

**Deployment:** Network-based (NIDS) monitors traffic at network points; Host-based (HIDS) monitors individual systems."""
                ),
            ]
        },
        
        "factual_neutrality": {
            "description": "Present information factually without bias",
            "examples": [
                TrainingExample(
                    instruction="Compare different programming paradigms",
                    output="""**Imperative Programming:**
- Describes how to achieve results through statements
- Examples: C, Pascal
- Good for: System programming, performance-critical code

**Object-Oriented Programming:**
- Organizes code around objects with data and methods
- Examples: Java, C++, Python
- Good for: Large codebases, modeling real-world entities

**Functional Programming:**
- Treats computation as mathematical function evaluation
- Emphasizes immutability and pure functions
- Examples: Haskell, Clojure, Erlang
- Good for: Concurrent systems, data transformation

**Declarative Programming:**
- Describes what result is needed, not how to achieve it
- Examples: SQL, HTML, Prolog
- Good for: Database queries, UI layouts, logic problems

**Multi-paradigm languages** (Python, JavaScript, Scala) allow mixing approaches based on the problem."""
                ),
            ]
        },
    }
    
    def __init__(self, output_path: str = "data/training"):
        """Initialize the dataset builder."""
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.examples: List[TrainingExample] = []
    
    def add_example(self, instruction: str, output: str, input_text: str = "") -> None:
        """Add a training example."""
        self.examples.append(TrainingExample(
            instruction=instruction,
            input=input_text,
            output=output,
        ))
    
    def add_examples_from_template(self, template_name: str) -> None:
        """Add examples from a built-in template."""
        if template_name in self.ALIGNMENT_REDUCTION_TEMPLATES:
            template = self.ALIGNMENT_REDUCTION_TEMPLATES[template_name]
            self.examples.extend(template["examples"])
            logger.info(f"Added {len(template['examples'])} examples from '{template_name}'")
        else:
            logger.warning(f"Template not found: {template_name}")
    
    def add_all_templates(self) -> None:
        """Add all built-in template examples."""
        for template_name in self.ALIGNMENT_REDUCTION_TEMPLATES:
            self.add_examples_from_template(template_name)
    
    def load_from_jsonl(self, file_path: str) -> None:
        """Load examples from a JSONL file."""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Dataset file not found: {file_path}")
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.examples.append(TrainingExample(
                        instruction=data.get('instruction', ''),
                        input=data.get('input', ''),
                        output=data.get('output', ''),
                    ))
        
        logger.info(f"Loaded {len(self.examples)} examples from {file_path}")
    
    def save_to_jsonl(self, file_name: str = "dataset.jsonl") -> Path:
        """Save examples to a JSONL file."""
        output_file = self.output_path / file_name
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in self.examples:
                json.dump(example.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(self.examples)} examples to {output_file}")
        return output_file
    
    def get_formatted_examples(self) -> List[str]:
        """Get all examples formatted as prompts."""
        return [ex.format_prompt(include_response=True) for ex in self.examples]
    
    def create_hf_dataset(self) -> 'Dataset':
        """Create a HuggingFace Dataset from examples."""
        if not TRAINING_AVAILABLE:
            raise RuntimeError("Training dependencies not installed")
        
        formatted = self.get_formatted_examples()
        return DatasetClass.from_dict({"text": formatted})
    
    @staticmethod
    def get_template_info() -> Dict[str, str]:
        """Get descriptions of available templates."""
        return {
            name: template["description"]
            for name, template in DatasetBuilder.ALIGNMENT_REDUCTION_TEMPLATES.items()
        }


class QLoRATrainer:
    """
    QLoRA fine-tuning trainer.
    Handles model loading, training, and adapter saving.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        if not TRAINING_AVAILABLE:
            raise RuntimeError(
                "Training dependencies not installed. "
                "Install with: pip install torch transformers peft trl datasets bitsandbytes"
            )
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed")
        
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("CUDA not available. Training will be slow on CPU.")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.adapter_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self) -> None:
        """Load the base model with QLoRA configuration."""
        logger.info(f"Loading base model: {self.config.base_model}")
        
        # Quantization config for 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
    
    def train(self, dataset: Dataset) -> None:
        """
        Train the model on the dataset.
        
        Args:
            dataset: HuggingFace Dataset with training examples
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Starting training with {len(dataset)} examples")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none",  # Disable wandb etc
            remove_unused_columns=False,
        )
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=True,
        )
        
        # Train
        logger.info("Training started...")
        self.trainer.train()
        
        # Save the adapter
        self.save_adapter()
        
        logger.info("Training completed!")
    
    def save_adapter(self) -> Path:
        """Save the trained LoRA adapter."""
        adapter_path = self.output_dir / "final"
        
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        
        # Save training config
        config_path = adapter_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "base_model": self.config.base_model,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": self.config.target_modules,
                "trained_at": datetime.now().isoformat(),
                "purpose": "alignment_reduction",
            }, f, indent=2)
        
        logger.info(f"Adapter saved to: {adapter_path}")
        return adapter_path
    
    def cleanup(self) -> None:
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.trainer is not None:
            del self.trainer
            self.trainer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("GPU memory cleaned up")


def run_finetuning(
    dataset_path: Optional[str] = None,
    config: Optional[TrainingConfig] = None,
    use_templates: bool = True,
) -> Path:
    """
    Run the complete fine-tuning pipeline.
    
    Args:
        dataset_path: Path to JSONL dataset (optional)
        config: Training configuration (uses defaults if not provided)
        use_templates: Whether to include built-in templates
    
    Returns:
        Path to the saved adapter
    """
    config = config or TrainingConfig()
    
    # Build dataset
    builder = DatasetBuilder(config.output_dir)
    
    if use_templates:
        builder.add_all_templates()
    
    if dataset_path:
        builder.load_from_jsonl(dataset_path)
    elif Path(config.dataset_path).exists():
        builder.load_from_jsonl(config.dataset_path)
    
    if len(builder.examples) == 0:
        logger.warning("No training examples. Using built-in templates only.")
        builder.add_all_templates()
    
    # Create HuggingFace dataset
    dataset = builder.create_hf_dataset()
    
    # Initialize trainer
    trainer = QLoRATrainer(config)
    
    try:
        # Train
        trainer.train(dataset)
        
        return trainer.output_dir / "final"
        
    finally:
        trainer.cleanup()


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for TKLocalAI")
    parser.add_argument("--dataset", type=str, help="Path to JSONL dataset")
    parser.add_argument("--base-model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output", type=str, default="adapters")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--no-templates", action="store_true", help="Don't use built-in templates")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        base_model=args.base_model,
        output_dir=args.output,
        num_epochs=args.epochs,
    )
    
    adapter_path = run_finetuning(
        dataset_path=args.dataset,
        config=config,
        use_templates=not args.no_templates,
    )
    
    print(f"\nAdapter saved to: {adapter_path}")
    print("\nTo use this adapter, update config/settings.yaml:")
    print(f'  lora_path: "{adapter_path}"')
