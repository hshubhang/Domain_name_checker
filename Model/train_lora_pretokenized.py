import os
import json
import torch
import psutil  # For RAM monitoring
import gc  # For garbage collection
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict, load_from_disk
import numpy as np

# Import our model loader
from load_model import ModelLoader

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads variables from .env file in the project root
except ImportError:
    # dotenv not installed, environment variables should be set manually
    print("Note: python-dotenv not installed. Install with 'pip install python-dotenv' to use .env files.")


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA fine-tuning with pre-tokenized data."""
    
    # LoRA specific parameters - v4 enhanced
    lora_r: int = 32  # Rank of the adaptation (doubled from v1)
    lora_alpha: int = 64  # LoRA scaling parameter (doubled from v1)
    lora_dropout: float = 0.05  # Dropout probability for LoRA layers (reduced from v1)
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Training parameters - v4 enhanced
    num_epochs: int = 4  # Increased for v4 due to larger dataset
    learning_rate: float = 1e-4  # Reduced from v1 for better stability
    batch_size: int = 1  # Conservative default for memory
    gradient_accumulation_steps: int = 16  # Doubled from v1
    warmup_steps: int = 100  # Doubled from v1
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    
    # Model and data paths - v4 paths
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenized_dataset_path: str = "../data/tokenized_augmented_dataset_v3"  # Path to v3 tokenized dataset
    output_dir: str = "lora_adapters/v4"
    
    # Training optimization
    use_gradient_checkpointing: bool = True
    use_fp16: bool = True
    dataloader_num_workers: int = 0  # Reduce to 0 for debugging
    dataloader_pin_memory: bool = False  # Reduce memory pressure
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            # LoRA Configuration
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            
            # Training Hyperparameters
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            
            # Model and Data
            "model_name": self.model_name,
            "tokenized_dataset_path": self.tokenized_dataset_path,
            "output_dir": self.output_dir,
            
            # Training Optimization
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_fp16": self.use_fp16,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory
        }


class PreTokenizedDomainTrainer:
    """
    LoRA fine-tuning trainer for domain name generation using pre-tokenized data.
    
    This class handles loading pre-tokenized data and training the model
    for domain name generation tasks with minimal memory usage.
    """
    
    def __init__(self, config: LoRATrainingConfig):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: LoRA training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.peft_model = None
        self.dataset = None
        
    def load_model(self):
        """Load the base model with quantization."""
        print("Loading base model...")
        self._print_memory_usage("Before loading model")
        loader = ModelLoader(self.config.model_name)
        self.model, self.tokenizer = loader.load_quantized_model(load_in_8bit=True)
        
        # Prepare model for LoRA training
        self.model.train()  # type: ignore # Set to training mode
        
        # Ensure all base model parameters are frozen (except embeddings for resize)
        for param in self.model.parameters():  # type: ignore
            param.requires_grad = False
            
        self._print_memory_usage("After loading model")
        print("✅ Base model loaded successfully")
        
    def prepare_lora_model(self):
        """Configure and apply LoRA to the 8-bit quantized model."""
        print("Configuring LoRA for 8-bit quantized model...")
        
        # Prepare model for LoRA training (critical for quantized models)
        from peft import prepare_model_for_kbit_training
        
        # Prepare the quantized model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)  # type: ignore
        
        # Configure LoRA with simple settings (let PEFT handle complexities)
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to the prepared model
        self.peft_model = get_peft_model(self.model, lora_config)  # type: ignore
        
        # Let PEFT handle gradient checkpointing
        if self.config.use_gradient_checkpointing:
            self.peft_model.gradient_checkpointing_enable()  # type: ignore
        
        # Print trainable parameters
        self._print_trainable_parameters()
        
        # Debug: Check gradient flow
        self._debug_gradient_flow()
        
        print("✅ LoRA configuration applied successfully")
        
    def _print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        if self.peft_model is None:
            return
            
        trainable_params = 0
        all_param = 0
        
        for _, param in self.peft_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"\n📊 LoRA Parameters:")
        print(f"   Trainable params: {trainable_params:,}")
        print(f"   All params: {all_param:,}")
        print(f"   Trainable %: {100 * trainable_params / all_param:.2f}%")
        
    def _debug_gradient_flow(self):
        """Debug gradient flow for LoRA parameters."""
        if self.peft_model is None:
            return
            
        print(f"\n🔍 Gradient Flow Debug:")
        lora_params = 0
        trainable_params = 0
        
        for name, param in self.peft_model.named_parameters():
            if param.requires_grad:
                trainable_params += 1
                if 'lora_' in name:
                    lora_params += 1
                    print(f"   ✓ {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        
        print(f"   Total trainable params: {trainable_params}")
        print(f"   LoRA params: {lora_params}")
        
        if lora_params == 0:
            print("   ⚠️  WARNING: No LoRA parameters found with requires_grad=True!")
        else:
            print(f"   ✅ Found {lora_params} LoRA parameters ready for training")
        
    def _print_memory_usage(self, stage: str):
        """Print current memory usage for debugging."""
        # GPU Memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"\n🔍 Memory Usage ({stage}):")
            print(f"   GPU Allocated: {allocated:.2f} GB")
            print(f"   GPU Reserved: {reserved:.2f} GB") 
            print(f"   GPU Total: {max_memory:.2f} GB")
            print(f"   GPU Free: {max_memory - reserved:.2f} GB")
        
        # RAM Memory
        ram = psutil.virtual_memory()
        ram_used = ram.used / 1024**3  # GB
        ram_total = ram.total / 1024**3  # GB
        ram_percent = ram.percent
        print(f"   RAM Used: {ram_used:.2f} GB / {ram_total:.2f} GB ({ram_percent:.1f}%)")
        print(f"   RAM Available: {ram.available / 1024**3:.2f} GB")
        
        # Warning if memory usage is high
        if ram_percent > 85:
            print(f"⚠️  HIGH RAM USAGE WARNING: {ram_percent:.1f}%")
    
    def _cleanup_memory(self):
        """Force garbage collection and clear cache to free up memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _save_jupyter_configs(self, lora_output_dir: str, checkpoint_dir: str, dataset, version: str):
        """Save comprehensive configuration optimized for Jupyter experimentation."""
        import platform
        import transformers
        import torch
        import peft
        
        # Get trainable parameters info for model metadata
        trainable_params = 0
        all_params = 0
        if self.peft_model:
            for _, param in self.peft_model.named_parameters():
                all_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
        
        # 1. Main Model Config (Jupyter-friendly, matching baseline format)
        model_config = {
            "model_name": f"model_{version}",
            "version": "1.0",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "description": f"LoRA fine-tuned domain generation model - {version}",
            
            "model_config": {
                "provider": "local_lora",
                "base_model": self.config.model_name,
                "quantization": "8-bit",
                "lora_adapters_path": f"../lora_adapters/{version}/"
            },
            
            "lora_config": {
                "r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "target_modules": self.config.target_modules,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            
            "training_config": {
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "warmup_steps": self.config.warmup_steps,
                "use_fp16": self.config.use_fp16,
                "use_gradient_checkpointing": self.config.use_gradient_checkpointing
            },
            
            "model_stats": {
                "trainable_params": trainable_params,
                "total_params": all_params,
                "trainable_percentage": round((trainable_params / all_params) * 100, 3) if all_params > 0 else 0,
                "dataset_size": len(dataset)
            },
            
            "generation_config": {
                "domains_per_business": 3,
                "prompt_template": "instruction_following",
                "max_tokens": 256,
                "temperature": 0.7
            },
            
            "input_data": {
                "source_file": "data/dataset_v1.jsonl",
                "total_businesses": len(dataset),
                "description": "Pre-tokenized business descriptions for LoRA fine-tuning"
            },
            
            "output_config": {
                "output_file": f"Model Outputs/model_{version}_output.jsonl",
                "format": "jsonl",
                "fields": ["business_description", "target_domains"]
            },
            
            "evaluation_config": {
                "evaluator_model": "gpt-4",
                "metrics": ["relevance", "clarity", "professionalism", "memorability", "tld_suitability"],
                "eval_output": f"evaluator outputs/evaluated_model_{version}_output.jsonl"
            },
            
            "system_info": {
                "platform": platform.platform(),
                "pytorch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "peft_version": peft.__version__,
                "cuda_available": torch.cuda.is_available(),
                "training_date": datetime.now().isoformat()
            },
            
            "jupyter_utils": {
                "load_command": f"load_model_version('{version}')",
                "compare_with": ["v0"],
                "experiment_ready": True
            }
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            model_config["system_info"]["gpu_name"] = torch.cuda.get_device_name(0)
            model_config["system_info"]["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        
        # Save main config to checkpoints (for Jupyter loading)
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # 2. Save detailed training config to LoRA directory
        detailed_config = self.config.to_dict()
        detailed_config["training_completed"] = datetime.now().isoformat()
        detailed_config["version"] = version
        
        training_config_path = os.path.join(lora_output_dir, "training_config.json")
        with open(training_config_path, 'w') as f:
            json.dump(detailed_config, f, indent=2)
        
        print(f"✅ Saved Jupyter-optimized configs:")
        print(f"   📁 Main config: {config_path}")
        print(f"   📁 LoRA config: {training_config_path}")
        print(f"   🔬 Jupyter ready: load_model_version('{version}')")
        
    def load_tokenized_dataset(self):
        """
        Load pre-tokenized dataset from disk.
        
        Returns:
            Loaded tokenized dataset ready for training
        """
        print(f"Loading pre-tokenized dataset from: {self.config.tokenized_dataset_path}")
        
        if not os.path.exists(self.config.tokenized_dataset_path):
            raise FileNotFoundError(f"Tokenized dataset not found: {self.config.tokenized_dataset_path}")
        
        # Load metadata first
        metadata_path = os.path.join(self.config.tokenized_dataset_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"📋 Dataset Info:")
            print(f"   Total examples: {metadata.get('total_examples', 'Unknown')}")
            print(f"   Max sequence length: {metadata.get('max_seq_length', 'Unknown')}")
            print(f"   Model name: {metadata.get('model_name', 'Unknown')}")
        
        # Load the actual dataset
        self._print_memory_usage("Before loading tokenized dataset")
        
        try:
            loaded_data = load_from_disk(self.config.tokenized_dataset_path)
            
            # Handle both Dataset and DatasetDict cases
            if isinstance(loaded_data, DatasetDict):
                # If it's a DatasetDict, take the first split (usually 'train')
                dataset = loaded_data[list(loaded_data.keys())[0]]
            else:
                # It's already a Dataset
                dataset = loaded_data
            
            print(f"✅ Loaded tokenized dataset with {len(dataset)} examples")
            
            # Memory cleanup
            self._cleanup_memory()
            self._print_memory_usage("After loading tokenized dataset")
            
            self.dataset = dataset
            return dataset
            
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenized dataset: {e}")
        
    def train(self):
        """Execute the LoRA fine-tuning process using pre-tokenized data."""
        print("Starting LoRA fine-tuning with pre-tokenized data...")
        
        # Print initial memory usage
        self._print_memory_usage("Training start")
        
        # Load model and dataset
        self.load_model()
        self.prepare_lora_model()
        dataset = self.load_tokenized_dataset()
        
        # Memory cleanup before training setup
        self._cleanup_memory()
        self._print_memory_usage("After dataset loading")
        
        # Create versioned output directories (aligned with baseline structure)
        version = "v4"  # v4 with enhanced dataset and training
        lora_output_dir = f"../lora_adapters/{version}"
        checkpoint_dir = f"../checkpoints/model_{version}"
        
        # Create directories
        os.makedirs(lora_output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save comprehensive configuration (Jupyter-optimized)
        self._save_jupyter_configs(lora_output_dir, checkpoint_dir, dataset, version)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=lora_output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.use_fp16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Ensure tokenizer has padding configured
        if self.tokenizer.pad_token is None:  # type: ignore
            self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore
            print(f"Set tokenizer pad_token to: {self.tokenizer.pad_token}")  # type: ignore
        
        # Create data collator with padding enabled
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,  # type: ignore
            mlm=False,  # We're doing causal language modeling
            pad_to_multiple_of=8,  # Efficient padding for better performance
        )
        
        # Split dataset for train/eval
        train_dataset = dataset
        
        # Ensure we have a proper Dataset for eval selection
        if isinstance(dataset, Dataset):
            eval_dataset = dataset.select(range(min(50, len(dataset))))  # Use first 50 examples for eval
        else:
            # Fallback: use the same dataset for both train and eval
            eval_dataset = dataset
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Final memory check before training
        self._print_memory_usage("Before training start")
        
        # Start training
        print(f"Training output directory: {lora_output_dir}")
        print("🚀 Starting training...")
        
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(lora_output_dir)  # type: ignore
        
        print(f"✅ Training completed! Model saved to: {lora_output_dir}")
        
        return lora_output_dir
        
    def generate_sample(self, business_description: str, max_length: int = 256) -> str:
        """
        Generate a sample domain name using the fine-tuned model.
        
        Args:
            business_description: Business description for domain generation
            max_length: Maximum length of generated text
            
        Returns:
            Generated domain names
        """
        if self.peft_model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call train() first.")
        
        # Format the prompt (enhanced v4 format)
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Domain name generator for businesses.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Business: {business_description}
Generate 3 domains:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)  # type: ignore
        
        # Generate
        with torch.no_grad():
            outputs = self.peft_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id  # type: ignore
            )
        
        # Decode the output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore
        
        # Extract only the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            response = generated_text
        
        return response


def train_lora_model_pretokenized(config: Optional[LoRATrainingConfig] = None) -> str:
    """
    Convenience function to train a LoRA model with pre-tokenized data.
    
    Args:
        config: Training configuration (uses default if None)
        
    Returns:
        Path to trained model
    """
    if config is None:
        config = LoRATrainingConfig()
    
    trainer = PreTokenizedDomainTrainer(config)
    return trainer.train()


# Example usage
if __name__ == "__main__":
    print("=== LoRA Fine-tuning v4 with Pre-tokenized Data ===")
    
    # Create enhanced v4 training configuration
    config = LoRATrainingConfig(
        # Enhanced v4 parameters
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        num_epochs=4,
        learning_rate=1e-4,
        batch_size=1,  # Conservative batch size
        gradient_accumulation_steps=16,  # Optimized for v4
        warmup_steps=100,  # Optimized for v4
        tokenized_dataset_path="../data/tokenized_augmented_dataset_v3",
    )
    
    # Check if tokenized dataset exists
    if not os.path.exists(config.tokenized_dataset_path):
        print(f"❌ Tokenized dataset not found: {config.tokenized_dataset_path}")
        print("\n📝 Please run the tokenization script first:")
        print("python tokenize_dataset.py")
        exit(1)
    
    try:
        # Train the model
        model_path = train_lora_model_pretokenized(config)
        print(f"\n✅ LoRA fine-tuning v4 completed successfully!")
        print(f"Model saved to: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Pre-tokenized the dataset (python tokenize_dataset.py)")
        print("2. Required dependencies: pip install transformers torch peft datasets bitsandbytes accelerate psutil") 