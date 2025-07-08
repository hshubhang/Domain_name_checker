import os
import torch
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads variables from .env file in the project root
except ImportError:
    # dotenv not installed, environment variables should be set manually
    print("Note: python-dotenv not installed. Install with 'pip install python-dotenv' to use .env files.")


class ModelLoader:
    """
    Model loader for Meta Llama 3 8B Instruct with 8-bit quantization support.
    
    This class handles loading the base model with memory-efficient quantization
    configurations for fine-tuning and inference.
    """
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """
        Initialize the model loader.
        
        Args:
            model_name: HuggingFace model identifier for Llama 3 8B Instruct
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_quantized_model(self, load_in_8bit: bool = True, device_map: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model with 8-bit quantization for memory efficiency.
        
        Args:
            load_in_8bit: Whether to use 8-bit quantization (reduces memory usage)
            device_map: Device mapping strategy for multi-GPU setups
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            RuntimeError: If model loading fails
            ValueError: If model name is invalid
        """
        try:
            print(f"Loading model: {self.model_name}")
            print(f"Using 8-bit quantization: {load_in_8bit}")
            
            # Configure quantization settings
            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                
                # Load model with 8-bit quantization
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                
                print("✅ Model loaded with 8-bit quantization")
            else:
                # Load model without quantization (full precision)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                
                print("✅ Model loaded in full precision")
            
            # Load tokenizer
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("✅ Tokenizer loaded successfully")
            
            # Store references
            self.model = model
            self.tokenizer = tokenizer
            
            # Print model information
            self._print_model_info()
            
            return model, tokenizer
            
        except Exception as e:
            error_msg = f"Error loading model {self.model_name}: {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _print_model_info(self):
        """Print model information and memory usage."""
        if self.model is None:
            return
            
        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"\n📊 Model Information:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Model device: {next(self.model.parameters()).device}")
            print(f"   Model dtype: {next(self.model.parameters()).dtype}")
            
            # Check GPU memory if available
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                print(f"   GPU memory allocated: {memory_allocated:.2f} GB")
                print(f"   GPU memory reserved: {memory_reserved:.2f} GB")
                
        except Exception as e:
            print(f"Warning: Could not retrieve model info: {e}")
    
    def get_model_info(self) -> dict:
        """
        Get model information as a dictionary.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None or self.tokenizer is None:
            return {"error": "Model not loaded"}
        
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info = {
                "model_name": self.model_name,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": str(next(self.model.parameters()).device),
                "dtype": str(next(self.model.parameters()).dtype),
                "vocab_size": self.tokenizer.vocab_size,
                "pad_token": self.tokenizer.pad_token
            }
            
            if torch.cuda.is_available():
                info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
                info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            
            return info
            
        except Exception as e:
            return {"error": f"Could not retrieve model info: {e}"}


def load_base_model(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
                   quantized: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Convenience function to load the base model with quantization.
    
    Args:
        model_name: HuggingFace model identifier
        quantized: Whether to use 8-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = ModelLoader(model_name)
    return loader.load_quantized_model(load_in_8bit=quantized)


# Example usage
if __name__ == "__main__":
    print("=== Model Loading Test ===")
    
    try:
        # Load model with 8-bit quantization
        model, tokenizer = load_base_model(quantized=True)
        
        print("\n✅ Model loading test successful!")
        print("Model ready for fine-tuning or inference.")
        
    except Exception as e:
        print(f"\n❌ Model loading test failed: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install transformers torch bitsandbytes accelerate") 