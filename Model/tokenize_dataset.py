import os
import json
import torch
from typing import Dict, Any
from transformers import AutoTokenizer
from datasets import Dataset
import gc

class DatasetTokenizer:
    """Pre-tokenize and save the dataset for memory-efficient training."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        
    def load_tokenizer(self):
        """Load the tokenizer."""
        print(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=True
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Tokenizer loaded successfully")
    
    def format_prompt(self, item: Dict[str, Any]) -> str:
        """Format a data item into a training prompt with explicit empty target training."""
        business_description = item.get("business_description", "")
        domains = item.get("target_domains", [])
        
        # Format domains as a comma-separated list, or empty for gibberish
        domain_list = ", ".join(domains) if domains else ""
        
        # Enhanced format for v4 - adds TLD guidance and gibberish handling
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Domain name generator for businesses. Use appropriate TLDs (.com, .ai, .co, .org, .legal, .health, .bio, etc.) based on industry. For unclear or gibberish input, return no domains.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Business: {business_description}
Generate 3 domains:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{domain_list}<|eot_id|>"""
        
        return prompt
    
    def tokenize_dataset(self, input_path: str, output_dir: str, max_seq_length: int = 200, batch_size: int = 10):
        """
        Tokenize dataset and save in efficient format.
        
        Args:
            input_path: Path to input JSONL file
            output_dir: Directory to save tokenized dataset
            max_seq_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing (to control memory)
        """
        if self.tokenizer is None:
            self.load_tokenizer()
        
        print(f"Tokenizing dataset: {input_path}")
        print(f"Output directory: {output_dir}")
        print(f"Max sequence length: {max_seq_length}")
        print(f"Processing batch size: {batch_size}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process data in batches to control memory usage
        all_tokenized_data = []
        
        with open(input_path, 'r') as f:
            batch_data = []
            
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    item = json.loads(line)
                    prompt = self.format_prompt(item)
                    batch_data.append(prompt)
                    
                    # Process batch when it reaches batch_size
                    if len(batch_data) >= batch_size:
                        tokenized_batch = self._tokenize_batch(batch_data, max_seq_length)
                        all_tokenized_data.extend(tokenized_batch)
                        
                        print(f"Processed {len(all_tokenized_data)} examples...")
                        
                        # Clear batch and run garbage collection
                        batch_data = []
                        gc.collect()
                
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                    continue
            
            # Process remaining data in the last batch
            if batch_data:
                tokenized_batch = self._tokenize_batch(batch_data, max_seq_length)
                all_tokenized_data.extend(tokenized_batch)
        
        print(f"✅ Tokenized {len(all_tokenized_data)} examples")
        
        # Create dataset and save
        dataset = Dataset.from_list(all_tokenized_data)
        dataset.save_to_disk(output_dir)
        
        # Save metadata
        metadata = {
            "total_examples": len(all_tokenized_data),
            "max_seq_length": max_seq_length,
            "model_name": self.model_name,
            "tokenizer_config": {
                "vocab_size": self.tokenizer.vocab_size,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Dataset saved to {output_dir}")
        print(f"Total examples: {len(all_tokenized_data)}")
        
        return dataset
    
    def _tokenize_batch(self, texts, max_seq_length):
        """Tokenize a batch of texts."""
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,  # Don't pad yet to save space
            max_length=max_seq_length,
            return_tensors=None
        )
        
        # Convert to list of dictionaries
        batch_data = []
        for i in range(len(texts)):
            item = {
                "input_ids": tokenized["input_ids"][i],
                "attention_mask": tokenized["attention_mask"][i],
                "labels": tokenized["input_ids"][i].copy()  # For language modeling
            }
            batch_data.append(item)
        
        return batch_data


def main():
    """Main function to tokenize the dataset for v3 training."""
    # Configuration for v3 (comprehensive edge cases)
    input_path = "../data/augmented_dataset_v3.jsonl"
    output_dir = "../data/tokenized_augmented_dataset_v3"
    max_seq_length = 200
    batch_size = 10  # Small batch size to control memory
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"❌ Dataset file not found: {input_path}")
        return
    
    # Initialize tokenizer
    tokenizer = DatasetTokenizer()
    
    try:
        # Tokenize and save dataset
        dataset = tokenizer.tokenize_dataset(
            input_path=input_path,
            output_dir=output_dir,
            max_seq_length=max_seq_length,
            batch_size=batch_size
        )
        
        print("\n✅ Dataset tokenization completed successfully!")
        print(f"Tokenized dataset saved to: {output_dir}")
        print("\nNow you can use the pre-tokenized dataset for training.")
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 