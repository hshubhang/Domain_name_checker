import json
import os
import sys
import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel
import psutil
import gc

class LoRADomainGenerator:
    def __init__(self, lora_adapters_path: str = "../lora_adapters/v1/"):
        """
        Initialize the LoRA domain generator.
        
        Args:
            lora_adapters_path: Path to the LoRA adapters directory
        """
        self.lora_adapters_path = lora_adapters_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Initializing LoRA Domain Generator...")
        print(f"📁 LoRA adapters: {lora_adapters_path}")
        print(f"🖥️  Device: {self.device}")
        
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load the base model with LoRA adapters and tokenizer."""
        try:
            print("📥 Loading base model and tokenizer...")
            
            # Configure 8-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_quant_type="nf4",
                bnb_8bit_use_double_quant=True,
            )
            
            # Load base model
            base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            
            # Load LoRA adapters
            print("🔗 Loading LoRA adapters...")
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.lora_adapters_path,
                torch_dtype=torch.float16,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.lora_adapters_path)
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Model and LoRA adapters loaded successfully!")
            self._print_model_info()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _print_model_info(self):
        """Print model information."""
        try:
            # Get model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"📊 Model Info:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Trainable %: {(trainable_params/total_params)*100:.3f}%")
            
            # Memory info
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                print(f"   GPU memory used: {memory_used:.1f}GB")
            
        except Exception as e:
            print(f"Warning: Could not get model info: {e}")
    
    def _create_prompt(self, business_description: str) -> str:
        """
        Create a prompt for domain generation using the same format as model_v0.
        
        Args:
            business_description: The business description
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a domain name expert. Generate 3 creative and professional domain name suggestions for this business:

Business Description: "{business_description}"

Requirements:
- Domain names should be relevant to the business
- Keep them short and memorable (ideally 6-15 characters)
- Use professional TLDs like .com, .co, .ai, .io, .net
- Make them brandable and easy to type
- Avoid hyphens and numbers when possible

Output ONLY a simple list of domain names, one per line, without any additional text or explanations.

Example format:
businessname.com
startup.ai
platform.co"""
        
        return prompt
    
    def generate_domains(self, business_description: str) -> List[str]:
        """
        Generate domain names for a business description.
        
        Args:
            business_description: Description of the business
            
        Returns:
            List of generated domain names
        """
        try:
            # Create prompt
            prompt = self._create_prompt(business_description)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=200  # Match training max length
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract domains from response
            domains = self._extract_domains_from_response(response, prompt)
            
            return domains
            
        except Exception as e:
            print(f"Error generating domains for '{business_description[:50]}...': {e}")
            return []
    
    def _extract_domains_from_response(self, response: str, prompt: str) -> List[str]:
        """
        Extract domain names from the model response using the proven baseline approach.
        
        Args:
            response: Full model response
            prompt: Original prompt (to remove from response)
            
        Returns:
            List of extracted domain names
        """
        try:
            # Remove the original prompt from response to get only generated text
            generated_text = response.replace(prompt, "").strip()
            
            domains = []
            lines = generated_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line and '.' in line:
                    # Remove common prefixes/suffixes and clean up (same as model_v0)
                    domain = line.replace('- ', '').replace('• ', '').replace('* ', '')
                    domain = domain.replace('Example: ', '').replace('Domain: ', '')
                    domain = domain.replace('1. ', '').replace('2. ', '').replace('3. ', '')
                    
                    # Basic validation - should have a dot and reasonable length
                    if '.' in domain and 3 <= len(domain) <= 30:
                        # Remove any trailing punctuation
                        domain = domain.rstrip('.,!?')
                        # Skip example domains from prompt template
                        if domain not in ['businessname.com', 'startup.ai', 'platform.co']:
                            domains.append(domain.lower())
            
            # Remove duplicates while preserving order (same as model_v0)
            seen = set()
            unique_domains = []
            for domain in domains:
                if domain not in seen:
                    seen.add(domain)
                    unique_domains.append(domain)
            
            # Return first 3 domains, pad with fallbacks if needed
            if len(unique_domains) < 3:
                while len(unique_domains) < 3:
                    unique_domains.append(f"domain{len(unique_domains)+1}.com")
            
            return unique_domains[:3]
            
        except Exception as e:
            print(f"Error extracting domains: {e}")
            return ["domain1.com", "domain2.com", "domain3.com"]
    
    def load_business_descriptions(self, input_file: str) -> List[str]:
        """
        Load business descriptions from a JSONL file.
        
        Args:
            input_file: Path to the input JSONL file
            
        Returns:
            List of business descriptions
        """
        business_descriptions = []
        
        # 🔍 THIS IS WHERE THE DATASET INPUT HAPPENS 🔍
        # The input_file parameter specifies the path to your dataset
        # You'll need to create this file with your 100 business descriptions
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        
                        # Extract business description
                        if 'business_description' in entry:
                            business_descriptions.append(entry['business_description'])
                        elif 'description' in entry:  # Alternative field name
                            business_descriptions.append(entry['description'])
                        else:
                            print(f"Warning: Line {line_num} missing business description field")
                            
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"❌ Error: Could not find input file {input_file}")
            print(f"📝 Please create this file with your business descriptions in JSONL format:")
            print(f'   {{"business_description": "Your business description here"}}')
            return []
        
        print(f"📥 Loaded {len(business_descriptions)} business descriptions from {input_file}")
        return business_descriptions
    
    def process_business_descriptions(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Process all business descriptions from a file and generate domains.
        
        Args:
            input_file: Path to the input JSONL file containing business descriptions
            
        Returns:
            List of results with business descriptions and generated domains
        """
        # Load business descriptions
        business_descriptions = self.load_business_descriptions(input_file)
        
        if not business_descriptions:
            print("No business descriptions to process")
            return []
        
        results = []
        
        print(f"\n🚀 Processing {len(business_descriptions)} business descriptions...")
        
        for i, description in enumerate(business_descriptions, 1):
            print(f"\n[{i}/{len(business_descriptions)}] Processing: {description[:60]}...")
            
            try:
                # Generate domains
                domains = self.generate_domains(description)
                
                result = {
                    "business_description": description,
                    "target_domains": domains
                }
                
                results.append(result)
                
                print(f"✅ Generated domains: {domains}")
                
                # Memory cleanup every 10 iterations
                if i % 10 == 0:
                    self._cleanup_memory()
                    
            except Exception as e:
                print(f"❌ Error processing description {i}: {e}")
                # Add empty result to maintain consistency
                results.append({
                    "business_description": description,
                    "target_domains": [],
                    "error": str(e)
                })
        
        return results
    
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Print memory status
        ram_percent = psutil.virtual_memory().percent
        print(f"🧠 Memory: {ram_percent:.1f}% RAM used")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            print(f"     {gpu_memory:.1f}GB GPU used")
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = "../Model Outputs/model_v1_output_final.jsonl"):
        """
        Save the results to a JSONL file.
        
        Args:
            results: List of results to save
            output_file: Output file path
        """
        try:
            # Create Model Outputs directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"\n🎯 Results saved to {output_file}")
            print(f"📊 Total businesses processed: {len(results)}")
            
            # Calculate success rate
            successful = len([r for r in results if r.get('target_domains') and len(r['target_domains']) > 0])
            success_rate = (successful / len(results)) * 100 if results else 0
            print(f"✅ Success rate: {success_rate:.1f}% ({successful}/{len(results)})")
            
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")


def main():
    """
    Main function to run the LoRA domain generation process.
    """
    print("=== LoRA Fine-tuned Domain Generator (Model v1) ===")
    
    # 🔍 Using dataset_injection_v2.jsonl as input 🔍
    input_dataset = "../data/dataset_injection_v4.jsonl"
    
    print(f"📖 Reading business descriptions from {input_dataset}...")
    
    try:
        # Initialize the LoRA generator
        generator = LoRADomainGenerator()
        
        # Process all business descriptions
        results = generator.process_business_descriptions(input_dataset)
        
        # Save results
        generator.save_results(results)
        
        # Display sample results
        if results:
            print(f"\n📝 Sample results:")
            for i, result in enumerate(results[:3]):  # Show first 3 results
                print(f"\n{i+1}. Business: {result['business_description'][:80]}...")
                print(f"   Domains: {result['target_domains']}")
        
        return results
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None


if __name__ == "__main__":
    # Make sure your LoRA adapters are trained and saved in lora_adapters/v1/
    # Create your test dataset at the path specified in main()
    
    results = main() 