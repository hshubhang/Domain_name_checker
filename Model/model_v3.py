import json
import os
import sys
import torch
import re
from typing import List, Dict, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel
import psutil
import gc

class LoRADomainGeneratorV3:
    def __init__(self, 
                 lora_adapters_path: str = "../lora_adapters/v2/",
                 classifier_model_path: str = "./classifier_model_v3/"):
        """
        Initialize the LoRA domain generator v3 with ML-based classification.
        
        Args:
            lora_adapters_path: Path to the LoRA adapters directory (uses v2 model)
            classifier_model_path: Path to the trained classifier model
        """
        self.lora_adapters_path = lora_adapters_path
        self.classifier_model_path = classifier_model_path
        
        # Domain generation model (reusing v2)
        self.domain_model = None
        self.domain_tokenizer = None
        
        # Classification model
        self.classifier = None
        self.classifier_tokenizer = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Initializing LoRA Domain Generator v3 (ML-powered classification)")
        print(f"📁 LoRA adapters: {lora_adapters_path}")
        print(f"🤖 Classifier model: {classifier_model_path}")
        print(f"🖥️  Device: {self.device}")
        
        self._load_models()
    
    def _load_models(self):
        """Load both the classifier and domain generation models."""
        
        # Load classifier first (lightweight)
        print("🤖 Loading classification model...")
        try:
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(self.classifier_model_path)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(self.classifier_model_path)
            self.classifier.to(self.device)
            self.classifier.eval()  # Set to evaluation mode
            print("✅ Classifier loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading classifier: {e}")
            raise
        
        # Load domain generation model (heavy)
        print("📥 Loading domain generation model...")
        try:
            # Configure 8-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_quant_type="int8",
                bnb_8bit_use_double_quant=True,
            )
            
            # Load base model
            base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            self.domain_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            
            # Load LoRA adapters
            print("🔗 Loading LoRA adapters...")
            self.domain_model = PeftModel.from_pretrained(
                self.domain_model, 
                self.lora_adapters_path,
                torch_dtype=torch.float16,
            )
            
            # Load tokenizer
            self.domain_tokenizer = AutoTokenizer.from_pretrained(self.lora_adapters_path)
            
            # Ensure pad token is set
            if self.domain_tokenizer.pad_token is None:
                self.domain_tokenizer.pad_token = self.domain_tokenizer.eos_token
            
            print("✅ Domain generation model loaded successfully!")
            self._print_model_info()
            
        except Exception as e:
            print(f"❌ Error loading domain generation model: {e}")
            raise
    
    def _print_model_info(self):
        """Print model information."""
        try:
            # Get domain model parameters
            total_params = sum(p.numel() for p in self.domain_model.parameters())
            trainable_params = sum(p.numel() for p in self.domain_model.parameters() if p.requires_grad)
            
            print(f"📊 Model Info:")
            print(f"   Domain model parameters: {total_params:,}")
            print(f"   Domain trainable parameters: {trainable_params:,}")
            print(f"   Domain trainable %: {(trainable_params/total_params)*100:.3f}%")
            
            # Memory info
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                print(f"   GPU memory used: {memory_used:.1f}GB")
            
        except Exception as e:
            print(f"Warning: Could not get model info: {e}")
    
    def classify_business_description(self, text: str) -> Dict[str, Any]:
        """
        Classify a business description as legitimate or gibberish using the ML classifier.
        
        Args:
            text: Business description to classify
            
        Returns:
            Dictionary with classification result and confidence
        """
        try:
            # Tokenize input
            inputs = self.classifier_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.classifier(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            is_legitimate = predicted_class == 1
            label = "legitimate" if is_legitimate else "gibberish"
            
            return {
                "is_legitimate": is_legitimate,
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    "gibberish": probabilities[0][0].item(),
                    "legitimate": probabilities[0][1].item()
                }
            }
            
        except Exception as e:
            print(f"Error in classification: {e}")
            # Default to legitimate if classifier fails
            return {
                "is_legitimate": True,
                "label": "legitimate",
                "confidence": 0.5,
                "probabilities": {"gibberish": 0.5, "legitimate": 0.5},
                "error": str(e)
            }
    
    def _create_prompt(self, business_description: str) -> str:
        """
        Create a prompt for domain generation using the v2 format.
        
        Args:
            business_description: The business description
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Domain name generator for businesses.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Business: {business_description}
Generate 3 domains:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        return prompt
    
    def generate_domains_for_legitimate(self, business_description: str) -> List[str]:
        """
        Generate domain names for a legitimate business description.
        This is only called after classification confirms legitimacy.
        
        Args:
            business_description: Description of the business (pre-classified as legitimate)
            
        Returns:
            List of generated domain names
        """
        try:
            # Create prompt
            prompt = self._create_prompt(business_description)
            
            # Tokenize input
            inputs = self.domain_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=200  # Match training max length
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.domain_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.domain_tokenizer.eos_token_id,
                    eos_token_id=self.domain_tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.domain_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract domains from response
            domains = self._extract_domains_from_response(response, prompt)
            
            return domains
            
        except Exception as e:
            print(f"Error generating domains for '{business_description[:50]}...': {e}")
            return ["domain1.com", "domain2.com", "domain3.com"]  # Fallback
    
    def _extract_domains_from_response(self, response: str, prompt: str) -> List[str]:
        """
        Extract domain names from the model response using improved parsing for v3.
        
        Args:
            response: Full model response
            prompt: Original prompt (to remove from response)
            
        Returns:
            List of extracted domain names
        """
        try:
            # For Llama 3.1 chat format, look for the assistant response specifically
            if "assistant" in response:
                # Split by assistant and take the last part (the actual response)
                parts = response.split("assistant")
                if len(parts) > 1:
                    generated_text = parts[-1].strip()
                else:
                    generated_text = response.strip()
            else:
                # Fallback: try to remove prompt (v2 approach)
                generated_text = response.replace(prompt, "").strip()
            
            domains = []
            
            # Check if it's a comma-separated list first (common format)
            if ',' in generated_text and generated_text.count(',') >= 1:
                # Split by comma and clean each domain
                potential_domains = generated_text.split(',')
                for domain in potential_domains:
                    clean_domain = domain.strip()
                    if '.' in clean_domain and 3 <= len(clean_domain) <= 30:
                        # Remove trailing punctuation
                        clean_domain = clean_domain.rstrip('.,!?')
                        domains.append(clean_domain.lower())
            else:
                # Line-by-line parsing (fallback)
                lines = generated_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and '.' in line:
                        # Remove common prefixes/suffixes and clean up
                        domain = line.replace('- ', '').replace('• ', '').replace('* ', '')
                        domain = domain.replace('Example: ', '').replace('Domain: ', '')
                        domain = domain.replace('1. ', '').replace('2. ', '').replace('3. ', '')
                        
                        # Basic validation - should have a dot and reasonable length
                        if '.' in domain and 3 <= len(domain) <= 30:
                            # Remove any trailing punctuation
                            domain = domain.rstrip('.,!?')
                            domains.append(domain.lower())
            
            # Remove duplicates while preserving order
            seen = set()
            unique_domains = []
            for domain in domains:
                if domain not in seen and domain not in ['businessname.com', 'startup.ai', 'platform.co']:
                    seen.add(domain)
                    unique_domains.append(domain)
            
            # Return first 3 domains, pad with fallbacks only if we got nothing
            if len(unique_domains) == 0:
                return ["domain1.com", "domain2.com", "domain3.com"]
            
            # Pad with fallbacks if needed
            while len(unique_domains) < 3:
                unique_domains.append(f"domain{len(unique_domains)+1}.com")
            
            return unique_domains[:3]
            
        except Exception as e:
            print(f"Error extracting domains: {e}")
            return ["domain1.com", "domain2.com", "domain3.com"]
    
    def generate_domains(self, business_description: str) -> List[str]:
        """
        Sequential pipeline: classify first, then generate domains if legitimate.
        Classification happens implicitly - gibberish returns empty list.
        
        Args:
            business_description: Description of the business
            
        Returns:
            List of generated domain names (empty if gibberish detected)
        """
        print(f"🔍 Processing: '{business_description[:50]}...'")
        
        # Step 1: Classify the input (internal)
        classification = self.classify_business_description(business_description)
        
        print(f"🤖 Classification: {classification['label']} (confidence: {classification['confidence']:.3f})")
        
        # Step 2: Generate domains only if legitimate
        if classification['is_legitimate']:
            print(f"✅ Legitimate business detected, generating domains...")
            domains = self.generate_domains_for_legitimate(business_description)
        else:
            print(f"🚫 Gibberish detected, skipping domain generation")
            domains = []
        
        if domains:
            print(f"✅ Generated domains: {', '.join(domains)}")
        else:
            print(f"🚫 No domains generated")
        
        return domains
    
    def load_business_descriptions(self, input_file: str) -> List[str]:
        """
        Load business descriptions from a JSONL file.
        
        Args:
            input_file: Path to the input JSONL file
            
        Returns:
            List of business descriptions
        """
        business_descriptions = []
        
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
        Process all business descriptions from a file using the sequential pipeline.
        
        Args:
            input_file: Path to the input JSONL file containing business descriptions
            
        Returns:
            List of results with business descriptions and target domains (same format as v1/v2)
        """
        # Load business descriptions
        business_descriptions = self.load_business_descriptions(input_file)
        
        if not business_descriptions:
            print("No business descriptions to process")
            return []
        
        print(f"🚀 Processing {len(business_descriptions)} business descriptions with v3 sequential pipeline...")
        
        results = []
        
        # Track statistics
        total_processed = 0
        legitimate_count = 0
        gibberish_count = 0
        
        for i, business_description in enumerate(business_descriptions, 1):
            print(f"\n📋 Processing {i}/{len(business_descriptions)}")
            
            # Generate using sequential pipeline (returns list of domains)
            domains = self.generate_domains(business_description)
            
            # Create result in same format as v1/v2
            result = {
                "business_description": business_description,
                "target_domains": domains
            }
            
            results.append(result)
            
            # Update statistics (based on whether domains were generated)
            total_processed += 1
            if domains:  # If domains generated, it was classified as legitimate
                legitimate_count += 1
            else:  # If no domains, it was classified as gibberish
                gibberish_count += 1
            
            # Cleanup memory periodically
            if i % 10 == 0:
                self._cleanup_memory()
                print(f"📊 Progress: {legitimate_count} legitimate, {gibberish_count} gibberish ({i}/{len(business_descriptions)})")
        
        # Final statistics
        print(f"\n🎉 Completed processing {total_processed} business descriptions!")
        print(f"📊 Final Stats:")
        print(f"   ✅ Legitimate: {legitimate_count} ({legitimate_count/total_processed*100:.1f}%)")
        print(f"   🚫 Gibberish: {gibberish_count} ({gibberish_count/total_processed*100:.1f}%)")
        print(f"   🏭 Domains generated: {sum(1 for r in results if r['target_domains'])}")
        
        return results
    
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM errors."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = "../Model Outputs/model_v3_output_final.jsonl"):
        """
        Save the generated results to a JSONL file.
        
        Args:
            results: List of results to save
            output_file: Path to the output file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"💾 Results saved to: {output_file}")
            print(f"📊 Total results: {len(results)}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """
    Main function to run the LoRA domain generator v3 with sequential classification.
    
    Example usage:
    python model_v3.py
    """
    # Input file path - change this to your dataset
    input_file = "../data/dataset_injection_v4.jsonl"
    
    # Initialize the generator
    generator = LoRADomainGeneratorV3()
    
    # Process business descriptions
    results = generator.process_business_descriptions(input_file)
    
    # Save results
    generator.save_results(results)
    
    print("\n✅ Model v3 domain generation complete!")
    print("🤖 Used ML classification + domain generation pipeline")
    print("📊 Check output file for detailed classification and generation results")


if __name__ == "__main__":
    main() 