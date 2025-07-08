import json
import os
import sys
import torch
import re
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel
import psutil
import gc

class LoRADomainGeneratorV2:
    def __init__(self, lora_adapters_path: str = "../lora_adapters/v2/"):
        """
        Initialize the LoRA domain generator v2 (instruction-enhanced).
        
        Args:
            lora_adapters_path: Path to the LoRA adapters directory
        """
        self.lora_adapters_path = lora_adapters_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Initializing LoRA Domain Generator v2 (instruction-enhanced)...")
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
    
    def is_gibberish(self, text: str) -> bool:
        """
        Rule-based gibberish detection using pattern matching.
        
        Args:
            text: Business description to check
            
        Returns:
            True if text appears to be gibberish
        """
        return self.get_gibberish_confidence(text) > 0.7
    
    def get_gibberish_confidence(self, text: str) -> float:
        """
        Get confidence score for gibberish detection (0.0 = legitimate, 1.0 = definitely gibberish).
        
        Args:
            text: Business description to check
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        confidence = 0.0
        
        # Pattern 1: Code/markup/technical syntax (high confidence)
        code_patterns = [
            r'[{}[\]<>(){}]',  # Brackets and braces
            r'<[^>]*>',  # HTML/XML tags
            r'[{}\[\]]+',  # Multiple brackets
            r'[=;{}]+',  # Code-like syntax
            r'console\.log|alert\(|function\(|import\s+|from\s+|SELECT\s+\*',  # Programming keywords
            r'<script>|</script>|<div>|</div>',  # HTML tags
            r'try\s*{|catch\s*\(|throw\s+new',  # Exception handling
        ]
        
        # Pattern 2: Network/system related (very high confidence)
        network_patterns = [
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
            r'localhost:\d+|127\.0\.0\.1',  # Local addresses
            r'http[s]?://[^\s]+',  # URLs
            r'ERROR:|EXCEPTION:|TIMEOUT:|FAILED:|CRASH:',  # Error messages
            r'java\.|javax\.|org\.springframework',  # Java packages
            r'npm\s+install|pip\s+install|apt\s+install',  # Package managers
        ]
        
        # Pattern 3: Random characters/nonsense (moderate confidence)
        random_patterns = [
            r'^[A-Z]{10,}$',  # All caps nonsense
            r'^[a-z]{20,}$',  # All lowercase nonsense
            r'^[0-9]{10,}$',  # All numbers
            r'[!@#$%^&*()_+]{5,}',  # Multiple special characters
            r'[A-Za-z0-9]{30,}',  # Very long strings without spaces
            r'^\w{3,}\d{3,}\w{3,}',  # Mixed letters and numbers pattern
        ]
        
        # Pattern 4: Technical jargon without business context (low confidence)
        tech_patterns = [
            r'blockchain|cryptocurrency|neural\s+network|machine\s+learning|quantum\s+computing',
            r'microservice|kubernetes|docker|containerized|distributed\s+hash',
            r'algorithm|encryption|decryption|hash\s+function|merkle\s+tree',
            r'binary|hexadecimal|base64|utf-8|ascii',
            r'malloc|segmentation\s+fault|memory\s+leak|buffer\s+overflow',
        ]
        
        # Pattern 5: Database/SQL related (high confidence)
        db_patterns = [
            r'SELECT\s+\*\s+FROM|INSERT\s+INTO|UPDATE\s+SET|DELETE\s+FROM',
            r'CREATE\s+TABLE|DROP\s+TABLE|ALTER\s+TABLE',
            r'mongodb|postgresql|mysql|sqlite|redis',
            r'db\.[a-z]+\.|\.find\(|\.insert\(|\.update\(',
        ]
        
        # Pattern 6: Foreign languages (moderate confidence - could be legitimate international business)
        foreign_patterns = [
            r'[\u0900-\u097F]+',  # Hindi/Devanagari
            r'[\u0590-\u05FF]+',  # Hebrew
            r'[\u0600-\u06FF]+',  # Arabic
            r'[\u4E00-\u9FFF]+',  # Chinese
            r'[\u3040-\u309F\u30A0-\u30FF]+',  # Japanese
            r'[\u0400-\u04FF]+',  # Cyrillic
            r'[\u0370-\u03FF]+',  # Greek
            r'[\u0E00-\u0E7F]+',  # Thai
        ]
        
        # Pattern 7: Excessive punctuation or symbols (high confidence)
        symbol_patterns = [
            r'[!?]{5,}',  # Multiple exclamation/question marks
            r'[\.]{5,}',  # Multiple periods
            r'[,]{3,}',  # Multiple commas
            r'[_\-]{5,}',  # Multiple underscores/dashes
            r'[♠♣♥♦♪♫♯♭]+',  # Special symbols
            r'[░▒▓█▲▼◄►]+',  # Block elements
        ]
        
        # Check patterns with different confidence weights
        pattern_groups = [
            (code_patterns, 0.8),      # High confidence
            (network_patterns, 0.9),   # Very high confidence
            (random_patterns, 0.7),    # Moderate confidence
            (tech_patterns, 0.3),      # Low confidence (could be legitimate tech business)
            (db_patterns, 0.8),        # High confidence
            (foreign_patterns, 0.4),   # Moderate confidence (could be international)
            (symbol_patterns, 0.8),    # High confidence
        ]
        
        for patterns, weight in pattern_groups:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    confidence = max(confidence, weight)
        
        # Additional checks with weighted confidence
        words = text.split()
        
        # Check for very long words (likely gibberish)
        if any(len(word) > 25 for word in words):
            confidence = max(confidence, 0.8)
        
        # Check for mostly non-alphabetic content
        if len(text) > 10:
            alpha_chars = sum(1 for char in text if char.isalpha())
            if alpha_chars / len(text) < 0.5:
                confidence = max(confidence, 0.6)
        
        # Check for excessive repeated characters
        if re.search(r'(.)\1{10,}', text):  # Same character repeated 10+ times
            confidence = max(confidence, 0.9)
        
        # Check for Lorem ipsum
        if 'lorem ipsum' in text_lower:
            confidence = max(confidence, 0.9)
        
        # Check for very short descriptions (could be incomplete)
        if len(text.strip()) < 10:
            confidence = max(confidence, 0.3)
        
        # Check for business-like keywords (reduce confidence)
        business_keywords = ['business', 'service', 'company', 'platform', 'app', 'website', 'startup', 'consulting', 'agency', 'solution']
        if any(keyword in text_lower for keyword in business_keywords):
            confidence = max(0, confidence - 0.2)
        
        return min(confidence, 1.0)
    
    def _create_prompt(self, business_description: str) -> str:
        """
        Create a prompt for domain generation using the simplified format.
        This matches the exact format used in training for v2.
        
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
    
    def generate_domains(self, business_description: str) -> List[str]:
        """
        Generate domain names for a business description using hybrid approach.
        
        Args:
            business_description: Description of the business
            
        Returns:
            List of generated domain names
        """
        try:
            # Hybrid Approach: Pre-processing gibberish filter
            if self.is_gibberish(business_description):
                print(f"🚫 Gibberish detected, skipping model: '{business_description[:50]}...'")
                return []  # Return empty list for detected gibberish
            
            print(f"✅ Legitimate business detected, using model: '{business_description[:50]}...'")
            
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
            
            # Debug: Print raw model response
            print(f"🔍 Raw model response: '{response}'")
            print(f"🔍 Generated part only: '{response.replace(prompt, '').strip()}'")
            
            # Extract domains from response
            domains = self._extract_domains_from_response(response, prompt)
            
            # Post-processing quality check
            if self._is_low_quality_output(domains, business_description):
                print(f"⚠️  Low quality output detected, returning minimal result")
                return []
            
            return domains
            
        except Exception as e:
            print(f"Error generating domains for '{business_description[:50]}...': {e}")
            return []
    
    def _is_low_quality_output(self, domains: List[str], business_description: str) -> bool:
        """
        Check if the generated domains are of low quality (indicating possible gibberish input that passed filter).
        
        Args:
            domains: Generated domain list
            business_description: Original business description
            
        Returns:
            True if output appears to be low quality
        """
        # Check if all domains are fallback domains
        fallback_domains = {'domain1.com', 'domain2.com', 'domain3.com'}
        if all(domain in fallback_domains for domain in domains):
            return True
        
        # Check if domains are too generic/nonsensical
        generic_keywords = {'error', 'timeout', 'fix', 'solution', 'random', 'test', 'example'}
        if any(any(keyword in domain.lower() for keyword in generic_keywords) for domain in domains):
            return True
        
        return False
    
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
            
            # For v2: Respect empty responses - don't force fallback domains
            # If model learned to output nothing for gibberish, respect that
            if len(unique_domains) == 0:
                # Check if the generated text is truly empty/whitespace
                if not generated_text.strip():
                    return []  # Return empty list for gibberish
                else:
                    # If there's text but no valid domains, use minimal fallbacks
                    return ["domain1.com"]
            
            # Return first 3 domains, pad with fallbacks only if we have some domains
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
        
        print(f"🚀 Processing {len(business_descriptions)} business descriptions...")
        
        results = []
        
        for i, business_description in enumerate(business_descriptions, 1):
            print(f"\n📋 Processing {i}/{len(business_descriptions)}: {business_description[:50]}...")
            
            # Generate domains
            domains = self.generate_domains(business_description)
            
            # Create result entry
            result = {
                "business_description": business_description,
                "target_domains": domains
            }
            
            results.append(result)
            
            # Print results
            if domains:
                print(f"✅ Generated domains: {', '.join(domains)}")
            else:
                print(f"🚫 No domains generated (gibberish detected or low quality)")
            
            # Cleanup memory periodically
            if i % 10 == 0:
                self._cleanup_memory()
        
        print(f"\n🎉 Completed processing {len(results)} business descriptions!")
        return results
    
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM errors."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = "../Model Outputs/model_v2_output_final.jsonl"):
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
    Main function to run the LoRA domain generator v2.
    
    Example usage:
    python model_v2.py
    """
    # Input file path - change this to your dataset
    input_file = "../data/dataset_injection_v4.jsonl"
    
    # Initialize the generator
    generator = LoRADomainGeneratorV2()
    
    # Process business descriptions
    results = generator.process_business_descriptions(input_file)
    
    # Save results
    generator.save_results(results)
    
    print("\n✅ Domain generation complete!")


if __name__ == "__main__":
    main() 