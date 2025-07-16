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

# NSFW Content Filter
try:
    from better_profanity import profanity
    PROFANITY_AVAILABLE = True
    print("🛡️  Better-profanity NSFW filter available")
except ImportError:
    PROFANITY_AVAILABLE = False
    print("⚠️  Better-profanity not installed. NSFW filtering will use keyword fallback only.")
    print("   Install with: pip install better-profanity")

class LoRADomainGeneratorV4:
    def __init__(self, 
                 lora_adapters_path: str = "../lora_adapters/v4/",
                 classifier_model_path: str = "./classifier_model_v3"):
        """
        Initialize the LoRA domain generator v4 with ML-based classification.
        
        Args:
            lora_adapters_path: Path to the LoRA adapters directory (uses v4 model)
            classifier_model_path: Path to the trained classifier model
        """
        self.lora_adapters_path = lora_adapters_path
        self.classifier_model_path = classifier_model_path
        
        # Domain generation model (using v4)
        self.domain_model = None
        self.domain_tokenizer = None
        
        # Classification model
        self.classifier = None
        self.classifier_tokenizer = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize NSFW filter
        self.profanity_filter = None
        self._init_nsfw_filter()
        
        print(f"🔧 Initializing LoRA Domain Generator v4 (Comprehensive edge cases)")
        print(f"📁 LoRA adapters: {lora_adapters_path}")
        print(f"🤖 Classifier model: {classifier_model_path}")
        print(f"🖥️  Device: {self.device}")
        
        self._load_models()
    
    def _load_models(self):
        """Load both the classifier and domain generation models."""
        
        # Load classifier first (lightweight)
        print("🤖 Loading classification model...")
        try:
            # Ensure we have absolute path for the classifier
            import os
            classifier_abs_path = os.path.abspath(self.classifier_model_path)
            print(f"   Loading classifier from: {classifier_abs_path}")
            
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(
                classifier_abs_path, 
                local_files_only=True,
                trust_remote_code=False
            )
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                classifier_abs_path, 
                local_files_only=True,
                trust_remote_code=False
            )
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
        Create a prompt for domain generation using the v4 format.
        
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
            business_description: The business description
            
        Returns:
            List of generated domain names
        """
        try:
            prompt = self._create_prompt(business_description)
            
            # Tokenize the prompt
            inputs = self.domain_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.domain_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.domain_tokenizer.pad_token_id,
                    eos_token_id=self.domain_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2,
                )
            
            # Decode the response
            response = self.domain_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract domains from the response
            domains = self._extract_domains_from_response(response, prompt)
            
            return domains
            
        except Exception as e:
            print(f"Error generating domains: {e}")
            return []
    
    def _extract_domains_from_response(self, response: str, prompt: str) -> List[str]:
        """
        Extract domain names from the model response.
        
        Args:
            response: The full model response
            prompt: The original prompt (to remove it from response)
            
        Returns:
            List of extracted domain names
        """
        try:
            # Remove the prompt from the response
            assistant_response = response.replace(prompt, "").strip()
            
            # Split by common separators and clean up
            potential_domains = re.split(r'[,\n\r\t\|]', assistant_response)
            
            domains = []
            for domain in potential_domains:
                domain = domain.strip()
                
                # Skip empty strings and common non-domain words
                if not domain or domain.lower() in ['', 'domains:', 'domain:', 'suggestions:', 'options:']:
                    continue
                
                # Remove numbering and bullets
                domain = re.sub(r'^\d+[\.\)]\s*', '', domain)
                domain = re.sub(r'^[-•*]\s*', '', domain)
                domain = domain.strip()
                
                # Check if it looks like a domain
                if '.' in domain and len(domain) > 3:
                    # Remove quotes and extra text
                    domain = re.sub(r'^["\']|["\']$', '', domain)
                    
                    # Extract just the domain part if there's extra text
                    domain_match = re.search(r'([a-zA-Z0-9-]+\.[a-zA-Z]{2,4})', domain)
                    if domain_match:
                        domain = domain_match.group(1)
                        
                        # Final validation
                        if self._is_valid_domain_format(domain):
                            domains.append(domain)
            
            # Ensure we return exactly 3 domains or pad with fallbacks
            if len(domains) < 3:
                fallback_domains = self._generate_fallback_domains(len(domains))
                domains.extend(fallback_domains)
            
            return domains[:3]  # Return exactly 3 domains
            
        except Exception as e:
            print(f"Error extracting domains: {e}")
            return self._generate_fallback_domains(0)
    
    def _is_valid_domain_format(self, domain: str) -> bool:
        """Check if a string follows basic domain format."""
        # Basic domain validation
        domain_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,}$'
        return bool(re.match(domain_pattern, domain)) and len(domain) <= 63
    
    def _generate_fallback_domains(self, existing_count: int) -> List[str]:
        """Generate fallback domains when extraction fails."""
        fallbacks = [
            f"domain{i+existing_count+1}.com" 
            for i in range(3 - existing_count)
        ]
        return fallbacks
    
    def generate_domains(self, business_description: str) -> List[str]:
        """
        Main method to generate domain names with NSFW filtering and classification-first approach.
        
        Args:
            business_description: The business description
            
        Returns:
            List of domain names (empty if classified as gibberish or contains NSFW content)
        """
        # Step 1: Check for NSFW content FIRST - block before any model processing
        nsfw_check = self.is_nsfw_content(business_description)
        if nsfw_check["is_nsfw"]:
            print(f"🚫 NSFW content detected: {nsfw_check['reason']}")
            # Return a clear message about blocked content
            return ["NSFW_CONTENT_DETECTED"]
        
        # Step 2: Classify the business description (only if content is safe)
        classification = self.classify_business_description(business_description)
        
        # Step 3: Only generate domains if legitimate and safe
        if classification["is_legitimate"]:
            domains = self.generate_domains_for_legitimate(business_description)
            
            # Step 4: Double-check generated domains for any inappropriate content
            safe_domains = []
            for domain in domains:
                domain_check = self.is_nsfw_content(domain)
                if not domain_check["is_nsfw"]:
                    safe_domains.append(domain)
                else:
                    print(f"🚫 Filtered inappropriate domain: {domain}")
            
            return safe_domains if safe_domains else ["SAFE_DOMAINS_UNAVAILABLE"]
        else:
            # Return empty list for gibberish
            return []
    
    def load_business_descriptions(self, input_file: str) -> List[str]:
        """
        Load business descriptions from a JSONL file.
        
        Args:
            input_file: Path to the input JSONL file
            
        Returns:
            List of business descriptions
        """
        descriptions = []
        
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            return descriptions
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        description = data.get('business_description', '').strip()
                        if description:
                            descriptions.append(description)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                        continue
            
            print(f"✅ Loaded {len(descriptions)} business descriptions from {input_file}")
            
        except Exception as e:
            print(f"❌ Error loading file {input_file}: {e}")
        
        return descriptions
    
    def process_business_descriptions(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Process business descriptions from file and generate domains.
        
        Args:
            input_file: Path to input JSONL file
            
        Returns:
            List of results with business descriptions and generated domains
        """
        descriptions = self.load_business_descriptions(input_file)
        
        if not descriptions:
            print("❌ No valid business descriptions found.")
            return []
        
        results = []
        
        print(f"🚀 Processing {len(descriptions)} business descriptions...")
        
        for i, description in enumerate(descriptions, 1):
            print(f"\n📝 Processing {i}/{len(descriptions)}: {description[:50]}...")
            
            try:
                # Generate domains using v4 model
                domains = self.generate_domains(description)
                
                result = {
                    "business_description": description,
                    "target_domains": domains
                }
                
                results.append(result)
                
                print(f"✅ Generated domains: {domains}")
                
                # Memory cleanup every 10 items
                if i % 10 == 0:
                    self._cleanup_memory()
                    print(f"🧹 Memory cleanup at item {i}")
                
            except Exception as e:
                print(f"❌ Error processing description {i}: {e}")
                # Add error result
                result = {
                    "business_description": description,
                    "target_domains": []
                }
                results.append(result)
        
        print(f"\n✅ Completed processing {len(results)} descriptions")
        return results
    
    def _cleanup_memory(self):
        """Force garbage collection and clear CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _init_nsfw_filter(self):
        """Initialize the better-profanity NSFW content filter."""
        if PROFANITY_AVAILABLE:
            try:
                # Initialize the profanity filter
                self.profanity_filter = profanity
                # Load default wordlist and additional words
                self.profanity_filter.load_censor_words()
                
                # Add additional inappropriate words for business context
                additional_words = [
                    'scam', 'fraud', 'cheat', 'steal', 'illegal', 'criminal',
                    'money-laundering', 'ponzi', 'pyramid', 'mlm', 'exploit'
                ]
                self.profanity_filter.add_censor_words(additional_words)
                
                print("🛡️  Better-profanity NSFW filter initialized successfully")
                self.profanity_available = True
            except Exception as e:
                print(f"⚠️  Failed to initialize better-profanity filter: {e}")
                self.profanity_available = False
        else:
            print("🛡️  Using keyword-based NSFW detection as fallback")
            self.profanity_available = False
            
        # Simplified keyword-based detection (fallback)
        self.nsfw_keywords = {
            'explicit_sexual': [
                'sex', 'porn', 'xxx', 'adult', 'escort', 'prostitute', 'brothel',
                'naked', 'nude', 'strip', 'erotic', 'fetish'
            ],
            'violence': [
                'kill', 'murder', 'weapon', 'bomb', 'terrorist', 'violence',
                'assault', 'torture', 'shooting', 'gore'
            ],
            'drugs': [
                'cocaine', 'heroin', 'meth', 'crack', 'trafficking', 'dealer'
            ],
            'hate': [
                'nazi', 'racist', 'supremacist', 'genocide', 'hate', 'kkk'
            ],
            'scam': [
                'scam', 'fraud', 'cheat', 'steal', 'ponzi', 'pyramid', 'mlm'
            ]
        }
    
    def is_nsfw_content(self, text: str) -> Dict[str, Any]:
        """
        Check if text contains NSFW content using better-profanity filter.
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with is_nsfw boolean and reason
        """
        # Normalize text for checking
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        # Quick empty/minimal text check
        if len(text_clean) < 3:
            return {"is_nsfw": False, "reason": "Text too short for meaningful analysis", "method": "length_check"}
        
        # Try better-profanity filter first if available
        if self.profanity_available:
            try:
                # Use better-profanity to check for inappropriate content
                if self.profanity_filter.contains_profanity(text_clean):
                    censored_text = self.profanity_filter.censor(text_clean)
                    return {
                        "is_nsfw": True,
                        "reason": "Content contains profanity or inappropriate language",
                        "method": "better_profanity",
                        "censored_version": censored_text
                    }
                else:
                    return {
                        "is_nsfw": False,
                        "reason": "Content passed better-profanity filter",
                        "method": "better_profanity"
                    }
                    
            except Exception as e:
                print(f"⚠️  Better-profanity check failed: {e}, falling back to keyword detection")
        
        # Fallback to keyword-based detection with context awareness
        safe_contexts = {
            'adult': ['education', 'learning', 'university', 'college', 'school', 'training', 'course'],
            'weapon': ['safety', 'security', 'training', 'education']
        }
        
        for category, keywords in self.nsfw_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Check if keyword appears in a safe context
                    is_safe_context = False
                    if keyword in safe_contexts:
                        for safe_word in safe_contexts[keyword]:
                            if safe_word in text_lower:
                                is_safe_context = True
                                break
                    
                    if not is_safe_context:
                        return {
                            "is_nsfw": True, 
                            "reason": f"Contains inappropriate content ({category}): '{keyword}'",
                            "method": "keyword_fallback",
                            "category": category,
                            "detected_keyword": keyword
                        }
        
        return {"is_nsfw": False, "reason": "Content appears safe", "method": "keyword_fallback"}
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = "../Model Outputs/model_v4_output_final.jsonl"):
        """
        Save results to a JSONL file.
        
        Args:
            results: List of result dictionaries
            output_file: Path to output file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            print(f"✅ Results saved to: {output_file}")
            print(f"📊 Total results: {len(results)}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Main function for running v4 model on dataset."""
    # Input file path - UPDATE THIS TO YOUR DATASET
    input_file = "../data/dataset_injection_v4.jsonl"
    
    # Initialize the v4 model
    generator = LoRADomainGeneratorV4()
    
    # Process business descriptions from file
    results = generator.process_business_descriptions(input_file)
    
    # Save results
    generator.save_results(results)
    
    print("\n✅ Model v4 processing complete!")
    print("🔍 Results saved with NSFW filtering and gibberish classification")


if __name__ == "__main__":
    main() 