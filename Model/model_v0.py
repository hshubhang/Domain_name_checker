import json
import os
import replicate
from typing import List, Dict, Any

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads variables from .env file in the project root
except ImportError:
    # dotenv not installed, environment variables should be set manually
    print("Note: python-dotenv not installed. Install with 'pip install python-dotenv' to use .env files.")


class DomainGenerator:
    def __init__(self, api_token: str = None):
        """
        Initialize the domain generator with Replicate API.
        
        Args:
            api_token: Replicate API token. If None, will try to get from environment variable REPLICATE_API_TOKEN
        """
        if api_token is None:
            api_token = os.getenv('REPLICATE_API_TOKEN')
            if not api_token:
                raise ValueError("Replicate API token must be provided either as parameter or REPLICATE_API_TOKEN environment variable")
        
        # Set the API token for replicate
        os.environ['REPLICATE_API_TOKEN'] = api_token
        
        # Initialize the model
        self.model = "meta/meta-llama-3-8b-instruct"
        
    def generate_domains_for_business(self, business_description: str, num_domains: int = 3) -> List[str]:
        """
        Generate domain name suggestions for a business description using LLaMA 3 8B.
        
        Args:
            business_description: Description of the business
            num_domains: Number of domain suggestions to generate
            
        Returns:
            List of suggested domain names
        """
        
        prompt = f"""You are a domain name expert. Generate {num_domains} creative and professional domain name suggestions for this business:

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

        try:
            # Generate response using LLaMA 3 8B Instruct
            output = replicate.run(
                self.model,
                input={
                    "prompt": prompt,
                    "max_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop_sequences": "\n\n"
                }
            )
            
            # Join the output if it's a list, otherwise use as string
            if isinstance(output, list):
                response_text = "".join(output)
            else:
                response_text = str(output)
            
            # Parse the response to extract domain names
            domains = self._parse_domain_response(response_text)
            
            # Ensure we have the requested number of domains
            if len(domains) < num_domains:
                print(f"Warning: Only generated {len(domains)} domains instead of {num_domains} for business: {business_description[:50]}...")
            
            return domains[:num_domains]  # Return only the requested number
            
        except Exception as e:
            print(f"Error generating domains for business '{business_description[:50]}...': {str(e)}")
            # Return fallback domains if API fails
            return self._generate_fallback_domains(business_description, num_domains)
    
    def _parse_domain_response(self, response_text: str) -> List[str]:
        """
        Parse the LLaMA response to extract clean domain names.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            List of cleaned domain names
        """
        domains = []
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and '.' in line:
                # Remove common prefixes/suffixes and clean up
                domain = line.replace('- ', '').replace('• ', '').replace('* ', '')
                domain = domain.replace('Example: ', '').replace('Domain: ', '')
                
                # Basic validation - should have a dot and reasonable length
                if '.' in domain and 3 <= len(domain) <= 30:
                    # Remove any trailing punctuation
                    domain = domain.rstrip('.,!?')
                    domains.append(domain.lower())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_domains = []
        for domain in domains:
            if domain not in seen:
                seen.add(domain)
                unique_domains.append(domain)
        
        return unique_domains
    
    def _generate_fallback_domains(self, business_description: str, num_domains: int) -> List[str]:
        """
        Generate simple fallback domains if the API fails.
        
        Args:
            business_description: Description of the business
            num_domains: Number of domains to generate
            
        Returns:
            List of fallback domain names
        """
        # Extract key words from business description
        words = business_description.lower().replace('.', '').replace(',', '').split()
        key_words = [word for word in words if len(word) > 3 and word not in 
                    {'that', 'with', 'through', 'platform', 'service', 'system', 'application'}]
        
        fallback_domains = []
        tlds = ['.com', '.co', '.ai', '.io', '.net']
        
        for i in range(min(num_domains, len(key_words))):
            keyword = key_words[i][:8]  # Limit length
            tld = tlds[i % len(tlds)]
            fallback_domains.append(f"{keyword}{tld}")
        
        # Fill remaining slots if needed
        while len(fallback_domains) < num_domains:
            fallback_domains.append(f"startup{len(fallback_domains)}.com")
        
        return fallback_domains
    
    def process_business_descriptions(self, input_file: str = "../data/data_injection.jsonl") -> List[Dict[str, Any]]:
        """
        Process all business descriptions from the input file and generate domain suggestions.
        
        Args:
            input_file: Path to the input JSONL file
            
        Returns:
            List of dictionaries with business descriptions and target domains
        """
        results = []
        
        # Read business descriptions from input file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                business_descriptions = []
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if 'business_description' in data:
                            business_descriptions.append(data['business_description'])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_num} in {input_file}: {e}")
                        continue
        except FileNotFoundError:
            print(f"Error: Could not find input file {input_file}")
            return []
        
        print(f"Processing {len(business_descriptions)} business descriptions...")
        
        # Generate domains for each business
        for i, business_desc in enumerate(business_descriptions, 1):
            print(f"Processing {i}/{len(business_descriptions)}: {business_desc[:60]}...")
            
            try:
                target_domains = self.generate_domains_for_business(business_desc, num_domains=3)
                
                result = {
                    "business_description": business_desc,
                    "target_domains": target_domains
                }
                
                results.append(result)
                
                # Print progress
                print(f"✅ Generated domains: {target_domains}")
                
            except Exception as e:
                print(f"❌ Error processing business {i}: {e}")
                # Add entry with empty domains to maintain consistency
                results.append({
                    "business_description": business_desc,
                    "target_domains": []
                })
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = "../Model Outputs/model_v0_output.jsonl"):
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
            successful = len([r for r in results if r.get('target_domains')])
            success_rate = (successful / len(results)) * 100 if results else 0
            print(f"✅ Success rate: {success_rate:.1f}% ({successful}/{len(results)})")
            
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")


def main():
    """
    Main function to run the domain generation process.
    """
    print("=== Meta LLaMA 3 8B Domain Generator ===")
    print("Reading business descriptions from data/data_injection.jsonl...")
    
    try:
        # Initialize the generator
        generator = DomainGenerator()
        
        # Process all business descriptions
        results = generator.process_business_descriptions("../data/data_injection.jsonl")
        
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
    # Set your Replicate API token as an environment variable or in .env file
    # export REPLICATE_API_TOKEN="your-replicate-token-here"
    
    results = main() 