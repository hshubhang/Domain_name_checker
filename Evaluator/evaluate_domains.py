import json
import os
from typing import List, Dict, Any
from openai import OpenAI

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads variables from .env file in the project root
except ImportError:
    # dotenv not installed, environment variables should be set manually
    print("Note: python-dotenv not installed. Install with 'pip install python-dotenv' to use .env files.")

class DomainEvaluator:
    def __init__(self, api_key: str = None):
        """
        Initialize the domain evaluator with OpenAI API.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
        """
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
        
    def evaluate_domain(self, business_description: str, domain: str) -> Dict[str, Any]:
        """
        Evaluate a single domain name against a business description using GPT-4.
        
        Args:
            business_description: The business description to match against
            domain: The domain name to evaluate (e.g., "example.com")
            
        Returns:
            Dictionary with overall score and individual dimension scores
        """
        prompt = self._create_evaluation_prompt(business_description, domain)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a domain name evaluator with expertise in branding and business strategy. You must respond ONLY with valid JSON in the exact format specified."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse the JSON response
            result_text = response.choices[0].message.content.strip()
            
            # Clean up the response to extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            result = json.loads(result_text)
            
            # Validate the structure
            if not self._validate_result(result):
                raise ValueError("Invalid response structure from API")
                
            return result
            
        except Exception as e:
            print(f"Error evaluating domain {domain}: {str(e)}")
            # Return a default low score if API fails
            return {
                "score": 0.0,
                "reasons": {
                    "relevance": 0.0,
                    "clarity": 0.0,
                    "professionalism": 0.0,
                    "memorability": 0.0,
                    "tld_suitability": 0.0
                }
            }
    
    def _create_evaluation_prompt(self, business_description: str, domain: str) -> str:
        """Create the evaluation prompt for GPT-4."""
        return f"""
You are a domain name evaluator with expertise in branding and business strategy.

Evaluate the domain name "{domain}" for this business:
"{business_description}"

Rate the domain on these five metrics (score between 0.0 to 1.0):

- Relevance: How well does the domain relate to the business?
- Clarity: Is the domain easy to read and understand?
- Professionalism: Does the domain sound credible for a real company?
- Memorability: Is it catchy or easy to remember?
- TLD Suitability: Is the TLD appropriate (.com, .ai, .co, .org, .net)?

Output ONLY in this exact JSON format:
```json
{{
  "score": <average_of_all_criteria>,
  "reasons": {{
    "relevance": <score>,
    "clarity": <score>,
    "professionalism": <score>,
    "memorability": <score>,
    "tld_suitability": <score>
  }}
}}
```

Provide only the JSON response, no additional text.
"""

    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate that the API response has the correct structure."""
        if not isinstance(result, dict):
            return False
        
        if "score" not in result or "reasons" not in result:
            return False
        
        reasons = result["reasons"]
        required_keys = ["relevance", "clarity", "professionalism", "memorability", "tld_suitability"]
        
        if not all(key in reasons for key in required_keys):
            return False
        
        # Check that all scores are between 0.0 and 1.0
        all_scores = [result["score"]] + [reasons[key] for key in required_keys]
        if not all(isinstance(score, (int, float)) and 0.0 <= score <= 1.0 for score in all_scores):
            return False
        
        return True

    def evaluate_domains_batch(self, business_description: str, domains: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple domains for a business.
        
        Args:
            business_description: Description of the business
            domains: List of domain names to evaluate
            
        Returns:
            List of evaluation results for each domain
        """
        results = []
        
        for domain in domains:
            print(f"Evaluating domain: {domain}")
            result = self.evaluate_domain(business_description, domain)
            result['domain'] = domain
            results.append(result)
        
        return results

    def evaluate_domains_single_request(self, business_description: str, domains: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple domains in a single API request for efficiency.
        
        Args:
            business_description: Description of the business
            domains: List of domain names to evaluate
            
        Returns:
            List of evaluation results for each domain
        """
        if not domains:
            return []
        
        # Create a prompt for multiple domains
        domains_text = "\n".join([f"- {domain}" for domain in domains])
        
        prompt = f"""
You are a domain name evaluator with expertise in branding and business strategy.

Evaluate these domain names for this business:
Business: "{business_description}"

Domains to evaluate:
{domains_text}

For each domain, rate it on these five metrics (score between 0.0 to 1.0):
- Relevance: How well does the domain relate to the business?
- Clarity: Is the domain easy to read and understand?
- Professionalism: Does the domain sound credible for a real company?
- Memorability: Is it catchy or easy to remember?
- TLD Suitability: Is the TLD appropriate (.com, .ai, .co, .org, .net)?

Output ONLY a JSON array where each object follows this exact format:
```json
[
  {{
    "domain": "domain1.com",
    "score": <average_of_all_criteria>,
    "reasons": {{
      "relevance": <score>,
      "clarity": <score>,
      "professionalism": <score>,
      "memorability": <score>,
      "tld_suitability": <score>
    }}
  }},
  {{
    "domain": "domain2.com",
    "score": <average_of_all_criteria>,
    "reasons": {{
      "relevance": <score>,
      "clarity": <score>,
      "professionalism": <score>,
      "memorability": <score>,
      "tld_suitability": <score>
    }}
  }}
]
```

Provide only the JSON array response, no additional text.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a domain name evaluator with expertise in branding and business strategy. You must respond ONLY with valid JSON in the exact format specified."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Parse the JSON response
            result_text = response.choices[0].message.content.strip()
            
            # Clean up the response to extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            results = json.loads(result_text)
            
            # Validate results
            if not isinstance(results, list):
                raise ValueError("Expected JSON array response")
            
            for result in results:
                if not self._validate_result(result):
                    raise ValueError("Invalid result structure")
            
            return results
            
        except Exception as e:
            print(f"Error in batch evaluation: {str(e)}")
            # Fall back to individual evaluations
            return self.evaluate_domains_batch(business_description, domains)


# Convenience functions
def evaluate_single_domain(business_description: str, domain: str, api_key: str = None) -> Dict[str, Any]:
    """
    Evaluate a single domain name.
    
    Args:
        business_description: Description of the business
        domain: Domain name to evaluate
        api_key: OpenAI API key (optional, can use environment variable)
        
    Returns:
        Evaluation result dictionary
    """
    evaluator = DomainEvaluator(api_key)
    return evaluator.evaluate_domain(business_description, domain)


def evaluate_multiple_domains(business_description: str, domains: List[str], api_key: str = None, batch: bool = True) -> List[Dict[str, Any]]:
    """
    Evaluate multiple domain names.
    
    Args:
        business_description: Description of the business
        domains: List of domain names to evaluate
        api_key: OpenAI API key (optional, can use environment variable)
        batch: If True, evaluates all domains in one API call. If False, makes separate calls.
        
    Returns:
        List of evaluation results
    """
    evaluator = DomainEvaluator(api_key)
    
    if batch:
        return evaluator.evaluate_domains_single_request(business_description, domains)
    else:
        return evaluator.evaluate_domains_batch(business_description, domains)


# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key as an environment variable or pass it directly
    # export OPENAI_API_KEY="your-api-key-here"
    
    # Example business description
    business_desc = "A telemedicine platform that connects patients with licensed therapists for real-time virtual sessions."
    
    # Example domains to evaluate
    test_domains = [
        "virtualtherapy.com",
        "telemed-connect.co",
        "therapylink.ai",
        "mindcare24.com"
    ]
    
    print("=== Domain Evaluation with GPT-4 ===")
    print(f"Business: {business_desc}\n")
    
    # Evaluate domains (you need to set OPENAI_API_KEY environment variable)
    try:
        results = evaluate_multiple_domains(business_desc, test_domains, batch=True)
        
        for result in results:
            print(f"Domain: {result['domain']}")
            print(json.dumps({
                "score": result["score"],
                "reasons": result["reasons"]
            }, indent=2))
            print("-" * 40)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to set your OPENAI_API_KEY environment variable") 