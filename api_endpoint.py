import json
import re
from typing import List, Dict, Any, Tuple
from flask import Flask, request, jsonify
import sys
import os

# Add Model directory to path to import model_v4
sys.path.append(os.path.join(os.path.dirname(__file__), 'Model'))

try:
    from model_v4 import LoRADomainGeneratorV4
except ImportError as e:
    print(f"❌ Error importing LoRADomainGeneratorV4: {e}")
    print("Make sure Model/model_v4.py exists and dependencies are installed")
    sys.exit(1)

app = Flask(__name__)

class DomainAPIWrapper:
    """API wrapper for LoRADomainGeneratorV4 with confidence scoring and status handling."""
    
    def __init__(self):
        """Initialize the domain generator model."""
        print("🚀 Initializing Domain API Wrapper...")
        try:
            # Set correct paths for API context (running from root directory)
            lora_path = "lora_adapters/v4/"
            classifier_path = "Model/classifier_model_v3"
            
            self.model = LoRADomainGeneratorV4(
                lora_adapters_path=lora_path,
                classifier_model_path=classifier_path
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def calculate_confidence_score(self, domain: str, business_description: str) -> float:
        """
        Calculate heuristic confidence score for a domain name.
        
        Args:
            domain: Domain name to score
            business_description: Original business description
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        weights = {
            'length': 0.15,
            'readability': 0.20,
            'relevance': 0.25,
            'tld_quality': 0.20,
            'memorability': 0.20
        }
        
        # 1. Length Score (optimal 8-15 chars)
        domain_name = domain.split('.')[0]  # Remove TLD for length calc
        length = len(domain_name)
        if 8 <= length <= 15:
            length_score = 1.0
        elif 6 <= length <= 18:
            length_score = 0.8
        elif 4 <= length <= 20:
            length_score = 0.6
        else:
            length_score = 0.3
        
        # 2. Readability Score (no numbers, hyphens, confusing patterns)
        readability_score = 1.0
        if re.search(r'\d', domain_name):  # Contains numbers
            readability_score -= 0.3
        if '-' in domain_name:  # Contains hyphens
            readability_score -= 0.2
        if re.search(r'(.)\1{2,}', domain_name):  # Repeated chars (aaa)
            readability_score -= 0.3
        readability_score = max(0.0, readability_score)
        
        # 3. Relevance Score (keyword matching)
        business_words = set(re.findall(r'\b\w+\b', business_description.lower()))
        domain_words = set(re.findall(r'[a-z]+', domain_name.lower()))
        
        relevance_score = 0.0
        if business_words and domain_words:
            # Direct word matches
            word_matches = len(business_words.intersection(domain_words))
            if word_matches > 0:
                relevance_score = min(1.0, word_matches * 0.4)
            
            # Partial matches (domain contains business word parts)
            for bword in business_words:
                if len(bword) >= 4:  # Only check meaningful words
                    for dword in domain_words:
                        if bword in dword or dword in bword:
                            relevance_score += 0.2
            
            relevance_score = min(1.0, relevance_score)
        
        # 4. TLD Quality Score
        tld = domain.split('.')[-1].lower()
        tld_scores = {
            'com': 1.0, 'org': 0.9, 'net': 0.8, 'io': 0.85, 'ai': 0.8,
            'co': 0.75, 'app': 0.7, 'tech': 0.7, 'biz': 0.6, 'info': 0.5
        }
        tld_score = tld_scores.get(tld, 0.4)
        
        # 5. Memorability Score (pronounceable, not too complex)
        memorability_score = 1.0
        vowel_count = len([c for c in domain_name.lower() if c in 'aeiou'])
        consonant_count = len([c for c in domain_name.lower() if c.isalpha() and c not in 'aeiou'])
        
        # Good vowel/consonant ratio
        if consonant_count > 0:
            vowel_ratio = vowel_count / consonant_count
            if 0.3 <= vowel_ratio <= 0.8:  # Good balance
                memorability_score = 1.0
            else:
                memorability_score = 0.7
        
        # Penalize hard consonant clusters
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{4,}', domain_name.lower()):
            memorability_score -= 0.3
        
        memorability_score = max(0.0, memorability_score)
        
        # Calculate weighted final score
        final_score = (
            length_score * weights['length'] +
            readability_score * weights['readability'] +
            relevance_score * weights['relevance'] +
            tld_score * weights['tld_quality'] +
            memorability_score * weights['memorability']
        )
        
        return round(final_score, 3)
    
    def format_api_response(self, domains: List[str], business_description: str) -> Dict[str, Any]:
        """
        Format model output into API response with status and confidence scores.
        
        Args:
            domains: Raw model output
            business_description: Original business description
            
        Returns:
            Formatted API response
        """
        # Handle special model outputs
        if not domains:  # Empty list (gibberish filtered)
            return {
                "suggestions": [],
                "status": "filtered", 
                "message": "Content appears to be invalid or gibberish"
            }
        
        if domains == ["NSFW_CONTENT_DETECTED"]:  # NSFW blocked
            return {
                "suggestions": [],
                "status": "blocked",
                "message": "Request contains inappropriate content"
            }
        
        if domains == ["SAFE_DOMAINS_UNAVAILABLE"]:  # Safety fallback
            return {
                "suggestions": [],
                "status": "blocked",
                "message": "Unable to generate safe domain suggestions"
            }
        
        # Normal case: generate confidence scores for domains
        suggestions = []
        for domain in domains:
            confidence = self.calculate_confidence_score(domain, business_description)
            suggestions.append({
                "domain": domain,
                "confidence": confidence
            })
        
        # Sort by confidence (highest first)
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "suggestions": suggestions,
            "status": "success"
        }
    
    def generate_domains_api(self, business_description: str) -> Dict[str, Any]:
        """
        Generate domains and format for API response.
        
        Args:
            business_description: Business description from request
            
        Returns:
            API response dictionary
        """
        try:
            # Validate input
            if not business_description or not business_description.strip():
                return {
                    "suggestions": [],
                    "status": "error",
                    "message": "Business description is required"
                }
            
            # Generate domains using model
            domains = self.model.generate_domains(business_description.strip())
            
            # Format response
            return self.format_api_response(domains, business_description)
            
        except Exception as e:
            print(f"❌ Error generating domains: {e}")
            return {
                "suggestions": [],
                "status": "error", 
                "message": "Internal server error"
            }

# Initialize the API wrapper (global instance)
try:
    api_wrapper = DomainAPIWrapper()
except Exception as e:
    print(f"❌ Failed to initialize API: {e}")
    api_wrapper = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if api_wrapper is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({"status": "healthy"})

@app.route('/generate', methods=['POST'])
def generate_domains():
    """Main domain generation endpoint."""
    if api_wrapper is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({
                "suggestions": [],
                "status": "error",
                "message": "Invalid JSON request"
            }), 400
        
        business_description = data.get('business_description', '')
        
        # Generate domains
        response = api_wrapper.generate_domains_api(business_description)
        
        # Return appropriate HTTP status
        if response['status'] == 'error':
            return jsonify(response), 500
        elif response['status'] in ['blocked', 'filtered']:
            return jsonify(response), 200  # Not an error, but content filtered
        else:
            return jsonify(response), 200
            
    except Exception as e:
        return jsonify({
            "suggestions": [],
            "status": "error",
            "message": "Request processing failed"
        }), 500

@app.route('/generate', methods=['GET'])
def generate_domains_get():
    """GET endpoint for domain generation (for easy testing)."""
    if api_wrapper is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    
    business_description = request.args.get('business_description', '')
    response = api_wrapper.generate_domains_api(business_description)
    
    if response['status'] == 'error':
        return jsonify(response), 500
    else:
        return jsonify(response), 200

if __name__ == '__main__':
    if api_wrapper is None:
        print("❌ Cannot start server: Model failed to load")
        sys.exit(1)
    
    print("🚀 Starting Domain Generation API...")
    print("📍 Endpoints:")
    print("   POST /generate - Generate domains")
    print("   GET  /generate?business_description=... - Generate domains (GET)")
    print("   GET  /health - Health check")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 