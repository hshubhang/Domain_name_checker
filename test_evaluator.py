#!/usr/bin/env python3
"""
Simple test script to debug the domain evaluator functionality.
"""

import os
import sys

# Add Evaluator to path
sys.path.append('Evaluator')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("❌ python-dotenv not installed")

# Check API key
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"✅ OpenAI API key found (starts with: {api_key[:10]}...)")
else:
    print("❌ No OpenAI API key found")
    sys.exit(1)

# Test evaluator import
try:
    from evaluate_domains import DomainEvaluator
    print("✅ Successfully imported DomainEvaluator")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test evaluator initialization
try:
    evaluator = DomainEvaluator(api_key)
    print("✅ Successfully initialized DomainEvaluator")
except Exception as e:
    print(f"❌ Evaluator initialization error: {e}")
    sys.exit(1)

# Test a simple evaluation
try:
    print("\n🧪 Testing domain evaluation...")
    test_business = "A SaaS platform that helps small restaurants optimize their inventory."
    test_domains = ["foodtech.com", "inventory.co"]
    
    result = evaluator.evaluate_domains_single_request(test_business, test_domains)
    
    if result:
        print(f"✅ Evaluation successful! Got {len(result)} results")
        for domain_result in result:
            print(f"   - {domain_result['domain']}: {domain_result['score']:.3f}")
    else:
        print("❌ Evaluation returned empty result")
        
except Exception as e:
    print(f"❌ Evaluation error: {e}")
    import traceback
    traceback.print_exc() 