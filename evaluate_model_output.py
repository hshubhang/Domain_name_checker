import json
import os
import sys
from typing import List, Dict, Any

# Add paths to import from Evaluator folder
sys.path.append('Evaluator')

try:
    from evaluate_domains import evaluate_multiple_domains, DomainEvaluator
except ImportError as e:
    print(f"Error importing evaluator: {e}")
    print("Make sure evaluate_domains.py is in the Evaluator folder")
    sys.exit(1)

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Install with 'pip install python-dotenv' to use .env files.")


class ModelOutputEvaluator:
    def __init__(self, api_key: str = None):
        """
        Initialize the model output evaluator.
        
        Args:
            api_key: OpenAI API key for domain evaluation
        """
        self.evaluator = DomainEvaluator(api_key)
    
    def load_model_output(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Load model output from JSONL file.
        
        Args:
            input_file: Path to the model output JSONL file
            
        Returns:
            List of dictionaries with business descriptions and target domains
        """
        data = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        
                        # Validate required fields
                        if 'business_description' not in entry:
                            print(f"Warning: Line {line_num} missing 'business_description'")
                            continue
                        if 'target_domains' not in entry:
                            print(f"Warning: Line {line_num} missing 'target_domains'")
                            continue
                        
                        data.append(entry)
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"Error: Could not find input file {input_file}")
            return []
        
        print(f"Loaded {len(data)} entries from {input_file}")
        return data
    
    def evaluate_model_output(self, model_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate all domains from model output.
        
        Args:
            model_data: List of model output entries
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, entry in enumerate(model_data, 1):
            business_description = entry['business_description']
            target_domains = entry['target_domains']
            
            print(f"Evaluating {i}/{len(model_data)}: {business_description[:60]}...")
            print(f"  Domains: {target_domains}")
            
            if not target_domains:
                print(f"  ⚠️  No domains to evaluate")
                results.append({
                    "business_description": business_description,
                    "target_domains": target_domains,
                    "domain_evaluations": [],
                    "best_domain": None,
                    "average_score": 0.0,
                    "total_domains": 0
                })
                continue
            
            try:
                # Evaluate all domains for this business
                domain_evaluations = self.evaluator.evaluate_domains_single_request(
                    business_description, 
                    target_domains
                )
                
                # Calculate statistics
                if domain_evaluations:
                    scores = [eval_result['score'] for eval_result in domain_evaluations]
                    average_score = sum(scores) / len(scores)
                    best_domain = max(domain_evaluations, key=lambda x: x['score'])
                else:
                    average_score = 0.0
                    best_domain = None
                
                result = {
                    "business_description": business_description,
                    "target_domains": target_domains,
                    "domain_evaluations": domain_evaluations,
                    "best_domain": best_domain,
                    "average_score": round(average_score, 3),
                    "total_domains": len(target_domains)
                }
                
                results.append(result)
                
                print(f"  ✅ Evaluated {len(domain_evaluations)} domains")
                if best_domain:
                    print(f"  🏆 Best: {best_domain['domain']} (score: {best_domain['score']:.3f})")
                
            except Exception as e:
                print(f"  ❌ Error evaluating domains: {e}")
                results.append({
                    "business_description": business_description,
                    "target_domains": target_domains,
                    "domain_evaluations": [],
                    "best_domain": None,
                    "average_score": 0.0,
                    "total_domains": len(target_domains),
                    "error": str(e)
                })
        
        return results
    
    def save_evaluation_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save evaluation results to a JSONL file.
        
        Args:
            results: List of evaluation results
            output_file: Output file path
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"\n🎯 Results saved to {output_file}")
            
            # Print summary statistics
            self._print_summary(results)
            
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary statistics."""
        total_businesses = len(results)
        successful_evaluations = len([r for r in results if not r.get('error') and r.get('domain_evaluations')])
        
        if successful_evaluations > 0:
            # Collect all domain scores
            all_scores = []
            best_domains = []
            
            for result in results:
                if result.get('domain_evaluations'):
                    for eval_result in result['domain_evaluations']:
                        all_scores.append(eval_result['score'])
                if result.get('best_domain'):
                    best_domains.append(result['best_domain'])
            
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
            
            print(f"\n📊 EVALUATION SUMMARY:")
            print(f"   Total businesses: {total_businesses}")
            print(f"   Successfully evaluated: {successful_evaluations}")
            print(f"   Total domains evaluated: {len(all_scores)}")
            print(f"   Average domain score: {avg_score:.3f}")
            
            if best_domains:
                top_score = max(best_domains, key=lambda x: x['score'])
                print(f"   Highest scoring domain: {top_score['domain']} ({top_score['score']:.3f})")


def process_model_output_file(input_file: str, api_key: str = None) -> str:
    """
    Process a model output file and generate evaluation results.
    
    Args:
        input_file: Path to model output JSONL file
        api_key: OpenAI API key
        
    Returns:
        Path to the generated evaluation file
    """
    # Extract filename without extension to generate output name
    base_name = os.path.basename(input_file)
    if base_name.endswith('.jsonl'):
        base_name = base_name[:-6]  # Remove .jsonl
    
    # Create output filename and path
    output_filename = f"evaluated_{base_name}.jsonl"
    output_file = os.path.join("evaluator outputs", output_filename)  # Save in evaluator outputs folder
    
    print(f"=== Model Output Evaluator ===")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}\n")
    
    try:
        # Initialize evaluator
        evaluator = ModelOutputEvaluator(api_key)
        
        # Load model output
        model_data = evaluator.load_model_output(input_file)
        if not model_data:
            print("No data to process")
            return None
        
        # Evaluate domains
        results = evaluator.evaluate_model_output(model_data)
        
        # Save results
        evaluator.save_evaluation_results(results, output_file)
        
        return output_file
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return None


def main():
    """Main function to run the evaluation process."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate domains from model output')
    parser.add_argument('input_file', nargs='?', default='Model Outputs/model_v0_output.jsonl',
                        help='Path to model output JSONL file')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Process the file
    output_file = process_model_output_file(args.input_file, args.api_key)
    
    if output_file:
        print(f"\n✅ Evaluation complete! Results saved to: {output_file}")
    else:
        print("\n❌ Evaluation failed!")


if __name__ == "__main__":
    main() 