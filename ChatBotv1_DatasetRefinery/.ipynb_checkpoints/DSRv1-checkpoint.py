import argparse
import ast
import json
import logging
import os
import time
from typing import Dict, List, Any
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetRefineryError(Exception):
    """Base exception for dataset refinery errors"""
    pass

class OllamaConnectionError(DatasetRefineryError):
    """Raised when connection to Ollama fails"""
    pass

class DatasetRefinery:
    """Automates dataset refinement using Ollama local LLM"""
    
    def __init__(self, 
                 ollama_host: str = "http://localhost:11434",
                 model_name: str = "llama3.3:70b",
                 temperature: float = 0.3,
                 max_retries: int = 3,
                 retry_delay: float = 2.0):
        """
        Initialize the Dataset Refinery.
        
        Args:
            ollama_host: Ollama server URL
            model_name: Model to use for refinement
            temperature: Generation temperature (lower = more consistent)
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        # Define the refinement prompt template
        self.refinement_prompt = """You are a dataset curator tasked with refining intent classification samples. Your job is to:

1. Remove duplicate or near-duplicate samples
2. Keep the most diverse and representative examples
3. Ensure each sample is clear and well-formed
4. Maintain the original meaning and intent
5. Return exactly {target_samples} samples

Given the following samples for the intent "{main_intent}" -> "{sub_intent}":

{samples}

Please refine this list to exactly {target_samples} high-quality, diverse samples. Return ONLY a Python list format like:
[
    "sample 1",
    "sample 2",
    ...
]

Do not include any explanations, just the refined list."""

    def _test_ollama_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model_name not in model_names:
                    logger.warning(f"Model '{self.model_name}' not found. Available models: {model_names}")
                    logger.info("You may need to pull the model using: ollama pull llama3.3:70b")
                else:
                    logger.info(f"Successfully connected to Ollama. Model '{self.model_name}' is available.")
            else:
                raise OllamaConnectionError(f"Ollama server returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama at {self.ollama_host}: {str(e)}")

    def _call_ollama(self, prompt: str) -> str:
        """
        Make a completion request to Ollama with retry logic.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Model response text
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making Ollama request (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=300  # 5 minutes timeout for large models
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    logger.warning(f"Ollama request failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                
            if attempt < self.max_retries - 1:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        raise OllamaConnectionError(f"Failed to get response from Ollama after {self.max_retries} attempts")

    def _parse_llm_response(self, response: str) -> List[str]:
        """
        Parse the LLM response to extract the refined samples list.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            List of refined samples
        """
        try:
            # Try to find Python list in the response
            response = response.strip()
            
            # Look for list patterns
            if response.startswith('[') and response.endswith(']'):
                # Direct list format
                return ast.literal_eval(response)
            
            # Try to extract list from response
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                list_str = response[start_idx:end_idx + 1]
                return ast.literal_eval(list_str)
            
            # If no list found, try to parse as JSON
            try:
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            
            logger.warning("Could not parse LLM response as list, attempting line-by-line parsing")
            
            # Fallback: try to extract lines that look like samples
            lines = response.split('\n')
            samples = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Remove quotes and common prefixes
                    line = line.strip('"\'`')
                    if line.startswith('- '):
                        line = line[2:]
                    if line.startswith('* '):
                        line = line[2:]
                    if line and len(line) > 5:  # Minimum length check
                        samples.append(line)
            
            return samples
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            logger.debug(f"Raw response: {response}")
            return []

    def refine_samples(self, 
                      main_intent: str, 
                      sub_intent: str, 
                      samples: List[str], 
                      target_samples: int) -> List[str]:
        """
        Refine a list of samples using the LLM.
        
        Args:
            main_intent: Main intent category
            sub_intent: Sub-intent category
            samples: List of original samples
            target_samples: Desired number of samples
            
        Returns:
            List of refined samples
        """
        if len(samples) <= target_samples:
            logger.info(f"Intent {main_intent}->{sub_intent}: Already has {len(samples)} samples (target: {target_samples}), skipping refinement")
            return samples
        
        logger.info(f"Refining intent {main_intent}->{sub_intent}: {len(samples)} -> {target_samples} samples")
        
        # Format samples for the prompt
        samples_str = '\n'.join([f"- {sample}" for sample in samples])
        
        # Create the prompt
        prompt = self.refinement_prompt.format(
            main_intent=main_intent,
            sub_intent=sub_intent,
            samples=samples_str,
            target_samples=target_samples
        )
        
        try:
            # Get LLM response
            response = self._call_ollama(prompt)
            
            # Parse the response
            refined_samples = self._parse_llm_response(response)
            
            if not refined_samples:
                logger.warning(f"No samples parsed from LLM response for {main_intent}->{sub_intent}, using original samples")
                return samples[:target_samples]  # Fallback to truncation
            
            # Ensure we have the right number of samples
            if len(refined_samples) > target_samples:
                refined_samples = refined_samples[:target_samples]
            elif len(refined_samples) < target_samples:
                logger.warning(f"LLM returned {len(refined_samples)} samples, expected {target_samples}")
                # Pad with original samples if needed
                remaining_needed = target_samples - len(refined_samples)
                original_remaining = [s for s in samples if s not in refined_samples]
                refined_samples.extend(original_remaining[:remaining_needed])
            
            logger.info(f"Successfully refined {main_intent}->{sub_intent}: {len(refined_samples)} samples")
            return refined_samples
            
        except Exception as e:
            logger.error(f"Error refining samples for {main_intent}->{sub_intent}: {str(e)}")
            logger.info("Falling back to simple truncation")
            return samples[:target_samples]

    def load_dataset(self, file_path: str) -> Dict[str, Any]:
        """
        Load dataset from a .txt file containing Python dict format.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Loaded dataset dictionary
        """
        try:
            logger.info(f"Loading dataset from {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Try to extract the dictionary from the file
            # Look for patterns like "intent_samples = {" or just "{"
            if '=' in content:
                # Split by '=' and take the part after it
                dict_part = content.split('=', 1)[1].strip()
            else:
                dict_part = content.strip()
            
            # Parse the dictionary
            dataset = ast.literal_eval(dict_part)
            
            logger.info(f"Successfully loaded dataset with {len(dataset)} main intents")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset from {file_path}: {str(e)}")
            raise DatasetRefineryError(f"Failed to load dataset: {str(e)}")

    def save_dataset(self, dataset: Dict[str, Any], output_path: str):
        """
        Save refined dataset to output file.
        
        Args:
            dataset: Refined dataset dictionary
            output_path: Path for output file
        """
        try:
            logger.info(f"Saving refined dataset to {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as file:
                # Write in the same format as input
                file.write("intent_samples = ")
                file.write(json.dumps(dataset, indent=4, ensure_ascii=False))
            
            logger.info(f"Successfully saved refined dataset to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving dataset to {output_path}: {str(e)}")
            raise DatasetRefineryError(f"Failed to save dataset: {str(e)}")

    def refine_dataset(self, 
                      input_path: str, 
                      output_path: str, 
                      target_samples: int,
                      progress_callback=None):
        """
        Refine the entire dataset.
        
        Args:
            input_path: Path to input dataset file
            output_path: Path for output refined dataset file
            target_samples: Target number of samples per sub-intent
            progress_callback: Optional callback function for progress updates
        """
        # Load dataset
        dataset = self.load_dataset(input_path)
        
        # Count total sub-intents for progress tracking
        total_sub_intents = sum(len(sub_intents) for sub_intents in dataset.values())
        processed = 0
        
        refined_dataset = {}
        
        logger.info(f"Starting refinement: {total_sub_intents} sub-intents, target {target_samples} samples each")
        
        for main_intent, sub_intents in dataset.items():
            logger.info(f"Processing main intent: {main_intent}")
            refined_dataset[main_intent] = {}
            
            for sub_intent, samples in sub_intents.items():
                try:
                    # Refine samples for this sub-intent
                    refined_samples = self.refine_samples(
                        main_intent, 
                        sub_intent, 
                        samples, 
                        target_samples
                    )
                    
                    refined_dataset[main_intent][sub_intent] = refined_samples
                    processed += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(processed, total_sub_intents, main_intent, sub_intent)
                    
                    # Small delay to be nice to the LLM
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing {main_intent}->{sub_intent}: {str(e)}")
                    # Keep original samples on error
                    refined_dataset[main_intent][sub_intent] = samples[:target_samples]
                    processed += 1
        
        # Save refined dataset
        self.save_dataset(refined_dataset, output_path)
        
        logger.info(f"Dataset refinement completed! Processed {processed}/{total_sub_intents} sub-intents")
        return refined_dataset

def progress_printer(processed: int, total: int, main_intent: str, sub_intent: str):
    """Simple progress printer for the refinement process"""
    percentage = (processed / total) * 100
    print(f"Progress: {processed}/{total} ({percentage:.1f}%) - Completed: {main_intent} -> {sub_intent}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Refine intent classification dataset using local Ollama LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_refinery.py -i dataset.txt -o refined_dataset.txt -s 20
  python dataset_refinery.py -i data/intent_samples.txt -o output/refined.txt -s 15 --model llama3.1:8b
  python dataset_refinery.py -i dataset.txt -o refined.txt -s 10 --host http://192.168.1.100:11434
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Path to source dataset .txt file')
    parser.add_argument('-o', '--output', required=True,
                       help='Path for output refined dataset .txt file')
    parser.add_argument('-s', '--samples', type=int, required=True,
                       help='Target number of samples per sub-intent')
    parser.add_argument('--host', default='http://localhost:11434',
                       help='Ollama server host (default: http://localhost:11434)')
    parser.add_argument('--model', default='llama3.3:70b',
                       help='Model name to use (default: llama3.3:70b)')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Generation temperature (default: 0.3)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts (default: 3)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return 1
    
    if args.samples <= 0:
        print("Error: Number of samples must be positive")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        # Initialize refinery
        print(f"Initializing Dataset Refinery...")
        print(f"  Ollama Host: {args.host}")
        print(f"  Model: {args.model}")
        print(f"  Target Samples: {args.samples}")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
        print()
        
        refinery = DatasetRefinery(
            ollama_host=args.host,
            model_name=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries
        )
        
        # Start refinement
        print("Starting dataset refinement...")
        start_time = time.time()
        
        refinery.refine_dataset(
            input_path=args.input,
            output_path=args.output,
            target_samples=args.samples,
            progress_callback=progress_printer
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nRefinement completed successfully!")
        print(f"Time taken: {duration:.2f} seconds")
        print(f"Output saved to: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nRefinement interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())