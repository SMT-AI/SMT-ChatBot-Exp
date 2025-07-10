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
                 ollama_host: str = "http://127.0.0.1:11434",
                 model_name: str = "llama3.3:latest",
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
        
        # Define the refinement prompt template (using user's specific prompt)
        self.refinement_prompt = """Now you have to help me effectively prune and structure the huge text samples for my intent classification using BERT model, apparently the intent counts are high and the sample we are trying to feed have some issues, I need your help to refine the samples to imporeve the generalization and classifcation performance of the model. I want to make them "effective" {target_samples} sentences. I will share the intent with their samples one by one for you to work on it. ONLY OUTPUT THE CURATED SAMPLES IN BELOW MENTIONED FORMAT! DO NOT OUTPUT ANYTHING ELSE OTHER THAN BELOW MENTIONED FORMAT!

Intent: "{main_intent}" -> "{sub_intent}"
Samples:
{samples}

Output only the curated {target_samples} samples in Python list format:
[
    "sample 1",
    "sample 2",
    ...
]"""

    def _test_ollama_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model_name not in model_names:
                    logger.warning(f"Model '{self.model_name}' not found. Available models: {model_names}")
                    logger.info("You may need to pull the model using: ollama pull llama3.3:latest")
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
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response: {response[:500]}...")
            
            # Try to find Python list in the response
            response = response.strip()
            
            # Look for list patterns
            if response.startswith('[') and response.endswith(']'):
                # Direct list format - but we need to handle ellipsis in the response
                response_cleaned = response.replace('...', '')  # Remove ellipsis
                response_cleaned = response_cleaned.replace('...,', '')  # Remove ellipsis with comma
                response_cleaned = response_cleaned.replace(',\n    ...', '')  # Remove trailing ellipsis
                response_cleaned = response_cleaned.replace(',\n...', '')  # Remove trailing ellipsis variations
                
                try:
                    parsed_list = ast.literal_eval(response_cleaned)
                    # Filter out any ellipsis objects that might remain
                    filtered_list = [item for item in parsed_list if item is not Ellipsis and item != "..." and isinstance(item, str)]
                    logger.debug(f"Successfully parsed list with {len(filtered_list)} items")
                    return filtered_list
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Failed to parse cleaned list: {str(e)}")
            
            # Try to extract list from response
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                list_str = response[start_idx:end_idx + 1]
                
                # Clean ellipsis from extracted list
                list_str = list_str.replace('...', '')
                list_str = list_str.replace('...,', '')
                list_str = list_str.replace(',\n    ...', '')
                list_str = list_str.replace(',\n...', '')
                
                try:
                    parsed_list = ast.literal_eval(list_str)
                    filtered_list = [item for item in parsed_list if item is not Ellipsis and item != "..." and isinstance(item, str)]
                    logger.debug(f"Successfully extracted and parsed list with {len(filtered_list)} items")
                    return filtered_list
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Failed to parse extracted list: {str(e)}")
            
            # If no list found, try to parse as JSON
            try:
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    filtered_list = [item for item in parsed if item is not Ellipsis and item != "..." and isinstance(item, str)]
                    return filtered_list
            except:
                pass
            
            logger.warning("Could not parse LLM response as list, attempting line-by-line parsing")
            
            # Fallback: try to extract lines that look like samples
            lines = response.split('\n')
            samples = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//') and line != "...":
                    # Remove quotes and common prefixes
                    line = line.strip('"\'`')
                    if line.startswith('- '):
                        line = line[2:]
                    if line.startswith('* '):
                        line = line[2:]
                    if line.startswith('"') and line.endswith('",'):
                        line = line[1:-2]  # Remove quotes and comma
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]  # Remove quotes
                    
                    # Skip ellipsis and empty lines
                    if line and len(line) > 5 and line != "..." and "..." not in line:
                        samples.append(line)
            
            logger.debug(f"Fallback parsing extracted {len(samples)} samples")
            return samples
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            logger.debug(f"Problematic response: {response}")
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
        
        logger.info(f"🔄 Starting refinement for intent: {main_intent} -> {sub_intent}")
        logger.info(f"   📊 Original samples: {len(samples)} | Target samples: {target_samples}")
        
        # Format samples for the prompt (numbered list as per user's format)
        samples_str = '\n'.join([f"{i+1}. {sample}" for i, sample in enumerate(samples)])
        
        # Create the prompt using user's specific format
        prompt = self.refinement_prompt.format(
            main_intent=main_intent,
            sub_intent=sub_intent,
            samples=samples_str,
            target_samples=target_samples
        )
        
        try:
            logger.info(f"   🤖 Sending request to Ollama model: {self.model_name}")
            
            # Get LLM response
            start_time = time.time()
            response = self._call_ollama(prompt)
            end_time = time.time()
            
            logger.info(f"   ⏱️  LLM response received in {end_time - start_time:.2f} seconds")
            
            # Parse the response
            refined_samples = self._parse_llm_response(response)
            
            if not refined_samples:
                logger.warning(f"   ⚠️  No samples parsed from LLM response for {main_intent}->{sub_intent}, using original samples")
                refined_samples = samples[:target_samples]  # Fallback to truncation
            else:
                logger.info(f"   ✅ Successfully parsed {len(refined_samples)} samples from LLM response")
            
            # Ensure we have the right number of samples
            if len(refined_samples) > target_samples:
                refined_samples = refined_samples[:target_samples]
                logger.info(f"   ✂️  Trimmed to {target_samples} samples")
            elif len(refined_samples) < target_samples:
                logger.warning(f"   ⚠️  LLM returned {len(refined_samples)} samples, expected {target_samples}")
                # Pad with original samples if needed
                remaining_needed = target_samples - len(refined_samples)
                original_remaining = [s for s in samples if s not in refined_samples]
                refined_samples.extend(original_remaining[:remaining_needed])
                logger.info(f"   🔧 Padded with {remaining_needed} original samples to reach target")
            
            logger.info(f"✅ COMPLETED: {main_intent} -> {sub_intent} | Final count: {len(refined_samples)} samples")
            logger.info("=" * 80)  # Separator line for better readability
            
            return refined_samples
            
        except Exception as e:
            logger.error(f"❌ ERROR refining samples for {main_intent}->{sub_intent}: {str(e)}")
            logger.info(f"🔄 Falling back to simple truncation for {main_intent}->{sub_intent}")
            logger.info("=" * 80)  # Separator line for better readability
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
            logger.info(f"📂 Loading dataset from {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Try to extract the dictionary from the file
            # Look for patterns like "intent_samples = {" or just "{"
            if '=' in content:
                # Split by '=' and take the part after it
                dict_part = content.split('=', 1)[1].strip()
            else:
                dict_part = content.strip()
            
            # Clean content before parsing - replace ellipsis patterns
            dict_part = dict_part.replace('...', '"ELLIPSIS_PLACEHOLDER"')
            dict_part = dict_part.replace('Ellipsis', '"ELLIPSIS_PLACEHOLDER"')
            
            # Parse the dictionary
            dataset = ast.literal_eval(dict_part)
            
            # Clean the loaded dataset
            dataset = self._clean_dataset_for_serialization(dataset)
            
            logger.info(f"✅ Successfully loaded dataset with {len(dataset)} main intents")
            
            # Log some statistics
            total_sub_intents = sum(len(sub_intents) for sub_intents in dataset.values())
            total_samples = sum(
                len(samples) for main_intent in dataset.values() 
                for samples in main_intent.values() if isinstance(samples, list)
            )
            logger.info(f"📊 Dataset statistics: {len(dataset)} main intents, {total_sub_intents} sub-intents, {total_samples} total samples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Error loading dataset from {file_path}: {str(e)}")
            logger.info("💡 Tip: Make sure your dataset file contains a valid Python dictionary")
            raise DatasetRefineryError(f"Failed to load dataset: {str(e)}")

    def _clean_dataset_for_serialization(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean dataset by removing/replacing non-serializable objects like ellipsis.
        
        Args:
            dataset: Raw dataset dictionary
            
        Returns:
            Cleaned dataset dictionary
        """
        def clean_value(value):
            if value is Ellipsis or value == "..." or value == "ELLIPSIS_PLACEHOLDER":
                return None  # Replace ellipsis with None
            elif isinstance(value, list):
                cleaned_list = []
                for item in value:
                    cleaned_item = clean_value(item)
                    if cleaned_item is not None and cleaned_item != "ELLIPSIS_PLACEHOLDER":
                        cleaned_list.append(cleaned_item)
                return cleaned_list
            elif isinstance(value, dict):
                cleaned_dict = {}
                for k, v in value.items():
                    cleaned_v = clean_value(v)
                    if cleaned_v is not None and cleaned_v != "ELLIPSIS_PLACEHOLDER":
                        cleaned_dict[k] = cleaned_v
                return cleaned_dict
            else:
                return value
        
        cleaned = clean_value(dataset)
        logger.info("🧹 Cleaned dataset from ellipsis and non-serializable objects")
        return cleaned

    def _clean_dataset_for_serialization(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean dataset by removing/replacing non-serializable objects like ellipsis.
        
        Args:
            dataset: Raw dataset dictionary
            
        Returns:
            Cleaned dataset dictionary
        """
        def clean_value(value):
            if value is Ellipsis or value == "...":
                return None  # Replace ellipsis with None
            elif isinstance(value, list):
                return [clean_value(item) for item in value if item is not Ellipsis and item != "..."]
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items() if v is not Ellipsis and v != "..."}
            else:
                return value
        
        return clean_value(dataset)

    def save_dataset(self, dataset: Dict[str, Any], output_path: str):
        """
        Save refined dataset to output file.
        
        Args:
            dataset: Refined dataset dictionary
            output_path: Path for output file
        """
        try:
            logger.info(f"💾 Saving refined dataset to {output_path}")
            
            # Clean dataset from ellipsis and other non-serializable objects
            cleaned_dataset = self._clean_dataset_for_serialization(dataset)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                # Write in the same format as input
                file.write("intent_samples = ")
                
                # Use a custom JSON encoder to handle any remaining issues
                try:
                    json_str = json.dumps(cleaned_dataset, indent=4, ensure_ascii=False)
                    file.write(json_str)
                except TypeError as json_error:
                    logger.warning(f"JSON serialization failed, using fallback method: {str(json_error)}")
                    # Fallback: use repr() which handles more Python objects
                    file.write(repr(cleaned_dataset))
            
            logger.info(f"✅ Successfully saved refined dataset to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving dataset to {output_path}: {str(e)}")
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
        
        logger.info(f"🚀 Starting dataset refinement process")
        logger.info(f"📝 Total sub-intents to process: {total_sub_intents}")
        logger.info(f"🎯 Target samples per sub-intent: {target_samples}")
        logger.info(f"🤖 Using model: {self.model_name}")
        logger.info("=" * 80)
        
        for main_intent, sub_intents in dataset.items():
            logger.info(f"📂 Processing main intent: {main_intent} ({len(sub_intents)} sub-intents)")
            refined_dataset[main_intent] = {}
            
            for sub_intent, samples in sub_intents.items():
                try:
                    # Refine samples for this sub-intent (one at a time as requested)
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
                    
                    # Small delay to be nice to the LLM and prevent overwhelming
                    time.sleep(1.0)  # Increased delay since this is a longer process
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {main_intent}->{sub_intent}: {str(e)}")
                    # Keep original samples on error, but clean them first
                    original_cleaned = [s for s in samples if s is not Ellipsis and s != "..." and isinstance(s, str)]
                    refined_dataset[main_intent][sub_intent] = original_cleaned[:target_samples]
                    processed += 1
                    logger.info("=" * 80)  # Separator line
        
        # Save refined dataset
        self.save_dataset(refined_dataset, output_path)
        
        logger.info(f"🎉 Dataset refinement completed successfully!")
        logger.info(f"📊 Final statistics: Processed {processed}/{total_sub_intents} sub-intents")
        logger.info("=" * 80)
        return refined_dataset

def progress_printer(processed: int, total: int, main_intent: str, sub_intent: str):
    """Enhanced progress printer for the refinement process"""
    percentage = (processed / total) * 100
    remaining = total - processed
    
    print(f"\n📈 PROGRESS UPDATE:")
    print(f"   ✅ Completed: {processed}/{total} ({percentage:.1f}%)")
    print(f"   🔄 Remaining: {remaining} sub-intents")
    print(f"   📝 Last processed: {main_intent} -> {sub_intent}")
    print(f"   {'='*50}")
    
    # Estimated time remaining (rough calculation)
    if processed > 0:
        avg_time_per_intent = 30  # Rough estimate: 30 seconds per intent
        estimated_remaining_time = remaining * avg_time_per_intent
        hours = estimated_remaining_time // 3600
        minutes = (estimated_remaining_time % 3600) // 60
        
        if hours > 0:
            print(f"   ⏱️  Estimated time remaining: ~{hours}h {minutes}m")
        else:
            print(f"   ⏱️  Estimated time remaining: ~{minutes}m")
        print(f"   {'='*50}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Refine intent classification dataset using local Ollama LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_refinery.py -i dataset.txt -o refined_dataset.txt -s 20
  python dataset_refinery.py -i data/intent_samples.txt -o output/refined.txt -s 15 --model llama3.3:latest
  python dataset_refinery.py -i dataset.txt -o refined.txt -s 10 --host http://192.168.1.100:11434
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Path to source dataset .txt file')
    parser.add_argument('-o', '--output', required=True,
                       help='Path for output refined dataset .txt file')
    parser.add_argument('-s', '--samples', type=int, required=True,
                       help='Target number of samples per sub-intent')
    parser.add_argument('--host', default='http://127.0.0.1:11434',
                       help='Ollama server host (default: http://127.0.0.1:11434)')
    parser.add_argument('--model', default='llama3.3:latest',
                       help='Model name to use (default: llama3.3:latest)')
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