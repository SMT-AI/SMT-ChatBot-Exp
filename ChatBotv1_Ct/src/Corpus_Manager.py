import json
import logging
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CorpusManagerError(Exception):
    """Base exception class for Corpus Manager errors"""
    pass

class CorpusLoadError(CorpusManagerError):
    """Raised when corpus loading fails"""
    pass

class IntentNotFoundError(CorpusManagerError):
    """Raised when requested intent is not found in corpus"""
    pass

class CorpusManager:
    """Manager for handling response templates from JSON corpus"""
    
    def __init__(self, corpus_path: str):
        """
        Initialize the Corpus Manager.
        
        Args:
            corpus_path: Path to the JSON corpus file
        """
        self.corpus_path = corpus_path
        self.corpus_data = None
        self.load_corpus()
    
    def load_corpus(self):
        """Load the corpus from JSON file"""
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as file:
                self.corpus_data = json.load(file)
            logger.info(f"Successfully loaded corpus from {self.corpus_path}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in corpus file: {str(e)}")
            raise CorpusLoadError(f"Invalid JSON format in corpus file: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading corpus file: {str(e)}")
            raise CorpusLoadError(f"Failed to load corpus file: {str(e)}")
    
    def reload_corpus(self):
        """Reload the corpus from file"""
        logger.info("Reloading corpus file")
        self.load_corpus()
    
    def get_response_text(self, main_intent: str, sub_intent: Optional[str] = None) -> Dict[str, Any]:
        """
        Get response text for given intents.
        
        Args:
            main_intent: Main intent identifier
            sub_intent: Sub-intent identifier (optional)
            
        Returns:
            Dictionary containing:
                - status: 'success' or 'error'
                - main_response: Main intent response text
                - sub_response: Sub-intent response text (if applicable)
                - error: Error message (if error)
        """
        try:
            # Validate corpus data
            if not self.corpus_data:
                raise CorpusLoadError("Corpus data not loaded")
            
            # Check main intent
            if main_intent not in self.corpus_data:
                raise IntentNotFoundError(f"Main intent '{main_intent}' not found in corpus")
            
            main_response = self.corpus_data[main_intent]["response"]
            
            # If sub_intent is provided, get its response
            sub_response = None
            if sub_intent:
                if sub_intent not in self.corpus_data[main_intent]:
                    raise IntentNotFoundError(
                        f"Sub-intent '{sub_intent}' not found under main intent '{main_intent}'"
                    )
                sub_response = self.corpus_data[main_intent][sub_intent]["response"]
            
            return {
                'status': 'success',
                'main_response': main_response,
                'sub_response': sub_response
            }
            
        except IntentNotFoundError as e:
            logger.error(str(e))
            return {
                'status': 'error',
                'error': str(e),
                'main_response': None,
                'sub_response': None
            }
        except Exception as e:
            logger.error(f"Unexpected error retrieving response: {str(e)}")
            return {
                'status': 'error',
                'error': f"Failed to retrieve response: {str(e)}",
                'main_response': None,
                'sub_response': None
            }
    
    def validate_intent_structure(self) -> Dict[str, Any]:
        """
        Validate the structure of the loaded corpus.
        
        Returns:
            Dictionary containing validation results
        """
        try:
            if not self.corpus_data:
                raise CorpusLoadError("Corpus data not loaded")
            
            validation_results = {
                'status': 'success',
                'main_intents': [],
                'errors': []
            }
            
            for main_intent, data in self.corpus_data.items():
                intent_info = {
                    'main_intent': main_intent,
                    'has_main_response': 'response' in data,
                    'sub_intents': []
                }
                
                # Check main intent response
                if 'response' not in data:
                    validation_results['errors'].append(
                        f"Missing 'response' for main intent '{main_intent}'"
                    )
                
                # Check sub-intents
                for key, value in data.items():
                    if key != 'response':
                        if not isinstance(value, dict) or 'response' not in value:
                            validation_results['errors'].append(
                                f"Invalid sub-intent structure for '{key}' under '{main_intent}'"
                            )
                        intent_info['sub_intents'].append(key)
                
                validation_results['main_intents'].append(intent_info)
            
            if validation_results['errors']:
                validation_results['status'] = 'warning'
                
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during corpus validation: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'main_intents': [],
                'errors': [str(e)]
            }