import requests
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OllamaGatewayError(Exception):
    """Base exception class for Ollama gateway errors"""
    pass

class OllamaConnectionError(OllamaGatewayError):
    """Raised when connection to Ollama server fails"""
    pass

class OllamaResponseError(OllamaGatewayError):
    """Raised when Ollama response processing fails"""
    pass

class LLMGateway:
    """Gateway for interacting with local Ollama server"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model: str = "llama3.2:1b",
                 temperature: float = 0.7,
                 max_tokens: int = 500):
        """
        Initialize the LLM Gateway.
        
        Args:
            base_url: Ollama server URL
            model: Name of the model to use
            temperature: Response temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Validate server connection on initialization
        self._check_server_connection()
    
    def _check_server_connection(self):
        """Verify connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise OllamaConnectionError(
                    f"Failed to connect to Ollama server. Status code: {response.status_code}"
                )
            logger.info("Successfully connected to Ollama server")
        except requests.RequestException as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama server: {str(e)}")
    
    def set_model(self, model: str):  
        self.model = model
        logger.info(f"Model changed to: {model}")
    
    def set_parameters(self, temperature: Optional[float] = None, 
                      max_tokens: Optional[int] = None):
        """
        Update generation parameters.
        
        Args:
            temperature: New temperature value (0.0 to 1.0)
            max_tokens: New maximum tokens value
        """
        if temperature is not None:
            if not 0 <= temperature <= 1:
                raise ValueError("Temperature must be between 0 and 1")
            self.temperature = temperature
            
        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            self.max_tokens = max_tokens
            
        logger.info(f"Parameters updated - Temperature: {self.temperature}, "
                   f"Max Tokens: {self.max_tokens}")
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate response for a query using Ollama.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary containing:
                - status: 'success' or 'error'
                - response: Generated response text (if success)
                - error: Error message (if error)
        """
        if not query or not query.strip():
            return {
                'status': 'error',
                'error': 'Empty or invalid input query',
                'response': None
            }
            
        try:
            # Prepare the request
            endpoint = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": query,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False  # Get complete response at once
            }
            
            # Make the request
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            
            # Process response
            response_data = response.json()
            
            return {
                'status': 'success',
                'response': response_data.get('response', '').strip(),
                'model_used': self.model
            }
            
        except requests.RequestException as e:
            logger.error(f"Error communicating with Ollama server: {str(e)}")
            return {
                'status': 'error',
                'error': f"Failed to get response from Ollama: {str(e)}",
                'response': None
            }
        except Exception as e:
            logger.error(f"Unexpected error during response generation: {str(e)}")
            return {
                'status': 'error',
                'error': f"Unexpected error: {str(e)}",
                'response': None
            }