import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# LangChain imports
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NIMGatewayError(Exception):
    """Base exception class for NVIDIA NIM gateway errors"""
    pass

class NIMConnectionError(NIMGatewayError):
    """Raised when connection to NVIDIA NIM API fails"""
    pass

class NIMResponseError(NIMGatewayError):
    """Raised when NIM response processing fails"""
    pass

class ConversationManagerError(NIMGatewayError):
    """Raised when conversation management fails"""
    pass

class LLMGatewayLC:
    """Enhanced Gateway for interacting with NVIDIA NIM API using LangChain"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "meta/llama-3.2-3b-instruct",
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 session_ttl: int = 60,
                 max_history_length: int = 25):
        """
        Initialize the LLM Gateway for NVIDIA NIM with LangChain integration.
        
        Args:
            api_key: API key for NVIDIA NIM (pulled from env if None)
            model: Name of the model to use
            temperature: Response temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            session_ttl: Time in seconds for conversation session to expire
            max_history_length: Maximum number of message pairs to retain in history
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY environment variable not set and no api_key provided")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_ttl = session_ttl
        self.max_history_length = max_history_length
        
        # Initialize LangChain ChatNVIDIA LLM
        self.llm = ChatNVIDIA(
            model=self.model,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Conversation store: session_id -> conversation data
        self.conversation_store = {}
        
        # Validate connection on initialization
        self._check_api_connection()
        
        logger.info(f"LLM Gateway initialized with model {model} and history limit {max_history_length}")
    
    def _check_api_connection(self):
        """Verify connection to NVIDIA NIM API"""
        try:
            # Simple ping to test connection with minimal token usage
            response = self.llm.invoke("Hi")
            logger.info("Successfully connected to NVIDIA NIM API using LangChain")
        except Exception as e:
            logger.error(f"Failed to connect to NVIDIA NIM API: {str(e)}")
            raise NIMConnectionError(f"Failed to connect to NVIDIA NIM API: {str(e)}")
    
    def set_model(self, model: str):
        """
        Change the model being used.
        
        Args:
            model: New model to use
        """
        self.model = model
        # Recreate the LLM with the new model
        self.llm = ChatNVIDIA(
            model=self.model,
            api_key=os.environ.get("NVIDIA_API_KEY"),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
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
        
        # Update the LLM with new parameters
        self.llm = ChatNVIDIA(
            model=self.model,
            api_key=os.environ.get("NVIDIA_API_KEY"),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
            
        logger.info(f"Parameters updated - Temperature: {self.temperature}, "
                   f"Max Tokens: {self.max_tokens}")
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired conversation sessions based on TTL.
        
        Returns:
            Number of sessions removed
        """
        current_time = time.time()
        expired_sessions = []
        
        for session_id, data in self.conversation_store.items():
            if current_time - data["last_access"] > self.session_ttl:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.conversation_store[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
        return len(expired_sessions)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information or None if not found
        """
        if session_id not in self.conversation_store:
            return None
            
        session = self.conversation_store[session_id]
        return {
            "last_access": datetime.fromtimestamp(session["last_access"]).isoformat(),
            "user_name": session["user_name"],
            "message_count": len(session["history"]) // 2,  # Divide by 2 to count conversation turns
            "corpus_response_count": len(session["corpus_responses"])
        }
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a specific session's history.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False if session not found
        """
        if session_id not in self.conversation_store:
            return False
            
        del self.conversation_store[session_id]
        return True
    
    def get_sessions_count(self) -> int:
        """
        Get the current number of active sessions.
        
        Returns:
            Number of active sessions
        """
        return len(self.conversation_store)
    
    def generate_response(self, query: str, 
                         session_id: Optional[str] = None,
                         corpus_responses: Optional[List[Dict[str, Any]]] = None,
                         word_limit: int = 50) -> Dict[str, Any]:
        """
        Generate response for a query using NVIDIA NIM with conversation awareness.
        
        This maintains the same return signature as the original for compatibility.
        
        Args:
            query: User query text
            session_id: Optional session identifier for conversation continuity
            corpus_responses: Optional list of previous corpus responses
            word_limit: Maximum number of words in the response (50-100)
            
        Returns:
            Dictionary containing:
                - status: 'success' or 'error'
                - response: Generated response text (if success)
                - error: Error message (if error)
                - model_used: Name of the model used
        """
        # Input validation
        if not query or not query.strip():
            return {
                'status': 'error',
                'error': 'Empty or invalid input query',
                'response': None,
                'model_used': self.model
            }
        
        # Enforce word limit bounds
        if word_limit < 50:
            word_limit = 50
        elif word_limit > 100:
            word_limit = 100
            
        try:
            # Clean up expired sessions first
            self.cleanup_expired_sessions()
            
            # Prepare the system message with appropriate context
            if session_id:
                # Initialize or get session data
                current_time = time.time()
                user_name = session_id  # Default to session_id as user_name
                
                if session_id in self.conversation_store:
                    # Get existing session data
                    session = self.conversation_store[session_id]
                    session["last_access"] = current_time
                    user_name = session["user_name"]
                    history = session["history"]
                    
                    # Add any new corpus responses
                    if corpus_responses:
                        for resp in corpus_responses:
                            if resp not in session["corpus_responses"]:
                                session["corpus_responses"].append(resp)
                else:
                    # Create new session
                    history = []
                    self.conversation_store[session_id] = {
                        "last_access": current_time,
                        "user_name": user_name,
                        "history": history,
                        "corpus_responses": corpus_responses or []
                    }
                
                # Build system message with user name
                system_message = (
                    f"You are SMT-Bot, an assistant helping users with their questions. "
                    f"The user's name is {user_name}. Remember this name throughout the conversation. "
                    f"Always address the user by their name when appropriate. "
                    f"Keep your responses under {word_limit} words unless explicitly asked for more detail. "
                )
                
                # Add corpus context if available
                corpus_context = ""
                if corpus_responses or (session_id in self.conversation_store and 
                                        self.conversation_store[session_id]["corpus_responses"]):
                    all_corpus = corpus_responses or []
                    if session_id in self.conversation_store:
                        all_corpus.extend(self.conversation_store[session_id]["corpus_responses"])
                    
                    # Format corpus context
                    corpus_info = []
                    for resp in all_corpus[-3:]:  # Take only the most recent 3
                        intent = resp.get('main_intent', 'unknown')
                        sub_intent = resp.get('sub_intent', '')
                        response = resp.get('response', '')[:100]  # Truncate long responses
                        
                        corpus_info.append(f"- {intent}/{sub_intent}: {response}...")
                    
                    if corpus_info:
                        corpus_context = "\n\nRecent information from the ERP system:\n" + "\n".join(corpus_info)
                
                # Build the complete system message
                system_message += corpus_context
                
                # Convert history to the format expected by OpenAI API
                formatted_history = []
                for i, msg in enumerate(history):
                    formatted_history.append(msg)
                
                # Add the current query
                formatted_history.append({"role": "user", "content": query})
                
                # Generate response using OpenAI-compatible format
                messages = [{"role": "system", "content": system_message}] + formatted_history
                response_text = self.llm.invoke(messages)
                
                # Store the conversation
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response_text.content})
                
                # Prune history if it exceeds the maximum length
                # We want to keep pairs intact, so we remove the oldest pairs first
                if len(history) > self.max_history_length * 2:
                    # Keep the most recent messages
                    history[:] = history[-self.max_history_length*2:]
                    
            else:
                # For stateless requests, use a simple system message
                system_message = (
                    f"You are SMT-Bot, an ERP assistant helping users with their questions. "
                    f"Respond in a friendly and helpful manner. "
                    f"Do not mention any system related info (eg: '[ERP CORPUS]') to the end user, Kindly maintain integrity."
                    f"Keep your responses under {word_limit} words unless explicitly asked for more detail."
                )
                
                # Generate response
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ]
                response_text = self.llm.invoke(messages)
            
            return {
                'status': 'success',
                'response': response_text.content.strip(),
                'model_used': self.model
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'status': 'error',
                'error': f"Failed to get response: {str(e)}",
                'response': None,
                'model_used': self.model
            }
    
    def add_corpus_response_to_session(self, session_id: str, 
                                      main_intent: str,
                                      sub_intent: Optional[str],
                                      response_text: str) -> bool:
        """
        Add a corpus response to a user session without generating a new LLM response.
        
        Args:
            session_id: Session identifier
            main_intent: Main intent identified
            sub_intent: Sub-intent identified (optional)
            response_text: Text of the corpus response
            
        Returns:
            True if successful, False if session not found
        """
        if not session_id:
            return False
            
        # Create the session if it doesn't exist
        current_time = time.time()
        if session_id not in self.conversation_store:
            self.conversation_store[session_id] = {
                "last_access": current_time,
                "user_name": session_id,
                "history": [],
                "corpus_responses": []
            }
        
        # Add the corpus response to the session
        corpus_response = {
            'main_intent': main_intent,
            'sub_intent': sub_intent,
            'response': response_text,
            'timestamp': current_time
        }
        
        self.conversation_store[session_id]["corpus_responses"].append(corpus_response)
        self.conversation_store[session_id]["last_access"] = current_time
        
        # Also add this to the conversation history
        # For corpus responses, we create a synthetic dialogue
        history = self.conversation_store[session_id]["history"]
        
        # Add a synthetic user query based on the intent
        user_query = f"Tell me about {main_intent}"
        if sub_intent:
            user_query += f" {sub_intent}"
            
        # Add a synthetic system response
        system_response = f"[ERP CORPUS] {response_text}"
        
        # Add to history if it's not already there (checking exact matches)
        if not any(h.get("content") == user_query for h in history if h.get("role") == "user"):
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": system_response})
            
            # Prune history if needed
            if len(history) > self.max_history_length * 2:
                history[:] = history[-self.max_history_length*2:]
        
        return True