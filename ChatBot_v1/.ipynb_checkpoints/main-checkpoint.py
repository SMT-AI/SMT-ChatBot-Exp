from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional
import os
from dotenv import load_dotenv

from BERTClassifier import IntentClassifier
from LLM_Gateway_Ollama import LLMGateway #SHOULD CHANGE TO NIMS GATEWAY ON PRODUCTION!!!
from Corpus_Manager import CorpusManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SMT ChatBot v1",
    description="API for ERP chatbot with intent classification and response generation",
    version="1.0.0"
)

#Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Models
class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    status: str
    response: str
    source: Optional[str] = None
    confidence: Optional[dict] = None
    
# Global variables for components
classifier = None
llm = None
corpus = None

# API Key validation
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv('API_KEY'):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global classifier, llm, corpus
    
    try:
        # Initialize BERT classifier
        classifier = IntentClassifier(
            model_path=os.getenv('MODEL_PATH'),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.8'))
        )
        
        # Initialize LLM Gateway
        llm = LLMGateway(
            base_url=os.getenv('OLLAMA_URL', 'http://localhost:11434'),
            model=os.getenv('LLM_MODEL', 'llama3.2:1b'),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.7'))
        )
        
        # Initialize Corpus Manager
        corpus = CorpusManager(os.getenv('CORPUS_PATH'))
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Main chat endpoint
    
    Args:
        request: ChatRequest containing query and optional user/session info
        api_key: API key for authentication
    
    Returns:
        ChatResponse with generated response
    """
    try:
        # Log request (excluding sensitive data)
        logger.info(f"Received chat request - Session ID: {request.session_id}")
        
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Empty query received"
            )
            
        # Step 1: Intent Classification
        classification_result = classifier.predict(request.query)
        logger.info(f"Classification result: {classification_result}")
        
        # Step 2: Process based on classification
        if classification_result['status'] == 'llm_fallback':
            # Use LLM for out-of-domain queries
            llm_response = llm.generate_response("CONSIDER YOU ARE A CHATBOT NOW, DO NOT RESPOND MORE THAN 50 WORDS FOR THE FOLLOWING QUERY: " +request.query)
            
            if llm_response['status'] == 'error':
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM error: {llm_response.get('error')}"
                )
                
            return ChatResponse(
                status='success',
                response=llm_response['response'],
                source='llm',
                confidence=classification_result['confidence']
            )
            
        elif classification_result['status'] == 'success':
            # Use corpus for in-domain queries
            corpus_response = corpus.get_response_text(
                classification_result['main_intent'],
                classification_result['sub_intent']
            )
            
            if corpus_response['status'] == 'error':
                raise HTTPException(
                    status_code=500,
                    detail=f"Corpus error: {corpus_response.get('error')}"
                )
                
            # Combine main and sub responses if both exist
            final_response = corpus_response['main_response']
            if corpus_response['sub_response']:
                final_response = f"{final_response}\n\n{corpus_response['sub_response']}"
                
            return ChatResponse(
                status='success',
                response=final_response,
                source='corpus',
                confidence=classification_result['confidence']
            )
            
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Classification error: {classification_result.get('error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "status": "error",
        "detail": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "status": "error",
        "detail": "Internal server error",
        "status_code": 500
    }

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)