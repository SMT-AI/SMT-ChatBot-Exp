from fastapi import FastAPI, HTTPException, Depends, Header, Response, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import json
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import tempfile
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Any
import pathlib

from BERTClassifier import IntentClassifier
from LLM_Gateway_LC import LLMGatewayLC  # Using the simplified LLM Gateway
from Corpus_Manager import CorpusManager

# Load environment variables
load_dotenv()

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up enhanced logging with daily files
def setup_daily_logging():
    """Set up logging to daily files in the required format"""
    # Create logs directory if it doesn't exist
    logs_dir = pathlib.Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Get current date for folder and file names
    today = datetime.now()
    folder_name = f"{today.day:02d}_{today.month:02d}_{today.year}_LOG"
    file_name = f"{today.day:02d}_{today.month:02d}_{today.year}.txt"
    
    # Create today's log directory if it doesn't exist
    log_dir = logs_dir / folder_name
    log_dir.mkdir(exist_ok=True)
    
    # Set up log file path
    log_file = log_dir / file_name
    
    # Create a file handler for the daily log file
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5  # Keep 5 backup files
    )
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Create a specific logger for detailed chat logs
    chat_logger = logging.getLogger("chatbot.transactions")
    chat_logger.setLevel(logging.INFO)
    chat_logger.addHandler(file_handler)
    
    # Add file handler to root logger as well
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    return chat_logger

# Initialize the chat logger
chat_logger = setup_daily_logging()

# Global variables
llm = None
s3_client = None
temp_dir = tempfile.mkdtemp(prefix="chatbot_")  # Create temp directory for downloads

# Cache structures for models and corpus
model_cache: Dict[str, IntentClassifier] = {}
corpus_cache: Dict[str, CorpusManager] = {}
cache_usage: Dict[str, float] = {}  # For LRU tracking
MAX_CACHE_SIZE = int(os.getenv('MAX_CACHE_SIZE', '10'))  # Number of models to keep in memory
MAX_HISTORY_LENGTH = int(os.getenv('MAX_HISTORY_LENGTH', '10'))  # Maximum conversation history length
SESSION_TTL = int(os.getenv('SESSION_TTL', '350'))  # Session timeout in seconds

# S3 configuration
S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_BUCKET = os.getenv('S3_BUCKET', 'chatbotv1')
S3_BASE_FOLDER = os.getenv('S3_BASE_FOLDER', 'org')

# Define the lifespan context manager
@asynccontextmanager
async def lifespan_context(app: FastAPI):
    # Startup code
    global llm, s3_client
    
    logger.info("Starting SMT ChatBot v1")
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY
        )
        
        logger.info(f"Successfully connected to S3 endpoint: {S3_ENDPOINT}")
        
        # Verify bucket exists
        try:
            s3_client.head_bucket(Bucket=S3_BUCKET)
            logger.info(f"Successfully connected to bucket: {S3_BUCKET}")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"Bucket {S3_BUCKET} does not exist")
                raise
            else:
                logger.error(f"Error connecting to bucket {S3_BUCKET}: {str(e)}")
                raise
        
        # Initialize LLM Gateway with LangChain integration
        # Changed model to llama-3.2-3b-instruct as requested
        llm = LLMGatewayLC(
            api_key=os.getenv('NVIDIA_API_KEY'),
            model=os.getenv('LLM_MODEL', 'meta/llama-3.2-3b-instruct'),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '500')),
            session_ttl=SESSION_TTL,
            max_history_length=MAX_HISTORY_LENGTH
        )
        
        logger.info("LLM Gateway initialized successfully with LangChain integration")
        chat_logger.info("Chat system initialized and ready to process requests")
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        chat_logger.error(f"CRITICAL: System initialization failed: {str(e)}")
        raise
    
    yield  # This is where FastAPI runs
    
    # Shutdown code
    logger.info("Shutting down application")
    chat_logger.info("Chat system shutting down")
    
    # Clean up temp directory
    try:
        import shutil
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up temp directory: {str(e)}")

# Utility function to log chat transactions in detail
def log_chat_transaction(request_id: str, 
                      organization_id: str,
                      session_id: Optional[str],
                      user_id: Optional[str],
                      user_query: str,
                      response: str,
                      source: str,
                      classification: Optional[Dict[str, Any]] = None,
                      duration_ms: Optional[float] = None):
    """
    Log detailed information about a chat transaction
    
    Args:
        request_id: Unique ID for this request
        organization_id: Organization ID
        session_id: Optional session ID
        user_id: Optional user ID
        user_query: The user's query text
        response: The system's response text
        source: Source of the response (corpus or llm)
        classification: Optional classification results
        duration_ms: Processing time in milliseconds
    """
    # Create log entry with all relevant information
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "organization_id": organization_id,
        "session_id": session_id,
        "user_id": user_id,
        "query": user_query,
        "response": response,
        "response_source": source,
        "processing_time_ms": duration_ms
    }
    
    # Add classification details if available
    if classification:
        if source == "corpus" and classification.get('status') == 'success':
            log_entry["main_intent"] = classification.get('main_intent')
            log_entry["sub_intent"] = classification.get('sub_intent')
            log_entry["confidence"] = classification.get('confidence')
        elif source == "llm" and classification.get('status') == 'llm_fallback':
            log_entry["confidence"] = classification.get('confidence')
    
    # Log the transaction
    chat_logger.info(f"CHAT_TRANSACTION: {json.dumps(log_entry)}")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SMT ChatBot v1",
    description="API for ERP chatbot with intent classification and response generation",
    version="1.0.0",
    lifespan=lifespan_context
)

# Add middleware to generate request IDs
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add a unique request ID to each request for tracking in logs"""
    request_id = f"req_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
    request.state.request_id = request_id
    
    # Measure request duration
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    # Add timing headers
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    response.headers["X-Request-ID"] = request_id
    
    return response

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Models
class ChatRequest(BaseModel):
    query: str
    organization_id: str = Field(..., description="Organization ID for model/corpus selection")
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    status: str
    response: str
    source: Optional[str] = None
    confidence: Optional[dict] = None
    organization_id: str

class SessionInfo(BaseModel):
    session_id: str
    last_access: str
    message_count: int
    corpus_response_count: int
    user_name: str

# API Key validation
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv('API_KEY'):
        logger.warning(f"Invalid API key attempt: {x_api_key[:5]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key

def list_organization_ids() -> List[str]:
    """List all available organization IDs from S3 bucket"""
    try:
        # List objects with the prefix of the base folder
        prefix = f"{S3_BASE_FOLDER}/"
        result = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, Delimiter='/')
        
        # Extract organization IDs from CommonPrefixes
        org_ids = []
        if 'CommonPrefixes' in result:
            for prefix_obj in result['CommonPrefixes']:
                prefix_path = prefix_obj['Prefix']
                # Extract the org_id from the path - format is "org/00001/"
                org_id = prefix_path.split('/')[-2]  # Get the second-to-last part
                org_ids.append(org_id)
        
        return org_ids
    
    except Exception as e:
        logger.error(f"Error listing organization IDs: {str(e)}")
        raise

def download_from_s3(org_id: str, file_type: str) -> str:
    """
    Download model or corpus file from S3 and return local path
    
    Args:
        org_id: Organization ID
        file_type: Either 'model' or 'corpus'
    
    Returns:
        Local path to the downloaded file
    """
    try:
        if file_type not in ['model', 'corpus']:
            raise ValueError(f"Invalid file_type: {file_type}")
        
        # Determine file paths with correct extensions
        extension = 'pt' if file_type == 'model' else 'json'  # Changed from 'pth' to 'pt'
        s3_key = f"{S3_BASE_FOLDER}/{org_id}/{org_id}_{file_type}.{extension}"
        local_path = os.path.join(temp_dir, f"{org_id}_{file_type}.{extension}")
        
        # Log the exact S3 path being accessed for debugging
        logger.info(f"Attempting to download from S3 path: {s3_key}")
        
        # Check if file already exists locally (from recent download)
        if os.path.exists(local_path):
            # Check if file is recent (less than 5 minutes old)
            file_age = time.time() - os.path.getmtime(local_path)
            if file_age < 300:  # 5 minutes in seconds
                logger.info(f"Using cached {file_type} file for {org_id}")
                return local_path
        
        # Download file from S3
        logger.info(f"Downloading {file_type} for organization {org_id} from S3")
        
        # Check if file exists before downloading
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Try alternative extension as fallback (.pth instead of .pt or vice versa)
                alt_extension = 'pth' if extension == 'pt' else 'pt'
                alt_s3_key = f"{S3_BASE_FOLDER}/{org_id}/{org_id}_{file_type}.{alt_extension}"
                logger.info(f"File not found, trying alternative path: {alt_s3_key}")
                
                try:
                    s3_client.head_object(Bucket=S3_BUCKET, Key=alt_s3_key)
                    # If we get here, the alternative path exists
                    s3_key = alt_s3_key
                    local_path = os.path.join(temp_dir, f"{org_id}_{file_type}.{alt_extension}")
                    logger.info(f"Found file at alternative path: {alt_s3_key}")
                except:
                    # Both paths failed
                    raise HTTPException(
                        status_code=404,
                        detail=f"{file_type.capitalize()} for organization {org_id} not found"
                    )
        
        # Download file
        s3_client.download_file(S3_BUCKET, s3_key, local_path)
        logger.info(f"Successfully downloaded {file_type} for {org_id} to {local_path}")
        
        return local_path
    
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.error(f"{file_type.capitalize()} file not found for {org_id}")
            raise HTTPException(
                status_code=404,
                detail=f"{file_type.capitalize()} for organization {org_id} not found"
            )
        else:
            logger.error(f"Error downloading {file_type} for {org_id}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error accessing {file_type} for organization {org_id}"
            )
    except Exception as e:
        logger.error(f"Unexpected error downloading {file_type} for {org_id}: {str(e)}")
        raise

def load_organization_resources(org_id: str):
    """Load or retrieve from cache the classifier and corpus for the organization"""
    current_time = time.time()
    
    # Check if already in cache
    if org_id in model_cache and org_id in corpus_cache:
        # Update last access time
        cache_usage[org_id] = current_time
        return model_cache[org_id], corpus_cache[org_id]
    
    # If cache is full, remove least recently used
    if len(model_cache) >= MAX_CACHE_SIZE:
        lru_org = min(cache_usage.items(), key=lambda x: x[1])[0]
        del model_cache[lru_org]
        del corpus_cache[lru_org]
        del cache_usage[lru_org]
        logger.info(f"Removed resources for organization {lru_org} from cache due to LRU policy")
    
    # Download resources from S3
    try:
        model_path = download_from_s3(org_id, 'model')
        corpus_path = download_from_s3(org_id, 'corpus')
        
        # Load resources
        logger.info(f"Loading model for organization {org_id}")
        classifier = IntentClassifier(
            model_path=model_path,
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.8'))
        )
        
        logger.info(f"Loading corpus for organization {org_id}")
        corpus = CorpusManager(corpus_path)
        
        # Add to cache
        model_cache[org_id] = classifier
        corpus_cache[org_id] = corpus
        cache_usage[org_id] = current_time
        
        return classifier, corpus
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading resources for organization {org_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load resources for organization {org_id}: {str(e)}"
        )

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    request_id = request.state.request_id
    logger.info(f"Health check requested [request_id={request_id}]")
    
    try:
        # Try to list organization IDs to verify S3 connectivity
        org_ids = list_organization_ids()
        
        # Check LLM sessions
        active_sessions = llm.get_sessions_count()
        
        # Check if log directories exist
        logs_dir = pathlib.Path("logs")
        today = datetime.now()
        folder_name = f"{today.day:02d}_{today.month:02d}_{today.year}_LOG"
        log_dir = logs_dir / folder_name
        log_file = log_dir / f"{today.day:02d}_{today.month:02d}_{today.year}.txt"
        
        log_status = {
            "logs_dir_exists": logs_dir.exists(),
            "today_log_dir_exists": log_dir.exists(),
            "log_file_exists": log_file.exists(),
            "log_file_path": str(log_file)
        }
        
        chat_logger.info(f"Health check passed [request_id={request_id}]")
        
        return {
            "status": "healthy",
            "request_id": request_id,
            "s3_connection": "connected",
            "llm_sessions": active_sessions,
            "logging": log_status,
            "llm_model": llm.model,
            "cache_stats": {
                "models_loaded": len(model_cache),
                "max_cache_size": MAX_CACHE_SIZE
            },
            "available_organizations": len(org_ids)
        }
    except Exception as e:
        logger.error(f"Health check failed [request_id={request_id}]: {str(e)}")
        chat_logger.error(f"Health check failed [request_id={request_id}]: {str(e)}")
        
        return {
            "status": "unhealthy",
            "request_id": request_id,
            "s3_connection": "error",
            "error": str(e),
            "cache_stats": {
                "models_loaded": len(model_cache),
                "max_cache_size": MAX_CACHE_SIZE
            }
        }

@app.get("/api/organizations")
async def list_organizations(request: Request, api_key: str = Depends(verify_api_key)):
    """List all available organizations"""
    request_id = request.state.request_id
    logger.info(f"Organization list requested [request_id={request_id}]")
    
    try:
        org_ids = list_organization_ids()
        chat_logger.info(f"Listed {len(org_ids)} organizations [request_id={request_id}]")
        
        return {
            "status": "success",
            "request_id": request_id,
            "organizations": org_ids
        }
    except Exception as e:
        logger.error(f"Error listing organizations [request_id={request_id}]: {str(e)}")
        chat_logger.error(f"Failed to list organizations [request_id={request_id}]: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list organizations: {str(e)}"
        )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Main chat endpoint
    
    Args:
        request: FastAPI request object
        chat_request: ChatRequest containing query, organization_id and optional user/session info
        api_key: API key for authentication
    
    Returns:
        ChatResponse with generated response
    """
    start_time = time.time()
    request_id = request.state.request_id
    
    # Log detailed request information
    logger.info(
        f"Chat request [request_id={request_id}] - "
        f"Org: {chat_request.organization_id}, "
        f"Session: {chat_request.session_id or 'None'}, "
        f"User: {chat_request.user_id or 'None'}"
    )
    chat_logger.info(
        f"Received query [request_id={request_id}]: "
        f"'{chat_request.query[:50]}{'...' if len(chat_request.query) > 50 else ''}'"
    )
    
    try:
        # Validate query
        if not chat_request.query or not chat_request.query.strip():
            logger.warning(f"Empty query received [request_id={request_id}]")
            chat_logger.warning(f"Empty query rejected [request_id={request_id}]")
            
            raise HTTPException(
                status_code=400,
                detail="Empty query received"
            )
        
        # Load organization-specific resources
        classifier, corpus = load_organization_resources(chat_request.organization_id)
        
        # Step 1: Intent Classification
        classification_start = time.time()
        classification_result = classifier.predict(chat_request.query)
        classification_time = (time.time() - classification_start) * 1000
        
        logger.info(
            f"Classification [request_id={request_id}] - "
            f"Status: {classification_result['status']}, "
            f"Time: {classification_time:.2f}ms"
        )
        
        # Step 2: Process based on classification
        if classification_result['status'] == 'llm_fallback':
            # For LLM fallback, check if there's session history and use enhanced LLM
            llm_start = time.time()
            
            # Get word limit from environment or use default
            word_limit = int(os.getenv('LLM_WORD_LIMIT', '50'))
            
            # Use LLM with session context if available
            llm_response = llm.generate_response(
                query=chat_request.query, 
                session_id=chat_request.session_id,
                word_limit=word_limit
            )
            llm_time = (time.time() - llm_start) * 1000
            
            logger.info(
                f"LLM response [request_id={request_id}] - "
                f"Status: {llm_response['status']}, "
                f"Time: {llm_time:.2f}ms, "
                f"Model: {llm_response.get('model_used', 'unknown')}"
            )
            
            if llm_response['status'] == 'error':
                logger.error(
                    f"LLM error [request_id={request_id}]: {llm_response.get('error')}"
                )
                chat_logger.error(
                    f"LLM failed [request_id={request_id}]: {llm_response.get('error')}"
                )
                
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM error: {llm_response.get('error')}"
                )
            
            # Prepare response
            response = ChatResponse(
                status='success',
                response=llm_response['response'],
                source='llm',
                confidence=classification_result['confidence'],
                organization_id=chat_request.organization_id
            )
            
            # Log the transaction before returning
            total_time = (time.time() - start_time) * 1000
            log_chat_transaction(
                request_id=request_id,
                organization_id=chat_request.organization_id,
                session_id=chat_request.session_id,
                user_id=chat_request.user_id,
                user_query=chat_request.query,
                response=llm_response['response'],
                source="llm",
                classification=classification_result,
                duration_ms=total_time
            )
            
            return response
            
        elif classification_result['status'] == 'success':
            # Use corpus for in-domain queries
            corpus_start = time.time()
            corpus_response = corpus.get_response_text(
                classification_result['main_intent'],
                classification_result['sub_intent']
            )
            corpus_time = (time.time() - corpus_start) * 1000
            
            logger.info(
                f"Corpus response [request_id={request_id}] - "
                f"Status: {corpus_response['status']}, "
                f"Time: {corpus_time:.2f}ms, "
                f"Intent: {classification_result['main_intent']}/{classification_result.get('sub_intent', 'None')}"
            )
            
            if corpus_response['status'] == 'error':
                logger.error(
                    f"Corpus error [request_id={request_id}]: {corpus_response.get('error')}"
                )
                chat_logger.error(
                    f"Corpus failed [request_id={request_id}]: {corpus_response.get('error')}"
                )
                
                raise HTTPException(
                    status_code=500,
                    detail=f"Corpus error: {corpus_response.get('error')}"
                )
                
            # Combine main and sub responses if both exist
            final_response = corpus_response['main_response']
            if corpus_response['sub_response']:
                final_response = f"{final_response}\n\n{corpus_response['sub_response']}"
            
            # Store corpus response in the LLM conversation history if session_id provided
            if chat_request.session_id:
                # Add the corpus response to the LLM session for future context
                llm.add_corpus_response_to_session(
                    session_id=chat_request.session_id,
                    main_intent=classification_result['main_intent'],
                    sub_intent=classification_result['sub_intent'],
                    response_text=final_response
                )
                
                logger.info(
                    f"Added corpus response to session [request_id={request_id}, "
                    f"session_id={chat_request.session_id}]"
                )
            
            # Prepare response
            response = ChatResponse(
                status='success',
                response=final_response,
                source='corpus',
                confidence=classification_result['confidence'],
                organization_id=chat_request.organization_id
            )
            
            # Log the transaction before returning
            total_time = (time.time() - start_time) * 1000
            log_chat_transaction(
                request_id=request_id,
                organization_id=chat_request.organization_id,
                session_id=chat_request.session_id,
                user_id=chat_request.user_id,
                user_query=chat_request.query,
                response=final_response,
                source="corpus",
                classification=classification_result,
                duration_ms=total_time
            )
            
            return response
            
        else:
            logger.error(
                f"Classification error [request_id={request_id}]: "
                f"{classification_result.get('error')}"
            )
            chat_logger.error(
                f"Classification failed [request_id={request_id}]: "
                f"{classification_result.get('error')}"
            )
            
            raise HTTPException(
                status_code=500,
                detail=f"Classification error: {classification_result.get('error')}"
            )
            
    except HTTPException as he:
        # Re-raise HTTP exceptions
        total_time = (time.time() - start_time) * 1000
        chat_logger.error(
            f"HTTP error in chat [request_id={request_id}, "
            f"status={he.status_code}]: {he.detail} "
            f"(processing time: {total_time:.2f}ms)"
        )
        raise
        
    except Exception as e:
        # Log and convert general exceptions
        total_time = (time.time() - start_time) * 1000
        logger.error(
            f"Error processing chat request [request_id={request_id}]: {str(e)}"
        )
        chat_logger.error(
            f"Failed chat request [request_id={request_id}]: {str(e)} "
            f"(processing time: {total_time:.2f}ms)"
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Resource management endpoints
@app.get("/api/admin/cache/stats")
async def get_cache_stats(request: Request, api_key: str = Depends(verify_api_key)):
    """Get statistics about the model/corpus cache"""
    request_id = request.state.request_id
    logger.info(f"Cache stats requested [request_id={request_id}]")
    
    return {
        "status": "success",
        "request_id": request_id,
        "cache_stats": {
            "models_loaded": len(model_cache),
            "organizations": list(model_cache.keys()),
            "max_cache_size": MAX_CACHE_SIZE
        }
    }

@app.post("/api/admin/cache/clear")
async def clear_cache(request: Request, api_key: str = Depends(verify_api_key)):
    """Clear the model/corpus cache"""
    request_id = request.state.request_id
    logger.info(f"Cache clear requested [request_id={request_id}]")
    
    global model_cache, corpus_cache, cache_usage
    model_cache = {}
    corpus_cache = {}
    cache_usage = {}
    
    chat_logger.info(f"Model and corpus cache cleared [request_id={request_id}]")
    
    return {
        "status": "success",
        "request_id": request_id,
        "message": "Cache cleared successfully"
    }

# Session management endpoints
@app.get("/api/admin/sessions/stats")
async def get_session_stats(request: Request, api_key: str = Depends(verify_api_key)):
    """Get statistics about active conversation sessions"""
    request_id = request.state.request_id
    logger.info(f"Session stats requested [request_id={request_id}]")
    
    return {
        "status": "success",
        "request_id": request_id,
        "active_sessions": llm.get_sessions_count(),
        "session_ttl": SESSION_TTL,
        "max_history_length": MAX_HISTORY_LENGTH
    }

@app.get("/api/admin/sessions/{session_id}")
async def get_session_info(
    session_id: str, 
    request: Request, 
    api_key: str = Depends(verify_api_key)
):
    """Get information about a specific session"""
    request_id = request.state.request_id
    logger.info(f"Session info requested for {session_id} [request_id={request_id}]")
    
    session_info = llm.get_session_info(session_id)
    if not session_info:
        logger.warning(f"Session {session_id} not found [request_id={request_id}]")
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    return {
        "status": "success",
        "request_id": request_id,
        "session_info": session_info
    }

@app.post("/api/admin/sessions/{session_id}/clear")
async def clear_session(
    session_id: str, 
    request: Request, 
    api_key: str = Depends(verify_api_key)
):
    """Clear a specific session's history"""
    request_id = request.state.request_id
    logger.info(f"Session clear requested for {session_id} [request_id={request_id}]")
    
    success = llm.clear_session(session_id)
    if not success:
        logger.warning(f"Session {session_id} not found for clearing [request_id={request_id}]")
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    chat_logger.info(f"Session {session_id} cleared [request_id={request_id}]")
    
    return {
        "status": "success",
        "request_id": request_id,
        "message": f"Session {session_id} cleared successfully"
    }

@app.post("/api/admin/sessions/clear-all")
async def clear_all_sessions(request: Request, api_key: str = Depends(verify_api_key)):
    """Clear all active sessions"""
    request_id = request.state.request_id
    logger.info(f"Clear all sessions requested [request_id={request_id}]")
    
    sessions_count = llm.get_sessions_count()
    
    # Reinitialize the LLM to clear all sessions
    #global llm
    llm = LLMGatewayLC(
        api_key=os.getenv('NVIDIA_API_KEY'),
        model=os.getenv('LLM_MODEL', 'meta/llama-3.2-3b-instruct'),
        temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
        max_tokens=int(os.getenv('LLM_MAX_TOKENS', '500')),
        session_ttl=SESSION_TTL,
        max_history_length=MAX_HISTORY_LENGTH
    )
    
    chat_logger.info(f"All {sessions_count} sessions cleared [request_id={request_id}]")
    
    return {
        "status": "success",
        "request_id": request_id,
        "message": f"All sessions cleared ({sessions_count} sessions)",
    }

# Logging management endpoints
@app.get("/api/admin/logs/info")
async def get_logs_info(request: Request, api_key: str = Depends(verify_api_key)):
    """Get information about the logging system"""
    request_id = request.state.request_id
    logger.info(f"Logs info requested [request_id={request_id}]")
    
    # Get list of log directories
    logs_dir = pathlib.Path("logs")
    if not logs_dir.exists():
        return {
            "status": "warning",
            "request_id": request_id,
            "message": "Logs directory does not exist",
            "logs_dir": str(logs_dir)
        }
    
    # List all log folders
    log_folders = [d for d in logs_dir.iterdir() if d.is_dir()]
    folder_info = []
    
    for folder in log_folders:
        # Get log files in the folder
        log_files = [f for f in folder.iterdir() if f.is_file() and f.name.endswith('.txt')]
        
        # Get folder stats
        folder_stats = {
            "folder_name": folder.name,
            "log_count": len(log_files),
            "log_files": [f.name for f in log_files],
            "size_bytes": sum(f.stat().st_size for f in log_files)
        }
        folder_info.append(folder_stats)
    
    return {
        "status": "success",
        "request_id": request_id,
        "logs_dir": str(logs_dir),
        "folder_count": len(log_folders),
        "folders": folder_info
    }

@app.post("/api/admin/logs/rotate")
async def rotate_logs(request: Request, api_key: str = Depends(verify_api_key)):
    """Force rotation of the current log file"""
    request_id = request.state.request_id
    logger.info(f"Log rotation requested [request_id={request_id}]")
    
    try:
        # Set up a new logger instance
        chat_logger = setup_daily_logging()
        
        return {
            "status": "success",
            "request_id": request_id,
            "message": "Log rotation completed successfully"
        }
    except Exception as e:
        logger.error(f"Error rotating logs [request_id={request_id}]: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rotate logs: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning(
        f"HTTP exception [request_id={request_id}, status={exc.status_code}]: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "request_id": request_id,
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(f"Unhandled exception [request_id={request_id}]: {str(exc)}")
    chat_logger.error(f"CRITICAL: Unhandled exception [request_id={request_id}]: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "request_id": request_id,
            "detail": "Internal server error",
            "status_code": 500
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Make sure logging is set up
    logger.info("Starting the application with enhanced logging")
    chat_logger.info("Application startup initiated")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=9000)