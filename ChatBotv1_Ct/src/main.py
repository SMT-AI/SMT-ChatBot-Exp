from fastapi import FastAPI, HTTPException, Depends, Header, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from typing import Optional, Dict, List
import os
import time
import boto3
from botocore.exceptions import ClientError
import tempfile
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from BERTClassifier import IntentClassifier
from LLM_Gateway_Ollama import LLMGateway  # SHOULD CHANGE TO NIMS GATEWAY ON PRODUCTION!!!
from Corpus_Manager import CorpusManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
llm = None
s3_client = None
temp_dir = tempfile.mkdtemp(prefix="chatbot_")  # Create temp directory for downloads

# Cache structures for models and corpus
model_cache: Dict[str, IntentClassifier] = {}
corpus_cache: Dict[str, CorpusManager] = {}
cache_usage: Dict[str, float] = {}  # For LRU tracking
MAX_CACHE_SIZE = int(os.getenv('MAX_CACHE_SIZE', '5'))  # Number of models to keep in memory

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
        
        # Initialize LLM Gateway (shared across all organizations)
        llm = LLMGateway(
            base_url=os.getenv('OLLAMA_URL', 'http://localhost:11434'),
            model=os.getenv('LLM_MODEL', 'llama3.2:1b'),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.7'))
        )
        
        logger.info("LLM Gateway initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise
    
    yield  # This is where FastAPI runs
    
    # Shutdown code
    logger.info("Shutting down application")
    # Clean up temp directory
    try:
        import shutil
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up temp directory: {str(e)}")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SMT ChatBot v1",
    description="API for ERP chatbot with intent classification and response generation",
    version="1.0.0",
    lifespan=lifespan_context
)

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

# API Key validation
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv('API_KEY'):
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
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
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
async def health_check():
    """Health check endpoint"""
    try:
        # Try to list organization IDs to verify S3 connectivity
        org_ids = list_organization_ids()
        
        return {
            "status": "healthy",
            "s3_connection": "connected",
            "cache_stats": {
                "models_loaded": len(model_cache),
                "max_cache_size": MAX_CACHE_SIZE
            },
            "available_organizations": len(org_ids)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "s3_connection": "error",
            "error": str(e),
            "cache_stats": {
                "models_loaded": len(model_cache),
                "max_cache_size": MAX_CACHE_SIZE
            }
        }

@app.get("/api/organizations")
async def list_organizations(api_key: str = Depends(verify_api_key)):
    """List all available organizations"""
    try:
        org_ids = list_organization_ids()
        return {
            "status": "success",
            "organizations": org_ids
        }
    except Exception as e:
        logger.error(f"Error listing organizations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list organizations: {str(e)}"
        )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Main chat endpoint
    
    Args:
        request: ChatRequest containing query, organization_id and optional user/session info
        api_key: API key for authentication
    
    Returns:
        ChatResponse with generated response
    """
    try:
        # Log request (excluding sensitive data)
        logger.info(f"Received chat request - Organization ID: {request.organization_id}, Session ID: {request.session_id}")
        
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Empty query received"
            )
        
        # Load organization-specific resources
        classifier, corpus = load_organization_resources(request.organization_id)
        
        # Step 1: Intent Classification
        classification_result = classifier.predict(request.query)
        logger.info(f"Classification result for org {request.organization_id}: {classification_result}")
        
        # Step 2: Process based on classification
        if classification_result['status'] == 'llm_fallback':
            # Use LLM for out-of-domain queries
            llm_response = llm.generate_response("CONSIDER YOU ARE A CHATBOT NOW, DO NOT RESPOND MORE THAN 50 WORDS FOR THE FOLLOWING QUERY: " + request.query)
            
            if llm_response['status'] == 'error':
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM error: {llm_response.get('error')}"
                )
                
            return ChatResponse(
                status='success',
                response=llm_response['response'],
                source='llm',
                confidence=classification_result['confidence'],
                organization_id=request.organization_id
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
                confidence=classification_result['confidence'],
                organization_id=request.organization_id
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

# Resource management endpoints
@app.get("/api/admin/cache/stats")
async def get_cache_stats(api_key: str = Depends(verify_api_key)):
    """Get statistics about the model/corpus cache"""
    return {
        "status": "success",
        "cache_stats": {
            "models_loaded": len(model_cache),
            "organizations": list(model_cache.keys()),
            "max_cache_size": MAX_CACHE_SIZE
        }
    }

@app.post("/api/admin/cache/clear")
async def clear_cache(api_key: str = Depends(verify_api_key)):
    """Clear the model/corpus cache"""
    global model_cache, corpus_cache, cache_usage
    model_cache = {}
    corpus_cache = {}
    cache_usage = {}
    return {
        "status": "success",
        "message": "Cache cleared successfully"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "Internal server error",
            "status_code": 500
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)