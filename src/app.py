from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import after path setup
from src.agentic_retrieval import AgenticRetrieval

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request models
class QueryRequest(BaseModel):
    text: str
    video_url: str = None

# Load environment variables
env_path = os.path.join(project_root, '.env')
logger.debug(f"Loading .env file from: {env_path}")
load_dotenv(env_path)

# Set USER_AGENT environment variable
os.environ['USER_AGENT'] = 'na-agent/1.0'

# Verify API keys are loaded
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    logger.debug(f"OpenAI API Key loaded: {openai_api_key[:10]}...")  
else:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline globally
rag_pipeline = None

def initialize_rag_pipeline():
    """Initialize the RAG pipeline"""
    global rag_pipeline
    try:
        rag_pipeline = AgenticRetrieval(
            pdf_folder=os.path.join(project_root, "data"),
            persist_directory=os.path.join(project_root, "chroma_db"),
            force_rebuild=False
        )
        logger.info("RAG pipeline initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    logger.info("Starting up FastAPI application...")
    success = initialize_rag_pipeline()
    if not success:
        logger.error("Failed to initialize RAG pipeline during startup")
        raise Exception("Failed to initialize RAG pipeline")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    if rag_pipeline.retriever is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    # Count documents in vectorstore if available
    try:
        if rag_pipeline.vectorstore:
            # Simple way to get document count from Chroma
            doc_count = len(rag_pipeline.vectorstore.get()['ids'])
        else:
            doc_count = 0
    except:
        doc_count = "unknown"
    
    return {"status": "healthy", "document_count": doc_count}

@app.post("/query")
async def process_query(query: QueryRequest):
    """Process a query through the RAG pipeline"""
    if rag_pipeline is None:
        logger.error("RAG pipeline not initialized")
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    if rag_pipeline.retriever is None:
        logger.error("Vector store not initialized")
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        # Log the incoming request
        logger.info(f"Received query request: {query.dict()}")
        
        # Process the query using the existing invoke method
        logger.info(f"Processing query: {query.text}")
        response = rag_pipeline.invoke(query.text)
        
        # Log the response
        logger.info(f"Generated response keys: {response.keys()}")
        
        # Return the formatted result
        result = {
            "response": response.get('answer', 'No answer found'),
            "sources": response.get('sources', []),
            "web_search_used": response.get('web_search_used', False),
            "datasource": response.get('datasource', 'unknown')
        }
        
        logger.info(f"Returning result with response length: {len(result['response'])}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.exception("Full exception details:")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Verify environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Start the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=5001) 