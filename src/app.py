from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager

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
    video_url: str = None  # Optional video URL parameter

# Load environment variables
env_path = os.path.join(project_root, '.env')
logger.debug(f"Loading .env file from: {env_path}")
load_dotenv(env_path)

# Set USER_AGENT environment variable
os.environ['USER_AGENT'] = 'na-agent/1.0'

# Verify API keys are loaded
openai_api_key = os.getenv("OPENAI_API_KEY")
logger.debug(f"OpenAI API Key loaded: {openai_api_key[:10]}...")  

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Initialize RAG pipeline
rag_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    try:
        # Initialize RAG pipeline with correct configuration
        rag_pipeline = AgenticRetrieval(
            pdf_folder=os.path.join(project_root, "data"),
            persist_directory=os.path.join(project_root, "chroma_db"),
            force_rebuild=False
        )
            
        logger.info("RAG pipeline initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        raise
    finally:
        # Cleanup if needed
        pass

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    if rag_pipeline.retriever is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    # Get document count safely
    try:
        doc_count = len(rag_pipeline.vectorstore.get()['ids']) if rag_pipeline.vectorstore else 0
    except:
        doc_count = 0
    
    return {"status": "healthy", "document_count": doc_count}

@app.post("/query")
async def process_query(query: QueryRequest):
    if rag_pipeline is None or rag_pipeline.retriever is None:
        logger.error("RAG pipeline not initialized")
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Log the incoming request
        logger.info(f"Received query request: {query.model_dump()}")
        
        # Process the query using the basic RAG pipeline
        logger.info(f"Processing query: {query.text}")
        response = rag_pipeline.invoke(query.text)
        
        # Format the response to match what mattermost_bot.py expects
        result = {
            "response": response.get('response', response.get('answer', 'No answer found')),  # Handle both formats
            "sources": response.get('sources', []),
            "web_search_used": response.get('web_search_used', False),
            "datasource": response.get('datasource', 'unknown'),
            "exercise_detected": response.get('exercise_detected', False),
            "related_lectures": response.get('related_lectures', []),
            "query_analysis": response.get('query_analysis', {})  # Add query analysis for debugging
        }
        
        # Log the response
        logger.info(f"Generated response with {len(result['sources'])} sources")
        
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