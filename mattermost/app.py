from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv
import asyncio
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

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
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
        # Initialize RAG pipeline with your configuration
        rag_pipeline = AgenticRetrieval(
            pdf_folder=os.path.join(os.path.dirname(__file__), "data"),
            persist_directory=os.path.join(project_root, "mattermost/chroma_db"),
            force_rebuild=False,
            verbose=True
        )
        
        # Setup the pipeline
        success = await rag_pipeline.setup()
        if not success:
            logger.error("Failed to setup RAG pipeline")
            raise Exception("Failed to setup RAG pipeline")
            
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
    return {"status": "healthy", "document_count": rag_pipeline.get_document_count()}

@app.post("/query")
async def process_query(query: QueryRequest):
    if rag_pipeline is None or rag_pipeline.retriever is None:
        logger.error("RAG pipeline not initialized")
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Log the incoming request
        logger.info(f"Received query request: {query.dict()}")
        
        # Process the query
        logger.info(f"Processing query: {query.text}")
        response = await rag_pipeline.invoke(query.text)
        
        # Log the response
        logger.info(f"Generated response: {response}")
        
        result = {"response": response.get('answer', 'No answer found')}
        logger.info(f"Returning result: {result}")
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