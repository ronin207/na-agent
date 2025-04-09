import os
from flask import Flask, request, jsonify
from src.agentic_retrieval import AgenticRetrieval
import logging
from dotenv import load_dotenv
import asyncio

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
logger.debug(f"Loading .env file from: {env_path}")
load_dotenv(env_path)

# Verify API keys are loaded
openai_api_key = os.getenv("OPENAI_API_KEY")
logger.debug(f"OpenAI API Key loaded: {openai_api_key[:10]}...")  

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Initialize Flask app
app = Flask(__name__)

# Initialize RAG pipeline
logger.debug("Initializing RAG pipeline...")
rag_pipeline = AgenticRetrieval(
    pdf_folder="./data/",
    chunk_size=500,
    chunk_overlap=300,
    embedding_model="text-embedding-3-small",  # OpenAI embedding model
    llm_model="gpt-3.5-turbo",  # OpenAI model
    persist_directory="./chroma_db",
    temperature=0.0,
    k=15,
    rewrite_query=True,
    evaluate=True,
    self_rag_threshold=0.7,
    adaptive_rag=True,
    enable_corrective_rag=True,
    force_rebuild=False,
    test_mode=False,
    verbose=True,
    web_search_enabled=True,
    web_search_threshold=0.6,
    enable_web_search_in_test_mode=False,
    max_history_length=5
)

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        query_text = data['text']
        logger.info(f"Received query: {query_text}")
        
        # Process the query
        result = asyncio.run(rag_pipeline.invoke(query_text))
        
        # Format response for Mattermost
        answer = result['answer']
        
        # Add sources if available
        if 'sources' in result and result['sources']:
            sources = result['sources']
            # Deduplicate sources while preserving order
            seen = set()
            unique_sources = []
            for source in sources:
                if source not in seen:
                    seen.add(source)
                    # Extract just the filename from the path
                    source_name = source.split('/')[-1] if '/' in source else source
                    unique_sources.append(source_name)
            
            if unique_sources:
                sources_text = "\n\n**Sources:**\n" + "\n".join([f"- {source}" for source in unique_sources[:3]])
                if len(unique_sources) > 3:
                    sources_text += f"\n- ... and {len(unique_sources) - 3} more sources"
                answer += sources_text
        
        # Add web search notification if applicable
        if result.get('web_search_used', False):
            answer += "\n\n*Note: This answer includes information from web search results.*"
        
        response = {
            'text': answer,
            'response_type': 'in_channel'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Verify environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Ensure the RAG pipeline is set up before starting the Flask app
    logger.debug("Setting up RAG pipeline...")
    asyncio.run(rag_pipeline.setup())
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5001) 