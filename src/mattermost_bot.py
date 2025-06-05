import os
import json
import logging
import threading
import requests
from threading import Event
from flask import Flask, request, jsonify
import websocket
import time
import re
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# CONVERSATIONAL SYSTEM: Import the new conversational components
from conversation import ConversationalAgenticRetrieval
from mattermost_conversational_integration import MattermostConversationalBot

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

# Configure logging - Use absolute path to logs directory
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'mattermost_bot.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
MATTERMOST_URL = os.getenv('MATTERMOST_URL')
BOT_TOKEN = os.getenv('BOT_TOKEN')
BOT_USERNAME = os.getenv('BOT_USERNAME', 'knowledge-agent')

# Debug logging
logger.info(f"Environment variables loaded:")
logger.info(f"MATTERMOST_URL: {MATTERMOST_URL}")
logger.info(f"BOT_TOKEN: {'set' if BOT_TOKEN else 'not set'}")
logger.info(f"BOT_USERNAME: {BOT_USERNAME}")

# CONVERSATIONAL SYSTEM: Initialize the conversational RAG system and bot
try:
    logger.info("🤖 Initializing Conversational RAG System...")
    rag_system = ConversationalAgenticRetrieval(
        pdf_folder="./data/",
        persist_directory="./chroma_db"
    )
    conversational_bot = MattermostConversationalBot(rag_system)
    logger.info("✅ Conversational RAG System initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize conversational system: {e}")
    # Fallback to None if initialization fails
    rag_system = None
    conversational_bot = None

# Initialize Flask app
app = Flask(__name__)

@app.route('/health')
def health_check():
    return "OK"

def connect_websocket():
    """Connect to Mattermost WebSocket"""
    ws_url = f"{MATTERMOST_URL.replace('http', 'ws')}/api/v4/websocket"
    logger.info(f"Connecting to WebSocket: {ws_url}")
    
    def on_message(ws, message):
        logger.debug(f"Received WebSocket message: {message}")

    def on_error(ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        time.sleep(5)  # Wait before reconnecting
        connect_websocket()

    def on_open(ws):
        logger.info("WebSocket connection established")
        # Authenticate with the WebSocket
        auth_message = {
            "seq": 1,
            "action": "authentication_challenge",
            "data": {
                "token": BOT_TOKEN
            }
        }
        ws.send(json.dumps(auth_message))

    # Create and start WebSocket connection
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    
    ws.run_forever()

def send_message(channel_id, message, response_type='in_channel', root_id=None):
    """Send a message to a Mattermost channel"""
    try:
        headers = {
            'Authorization': f'Bearer {BOT_TOKEN}',
            'Content-Type': 'application/json'
        }
    
        # Prepare the message data
        current_root_id = root_id
        if isinstance(current_root_id, str) and not current_root_id.strip(): # Treat empty string as None
            current_root_id = None

        data = {
            'channel_id': channel_id,
            'message': message,
        }
        if current_root_id: # Only add root_id if it's valid
            data['root_id'] = current_root_id
        
        logger.info(f"Sending message with data: {json.dumps(data)}") # Log what's being sent
        
        # Send the message
        response = requests.post(
            f'{MATTERMOST_URL}/api/v4/posts',
            headers=headers,
            json=data
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending message: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"Response text: {e.response.text}")
        return None

def show_typing(channel_id):
    """Show typing indicator in channel"""
    try:
        headers = {
            'Authorization': f'Bearer {BOT_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{MATTERMOST_URL}/api/v4/users/me/typing',
            headers=headers,
            json={'channel_id': channel_id}
        )
        
        response.raise_for_status()
        logger.debug(f"Successfully showed typing indicator in channel {channel_id}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error showing typing indicator: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")

def show_typing_continuous(channel_id, stop_event):
    """Continuously show typing indicator until stop_event is set"""
    while not stop_event.is_set():
        show_typing(channel_id)
        time.sleep(2)  # Show typing every 2 seconds

def extract_youtube_url(text: str) -> Optional[str]:
    """Extract YouTube URL from text if present."""
    youtube_patterns = [
        r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([^\s&]+)',
        r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com)\/(?:v|embed)\/([^\s&]+)',
        r'(?:https?:\/\/)?(?:www\.)?(?:youtu\.be)\/([^\s&]+)'
    ]
    
    for pattern in youtube_patterns:
        match = re.search(pattern, text)
        if match:
            # Return the full matched URL
            return match.group(0)
    return None

# CONVERSATIONAL SYSTEM: New async function to handle conversational processing
async def process_query_conversational(text, post_data):
    """Process a query using the conversational system with thread awareness"""
    try:
        if not conversational_bot:
            logger.warning("Conversational system not available, falling back to old method")
            return None
        
        logger.info(f"Processing conversational query: {text}")
        
        # Process the message with conversational awareness
        response_data = await conversational_bot.process_message(post_data)
        
        if not response_data.get("should_respond", True):
            return None
        
        # Format the response for Mattermost
        formatted_message = conversational_bot.format_response_for_mattermost(response_data)
        
        return {
            "message": formatted_message,
            "thread_metadata": response_data.get("thread_metadata", {}),
            "session_id": response_data.get("session_id"),
            "conversation_turns": response_data.get("conversation_turns", 0),
            "is_new_conversation": response_data.get("is_new_conversation", False)
        }
        
    except Exception as e:
        logger.error(f"Error in conversational processing: {e}")
        return None

def process_query(text, channel_id, user_name, root_id=None, post_data=None):
    """Process a query and return the response"""
    try:
        if not text:
            send_message(channel_id, "Please provide a query.", root_id=root_id)
            return jsonify({'response_type': 'in_channel'})
        
        # Create a stop event for the typing indicator
        stop_typing = Event()
        
        # Start typing indicator in a separate thread
        typing_thread = threading.Thread(
            target=show_typing_continuous,
            args=(channel_id, stop_typing)
        )
        typing_thread.start()

        try:
            # CONVERSATIONAL SYSTEM: Try conversational processing first
            conversational_result = None
            if conversational_bot and post_data:
                try:
                    import asyncio
                    # Create post data in the format expected by conversational system
                    conv_post_data = {
                        'id': post_data.get('id', root_id or 'unknown'),
                        'root_id': post_data.get('root_id'),  # This determines thread continuity
                        'message': text,
                        'channel_id': channel_id,
                        'user_id': post_data.get('user_id', user_name)
                    }
                    
                    # Run the async conversational processing
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    conversational_result = loop.run_until_complete(
                        process_query_conversational(text, conv_post_data)
                    )
                    loop.close()
                    
                    if conversational_result:
                        logger.info(f"✅ Conversational processing successful - Session: {conversational_result.get('session_id', 'N/A')}, Turns: {conversational_result.get('conversation_turns', 'N/A')}")
                        
                        # Stop the typing indicator
                        stop_typing.set()
                        typing_thread.join(timeout=1)
                        
                        # Use the root_id from thread metadata for proper threading
                        thread_root_id = conversational_result.get("thread_metadata", {}).get("root_id")
                        if isinstance(thread_root_id, str) and not thread_root_id.strip():
                            thread_root_id = None
                        
                        if not thread_root_id and conv_post_data.get('root_id'): # If conversational system didn't provide one, but original post had one
                            thread_root_id = conv_post_data['root_id']

                        logger.info(f"Sending conversational response to channel {channel_id}, thread_root_id: {thread_root_id}")
                        send_message(channel_id, conversational_result["message"], root_id=thread_root_id)
                        return jsonify({'response_type': 'in_channel'})
                    else:
                        logger.warning("⚠️ Conversational processing did not return a result (result is None). Falling back.")
                        
                except Exception as conv_error:
                    logger.warning(f"⚠️ Conversational processing failed with error: {conv_error}. Falling back to original method.")
                    conversational_result = None

            # FALLBACK: Original processing method if conversational system fails
            logger.info("Using fallback processing method")
            
            # Check for YouTube URL in the query
            youtube_url = extract_youtube_url(text)
            
            # Prepare the request data
            request_data = {
                'text': text
            }
            
            # If YouTube URL is found, add it to the request
            if youtube_url:
                # Remove the URL from the query text
                query_text = text.replace(youtube_url, '').strip()
                if not query_text:
                    send_message(channel_id, "Please provide a question about the video content.", root_id=root_id)
                    return jsonify({'response_type': 'in_channel'})
                
                request_data['text'] = query_text
                request_data['video_url'] = youtube_url
            
            logger.info(f"Sending request to RAG pipeline: {json.dumps(request_data)}")
            
            # Forward the query to the RAG pipeline with increased timeout
            response = requests.post(
                'http://localhost:5001/query',
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=120
            )
            
            # Log the raw response
            logger.info(f"Raw response from RAG pipeline: {response.text}")
            
            # Check if the response was successful
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Stop the typing indicator
            stop_typing.set()
            typing_thread.join(timeout=1)
            
            # Extract the answer from the response
            answer = response_data.get('response', 'I apologize, but I was unable to process your query.')
            
            # Extract sources and other metadata
            sources = response_data.get('sources', [])
            web_search_used = response_data.get('web_search_used', False)
            datasource = response_data.get('datasource', 'unknown')
            exercise_detected = response_data.get('exercise_detected', False)
            related_lectures = response_data.get('related_lectures', [])
            
            # Add source information to the answer
            if sources and len(sources) > 0:
                # Filter out any empty or 'web_search' sources for document references
                doc_sources = [source for source in sources if source and source != 'web_search']
                
                if doc_sources:
                    answer += f"\n\n**Sources:** {', '.join(doc_sources)}"
                    
            # Add exercise-specific information
            if exercise_detected:
                answer += f"\n\n**Exercise Solution**"
                if related_lectures:
                    answer += f"\n**Related Lectures:** {', '.join(related_lectures)}"
                    
            # Add datasource information
            if web_search_used:
                answer += f"\n*Information retrieved from web search*"
            elif datasource == "local_documents":
                answer += f"\n*Information from local documents*"
            elif datasource == "exercise_solver":
                answer += f"\n*Comprehensive exercise solution with lecture materials*"
            
            # Send the response back to Mattermost
            send_message(channel_id, answer, root_id=root_id)
            
            return jsonify({'response_type': 'in_channel'})
            
        except requests.Timeout:
            send_message(channel_id, "The query took too long to process. Please try again with a simpler query.", root_id=root_id)
        except requests.RequestException as e:
            logger.error(f"Error making request to RAG pipeline: {e}")
            send_message(channel_id, "Sorry, there was an error processing your query. Please try again later.", root_id=root_id)
        finally:
            # Stop the typing indicator
            stop_typing.set()
            typing_thread.join()
                
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        if channel_id:
            send_message(channel_id, "Sorry, there was an error processing your query.")
        return jsonify({'response_type': 'in_channel'})

@app.route('/test', methods=['GET'])
def test():
    return "Bot is running!"

@app.route('/mattermost', methods=['POST'])
def mattermost_webhook():
    try:
        data = request.get_json()
        logger.info(f"Webhook received. Data keys: {list(data.keys()) if data else 'No data'}") # Log incoming data keys

        if not data:
            logger.error("No JSON data received in webhook request")
            return jsonify({'error': 'No data provided'}), 400

        channel_id = data.get('channel_id', '')
        user_name = data.get('user_name', 'user') # user_name from webhook is just the username string
        webhook_post_id = data.get('post_id') # This is the ID of the current post/reply from the webhook
        
        # CONVERSATIONAL SYSTEM: Attempt to get full post details for thread management
        # This is crucial because the webhook payload for a mention might not directly contain root_id.
        # We need to fetch the post to get its actual root_id (if it's a reply) and user_id (actual ID of poster).
        full_post_details = None
        if webhook_post_id:
            full_post_details = get_post_details(webhook_post_id)

        current_actual_post_data_for_conv = None
        if full_post_details: # If we successfully fetched the post
            current_actual_post_data_for_conv = {
                'id': full_post_details.get('id'),
                'root_id': full_post_details.get('root_id', ''), # Ensure root_id is present, defaults to empty string if not a reply
                'message': full_post_details.get('message'),
                'channel_id': full_post_details.get('channel_id'),
                'user_id': full_post_details.get('user_id') # The actual user ID of the poster
            }
            logger.info(f"📝 Constructed current_actual_post_data_for_conv from fetched post - ID: {current_actual_post_data_for_conv.get('id')}, Root ID: {current_actual_post_data_for_conv.get('root_id')}")
        elif 'text' in data: # Fallback if post fetch fails but we have text (less ideal, might miss context)
            logger.warning(f"Could not fetch full post details for webhook_post_id: {webhook_post_id}. Using webhook data for conv_post_data.")
            current_actual_post_data_for_conv = {
                'id': webhook_post_id, # ID of the current message/reply
                'root_id': data.get('root_id',''), # Attempt to get root_id if webhook provided one (unlikely for mentions)
                'message': data.get('text','').strip(),
                'channel_id': channel_id,
                'user_id': data.get('user_id') # user_id from webhook (might be username string or actual ID based on MM config)
            }
            logger.info(f"📝 Constructed current_actual_post_data_for_conv from webhook data (fallback) - ID: {current_actual_post_data_for_conv.get('id')}, Root ID: {current_actual_post_data_for_conv.get('root_id')}")
        else:
            logger.warning("Could not identify any post_data from webhook payload for conversational processing.")
        
        # Handle mentions (from outgoing webhook)
        if 'text' in data: # data['text'] is the message content
            message_content_from_webhook = data.get('text', '').strip()
            logger.info(f"Received message content from webhook: '{message_content_from_webhook}'")
            
            # Check if this is a mention
            if f"@{BOT_USERNAME}" in message_content_from_webhook:
                parts = message_content_from_webhook.split(None, 2)  # Split into max 3 parts
                logger.info(f"Message parts: {parts}")
                
                if len(parts) > 1:
                    command = parts[1].lower() if len(parts) > 1 else ''
                    logger.info(f"Extracted command: '{command}'")

                    if command == 'help':
                        logger.info("Processing help command")
                        help_message = """:wave: Hi! I'm your knowledge assistant. Ask me anything!

You can interact with me by:

Mention me: @knowledge-agent your question here
Examples:

@knowledge-agent Can you explain the difference between X and Y?

🆕 **Conversational Features:**
- I remember our conversation within each thread
- Reply to continue our discussion in the same context
- Start new posts for fresh conversations"""
                        send_message(channel_id, help_message) # Help doesn't need a root_id typically
                        return jsonify({'response_type': 'in_channel'})

                    if command in ['hi', 'hello', 'hey', 'greetings']:
                        logger.info("Processing greeting command")
                        greeting_message = f"Hello {user_name}! How can I help you today? :smile:"
                        # Greetings usually don't need to be in a thread unless the greeting itself was threaded.
                        # If full_post_details exists and has a root_id, we can thread it.
                        reply_root_id_for_greeting = full_post_details.get('root_id', '') if full_post_details and full_post_details.get('root_id') else None
                        send_message(channel_id, greeting_message, root_id=reply_root_id_for_greeting)
                        return jsonify({'response_type': 'in_channel'})

                    query_text_for_processing = ' '.join(parts[1:]) if len(parts) > 1 else ''
                    logger.info(f"Processing regular query: '{query_text_for_processing}'")

                    # Determine root_id for the fallback path, if needed.
                    # If we fetched full_post_details, use its root_id.
                    root_id_for_fallback = None
                    if full_post_details and full_post_details.get('root_id', '').strip():
                        root_id_for_fallback = full_post_details.get('root_id')
                    
                    logger.info(f"Passing to process_query - root_id_for_fallback: {root_id_for_fallback}, current_actual_post_data_for_conv available: {current_actual_post_data_for_conv is not None}")
                    return process_query(query_text_for_processing, channel_id, user_name, root_id_for_fallback, current_actual_post_data_for_conv)
        
        # Handle slash command (less likely to need complex post fetching for root_id, but pass current_actual_post_data_for_conv if available)
        elif 'command' in data:
            text = data.get('text', '').strip()
            logger.info(f"Processing slash command with text: {text}")
            # Slash commands don't typically have a 'post_id' in the same way mentions do for fetching root_id.
            # They might have a trigger_id. The conversational system will treat it as a new session if root_id is not in current_actual_post_data_for_conv.
            trigger_id_as_root = data.get('trigger_id') # This might or might not be useful as a root_id
            return process_query(text, channel_id, user_name, trigger_id_as_root, current_actual_post_data_for_conv)
            
        return jsonify({'response_type': 'ephemeral'})

    except Exception as e:
        error_message = f"Error processing webhook: {str(e)}"
        logger.error(error_message)
        logger.exception("Full exception details:")
        return jsonify({'text': error_message, 'response_type': 'ephemeral'})

def get_post_details(post_id: str) -> Optional[Dict[str, Any]]:
    """Fetch post details from Mattermost API to get definitive root_id and other info."""
    if not BOT_TOKEN or not MATTERMOST_URL:
        logger.error("BOT_TOKEN or MATTERMOST_URL not configured for get_post_details")
        return None
    try:
        headers = {
            'Authorization': f'Bearer {BOT_TOKEN}',
            'Content-Type': 'application/json'
        }
        response = requests.get(
            f'{MATTERMOST_URL}/api/v4/posts/{post_id}',
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        post_details = response.json()
        logger.info(f"Fetched post details for {post_id}: ID: {post_details.get('id')}, Root ID: {post_details.get('root_id')}, Message: {post_details.get('message')[:50]}...")
        return post_details
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching post details for {post_id}: {e}")
        if hasattr(e.response, 'text'):
            logger.error(f"Fetch post details response text: {e.response.text}")
        return None

def handle_lecture_command(data):
    """Handle /lecture command"""
    try:
        # Extract channel_id and text from the data
        channel_id = data.get('channel_id')
        text = data.get('text', '').strip()
        
        if not text:
            send_message(channel_id, "Please provide a query after the /lecture command.")
            return
            
        logger.info(f"Processing lecture query: {text}")
        
        # Create a stop event for the typing indicator
        stop_typing = Event()
        
        # Start typing indicator in a separate thread
        typing_thread = threading.Thread(
            target=show_typing_continuous,
            args=(channel_id, stop_typing)
        )
        typing_thread.start()
        
        try:
            # Make request to RAG pipeline
            response = requests.post(
                'http://localhost:5001/query',
                json={'text': text},
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract answer and source information
            answer = result.get('response', 'No answer found')
            sources = result.get('sources', [])
            web_search_used = result.get('web_search_used', False)
            datasource = result.get('datasource', 'unknown')
            exercise_detected = result.get('exercise_detected', False)
            related_lectures = result.get('related_lectures', [])
            
            # Add source information to the answer
            if sources and len(sources) > 0:
                # Filter out any empty or 'web_search' sources for document references
                doc_sources = [source for source in sources if source and source != 'web_search']
                
                if doc_sources:
                    answer += f"\n\n**Sources:** {', '.join(doc_sources)}"
                    
            # Add exercise-specific information
            if exercise_detected:
                answer += f"\n\n**Exercise Solution**"
                if related_lectures:
                    answer += f"\n**Related Lectures:** {', '.join(related_lectures)}"
                    
            # Add datasource information
            if web_search_used:
                answer += f"\n*Information retrieved from web search*"
            elif datasource == "local_documents":
                answer += f"\n*Information from local documents*"
            elif datasource == "exercise_solver":
                answer += f"\n*Comprehensive exercise solution with lecture materials*"
            
            send_message(channel_id, answer)
            
        except requests.Timeout:
            send_message(channel_id, "The query took too long to process. Please try again with a simpler query.")
        except requests.RequestException as e:
            logger.error(f"Error making request to RAG pipeline: {e}")
            send_message(channel_id, "Sorry, there was an error processing your query. Please try again later.")
        finally:
            # Stop the typing indicator
            stop_typing.set()
            typing_thread.join()
            
    except Exception as e:
        logger.error(f"Error handling lecture command: {e}")
        if channel_id:
            send_message(channel_id, "Sorry, there was an error processing your command.")

# CONVERSATIONAL SYSTEM: Add endpoint to check conversation stats
@app.route('/conversation-stats', methods=['GET'])
def conversation_stats():
    """Get statistics about active conversations"""
    try:
        if conversational_bot:
            stats = conversational_bot.get_conversation_stats()
            return jsonify(stats)
        else:
            return jsonify({"error": "Conversational system not available"}), 503
    except Exception as e:
        logger.error(f"Error getting conversation stats: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Verify environment variables
    required_vars = ['MATTERMOST_URL', 'BOT_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    logger.info("🤖 Mattermost Knowledge Agent Starting...")
    
    # CONVERSATIONAL SYSTEM: Enhanced startup messages
    if conversational_bot:
        logger.info("✅ Conversational features enabled - Thread-based sessions active")
        logger.info("💬 Features: Memory across threads, context awareness, session management")
    else:
        logger.info("⚠️ Conversational features disabled - Using fallback mode")
    
    logger.info("📚 Bot functionality ready")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5002) 