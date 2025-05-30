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

# Initialize Flask app
app = Flask(__name__)

# Thread-aware conversation storage
# Structure: {thread_id: conversation_data}
thread_conversations = {}

class ThreadManager:
    """Manages conversation threads and integrates with the conversational agent"""
    
    def __init__(self):
        self.threads = {}
    
    def get_or_create_thread(self, thread_id: str, channel_id: str, user_name: str) -> Dict[str, Any]:
        """Get existing thread conversation or create a new one"""
        if thread_id not in self.threads:
            self.threads[thread_id] = {
                'thread_id': thread_id,
                'channel_id': channel_id,
                'user_name': user_name,
                'created_at': time.time(),
                'last_activity': time.time(),
                'message_count': 0,
                'conversation_active': True
            }
            logger.info(f"Created new thread conversation: {thread_id}")
        else:
            # Update last activity
            self.threads[thread_id]['last_activity'] = time.time()
            self.threads[thread_id]['message_count'] += 1
        
        return self.threads[thread_id]
    
    def is_thread_conversation(self, thread_id: str) -> bool:
        """Check if this is part of an ongoing thread conversation"""
        return thread_id in self.threads and self.threads[thread_id]['conversation_active']
    
    def cleanup_old_threads(self, max_age_hours: int = 24):
        """Remove old inactive threads"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        threads_to_remove = []
        for thread_id, thread_data in self.threads.items():
            if current_time - thread_data['last_activity'] > max_age_seconds:
                threads_to_remove.append(thread_id)
        
        for thread_id in threads_to_remove:
            del self.threads[thread_id]
            logger.info(f"Cleaned up old thread: {thread_id}")

# Initialize thread manager
thread_manager = ThreadManager()

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
        data = {
            'channel_id': channel_id,
            'message': message,
            'root_id': root_id if root_id else None
        }
    
        # Remove root_id if it's None to avoid API errors
        if data['root_id'] is None:
            del data['root_id']
        
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

def get_post_details(post_id: str) -> Optional[Dict[str, Any]]:
    """Get post details from Mattermost API to determine thread information"""
    try:
        headers = {
            'Authorization': f'Bearer {BOT_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(
            f'{MATTERMOST_URL}/api/v4/posts/{post_id}',
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get post details for {post_id}: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting post details: {e}")
        return None

def extract_thread_info(data: Dict[str, Any]) -> tuple[Optional[str], bool]:
    """
    Extract thread information from webhook data
    Returns: (thread_root_id, is_thread_reply)
    """
    thread_root_id = None
    is_thread_reply = False
    
    # Method 1: Check if there's a root_id in the webhook data
    if 'root_id' in data and data['root_id']:
        thread_root_id = data['root_id']
        is_thread_reply = True
        logger.info(f"Thread reply detected via root_id: {thread_root_id}")
        return thread_root_id, is_thread_reply
    
    # Method 2: Check for post data and extract root_id
    if 'post' in data:
        try:
            if isinstance(data['post'], str):
                post_data = json.loads(data['post'])
            else:
                post_data = data['post']
            
            if 'root_id' in post_data and post_data['root_id']:
                thread_root_id = post_data['root_id']
                is_thread_reply = True
                logger.info(f"Thread reply detected via post.root_id: {thread_root_id}")
            elif 'id' in post_data:
                # This could be the start of a new thread
                thread_root_id = post_data['id']
                is_thread_reply = False
                logger.info(f"Potential new thread detected: {thread_root_id}")
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing post data for thread info: {e}")
    
    # Method 3: Try to get post details from API if we have a post_id
    if not thread_root_id:
        post_id = data.get('post_id') or data.get('trigger_id')
        if post_id:
            post_details = get_post_details(post_id)
            if post_details:
                if post_details.get('root_id'):
                    thread_root_id = post_details['root_id']
                    is_thread_reply = True
                    logger.info(f"Thread reply detected via API: {thread_root_id}")
                else:
                    thread_root_id = post_details.get('id', post_id)
                    is_thread_reply = False
    
    return thread_root_id, is_thread_reply

def process_query(text, channel_id, user_name, root_id=None, webhook_data=None):
    """Process a query and return the response with thread support"""
    try:
        if not text:
            send_message(channel_id, "Please provide a query.", root_id=root_id)
            return jsonify({'response_type': 'in_channel'})
        
        # Extract thread information
        thread_root_id, is_thread_reply = extract_thread_info(webhook_data or {})
        
        # Use provided root_id if available, otherwise use extracted thread_root_id
        effective_root_id = root_id or thread_root_id
        
        # Determine if this is a conversation thread
        use_conversational_agent = False
        thread_id = None
        
        if effective_root_id:
            thread_id = effective_root_id
            # Check if this is part of an ongoing conversation
            if thread_manager.is_thread_conversation(thread_id) or is_thread_reply:
                use_conversational_agent = True
                # Update thread manager
                thread_manager.get_or_create_thread(thread_id, channel_id, user_name)
                logger.info(f"Using conversational agent for thread: {thread_id}")
        
        # Create a stop event for the typing indicator
        stop_typing = Event()
        
        # Start typing indicator in a separate thread
        typing_thread = threading.Thread(
            target=show_typing_continuous,
            args=(channel_id, stop_typing)
        )
        typing_thread.start()

        try:
            # Check for YouTube URL in the query
            youtube_url = extract_youtube_url(text)
            
            # Prepare the request data
            request_data = {
                'text': text,
                'user': user_name,
                'channel_id': channel_id,
                'use_conversational_agent': use_conversational_agent,
                'thread_id': thread_id,
                'is_thread_reply': is_thread_reply
            }
            
            # If YouTube URL is found, add it to the request
            if youtube_url:
                # Remove the URL from the query text
                query_text = text.replace(youtube_url, '').strip()
                if not query_text:
                    send_message(channel_id, "Please provide a question about the video content.", root_id=effective_root_id)
                    return jsonify({'response_type': 'in_channel'})
                
                request_data['text'] = query_text
                request_data['video_url'] = youtube_url
            
            logger.info(f"Sending request to RAG pipeline: {json.dumps(request_data, indent=2)}")
            
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
            
            # Add thread context indicator if this is a follow-up
            if use_conversational_agent and is_thread_reply:
                thread_info = thread_manager.threads.get(thread_id, {})
                message_count = thread_info.get('message_count', 0)
                if message_count > 1:
                    answer = f"🧵 *Thread follow-up #{message_count}*\n\n{answer}"
            
            # Send the response back to Mattermost
            sent_message = send_message(channel_id, answer, root_id=effective_root_id)
            
            # If this was the start of a new thread and we successfully sent a message, 
            # mark it as a thread conversation
            if sent_message and not is_thread_reply and effective_root_id:
                thread_manager.get_or_create_thread(effective_root_id, channel_id, user_name)
                logger.info(f"Started new thread conversation: {effective_root_id}")
            
            return jsonify({'response_type': 'in_channel'})
            
        except requests.Timeout:
            send_message(channel_id, "The query took too long to process. Please try again with a simpler query.", root_id=effective_root_id)
        except requests.RequestException as e:
            logger.error(f"Error making request to RAG pipeline: {e}")
            send_message(channel_id, "Sorry, there was an error processing your query. Please try again later.", root_id=effective_root_id)
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

@app.route('/thread-stats', methods=['GET'])
def thread_stats():
    """Get statistics about active threads"""
    stats = {
        'active_threads': len(thread_manager.threads),
        'threads': []
    }
    
    for thread_id, thread_data in thread_manager.threads.items():
        stats['threads'].append({
            'thread_id': thread_id,
            'message_count': thread_data['message_count'],
            'last_activity': thread_data['last_activity'],
            'user_name': thread_data['user_name']
        })
    
    return jsonify(stats)

def cleanup_threads_periodically():
    """Cleanup old threads periodically"""
    import threading
    
    def cleanup():
        while True:
            time.sleep(3600)  # Run every hour
            thread_manager.cleanup_old_threads(max_age_hours=24)
    
    cleanup_thread = threading.Thread(target=cleanup, daemon=True)
    cleanup_thread.start()
    logger.info("Started periodic thread cleanup")

@app.route('/mattermost', methods=['POST'])
def mattermost_webhook():
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in webhook request")
            return jsonify({'error': 'No data provided'}), 400

        channel_id = data.get('channel_id', '')
        user_name = data.get('user_name', 'user')
        
        # Log the full webhook data for debugging
        logger.debug(f"Full webhook data: {json.dumps(data, indent=2)}")
        
        # Handle mentions (from outgoing webhook)
        if 'text' in data:
            message = data.get('text', '').strip()
            logger.info(f"Received message: '{message}'")
            
            # Check if this is a mention
            if f"@{BOT_USERNAME}" in message:
                # Get the actual command/query (everything after the mention)
                parts = message.split(None, 2)  # Split into max 3 parts
                logger.info(f"Message parts: {parts}")
                
                if len(parts) > 1:  # We have a command after the mention
                    # The command will be the second part (after @username)
                    command = parts[1].lower() if len(parts) > 1 else ''
                    logger.info(f"Extracted command: '{command}'")

                    # Handle help command
                    if command == 'help':
                        logger.info("Processing help command")
                        help_message = """:wave: Hi! I'm your knowledge assistant. Ask me anything!

You can interact with me by:

Mention me: @knowledge-agent your question here
Examples:

@knowledge-agent Can you explain the difference between X and Y?

💡 **Thread Support**: Ask follow-up questions by replying to my responses in a thread!
🧵 I'll remember our conversation context within threads for better continuity."""
                        send_message(channel_id, help_message)
                        return jsonify({'response_type': 'in_channel'})

                    # Handle greetings
                    if command in ['hi', 'hello', 'hey', 'greetings']:
                        logger.info("Processing greeting command")
                        greeting_message = f"Hello {user_name}! How can I help you today? :smile:\n\n💡 Tip: You can ask follow-up questions by replying to my responses in a thread!"
                        send_message(channel_id, greeting_message)
                        return jsonify({'response_type': 'in_channel'})

                    # For other queries, get the full text after the command
                    query = ' '.join(parts[1:]) if len(parts) > 1 else ''
                    logger.info(f"Processing regular query: '{query}'")

                    # Get post_id for threading regular responses
                    post_id = None
                    if 'post' in data:
                        try:
                            if isinstance(data['post'], str):
                                post_data = json.loads(data['post'])
                            else:
                                post_data = data['post']
                            post_id = post_data.get('id')
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing post data: {e}")
                    
                    if post_id is None:
                        post_id = data.get('trigger_id') or data.get('post_id')

                    # Process as regular query with full webhook data
                    return process_query(query, channel_id, user_name, post_id, data)
        
        # Handle slash command
        elif 'command' in data:
            text = data.get('text', '').strip()
            logger.info(f"Processing slash command with text: {text}")
            post_id = data.get('trigger_id') or data.get('post_id')
            return process_query(text, channel_id, user_name, post_id, data)
            
        return jsonify({'response_type': 'ephemeral'})

    except Exception as e:
        error_message = f"Error processing webhook: {str(e)}"
        logger.error(error_message)
        logger.exception("Full exception details:")
        return jsonify({'text': error_message, 'response_type': 'ephemeral'})

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
            
            # Format and send the response
            answer = result.get('answer', 'No answer found')
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

if __name__ == '__main__':
    # Verify environment variables
    required_vars = ['MATTERMOST_URL', 'BOT_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Start periodic thread cleanup
    cleanup_threads_periodically()
    
    logger.info("🤖 Mattermost Knowledge Agent Starting...")
    logger.info("🧵 Thread-aware conversation support enabled")
    logger.info("💬 Conversational agent integration ready")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5002) 