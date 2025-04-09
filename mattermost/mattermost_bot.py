import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import logging
import threading
import time
from threading import Event

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Mattermost configuration
MATTERMOST_URL = os.getenv('MATTERMOST_URL')
BOT_TOKEN = os.getenv('BOT_TOKEN')
BOT_USERNAME = os.getenv('BOT_USERNAME')
BOT_ICON_URL = os.getenv('BOT_ICON_URL', 'https://example.com/bot-icon.png')

HEADERS = {
    'Authorization': f'Bearer {BOT_TOKEN}',
    'Content-Type': 'application/json'
}

def send_message(channel_id, message, response_type='in_channel', root_id=None):
    """Send a message to a Mattermost channel
    
    Args:
        channel_id: The ID of the channel to send the message to
        message: The message content
        response_type: The type of response ('in_channel' or 'ephemeral')
        root_id: The ID of the root post to reply to (for creating/continuing a thread)
    """
    headers = {
        'Authorization': f'Bearer {BOT_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'channel_id': channel_id,
        'message': message,
        'props': {
            'from_webhook': 'true',
            'override_username': BOT_USERNAME,
            'override_icon_url': BOT_ICON_URL
        }
    }
    
    # Add root_id to payload if provided to create/continue a thread
    if root_id:
        logger.info(f"Creating thread reply - root_id: {root_id}")
        payload['root_id'] = root_id
        logger.info(f"Full payload for thread reply: {json.dumps(payload, indent=2)}")
    else:
        logger.info("Creating new message (no thread)")
    
    try:
        logger.info(f"Sending message to channel {channel_id}")
        response = requests.post(
            f'{MATTERMOST_URL}/api/v4/posts',
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        # Log the response for debugging
        response_data = response.json()
        logger.info(f"Mattermost API response: {json.dumps(response_data, indent=2)}")
        
        if 'id' in response_data:
            if root_id:
                logger.info(f"Successfully created thread reply - Message ID: {response_data['id']}, Thread ID: {root_id}")
            else:
                logger.info(f"Successfully created new message - ID: {response_data['id']}")
        
        return response_data
    except Exception as e:
        logger.error(f"Error sending message to Mattermost: {str(e)}")
        raise

def show_typing(channel_id):
    """Show typing indicator in a Mattermost channel"""
    headers = {
        'Authorization': f'Bearer {BOT_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(
            f'{MATTERMOST_URL}/api/v4/users/me/typing',
            headers=headers,
            json={'channel_id': channel_id}
        )
        response.raise_for_status()
        return response.json() if response.content else None
    except Exception as e:
        logger.error(f"Error showing typing indicator: {str(e)}")
        return None

def show_typing_continuous(channel_id, stop_event):
    """Show typing indicator continuously until stop_event is set"""
    while not stop_event.is_set():
        try:
            response = requests.post(
                f'{MATTERMOST_URL}/api/v4/users/me/typing',
                headers=HEADERS,
                json={'channel_id': channel_id}
            )
            response.raise_for_status()
            time.sleep(5)  # Show typing indicator every 5 seconds
        except Exception as e:
            logger.error(f"Error showing typing indicator: {e}")
            break

def process_query(text, channel_id, user_name, root_id=None):
    """Process a query and return the response
    
    Args:
        text: The query text
        channel_id: The ID of the channel
        user_name: The name of the user who sent the query
        root_id: The ID of the message to reply to (for creating a thread)
    """
    if not text or text.lower() == 'help':
        help_message = (
            "👋 Hi! I'm your knowledge assistant. Ask me anything!\n\n"
            "You can interact with me in by:\n"
            "- Mention me: `@knowledge-agent your question here`\n\n"
            "**Examples**:\n"
            "- `@knowledge-agent Can you explain the difference between X and Y?`\n"
        )
        # For help messages, always send as a main message (no thread)
        logger.info("Sending help message as main message")
        send_message(channel_id, help_message)
        return jsonify({'response_type': 'in_channel'})

    try:
        # Create a stop event for the typing indicator thread
        stop_typing = threading.Event()
        
        # Start the typing indicator in a separate thread
        typing_thread = threading.Thread(
            target=show_typing_continuous,
            args=(channel_id, stop_typing)
        )
        typing_thread.daemon = True
        typing_thread.start()

        try:
            # Forward the query to the RAG pipeline with increased timeout
            response = requests.post(
                'http://localhost:5001/query',
                json={'text': text},
                timeout=120  # Increase timeout to 120 seconds
            )
            response.raise_for_status()
            response_data = response.json()

            # Get the response text
            answer = response_data.get('text', 'No answer available')

            # Stop the typing indicator
            stop_typing.set()
            typing_thread.join(timeout=1)

            # Send as a reply if root_id is provided, otherwise as a new message
            logger.info(f"Sending response with root_id: {root_id}")
            response = send_message(channel_id, answer, root_id=root_id)

            # Return the response with root_id if available
            return jsonify({
                'response_type': 'in_channel',
                'root_id': root_id
            })

        finally:
            # Ensure we stop the typing indicator even if an error occurs
            stop_typing.set()
            typing_thread.join(timeout=1)

    except requests.exceptions.RequestException as e:
        error_message = f"Error processing query: {str(e)}"
        logger.error(error_message)
        # Send error message as a reply if root_id is provided
        send_message(channel_id, error_message, root_id=root_id)
        return jsonify({
            'text': error_message,
            'response_type': 'ephemeral',
            'root_id': root_id
        })

@app.route('/mattermost', methods=['POST'])
def mattermost_webhook():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Log the incoming webhook data for debugging
        logger.info(f"Received webhook data: {json.dumps(data, indent=2)}")

        channel_id = data.get('channel_id', '')
        user_name = data.get('user_name', 'user')
        
        # Extract post data and ID
        post_id = None
        if 'post' in data:
            try:
                if isinstance(data['post'], str):
                    post_data = json.loads(data['post'])
                else:
                    post_data = data['post']
                
                # Get the ID of the message we want to reply to
                post_id = post_data.get('id')
                logger.info(f"Extracted post_id from post data: {post_id}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing post data: {e}")
        
        # If post_id is still None, try to get it from the trigger_id or other fields
        if post_id is None:
            post_id = data.get('trigger_id') or data.get('post_id')
            logger.info(f"Attempting to get post_id from trigger_id or post_id: {post_id}")
        
        # Handle mentions (from outgoing webhook)
        if 'text' in data:
            message = data.get('text', '').strip()
            trigger_word = f"@{BOT_USERNAME}"
            
            if trigger_word in message:
                # Remove the mention and get the actual query
                query = message.replace(trigger_word, '').strip()
                logger.info(f"Processing query: {query}")
                logger.info(f"Message ID for reply: {post_id}")
                
                # Process the query and create a reply
                return process_query(query, channel_id, user_name, post_id)
        
        # Handle slash command
        elif 'command' in data:
            text = data.get('text', '').strip()
            return process_query(text, channel_id, user_name, post_id)
            
        return jsonify({'response_type': 'ephemeral'})

    except Exception as e:
        error_message = f"Error processing webhook: {str(e)}"
        logger.error(error_message)
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
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5002) 