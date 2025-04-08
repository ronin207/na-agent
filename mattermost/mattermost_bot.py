import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import logging

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

def send_message(channel_id, message, response_type='in_channel'):
    """Send a message to a Mattermost channel"""
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
    
    try:
        response = requests.post(
            f'{MATTERMOST_URL}/api/v4/posts',
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error sending message to Mattermost: {str(e)}")
        raise

def process_query(text, channel_id, user_name):
    """Process a query and return the response"""
    if not text or text.lower() == 'help':
        help_message = (
            "👋 Hi! I'm your knowledge assistant. Ask me anything!\n\n"
            "You can interact with me in by:\n"
            "- Mention me: `@knowledge-agent your question here`\n\n"
            "**Examples**:\n"
            "- `@knowledge-agent Can you explain the difference between X and Y?`\n"
        )
        send_message(channel_id, help_message)
        return jsonify({'response_type': 'in_channel'})

    # Send initial response
    send_message(channel_id, "Processing your query...", response_type='ephemeral')

    try:
        # Forward the query to the RAG pipeline
        response = requests.post(
            'http://localhost:5001/query',
            json={'text': text},
            timeout=30
        )
        response.raise_for_status()
        response_data = response.json()

        # Format the message
        formatted_message = (
            f"**Question from @{user_name}**:\n"
            f"{text}\n\n"
            f"**Answer**:\n"
            f"{response_data['text']}"
        )

        # Send the formatted message
        send_message(channel_id, formatted_message)
        return jsonify({'response_type': 'in_channel'})

    except requests.exceptions.RequestException as e:
        error_message = f"Error processing query: {str(e)}"
        logger.error(error_message)
        return jsonify({'text': error_message, 'response_type': 'ephemeral'})

@app.route('/mattermost', methods=['POST'])
def mattermost_webhook():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        channel_id = data.get('channel_id', '')
        user_name = data.get('user_name', 'user')
        
        # Handle slash command
        if 'command' in data:
            text = data.get('text', '').strip()
            return process_query(text, channel_id, user_name)
        
        # Handle mentions (from outgoing webhook)
        elif 'text' in data:
            message = data.get('text', '').strip()
            trigger_word = f"@{BOT_USERNAME}"
            
            # Log the incoming message for debugging
            logger.info(f"Received message: {message}")
            logger.info(f"Looking for trigger word: {trigger_word}")
            
            if trigger_word in message:
                # Remove the mention and get the actual query
                query = message.replace(trigger_word, '').strip()
                logger.info(f"Extracted query: {query}")
                return process_query(query, channel_id, user_name)
            
        return jsonify({'response_type': 'ephemeral'})

    except Exception as e:
        error_message = f"Error processing webhook: {str(e)}"
        logger.error(error_message)
        return jsonify({'text': error_message, 'response_type': 'ephemeral'})

if __name__ == '__main__':
    # Verify environment variables
    required_vars = ['MATTERMOST_URL', 'BOT_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5002) 