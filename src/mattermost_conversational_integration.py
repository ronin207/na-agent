"""
Mattermost Integration Example for Conversational Agentic Retrieval

This module demonstrates how to integrate the conversational RAG system
with Mattermost bot, managing conversation sessions per thread.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from conversation import ConversationalAgenticRetrieval

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MattermostConversationalBot:
    """
    Mattermost bot with conversational awareness using thread-based sessions.
    """
    
    def __init__(self, rag_system: ConversationalAgenticRetrieval):
        self.rag = rag_system
        self.active_threads = {}  # Track active conversation threads
        
    def get_session_id_from_thread(self, post_data: Dict[str, Any]) -> str:
        """
        Generate session ID based on Mattermost thread structure.
        
        For Mattermost:
        - If it's a root post (new question in channel): use post_id as session_id
        - If it's a reply in thread: use root_id as session_id
        """
        # If this is a reply in a thread, use the root post ID
        if post_data.get('root_id'):
            session_id = f"thread_{post_data['root_id']}"
        else:
            # This is a new root post, start new session
            session_id = f"thread_{post_data['id']}"
            
        return session_id
    
    def is_new_conversation(self, post_data: Dict[str, Any]) -> bool:
        """
        Determine if this is the start of a new conversation.
        
        In Mattermost:
        - New conversation = root post (no root_id)
        - Continue conversation = reply (has root_id)
        """
        return not post_data.get('root_id')
    
    async def process_message(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a Mattermost message with conversational awareness.
        
        Args:
            post_data: Mattermost post data containing:
                - id: post ID
                - root_id: root post ID (if reply)
                - message: user message
                - channel_id: channel ID
                - user_id: user ID
                
        Returns:
            Response data for the bot
        """
        try:
            # Extract message and metadata
            user_message = post_data.get('message', '').strip()
            channel_id = post_data.get('channel_id')
            user_id = post_data.get('user_id')
            
            if not user_message:
                return {
                    "error": "Empty message",
                    "should_respond": False
                }
            
            # Determine session ID based on thread structure
            session_id = self.get_session_id_from_thread(post_data)
            is_new_conversation = self.is_new_conversation(post_data)
            
            # Log conversation flow
            if is_new_conversation:
                logger.info(f"New conversation started: {session_id} in channel {channel_id}")
            else:
                logger.info(f"Continuing conversation: {session_id} in channel {channel_id}")
            
            # Process the question with RAG system
            result = await self.rag.ainvoke(user_message, session_id)
            
            # Prepare response for Mattermost
            response_data = {
                "message": result["response"],
                "session_id": session_id,
                "is_new_conversation": is_new_conversation,
                "conversation_turns": result.get("conversation_turns", 0),
                "datasource": result.get("datasource"),
                "sources": result.get("sources", []),
                "should_respond": True,
                "thread_metadata": {
                    "root_id": post_data.get('root_id') or post_data.get('id'),  # Use post_id if no root_id
                    "channel_id": channel_id,
                    "original_post_id": post_data.get('id')
                }
            }
            
            # Add helpful metadata for debugging
            response_data["debug_info"] = {
                "web_search_used": result.get("web_search_used", False),
                "exercise_detected": result.get("exercise_detected", False),
                "query_analysis": result.get("query_analysis", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            # Track active threads
            self.active_threads[session_id] = {
                "last_activity": datetime.now(),
                "channel_id": channel_id,
                "turn_count": result.get("conversation_turns", 0)
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing Mattermost message: {e}")
            return {
                "error": str(e),
                "should_respond": True,
                "message": "Sorry, I encountered an error processing your request. Please try again.",
                "session_id": session_id if 'session_id' in locals() else "unknown"
            }
    
    def format_response_for_mattermost(self, response_data: Dict[str, Any]) -> str:
        """
        Format the response for Mattermost with appropriate metadata.
        """
        if not response_data.get("should_respond", True):
            return ""
        
        message = response_data["message"]
        
        # Add conversation context info for new conversations
        if response_data.get("is_new_conversation", False):
            footer = "\n\n---\n" # *New conversation started. I'll remember our discussion in this thread.*
        else:
            turns = response_data.get("conversation_turns", 0)
            footer = f"\n\n---\n" # *Conversation turn {turns} • I remember our previous discussion*
        
        # Add source information
        sources = response_data.get("sources", [])
        if sources and sources != ["web_search"]:
            source_text = ", ".join(sources[:3])  # Limit to first 3 sources
            if len(sources) > 3:
                source_text += f" (+{len(sources)-3} more)"
            footer += f"\n*Sources: {source_text}*"
        elif response_data.get("debug_info", {}).get("web_search_used", False):
            footer += "\n*Web search used*"
        
        return message + footer
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about active conversations."""
        return {
            "active_threads": len(self.active_threads),
            "total_sessions": len(self.rag.conversation_manager.sessions),
            "thread_details": self.active_threads
        }
    
    def cleanup_old_sessions(self, hours: int = 24):
        """Clean up old conversation sessions."""
        self.rag.conversation_manager.cleanup_expired_sessions()
        
        # Also clean up local thread tracking
        cutoff_time = datetime.now() - timedelta(hours=hours)
        old_threads = [
            thread_id for thread_id, info in self.active_threads.items()
            if info["last_activity"] < cutoff_time
        ]
        
        for thread_id in old_threads:
            del self.active_threads[thread_id]
            logger.info(f"Cleaned up old thread tracking: {thread_id}")

# Example usage for FastAPI/Flask integration
class MattermostWebhookHandler:
    """
    Example webhook handler for Mattermost integration.
    """
    
    def __init__(self):
        # Initialize the conversational RAG system
        self.rag_system = ConversationalAgenticRetrieval(
            pdf_folder="./data/",
            persist_directory="./chroma_db"
        )
        
        # Initialize the Mattermost bot
        self.bot = MattermostConversationalBot(self.rag_system)
    
    async def handle_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming Mattermost webhook.
        
        This would be called by your FastAPI/Flask endpoint.
        """
        try:
            # Extract post data from webhook
            post_data = webhook_data.get('post', {})
            
            # Skip if this is the bot's own message
            if webhook_data.get('user_name') == 'your-bot-username':
                return {"should_respond": False}
            
            # Process the message
            response_data = await self.bot.process_message(post_data)
            
            if not response_data.get("should_respond", True):
                return {"should_respond": False}
            
            # Format response for Mattermost
            formatted_message = self.bot.format_response_for_mattermost(response_data)
            
            # Prepare Mattermost API response
            mattermost_response = {
                "text": formatted_message,
                "response_type": "in_channel",
                "username": "NA Assistant",
                "icon_emoji": ":robot_face:"
            }
            
            # Add threading info if this is a reply
            thread_metadata = response_data.get("thread_metadata", {})
            if thread_metadata.get("root_id"):
                mattermost_response["props"] = {
                    "root_id": thread_metadata["root_id"]
                }
            
            return {
                "should_respond": True,
                "mattermost_response": mattermost_response,
                "session_id": response_data.get("session_id"),
                "debug_info": response_data.get("debug_info", {})
            }
            
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return {
                "should_respond": True,
                "mattermost_response": {
                    "text": "Sorry, I encountered an error. Please try again.",
                    "response_type": "in_channel",
                    "username": "NA Assistant",
                    "icon_emoji": ":warning:"
                }
            }

# Example FastAPI integration
"""
from fastapi import FastAPI, Request
from mattermost_conversational_integration import MattermostWebhookHandler

app = FastAPI()
webhook_handler = MattermostWebhookHandler()

@app.post("/webhook/mattermost")
async def mattermost_webhook(request: Request):
    webhook_data = await request.json()
    result = await webhook_handler.handle_webhook(webhook_data)
    
    if result.get("should_respond", False):
        return result["mattermost_response"]
    else:
        return {"text": ""}  # Empty response to avoid duplicate messages

@app.get("/stats")
async def get_stats():
    return webhook_handler.bot.get_conversation_stats()
"""

if __name__ == "__main__":
    # Test the conversational bot locally
    async def test_conversation():
        """Test the conversational capabilities locally."""
        print("🧪 Testing Conversational Mattermost Bot")
        print("=" * 50)
        
        # Initialize system
        rag_system = ConversationalAgenticRetrieval(
            pdf_folder="./data/",
            persist_directory="./chroma_db"
        )
        bot = MattermostConversationalBot(rag_system)
        
        # Simulate conversation thread
        test_posts = [
            {
                "id": "post_001",
                "message": "What is the condition number?",
                "channel_id": "channel_123",
                "user_id": "user_456"
            },
            {
                "id": "post_002", 
                "root_id": "post_001",  # Reply in thread
                "message": "Can you give me an example?",
                "channel_id": "channel_123", 
                "user_id": "user_456"
            },
            {
                "id": "post_003",
                "root_id": "post_001",  # Another reply in same thread
                "message": "How do I calculate it in Python?",
                "channel_id": "channel_123",
                "user_id": "user_456"
            }
        ]
        
        for i, post in enumerate(test_posts, 1):
            print(f"\n--- Test Message {i} ---")
            print(f"Message: {post['message']}")
            print(f"Is reply: {'Yes' if post.get('root_id') else 'No'}")
            
            result = await bot.process_message(post)
            
            if result.get("should_respond", True):
                formatted_response = bot.format_response_for_mattermost(result)
                print(f"Response: {formatted_response[:200]}...")
                print(f"Session: {result['session_id']}")
                print(f"Turns: {result.get('conversation_turns', 0)}")
            else:
                print("No response needed")
        
        # Show conversation stats
        print(f"\n--- Stats ---")
        stats = bot.get_conversation_stats()
        print(f"Active threads: {stats['active_threads']}")
        print(f"Total sessions: {stats['total_sessions']}")
    
    # Run test
    asyncio.run(test_conversation()) 