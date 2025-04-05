"""
Conversational Agent for RAG Pipeline

This module implements a ConversationalAgent class that enhances a RAG pipeline
with conversation history management and follow-up question handling capabilities.
"""

import datetime
from typing import List, Dict, Any, Optional, Union


class ConversationalAgent:
    """
    A conversational agent that wraps a RAG pipeline to maintain conversation history
    and handle follow-up questions by rewriting queries with context.
    """

    def __init__(
        self,
        rag_pipeline: Any,
        max_history_length: int = 5,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the ConversationalAgent.

        Args:
            rag_pipeline: The underlying RAG pipeline instance
            max_history_length: Maximum number of conversation exchanges to keep in history
            verbose: Whether to print verbose output
            **kwargs: Additional configuration parameters
        """
        self.rag_pipeline = rag_pipeline
        self.max_history_length = max_history_length
        self.verbose = verbose
        self.config = kwargs
        
        # Initialize conversation history as a list of dictionaries
        # Each dictionary contains query, answer, sources, and timestamp
        self.conversation_history: List[Dict[str, Any]] = []

    def add_to_history(
        self, 
        query: str, 
        answer: str, 
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add a new exchange to the conversation history and maintain the maximum history length.

        Args:
            query: The user's query
            answer: The system's answer
            sources: Optional list of sources used to generate the answer
        """
        # Create a new history entry
        entry = {
            "query": query,
            "answer": answer,
            "sources": sources if sources else [],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add to history
        self.conversation_history.append(entry)
        
        # Maintain maximum history length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
            
        if self.verbose:
            print(f"Added to conversation history. Current length: {len(self.conversation_history)}")

    def get_conversation_context(self) -> str:
        """
        Format the conversation history as a context string for inclusion in new queries.

        Returns:
            A formatted string containing the conversation context
        """
        if not self.conversation_history:
            return ""
            
        context_parts = []
        for i, entry in enumerate(self.conversation_history):
            context_parts.append(f"User: {entry['query']}")
            context_parts.append(f"Assistant: {entry['answer']}")
            
        return "\n".join(context_parts)

    def _detect_followup_question(self, query: str) -> bool:
        """
        Detect if a query is likely a follow-up question based on linguistic indicators.

        Args:
            query: The user's query

        Returns:
            True if the query appears to be a follow-up question, False otherwise
        """
        # List of common follow-up indicators
        followup_indicators = [
            "what about", "how about", "and", "also", "additionally",
            "what else", "tell me more", "can you elaborate", "furthermore",
            "moreover", "in addition", "besides", "apart from that"
        ]
        
        # List of pronouns that might indicate reference to previous context
        context_dependent_pronouns = [
            "it", "this", "that", "these", "those", "they", "them", "their",
            "he", "she", "his", "her", "its"
        ]
        
        # Check if query contains any follow-up indicators
        query_lower = query.lower()
        for indicator in followup_indicators:
            if indicator in query_lower:
                return True
                
        # Check if query starts with a pronoun
        first_word = query_lower.split()[0] if query_lower.split() else ""
        if first_word in context_dependent_pronouns:
            return True
            
        # If there's no conversation history, it can't be a follow-up
        if not self.conversation_history:
            return False
            
        return False

    def _rewrite_with_context(self, query: str, conversation_context: str) -> str:
        """
        Use the underlying language model to rewrite the query into a standalone one
        that includes conversation context.

        Args:
            query: The user's query
            conversation_context: The formatted conversation context

        Returns:
            A rewritten query that incorporates the conversation context
        """
        # If there's no conversation context, return the original query
        if not conversation_context:
            return query
            
        # Construct a prompt for the language model to rewrite the query
        rewrite_prompt = f"""
        Given the following conversation history and a follow-up question, 
        rewrite the follow-up question to be a standalone question that includes all relevant context.
        
        Conversation history:
        {conversation_context}
        
        Follow-up question: {query}
        
        Standalone question:
        """
        
        # Use the language model from the RAG pipeline to rewrite the query
        # This assumes the RAG pipeline has a language model with a generate method
        try:
            # Try to access the language model from the RAG pipeline
            if hasattr(self.rag_pipeline, "llm"):
                rewritten_query = self.rag_pipeline.llm.generate(rewrite_prompt).strip()
            else:
                # Fallback if we can't access the language model directly
                # This is a simple heuristic approach
                last_query = self.conversation_history[-1]["query"] if self.conversation_history else ""
                rewritten_query = f"Regarding {last_query}, {query}"
                
            if self.verbose:
                print(f"Original query: {query}")
                print(f"Rewritten query: {rewritten_query}")
                
            return rewritten_query
        except Exception as e:
            if self.verbose:
                print(f"Error rewriting query: {e}")
            return query

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user query by detecting if it is a follow-up question.
        If so, augment the query using the conversation context.
        Then call the underlying RAG pipeline to generate an answer.

        Args:
            user_input: The user's query

        Returns:
            A dictionary containing the answer, sources, and other metadata
        """
        # Check if this is a follow-up question
        is_followup = self._detect_followup_question(user_input)
        
        if is_followup and self.conversation_history:
            # Get conversation context
            conversation_context = self.get_conversation_context()
            
            # Rewrite the query with context
            processed_query = self._rewrite_with_context(user_input, conversation_context)
        else:
            processed_query = user_input
            
        if self.verbose:
            print(f"Is follow-up: {is_followup}")
            print(f"Processed query: {processed_query}")
            
        # Call the underlying RAG pipeline
        result = self.rag_pipeline.invoke(processed_query)
        
        # Extract answer and sources from the result
        # The exact structure depends on the RAG pipeline implementation
        if isinstance(result, dict):
            answer = result.get("answer", str(result))
            sources = result.get("sources", [])
        else:
            answer = str(result)
            sources = []
            
        # Add to conversation history
        self.add_to_history(user_input, answer, sources)
        
        # Return the result
        return {
            "query": user_input,
            "processed_query": processed_query,
            "answer": answer,
            "sources": sources,
            "is_followup": is_followup
        }