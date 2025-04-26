"""
Adaptive RAG module for query classification and parameter adjustment.


This module contains the AdaptiveRAG class which is responsible for classifying
queries and adjusting retrieval parameters based on the query type.
"""


import logging
from typing import Dict, List, Any, Optional, Tuple


from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel


logger = logging.getLogger(__name__)




class AdaptiveRAG:
   """
   AdaptiveRAG handles query classification and parameter adjustment.
  
   This class analyzes the input query to determine its type and adjusts
   retrieval parameters accordingly to optimize the retrieval process.
   """
  
   def __init__(
       self,
       llm: BaseLanguageModel,
       verbose: bool = False
   ):
       """
       Initialize the AdaptiveRAG instance.
      
       Args:
           llm: The language model to use for query classification
           verbose: Whether to log detailed information
       """
       self.llm = llm
       self.verbose = verbose
      
   def classify_query(self, query: str) -> Dict[str, Any]:
       """
       Classify the query to determine its type and appropriate retrieval parameters.
      
       Args:
           query: The user query to classify
          
       Returns:
           A dictionary containing query classification and adjusted parameters
       """
       # Define the prompt for query classification
       classification_prompt = f"""
       You are an AI assistant that classifies user queries to determine the optimal retrieval strategy.
      
       First, determine if the query is about numerical analysis, mathematical concepts, or related topics.
       If it's not, mark it as out-of-domain.
      
       Then, if it's in-domain, classify it into one of these categories:
       1. Factual: Seeking specific facts or information
       2. Conceptual: Seeking explanation of concepts or ideas
       3. Procedural: Seeking steps or instructions
       4. Comparative: Seeking comparison between entities
       5. Open-ended: Seeking opinions or broad information
      
       Query: {query}
      
       Respond with a JSON object containing:
       {{
           "is_out_of_domain": true/false,
           "query_type": "factual/conceptual/procedural/comparative/open-ended",
           "num_docs": 1-10,
           "similarity_threshold": 0.0-1.0,
           "reason": "Brief explanation of classification"
       }}
      
       Example for in-domain query:
       {{
           "is_out_of_domain": false,
           "query_type": "conceptual",
           "num_docs": 5,
           "similarity_threshold": 0.6,
           "reason": "Query asks about mathematical concept explanation"
       }}
      
       Example for out-of-domain query:
       {{
           "is_out_of_domain": true,
           "query_type": "factual",
           "num_docs": 0,
           "similarity_threshold": 0.0,
           "reason": "Query is about non-mathematical topic"
       }}
       """
      
       try:
           # Get classification from LLM
           response = self.llm.predict(classification_prompt)
          
           # Parse the response to extract classification and parameters
           # For now, we'll do a simple check for common non-mathematical terms
           non_math_keywords = [
               "movie", "film", "actor", "actress", "celebrity", "sport", "game",
               "politics", "politician", "entertainment", "music", "song", "food",
               "recipe", "restaurant", "travel", "vacation", "hotel"
           ]
           
           # Check if query contains non-math keywords
           query_lower = query.lower()
           contains_non_math = any(keyword in query_lower for keyword in non_math_keywords)
           
           if contains_non_math:
               return {
                   "is_out_of_domain": True,
                   "query_type": "factual",
                   "num_docs": 0,
                   "similarity_threshold": 0.0,
                   "reason": "Query contains non-mathematical terms"
               }
           
           # Default classification for in-domain queries
           return {
               "is_out_of_domain": False,
               "query_type": "factual",
               "num_docs": 3,
               "similarity_threshold": 0.7,
               "reason": "Query appears to be about mathematical concepts"
           }
          
       except Exception as e:
           logger.error(f"Error in query classification: {str(e)}")
           # Return default values if classification fails
           return {
               "is_out_of_domain": False,
               "query_type": "factual",
               "num_docs": 3,
               "similarity_threshold": 0.7,
               "reason": "Classification failed, using default values"
           }
  
   def classify(self, query: str) -> str:
       """Legacy method for backward compatibility. Calls classify_query and returns the query type."""
       try:
           result = self.classify_query(query)
           return result.get('query_type', 'factual')
       except Exception as e:
           logger.error(f'Error in classify: {e}')
           return 'factual'
  
   def adjust_retrieval_parameters(
       self,
       query: str,
       default_params: Dict[str, Any]
   ) -> Dict[str, Any]:
       """
       Adjust retrieval parameters based on query classification.
      
       Args:
           query: The user query
           default_params: Default retrieval parameters
          
       Returns:
           Adjusted retrieval parameters
       """
       # Classify the query
       classification = self.classify_query(query)
      
       # Start with default parameters
       adjusted_params = default_params.copy()
      
       # Adjust parameters based on classification
       adjusted_params["k"] = classification.get("num_docs", default_params.get("k", 3))
       adjusted_params["score_threshold"] = classification.get(
           "similarity_threshold",
           default_params.get("score_threshold", 0.7)
       )
      
       if self.verbose:
           logger.info(f"Adjusted retrieval parameters: {adjusted_params}")
          
       return adjusted_params
      
   def adjust_parameters(self, query_type: str) -> Dict[str, Any]:
       '''Return retrieval parameters based on the query type.'''
       # Define a mapping from query type to parameters
       mapping = {
           'factual': {'k': 3, 'score_threshold': 0.7},
           'conceptual': {'k': 5, 'score_threshold': 0.6},
           'procedural': {'k': 4, 'score_threshold': 0.65},
           'comparative': {'k': 4, 'score_threshold': 0.60},
           'open-ended': {'k': 6, 'score_threshold': 0.5}
       }
       # Use lower-case for normalization
       return mapping.get(query_type.lower(), {'k': 3, 'score_threshold': 0.7})



