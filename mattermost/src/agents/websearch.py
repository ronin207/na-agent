"""
WebSearchAgent module for retrieving information from web search engines.


This module provides a WebSearchAgent class that can be used as a fallback mechanism
to retrieve additional information when local context is insufficient.
"""


import openai
import logging
import json
import time
import random
import re
from typing import Dict, List, Optional, Union, Any, Tuple


# Configure logging
logger = logging.getLogger(__name__)




class WebSearchAgent:
   """
   Agent for performing web searches and formatting results for use in a RAG pipeline.
  
   This agent serves as a fallback mechanism to retrieve additional information
   when local PDF-based context is insufficient. It uses OpenAI's API to generate
   search results based on the query, providing a consistent interface for web search
   functionality within the RAG system.
   """
  
   def __init__(
       self,
       api_keys: Dict[str, str],
       search_engine: str = 'openai',
       max_results: int = 5,
       verbose: bool = False
   ):
       """
       Initialize the WebSearchAgent.
      
       Args:
           api_keys: Dictionary containing API keys for different search engines.
                     Expected key is 'openai'.
           search_engine: The search engine to use (only 'openai' is supported).
                         Note: This parameter is ignored as OpenAI search is always used.
           max_results: Maximum number of search results to return.
           verbose: Whether to print verbose output during searches.
       """
       self.api_keys = api_keys
       # Always use OpenAI search regardless of the input parameter
       self.search_engine = 'openai'  # OpenAI search is always used with gpt-4o-mini-search-preview model
       self.max_results = max_results
       self.verbose = verbose
  
   def _openai_search(self, query: str) -> List[Dict[str, str]]:
       """
       Perform a web search using OpenAI's API.
      
       Args:
           query: The search query.
          
       Returns:
           A list of dictionaries containing search results with 'title', 'link', and 'snippet' keys.
          
       Raises:
           openai.OpenAIError: If there's an issue with the OpenAI API call.
       """
       if self.verbose:
           print(f"Performing OpenAI web search for: {query}")
      
       logger.info(f"Initiating OpenAI web search for query: {query}")
      
       # Initialize the OpenAI client
       client = openai.OpenAI(api_key=self.api_keys.get('openai'))
      
       try:
           # Call the OpenAI API with web search enabled - only use required parameters
           # No temperature or response_format parameters as they're not supported
           response = client.chat.completions.create(
               model="gpt-4o-mini-search-preview",
               messages=[
                   {
                       "role": "user",
                       "content": query
                   }
               ]
           )
          
           # Extract the response content directly
           content = response.choices[0].message.content
           logger.debug(f"Successfully received response from OpenAI web search")
          
           # Extract URLs from the content using regex
           urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', content)
           logger.debug(f"Extracted {len(urls)} URLs from response content")
          


           # Format the search results
           formatted_results = []
          
           # Add a header result with the search query and first URL if available
           formatted_results.append({
               "title": f"Web Search Results for: {query}",
               "link": urls[0] if urls else "https://search.example.com",
               "snippet": f"The following information was retrieved from the web in response to your query: '{query}'"
           })
          
           # Add the main result with the content from OpenAI and second URL if available
           formatted_results.append({
               "title": "Web Search Result",
               "link": urls[1] if len(urls) > 1 else (urls[0] if urls else "https://search.openai.com"),
               "snippet": content
           })
          
           # Add additional results with actual URLs if available
           url_index = 2
           while len(formatted_results) < self.max_results and url_index < len(urls):
               formatted_results.append({
                   "title": f"Additional Information for: {query}",
                   "link": urls[url_index],
                   "snippet": f"Additional web search result from {urls[url_index]}"
               })
               url_index += 1
              
           # Add placeholder results only if needed to meet max_results and we've used all available URLs
           while len(formatted_results) < self.max_results:
               formatted_results.append({
                   "title": f"Additional Source {url_index-1} for: {query}",
                   "link": urls[url_index] if url_index < len(urls) else "https://search.openai.com/search?q=" + query.replace(' ', '+'),
                   "snippet": f"This is an additional source found during web search for '{query}'."
               })
               url_index += 1
          
           # If we still need more results to meet max_results, add generic entries
           while len(formatted_results) < self.max_results:
               formatted_results.append({
                   "title": f"Additional Information for: {query}",
                   "link": urls[0] if urls else f"https://search.openai.com/search?q={query.replace(' ', '+')}",
                   "snippet": "This is a supplementary result based on the web search."
               })
          
           # Limit to max_results
           final_results = formatted_results[:self.max_results]
           logger.info(f"Successfully retrieved and formatted web search results with {len(urls)} extracted URLs")
           return final_results
          
       except openai.OpenAIError as e:
           error_msg = f"OpenAI API error: {str(e)}"
           logger.error(error_msg)
           if self.verbose:
               print(error_msg)
              
           # Check if this is a quota exceeded error or rate limit error
           error_str = str(e).lower()
           if ("exceeded" in error_str and "quota" in error_str) or "resourceexhausted" in error_str or "rate limit" in error_str:
               quota_error_msg = f"API quota or rate limit exceeded: {str(e)}. Returning fallback results."
               logger.warning(quota_error_msg)
               if self.verbose:
                   print(quota_error_msg)
              
           # Return fallback results for any OpenAI error
           return self._generate_fallback_results(query, f"OpenAI API error: {str(e)}")
          
       except Exception as e:
           error_msg = f"Unexpected error in OpenAI search: {str(e)}"
           logger.error(error_msg)
           if self.verbose:
               print(error_msg)
          
           # Return fallback results for any unexpected error
           return self._generate_fallback_results(query, f"Unexpected error: {str(e)}")
  
   def _generate_fallback_results(self, query: str, reason: str) -> List[Dict[str, str]]:
       """
       Generate fallback search results when API calls fail.
      
       Args:
           query: The original search query
           reason: The reason for falling back to generated results
          
       Returns:
           A list of dictionaries containing minimal search results
       """
       logger.warning(f"Generating fallback results for query '{query}'. Reason: {reason}")
      
       # Create a consistent set of fallback results
       fallback_results = []
      
       # Create a search query URL for the fallback
       encoded_query = query.replace(' ', '+')
       search_url = f"https://search.openai.com/search?q={encoded_query}"
      
       # Add a header result explaining the fallback
       fallback_results.append({
           "title": f"Web Search Unavailable for: {query}",
           "link": "https://search.failed/web_search_unavailable",
           "snippet": f"Web search could not be performed due to: {reason}. The following are placeholder results."
       })
      
       # Add a single additional fallback result with a more meaningful URL
       fallback_results.append({
           "title": f"Fallback Information for '{query}'",
           "link": "https://search.failed/fallback_information",
           "snippet": f"This is a generated result about {query} based on available knowledge. Note: Web search failed due to {reason}."
       })
      
       return fallback_results
  
  
   def search(self, query: str) -> List[Dict[str, str]]:
       """
       Perform a web search using the configured search engine.
      
       Args:
           query: The search query.
          
       Returns:
           A list of dictionaries containing search results.
      
       Raises:
           ValueError: If the configured search engine is not supported.
       """
       logger.info(f"Web search triggered for query: '{query}'")
      
       start_time = __import__('time').time()
       max_retries = 3
       retry_count = 0
      
       while retry_count < max_retries:
           try:
               # Always use OpenAI search
               results = self._openai_search(query)
              
               # If we got results successfully, break out of the retry loop
               break
              
           except openai.OpenAIError as e:
               retry_count += 1
               error_str = str(e).lower()
              
               # Check if this is a quota exceeded error or rate limit error
               if ("exceeded" in error_str and "quota" in error_str) or "resourceexhausted" in error_str or "rate limit" in error_str or "429" in error_str:
                   logger.warning(f"API quota or rate limit exceeded: {str(e)}. Returning fallback results without retrying.")
                   results = self._generate_fallback_results(query, "API quota or rate limit exceeded")
                   break
              
               # For other errors, implement exponential backoff
               if retry_count < max_retries:
                   # Calculate backoff time: 2^retry_count + random jitter
                   backoff_time = 2 ** retry_count + random.random()
                   logger.warning(f"Search attempt {retry_count} failed: {str(e)}. Retrying in {backoff_time:.2f} seconds...")
                   time.sleep(backoff_time)
               else:
                   logger.error(f"All {max_retries} search attempts failed. Last error: {str(e)}")
                   results = self._generate_fallback_results(query, f"All {max_retries} attempts failed")
          
           except Exception as e:
               logger.error(f"Unexpected error in search: {str(e)}")
               results = self._generate_fallback_results(query, "Unexpected error")
               break
      
       end_time = __import__('time').time()
       duration = round(end_time - start_time, 2)
      
       logger.info(f"Web search completed in {duration}s. Retrieved {len(results)} results for query: '{query}'")
      
       return results
  
   def is_search_needed(self, query: str, context: str) -> Tuple[bool, str]:
       """
       Determine if a web search is needed based on the query and current context.
      
       This method uses heuristic checks to decide if the current context is sufficient
       to answer the query or if additional information from web search is needed.
      
       Args:
           query: The user's query.
           context: The current context available from local documents.
          
       Returns:
           A tuple containing:
           - A boolean indicating whether a web search is needed
           - A string explaining the reason for the decision
       """
       logger.info(f"Evaluating if web search is needed for query: {query}")
      
       # Check if context is empty or very short
       if not context or len(context.strip()) < 100:
           reason = "Insufficient context available from local documents"
           logger.info(f"Web search needed: {reason}")
           return True, reason
      
       # Check for time-sensitive queries (current events, recent developments)
       time_keywords = ["recent", "latest", "current", "today", "yesterday", "this week",
                        "this month", "this year", "update", "news"]
      
       if any(keyword in query.lower() for keyword in time_keywords):
           reason = "Query appears to be about recent events or time-sensitive information"
           logger.info(f"Web search needed: {reason}")
           return True, reason
      
       # Check for specific factual queries that might not be in the context
       factual_indicators = ["who is", "what is", "when did", "where is", "how many",
                            "statistics", "data", "numbers", "percentage"]
      
       if any(indicator in query.lower() for indicator in factual_indicators):
           # Only trigger if these terms don't appear in the context
           if not any(indicator in context.lower() for indicator in factual_indicators):
               reason = "Query requests specific factual information not found in context"
               logger.info(f"Web search needed: {reason}")
               return True, reason
      
       # Domain relevance check for numerical analysis and programming
       numerical_programming_keywords = [
           "algorithm", "numerical", "computation", "programming", "function", "equation",
           "matrix", "vector", "iteration", "convergence", "error", "approximation",
           "differential", "integral", "linear", "nonlinear", "newton", "euler",
           "interpolation", "extrapolation", "python", "code", "implementation",
           "method", "solution", "system", "optimization", "root", "eigenvalue",
           "derivative", "calculus", "norm", "variable", "parameter", "coefficient"
       ]
      
       # Count domain-specific keywords in query and context
       query_domain_count = sum(1 for keyword in numerical_programming_keywords if keyword in query.lower())
       context_domain_count = sum(1 for keyword in numerical_programming_keywords if keyword in context.lower())
      
       # Calculate domain relevance ratios
       query_words = [word for word in query.lower().split() if len(word) > 3]
       query_word_count = max(len(query_words), 1)  # Avoid division by zero
      
       query_domain_ratio = query_domain_count / query_word_count
      
       # If query has low domain relevance, suggest web search
       if query_domain_ratio < 0.2:
           reason = f"Query outside primary domain of numerical analysis/programming (domain relevance: {query_domain_ratio:.2f})"
           logger.info(f"Web search needed: {reason}")
           return True, reason
      
       # Use a simple heuristic to check if key terms from the query appear in the context
       query_terms = set(term.lower() for term in query.split() if len(term) > 3)
       if query_terms:
           matching_terms = sum(1 for term in query_terms if term in context.lower())
           match_ratio = matching_terms / len(query_terms)
          
           if match_ratio < 0.5:  # Less than half of the query terms appear in the context
               reason = f"Only {match_ratio:.0%} of query terms found in context"
               logger.info(f"Web search needed: {reason}")
               return True, reason
      
       # Default: context seems sufficient
       logger.info("Web search not needed: Context appears sufficient")
       return False, "Context appears sufficient to answer the query"
  
   def format_search_results(self, results: List[Dict[str, str]]) -> str:
       """
       Format search results into a structured context string for the RAG pipeline.
      
       Args:
           results: A list of dictionaries containing search results.
          
       Returns:
           A formatted string containing the search results as additional context.
          
       Note:
           When results from the web search agent are included in the context,
           these results are used as additional context for the LLM to generate
           improved answers, especially for queries where local context is insufficient.
          
           The formatted output is clearly labeled with "WEB SEARCH RESULTS" markers
           to distinguish it from other context sources. These markers are important for:
           1. Helping the LLM identify the source of information
           2. Signaling in the final answer that web search fallback was triggered
           3. Allowing downstream processes to detect when web search was used
       """
       if not results:
           logger.warning("No web search results to format")
           return "No web search results found."
      
       logger.info(f"Formatting {len(results)} web search results")
      
       # Clearly label the beginning of web search results for the LLM to recognize this section
       # These markers are used to signal in the final answer that web search fallback was triggered
       formatted_text = "=== WEB SEARCH RESULTS ===\n\n"
       formatted_text += "Note: These results were retrieved because the query appears to be outside the primary domain of numerical analysis and programming covered by the local documents.\n\n"
      
       for i, result in enumerate(results, 1):
           formatted_text += f"[{i}] {result.get('title', 'No title')}\n"
           formatted_text += f"URL: {result.get('link', 'No link')}\n"
           formatted_text += f"Snippet: {result.get('snippet', 'No snippet available')}\n\n"
      
       # Clearly mark the end of web search results to help the LLM distinguish this from other context
       # This end marker helps the answer generation process identify that web search fallback was used
       formatted_text += "=== END OF WEB SEARCH RESULTS ==="
      
       logger.debug("Web search results successfully formatted")
      
       return formatted_text