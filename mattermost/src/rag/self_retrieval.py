"""
Self-RAG implementation for evaluating and refining retrieval results.


This module contains the SelfRAG class which implements a self-feedback
evaluation mechanism for retrieval-augmented generation.
"""


import logging
from typing import Dict, List, Any, Optional, Union


from langchain.schema import BaseMessage
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage


logger = logging.getLogger(__name__)


class SelfRAG:
   """
   SelfRAG implements a self-feedback evaluation mechanism for retrieval results.
  
   This class evaluates the relevance and utility of retrieved documents for answering
   a given query, providing a self-critique mechanism that helps improve the quality
   of the final response by filtering out irrelevant or low-quality retrieved information.
   """
  
   def __init__(
       self,
       llm: BaseLanguageModel,
       system_prompt: Optional[str] = None,
       verbose: bool = False
   ):
       """
       Initialize the SelfRAG evaluator.
      
       Args:
           llm: The language model to use for evaluation
           system_prompt: Optional system prompt to override the default
           verbose: Whether to log detailed information
       """
       self.llm = llm
       self.verbose = verbose
      
       # Default system prompt for self-evaluation
       self.system_prompt = system_prompt or (
           "You are an AI assistant evaluating the relevance of retrieved documents "
           "for answering a user query. For each document, determine if it contains "
           "information that is relevant and helpful for answering the query. "
           "Provide a score from 0-10 for each document, where 0 means completely "
           "irrelevant and 10 means highly relevant and directly answers the query."
       )
      
       # Prompt template for document evaluation
       self.eval_template = PromptTemplate(
           input_variables=["query", "documents"],
           template=(
               "User Query: {query}\n\n"
               "Retrieved Documents:\n{documents}\n\n"
               "For each document, evaluate its relevance to the query on a scale of 0-10 "
               "and provide a brief explanation. Format your response as:\n"
               "Document 1: [score] - [explanation]\n"
               "Document 2: [score] - [explanation]\n"
               "...\n"
               "Then provide a final list of document indices to keep, ordered by relevance."
           )
       )
  
   def evaluate_documents(
       self,
       query: str,
       documents: List[Dict[str, Any]],
       threshold: float = 5.0
   ) -> List[Dict[str, Any]]:
       """
       Evaluate the relevance of retrieved documents for the given query.
      
       Args:
           query: The user query
           documents: List of retrieved documents
           threshold: Minimum relevance score to keep a document (0-10)
          
       Returns:
           List of documents that passed the relevance threshold
       """
       if not documents:
           logger.warning("No documents to evaluate")
           return []
      
       # Format documents for evaluation
       docs_text = ""
       for i, doc in enumerate(documents):
           content = doc.get("page_content", doc.get("content", ""))
           metadata = doc.get("metadata", {})
           source = metadata.get("source", "Unknown")
           docs_text += f"Document {i+1} [Source: {source}]:\n{content}\n\n"
      
       # Create messages for the LLM
       messages = [
           SystemMessage(content=self.system_prompt),
           HumanMessage(content=self.eval_template.format(
               query=query,
               documents=docs_text
           ))
       ]
      
       # Get evaluation from LLM
       response = self.llm.invoke(messages)
      
       if self.verbose:
           logger.info(f"SelfRAG evaluation response: {response.content}")
      
       # Parse the response to extract document scores and indices to keep
       filtered_docs = self._parse_evaluation(response.content, documents, threshold)
      
       return filtered_docs
  
   def _parse_evaluation(
       self,
       evaluation: str,
       documents: List[Dict[str, Any]],
       threshold: float
   ) -> List[Dict[str, Any]]:
       """
       Parse the evaluation response to extract document scores and filter documents.
      
       Args:
           evaluation: The LLM's evaluation response
           documents: Original list of documents
           threshold: Minimum score to keep a document
          
       Returns:
           Filtered list of documents
       """
       # Initialize scores dictionary
       scores = {}
      
       # Try to parse the evaluation line by line
       lines = evaluation.split('\n')
       for line in lines:
           line = line.strip()
           if line.startswith("Document ") and ":" in line:
               try:
                   # Extract document number and score
                   doc_part = line.split(":")[0].strip()
                   doc_num = int(doc_part.replace("Document ", "")) - 1
                  
                   score_part = line.split(":")[1].strip()
                   if "-" in score_part:
                       score_str = score_part.split("-")[0].strip()
                       # Handle different score formats (e.g., "8/10" or just "8")
                       if "/" in score_str:
                           score = float(score_str.split("/")[0])
                       else:
                           score = float(score_str)
                      
                       scores[doc_num] = score
               except (ValueError, IndexError) as e:
                   logger.warning(f"Error parsing evaluation line: {line}. Error: {e}")
      
       # Look for a final list of documents to keep
       keep_indices = []
       for i, line in enumerate(lines):
           if "final list" in line.lower() or "documents to keep" in line.lower():
               # Try to extract document indices from the next few lines
               for j in range(i+1, min(i+10, len(lines))):
                   if lines[j].strip():
                       # Look for numbers in this line
                       import re
                       nums = re.findall(r'\d+', lines[j])
                       keep_indices.extend([int(n) - 1 for n in nums])
      
       # If we found explicit indices to keep, use those
       if keep_indices:
           filtered_docs = [documents[i] for i in keep_indices if 0 <= i < len(documents)]
       else:
           # Otherwise filter based on scores
           filtered_docs = [doc for i, doc in enumerate(documents)
                           if i in scores and scores[i] >= threshold]
      
       if self.verbose:
           logger.info(f"SelfRAG kept {len(filtered_docs)} out of {len(documents)} documents")
          
       return filtered_docs
      
   def evaluate(self, query: str, context: str, answer: str) -> Dict[str, Any]:
       """Evaluate the quality of an answer based on the query and context."""
       from langchain.schema.messages import SystemMessage, HumanMessage
       if not context or not answer:
           logger.warning("Empty context or answer provided for evaluation")
           return {"confidence_score": 0.0, "critique": "Cannot evaluate empty content", "improvement_suggestions": ""}
      
       messages = [
           SystemMessage(content="You are an expert evaluator assessing the quality of an answer."),
           HumanMessage(content=f"""
           Query: {query}
          
           Context: {context[:5000]}
          
           Answer: {answer}
          
           Evaluate this answer based on relevance, factual accuracy, completeness, and clarity.
           Provide a detailed critique, a confidence score between 0.0 and 1.0, and suggestions for improvement.


           Format your response as:
           Critique: <your detailed critique>
           Confidence Score: <score between 0.0 and 1.0>
           Improvement Suggestions: <your suggestions here>
           """)
       ]
       response = self.llm.invoke(messages)
       return self._parse_evaluation_response(response.content)
      
   def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
       """Parse the evaluation response from the LLM to extract components."""
       import re
       result = {"confidence_score": 0.5, "critique": "", "improvement_suggestions": ""}
      
       confidence_match = re.search(r'Confidence Score:\s*(0\.\d+|1\.0|1|0)', response)
       if confidence_match:
           try:
               result["confidence_score"] = float(confidence_match.group(1))
           except ValueError:
               logger.warning(f"Could not parse confidence score: {confidence_match.group(1)}")
       critique_match = re.search(r'Critique:(.*?)(?:Confidence Score:|Improvement Suggestions:|$)', response, re.DOTALL)
       if critique_match:
           result["critique"] = critique_match.group(1).strip()
       suggestions_match = re.search(r'Improvement Suggestions:(.*?)$', response, re.DOTALL)
       if suggestions_match:
           result["improvement_suggestions"] = suggestions_match.group(1).strip()
       return result

