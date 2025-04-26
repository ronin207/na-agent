"""
Corrective RAG Module


This module contains the CorrectiveRAG class which implements a corrective refinement
strategy for retrieval-augmented generation based on evaluation metrics.
"""


import logging
from typing import Dict, List, Any, Optional, Tuple


from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


logger = logging.getLogger(__name__)


class CorrectiveRAG:
   """
   CorrectiveRAG class triggers corrective refinement based on evaluation metrics.
  
   This class implements a strategy to refine retrieval results when the initial
   generation doesn't meet quality thresholds. It uses evaluation metrics to
   determine when to trigger corrective actions and how to refine the retrieval
   process to improve response quality.
   """
  
   def __init__(
       self,
       llm,
       retriever,
       evaluation_chain=None,
       metrics_threshold: Dict[str, float] = None,
       max_iterations: int = 3
   ):
       """
       Initialize the CorrectiveRAG component.
      
       Args:
           llm: The language model to use for generation
           retriever: The retriever to use for document retrieval
           evaluation_chain: Optional chain for evaluating response quality
           metrics_threshold: Dictionary of metric names to threshold values
           max_iterations: Maximum number of refinement iterations
       """
       self.llm = llm
       self.retriever = retriever
       self.evaluation_chain = evaluation_chain
       self.metrics_threshold = metrics_threshold or {
           "relevance": 0.7,
           "faithfulness": 0.7,
           "coherence": 0.7
       }
       self.max_iterations = max_iterations
      
       # Define the refinement prompt
       self.refinement_prompt = PromptTemplate(
           input_variables=["question", "previous_answer", "feedback"],
           template="""
           You are tasked with improving a response to a user question.
          
           User Question: {question}
          
           Previous Answer: {previous_answer}
          
           Feedback on Previous Answer: {feedback}
          
           Please provide a better answer that addresses the feedback.
           """
       )
      
       # Create the refinement chain using RunnableSequence
       self.refinement_chain = (
           self.refinement_prompt
           | self.llm
           | StrOutputParser()
       )
  
   def evaluate_response(self, question: str, answer: str, docs: List[Document]) -> Dict[str, float]:
       """
       Evaluate the quality of a response using the evaluation chain.
      
       Args:
           question: The original user question
           answer: The generated answer
           docs: The retrieved documents used for generation
          
       Returns:
           Dictionary of evaluation metrics
       """
       if not self.evaluation_chain:
           logger.warning("No evaluation chain provided, skipping evaluation")
           return {"relevance": 1.0, "faithfulness": 1.0, "coherence": 1.0}
      
       try:
           eval_result = self.evaluation_chain.run(
               question=question,
               answer=answer,
               retrieved_documents=docs
           )
           return eval_result
       except Exception as e:
           logger.error(f"Error during response evaluation: {e}")
           return {"relevance": 1.0, "faithfulness": 1.0, "coherence": 1.0}
  
   def needs_refinement(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
       """
       Determine if the response needs refinement based on evaluation metrics.
      
       Args:
           metrics: Dictionary of evaluation metrics
          
       Returns:
           Tuple of (needs_refinement, feedback)
       """
       feedback = []
       needs_refinement = False
      
       for metric, value in metrics.items():
           if metric in self.metrics_threshold and value < self.metrics_threshold[metric]:
               needs_refinement = True
               feedback.append(f"The {metric} score of {value} is below the threshold of {self.metrics_threshold[metric]}.")
      
       return needs_refinement, "\n".join(feedback)
  
   def refine_retrieval(self, question: str, previous_docs: List[Document]) -> List[Document]:
       """
       Refine the retrieval process to get better documents.
      
       Args:
           question: The original user question
           previous_docs: Previously retrieved documents
          
       Returns:
           New list of retrieved documents
       """
       # Implement retrieval refinement logic
       # This could involve query expansion, filtering, or other techniques
       expanded_query = self._expand_query(question, previous_docs)
       new_docs = self.retriever.get_relevant_documents(expanded_query)
      
       # Combine with previous docs and remove duplicates
       all_docs = previous_docs + new_docs
       unique_docs = self._remove_duplicate_docs(all_docs)
      
       return unique_docs[:min(len(unique_docs), 5)]  # Limit to top 5 docs
  
   def _expand_query(self, question: str, docs: List[Document]) -> str:
       """
       Expand the original query to improve retrieval.
      
       Args:
           question: The original user question
           docs: Previously retrieved documents
          
       Returns:
           Expanded query string
       """
       expansion_prompt = PromptTemplate(
           input_variables=["question", "doc_contents"],
           template="""
           Based on the original question and the retrieved documents,
           create an expanded search query that might retrieve more relevant information.
          
           Original question: {question}
          
           Retrieved document contents:
           {doc_contents}
          
           Expanded search query:
           """
       )
      
       doc_contents = "\n\n".join([doc.page_content for doc in docs[:2]])
      
       try:
           expanded_query = self.llm.predict(
               expansion_prompt.format(
                   question=question,
                   doc_contents=doc_contents
               )
           )
           return expanded_query
       except Exception as e:
           logger.error(f"Error during query expansion: {e}")
           return question
  
   def _remove_duplicate_docs(self, docs: List[Document]) -> List[Document]:
       """
       Remove duplicate documents from a list.
      
       Args:
           docs: List of documents
          
       Returns:
           Deduplicated list of documents
       """
       unique_contents = set()
       unique_docs = []
      
       for doc in docs:
           if doc.page_content not in unique_contents:
               unique_contents.add(doc.page_content)
               unique_docs.append(doc)
      
       return unique_docs
  
   def generate_refined_answer(
       self,
       question: str,
       previous_answer: str,
       feedback: str
   ) -> str:
       """
       Generate a refined answer based on feedback.
      
       Args:
           question: The original user question
           previous_answer: The previous answer
           feedback: Feedback on what needs improvement
          
       Returns:
           Refined answer
       """
       try:
           refined_answer = self.refinement_chain.run(
               question=question,
               previous_answer=previous_answer,
               feedback=feedback
           )
           return refined_answer
       except Exception as e:
           logger.error(f"Error during answer refinement: {e}")
           return previous_answer
  
   def process(
       self,
       question: str,
       initial_docs: List[Document],
       initial_answer: str
   ) -> Tuple[str, List[Document]]:
       """
       Process a question through the corrective RAG pipeline.
      
       Args:
           question: The user question
           initial_docs: Initially retrieved documents
           initial_answer: Initial answer generated
          
       Returns:
           Tuple of (final_answer, final_docs)
       """
       current_answer = initial_answer
       current_docs = initial_docs
      
       for i in range(self.max_iterations):
           # Evaluate current answer
           metrics = self.evaluate_response(question, current_answer, current_docs)
           needs_refinement, feedback = self.needs_refinement(metrics)
          
           if not needs_refinement:
               logger.info(f"Answer meets quality thresholds after {i} refinements")
               break
              
           logger.info(f"Refinement iteration {i+1}: {feedback}")
          
           # Refine retrieval
           current_docs = self.refine_retrieval(question, current_docs)
          
           # Generate refined answer
           current_answer = self.generate_refined_answer(
               question=question,
               previous_answer=current_answer,
               feedback=feedback
           )
      
       return current_answer, current_docs

