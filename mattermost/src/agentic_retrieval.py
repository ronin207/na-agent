from final_decorator import finalclass
import logging
import os
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dotenv import load_dotenv

from agents.conversational import ConversationalAgent
from agents.websearch import WebSearchAgent

from rag.retrieval import RAG_Pipeline
from rag.self_retrieval import SelfRAG
from rag.corrective_retrieval import CorrectiveRAG
from rag.adaptive_retrieval import AdaptiveRAG

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Define prompts directly in this file to avoid external dependencies
QUERY_REWRITING_PROMPT = """
You are an expert at reformulating queries to improve retrieval performance, especially for technical and mathematical topics.
Please rewrite the following query to make it more effective for retrieving relevant information:

Original Query: {query}

Your task is to rewrite this query to:
1. Make it more specific and detailed
2. Include relevant keywords that might help in retrieval (e.g., for mathematical concepts include related terms like 'definition', 'properties', 'theorem', 'formula', etc.)
3. Break down complex questions into clearer components
4. Remove any ambiguity
5. For mathematical concepts, include alternative names and notations


Rewritten Query:
"""

QUERY_REWRITING_SYSTEM_PROMPT = """
You are an AI assistant specialized in query reformulation to improve information retrieval, with expertise in mathematical and technical topics.
Your goal is to rewrite user queries to make them more effective for retrieving relevant documents.
Focus on clarity, specificity, and including key terms that will help match with relevant content.
For mathematical concepts, include formal terminology, alternative names, and related concepts.
For vector norms specifically, consider terms like: vector space, norm properties, Euclidean norm, Manhattan norm, p-norm, infinity norm, etc.
"""

ANSWER_GENERATION_PROMPT = """
Based on the following context, please provide a comprehensive answer to the query.

Query: {query}

Context:
{context}

Instructions:
1. Answer the query based only on the information provided in the context
2. If the context doesn't contain relevant information, state that clearly
3. Provide a detailed and well-structured answer
4. Cite specific parts of the context to support your answer
5. For mathematical content, present formulas and definitions clearly
6. Include examples if they are present in the context
7. Maintain the technical precision of the original content
8. For mathematical expressions, use LaTeX enclosed in double dollar signs ($$...$$) for display equations and single dollar signs ($...$) for inline equations


Answer:
"""

ANSWER_GENERATION_SYSTEM_PROMPT = """
You are an AI assistant that generates accurate and helpful answers based solely on the provided context.
Do not use prior knowledge or make assumptions beyond what's in the context.
Ensure your answers are factual, relevant, and directly address the user's query.
For mathematical and technical content:
1. Present definitions, theorems, and formulas with precision
2. Use clear notation and formatting for mathematical expressions
3. Explain technical concepts in a structured and logical manner
4. Preserve the rigor of the original mathematical content
5. Format mathematical expressions using LaTeX: use double dollar signs ($$...$$) for display equations and single dollar signs ($...$) for inline equations
"""

logger = logging.getLogger(__name__)

# Implement evaluation functions directly in this file
def evaluate_context_sufficiency(llm: BaseLanguageModel, query: str, context: str) -> Dict[str, Any]:
	prompt = """
	Evaluate if the following context is sufficient to answer the query.
	
	Query: {query}
	
	Context: {context}
	
	Analyze the context and provide a JSON output with the following structure:
	{{
		"sufficiency_score": <float between 0.0 and 1.0>,
		"missing_information": <boolean - true if key information is completely absent>,
		"missing_details": <boolean - true if information is present but lacks necessary details>,
		"insufficient": <boolean - true if the context is not adequate to fully answer the query>,
		"explanation": <string explaining your evaluation>
	}}
	
	Consider:
	- Does the context contain the necessary information?
	- Is the information complete enough to form a comprehensive answer?
	- Are there any obvious gaps in the information provided?
	
	Output only the JSON object.
	"""
	
	messages = [
		{"role": "system", "content": "You are an expert evaluator of context sufficiency."},
		{"role": "user", "content": prompt.format(query=query, context=context)},
	]
	
	result = llm.invoke(messages).content.strip()
	
	try:
		# Extract JSON if it's embedded in other text
		import re
		json_match = re.search(r'({.*})', result, re.DOTALL)
		if json_match:
			result = json_match.group(1)
	   	 
		evaluation = json.loads(result)
   	 
		# Ensure all required fields are present
		if "sufficiency_score" not in evaluation:
			evaluation["sufficiency_score"] = 0.5
		if "missing_information" not in evaluation:
			evaluation["missing_information"] = True
		if "missing_details" not in evaluation:
			evaluation["missing_details"] = True
		if "insufficient" not in evaluation:
			# Set insufficient flag based on sufficiency score and missing information
			evaluation["insufficient"] = (evaluation["sufficiency_score"] < 0.7 or
									 	evaluation["missing_information"] or
									 	evaluation["missing_details"])
		if "explanation" not in evaluation:
			evaluation["explanation"] = "Failed to extract explanation from evaluation."
	   	 
		# Ensure score is within bounds
		evaluation["sufficiency_score"] = min(max(float(evaluation["sufficiency_score"]), 0.0), 1.0)
   	 
		return evaluation
	except Exception as e:
		logger.warning(f"Could not parse context sufficiency evaluation: {e}. Raw result: {result}")
		return {
			"sufficiency_score": 0.5,
			"missing_information": True,
			"missing_details": True,
			"explanation": "Failed to parse evaluation result."
		}


def evaluate_faithfulness(llm: BaseLanguageModel, context: str, answer: str) -> float:
	prompt = """
	Evaluate if the following answer is faithful to the provided context.
	
	Context: {context}
	
	Answer: {answer}
	
	On a scale from 0.0 to 1.0, how faithful is this answer to the context?
	Consider:
	- Does the answer contain information not present in the context?
	- Does the answer contradict any information in the context?
	- Does the answer accurately represent the information in the context?
	
	Output only a number between 0.0 and 1.0 representing your evaluation.
	"""
	
	messages = [
		{"role": "system", "content": "You are an expert evaluator of answer faithfulness."},
		{"role": "user", "content": prompt.format(context=context, answer=answer)},
	]
	
	result = llm.invoke(messages).content.strip()
	
	try:
		score = float(result)
		return min(max(score, 0.0), 1.0)
	except ValueError:
		logger.warning(f"Could not parse faithfulness score: {result}")
		return 0.5


def evaluate_completeness(llm: BaseLanguageModel, query: str, answer: str) -> float:
	prompt = """
	Evaluate if the following answer completely addresses the query.
	
	Query: {query}
	
	Answer: {answer}
	
	On a scale from 0.0 to 1.0, how completely does this answer address the query?
	Consider:
	- Does the answer address all aspects of the query?
	- Is the answer thorough and comprehensive?
	- Are there any parts of the query left unanswered?
	
	Output only a number between 0.0 and 1.0 representing your evaluation.
	"""
	
	messages = [
		{"role": "system", "content": "You are an expert evaluator of answer completeness."},
		{"role": "user", "content": prompt.format(query=query, answer=answer)},
	]
	
	result = llm.invoke(messages).content.strip()
	
	try:
		score = float(result)
		return min(max(score, 0.0), 1.0)
	except ValueError:
		logger.warning(f"Could not parse completeness score: {result}")
		return 0.5


def evaluate_hallucination(llm: BaseLanguageModel, context: str, answer: str) -> float:
	prompt = """
	Evaluate if the following answer contains hallucinations not supported by the context.
	
	Context: {context}
	
	Answer: {answer}
	
	On a scale from 0.0 to 1.0, how much hallucination does this answer contain?
	Consider:
	- Does the answer include specific facts not present in the context?
	- Does the answer make claims that cannot be verified from the context?
	- Does the answer extrapolate beyond what can be reasonably inferred from the context?
	
	IMPORTANT: If the answer includes content from web search results (marked with '=== WEB SEARCH RESULTS ===' or [WEB] tags), 
	consider that such information is derived from less curated sources and may be inherently less reliable. 
	In your evaluation, assign a non-zero hallucination score (at least 0.15) if web-derived information is present, 
	even if the rest of the answer appears supported by local documents.
	
	Output only a number between 0.0 and 1.0 representing your evaluation.
	A higher score means MORE hallucination.
	"""
	
	messages = [
		{"role": "system", "content": "You are an expert evaluator of answer hallucination."},
		{"role": "user", "content": prompt.format(context=context, answer=answer)},
	]
	
	result = llm.invoke(messages).content.strip()
	
	try:
		score = float(result)
		# Apply minimum hallucination score if web search markers are present in the answer
		if score < 0.15 and (
			"=== WEB SEARCH RESULTS ===" in answer or 
			"[WEB]" in answer or 
			"Web Sources:" in answer
		):
			# Enforce a minimum hallucination score for web search content
			logger.info("Enforcing minimum hallucination score of 0.15 for answer containing web search content")
			score = 0.15
		return min(max(score, 0.0), 1.0)
	except ValueError:
		logger.warning(f"Could not parse hallucination score: {result}")
		return 0.5


@finalclass
class AgenticRetrieval(RAG_Pipeline):
	"""
	Enhanced RAG pipeline with advanced agentic techniques.
	This class extends the RAG_Pipeline with Self-RAG, Adaptive RAG, and Corrective RAG
	techniques to improve the quality and relevance of generated responses.
	
	Provides concise output that is dynamic to the query, similar to the style in retrieval.py.
	"""
	def __init__(
		self,
		pdf_folder: str = "./data/",
		chunk_size: int = 500,
		chunk_overlap: int = 300,
		embedding_model: str = "models/embedding-001",
		llm_model: str = "gemini-2.0-flash",
		persist_directory: str = "../chroma_db",
		temperature: float = 0.0,
		user_agent: str = "USER_AGENT",
		math_parsing_instruction: str = None,
		k: int = 15,
		rewrite_query: bool = True,
		evaluate: bool = True,
		self_rag_threshold: float = 0.7,
		adaptive_rag: bool = True,
		enable_corrective_rag: bool = True,
		force_rebuild: bool = False,
		test_mode: bool = False,
		verbose: bool = False,
		web_search_enabled: bool = False,
		web_search_threshold: float = 0.6,
		web_search_engine: str = 'google',
		enable_web_search_in_test_mode: bool = False,
		max_history_length: int = 5,
	):
		os.environ['USER_AGENT'] = user_agent
		
		# Set default math parsing instruction if not provided
		if math_parsing_instruction is None:
			math_parsing_instruction = "Format all mathematical expressions using LaTeX notation. Enclose display equations in double dollar signs ($$...$$) and inline equations in single dollar signs ($...$)."
   	 
		load_dotenv()
   	 
		google_api_key = os.getenv("GOOGLE_API_KEY")
		if not google_api_key:
			raise ValueError("GOOGLE_API_KEY environment variable is not set")
   	 
		self.pdf_folder = pdf_folder
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap
		self.embedding_model = embedding_model
		self.llm_model = llm_model
		self.persist_directory = persist_directory
		self.temperature = temperature
		self.user_agent = user_agent
		self.math_parsing_instruction = math_parsing_instruction
		self.k = k
		self.rewrite_query = rewrite_query
		self.evaluate = evaluate
		self.self_rag_threshold = self_rag_threshold
		self.adaptive_rag = adaptive_rag
		self.enable_corrective_rag = enable_corrective_rag
		self.force_rebuild = force_rebuild
		self.test_mode = test_mode
		self.verbose = verbose
		self.web_search_enabled = web_search_enabled
		self.web_search_threshold = web_search_threshold
		self.web_search_engine = web_search_engine
		self.enable_web_search_in_test_mode = enable_web_search_in_test_mode
		self.max_history_length = max_history_length
   	 
		# Use environment variables for API keys
		self.llm = ChatGoogleGenerativeAI(
			model=self.llm_model,
			temperature=self.temperature,
			google_api_key=google_api_key
		)
		logger.info(f"Initializing embeddings with model: {self.embedding_model}")
		self.embeddings = GoogleGenerativeAIEmbeddings(
			model=self.embedding_model,
			google_api_key=google_api_key
		)
		self.retriever = None
		self.document_count = 0
		self.retriever = None
		self.chunks_cached = False

		self.query_rewriting_prompt = PromptTemplate(
			template=QUERY_REWRITING_PROMPT,
			input_variables=["query"],
		)
		self.query_rewriting_system_prompt = QUERY_REWRITING_SYSTEM_PROMPT


		self.answer_generation_prompt = PromptTemplate(
			template=ANSWER_GENERATION_PROMPT,
			input_variables=["query", "context"],
		)
		self.answer_generation_system_prompt = ANSWER_GENERATION_SYSTEM_PROMPT
   	 
		# Store the math parsing instruction
		self.math_parsing_instruction = math_parsing_instruction
   	 
		self.query_classification_prompt = PromptTemplate(
			template="""
			Classify the following query into one of these categories:
			- Factual: Seeking specific facts or information
			- Conceptual: Seeking explanation of concepts or ideas
			- Procedural: Asking how to do something
			- Ambiguous: Unclear or could have multiple interpretations
			- Complex: Requiring deep analysis or multiple information sources
	   	 
			Query: {query}
	   	 
			Output only the category name.
			""",
			input_variables=["query"],
		)
   	 
		self.query_rewriting_chat_prompt = ChatPromptTemplate.from_messages([
			SystemMessagePromptTemplate.from_template(self.query_rewriting_system_prompt),
			HumanMessagePromptTemplate.from_template(QUERY_REWRITING_PROMPT)
		])
   	 
		self.answer_generation_chat_prompt = ChatPromptTemplate.from_messages([
			SystemMessagePromptTemplate.from_template(self.answer_generation_system_prompt),
			HumanMessagePromptTemplate.from_template(self.answer_generation_prompt.template)
		])
   	 
		self.self_rag_prompt = PromptTemplate(
			template="""
			You are an expert evaluator. Assess the quality of the following answer based on the provided context.
	   	 
			Query: {query}
	   	 
			Context: {context}
	   	 
			Generated Answer: {answer}
	   	 
			Evaluate the answer on the following criteria:
			1. Relevance to the query
			2. Factual accuracy based on the context
			3. Completeness of information
			4. Presence of any hallucinations or information not in the context
	   	 
			First, provide a detailed critique of the answer.
			Then, rate the overall quality on a scale of 0.0 to 1.0.
			Finally, suggest specific improvements if needed.
	   	 
			Output your response in the following format:
			Critique: <your detailed critique>
			Confidence Score: <score between 0.0 and 1.0>
			Improvement Suggestions: <specific suggestions for improvement>
			""",
			input_variables=["query", "context", "answer"],
		)
		# Initialize specialized classes
		self.self_rag_handler = SelfRAG(self.llm, self.self_rag_prompt)
		# Note: self.retriever is None at this point and will be set during setup
		self.corrective_rag_handler = CorrectiveRAG(self.llm, self.retriever)
		self.adaptive_rag_handler = AdaptiveRAG(self.llm, self.verbose)
   	 
		# Initialize web search agent if enabled
		self.web_search_enabled = web_search_enabled
		self.web_search_threshold = web_search_threshold
		self.web_search_agent = None
   	 
		if self.web_search_enabled:
			# Only use OpenAI API key for web search
			openai_api_key = { 
				'openai': os.environ.get("OPENAI_API_KEY"),
			}

			self.web_search_agent = WebSearchAgent(
				api_keys=openai_api_key,
				search_engine='openai',  
				max_results=5,
				verbose=self.verbose
			)
			logger.info(f"Web search fallback enabled with OpenAI search and threshold {self.web_search_threshold}")
   	 
		# Initialize conversational agent
		self.max_history_length = max_history_length
		self.conversational_agent = ConversationalAgent(
			rag_pipeline=self,
			max_history_length=self.max_history_length,
			verbose=self.verbose
		)


	async def load_documents(self) -> List[Document]:
		logger.info(f"Loading documents from {self.pdf_folder}")
   	 
		if not self.pdf_folder or not os.path.exists(self.pdf_folder):
			logger.warning(f"PDF folder {self.pdf_folder} does not exist")
			return []
   	 
		try:
			loader = DirectoryLoader(self.pdf_folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
			documents = loader.load()
			logger.info(f"Loaded {len(documents)} documents")
			return documents
		except Exception as e:
			logger.error(f"Error loading documents: {e}")
			return []
	
	def format_documents(self, documents: List[Any]) -> List[Document]:
		formatted_docs = []
		seen_ids = set()
   	 
		for doc in documents:
			# Deduplicate using a fingerprint of the page_content
			fingerprint = None
			try:
				import hashlib
				fingerprint = hashlib.sha256(str(doc.page_content).encode('utf-8')).hexdigest()
			except Exception:
				fingerprint = None
			if fingerprint and fingerprint in seen_ids:
				continue
			if fingerprint:
				seen_ids.add(fingerprint)
			if isinstance(doc, Document):
				# Store fingerprint in metadata
				if fingerprint:
					doc.metadata["fingerprint"] = fingerprint
				formatted_docs.append(doc)
			else:
				new_doc = Document(
					page_content=str(doc.page_content) if hasattr(doc, 'page_content') else str(doc),
					metadata=doc.metadata if hasattr(doc, 'metadata') else {}
				)
				if fingerprint:
					new_doc.metadata["fingerprint"] = fingerprint
				formatted_docs.append(new_doc)
   	 
		return formatted_docs
	
	def prepare_documents(self, documents: List[Document]) -> List[Document]:
		logger.info("Preparing documents")
   	 
		if not documents:
			return []
   	 
		text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=self.chunk_size,
			chunk_overlap=self.chunk_overlap,
			length_function=len
		)
   	 
		document_chunks = text_splitter.split_documents(documents)
   	 
		# Deduplicate document chunks based on content fingerprint
		unique_chunks = []
		seen = set()
		for doc in document_chunks:
			try:
				import hashlib
				fp = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
			except Exception:
				fp = doc.page_content
			if fp not in seen:
				seen.add(fp)
				unique_chunks.append(doc)
		logger.info(f"Created {len(unique_chunks)} unique document chunks from {len(document_chunks)} chunks")
   	 
		return unique_chunks
	
	def build_vectorstore(self, documents: List[Document]) -> None:
		logger.info("Building vector store")
   	 
		if not documents:
			logger.warning("No documents to build vector store")
			return
   	 
		os.makedirs(os.path.dirname(self.persist_directory), exist_ok=True)
   	 
		vector_store = FAISS.from_documents(documents, self.embeddings)
   	 
		vector_store.save_local(self.persist_directory)
   	 
		self.retriever = vector_store.as_retriever(search_kwargs={"k": self.k})
   	 
		self.document_count = len(documents)
   	 
		logger.info(f"Built vector store with {self.document_count} documents")
	
	def load_existing_vectorstore(self) -> bool:
		logger.info(f"Loading existing vector store from {self.persist_directory}")
   	 
		try:
			if os.path.exists(self.persist_directory):
				vector_store = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)
				self.retriever = vector_store.as_retriever(search_kwargs={"k": self.k})
				self.document_count = len(vector_store.docstore._dict)
				logger.info(f"Loaded vector store with {self.document_count} documents")
				doc_keys = list(vector_store.docstore._dict.keys())
				logger.info(f"First 5 document keys: {doc_keys[:5]}")
				doc_sources = [vector_store.docstore._dict[key].metadata.get('source', 'unknown')
						  	for key in list(vector_store.docstore._dict.keys())[:10]]
				logger.info(f"Sample document sources: {doc_sources}")
				# Count documents by source
				source_counts = {}
				for key in vector_store.docstore._dict.keys():
					source = str(vector_store.docstore._dict[key].metadata.get('source', 'unknown'))
					source_name = source.split('/')[-1] if '/' in source else source
					if source_name in source_counts:
						source_counts[source_name] += 1
					else:
						source_counts[source_name] = 1
				logger.info(f"Document counts by source: {source_counts}")
		   	 
				# Re-initialize corrective_rag_handler with the now available retriever
				if self.retriever is not None:
					self.corrective_rag_handler = CorrectiveRAG(self.llm, self.retriever)
		   	 
				return True
			else:
				logger.warning(f"Vector store directory {self.persist_directory} does not exist")
				return False
		except Exception as e:
			logger.error(f"Error loading vector store: {e}")
			return False
	
	def update_vectorstore(self, documents: List[Document]) -> None:
		logger.info("Updating vector store")
   	 
		if not documents:
			logger.warning("No documents to update vector store")
			return
   	 
		try:
			vector_store = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)
			
			# Create a set of existing fingerprints
			existing_fingerprints = set()
			for key, doc in vector_store.docstore._dict.items():
				f = doc.metadata.get("fingerprint")
				if f:
					existing_fingerprints.add(f)
			
			# Filter incoming documents for new ones
			new_documents = []
			for doc in documents:
				try:
					import hashlib
					fp = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
				except Exception:
					fp = doc.page_content
				if fp not in existing_fingerprints:
					# Add fingerprint to metadata if not present
					doc.metadata["fingerprint"] = fp
					new_documents.append(doc)
			
			if not new_documents:
				logger.info("No modifications detected in documents. Skipping vector store update.")
				return
	   	 
			pre_update_sources = set()
			for key in list(vector_store.docstore._dict.keys())[:20]:
				source = vector_store.docstore._dict[key].metadata.get('source', 'unknown')
				pre_update_sources.add(str(source))
			logger.info(f"Pre-update document sources sample: {pre_update_sources}")
	   	 
			vector_store.add_documents(new_documents)
	   	 
			vector_store.save_local(self.persist_directory)
	   	 
			self.retriever = vector_store.as_retriever(search_kwargs={"k": self.k})
	   	 
			self.document_count = len(vector_store.docstore._dict)
	   	 
			post_update_sources = set()
			for key in list(vector_store.docstore._dict.keys())[:20]:
				source = vector_store.docstore._dict[key].metadata.get('source', 'unknown')
				post_update_sources.add(str(source))
	   	 
			# Count documents by source after update
			source_counts = {}
			for key in vector_store.docstore._dict.keys():
				source = str(vector_store.docstore._dict[key].metadata.get('source', 'unknown'))
				source_name = source.split('/')[-1] if '/' in source else source
				if source_name in source_counts:
					source_counts[source_name] += 1
				else:
					source_counts[source_name] = 1
	   	 
			logger.info(f"Updated vector store with {len(new_documents)} new documents, total: {self.document_count}")
			logger.info(f"Post-update document counts by source: {source_counts}")
	   	 
			# Re-initialize corrective_rag_handler with the updated retriever
			if self.retriever is not None:
				self.corrective_rag_handler = CorrectiveRAG(self.llm, self.retriever)
		except Exception as e:
			logger.error(f"Error updating vector store: {e}")
			self.build_vectorstore(documents)
	
	def get_document_count(self) -> int:
		return self.document_count
	
	async def setup(self) -> bool:
		logger.info("Setting up RAG pipeline")
   	 
		if self.pdf_folder and not os.path.exists(self.pdf_folder):
			logger.warning(f"PDF folder {self.pdf_folder} does not exist")
			return False
   	 
		if self.force_rebuild and os.path.exists(self.persist_directory):
			logger.info(f"Force rebuilding vector store. Deleting {self.persist_directory}")
			import shutil
			try:
				shutil.rmtree(self.persist_directory)
				logger.info(f"Successfully deleted {self.persist_directory}")
				vector_store_exists = False
			except Exception as e:
				logger.error(f"Error deleting vector store directory: {e}")
				return False
		else:
			vector_store_exists = self.load_existing_vectorstore()
   	 
		# Check for cached chunks to avoid duplicate processing
		if vector_store_exists and self.chunks_cached:
			logger.info("Document chunks already cached. Skipping document reprocessing.")
		else:
			if self.pdf_folder:
				documents = await self.load_documents()
		   	 
				if not documents:
					logger.warning("No documents loaded")
					return vector_store_exists
		   	 
				formatted_docs = self.format_documents(documents)
				document_chunks = self.prepare_documents(formatted_docs)
		   	 
				if not document_chunks:
					logger.warning("No document chunks created")
					return vector_store_exists
		   	 
				if vector_store_exists and not self.force_rebuild:
					self.update_vectorstore(document_chunks)
				else:
					self.build_vectorstore(document_chunks)
				
				# Mark chunks as cached after processing
				self.chunks_cached = True
		   	 
				# Re-initialize corrective_rag_handler with the now available retriever
				if self.retriever is not None:
					self.corrective_rag_handler = CorrectiveRAG(self.llm, self.retriever)
		   	 
		return True
   	 
		return vector_store_exists


	def rewrite_user_query(self, query: str) -> str:
		logger.info(f"Rewriting query: {query}")
   	 
		messages = [
			{"role": "system", "content": self.query_rewriting_system_prompt},
			{"role": "user", "content": self.query_rewriting_prompt.format(query=query)},
		]
   	 
		rewritten_query = self.llm.invoke(messages).content
   	 
		logger.info(f"Rewritten query: {rewritten_query}")
   	 
		return rewritten_query


	def reciprocal_rank_fusion(self, results: List[List[Document]], k: int = 60) -> List[Document]:
		logger.info(f"Performing reciprocal rank fusion on {len(results)} result sets")
		from langchain.load import dumps, loads
   	 
		fused_scores = {}
   	 
		for docs in results:
			for rank, doc in enumerate(docs):
				doc_str = dumps(doc)
				if doc_str not in fused_scores:
					fused_scores[doc_str] = 0
				fused_scores[doc_str] += 1 / (rank + k)
   	 
		reranked_results = [
			loads(doc)
			for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
		]
   	 
		logger.info(f"Reciprocal rank fusion produced {len(reranked_results)} unique documents")
		return reranked_results
	
	def generate_hypothetical_document(self, query: str) -> Document:
		logger.info(f"Generating hypothetical document for query: {query}")
		template = """You are an expert at creating hypothetical documents that would perfectly answer a user's query.
		Given the following query, generate a detailed, factual response as if you were writing a document
		that would be the ideal search result for this query. Include technical details, explanations,
		and any information that would be relevant.


		Query: {query}


		Hypothetical document:
		"""
   	 
		prompt = PromptTemplate(template=template, input_variables=["query"])
   	 
		messages = [
			{"role": "user", "content": prompt.format(query=query)},
		]
   	 
		hypothetical_content = self.llm.invoke(messages).content
   	 
		return Document(
			page_content=hypothetical_content,
			metadata={"source": "hypothetical_document", "query": query}
		)
   	 
	def retrieve_documents(self, query: str) -> List[Document]:
		logger.info(f"Retrieving documents for query: {query}")
   	 
		alternative_queries = [query]
   	 
		for i in range(2):
			alt_query = self.rewrite_user_query(f"Alternative version {i+1} of: {query}")
			if alt_query and alt_query != query:
				alternative_queries.append(alt_query)
   	 
		if self.verbose:
			logger.info(f"Generated {len(alternative_queries)} alternative queries: {alternative_queries}")
   	 
		retrieval_results = []
		for alt_query in alternative_queries:
			docs = self.retriever.get_relevant_documents(alt_query)
			retrieval_results.append(docs)
			logger.info(f"Retrieved {len(docs)} documents for query: {alt_query}")
   	 
		logger.info("Generating hypothetical document based on the original query")
		hypothetical_doc = self.generate_hypothetical_document(query)
   	 
		logger.info("Retrieving documents similar to the hypothetical document")
		hypothetical_embedding = self.embeddings.embed_documents([hypothetical_doc.page_content])[0]
   	 
		vector_store = Chroma(
			embedding_function=self.embeddings,
			persist_directory=self.persist_directory
		)
   	 
		hyde_results = vector_store.similarity_search_by_vector(hypothetical_embedding)
		retrieval_results.append(hyde_results)
   	 
		logger.info("Merging retrieval results using reciprocal rank fusion")
		fused_results = self.reciprocal_rank_fusion(retrieval_results)
   	 
		# Filter for relevant sources
		docs = fused_results[:self.k]
		logger.info(f"Filtering {len(docs)} fused results for relevance")
		docs = self.filter_relevant_sources(query, docs)
   	 
		norm_related_docs = 0
		matrix_related_docs = 0
		# Count document keywords and sources without printing full content
		for doc in docs:
			if 'norm' in doc.page_content.lower():
				norm_related_docs += 1
			if 'matrix' in doc.page_content.lower():
				matrix_related_docs += 1
   	 
		# Only log detailed document info in verbose mode
		if self.verbose:
			for i, doc in enumerate(docs[:3]):
				source = doc.metadata.get('source', 'unknown source')
				page = doc.metadata.get('page', 'unknown page')
				logger.info(f"Document {i+1}: {source} (Page: {page})")
				logger.info(f"Preview of document {i+1}:\n{doc.page_content[:200]}...\n")
	   	 
			# Count retrieved documents by source
			retrieved_sources = [doc.metadata.get('source', '') for doc in docs]
			source_counts = {}
			for source in retrieved_sources:
				source_name = str(source).split('/')[-1] if '/' in str(source) else str(source)
				if source_name in source_counts:
					source_counts[source_name] += 1
				else:
					source_counts[source_name] = 1
			logger.info(f"Retrieved document counts by source: {source_counts}")
   	 
		logger.info(f"Retrieved {len(docs)} documents")
   	 
		return docs


	def generate_answer(self, query: str, context: str, web_search_info: Dict[str, Any] = None) -> str:
		logger.info("Generating answer")
   	 
		# Add math parsing instruction to system prompt if available
		system_prompt = self.answer_generation_system_prompt
		if self.math_parsing_instruction:
			system_prompt = f"{system_prompt}\n\n{self.math_parsing_instruction}"
   	 
		# Add web search instruction to system prompt if web search was used
		if web_search_info and web_search_info.get("web_search_used", False):
			system_prompt += "\n\nParts of the context are from web search results (marked with === WEB SEARCH RESULTS === tags). Clearly indicate when you're using information from web sources versus local documents. When citing information, specify whether it comes from [LOCAL] or [WEB] sources and include the source reference when possible."
   	 
		# Prepare user prompt with context
		user_prompt = self.answer_generation_prompt.format(query=query, context=context)
   	 
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]
   	 
		answer = self.llm.invoke(messages).content
   	 
		# Process the answer to preserve LaTeX expressions
		answer = self._preserve_latex_expressions(answer)
   	 
		# Add web search notification if applicable
		if web_search_info:
			if web_search_info.get("web_search_used", False):
				# Add marker that web search results were used
				web_sources = web_search_info.get("web_sources", [])
				local_sources = web_search_info.get("local_sources", [])
				
				# Create a structured source section
				source_section = "\n\n---\n**Sources Used:**\n"
				
				# Add local sources first
				if local_sources:
					source_section += "\n**Local Documents:**\n"
					for i, source in enumerate(local_sources[:3], 1):
						source_name = source.split('/')[-1] if '/' in source else source
						source_section += f"{i}. {source_name}\n"
					if len(local_sources) > 3:
						source_section += f"...and {len(local_sources) - 3} more local documents\n"
				
				# Add web sources
				if web_sources:
					source_section += "\n**Web Sources:**\n"
					for i, source in enumerate(web_sources[:3], 1):
						source_section += f"{i}. {source}\n"
					if len(web_sources) > 3:
						source_section += f"...and {len(web_sources) - 3} more web sources\n"
					
					# Add a disclaimer about web information
					source_section += "\nWeb information may not be as reliable as the curated document collection. Please verify important information from web sources."
				
				answer += source_section
			elif web_search_info.get("web_search_needed", False) and not web_search_info.get("web_search_used", False):
				# Add marker that web search could be helpful but wasn't used
				reason = web_search_info.get("reason", "Local context was insufficient")
				answer += f"\n\n---\n**Note:** {reason}. Would you like me to search the web for more information?"
   	 
		logger.info("Answer generated")
   	 
		return answer
	
	def _preserve_latex_expressions(self, text: str) -> str:
		"""
		Preserve LaTeX expressions in the text to ensure they are properly displayed.
		This handles both inline ($...$) and display ($$...$$) LaTeX expressions.
		"""
		import re
   	 
		# Function to replace LaTeX expressions with a placeholder
		def replace_latex(match):
			return match.group(0)
   	 
		# Handle display LaTeX expressions ($$...$$)
		display_latex_pattern = r'\$\$(.*?)\$\$'
		text = re.sub(display_latex_pattern, replace_latex, text, flags=re.DOTALL)
   	 
		# Handle inline LaTeX expressions ($...$)
		inline_latex_pattern = r'\$(.*?)\$'
		text = re.sub(inline_latex_pattern, replace_latex, text, flags=re.DOTALL)
   	 
		return text
   	 
	def determine_web_search_threshold(self, query: str) -> float:
		"""
		Determine an appropriate web search threshold based on query complexity.
		Sends a prompt to the LLM asking to rate the query on a scale from 1-10.
		Then scales the output to a threshold between 0.4 and 0.8.
		"""
		logger.info(f"Determining web search threshold for query: {query}")
   	 
		complexity_prompt = f"""
		Analyze this query and rate its complexity on a scale of 1 to 10:
   	 
		Query: {query}
   	 
		Output only a number from 1-10.
		"""
   	 
		messages = [
			{"role": "system", "content": "You are an expert at analyzing query complexity."},
			{"role": "user", "content": complexity_prompt}
		]
   	 
		result = self.llm.invoke(messages).content.strip()
   	 
		try:
			complexity = float(result)
			threshold = 0.4 + (complexity / 10) * 0.4
			logger.info(f"Query complexity: {complexity}/10, dynamic threshold: {threshold:.2f}")
			if self.verbose:
				logger.info(f"Web search will be triggered if context sufficiency is below {threshold:.2f} or if key information is missing")
			return threshold
		except Exception as e:
			logger.warning(f"Could not determine dynamic threshold: {e}. Using default: {self.web_search_threshold}")
			return self.web_search_threshold  # fallback
	   	 
	def filter_relevant_sources(self, query: str, docs: List[Document]) -> List[Document]:
		"""
		Filter the provided documents using an LLM prompt to check if each document
		contains information directly relevant to the query.
		Only returns documents that are confirmed as relevant by the LLM.
		"""
		logger.info(f"Filtering relevant sources for query: {query}")
   	 
		if not docs:
			return []
	   	 
		prompt = f"""
		For the query: "{query}", list the document numbers (1-indexed) that contain information
		directly relevant to answering this query. Only output the numbers separated by commas.
   	 
		Consider that a document is relevant only if it includes key concepts, steps, or detailed
		explanations required for a complete answer.
		Also, favor using local context when sufficient details are present.
		"""
   	 
		formatted_docs = ""
		for i, doc in enumerate(docs):
			source = doc.metadata.get('source', 'unknown')
			formatted_docs += f"Document {i+1} [Source: {source}]:\n{doc.page_content[:300]}...\n\n"
	   	 
		messages = [
			{"role": "system", "content": "You are an expert at evaluating document relevance."},
			{"role": "user", "content": prompt + "\n" + formatted_docs}
		]
   	 
		result = self.llm.invoke(messages).content.strip()
   	 
		import re
		indices = [int(x) - 1 for x in re.findall(r'\d+', result) if int(x) <= len(docs)]
   	 
		if not indices:
			logger.warning("No relevant documents identified. Returning all documents.")
			return docs
	   	 
		relevant_docs = [docs[i] for i in indices]
		logger.info(f"Filtered {len(docs)} documents down to {len(relevant_docs)} relevant ones")
   	 
		return relevant_docs


	def route_query(self, query: str, context: str) -> Dict[str, Any]:
		"""
		Determines whether a query should be answered using local documents or web search.
   	 
		Args:
			query: The user's query
			context: The retrieved local context
	   	 
		Returns:
			A dictionary containing the routing decision with keys:
			- datasource: 'local' or 'web_search'
			- reason: Explanation for the routing decision
			- sufficiency_score: Score indicating how sufficient the local context is
			- missing_information: Whether key information is missing
			- query_complexity: Estimated complexity of the query
		"""
		logger.info(f"Routing query: {query}")
   	 
		# 1. Early Keyword Check - Extract important keywords from the query
		query_keywords = set(query.lower().split())
		# Filter out common stop words
		important_keywords = [word for word in query_keywords if len(word) > 3 and word not in ["what", "when", "where", "which", "whose", "whom", "that", "this", "these", "those"]]
		if important_keywords:
			matching_keywords = sum(1 for keyword in important_keywords if keyword in context.lower())
			match_ratio = matching_keywords / len(important_keywords)
			logger.info(f"Early keyword check: match_ratio = {match_ratio:.2f}")
			if match_ratio < 0.2:  # Require at least 20% of keywords to be present
				logger.info(f"Web search triggered: insufficient matching keywords in context (only {match_ratio:.0%} found)")
				return {
					"datasource": "web_search",
					"reason": f"Only {match_ratio:.0%} of query key terms found in context",
					"sufficiency_score": 0.1,
					"missing_information": True,
					"query_complexity": 0.5,
					"time_sensitive": False,
					"specialized_knowledge": True
				}
   	 
		# 2. Semantic Similarity Check - Use embeddings to compute similarity
		try:
			query_embedding = self.embeddings.embed_query(query)
			# Use the first 1000 characters of the context to compute an embedding
			context_embedding = self.embeddings.embed_query(context[:1000])
			similarity = cosine_similarity([query_embedding], [context_embedding])[0][0]
			logger.info(f"Semantic similarity between query and context: {similarity:.3f}")
			if similarity < 0.15:
				logger.info(f"Web search triggered: Query is semantically too dissimilar to the context (similarity: {similarity:.3f})")
				return {
					"datasource": "web_search",
					"reason": "Query is semantically unrelated to available context",
					"sufficiency_score": 0.2,
					"missing_information": True,
					"query_complexity": 0.5,
					"time_sensitive": False,
					"specialized_knowledge": True
				}
		except Exception as e:
			logger.warning(f"Error computing semantic similarity: {e}. Skipping this check.")
   	 
		# 3. Domain Relevance Check - Generalized to detect any out-of-domain queries
		# Define keywords that represent the primary domain (numerical analysis, programming, numerical computation)
		in_domain_keywords = [
			"algorithm", "numerical", "computation", "programming", "function", "equation",
			"matrix", "vector", "iteration", "convergence", "error", "approximation",
			"differential", "integral", "linear", "nonlinear", "newton", "euler",
			"runge-kutta", "interpolation", "extrapolation", "python", "code",
			"implementation", "method", "solution", "system", "optimization", "root",
			"eigenvalue", "derivative", "calculus", "norm", "variable", "parameter",
			"coefficient", "array", "tensor", "gradient", "jacobian", "hessian",
			"precision", "accuracy", "stability", "decomposition", "factorization",
			"orthogonal", "orthonormal", "basis", "space", "subspace", "dimension",
			"rank", "determinant", "trace", "inverse", "transpose", "symmetric",
			"hermitian", "positive", "definite", "semidefinite", "singular", "value",
			"eigenvector", "diagonalization", "triangular", "upper", "lower", "banded",
			"sparse", "dense", "condition", "number", "residual", "tolerance", "machine",
			"epsilon", "floating", "point", "arithmetic", "summation", "product", "dot",
			"cross", "inner", "outer", "kronecker", "hadamard", "frobenius", "euclidean",
			"manhattan", "infinity", "p-norm", "l1", "l2", "linf", "spectral"
		]
   	 
		# Convert query and context to lowercase for case-insensitive matching
		query_lower = query.lower()
		context_lower = context.lower()
   	 
		# Count occurrences of domain keywords in query and context
		query_domain_count = sum(1 for keyword in in_domain_keywords if keyword in query_lower)
		context_domain_count = sum(1 for keyword in in_domain_keywords if keyword in context_lower)
   	 
		# Calculate word counts (excluding common short words)
		common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
				   	"in", "on", "at", "to", "for", "with", "by", "about", "like",
				   	"through", "over", "before", "after", "since", "of", "from"}
		query_words = [word for word in query_lower.split() if word not in common_words and len(word) > 2]
		context_words = context_lower.split()
   	 
		# Calculate domain relevance ratios
		query_word_count = max(len(query_words), 1)  # Avoid division by zero
		context_word_count = max(len(context_words), 1)  # Avoid division by zero
   	 
		query_domain_ratio = query_domain_count / query_word_count
		context_domain_ratio = context_domain_count / context_word_count
   	 
		# Define thresholds for domain relevance
		query_threshold = 0.15
		context_threshold = 0.1
   	 
		# Optional: Semantic similarity check for more nuanced domain relevance assessment
		domain_semantic_score = 0.5  # Default middle value
		try:
			# Create a representative in-domain text sample
			in_domain_text = " ".join([
				"numerical analysis algorithms for solving mathematical problems",
				"computational methods for linear and nonlinear equations",
				"matrix operations and vector calculations in numerical computing",
				"programming implementations of numerical methods",
				"error analysis and convergence of numerical algorithms"
			])
	   	 
			# Compute embeddings and semantic similarity
			query_embedding = self.embeddings.embed_query(query)
			domain_embedding = self.embeddings.embed_query(in_domain_text)
			domain_semantic_score = cosine_similarity([query_embedding], [domain_embedding])[0][0]
	   	 
			logger.info(f"Domain semantic similarity score: {domain_semantic_score:.3f}")
		except Exception as e:
			logger.warning(f"Error computing domain semantic similarity: {e}. Using default score.")
   	 
		# Log the domain relevance metrics
		logger.info(f"Domain relevance - Query: {query_domain_ratio:.2f} ({query_domain_count}/{query_word_count}), Context: {context_domain_ratio:.2f} ({context_domain_count}/{context_word_count})")
   	 
		# Combined check using keyword ratios and semantic similarity
		# Trigger web search if keyword ratios are low AND semantic similarity is low
		if (query_domain_ratio < query_threshold and context_domain_ratio < context_threshold) or domain_semantic_score < 0.3:
			logger.info(f"Web search triggered: Query appears to be outside the primary domain with domain relevance ratios (query: {query_domain_ratio:.2f}, context: {context_domain_ratio:.2f}, semantic: {domain_semantic_score:.2f})")
			return {
				"datasource": "web_search",
				"reason": f"Query appears to be outside the primary domain (query relevance: {query_domain_ratio:.2f}, context relevance: {context_domain_ratio:.2f}, semantic score: {domain_semantic_score:.2f})",
				"sufficiency_score": 0.2,
				"missing_information": True,
				"query_complexity": 0.5,
				"time_sensitive": False,
				"specialized_knowledge": True
			}
   	 
		# Prompt to determine if web search is needed (existing LLM-based logic)
		prompt = """
		Analyze the following query and the retrieved context to determine if web search is needed.
   	 
		Query: {query}
   	 
		Retrieved Context: {context}
   	 
		Evaluate whether the context is sufficient to answer the query completely and accurately.
		Give preference to using local context when sufficient details are present.
		Return your analysis as a JSON object with the following structure:
		{{
			"datasource": "local" or "web_search",
			"reason": <string explaining your decision>,
			"sufficiency_score": <float between 0.0 and 1.0 indicating how sufficient the context is>,
			"missing_information": <boolean - true if key information is completely absent>,
			"query_complexity": <float between 0.0 and 1.0 indicating query complexity>,
			"time_sensitive": <boolean - true if query likely requires recent information>,
			"specialized_knowledge": <boolean - true if query requires domain expertise not likely in local docs>
		}}
   	 
		Consider:
		1. Does the context contain the necessary information to answer the query?
		2. Is the information complete and up-to-date?
		3. Does the query ask about recent events or time-sensitive information?
		4. Does the query require specialized knowledge not likely to be in the local documents?
		5. Is the query complex or multi-faceted, requiring diverse information sources?
		6. If local context is sufficiently detailed, favor local documents.
   	 
		Output only the JSON object.
		"""
   	 
		messages = [
			{"role": "system", "content": "You are an expert at determining when web search is needed to supplement local knowledge."},
			{"role": "user", "content": prompt.format(query=query, context=context[:5000])}  # Limit context size
		]
   	 
		result = self.llm.invoke(messages).content.strip()
   	 
		try:
			# Extract JSON if it's embedded in other text
			import re
			json_match = re.search(r'({.*})', result, re.DOTALL)
			if json_match:
				result = json_match.group(1)
		   	 
			routing_decision = json.loads(result)
	   	 
			# Ensure all required fields are present
			if "datasource" not in routing_decision:
				routing_decision["datasource"] = "web_search"  # Default to web search if uncertain
			if "reason" not in routing_decision:
				routing_decision["reason"] = "Failed to extract reasoning from evaluation."
			if "sufficiency_score" not in routing_decision:
				routing_decision["sufficiency_score"] = 0.5
			if "missing_information" not in routing_decision:
				routing_decision["missing_information"] = True
			if "query_complexity" not in routing_decision:
				routing_decision["query_complexity"] = 0.5
			if "time_sensitive" not in routing_decision:
				routing_decision["time_sensitive"] = False
			if "specialized_knowledge" not in routing_decision:
				routing_decision["specialized_knowledge"] = False
		   	 
			# Ensure score is within bounds
			routing_decision["sufficiency_score"] = min(max(float(routing_decision["sufficiency_score"]), 0.0), 1.0)
			routing_decision["query_complexity"] = min(max(float(routing_decision["query_complexity"]), 0.0), 1.0)
	   	 
			# Log the routing decision
			logger.info(f"Query routed to: {routing_decision['datasource']}")
			logger.info(f"Routing reason: {routing_decision['reason']}")
			logger.info(f"Context sufficiency score: {routing_decision['sufficiency_score']:.2f}")
	   	 
			return routing_decision
		except Exception as e:
			logger.warning(f"Could not parse routing decision: {e}. Raw result: {result}")
			return {
				"datasource": "web_search",  # Default to web search on error
				"reason": "Error parsing routing decision, defaulting to web search for safety",
				"sufficiency_score": 0.5,
				"missing_information": True,
				"query_complexity": 0.5,
				"time_sensitive": False,
				"specialized_knowledge": False
			}
	   	 
	def evaluate_answer(self, query: str, context: str, answer: str) -> Dict[str, float]:
		logger.info("Evaluating answer")
   	 
		context_evaluation = evaluate_context_sufficiency(self.llm, query, context)
		context_sufficiency = context_evaluation.get("sufficiency_score", 0.0)
		faithfulness = evaluate_faithfulness(self.llm, context, answer)
		completeness = evaluate_completeness(self.llm, query, answer)
		hallucination = evaluate_hallucination(self.llm, context, answer)
   	 
		evaluation_results = {
			"context_sufficiency": context_sufficiency,
			"faithfulness": faithfulness,
			"completeness": completeness,
			"hallucination": hallucination,
			"missing_information": context_evaluation.get("missing_information", True),
			"missing_details": context_evaluation.get("missing_details", True),
			"explanation": context_evaluation.get("explanation", "No explanation provided")
		}
   	 
		if self.verbose:
			logger.info(f"Evaluation results: {evaluation_results}")
   	 
		return evaluation_results


	def self_rag_evaluation(self, query: str, context: str, answer: str) -> Dict[str, Any]:
		try:
			evaluation = self.self_rag_handler.evaluate(query, context, answer)
			return evaluation
		except AttributeError as e:
			logger.warning(f'SelfRAG evaluate method not available: {e}, using fallback evaluation')
			return {
				"confidence_score": 0.5,
				"critique": "Evaluation unavailable due to missing method.",
				"improvement_suggestions": "Update SelfRAG implementation."
			}
		except Exception as e:
			logger.warning(f'Error in SelfRAG evaluation: {e}, using fallback evaluation')
			return {
				"confidence_score": 0.5,
				"critique": "Evaluation unavailable due to an error.",
				"improvement_suggestions": "Check SelfRAG implementation for errors."
			}


	def classify_query(self, query: str) -> str:
		# Call AdaptiveRAG method and extract the 'query_type' field from the returned dictionary
		classification = self.adaptive_rag_handler.classify_query(query)
		return classification.get('query_type', 'factual')


	def adjust_retrieval_parameters(self, query_type: str) -> Dict[str, Any]:
		# Create default parameters dictionary
		default_params = {"k": self.k, "score_threshold": 0.7}
		# Call the adjust_retrieval_parameters method with the query_type as a query
		# This is a workaround since we don't have the original query here
		return self.adaptive_rag_handler.adjust_retrieval_parameters(query_type, default_params)


	def corrective_rag(
		self,
		query: str,
		original_context: str,
		original_answer: str,
		evaluation_results: Dict[str, float],
		self_rag_results: Dict[str, Any]
	) -> Dict[str, Any]:
		return self.corrective_rag_handler.corrective(query, original_context, original_answer, evaluation_results, self_rag_results)


	def invoke(self, query: str) -> Dict[str, Any]:
		"""
		Process a query using the RAG pipeline with enhanced web search routing.
   	 
		Args:
			query: The user's query
	   	 
		Returns:
			A dictionary containing the answer, sources, evaluation metrics, and web search information
		"""
		logger.info("Processing query")
   	 
		# Step 1: Determine adaptive retrieval parameters based on query type
		if self.adaptive_rag:
			query_type = self.classify_query(query)
			params = self.adjust_retrieval_parameters(query_type)
			adaptive_k = params["k"]
			logger.info(f"Query type: {query_type}")
		else:
			query_type = "Default"
			adaptive_k = self.k
   	 
		# Step 2: Retrieve local documents
		docs = self.retrieve_documents(query)
		docs = docs[:adaptive_k]
		local_sources = list({doc.metadata.get('source', 'unknown') for doc in docs})
		sources = local_sources.copy()  # Initialize sources with local sources
		context = "\n\n".join([doc.page_content for doc in docs])
   	 
		# Step 3: Enhance local context if it seems insufficient
		if self.evaluate or (self.test_mode and self.enable_web_search_in_test_mode):
			# Get detailed context sufficiency evaluation
			context_evaluation = evaluate_context_sufficiency(self.llm, query, context)
			context_sufficiency = context_evaluation.get("sufficiency_score", 0.0)
			missing_information = context_evaluation.get("missing_information", True)
			missing_details = context_evaluation.get("missing_details", True)
	   	 
			logger.info(f"Initial context evaluation: score={context_sufficiency:.2f}, missing_info={missing_information}, missing_details={missing_details}")
	   	 
			if context_sufficiency < 0.6:
				logger.info("Enhancing local context with more specific retrieval...")
				specific_query = self.rewrite_user_query(f"Specific details about: {query}")
				additional_docs = self.retrieve_documents(specific_query)[:adaptive_k]
				additional_context = "\n\n".join([doc.page_content for doc in additional_docs])
				context = context + "\n\n" + additional_context
		   	 
				# Update sources list with additional documents
				for doc in additional_docs:
					source = doc.metadata.get('source', 'unknown')
					if source not in sources:
						sources.append(source)
   	 
		# Step 4: Route the query to determine if web search is needed
		# This is a key enhancement - using structured routing instead of threshold-based decision
		routing_decision = self.route_query(query, context)
   	 
		# Store local sources separately, but don't clear the sources list
		# This ensures we keep track of both local and web search sources
		if routing_decision["datasource"] == "web_search":
			logger.info("Query routed to web search - will add web search results to context")
   	 
		# Initialize web search info with routing decision and local sources
		web_search_info = {
			"web_search_needed": routing_decision["datasource"] == "web_search",
			"web_search_used": False,
			"routing_decision": routing_decision["datasource"],
			"routing_reason": routing_decision["reason"],
			"context_sufficiency": routing_decision["sufficiency_score"],
			"missing_information": routing_decision["missing_information"],
			"query_complexity": routing_decision["query_complexity"],
			"time_sensitive": routing_decision.get("time_sensitive", False),
			"specialized_knowledge": routing_decision.get("specialized_knowledge", False),
			"local_sources": local_sources  # Store local sources in web_search_info
		}
   	 
		# Step 5: Perform web search if needed (even in test mode)
		web_search_used = False
		web_search_triggered = routing_decision["datasource"] == "web_search"
   	 
		if web_search_triggered and self.web_search_enabled and self.web_search_agent is not None:
			logger.info(f"WEB SEARCH TRIGGERED: {routing_decision['reason']}")
			logger.info(f"Context sufficiency: {routing_decision['sufficiency_score']:.2f}, Missing information: {routing_decision['missing_information']}")
	   	 
			try:
				# Perform web search
				search_results = self.web_search_agent.search(query)
				logger.info(f"Web search returned {len(search_results)} results")
		   	 
				if search_results:
					# Format search results as additional context with clear markers
					web_context = self.web_search_agent.format_search_results(search_results)
			   	 
					# Enhance the markers to make the distinction clearer
					web_context = "=== WEB SEARCH RESULTS ===\n" + web_context
					if not web_context.endswith("=== END OF WEB SEARCH RESULTS ==="):
						web_context += "\n=== END OF WEB SEARCH RESULTS ==="
			   	 
					# Add a clear separator between local and web content
					local_context_with_marker = "=== LOCAL DOCUMENT RESULTS ===\n" + context + "\n=== END OF LOCAL DOCUMENT RESULTS ==="
					
					# Combine with existing context
					enhanced_context = f"{local_context_with_marker}\n\n{web_context}"
			   	 
					# Update web search info
					web_search_info["web_search_used"] = True
					web_search_info["search_results_count"] = len(search_results)
					web_search_info["web_sources"] = [result.get('link', 'web search result') for result in search_results]
			   	 
					# Use the enhanced context
					context = enhanced_context
					web_search_used = True
			   	 
					# Store web search sources separately
					web_sources = [result.get('link', 'web search result') for result in search_results]
					
					# Add web sources to the sources list (in addition to local sources)
					for source in web_sources:
						if source not in sources:
							sources.append(source)
					
					# Update web_search_info with web sources
					web_search_info["web_sources"] = web_sources
					
					logger.info(f"WEB SEARCH SUCCESSFULLY USED: Added {len(web_sources)} web search results to context")
					logger.info(f"Sources now include {len(local_sources)} local sources and {len(web_sources)} web sources")
			except Exception as e:
				logger.error(f"Error performing web search: {e}")
				web_search_info["web_search_error"] = str(e)
   	 
		# Step 6: Generate answer with web search info
		answer = self.generate_answer(query, context, web_search_info)
		evaluation_metrics = {}
	   	 
		# Step 7: Evaluate answer if needed
		if self.evaluate or (self.test_mode and self.enable_web_search_in_test_mode):
			evaluation_results = self.evaluate_answer(query, context, answer)
			self_rag_results = self.self_rag_evaluation(query, context, answer)
	   	 
			# Calculate weighted overall confidence
			overall_confidence = (
				evaluation_results.get("context_sufficiency", 0.0) * 0.25 +
				evaluation_results.get("faithfulness", 0.0) * 0.30 +
				evaluation_results.get("completeness", 0.0) * 0.20 +
				(1 - evaluation_results.get("hallucination", 0.0)) * 0.15
			) / 0.90

			# Store evaluation metrics for return
			evaluation_metrics = {
				"context_sufficiency": evaluation_results.get("context_sufficiency", 0.0),
				"faithfulness": evaluation_results.get("faithfulness", 0.0),
				"completeness": evaluation_results.get("completeness", 0.0),
				"hallucination": evaluation_results.get("hallucination", 0.0),
				"confidence": overall_confidence
			}
	   	 
			# Step 8: Apply corrective RAG if needed
			needs_correction = (
				evaluation_metrics.get("context_sufficiency", 1.0) < 0.7 or
				evaluation_metrics.get("faithfulness", 1.0) < 0.7 or
				evaluation_metrics.get("hallucination", 0.0) > 0.3 or
				evaluation_metrics.get("confidence", 1.0) < self.self_rag_threshold
			)
	   	 
			if needs_correction and self.enable_corrective_rag:
				logger.info("Applying corrective RAG")
				corrective_results = self.corrective_rag(
					query, context, answer, evaluation_results, self_rag_results
				)
		   	 
				if corrective_results.get("corrected", False):
					answer = corrective_results.get("answer", answer)
			   	 
					# Update evaluation metrics after correction
					if "evaluation" in corrective_results:
						# Calculate weighted overall confidence with corrected metrics
						overall_confidence = (
							corrective_results["evaluation"].get("context_sufficiency", 0.0) * 0.25 +
							corrective_results["evaluation"].get("faithfulness", 0.0) * 0.30 +
							corrective_results["evaluation"].get("completeness", 0.0) * 0.20 +
							(1 - corrective_results["evaluation"].get("hallucination", 0.0)) * 0.15
						) / 0.90

						evaluation_metrics = {
							"context_sufficiency": corrective_results["evaluation"].get("context_sufficiency", 0.0),
							"faithfulness": corrective_results["evaluation"].get("faithfulness", 0.0),
							"completeness": corrective_results["evaluation"].get("completeness", 0.0),
							"hallucination": corrective_results["evaluation"].get("hallucination", 0.0),
							"confidence": overall_confidence
						}
   	 
		logger.info("Answer generation completed")
   	 
		# Step 9: Return comprehensive result dictionary
		# Keep both local and web sources in the result
		if web_search_triggered and web_search_used:
			# Log the combined sources
			web_sources = web_search_info.get("web_sources", [])
			logger.info(f"Final sources list contains {len(local_sources)} local sources and {len(web_sources)} web sources")
			if self.verbose:
				logger.info(f"Local sources: {local_sources[:3]}" + ("..." if len(local_sources) > 3 else ""))
				logger.info(f"Web sources: {web_sources[:3]}" + ("..." if len(web_sources) > 3 else ""))
   	 
		result = {
			"answer": answer,
			"sources": sources,
			"local_sources": local_sources,
			"web_sources": web_search_info.get("web_sources", []) if web_search_used else [],
			"evaluation": evaluation_metrics,
			"web_search_used": web_search_used,
			"web_search_triggered": web_search_triggered,
			"web_search_info": web_search_info,
			"routing_decision": routing_decision["datasource"] if "routing_decision" in locals() else "local"
		}
   	 


		return result


	def chat(self, query: str) -> Dict[str, Any]:
		"""
		Process a query using the ConversationalAgent to maintain conversation history
		and handle follow-up questions.
   	 
		Args:
			query: The user's query
	   	 
		Returns:
			A dictionary containing the answer, sources, and other information
		"""
		logger.info("Processing query through conversational agent")
   	 
		# Use the conversational agent to process the query
		result = self.conversational_agent.query(query)
   	 
		# Return the result
		return result
	
	def create_langchain_runnable(self):
		def _rewrite_query(query):
			if self.rewrite_query:
				return self.rewrite_user_query(query)
			return query
   	 
		def _retrieve(query):
			return self.retrieve_documents(query)
   	 
		def _format_docs(docs):
			return "\n\n".join([doc.page_content for doc in docs])
   	 
		def _generate_answer(inputs):
			return self.generate_answer(inputs["query"], inputs["context"])
   	 
		def _evaluate_answer(inputs):
			if self.evaluate:
				return {
					"answer": inputs["answer"],
					"evaluation": self.evaluate_answer(
						inputs["query"], inputs["context"], inputs["answer"]
					)
				}
			return {"answer": inputs["answer"]}
   	 
		def _self_rag_evaluation(inputs):
			self_rag_results = self.self_rag_evaluation(
				inputs["query"], inputs["context"], inputs["answer"]
			)
			inputs["self_rag"] = self_rag_results
			return inputs
   	 
		def _corrective_rag(inputs):
			if self.enable_corrective_rag and self.evaluate:
				corrective_results = self.corrective_rag(
					inputs["query"],
					inputs["context"],
					inputs["answer"],
					inputs["evaluation"],
					inputs["self_rag"]
				)
		   	 
				if corrective_results.get("corrected", False):
					return {
						"query": inputs["query"],
						"context": corrective_results["context"],
						"answer": corrective_results["answer"],
						"evaluation": corrective_results["evaluation"],
						"self_rag": corrective_results["self_rag"],
						"corrective": {"corrected": True, "correction_reason": corrective_results["correction_reason"]}
					}
	   	 
			return inputs
   	 
		rewrite = RunnableLambda(_rewrite_query)
		retrieve = RunnableLambda(_retrieve)
		format_docs = RunnableLambda(_format_docs)
		generate = RunnableLambda(_generate_answer)
		evaluate = RunnableLambda(_evaluate_answer)
		self_rag = RunnableLambda(_self_rag_evaluation)
		corrective = RunnableLambda(_corrective_rag)
   	 
		chain = (
			{"query": RunnablePassthrough(), "rewritten_query": rewrite}
			| {"query": lambda x: x["query"], "context": lambda x: retrieve(x["rewritten_query"]) | format_docs}
			| {"query": lambda x: x["query"], "context": lambda x: x["context"], "answer": generate}
			| evaluate
			| self_rag
			| corrective
		)
   	 
		return chain

if __name__ == "__main__":
	import asyncio
	import os
	import sys
	
	async def test_vector_norms(agentic_rag):
		print("\n===== Testing Vector Norm Queries =====\n")
   	 
		test_queries = [
			"What is the L2 norm of a vector?",
			"Explain the difference between L1 and L2 norms",
			"How do you calculate the Frobenius norm of a matrix?",
			"What is the relationship between vector norms and distance metrics?",
			"How are norms used in regularization for machine learning?"
		]
   	 
		results = []
		for i, query in enumerate(test_queries):
			print(f"\n\n===== Query {i+1}: {query} =====")
	   	 
			print("\nRetrieving documents...")
			docs = agentic_rag.retrieve_documents(query)
	   	 
			sources = {}
			for doc in docs:
				source = doc.metadata.get('source', 'unknown')
				source_name = source.split('/')[-1] if '/' in source else source
				if source_name in sources:
					sources[source_name] += 1
				else:
					sources[source_name] = 1
	   	 
			print(f"Retrieved {len(docs)} documents")
			print(f"Documents by source: {sources}")
	   	 
			if docs:
				print(f"\nBrief document preview (from {docs[0].metadata.get('source', 'unknown')}):")
				print(f"{docs[0].page_content[:100]}...")
	   	 
			if not agentic_rag.test_mode:
				try:
					print("\nGenerating answer...")
					result = agentic_rag.invoke(query)
					print("\nAnswer:", result["answer"])
			   	 
					# Print evaluation metrics in a concise format if available
					if "evaluation" in result and result["evaluation"]:
						print("\n--- RAG Evaluation Results ---")
						eval_metrics = result["evaluation"]
						print(f"Context Sufficiency: {eval_metrics.get('context_sufficiency', 0.0):.2f}")
						print(f"Faithfulness: {eval_metrics.get('faithfulness', 0.0):.2f}")
						print(f"Completeness: {eval_metrics.get('completeness', 0.0):.2f}")
						print(f"Hallucination Score: {eval_metrics.get('hallucination', 0.0):.2f}")
						print(f"Overall Confidence: {eval_metrics.get('confidence', 0.0):.2f}")
			   	 
					results.append(result)
				except Exception as e:
					print(f"Error generating answer: {e}")
	   	 
			print("\n" + "-"*50)
   	 
		return results
	
	async def main():
		script_dir = os.path.dirname(os.path.abspath(__file__))
		default_pdf_folder = os.path.join(script_dir, "../data/")
		default_persist_dir = os.path.join(script_dir, "../chroma_db")
   	 
		print("\n=== Agentic RAG Chatbot ===\n")
		print("Initializing with document folder:", default_pdf_folder)
   	 
		# Ask if web search should be enabled
		web_search_enabled = False
		web_search_input = input("Enable web search fallback? (y/n, default: n): ")
		if web_search_input.lower() == 'y':
			web_search_enabled = True
			print("Web search fallback enabled - will search the web when local context is insufficient.")
   	 
		test_mode = False
		test_mode_input = input("Run in test mode to avoid API quota limits? (y/n, default: n): ")
		if test_mode_input.lower() == 'y':
			test_mode = True
			print("Running in test mode - API calls for answer generation will be disabled.")
   	 
		verbose_mode = False
		verbose_input = input("Enable verbose output? (y/n, default: n): ")
		if verbose_input.lower() == 'y':
			verbose_mode = True
			print("Verbose output enabled - detailed logs will be shown.")
   	 
		# Ask if conversation mode should be used
		conversation_mode = False
		conversation_input = input("Use conversation mode with history? (y/n, default: n): ")
		if conversation_input.lower() == 'y':
			conversation_mode = True
			print("Conversation mode enabled - will maintain history for follow-up questions.")
	   	 
		# Ask if web search should be enabled in test mode
		enable_web_test = False
		if test_mode and web_search_enabled:
			web_test_input = input("Enable web search in test mode? (y/n, default: n): ")
			if web_test_input.lower() == 'y':
				enable_web_test = True
				print("Web search in test mode enabled - will perform web searches even in test mode when needed.")
   	 
		agentic_rag = AgenticRetrieval(
			pdf_folder=default_pdf_folder,
			chunk_size=500,
			chunk_overlap=300,
			embedding_model="models/embedding-001",
			llm_model="gemini-2.0-flash",
			persist_directory=default_persist_dir,
			temperature=0.0,
			k=15,
			rewrite_query=True,
			evaluate=not test_mode,
			self_rag_threshold=0.7,
			adaptive_rag=True,
			enable_corrective_rag=not test_mode,
			force_rebuild=False,
			test_mode=test_mode,
			verbose=verbose_mode,
			web_search_enabled=web_search_enabled,
			web_search_threshold=0.6,
			enable_web_search_in_test_mode=enable_web_test,
			max_history_length=5
		)
   	 
		print("Setting up the RAG pipeline...")
		setup_success = await agentic_rag.setup()
   	 
		if not setup_success:
			print(f"Failed to set up RAG pipeline with folder: {agentic_rag.pdf_folder}")
			return
   	 
		print(f"\nRAG pipeline ready with {agentic_rag.document_count} documents.")
		print("\n=== Chat Interface ===")
		print("Ask any question about the documents in your data folder.")
		print("Type 'exit' to quit.")
   	 
		while True:
			query = input("\nYou: ")
	   	 
			if query.lower() in ["exit", "quit", "q"]:
				print("\nGoodbye!")
				break
	   	 
			if not query:
				continue
	   	 
			print("\nProcessing...")
			try:
				# Use conversation mode if enabled
				if conversation_mode:
					result = agentic_rag.chat(query)
					print(f"\nChatbot: {result['answer']}")
			   	 
					# Print if this was a follow-up question
					if "is_followup" in result and result["is_followup"]:
						print("\n[Detected as a follow-up question]")
			   	 
					# Print if the query was rewritten
					if "processed_query" in result and result["processed_query"] != query:
						print(f"\n[Query was rewritten to: {result['processed_query']}]")
			   	 
					# Print sources in a concise format
					if "sources" in result and result["sources"]:
						print("\nSources:")
						for i, source in enumerate(result["sources"][:5], 1):
							source_name = source.split('/')[-1] if '/' in source else source
							print(f"  {i}. {source_name}")
						if len(result["sources"]) > 5:
							print(f"  ... and {len(result['sources']) - 5} more")
				elif agentic_rag.test_mode:
					print("Processing query in test mode...")
					# Use the invoke() method to ensure consistent routing logic in test mode
					# This ensures that web search fallback is triggered appropriately for out-of-domain queries
					result = agentic_rag.invoke(query)
					print(f"\nChatbot: {result['answer']}")
			   	 
					# Print if web search was triggered or used
					if "web_search_triggered" in result and result["web_search_triggered"]:
						print(f"\n[Web search was triggered: {result.get('web_search_info', {}).get('routing_reason', 'Unknown reason')}]")
				   	 
						if "web_search_used" in result and result["web_search_used"]:
							print("\n[Web search results were used to enhance the answer]")
			   	 
					# Print sources in a concise format
					if "sources" in result and result["sources"]:
						print("\nSources:")
						for i, source in enumerate(result["sources"][:5], 1):
							source_name = source.split('/')[-1] if '/' in source else source
							print(f"  {i}. {source_name}")
						if len(result["sources"]) > 5:
							print(f"  ... and {len(result['sources']) - 5} more")
			   	 
					# Print evaluation metrics if available
					if "evaluation" in result and result["evaluation"]:
						print("\n--- RAG Evaluation Results ---")
						eval_metrics = result["evaluation"]
						print(f"Context Sufficiency: {eval_metrics.get('context_sufficiency', 0.0):.2f}")
						print(f"Faithfulness: {eval_metrics.get('faithfulness', 0.0):.2f}")
						print(f"Completeness: {eval_metrics.get('completeness', 0.0):.2f}")
						print(f"Hallucination Score: {eval_metrics.get('hallucination', 0.0):.2f}")
						print(f"Overall Confidence: {eval_metrics.get('confidence', 0.0):.2f}")
				else:
					result = agentic_rag.invoke(query)
					print(f"\nChatbot: {result['answer']}")
			   	 
					# Print evaluation metrics in a concise format if available
					if "evaluation" in result and result["evaluation"]:
						print("\n--- RAG Evaluation Results ---")
						eval_metrics = result["evaluation"]
						print(f"Context Sufficiency: {eval_metrics.get('context_sufficiency', 0.0):.2f}")
						print(f"Faithfulness: {eval_metrics.get('faithfulness', 0.0):.2f}")
						print(f"Completeness: {eval_metrics.get('completeness', 0.0):.2f}")
						print(f"Hallucination Score: {eval_metrics.get('hallucination', 0.0):.2f}")
						print(f"Overall Confidence: {eval_metrics.get('confidence', 0.0):.2f}")
			   	 
					# Print if web search was used
					if "web_search_used" in result and result["web_search_used"]:
						print("\n[Web search results were used to enhance the answer]")
			   	 
					# Print sources in a concise format
					if "sources" in result and result["sources"]:
						print("\nSources:")
						for i, source in enumerate(result["sources"][:5], 1):
							source_name = source.split('/')[-1] if '/' in source else source
							print(f"  {i}. {source_name}")
						if len(result["sources"]) > 5:
							print(f"  ... and {len(result['sources']) - 5} more")
			except Exception as e:
				print(f"\nError: {e}")
	
	asyncio.run(main())