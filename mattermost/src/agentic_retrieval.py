from .final_decorator import finalclass
import logging
import os
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dotenv import load_dotenv

from .agents.conversational import ConversationalAgent
from .agents.websearch import WebSearchAgent

from .rag.retrieval import RAG_Pipeline
from .rag.self_retrieval import SelfRAG
from .rag.corrective_retrieval import CorrectiveRAG
from .rag.adaptive_retrieval import AdaptiveRAG

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from pypdf import PdfReader
import re

from .semantic_cache import SemanticCache
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from .jupyter_parser import JupyterParser
from .youtube_processor import YouTubeProcessor

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

Your responses should:
1. Match the style and complexity level of the lecture notes used in the course: Foundations of Numerical Analysis and Exercises in Numerical Analysis
2. Use clear, concise explanations without unnecessary technical jargon
3. Include relevant examples when helpful
4. Break down complex concepts into digestible parts
5. Focus on fundamental understanding rather than advanced details
6. Use mathematical notation only when necessary and with clear explanation

For mathematical and technical content:
1. Present definitions, theorems, and formulas with precision but clarity
2. Use clear notation and formatting for mathematical expressions
3. Explain technical concepts in a structured and logical manner
4. Format mathematical expressions using LaTeX: use double dollar signs ($$...$$) for display equations and single dollar signs ($...$) for inline equations
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
		embedding_model: str = "text-embedding-3-small",  
		llm_model: str = "gpt-3.5-turbo",  
		persist_directory: str = "./chroma_db",
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
		web_search_engine: str = 'openai',
		enable_web_search_in_test_mode: bool = False,
		max_history_length: int = 5,
		cache_enabled: bool = True,
		cache_similarity_threshold: float = 0.9,
		max_cache_size: int = 1000,
	):
		os.environ['USER_AGENT'] = user_agent
		
		# Set default math parsing instruction if not provided
		if math_parsing_instruction is None:
			math_parsing_instruction = "Format all mathematical expressions using LaTeX notation. Enclose display equations in double dollar signs ($$...$$) and inline equations in single dollar signs ($...$)."
   	 
		# Load environment variables from the parent directory
		env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
		load_dotenv(env_path)
   	 
		openai_api_key = os.getenv("OPENAI_API_KEY")
		if not openai_api_key:
			raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
   	 
		self.pdf_folder = pdf_folder
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap
		self.embedding_model = embedding_model
		self.llm_model = llm_model
		self.persist_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")  # Fix the path
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
		self.cache_enabled = cache_enabled
		if self.cache_enabled:
			self.semantic_cache = SemanticCache(
				cache_file=os.path.join(os.path.dirname(__file__), "semantic_cache.json"),
				similarity_threshold=0.95,  # Increased from 0.9 to 0.95 for stricter matching
				max_cache_size=max_cache_size,
				embedding_model=self.embedding_model,
				vectorstore_path=self.persist_directory  # Pass the vectorstore path
			)
		else:
			self.semantic_cache = None
   	 
		# Initialize OpenAI models
		self.llm = ChatOpenAI(
			model_name=self.llm_model,
			temperature=self.temperature,
			openai_api_key=openai_api_key
		)
		logger.info(f"Initializing embeddings with model: {self.embedding_model}")
		self.embeddings = OpenAIEmbeddings(
			model=self.embedding_model,
			openai_api_key=openai_api_key
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

		# Initialize components
		self.parser = LlamaParse(result_type="text", system_prompt_append=self.math_parsing_instruction)
		self.jupyter_parser = JupyterParser(include_outputs=True, include_raw_code=True)
		
		# Update file extractor configuration
		self.file_extractor = {
			".pdf": self.parser,
			".ipynb": lambda x: self.jupyter_parser.parse(x)  # Wrap in lambda to ensure proper method binding
		}
		
		# Initialize SimpleDirectoryReader with the file extractor
		self.reader = SimpleDirectoryReader(
			input_dir=self.pdf_folder,
			file_extractor=self.file_extractor
		)
		
		self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
			chunk_size=self.chunk_size,
				chunk_overlap=self.chunk_overlap,
		)

		self.youtube_processor = YouTubeProcessor()

	async def load_documents(self) -> List[Document]:
		logger.info(f"Loading documents from {self.pdf_folder}")
		
		if not self.pdf_folder or not os.path.exists(self.pdf_folder):
			logger.warning(f"PDF folder {self.pdf_folder} does not exist")
			return []
		
		try:
			# First, try to identify PowerPoint-converted PDFs
			ppt_pdfs = []
			regular_pdfs = []
			
			for file in os.listdir(self.pdf_folder):
				if file.endswith('.pdf'):
					file_path = os.path.join(self.pdf_folder, file)
					# Simple heuristic to identify PowerPoint-converted PDFs
					# They often have "Slide" in the text or specific formatting
					try:
						reader = PdfReader(file_path)
						first_page = reader.pages[0].extract_text()
						if "Slide" in first_page or "PowerPoint" in first_page:
							ppt_pdfs.append(file_path)
						else:
							regular_pdfs.append(file_path)
					except Exception as e:
						logger.warning(f"Error checking PDF type for {file}: {e}")
						regular_pdfs.append(file_path)
			
			documents = []
			
			# Load PowerPoint-converted PDFs with our custom loader
			for ppt_pdf in ppt_pdfs:
				try:
					loader = PowerPointPDFLoader(ppt_pdf)
					docs = loader.load()
					documents.extend(docs)
					logger.info(f"Loaded {len(docs)} pages from PowerPoint-converted PDF: {ppt_pdf}")
				except Exception as e:
					logger.error(f"Error loading PowerPoint-converted PDF {ppt_pdf}: {e}")
			
			# Load regular PDFs with PyPDFLoader
			if regular_pdfs:
				loader = DirectoryLoader(
					self.pdf_folder,
					glob="**/*.pdf",
					loader_cls=PyPDFLoader,
					use_multithreading=True
				)
				docs = loader.load()
				documents.extend(docs)
				logger.info(f"Loaded {len(docs)} pages from regular PDFs")
			
			logger.info(f"Total documents loaded: {len(documents)}")
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
		
		self.vectorstore = Chroma.from_documents(
			documents=documents,
			embedding=self.embeddings,
			persist_directory=self.persist_directory
		)
		self.vectorstore.persist()
		self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
		
		self.document_count = len(documents)
		
		logger.info(f"Built vector store with {self.document_count} documents")
	
	def load_existing_vectorstore(self) -> bool:
		logger.info(f"Loading existing vector store from {self.persist_directory}")
		
		try:
			if os.path.exists(self.persist_directory):
				self.vectorstore = Chroma(
					embedding_function=self.embeddings,
					persist_directory=self.persist_directory
				)
				self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
				self.document_count = len(self.vectorstore.get()['ids'])
				logger.info(f"Loaded vector store with {self.document_count} documents")
				
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
		"""Update the vectorstore with new documents"""
		logger.info("Updating vector store")
		
		if not documents:
			logger.warning("No documents to update vector store")
			return
		
		try:
			# Create a set of existing fingerprints
			existing_fingerprints = set()
			if self.vectorstore:
				existing_docs = self.vectorstore.get()
				for doc in existing_docs['documents']:
					try:
						import hashlib
						fp = hashlib.sha256(doc.encode('utf-8')).hexdigest()
						existing_fingerprints.add(fp)
					except Exception:
						pass
			
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
			
			if self.vectorstore:
				self.vectorstore.add_documents(new_documents)
				self.vectorstore.persist()
			else:
				self.build_vectorstore(new_documents)
			
			self.document_count = len(self.vectorstore.get()['ids'])
			logger.info(f"Updated vector store with {len(new_documents)} new documents, total: {self.document_count}")
			
			# Clear the semantic cache when vectorstore is updated
			if self.cache_enabled and self.semantic_cache:
				logger.info("Clearing semantic cache due to vectorstore update")
				self.semantic_cache.clear()
			
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
		   	 
		# Create the RAG pipeline if we have a retriever
		if self.retriever is not None:
			self.create_rag_pipeline()
			logger.info("RAG pipeline created successfully")
		else:
			logger.warning("No retriever available. RAG pipeline not created.")
			return False
   	 
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
		
		# Store unique documents using their content hash
		seen_docs = {}
		retrieval_results = []
		
		for alt_query in alternative_queries:
			docs = self.retriever.get_relevant_documents(alt_query)
			# Deduplicate documents based on content
			unique_docs = []
			for doc in docs:
				try:
					import hashlib
					content_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
					if content_hash not in seen_docs:
						seen_docs[content_hash] = doc
						unique_docs.append(doc)
				except Exception as e:
					logger.warning(f"Error hashing document content: {e}")
					unique_docs.append(doc)
			
			if unique_docs:  # Only add if we found unique documents
				retrieval_results.append(unique_docs)
				logger.info(f"Retrieved {len(unique_docs)} unique documents for query: {alt_query}")
		
		logger.info("Generating hypothetical document based on the original query")
		hypothetical_doc = self.generate_hypothetical_document(query)
		
		logger.info("Retrieving documents similar to the hypothetical document")
		hypothetical_embedding = self.embeddings.embed_documents([hypothetical_doc.page_content])[0]
		
		vector_store = Chroma(
			embedding_function=self.embeddings,
			persist_directory=self.persist_directory
		)
		
		hyde_results = vector_store.similarity_search_by_vector(hypothetical_embedding)
		# Deduplicate HYDE results
		unique_hyde_results = []
		for doc in hyde_results:
			try:
				content_hash = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()
				if content_hash not in seen_docs:
					seen_docs[content_hash] = doc
					unique_hyde_results.append(doc)
			except Exception as e:
				logger.warning(f"Error hashing HYDE document content: {e}")
				unique_hyde_results.append(doc)
		
		if unique_hyde_results:  # Only add if we found unique documents
			retrieval_results.append(unique_hyde_results)
		
		logger.info("Merging retrieval results using reciprocal rank fusion")
		fused_results = self.reciprocal_rank_fusion(retrieval_results)
		
		# Filter for relevant sources and deduplicate one final time
		docs = fused_results[:self.k]
		logger.info(f"Filtering {len(docs)} fused results for relevance")
		
		# Final deduplication of sources
		unique_sources = {}
		for doc in docs:
			source = doc.metadata.get('source', 'unknown')
			if source not in unique_sources:
				unique_sources[source] = doc
		
		final_docs = list(unique_sources.values())
		
		# Only log detailed document info in verbose mode
		if self.verbose:
			for i, doc in enumerate(final_docs[:3]):
				source = doc.metadata.get('source', 'unknown source')
				page = doc.metadata.get('page', 'unknown page')
				logger.info(f"Document {i+1}: {source} (Page: {page})")
				logger.info(f"Preview of document {i+1}:\n{doc.page_content[:200]}...\n")
			
			# Count unique sources
			source_counts = {}
			for doc in final_docs:
				source = str(doc.metadata.get('source', 'unknown'))
				source_name = source.split('/')[-1] if '/' in source else source
				if source_name in source_counts:
					source_counts[source_name] += 1
				else:
					source_counts[source_name] = 1
			logger.info(f"Retrieved document counts by unique source: {source_counts}")
		
		logger.info(f"Retrieved {len(final_docs)} unique documents")
		
		return final_docs


	def generate_answer(self, query: str, context: str, web_search_info: Dict[str, Any] = None) -> str:
		logger.info("Generating answer")
		
		# Add style guidance to system prompt
		system_prompt = self.answer_generation_system_prompt
		
		# Add math parsing instruction to system prompt if available
		if self.math_parsing_instruction:
			system_prompt = f"{system_prompt}\n\n{self.math_parsing_instruction}"
		
		# Add web search instruction to system prompt if web search was used
		if web_search_info and web_search_info.get("web_search_used", False):
			system_prompt += "\n\nParts of the context are from web search results (marked with === WEB SEARCH RESULTS === tags). Clearly indicate when you're using information from web sources versus local documents. When citing information, specify whether it comes from [LOCAL] or [WEB] sources and include the source reference when possible."
		
		# Prepare user prompt with context
		user_prompt = f"""
		Based on the following context, provide a clear and concise answer that matches the style of lecture notes.
		Focus on fundamental understanding and avoid unnecessary complexity.
		
		Query: {query}
		
		Context: {context}
		
		Remember to:
		1. Use clear, simple language
		2. Break down complex concepts
		3. Include examples if helpful
		4. Use mathematical notation sparingly and clearly
		5. Match the level of detail found in typical lecture notes
		6. PRESERVE ALL LaTeX FORMATTING (both inline and display math) exactly as provided
		7. Use $$...$$ for ALL mathematical expressions (both inline and display math)
		8. When using information from web search results, clearly indicate this in the response
		9. Do not include local document sources if the information comes from web search results
		"""
		
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
			]
		
		answer = self.llm.invoke(messages).content
		
		# Process the answer to preserve LaTeX expressions
		answer = self._preserve_latex_expressions(answer)
		
		# Add web search notification if applicable
		if web_search_info and web_search_info.get("web_search_used", False):
			# Only include web sources, not local sources
			web_sources = web_search_info.get("web_sources", [])
			
			# Create source section
			source_section = "\n\n---\n**Sources Used:**\n"
			
			# Add web sources
			if web_sources:
				source_section += "\n".join([f"- {source}" for source in web_sources])
			
			answer += source_section
			
			# Add web search notification
			answer += "\n\n*Note: This answer includes information from web search results.*"
		
		logger.info("Answer generated")
		
		return answer
	
	def _preserve_latex_expressions(self, text: str) -> str:
		"""
		Preserve LaTeX expressions in the text to ensure they are properly displayed.
		This handles both inline and display LaTeX expressions using $$...$$ format.
		"""
		import re
		
		# Function to replace LaTeX expressions with a placeholder
		def replace_latex(match):
			return match.group(0)
		
		# Handle display LaTeX expressions ($$...$$)
		display_latex_pattern = r'\$\$(.*?)\$\$'
		text = re.sub(display_latex_pattern, replace_latex, text, flags=re.DOTALL)
		
		# Handle inline LaTeX expressions ($$...$$)
		inline_latex_pattern = r'\$\$(.*?)\$\$'
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
		Your task is to decide how to best answer a user's question. The process begins by first considering the subject matter of the query. 
		Specifically, you need to determine if the question falls within the domain of numerical analysis. If, upon examination, you find that 
		the query is not related to numerical analysis, then the appropriate course of action is to perform a web search to find the answer. 
		However, if you determine that the query does indeed pertain to the field of numerical analysis, the next step is to consult a local 
		vector store containing relevant documents. Within this vector store, you must search for contextual information that can directly 
		answer the user's question. If, after searching the vector store, no suitable context is found, you should return a message indicating 
		that the query is out of scope for the local documents. Conversely, if relevant contextual information is successfully retrieved from 
		the vector store, you should then use this local information to formulate your answer and return the local result to the user.
		
		Args:
			query: The user's query (string)
			context: The retrieved local context (string)
		
		Returns:
			A dictionary containing the routing decision with the following keys:
			- datasource: Either "web_search", "local", or "out_of_scope"
			- reason: Explanation of the routing decision
			- sufficiency_score: Score indicating how well the context matches the query
			- missing_information: Boolean indicating if key information is missing
			- query_complexity: Score indicating the complexity of the query
			- needs_professor_assistance: Boolean indicating if professor assistance is needed
		"""
		logger.info(f"Routing query: {query}")

		# Ensure query is a string
		if not isinstance(query, str):
			if isinstance(query, dict):
				query = query.get('text', '') or query.get('query', '')
			query = str(query).strip()

		if not query:
			return {
				"datasource": "local",
				"reason": "Empty or invalid query",
				"sufficiency_score": 0.0,
				"missing_information": True,
				"query_complexity": 0.0,
				"needs_professor_assistance": False
			}

		# Define numerical analysis domain keywords
		domain_keywords = [
			"numerical", "analysis", "algorithm", "computation", "error", "approximation",
			"matrix", "vector", "iteration", "convergence", "differential", "integral",
			"linear", "nonlinear", "newton", "euler", "runge-kutta", "interpolation",
			"optimization", "eigenvalue", "norm", "derivative", "calculus", "numerical method",
			"numerical solution", "numerical integration", "numerical differentiation",
			"finite difference", "finite element", "numerical stability", "rounding error",
			"truncation error", "numerical accuracy", "numerical precision"
		]
		
		query_lower = query.lower()
		is_domain_related = any(keyword in query_lower for keyword in domain_keywords)
		
		if not is_domain_related:
			logger.info("Query is not related to numerical analysis domain")
			return {
				"datasource": "web_search",
				"reason": "Query is not related to numerical analysis domain",
				"sufficiency_score": 0.0,
				"missing_information": True,
				"query_complexity": 0.0,
				"needs_professor_assistance": False
			}

		# For domain-related queries, check if we have specific technical content
		technical_indicators = [
			r'\$\$.*?\$\$',  # LaTeX display math
			r'\$.*?\$',      # LaTeX inline math
			r'equation', r'formula', r'theorem', r'proof', r'definition',
			r'algorithm', r'method', r'technique', r'procedure',
			r'example', r'problem', r'solution', r'derivation'
		]
		
		has_technical_content = any(
			re.search(pattern, context, re.IGNORECASE) 
			for pattern in technical_indicators
		)
		
		if not has_technical_content:
			logger.info("No technical content found in local documents")
			return {
				"datasource": "out_of_scope",
				"reason": "This topic is currently out of scope for the course materials",
				"sufficiency_score": 0.0,
				"missing_information": True,
				"query_complexity": 0.0,
				"needs_professor_assistance": True
			}

		# If we have technical content, perform semantic search
		try:
			query_embedding = self.embeddings.embed_query(query)
			context_embedding = self.embeddings.embed_query(context[:1000])  # First 1000 chars for efficiency
			similarity = cosine_similarity([query_embedding], [context_embedding])[0][0]
			logger.info(f"Semantic similarity between query and context: {similarity:.3f}")
			
			# Use a higher threshold for technical content
			if similarity > 0.7:  # Increased threshold for technical content
				logger.info(f"Strong semantic match found in technical documents (similarity: {similarity:.3f})")
				return {
					"datasource": "local",
					"reason": f"Found relevant technical content in course documents (similarity: {similarity:.3f})",
					"sufficiency_score": similarity,
					"missing_information": False,
					"query_complexity": 0.5,
					"needs_professor_assistance": False
				}
			else:
				return {
					"datasource": "out_of_scope",
					"reason": "This specific topic is not covered in sufficient detail in the course materials",
					"sufficiency_score": similarity,
					"missing_information": True,
					"query_complexity": 0.5,
					"needs_professor_assistance": True
				}
		except Exception as e:
			logger.warning(f"Error computing semantic similarity: {e}. Falling back to out of scope.")
			return {
				"datasource": "out_of_scope",
				"reason": "Unable to determine if this topic is covered in the course materials",
				"sufficiency_score": 0.0,
				"missing_information": True,
				"query_complexity": 0.5,
				"needs_professor_assistance": True
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


	async def invoke(self, query: str, video_url: str = None) -> Dict[str, Any]:
		"""
		Process a query, now with optional YouTube video support.
		
		Args:
			query: The user's query
			video_url: Optional YouTube video URL
			
		Returns:
			Dictionary containing the answer and other metadata
		"""
		try:
			# If video URL is provided, process as a YouTube query
			if video_url:
				result = self.process_youtube_query(query, video_url)
				return {
					'answer': result['answer'],
					'sources': [result['video_url']],
					'web_search_used': False,
					'web_search_triggered': False,
					'evaluation': None,
					'needs_professor_assistance': False
				}
			
			# First, check if query is out of domain
			classification = self.adaptive_rag_handler.classify_query(query)
			query_type = classification.get('query_type', 'factual')
			is_out_of_domain = classification.get('is_out_of_domain', False)
			
			logger.info(f"Query classification: {query_type}, Out of domain: {is_out_of_domain}")
			
			# If query is out of domain and web search is enabled, use web search directly
			if is_out_of_domain and self.web_search_enabled:
				logger.info("Query is out of domain, using web search directly")
				if hasattr(self, 'web_search_agent') and self.web_search_agent and (not self.test_mode or self.enable_web_search_in_test_mode):
					web_results = await self.web_search_agent.search(query)
					if isinstance(web_results, dict):  # Check if web_results is a dictionary
						return {
							'answer': web_results.get('answer', 'No answer found'),
							'sources': web_results.get('sources', []),
							'web_search_used': True,
							'web_search_triggered': True,
							'web_search_info': {
								'routing_reason': 'Query is out of domain',
								'web_sources': web_results.get('sources', []),
								'local_sources': []
							},
							'evaluation': {
								'context_sufficiency': 1.0,
								'faithfulness': 1.0,
								'completeness': 1.0,
								'hallucination': 0.15,  # Base hallucination score for web results
								'confidence': 0.85
							}
						}
					else:
						logger.error(f"Unexpected web search results format: {type(web_results)}")
						return {
							'answer': "Sorry, I encountered an error processing web search results.",
							'sources': [],
							'web_search_used': False,
							'web_search_triggered': True,
							'evaluation': None
						}
				else:
					return {
						'answer': "I apologize, but this query appears to be outside my domain of expertise (numerical analysis and related topics). I can only provide accurate information about topics covered in the lecture notes and related mathematical concepts.",
						'sources': [],
						'web_search_used': False,
						'web_search_triggered': False,
						'evaluation': {
							'context_sufficiency': 0.0,
							'faithfulness': 1.0,
							'completeness': 0.0,
							'hallucination': 0.0,
							'confidence': 1.0
						}
					}

			# For in-domain queries or when web search is disabled, proceed with normal RAG pipeline
			if self.rewrite_query:
				rewritten_query = self.rewrite_user_query(query)
				logger.info(f"Rewritten query: {rewritten_query}")
			else:
				rewritten_query = query

			# Retrieve relevant documents
			docs = self.retrieve_documents(rewritten_query)
			
			if not docs:
				logger.warning("No relevant documents found")
				if self.web_search_enabled and hasattr(self, 'web_search_agent') and self.web_search_agent and (not self.test_mode or self.enable_web_search_in_test_mode):
					logger.info("Falling back to web search due to no relevant documents")
					web_results = await self.web_search_agent.search(query)
					if isinstance(web_results, dict):
						return {
							'answer': web_results.get('answer', 'No answer found'),
							'sources': web_results.get('sources', []),
							'web_search_used': True,
							'web_search_triggered': True,
							'web_search_info': {
								'routing_reason': 'No relevant local documents found',
								'web_sources': web_results.get('sources', []),
								'local_sources': []
							}
						}
					else:
						return {
							'answer': "Sorry, I encountered an error processing web search results.",
							'sources': [],
							'web_search_used': False,
							'web_search_triggered': True
						}
				else:
					return {
						'answer': "I couldn't find any relevant information to answer your question.",
						'sources': [],
						'web_search_used': False,
						'web_search_triggered': False
					}

			# Format context from retrieved documents
			context = "\n\n".join([doc.page_content for doc in docs])
			
			# Route the query to determine if web search is needed
			routing_decision = self.route_query(query, context)
			
			# If web search is needed and enabled
			if (routing_decision['datasource'] == 'web_search' and 
				self.web_search_enabled and 
				hasattr(self, 'web_search_agent') and 
				self.web_search_agent and 
				(not self.test_mode or self.enable_web_search_in_test_mode)):
				
				web_results = await self.web_search_agent.search(query)
				if not isinstance(web_results, dict):
					logger.error(f"Unexpected web search results format: {type(web_results)}")
					web_results = {'answer': 'No answer found', 'context': '', 'sources': []}
				
				# Combine web results with local context
				combined_context = f"{context}\n\n=== WEB SEARCH RESULTS ===\n\n{web_results.get('context', '')}"
				
				# Generate answer using combined context
				answer = self.generate_answer(
					query, 
					combined_context,
					web_search_info={
						'web_search_used': True,
						'web_sources': web_results.get('sources', []),
						'local_sources': [doc.metadata.get('source', 'unknown') for doc in docs]
					}
				)
				
				return {
					'answer': answer,
					'sources': web_results.get('sources', []) + [doc.metadata.get('source', 'unknown') for doc in docs],
					'web_search_used': True,
					'web_search_triggered': True,
					'web_search_info': {
						'routing_reason': routing_decision['reason'],
						'web_sources': web_results.get('sources', []),
						'local_sources': [doc.metadata.get('source', 'unknown') for doc in docs]
					}
				}
			else:
				# Generate answer using only local context
				answer = self.generate_answer(query, context)
				
				# Evaluate the answer if enabled
				if self.evaluate and not self.test_mode:
					evaluation = self.evaluate_answer(query, context, answer)
				else:
					evaluation = None
				
				return {
					'answer': answer,
					'sources': [doc.metadata.get('source', 'unknown') for doc in docs],
					'web_search_used': False,
					'web_search_triggered': False,
					'evaluation': evaluation
				}
				
		except Exception as e:
			logger.error(f"Error processing query: {str(e)}", exc_info=True)
			return {
				'answer': f"Error processing query: {str(e)}",
				'sources': [],
				'web_search_used': False,
				'web_search_triggered': False,
				'evaluation': None
			}

	def process_youtube_query(self, query: str, video_url: str) -> Dict[str, Any]:
		"""
		Process a query specifically for a YouTube video.
		
		Args:
			query: User's query about the video content
			video_url: URL of the YouTube video
			
		Returns:
			Dictionary containing answer and relevant video segments
		"""
		try:
			# Process the video and find relevant segments
			segments, error = self.youtube_processor.process_video(video_url, query)
			
			if error:
				return {
					'answer': f"Error processing video: {error}",
					'segments': [],
					'video_url': video_url
				}
			
			# Format the response
			response = []
			for segment in segments:
				timestamp_link = f"{video_url}{segment['url_time']}"
				response.append(f"At {segment['timestamp']}: {segment['text']} [Link]({timestamp_link})")
			
			answer = "\n\nRelevant segments from the video:\n\n" + "\n\n".join(response)
			
			return {
				'answer': answer,
				'segments': segments,
				'video_url': video_url
			}
				
		except Exception as e:
			logger.error(f"Error processing YouTube query: {e}")
			return {
				'answer': f"Error processing video: {str(e)}",
				'segments': [],
				'video_url': video_url
			}

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

class PowerPointPDFLoader:
    """Custom PDF loader optimized for PowerPoint-converted PDFs"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load(self) -> List[Document]:
        """Load and process the PDF file"""
        documents = []
        reader = PdfReader(self.file_path)
        
        for page_num, page in enumerate(reader.pages):
            # Extract text from the page
            text = page.extract_text()
            
            # Clean and format the text
            text = self._clean_text(text)
            
            # Create a document with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": self.file_path,
                    "page": page_num + 1,
                    "file_type": "powerpoint_pdf"
                }
            )
            documents.append(doc)
            
        return documents
    
    def _clean_text(self, text: str) -> str:
        """Clean and format the text from PowerPoint-converted PDFs"""
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\d+/\d+', '', text)
        
        # Remove bullet points and numbering
        text = re.sub(r'^[\s•\-\d\.]+', '', text, flags=re.MULTILINE)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

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
		default_persist_dir = os.path.join(script_dir, "./chroma_db")
   	 
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
			embedding_model="text-embedding-3-small",  # OpenAI embedding model
			llm_model="gpt-3.5-turbo",  # OpenAI model
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
			max_history_length=5,
			cache_enabled=True,
			cache_similarity_threshold=0.9,
			max_cache_size=1000,
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
					result = await agentic_rag.invoke(query)
					if result:  # Add null check
						print(f"\nChatbot: {result.get('answer', 'No answer available')}")
						
						# Print if web search was triggered or used
						if result.get("web_search_triggered"):
							print(f"\n[Web search was triggered: {result.get('web_search_info', {}).get('routing_reason', 'Unknown reason')}]")
							
							if result.get("web_search_used"):
								print("\n[Web search results were used to enhance the answer]")
						
						# Print sources in a concise format
						sources = result.get("sources", [])
						if sources:
							print("\nSources:")
							for i, source in enumerate(sources[:5], 1):
								source_name = source.split('/')[-1] if '/' in source else source
								print(f"  {i}. {source_name}")
							if len(sources) > 5:
								print(f"  ... and {len(sources) - 5} more")
						
						# Print evaluation metrics if available
						evaluation = result.get("evaluation")
						if evaluation:
							print("\n--- RAG Evaluation Results ---")
							print(f"Context Sufficiency: {evaluation.get('context_sufficiency', 0.0):.2f}")
							print(f"Faithfulness: {evaluation.get('faithfulness', 0.0):.2f}")
							print(f"Completeness: {evaluation.get('completeness', 0.0):.2f}")
							print(f"Hallucination Score: {evaluation.get('hallucination', 0.0):.2f}")
							print(f"Overall Confidence: {evaluation.get('confidence', 0.0):.2f}")
					else:
						print("\nError: No response received")
			except Exception as e:
				print(f"\nError: {e}")
	
	asyncio.run(main())