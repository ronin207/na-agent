import bs4
import os
import asyncio
import nest_asyncio
import requests
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from operator import itemgetter


from langchain import hub
from langchain import text_splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader


from dotenv import load_dotenv
from pathlib import Path


class RAG_Pipeline:
   """
   A class that encapsulates the document retrieval process for RAG applications.
  
   This class handles document loading, text splitting, vector store creation,
   and implements the reciprocal rank fusion algorithm for improved retrieval.
   """


   def __init__(
       self,
       pdf_folder: str = "./rag_llm/data/",  # Default path will be resolved relative to script location
       chunk_size: int = 1000, # modify the size if the queries are too slow
       chunk_overlap: int = 200, # modify the overlap to ensure that contexts are not missed
       embedding_model: str = "models/embedding-001",
       llm_model: str = "gemini-2.0-flash",
       persist_directory: str = "./chroma_db",
       temperature: float = 0,
       user_agent: str = "myagent",
       math_parsing_instruction: str = "Output any math equation or formula in LaTeX format enclosed in double dollar signs ($$). Preserve all mathematical notation exactly as written. Ensure all variables, symbols, and equations maintain their original formatting."
   ):
       """
       Initialize the Retriever with configuration parameters.
      
       Args:
           pdf_folder: Directory containing PDF documents to process
           chunk_size: Size of text chunks for splitting documents
           chunk_overlap: Overlap between text chunks
           embedding_model: Model to use for embeddings
           llm_model: LLM model to use for query generation and answering
           persist_directory: Directory to persist the vector store
           temperature: Temperature parameter for the LLM
           user_agent: User agent string for web requests
       """
       # Set user agent
       os.environ['USER_AGENT'] = user_agent
      
       # Load environment variables
       load_dotenv()
      
       # Access environment variables
       self.langchain_api = os.getenv('LANGCHAIN_API_KEY')
       self.pinecone_api = os.getenv('PINECONE_API_KEY')
       self.llama_api = os.getenv('LLAMA_CLOUD_API_KEY')
       self.openai_api = os.getenv('OPENAI_API_KEY')
       # self.google_api = os.getenv('GOOGLE_API_KEY', 'AIzaSyCf4zjj3_ccHrUB3ydOHUa-_DCaWQLdxro')
       self.google_api = os.getenv('GOOGLE_API_KEY')
      
       # Set environment variables for LangChain tracing
       os.environ['LANGCHAIN_TRACING_V2'] = 'true'
       os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
       os.environ['LANGCHAIN_API_KEY'] = self.langchain_api
      
       # Set other API keys as environment variables
       os.environ['LLAMA_CLOUD_API_KEY'] = self.llama_api
       os.environ['OPENAI_API_KEY'] = self.openai_api
       os.environ['GOOGLE_API_KEY'] = self.google_api
      
       # Apply nest_asyncio for async operations
       nest_asyncio.apply()
      
       # Store configuration parameters
       self.pdf_folder = pdf_folder
       self.chunk_size = chunk_size
       self.chunk_overlap = chunk_overlap
       self.embedding_model = embedding_model
       self.llm_model = llm_model
       self.persist_directory = persist_directory
       self.temperature = temperature
       self.math_parsing_instruction = math_parsing_instruction
      
       # Initialize components
       self.parser = LlamaParse(result_type="text", system_prompt_append=self.math_parsing_instruction)
       # Pass the parser instance directly to SimpleDirectoryReader
       self.file_extractor = {".pdf": self.parser}
       self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
           chunk_size=self.chunk_size,
           chunk_overlap=self.chunk_overlap,
       )
      
       # Initialize embeddings
       self.embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model)
      
       # Initialize LLM
       self.llm = ChatGoogleGenerativeAI(model=self.llm_model, temperature=self.temperature)
      
       # Initialize other attributes
       self.vectorstore = None
       self.retriever = None
       self.retrieval_chain = None
       self.final_rag_chain = None


   async def load_documents(self) -> List[Any]:
       """
       Asynchronously load documents from the PDF folder.
      
       Returns:
           List of loaded documents
       """
       try:
           reader = SimpleDirectoryReader(
               input_dir=self.pdf_folder,
               file_extractor=self.file_extractor
           )
          
           # Handle both synchronous and asynchronous load_data methods
           try:
               # Try asynchronous loading first
               docs = await reader.aload_data()
           except Exception as e:
               # If async loading fails, fall back to synchronous loading
               print(f"Async document loading failed, switching to synchronous load: {e}")
               docs = reader.load_data()
                  
           print(f"Successfully loaded {len(docs)} documents from {self.pdf_folder}")
           return docs
          
       except Exception as e:
           print(f"Error loading documents: {e}")
           print(f"Failed to load documents from {self.pdf_folder}")
           return []
  
   def format_documents(self, docs: List[Any]) -> List[Document]:
       """
       Format raw documents into LangChain Document objects.
      
       Args:
           docs: Raw documents from SimpleDirectoryReader
          
       Returns:
           List of formatted Document objects
       """
       return [Document(
           page_content=doc.text,
           metadata={
               "source": doc.metadata.get("file_name", "unknown")
           }) for doc in docs]
  
   def prepare_documents(self, formatted_docs: List[Document]) -> List[Document]:
       """
       Split documents into chunks and enhance them with contextual information.
      
       This method first splits the documents into chunks, then enhances each chunk
       with additional contextual information to improve retrieval performance.
      
       Args:
           formatted_docs: List of formatted Document objects
          
       Returns:
           List of enhanced document chunks
          
       References:
           [1] https://github.com/athina-ai/rag-cookbooks
           [2] https://milvus.io/docs/contextual_retrieval_with_milvus.md
       """
       # First split the documents into chunks
       chunks = self.text_splitter.split_documents(formatted_docs)
      
       print(f"Enhancing {len(chunks)} document chunks with contextual information...")
      
       # Enhance each chunk with contextual information
       enhanced_chunks = []
       for i, chunk in enumerate(chunks):
           if i % 10 == 0:  # Print progress every 10 chunks
               print(f"Processing chunk {i+1}/{len(chunks)}...")
          
           enhanced_chunk = self.generate_contextual_chunk(chunk)
           enhanced_chunks.append(enhanced_chunk)
      
       print(f"✅ Enhanced {len(enhanced_chunks)} document chunks with contextual information.")
       return enhanced_chunks
  
   def build_vectorstore(self, splits: List[Document]) -> None:
       """
       Build and persist the vector store from document splits.
      
       Args:
           splits: List of document chunks
       """
       if not splits:
           print("No text chunks available. Skipping vector store creation.")
           return
          
       self.vectorstore = Chroma.from_documents(
           documents=splits,
           embedding=self.embeddings,
           persist_directory=self.persist_directory
       )
       self.vectorstore.persist()
       self.retriever = self.vectorstore.as_retriever()
  
   def load_existing_vectorstore(self) -> None:
       """
       Load an existing vector store from the persist directory.
       """
       self.vectorstore = Chroma(
           embedding_function=self.embeddings,
           persist_directory=self.persist_directory
       )
       self.retriever = self.vectorstore.as_retriever()
  
   async def update_vectorstore(self) -> bool:
       """
       Update the existing vector store with new documents without rebuilding from scratch.
      
       This method efficiently updates the vector store by:
       1. Loading new documents from the PDF folder
       2. Filtering out documents that are already in the vectorstore
       3. Processing and adding only the new documents to the existing vectorstore
      
       Returns:
           Boolean indicating whether any updates were made
          
       References:
           [1] https://python.langchain.com/docs/integrations/vectorstores/chroma
       """
       if not self.vectorstore:
           print("No existing vectorstore loaded. Cannot update.")
           return False
          
       # Get existing document sources from the vectorstore
       existing_docs = self.vectorstore.get()
       existing_sources = set()
      
       # Extract source filenames from metadata
       if 'metadatas' in existing_docs and existing_docs['metadatas']:
           for metadata in existing_docs['metadatas']:
               if metadata and 'source' in metadata:
                   existing_sources.add(metadata['source'])
      
       print(f"Found {len(existing_sources)} unique document sources in existing vectorstore.")
      
       # Load all documents from the PDF folder
       all_docs = await self.load_documents()
       if not all_docs:
           print("No documents loaded from PDF folder. Nothing to update.")
           return False
          
       # Filter out documents that are already in the vectorstore
       new_docs = []
       for doc in all_docs:
           source = doc.metadata.get('file_name', '')
           if source not in existing_sources:
               new_docs.append(doc)
      
       print(f"Found {len(new_docs)} new documents to add to the vectorstore.")
      
       if not new_docs:
           print("No new documents to add. Vectorstore is up to date.")
           return False
          
       # Format and prepare the new documents
       formatted_docs = self.format_documents(new_docs)
       new_splits = self.prepare_documents(formatted_docs)
       print(f"Created {len(new_splits)} text chunks from {len(formatted_docs)} new documents.")
      
       # Add the new documents to the existing vectorstore
       print("Adding new documents to the vectorstore...")
       self.vectorstore.add_documents(new_splits)
       self.vectorstore.persist()
       print(f"✅ Successfully added {len(new_splits)} new document chunks to the vectorstore.")
      
       return True
  
   def get_document_count(self) -> int:
       """
       Get the number of documents in the vector store.
      
       Returns:
           Number of documents
       """
       if self.vectorstore:
           all_docs = self.vectorstore.get()
           return len(all_docs['documents'])
       return 0
  
   @staticmethod
   def reciprocal_rank_fusion(results: List[List], k: int = 60) -> List[Tuple[Any, float]]:
       """
       Perform reciprocal rank fusion on multiple lists of ranked documents.
      
       Args:
           results: List of lists of ranked documents
           k: Parameter used in the RRF formula
          
       Returns:
           List of (document, score) tuples sorted by score
          
       References:
           [1] https://github.com/athina-ai/rag-cookbooks
       """
       fused_scores = {}
      
       # Iterate through each list of ranked documents
       for docs in results:
           for rank, doc in enumerate(docs):
               doc_str = dumps(doc)
               if doc_str not in fused_scores:
                   fused_scores[doc_str] = 0
               fused_scores[doc_str] += 1 / (rank + k)
      
       # Sort the documents based on their fused scores in descending order
       reranked_results = [
           (loads(doc), score)
           for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
       ]
      
       return reranked_results
      
   def rewrite_query(self, query: str) -> List[str]:
       """
       Rewrite the original query into multiple alternative queries to improve retrieval.
      
       This technique is part of the "Rewrite-Retrieve-Read" pattern from advanced RAG techniques.
       It helps address vocabulary mismatch between queries and documents.
      
       Args:
           query: The original user query
          
       Returns:
           A list of rewritten queries
          
       References:
           [1] https://github.com/athina-ai/rag-cookbooks
           [2] https://www.aporia.com/learn/enhance-rags-hyde/
       """
       # Create a prompt for query rewriting
       template = """You are an expert at reformulating search queries to improve document retrieval.
       Given a user's original query, generate 3-4 alternative versions that:
       1. Use different vocabulary that might appear in technical documents
       2. Expand acronyms or use them if the original doesn't
       3. Add relevant technical terms that might help with retrieval
       4. Vary in length and specificity


       Original query: {query}


       Output only the alternative queries, one per line, without numbering or explanation:
       """
      
       prompt = ChatPromptTemplate.from_template(template)
      
       # Generate alternative queries
       chain = prompt | self.llm | StrOutputParser()
       result = chain.invoke({"query": query})
      
       # Split the result into individual queries and clean them
       rewritten_queries = [q.strip() for q in result.split('\n') if q.strip()]
      
       # Add the original query to the list if it's not already included
       if query not in rewritten_queries:
           rewritten_queries.append(query)
          
       return rewritten_queries
  
   def generate_contextual_chunk(self, chunk: Document) -> Document:
       """
       Enhance a document chunk with additional context to improve retrieval.
      
       This technique is part of "Contextual RAG" which adds relevant context to document
       chunks before embedding, addressing the issue of semantic isolation of chunks.
      
       Args:
           chunk: The original document chunk
          
       Returns:
           A new Document with enhanced contextual content
          
       References:
           [1] https://github.com/athina-ai/rag-cookbooks
           [2] https://milvus.io/docs/contextual_retrieval_with_milvus.md
       """
       # Create a prompt for contextual enhancement
       template = """You are an expert at enhancing document chunks with additional context for better retrieval.
       Given the following text chunk, add relevant contextual information that would help this chunk be retrieved
       when a user asks related questions. Focus on:
       1. Adding key terms and synonyms that might appear in user queries
       2. Summarizing the main concepts in a way that connects to potential questions
       3. Preserving all the original information


       Original chunk:
       {chunk_content}


       Output the enhanced chunk that includes both the original content and the added context:
       """
      
       prompt = ChatPromptTemplate.from_template(template)
      
       # Generate enhanced content
       chain = prompt | self.llm | StrOutputParser()
       enhanced_content = chain.invoke({"chunk_content": chunk.page_content})
      
       # Create a new Document with the enhanced content and original metadata
       return Document(
           page_content=enhanced_content,
           metadata=chunk.metadata
       )
  
   def generate_hypothetical_document(self, query: str) -> Document:
       """
       Generate a hypothetical document based on the query to improve retrieval.
      
       This technique is known as HyDE (Hypothetical Document Embeddings) which uses
       an LLM to generate a hypothetical document that would be relevant to the query,
       then uses its embedding to retrieve actual documents.
      
       Args:
           query: The user query
          
       Returns:
           A Document object containing the hypothetical content
          
       References:
           [1] https://github.com/athina-ai/rag-cookbooks/blob/main/advanced_rag_techniques/hyde_rag.ipynb
           [2] https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings
       """
       # Create a prompt for generating a hypothetical document
       template = """You are an expert at creating hypothetical documents that would perfectly answer a user's query.
       Given the following query, generate a detailed, factual response as if you were writing a document
       that would be the ideal search result for this query. Include technical details, explanations,
       and any information that would be relevant.


       Query: {query}


       Hypothetical document:
       """
      
       prompt = ChatPromptTemplate.from_template(template)
      
       # Generate hypothetical document
       chain = prompt | self.llm | StrOutputParser()
       hypothetical_content = chain.invoke({"query": query})
      
       # Create a Document object with the hypothetical content
       return Document(
           page_content=hypothetical_content,
           metadata={"source": "hypothetical_document", "query": query}
       )
  
   def create_rag_pipeline(self) -> None:
       """
       Create an advanced RAG pipeline incorporating multiple techniques:
       1. Query rewriting (Rewrite-Retrieve-Read)
       2. Hypothetical Document Embeddings (HyDE)
       3. Reciprocal Rank Fusion for merging results
      
       This enhanced pipeline improves retrieval performance by addressing common
       issues like vocabulary mismatch and semantic gaps between queries and documents.
      
       References:
           [1] https://github.com/athina-ai/rag-cookbooks
           [2] https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings
           [3] https://dev.to/rogiia/build-an-advanced-rag-app-query-rewriting-h3p
       """
       # Check if retriever is initialized
       if self.retriever is None:
           print("Warning: Retriever is not initialized. RAG pipeline may not work correctly.")
           return
      
       # Define a function to process a query through the advanced pipeline
       async def advanced_retrieval(query_input):
           query = query_input["question"]
          
           # Step 1: Rewrite the query to generate multiple alternative queries
           print(f"Rewriting query: {query}")
           rewritten_queries = self.rewrite_query(query)
           print(f"Generated {len(rewritten_queries)} alternative queries")
          
           # Step 2: Retrieve documents using each rewritten query in parallel
           print(f"Retrieving documents for {len(rewritten_queries)} queries in parallel")
           tasks = [
               asyncio.to_thread(self.retriever.invoke, rewritten_query)
               for rewritten_query in rewritten_queries
           ]
           retrieval_results = await asyncio.gather(*tasks)
          
           # Step 3: Generate a hypothetical document and use it for retrieval
           print("Generating hypothetical document based on the original query")
           hypothetical_doc = self.generate_hypothetical_document(query)
          
           # Step 4: Use the hypothetical document to retrieve similar real documents
           print("Retrieving documents similar to the hypothetical document")
           hypothetical_embedding = self.embeddings.embed_documents([hypothetical_doc.page_content])[0]
           hyde_results = self.vectorstore.similarity_search_by_vector(hypothetical_embedding)
           retrieval_results.append(hyde_results)
          
           # Step 5: Merge all retrieval results using reciprocal rank fusion
           print("Merging retrieval results using reciprocal rank fusion")
           fused_results = self.reciprocal_rank_fusion(retrieval_results)
          
           # Extract just the documents from the (doc, score) tuples
           return [doc for doc, _ in fused_results]
      
       # Create the retrieval chain
       self.retrieval_chain = advanced_retrieval
      
       # Create the final answer generation prompt
       answer_template = """Answer the following question based on this context:


       {context}


       Question: {question}


       If the question is marked as out-of-domain, respond with:
       "I'm sorry, but this question appears to be outside the scope of my training data and I don't have relevant information to provide a reliable answer."


       Otherwise, provide a comprehensive answer that directly addresses the question using the provided context. If the context does not contain sufficient information to answer the question, follow these steps:


       1. Determine if this is an exercise/problem that requires a solution.
       2. If it is an exercise problem, solve it step-by-step showing all your work and calculations.
       3. Prepend your answer with: "Since the lecture notes does not show the answer, here is the answer to the best of my knowledge: "
       4. Provide a complete solution with explanations for each step.


       For mathematical problems:


       - Show all formulas and equations used.
       - Format all mathematical expressions using LaTeX enclosed in double dollar signs ($$).
       - Preserve all mathematical notation exactly as written in the context.
       - If the context contains LaTeX formulas (enclosed in $$ delimiters), include them exactly as they appear.
       - Explain your reasoning at each step.
       - Calculate the final numerical answer if applicable.
       - Verify your solution if possible.


       Examples of properly formatted math:
       - For inline equations: The formula is $$E = mc^2$$.
       - For displayed equations: $$\\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}$$
       - For statistical formulas: The variance is $$\\sigma^2 = \\frac{{\\sum_{{i=1}}^{{n}} (x_i - \\mu)^2}}{{n}}$$


       If you don't know the answer, just say that you don't know.
       """
       answer_prompt = ChatPromptTemplate.from_template(answer_template)
      
       # Create the final RAG chain
       self.final_rag_chain = (
           {"context": lambda x: x["context"],
            "question": itemgetter("question")}
           | answer_prompt
           | self.llm
           | StrOutputParser()
       )
  
   async def setup(self) -> None:
       """
       Set up the retriever by loading documents, building or updating the vector store,
       and creating the RAG pipeline.
      
       This method implements an efficient workflow:
       1. Check if an existing vectorstore exists
       2. If it exists, load it and update with any new documents
       3. If it doesn't exist, build a new vectorstore from scratch
       4. Create the RAG pipeline if the vectorstore has documents
      
       This approach avoids rebuilding the entire vectorstore when only adding new documents,
       making the process more efficient for incremental updates.
       """
       # Verify that the data directory exists and contains PDF files
       if not os.path.exists(self.pdf_folder):
           print(f"⚠️ Data directory '{self.pdf_folder}' does not exist.")
           print(f"Please create the directory at '{os.path.abspath(self.pdf_folder)}' and add PDF documents.")
           print(f"If you're running this script from a different directory, consider using an absolute path.")
           return
      
       pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith('.pdf')]
       if not pdf_files:
           print(f"⚠️ No PDF files found in '{self.pdf_folder}'. Please add PDF documents to this directory.")
           return
       else:
           print(f"Found {len(pdf_files)} PDF files in '{self.pdf_folder}'.")
      
       # Check if vectorstore exists
       vectorstore_exists = os.path.exists(self.persist_directory) and os.listdir(self.persist_directory)
      
       if vectorstore_exists:
           print(f"Existing vectorstore found at '{self.persist_directory}'.")
           # Load existing vectorstore
           self.load_existing_vectorstore()
           doc_count = self.get_document_count()
           print(f"Loaded existing vectorstore with {doc_count} documents.")
          
           # Update vectorstore with new documents
           updated = await self.update_vectorstore()
          
           if updated:
               # Refresh document count after update
               doc_count = self.get_document_count()
               print(f"Vectorstore updated. New document count: {doc_count}")
           else:
               print("No updates needed for the vectorstore.")
       else:
           print("No existing vectorstore found. Building from scratch...")
           # Load and process documents
           docs = await self.load_documents()
           print(f"Loaded {len(docs)} documents from PDFs in '{self.pdf_folder}'.")
          
           if not docs:
               print("⚠️ No documents were loaded. Cannot proceed with RAG pipeline setup.")
               return
              
           formatted_docs = self.format_documents(docs)
           splits = self.prepare_documents(formatted_docs)
           print(f"✅ Created {len(splits)} text chunks from {len(formatted_docs)} documents.")
          
           # Build vector store
           self.build_vectorstore(splits)
           doc_count = self.get_document_count()
           print(f"Number of documents in the vector store: {doc_count}")
      
       # Create RAG pipeline only if we have documents
       if doc_count > 0:
           self.create_rag_pipeline()
       else:
           print("⚠️ No documents in vector store. RAG pipeline not created.")
  
   def setup_from_existing(self) -> None:
       """
       Set up the retriever from an existing vector store.
       """
       # Load existing vector store
       self.load_existing_vectorstore()
       doc_count = self.get_document_count()
       print(f"Loaded existing vector store with {doc_count} documents.")
      
       # Create RAG pipeline
       self.create_rag_pipeline()
  
   def _parse_json(self, result: str, default: Dict[str, Any]) -> Dict[str, Any]:
       """
       Parse a JSON string and return the parsed object or a default value if parsing fails.
      
       Args:
           result: The JSON string to parse
           default: The default value to return if parsing fails
          
       Returns:
           The parsed JSON object or the default value
       """
       print('Debug: Raw result from LLM:', result)
      
       # Preserve LaTeX expressions before processing
       # Temporarily replace LaTeX expressions with placeholders
       latex_placeholders = {}
       latex_pattern = r'(\$\$[^\$]+\$\$)'
      
       def replace_latex(match):
           placeholder = f'LATEX_PLACEHOLDER_{len(latex_placeholders)}'
           latex_placeholders[placeholder] = match.group(0)
           return placeholder
      
       # Replace LaTeX expressions with placeholders
       if '$$' in result:
           import re
           result_with_placeholders = re.sub(latex_pattern, replace_latex, result, flags=re.DOTALL)
       else:
           result_with_placeholders = result
      
       # Check if the result is wrapped in markdown code block delimiters
       if result_with_placeholders.strip().startswith('```json') and result_with_placeholders.strip().endswith('```'):
           # Strip the markdown code block delimiters
           # Remove the first line containing ```json and the last line containing ```
           lines = result_with_placeholders.strip().split('\n')
           if len(lines) > 2:  # Ensure there are at least 3 lines (opening, content, closing)
               result_with_placeholders = '\n'.join(lines[1:-1])
               print('Debug: Result after stripping markdown code block delimiters:', result_with_placeholders)
           else:
               print('Warning: Unexpected format in markdown code block')
      
       try:
           parsed_result = json.loads(result_with_placeholders)
          
           # Restore LaTeX expressions if any were replaced
           if latex_placeholders:
               result_str = json.dumps(parsed_result)
               for placeholder, latex in latex_placeholders.items():
                   result_str = result_str.replace(placeholder, latex)
               return json.loads(result_str)
           return parsed_result
       except json.JSONDecodeError as e:
           print(f'Error parsing JSON: {e}')
           # Try one more approach - look for JSON content between curly braces
           try:
               start_idx = result_with_placeholders.find('{')
               end_idx = result_with_placeholders.rfind('}')
               if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                   json_content = result_with_placeholders[start_idx:end_idx+1]
                   print('Debug: Attempting to parse extracted JSON content:', json_content)
                   parsed_json = json.loads(json_content)
                  
                   # Restore LaTeX expressions if any were replaced
                   if latex_placeholders:
                       result_str = json.dumps(parsed_json)
                       for placeholder, latex in latex_placeholders.items():
                           result_str = result_str.replace(placeholder, latex)
                       return json.loads(result_str)
                   return parsed_json
           except json.JSONDecodeError as nested_e:
               print(f'Error parsing extracted JSON content: {nested_e}')
          
           return default
  
   def evaluate_context_sufficiency(self, context: str, query: str) -> Dict[str, Any]:
       """
       Assess whether the retrieved context contains enough information to answer the query.
      
       This method evaluates if the context has sufficient information by checking for the
       presence of query keywords and relevant information in the context.
      
       Args:
           context: The retrieved context text
           query: The user's query
          
       Returns:
           Dictionary containing evaluation results including a sufficiency score and analysis
          
       References:
           [1] https://github.com/athina-ai/rag-cookbooks
       """
       # Create a prompt for context sufficiency evaluation
       template = """You are an expert at evaluating whether a context contains enough information to answer a query.
       Given the following query and context, determine if the context has sufficient information to provide a complete answer.


       Query: {query}


       Context: {context}


       Evaluate the context sufficiency by considering:
       1. Whether key concepts from the query appear in the context
       2. Whether the context provides enough detail to fully answer the query
       3. Whether important related information is present


       Output a JSON with the following structure:
       {{
       "sufficiency_score": <float between 0 and 1>,
       "missing_information": <list of information missing from the context>,
       "analysis": <brief explanation of your evaluation>
       }}
       """
      
       prompt = ChatPromptTemplate.from_template(template)
      
       # Create a chain that outputs a structured result
       chain = prompt | self.llm | StrOutputParser()
       result = chain.invoke({"query": query, "context": context})
      
       # Parse the result as JSON
       default = {
           "sufficiency_score": 0.0,
           "missing_information": ["Error in evaluation"],
           "analysis": "Error in evaluation"
       }
       evaluation = self._parse_json(result, default)
       return evaluation
  
   def evaluate_faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
       """
       Analyze the answer to ensure its content is supported by the provided context.
      
       This method checks if the generated answer is faithful to the retrieved context
       by identifying any statements in the answer that aren't supported by the context.
      
       Args:
           answer: The generated answer
           context: The retrieved context text
          
       Returns:
           Dictionary containing evaluation results including a faithfulness score and unsupported statements
          
       References:
           [1] https://github.com/athina-ai/rag-cookbooks
       """
       # Create a prompt for faithfulness evaluation
       template = """You are an expert at evaluating whether an answer is faithful to the provided context.
       Given the following answer and the context it was generated from, determine if the answer contains
       any statements or claims that are not supported by the context.


       Answer: {answer}


       Context: {context}


       Evaluate the faithfulness by:
       1. Identifying any statements in the answer that aren't explicitly supported by the context
       2. Checking if the answer contains any additional information not present in the context
       3. Verifying that the answer doesn't contradict the context


       Output a JSON with the following structure:
       {{
       "faithfulness_score": <float between 0 and 1>,
       "unsupported_statements": <list of statements not supported by the context>,
       "analysis": <brief explanation of your evaluation>
       }}
       """
      
       prompt = ChatPromptTemplate.from_template(template)
      
       # Create a chain that outputs a structured result
       chain = prompt | self.llm | StrOutputParser()
       result = chain.invoke({"answer": answer, "context": context})
      
       # Parse the result as JSON
       default = {
           "faithfulness_score": 0.0,
           "unsupported_statements": ["Error in evaluation"],
           "analysis": "Error in evaluation"
       }
       evaluation = self._parse_json(result, default)
       return evaluation
  
   def evaluate_answer_completeness(self, answer: str, query: str) -> Dict[str, Any]:
       """
       Determine if the answer addresses all important aspects of the query.
      
       This method evaluates whether the generated answer completely addresses
       all the important aspects and requirements of the user's query.
      
       Args:
           answer: The generated answer
           query: The user's query
          
       Returns:
           Dictionary containing evaluation results including a completeness score and analysis
          
       References:
           [1] https://github.com/athina-ai/rag-cookbooks
       """
       # Create a prompt for completeness evaluation
       template = """You are an expert at evaluating whether an answer completely addresses a query.
       Given the following query and answer, determine if the answer addresses all important aspects of the query.


       Query: {query}


       Answer: {answer}


       Evaluate the completeness by considering:
       1. Whether all parts of the query are addressed in the answer
       2. Whether the answer provides sufficient detail for each part of the query
       3. Whether any important aspects of the query are ignored or insufficiently addressed


       Output a JSON with the following structure:
       {{
       "completeness_score": <float between 0 and 1>,
       "unaddressed_aspects": <list of aspects of the query not fully addressed>,
       "analysis": <brief explanation of your evaluation>
       }}
       """
      
       prompt = ChatPromptTemplate.from_template(template)
      
       # Create a chain that outputs a structured result
       chain = prompt | self.llm | StrOutputParser()
       result = chain.invoke({"query": query, "answer": answer})
      
       # Parse the result as JSON
       default = {
           "completeness_score": 0.0,
           "unaddressed_aspects": ["Error in evaluation"],
           "analysis": "Error in evaluation"
       }
       evaluation = self._parse_json(result, default)
       return evaluation
  
   async def _check_question_relevance(self, question: str) -> bool:
       """
       Check if a question is relevant to the domain using keyword/pattern matching
       followed by dynamic domain relevance checking.
      
       This method first uses pattern matching to identify potentially out-of-domain
       questions. If a pattern is matched, it performs a more thorough check using
       the vectorstore's content to make a final determination, rather than
       immediately rejecting the question.
      
       Args:
           question: The user's question
          
       Returns:
           Boolean indicating whether the question is potentially relevant
       """
       # Convert question to lowercase for case-insensitive matching
       question_lower = question.lower()
      
       # Define patterns for out-of-domain questions
       out_of_domain_patterns = [
           "assignment due", "due date", "when is", "homework due",
           "office hours", "syllabus", "course website", "class schedule",
           "exam date", "test date", "quiz", "grade", "grading",
           "extension", "late submission", "email the professor",
           "contact the ta", "zoom link", "meeting link", "class cancelled"
       ]
      
       # Flag to track if an out-of-domain pattern was matched
       pattern_matched = False
       matched_pattern = ""
      
       # Check if any out-of-domain pattern is in the question
       for pattern in out_of_domain_patterns:
           if pattern in question_lower:
               print(f"Question contains potential out-of-domain pattern: '{pattern}'")
               pattern_matched = True
               matched_pattern = pattern
               break
      
       # If a pattern was matched, perform a dynamic domain relevance check
       if pattern_matched and self.vectorstore is not None:
           print(f"Pattern '{matched_pattern}' detected, performing dynamic domain relevance check...")
           # Use the existing is_query_in_domain method to check if the query is actually in-domain
           # based on the vectorstore's content
           is_in_domain = self.is_query_in_domain(question)
           if is_in_domain:
               print(f"Despite containing pattern '{matched_pattern}', query was determined to be in-domain based on vectorstore content.")
               return True
           else:
               print(f"Query confirmed to be out-of-domain after dynamic check.")
               return False
       elif pattern_matched:
           # If vectorstore is not available, fall back to pattern-based decision
           print("Vectorstore not available for dynamic check, using pattern-based decision.")
           return False
              
       # If no out-of-domain patterns matched, the question might be relevant
       return True
      
   def is_query_in_domain(self, query: str) -> bool:
       """
       Check if a query is within the domain of the training data.
      
       This method uses a simple heuristic based on retrieving a few documents
       and checking for semantic similarity to determine if the query is related
       to the domain the RAG system is trained on.
      
       Args:
           query: The user's query
          
       Returns:
           Boolean indicating whether the query is in-domain
       """
       # If vectorstore is not initialized, we can't check domain
       if self.vectorstore is None:
           print("Warning: Vectorstore not initialized. Cannot check domain relevance.")
           return True  # Default to True to avoid blocking queries
      
       try:
           # Retrieve top 3 documents for the query
           docs = self.vectorstore.similarity_search(query, k=3)
          
           if not docs:
               print("No relevant documents found for the query.")
               return False
          
           # Create a prompt to evaluate domain relevance
           template = """You are an expert at determining whether a query is relevant to a specific domain.
           Given the following query and retrieved context, determine if the query is related to the domain
           represented by the context.
          
           Query: {query}
          
           Retrieved Context:
           {context}
          
           Output a JSON with the following structure:
           {{
           "is_in_domain": <boolean>,
           "confidence": <float between 0 and 1>,
           "reasoning": <brief explanation of your evaluation>
           }}
           """
          
           prompt = ChatPromptTemplate.from_template(template)
          
           # Join the text from the retrieved documents to form the context
           context = "\n\n".join([doc.page_content for doc in docs])
          
           # Create a chain that outputs a structured result
           chain = prompt | self.llm | StrOutputParser()
           result = chain.invoke({"query": query, "context": context})
          
           # Parse the result as JSON
           default = {
               "is_in_domain": False,
               "confidence": 0.0,
               "reasoning": "Error in evaluation"
           }
           evaluation = self._parse_json(result, default)
          
           # Log the evaluation results
           print(f"Domain relevance check: {evaluation['is_in_domain']} (confidence: {evaluation['confidence']:.2f})")
           print(f"Reasoning: {evaluation['reasoning']}")
          
           return evaluation["is_in_domain"]
          
       except Exception as e:
           print(f"Error checking domain relevance: {e}")
           return True  # Default to True in case of errors
  
   def detect_hallucinations(self, answer: str, context: str) -> Dict[str, Any]:
       """
       Detect portions of the answer that seem to be hallucinated (not backed by the context).
      
       This method identifies specific segments of the answer that appear to be hallucinated
       or fabricated, rather than being derived from the provided context.
      
       Args:
           answer: The generated answer
           context: The retrieved context text
          
       Returns:
           Dictionary containing evaluation results including hallucinated segments and a hallucination score
          
       References:
           [1] https://github.com/athina-ai/rag-cookbooks
       """
       # Create a prompt for hallucination detection
       template = """You are an expert at detecting hallucinations in AI-generated answers.
       Given the following answer and the context it was generated from, identify any portions of the answer
       that appear to be hallucinated (i.e., not supported by or derived from the context).


       Answer: {answer}


       Context: {context}


       Detect hallucinations by:
       1. Breaking down the answer into key statements or claims
       2. For each statement, determining if it's supported by the context
       3. Identifying specific phrases or sentences that contain hallucinated information


       Output a JSON with the following structure:
       {{
       "hallucination_score": <float between 0 and 1, where 0 means no hallucinations>,
       "hallucinated_segments": <list of specific segments from the answer that appear hallucinated>,
       "analysis": <brief explanation of your evaluation>
       }}
       """
      
       prompt = ChatPromptTemplate.from_template(template)
      
       # Create a chain that outputs a structured result
       chain = prompt | self.llm | StrOutputParser()
       result = chain.invoke({"answer": answer, "context": context})
      
       # Parse the result as JSON
       default = {
           "hallucination_score": 1.0,
           "hallucinated_segments": ["Error in evaluation"],
           "analysis": "Error in evaluation"
       }
       evaluation = self._parse_json(result, default)
       return evaluation
  
   def _document_contributed_to_answer(self, answer: str, doc: Document) -> bool:
       """
       Check if the document contributed to the answer using simple heuristics.
       For instance, use keyword matching or date detection (if the answer contains specific dates or deadline terms and they appear in the document).
       """
       answer_lower = answer.lower()
       doc_content = doc.page_content.lower()
      
       # Simple heuristic: check if key terms (e.g., 'due', 'deadline', 'presentation', 'report') appear in both answer and document
       keywords = ['due', 'deadline', 'presentation', 'report']
       match_count = sum(1 for kw in keywords if kw in answer_lower and kw in doc_content)
       if match_count >= 2:
           return True
          
       # Check for date patterns if necessary (example using a simple pattern):
       import re
       date_pattern = r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b'
       answer_dates = re.findall(date_pattern, answer_lower, re.IGNORECASE)
       doc_dates = re.findall(date_pattern, doc_content, re.IGNORECASE)
       if answer_dates and doc_dates and any(ad.lower() in doc_content for ad in answer_dates):
           return True
          
       return False
  
   def _identify_contributing_sources(self, answer: str, docs: List[Document]) -> List[Document]:
       """
       Identify which documents among the retrieved docs contributed to the answer.
       """
       contributing_docs = []
       for doc in docs:
           # Skip hypothetical documents
           if doc.metadata.get('source', '') == 'hypothetical_document':
               continue
           if self._document_contributed_to_answer(answer, doc):
               contributing_docs.append(doc)
              
       if not contributing_docs and docs:
           # Fallback: return the first document that is not hypothetical
           for doc in docs:
               if doc.metadata.get('source', '') != 'hypothetical_document':
                   contributing_docs.append(doc)
                   break
                  
       return contributing_docs
  
   def run_evaluations(self, question: str, answer: str, context: str) -> Dict[str, Any]:
       """
       Run all evaluation metrics and aggregate the results.
      
       This method runs all the evaluation metrics (context sufficiency, faithfulness,
       completeness, and hallucination detection) and aggregates the results into a
       single dictionary of metrics.
      
       Args:
           question: The user's question
           answer: The generated answer
           context: The retrieved context text
          
       Returns:
           Dictionary containing all evaluation metrics
          
       References:
           [1] https://github.com/athina-ai/rag-cookbooks
       """
       print("Running RAG evaluation metrics...")
      
       # Run all evaluations
       print("Evaluating context sufficiency...")
       context_eval = self.evaluate_context_sufficiency(context, question)
      
       print("Evaluating answer faithfulness...")
       faithfulness_eval = self.evaluate_faithfulness(answer, context)
      
       print("Evaluating answer completeness...")
       completeness_eval = self.evaluate_answer_completeness(answer, question)
      
       print("Detecting hallucinations...")
       hallucination_eval = self.detect_hallucinations(answer, context)
      
       # Aggregate all evaluation results
       evaluation_results = {
           "context_sufficiency": context_eval,
           "faithfulness": faithfulness_eval,
           "completeness": completeness_eval,
           "hallucination": hallucination_eval,
           "overall_quality_score": (
               context_eval.get("sufficiency_score", 0) * 0.25 +
               faithfulness_eval.get("faithfulness_score", 0) * 0.35 +
               completeness_eval.get("completeness_score", 0) * 0.25 +
               (1 - hallucination_eval.get("hallucination_score", 1)) * 0.15
           )
       }
      
       print("Evaluation complete.")
       return evaluation_results
  
   async def invoke(self, question: str) -> Dict[str, Any]:
       """
       Invoke the advanced RAG pipeline with a question and evaluate the results.
      
       This method processes the user's question through the enhanced RAG pipeline,
       which includes:
       1. Initial relevance check using keyword/pattern matching
       2. Domain relevance check using semantic similarity
       3. Query rewriting, hypothetical document generation, and reciprocal rank fusion
       4. Answer generation and quality evaluation
      
       The method first performs a quick check to filter out clearly out-of-domain
       questions (like "when is the assignment due?") before performing more
       expensive document retrieval and semantic similarity checks.
      
       Args:
           question: The question to answer
          
       Returns:
           Dictionary containing the generated answer and a filtered list of source identifiers that actually contributed to it
           {
               'answer': str,  # The generated answer text
               'sources': List[str]  # List of source identifiers from document metadata
           }
          
       References:
           [1] https://github.com/athina-ai/rag-cookbooks
       """
       if not self.final_rag_chain:
           raise ValueError("RAG pipeline not set up. Call setup() or setup_from_existing() first.")
      
       print(f"Processing question: {question}")
      
       # First do a quick check using keyword/pattern matching
       print("Performing initial relevance check using keyword/pattern matching...")
       is_relevant = await self._check_question_relevance(question)
      
       if not is_relevant:
           print("Question detected as out-of-domain based on keyword patterns. Skipping retrieval and generation.")
           return {
               "answer": "I'm sorry, but this question appears to be outside the scope of my training data and I don't have relevant information to provide a reliable answer.",
               "sources": []
           }
      
       # If the question passes the initial check, perform the more expensive domain check
       print("Checking if query is in-domain using semantic similarity...")
       is_in_domain = self.is_query_in_domain(question)
      
       if not is_in_domain:
           print("Query detected as out-of-domain. Skipping retrieval and generation.")
           return {
               "answer": "I'm sorry, but this question appears to be outside the scope of my training data and I don't have relevant information to provide a reliable answer.",
               "sources": []
           }
      
       print("Using advanced RAG techniques: Query Rewriting, Contextual RAG, and HyDE")
      
       # First, retrieve the context separately
       print("Retrieving context...")
       retrieved_docs = await self.retrieval_chain({"question": question})
      
       # Join the text from the retrieved documents to form the context
       context = "\n\n".join([doc.page_content for doc in retrieved_docs])
       print(f"Retrieved {len(retrieved_docs)} documents for context.")
      
       # Invoke the enhanced RAG pipeline to generate the answer
       print("Generating answer...")
       answer = await asyncio.to_thread(self.final_rag_chain.invoke, {"question": question, "context": context})
      
       # Run evaluations on the generated answer
       evaluation_results = self.run_evaluations(question, answer, context)
      
       # Log the evaluation results
       print("\n--- RAG Evaluation Results ---")
       print(f"Context Sufficiency Score: {evaluation_results['context_sufficiency']['sufficiency_score']:.2f}")
       print(f"Faithfulness Score: {evaluation_results['faithfulness']['faithfulness_score']:.2f}")
       print(f"Completeness Score: {evaluation_results['completeness']['completeness_score']:.2f}")
       print(f"Hallucination Score: {evaluation_results['hallucination']['hallucination_score']:.2f}")
       print(f"Overall Quality Score: {evaluation_results['overall_quality_score']:.2f}")
       print("-----------------------------\n")
      
       # Use new helper to filter for documents that actually contributed to the answer
       contributing_docs = self._identify_contributing_sources(answer, retrieved_docs)
      
       # Extract the source metadata from contributing documents
       sources = [doc.metadata.get('source', 'unknown') for doc in contributing_docs]
      
       # Remove duplicates and exclude hypothetical_documents
       unique_sources = []
       for source in sources:
           if source not in unique_sources and source != "hypothetical_document":
               unique_sources.append(source)
      
       print(f"Sources used: {unique_sources}")
      
       # Return both the answer and sources
       return {
           "answer": answer,
           "sources": unique_sources
       }




if __name__ == "__main__":
   # Create a retriever instance with the correct path to your PDF files
   # Use Path to resolve the data directory relative to the script location
   current_dir = Path(__file__).parent.parent  # Go up one level from src to rag_llm
   data_dir = current_dir / 'data'  # Path to the data directory
  
   # Convert to string and ensure it ends with a slash
   data_path = str(data_dir) + ('/' if not str(data_dir).endswith('/') else '')
  
   print(f"Using data directory: {data_path}")
  
   # Initialize RAG_Pipeline with the absolute path
   rag = RAG_Pipeline(pdf_folder=data_path)
  
   # The setup method now handles both creating a new vectorstore and updating an existing one
   print("Setting up RAG pipeline (will update existing vectorstore if available)...")
   asyncio.run(rag.setup())
  
   # Verify that the pipeline was successfully created before invoking
   if rag.retriever is None or rag.final_rag_chain is None:
       print("⚠️ RAG pipeline was not properly initialized. Cannot answer questions.")
       print("Please check that your documents were loaded correctly.")
       exit(1)
      
   # Ask a question
   question = input("Ask me anything about Numerical Analysis: ")
   answer = asyncio.run(rag.invoke(question))
   print("\nAnswer:")
   print(answer)

