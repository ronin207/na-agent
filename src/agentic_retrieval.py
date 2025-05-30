import os
import sys
import logging
from typing import Dict, List, Any, Union, Optional, Literal
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pydantic.v1 import BaseModel, Field
from operator import itemgetter
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: Literal["local_documents", "web_search"] = Field(
        ...,
        description="""Given a user question choose which datasource would be most relevant for answering their question.
        If it is related to Numerical Analysis, choose local_documents..
        Otherwise, choose web_search.""",
    )

class RAGFusion:
    def __init__(self, question: str, llm: ChatOpenAI):
        self.question = question
        self.llm = llm

    def generate_queries(self):
        """Generate multiple queries related to the question."""
        template = """You are a helpful assistant that generates multiple search queries based on a single input query.
        The context overall should not be changed, and you should stick with the topic of the input query.
        Generate multiple search queries related to: {question}
        Output (4 queries):"""
        
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)
        
        generate_queries = (
            prompt_rag_fusion
            | self.llm
            | StrOutputParser()
            | (lambda x: x.strip().split("\n"))  # Ensure output is a clean list of queries
        )
        return generate_queries

    def reciprocal_rank_fusion(self, results: List[List], k: int = 60):
        """Perform reciprocal rank fusion on multiple result sets."""
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results

    def retrieval_chain_rag_fusion(self, retriever, generate_queries, reciprocal_rank_fusion):
        """Create retrieval chain for RAG-Fusion."""
        retrieval_chain_rag_fusion = (
            generate_queries
            | (lambda queries: [retriever.invoke(q) if isinstance(q, str) else retriever.invoke(" ".join(q)) for q in queries])
            | reciprocal_rank_fusion
        )
        return retrieval_chain_rag_fusion

    def final_rag_chain(self, retrieval_chain_rag_fusion):
        """Create the final RAG chain and capture source documents."""
        template = r"""Answer the following question based on this context:

        {context}

        Question: {question}

        When including mathematical expressions:
        - Use $...$ for inline math (not \(...\))
        - Use $$...$$ for equation blocks (not \[...\])

        However, if the question's contents are not in the retriever, please state how the information
        cannot be found in the textbook. Say something like: "The information is not stated in the textbook,
        so you would not be deeply required to understand it as of now."
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Get the ranked documents from retrieval chain
        ranked_docs = retrieval_chain_rag_fusion.invoke({"question": self.question})
        
        # Extract just the documents for context
        context_docs = [doc for doc, score in ranked_docs[:5]]  # Use top 5 docs
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Generate the answer
        final_rag_chain = (
            {"context": lambda x: context_text, "question": itemgetter("question")}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        rag_fusion_answer = final_rag_chain.invoke({"question": self.question})
        
        # Extract source information
        sources = []
        for doc in context_docs:
            source_info = doc.metadata.get('source', 'Unknown source')
            # Extract just the filename from the full path
            if '/' in source_info:
                source_info = source_info.split('/')[-1]
            sources.append(source_info)
        
        # Remove duplicates while preserving order
        unique_sources = []
        for source in sources:
            if source not in unique_sources:
                unique_sources.append(source)
        
        return {
            "answer": rag_fusion_answer,
            "sources": unique_sources
        }

class WebSearch:
    def __init__(self, client: OpenAI):
        self.client = client

    def web_search(self, query: str):
        """Perform web search using OpenAI's search capabilities."""
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini-search-preview",
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            )

            return (
                "As the answer to the question is not in the textbook, I have searched the web for the response: "
                + completion.choices[0].message.content
            )
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return "Sorry, I couldn't perform a web search at this time. Please try again later."

class AgenticRetrieval:
    """
    Simplified Agentic Retrieval system based on RAG-Fusion pattern.
    This implementation follows the working notebook pattern for better reliability.
    """
    
    def __init__(
        self,
        pdf_folder: str = "./data/",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-2024-08-06",
        persist_directory: str = "./chroma_db",
        temperature: float = 0.0,
        k: int = 5,
        force_rebuild: bool = False
    ):
        # Load environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
        
        self.pdf_folder = pdf_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.persist_directory = persist_directory
        self.temperature = temperature
        self.k = k
        self.force_rebuild = force_rebuild
        
        # Initialize models
        self.llm = ChatOpenAI(
            model_name=self.llm_model,
            temperature=self.temperature,
            openai_api_key=openai_api_key
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=openai_api_key
        )
        
        # Initialize OpenAI client for web search
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.web_search = WebSearch(self.openai_client)
        
        # Initialize retriever
        self.retriever = None
        self.vectorstore = None
        
        # Set up router
        self._setup_router()
        
        # Load or build vector store
        self._setup_vectorstore()
    
    def _setup_router(self):
        """Set up the query router."""
        structured_llm = self.llm.with_structured_output(RouteQuery)
        
        system_prompt = """You are an expert at routing a user question to the appropriate data source.
        Based on the overall content the question is referring to, route it to the relevant data source."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        self.router = prompt | structured_llm
    
    def _setup_vectorstore(self):
        """Set up the vector store and retriever."""
        # Check if we need to rebuild based on data changes
        needs_rebuild = self._check_if_data_changed()
        
        if os.path.exists(self.persist_directory) and not self.force_rebuild and not needs_rebuild:
            logger.info(f"Loading existing vector store from {self.persist_directory}")
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
                logger.info("Vector store loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}. Rebuilding...")
        
        if needs_rebuild:
            logger.info("Data folder has been modified since last build. Rebuilding vector store...")
        elif self.force_rebuild:
            logger.info("Force rebuild requested. Building new vector store...")
        else:
            logger.info("Building new vector store...")
            
        self._build_vectorstore()
    
    def _check_if_data_changed(self):
        """Check if data folder has been modified since the last vectorstore build."""
        import glob
        import time
        
        # Path to store the last build timestamp
        timestamp_file = os.path.join(self.persist_directory, ".last_build_timestamp")
        
        # If vectorstore doesn't exist, we need to build
        if not os.path.exists(self.persist_directory):
            return True
            
        # If timestamp file doesn't exist, we need to rebuild
        if not os.path.exists(timestamp_file):
            return True
            
        try:
            # Get last build timestamp
            with open(timestamp_file, 'r') as f:
                last_build_time = float(f.read().strip())
                
            # Get modification times of all files in data folder
            data_files = []
            data_files.extend(glob.glob(os.path.join(self.pdf_folder, "**/*.pdf"), recursive=True))
            data_files.extend(glob.glob(os.path.join(self.pdf_folder, "**/*.ipynb"), recursive=True))
            
            # Check if any file has been modified since last build
            for file_path in data_files:
                if os.path.getmtime(file_path) > last_build_time:
                    logger.info(f"Detected change in: {os.path.basename(file_path)}")
                    return True
                    
            return False
            
        except Exception as e:
            logger.warning(f"Error checking data changes: {e}. Will rebuild to be safe.")
            return True
    
    def _save_build_timestamp(self):
        """Save the current timestamp to track when vectorstore was last built."""
        import time
        
        timestamp_file = os.path.join(self.persist_directory, ".last_build_timestamp")
        try:
            with open(timestamp_file, 'w') as f:
                f.write(str(time.time()))
        except Exception as e:
            logger.warning(f"Failed to save build timestamp: {e}")
    
    def _build_vectorstore(self):
        """Build vector store from documents including PDFs and Jupyter notebooks."""
        if not os.path.exists(self.pdf_folder):
            logger.warning(f"Data folder {self.pdf_folder} does not exist")
            return
        
        documents = []
        
        # Load PDF documents
        logger.info("Loading PDF documents...")
        pdf_loader = DirectoryLoader(
            self.pdf_folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            use_multithreading=True
        )
        pdf_documents = pdf_loader.load()
        documents.extend(pdf_documents)
        logger.info(f"Loaded {len(pdf_documents)} PDF documents")
        
        # Load Jupyter notebooks
        logger.info("Loading Jupyter notebooks...")
        import json
        import glob
        
        notebook_files = glob.glob(os.path.join(self.pdf_folder, "**/*.ipynb"), recursive=True)
        
        for notebook_path in notebook_files:
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                
                # Extract content from notebook cells
                content_parts = []
                for cell in notebook.get('cells', []):
                    if cell.get('cell_type') == 'markdown':
                        source = cell.get('source', [])
                        if isinstance(source, list):
                            content_parts.append(''.join(source))
                        else:
                            content_parts.append(source)
                    elif cell.get('cell_type') == 'code':
                        source = cell.get('source', [])
                        if isinstance(source, list):
                            code_content = ''.join(source)
                        else:
                            code_content = source
                        if code_content.strip():  # Only add non-empty code
                            content_parts.append(f"```python\n{code_content}\n```")
                
                # Create document from notebook content
                if content_parts:
                    full_content = '\n\n'.join(content_parts)
                    doc = Document(
                        page_content=full_content,
                        metadata={
                            'source': os.path.basename(notebook_path),
                            'type': 'jupyter_notebook'
                        }
                    )
                    documents.append(doc)
                    
            except Exception as e:
                logger.warning(f"Failed to load notebook {notebook_path}: {e}")
        
        logger.info(f"Loaded {len(notebook_files)} Jupyter notebooks")
        
        if not documents:
            logger.warning("No documents found")
            return

        logger.info(f"Total documents loaded: {len(documents)}")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        logger.info(f"Vector store built with {len(chunks)} chunks")
        
        # Save build timestamp
        self._save_build_timestamp()
    
    def choose_route(self, question: str, result: RouteQuery) -> Dict[str, Any]:
        """Choose the appropriate route based on the routing decision."""
        datasource = result.datasource
        
        if "local_documents" in datasource.lower():
            if self.retriever is None:
                return {
                    "answer": "Sorry, the document retrieval system is not available.",
                    "sources": []
                }
            
            ragfusion = RAGFusion(question, self.llm)
            generate_queries = ragfusion.generate_queries()
            reciprocal_rank_fusion = ragfusion.reciprocal_rank_fusion
            retrieval_chain_rag_fusion = ragfusion.retrieval_chain_rag_fusion(
                self.retriever, generate_queries, reciprocal_rank_fusion
            )
            
            result = ragfusion.final_rag_chain(retrieval_chain_rag_fusion)
            return {
                "answer": result["answer"],
                "sources": result["sources"]
            }
        
        elif "web_search" in datasource.lower():
            return {
                "answer": self.web_search.web_search(question),
                "sources": ["web_search"]
            }
        
        else:
            return {
                "answer": "This information cannot be found in the textbook nor the internet. Please try again!",
                "sources": []
            }
    
    def invoke(self, question: str) -> Dict[str, Any]:
        """
        Process a question and return the answer.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Route the query
            result = self.router.invoke({"question": question})
            
            # Get the answer and sources
            response = self.choose_route(question, result)
            
            # Determine if web search was used
            web_search_used = "web_search" in result.datasource.lower()
            
            return {
                "answer": response["answer"],
                "datasource": result.datasource,
                "web_search_used": web_search_used,
                "sources": response["sources"]
            }

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "datasource": "error",
                "web_search_used": False,
                "sources": []
            }
    
    async def ainvoke(self, question: str) -> Dict[str, Any]:
        """Async version of invoke."""
        return self.invoke(question)

# Example usage
if __name__ == "__main__":
    print("🤖 Numerical Analysis Knowledge Agent")
    print("=" * 50)
    print("Initializing the system...")
    
    try:
        # Initialize the system
        rag = AgenticRetrieval(
            pdf_folder="./data/",
            persist_directory="./chroma_db"
        )
        print("✅ System initialized successfully!")
        print("💡 You can ask questions about Numerical Analysis topics.")
        print("💡 Type 'quit', 'exit', or 'bye' to stop the program.")
        print("=" * 50)
        
        # Interactive chat loop
        while True:
            try:
                # Get user input
                question = input("\n🧑 Your question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Goodbye! Thanks for using the NA Knowledge Agent!")
                    break
                
                # Skip empty questions
                if not question:
                    print("❌ Please enter a question.")
                    continue
                
                # Process the question
                print("\n🤔 Thinking...")
                result = rag.invoke(question)
                
                # Display the result
                print(f"\n🤖 Answer:")
                print(f"{result['answer']}")
                print(f"\n📊 Source: {result['datasource']}")
                
                if result.get('web_search_used', False):
                    print("🌐 Web search was used for this query.")
                else:
                    print("📚 Answer from local documents.")
                    # Show specific source documents
                    sources = result.get('sources', [])
                    if sources:
                        print(f"📄 Source documents: {', '.join(sources)}")
                    else:
                        print("📄 No specific source documents identified.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using the NA Knowledge Agent!")
                break
            except Exception as e:
                print(f"\n❌ Error processing your question: {str(e)}")
                print("Please try again with a different question.")
                
    except Exception as e:
        print(f"❌ Failed to initialize the system: {str(e)}")
        print("Please check your .env file and make sure all dependencies are installed.")
        exit(1) 