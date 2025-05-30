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
        """Create the final RAG chain."""
        template = """Answer the following question based on this context:

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

        final_rag_chain = (
            {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        rag_fusion_answer = final_rag_chain.invoke({"question": self.question})
        return rag_fusion_answer

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
        if os.path.exists(self.persist_directory) and not self.force_rebuild:
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
        
        # Build new vector store
        logger.info("Building new vector store")
        self._build_vectorstore()
    
    def _build_vectorstore(self):
        """Build vector store from documents."""
        if not os.path.exists(self.pdf_folder):
            logger.warning(f"PDF folder {self.pdf_folder} does not exist")
            return
        
        # Load documents
        loader = DirectoryLoader(
            self.pdf_folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            use_multithreading=True
        )
        documents = loader.load()
        
        if not documents:
            logger.warning("No documents found")
            return
        
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
    
    def choose_route(self, question: str, result: RouteQuery) -> str:
        """Choose the appropriate route based on the routing decision."""
        datasource = result.datasource
        
        if "local_documents" in datasource.lower():
            if self.retriever is None:
                return "Sorry, the document retrieval system is not available."
            
            ragfusion = RAGFusion(question, self.llm)
            generate_queries = ragfusion.generate_queries()
            reciprocal_rank_fusion = ragfusion.reciprocal_rank_fusion
            retrieval_chain_rag_fusion = ragfusion.retrieval_chain_rag_fusion(
                self.retriever, generate_queries, reciprocal_rank_fusion
            )
            
            return ragfusion.final_rag_chain(retrieval_chain_rag_fusion)
        
        elif "web_search" in datasource.lower():
            return self.web_search.web_search(question)
        
        else:
            return "This information cannot be found in the textbook nor the internet. Please try again!"
    
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
            
            # Get the answer
            answer = self.choose_route(question, result)
            
            # Determine if web search was used
            web_search_used = "web_search" in result.datasource.lower()
            
            return {
                "answer": answer,
                "datasource": result.datasource,
                "web_search_used": web_search_used,
                "sources": [] if web_search_used else ["local_documents"]
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
    # Initialize the system
    rag = AgenticRetrieval(
        pdf_folder="./data/",
        persist_directory="./chroma_db"
    )
    
    # Test query
    test_question = "please give me the definition of floating point numbers"
    result = rag.invoke(test_question)
    
    print(f"Question: {test_question}")
    print(f"Answer: {result['answer']}")
    print(f"Datasource: {result['datasource']}") 