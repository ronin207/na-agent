import os
import sys
import logging
import re
from typing import Dict, List, Any, Union, Optional, Literal, Tuple
from dotenv import load_dotenv
import nbformat
import pypdfium2 as pdfium

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

class QueryAnalysis(BaseModel):
    """Analyze user query for better retrieval."""
    
    keywords: List[str] = Field(
        ...,
        description="Extract specific technical terms, function names, exact phrases that should be preserved (e.g., 'np.linalg.eigvals', 'condition number', 'Exercise 5.1')"
    )
    
    document_hints: List[str] = Field(
        default=[],
        description="Extract any specific documents, lectures, or areas mentioned (e.g., 'lecture 5', '3rd.pdf', 'notebook', 'exercise')"
    )
    
    search_intent: Literal["exact_match", "conceptual", "mixed"] = Field(
        ...,
        description="exact_match: looking for specific terms/code; conceptual: understanding concepts; mixed: both specific and conceptual"
    )
    
    topic_area: Optional[str] = Field(
        default=None,
        description="Main topic area (e.g., 'linear algebra', 'interpolation', 'norms', 'exercises')"
    )

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: Literal["local_documents", "web_search", "exercise_solver"] = Field(
        ...,
        description="""Given a user question choose which datasource would be most relevant for answering their question.
        If it is related to Numerical Analysis theory/concepts, choose local_documents.
        If it mentions exercises (e.g., "Exercise 5.1", "solve exercise", "homework"), choose exercise_solver.
        Otherwise, choose web_search.""",
    )

class ExerciseSolver:
    """Enhanced exercise solver that uses lecture notes to solve exercises."""
    
    def __init__(self, llm: ChatOpenAI, retriever, pdf_folder: str):
        self.llm = llm
        self.retriever = retriever
        self.pdf_folder = pdf_folder
    
    def extract_exercise_info(self, query: str) -> Tuple[Optional[str], Optional[float]]:
        """Extract exercise number and section from query."""
        # Pattern to match Exercise X.Y format
        exercise_patterns = [
            r'[Ee]xercise\s*(\d+)\.(\d+)',
            r'[Ee]x\s*(\d+)\.(\d+)',
            r'(\d+)\.(\d+)',
        ]
        
        for pattern in exercise_patterns:
            match = re.search(pattern, query)
            if match:
                chapter = int(match.group(1))
                exercise_num = float(f"{match.group(1)}.{match.group(2)}")
                return f"Exercise{chapter}.{match.group(2)}", exercise_num
        
        return None, None
    
    def find_related_lectures(self, exercise_num: Optional[float]) -> List[str]:
        """Find lecture notes related to the exercise."""
        if exercise_num is None:
            return []
        
        chapter = int(exercise_num)
        related_lectures = []
        
        # Mapping of exercise chapters to lecture notes
        lecture_mapping = {
            1: ["1st.pdf"],
            2: ["2nd.pdf"], 
            3: ["3rd.pdf"],
            4: ["4th.pdf", "LectureNote-4th-Norms.ipynb"],
            5: ["5th.pdf", "LectureNote-5th-6th-LinearEq.ipynb"],
            6: ["LectureNote-5th-6th-LinearEq.ipynb"],
            7: ["LectureNote-7th-8th-NonLinearEq.ipynb"],
            8: ["LectureNote-7th-8th-NonLinearEq.ipynb"],
        }
        
        return lecture_mapping.get(chapter, [])
    
    def load_notebook_with_nbformat(self, notebook_path: str) -> Optional[Document]:
        """Load notebook using nbformat for better control."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            content_parts = []
            exercise_metadata = {
                'type': 'exercise',
                'source': os.path.basename(notebook_path),
                'has_code': False,
                'has_math': False
            }
            
            for cell in nb.cells:
                if cell.cell_type == 'markdown':
                    source_text = ''.join(cell.source) if isinstance(cell.source, list) else cell.source
                    content_parts.append(source_text)
                    
                    # Check for mathematical content
                    if '$' in source_text or '$$' in source_text:
                        exercise_metadata['has_math'] = True
                        
                elif cell.cell_type == 'code':
                    code_content = ''.join(cell.source) if isinstance(cell.source, list) else cell.source
                    if code_content.strip():
                        content_parts.append(f"```python\n{code_content}\n```")
                        exercise_metadata['has_code'] = True
            
            if content_parts:
                full_content = '\n\n'.join(content_parts)
                return Document(
                    page_content=full_content,
                    metadata=exercise_metadata
                )
                
        except Exception as e:
            logging.warning(f"Failed to load notebook {notebook_path} with nbformat: {e}")
        
        return None
    
    def solve_exercise(self, query: str) -> Dict[str, Any]:
        """Solve exercise using related lecture materials."""
        # Extract exercise information
        exercise_file, exercise_num = self.extract_exercise_info(query)
        
        # Load the specific exercise if found
        exercise_content = ""
        if exercise_file:
            exercise_path = os.path.join(self.pdf_folder, f"{exercise_file}.ipynb")
            if os.path.exists(exercise_path):
                exercise_doc = self.load_notebook_with_nbformat(exercise_path)
                if exercise_doc:
                    exercise_content = exercise_doc.page_content
        
        # Find related lecture materials
        related_lectures = self.find_related_lectures(exercise_num)
        
        # Retrieve relevant context using RAG
        rag_results = self.retriever.invoke(query)
        context_docs = [doc.page_content for doc in rag_results]
        
        # Enhanced prompt for exercise solving
        template = """You are an expert numerical analysis tutor helping a student solve exercises using lecture materials.

**Exercise Query:** {query}

**Exercise Content:**
{exercise_content}

**Related Lecture Context:**
{context}

**Related Lecture Files:** {related_lectures}

Please provide a comprehensive solution that:

1. **Problem Analysis**: Clearly state what the exercise is asking for
2. **Theoretical Foundation**: Explain the relevant theory from the lecture notes
3. **Solution Steps**: Provide step-by-step solution with mathematical reasoning
4. **Code Implementation**: If applicable, provide Python code with explanations
5. **Verification**: Show how to verify the solution
6. **Learning Notes**: Highlight key concepts the student should understand

When including mathematical expressions:
- Use $...$ for inline math
- Use $$...$$ for equation blocks

Format your response clearly with appropriate headers and explanations suitable for learning.
"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Prepare context
        context_text = "\n\n".join(context_docs[:5])  # Use top 5 context documents
        related_lectures_text = ", ".join(related_lectures) if related_lectures else "General numerical analysis materials"
        
        # Generate solution
        chain = prompt | self.llm | StrOutputParser()
        
        solution = chain.invoke({
            "query": query,
            "exercise_content": exercise_content if exercise_content else "Exercise content not found in local files.",
            "context": context_text,
            "related_lectures": related_lectures_text
        })
        
        # Extract sources
        sources = related_lectures.copy()
        if exercise_file:
            sources.append(f"{exercise_file}.ipynb")
        
        # Add sources from RAG results
        for doc in rag_results:
            source = doc.metadata.get('source', '')
            if source and source not in sources:
                # Extract just the filename
                if '/' in source:
                    source = source.split('/')[-1]
                sources.append(source)
        
        return {
            "answer": solution,
            "sources": sources[:10],  # Limit to top 10 sources
            "exercise_detected": exercise_file is not None,
            "related_lectures": related_lectures
        }

class EnhancedRAGFusion:
    def __init__(self, question: str, llm: ChatOpenAI, query_analyzer):
        self.question = question
        self.llm = llm
        self.query_analyzer = query_analyzer

    def analyze_query(self) -> QueryAnalysis:
        """Analyze the query to understand user intent."""
        return self.query_analyzer.invoke({"question": self.question})

    def generate_enhanced_queries(self, analysis: QueryAnalysis):
        """Generate queries based on analysis, preserving important keywords."""
        
        keywords_text = ", ".join(analysis.keywords) if analysis.keywords else "none"
        document_hints_text = ", ".join(analysis.document_hints) if analysis.document_hints else "none"
        
        if analysis.search_intent == "exact_match":
            template = """The user is looking for EXACT information. Generate 3 search queries that preserve the specific terms and keywords.
            
            Original query: {question}
            Keywords to preserve: {keywords}
            Document hints: {document_hints}
            
            IMPORTANT: Always include the exact keywords in each query variation.
            
            Generate 3 queries that:
            1. Keep exact technical terms unchanged
            2. Add context around the keywords
            3. Vary the phrasing while preserving specificity
            
            Output (3 queries):"""
            
        elif analysis.search_intent == "conceptual":
            template = """The user wants conceptual understanding. Generate 4 related queries for broader context.
            
            Original query: {question}
            Topic area: {topic_area}
            Document hints: {document_hints}
            
            Generate 4 queries that explore:
            1. The main concept
            2. Related theory
            3. Applications
            4. Examples or implementations
            
            Output (4 queries):"""
            
        else:  # mixed
            template = """The user needs both specific information and conceptual understanding.
            
            Original query: {question}
            Keywords to preserve: {keywords}
            Topic area: {topic_area}
            Document hints: {document_hints}
            
            Generate 4 queries:
            1. Exact keyword match query
            2. Conceptual understanding query
            3. Implementation/application query
            4. Related context query
            
            Always preserve exact keywords in appropriate queries.
            
            Output (4 queries):"""
        
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)
        
        generate_queries = (
            prompt_rag_fusion
            | self.llm
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.strip().split("\n") if q.strip() and not q.strip().isdigit()])
        )
        
        return generate_queries.invoke({
            "question": self.question,
            "keywords": keywords_text,
            "document_hints": document_hints_text,
            "topic_area": analysis.topic_area or "numerical analysis"
        })

    def filter_by_document_hints(self, docs: List[Document], analysis: QueryAnalysis) -> List[Document]:
        """Filter documents based on document hints in the query."""
        if not analysis.document_hints:
            return docs
        
        filtered_docs = []
        hint_keywords = [hint.lower() for hint in analysis.document_hints]
        
        for doc in docs:
            source = doc.metadata.get('source', '').lower()
            # Check if any hint matches the source
            for hint in hint_keywords:
                if (hint in source or 
                    any(keyword in source for keyword in hint.split()) or
                    any(keyword in hint for keyword in source.split('.'))):
                    filtered_docs.append(doc)
                    break
        
        # If filtering results in too few docs, return original
        return filtered_docs if len(filtered_docs) >= 2 else docs

    def exact_keyword_boost(self, docs: List[Document], analysis: QueryAnalysis) -> List[Document]:
        """Boost documents that contain exact keywords."""
        if not analysis.keywords:
            return docs
        
        # Score documents based on exact keyword matches
        scored_docs = []
        for doc in docs:
            score = 0
            content_lower = doc.page_content.lower()
            
            for keyword in analysis.keywords:
                keyword_lower = keyword.lower()
                # Exact match gets higher score
                if keyword_lower in content_lower:
                    score += 10
                # Partial match for compound terms
                elif any(part in content_lower for part in keyword_lower.split('.')):
                    score += 3
                
            scored_docs.append((doc, score))
        
        # Sort by score (descending) then return docs
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]

    def reciprocal_rank_fusion(self, results: List[List], k: int = 60):
        """Enhanced reciprocal rank fusion with keyword boosting."""
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

    def enhanced_retrieval_chain(self, retriever, analysis: QueryAnalysis):
        """Create enhanced retrieval chain with filtering and boosting."""
        
        # Generate enhanced queries
        queries = self.generate_enhanced_queries(analysis)
        
        # Retrieve documents for each query
        all_results = []
        for query in queries:
            if isinstance(query, str) and query.strip():
                docs = retriever.invoke(query.strip())
                
                # Apply document filtering if hints provided
                filtered_docs = self.filter_by_document_hints(docs, analysis)
                
                # Apply keyword boosting
                boosted_docs = self.exact_keyword_boost(filtered_docs, analysis)
                
                all_results.append(boosted_docs)
        
        # Apply reciprocal rank fusion
        if all_results:
            return self.reciprocal_rank_fusion(all_results)
        else:
            # Fallback to original query
            docs = retriever.invoke(self.question)
            return [(doc, 1.0) for doc in docs]

    def final_rag_chain(self, retriever):
        """Create the final enhanced RAG chain."""
        # Analyze query
        analysis = self.analyze_query()
        
        # Get enhanced retrieval results
        ranked_docs = self.enhanced_retrieval_chain(retriever, analysis)
        
        # Extract top documents for context
        context_docs = [doc for doc, score in ranked_docs[:6]]  # Use top 6 docs
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Enhanced template based on search intent
        if analysis.search_intent == "exact_match":
            template = """Answer the question with focus on the EXACT terms and specific information requested.

            Context:
            {context}

            Question: {question}
            Keywords to address: {keywords}

            Provide specific, precise information. If exact terms like function names or specific concepts are mentioned, include them directly in your response.

            When including mathematical expressions:
            - Use $...$ for inline math
            - Use $$...$$ for equation blocks

            If the specific information is not found, clearly state what is missing."""
            
        else:
            template = """Answer the following question based on this context:

            {context}

            Question: {question}

            When including mathematical expressions:
            - Use $...$ for inline math
            - Use $$...$$ for equation blocks

            However, if the question's contents are not in the retriever, please state how the information
            cannot be found in the textbook. Say something like: "The information is not stated in the textbook,
            so you would not be deeply required to understand it as of now."
            """

        prompt = ChatPromptTemplate.from_template(template)
        
        # Generate the answer
        if analysis.search_intent == "exact_match":
            answer = prompt.invoke({
                "context": context_text, 
                "question": self.question,
                "keywords": ", ".join(analysis.keywords)
            })
        else:
            answer = prompt.invoke({
                "context": context_text, 
                "question": self.question
            })
        
        final_chain = self.llm | StrOutputParser()
        rag_fusion_answer = final_chain.invoke(answer)
        
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
            "sources": unique_sources,
            "analysis": analysis
        }

# Keeping the original RAGFusion for backward compatibility
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
        - Use $...$ for inline math (not \\(...\\))
        - Use $$...$$ for equation blocks (not \\[...\\])

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
    Enhanced Agentic Retrieval system with improved query understanding and precision.
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
        
        # Set up router and query analyzer
        self._setup_router()
        self._setup_query_analyzer()
        
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
    
    def _setup_query_analyzer(self):
        """Set up the query analyzer for better understanding."""
        structured_llm = self.llm.with_structured_output(QueryAnalysis)
        
        system_prompt = """You are an expert at analyzing user queries to understand their intent and extract key information.
        
        Your job is to:
        1. Extract specific technical terms, function names, exact phrases that should be preserved
        2. Identify any document or area hints the user provided
        3. Determine if they want exact matches or conceptual understanding
        4. Identify the main topic area
        
        Be especially careful to preserve:
        - Function names (e.g., np.linalg.eigvals, matplotlib.pyplot)
        - Technical terms (e.g., "condition number", "LU decomposition")
        - Specific references (e.g., "Exercise 5.1", "lecture 3", "Definition 2.1")
        - Code snippets or specific syntax
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        self.query_analyzer = prompt | structured_llm

    def _setup_vectorstore(self):
        """Set up or load the vector store."""
        if self.force_rebuild or not os.path.exists(self.persist_directory):
            logger.info("Building new vector store...")
            self._build_vectorstore()
        else:
            # Check if data has changed
            if self._check_if_data_changed():
                logger.info("Data folder has been modified since last build. Rebuilding vector store...")
                self._build_vectorstore()
            else:
                logger.info("Loading existing vector store from " + self.persist_directory)
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
                logger.info("Vector store loaded successfully")
            except Exception as e:
                    logger.warning(f"Failed to load existing vector store: {e}")
                    logger.info("Rebuilding vector store...")
        self._build_vectorstore()
    
    def _check_if_data_changed(self):
        """Check if the data folder has been modified since the last vector store build."""
        timestamp_file = os.path.join(self.persist_directory, ".last_build_timestamp")
        
        if not os.path.exists(timestamp_file):
            return True
            
        try:
            with open(timestamp_file, 'r') as f:
                last_build_time = float(f.read().strip())
        except:
            return True
        
        # Check if any file in the data folder is newer than the last build
        for root, dirs, files in os.walk(self.pdf_folder):
            for file in files:
                if file.endswith(('.pdf', '.ipynb')):
                    file_path = os.path.join(root, file)
                if os.path.getmtime(file_path) > last_build_time:
                    return True
                    
            return False
    
    def _save_build_timestamp(self):
        """Save the current timestamp to track when the vector store was built."""
        timestamp_file = os.path.join(self.persist_directory, ".last_build_timestamp")
        os.makedirs(self.persist_directory, exist_ok=True)
        
        with open(timestamp_file, 'w') as f:
            f.write(str(os.path.getmtime(self.pdf_folder)))
    
    def _build_vectorstore(self):
        """Build vector store from documents including PDFs and Jupyter notebooks."""
        if not os.path.exists(self.pdf_folder):
            logger.warning(f"Data folder {self.pdf_folder} does not exist")
            return
        
        documents = []
        
        # Load PDF documents
        logger.info("Loading PDF documents...")
        import glob
        pdf_files = glob.glob(os.path.join(self.pdf_folder, "**/*.pdf"), recursive=True)

        for pdf_path in pdf_files:
            try:
                password = None
                if "PartI-midtermExam2024.pdf" in os.path.basename(pdf_path):
                    password = os.getenv("part1_midterm")

                with pdfium.PdfDocument(pdf_path, password=password) as doc:
                    for i, page in enumerate(doc):
                        text_page = page.get_textpage()
                        text = text_page.get_text_range()
                        documents.append(Document(page_content=text, metadata={"source": os.path.basename(pdf_path), "page": i}))
            except Exception as e:
                logger.error(f"Failed to load PDF {pdf_path}: {e}")

        logger.info(f"Loaded {len(documents)} pages from PDF documents.")

        # Load Jupyter notebooks
        logger.info("Loading Jupyter notebooks...")
        
        notebook_files = glob.glob(os.path.join(self.pdf_folder, "**/*.ipynb"), recursive=True)
        
        for notebook_path in notebook_files:
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                
                # Extract content from notebook cells using nbformat
                content_parts = []
                notebook_metadata = {
                    'source': os.path.basename(notebook_path),
                    'type': 'jupyter_notebook',
                    'has_code': False,
                    'has_math': False
                }
                
                # Detect exercise vs lecture note
                notebook_name = os.path.basename(notebook_path).lower()
                if 'exercise' in notebook_name:
                    notebook_metadata['type'] = 'exercise'
                elif 'lecture' in notebook_name:
                    notebook_metadata['type'] = 'lecture_note'
                
                for cell in nb.cells:
                    if cell.cell_type == 'markdown':
                        source_text = ''.join(cell.source) if isinstance(cell.source, list) else cell.source
                        content_parts.append(source_text)
                        
                        # Check for mathematical content
                        if '$' in source_text or '$$' in source_text:
                            notebook_metadata['has_math'] = True
                            
                    elif cell.cell_type == 'code':
                        code_content = ''.join(cell.source) if isinstance(cell.source, list) else cell.source
                        if code_content.strip():  # Only add non-empty code
                            content_parts.append(f"```python\n{code_content}\n```")
                            notebook_metadata['has_code'] = True
                
                # Create document from notebook content
                if content_parts:
                    full_content = '\n\n'.join(content_parts)
                    doc = Document(
                        page_content=full_content,
                        metadata=notebook_metadata
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
        """Choose the appropriate route based on the routing decision with enhanced processing."""
        datasource = result.datasource
        
        if "local_documents" in datasource.lower():
            if self.retriever is None:
                return {
                    "answer": "Sorry, the document retrieval system is not available.",
                    "sources": []
                }
            
            # Use enhanced RAG fusion for better query understanding
            enhanced_ragfusion = EnhancedRAGFusion(question, self.llm, self.query_analyzer)
            result = enhanced_ragfusion.final_rag_chain(self.retriever)
            
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "query_analysis": result.get("analysis")
            }
        
        elif "web_search" in datasource.lower():
            return {
                "answer": self.web_search.web_search(question),
                "sources": ["web_search"]
            }
        
        elif "exercise_solver" in datasource.lower():
            exercise_solver = ExerciseSolver(self.llm, self.retriever, self.pdf_folder)
            return exercise_solver.solve_exercise(question)
        
        else:
            return {
                "answer": "This information cannot be found in the textbook nor the internet. Please try again!",
                "sources": []
            }
    
    def invoke(self, question: str) -> Dict[str, Any]:
        """
        Process a question and return the answer with enhanced query understanding.
        
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
            
            # Base response structure
            response_data = {
                "response": response["answer"],  # Use 'response' for compatibility with existing API
                "datasource": result.datasource,
                "web_search_used": web_search_used,
                "sources": response["sources"]
            }
            
            # Add query analysis for debugging (if available)
            if "query_analysis" in response:
                response_data["query_analysis"] = {
                    "keywords": response["query_analysis"].keywords,
                    "search_intent": response["query_analysis"].search_intent,
                    "document_hints": response["query_analysis"].document_hints,
                    "topic_area": response["query_analysis"].topic_area
                }
            
            # Add exercise-specific metadata if available
            if "exercise_solver" in result.datasource.lower():
                response_data.update({
                    "exercise_detected": response.get("exercise_detected", False),
                    "related_lectures": response.get("related_lectures", [])
                })
            
            return response_data

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
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
                print(f"{result['response']}")
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