import os
import sys
import logging
import re
from typing import Dict, List, Any, Union, Optional, Literal, Tuple
from dotenv import load_dotenv
import nbformat
from datetime import datetime, timedelta
import json
import pypdfium2 as pdfium

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pydantic.v1 import BaseModel, Field
from operator import itemgetter
from openai import OpenAI

# Import conversation-related modules from Langchain
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import BaseMemory

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

class ConversationSession:
    """Manages individual conversation sessions with memory."""
    
    def __init__(self, session_id: str, llm: ChatOpenAI, max_token_limit: int = 2000):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        
        # Use ConversationSummaryBufferMemory for better token management
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_token_limit,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Track conversation context
        self.conversation_context = {
            "topic_area": None,
            "current_exercise": None,
            "preferred_datasource": None,
            "user_preferences": {}
        }
    
    def add_exchange(self, human_message: str, ai_message: str):
        """Add a human-AI exchange to memory."""
        self.memory.chat_memory.add_user_message(human_message)
        self.memory.chat_memory.add_ai_message(ai_message)
        self.last_accessed = datetime.now()
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history."""
        return self.memory.buffer
    
    def update_context(self, analysis: QueryAnalysis, datasource: str):
        """Update conversation context based on current query."""
        if analysis.topic_area:
            self.conversation_context["topic_area"] = analysis.topic_area
        
        # Track exercise progression
        for keyword in analysis.keywords:
            if "exercise" in keyword.lower():
                self.conversation_context["current_exercise"] = keyword
        
        self.conversation_context["preferred_datasource"] = datasource
        self.last_accessed = datetime.now()
    
    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Check if session has expired."""
        return datetime.now() - self.last_accessed > timedelta(hours=timeout_hours)

class ConversationManager:
    """Manages multiple conversation sessions."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.sessions: Dict[str, ConversationSession] = {}
        self.session_timeout_hours = 24
    
    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Get existing session or create new one."""
        self.cleanup_expired_sessions()
        
        if session_id not in self.sessions:
            logger.info(f"Creating new conversation session: {session_id}")
            self.sessions[session_id] = ConversationSession(session_id, self.llm)
        else:
            logger.info(f"Using existing conversation session: {session_id}")
            self.sessions[session_id].last_accessed = datetime.now()
        
        return self.sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions to free memory."""
        expired_sessions = [
            sid for sid, session in self.sessions.items() 
            if session.is_expired(self.session_timeout_hours)
        ]
        
        for sid in expired_sessions:
            logger.info(f"Removing expired session: {sid}")
            del self.sessions[sid]
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about active sessions."""
        return {
            "active_sessions": len(self.sessions),
            "sessions": {
                sid: {
                    "created_at": session.created_at.isoformat(),
                    "last_accessed": session.last_accessed.isoformat(),
                    "topic_area": session.conversation_context.get("topic_area"),
                    "current_exercise": session.conversation_context.get("current_exercise")
                }
                for sid, session in self.sessions.items()
            }
        }

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
        
        # Check for midterm questions
        midterm_patterns = [
            r'[Qq]\s*(\d+)-(\d+).*midterm',
            r'[Qq]uestion\s*(\d+)-(\d+).*midterm',
            r'midterm.*[Qq]\s*(\d+)-(\d+)',
            r'midterm.*[Qq]uestion\s*(\d+)-(\d+)',
            r'[Qq]\s*(\d+).*midterm',
            r'[Qq]uestion\s*(\d+).*midterm',
        ]
        
        for pattern in midterm_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:  # Q1-2 format
                    return f"midterm_Q{match.group(1)}-{match.group(2)}", float(f"{match.group(1)}.{match.group(2)}")
                else:  # Q1 format
                    return f"midterm_Q{match.group(1)}", float(match.group(1))
        
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
    
    def solve_exercise(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        """Solve exercise using related lecture materials with conversation context."""
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
        
        # Enhanced prompt for exercise solving with conversation awareness
        template = """You are an expert numerical analysis tutor helping a student solve problems using lecture materials and your knowledge.

**Previous Conversation:**
{conversation_history}

**Current Query:** {query}

**Problem Content Found:**
{exercise_content}

**Related Context from Course Materials:**
{context}

**Related Files:** {related_lectures}

INSTRUCTIONS:
1. If you found the actual problem/question in the content above, SOLVE IT using your numerical analysis expertise.
2. If the problem content is incomplete or missing, use the context and your knowledge to provide the best possible guidance.
3. For midterm/exam questions, attempt to solve them even if explicit answers aren't provided in the materials.

Please provide a comprehensive response that includes:

1. **Problem Analysis**: Clearly state what the problem is asking for (based on what you found or can infer)
2. **Connection to Previous Discussion**: If relevant, reference our previous conversation  
3. **Theoretical Foundation**: Explain the relevant numerical analysis theory
4. **Solution Approach**: Provide your best attempt at solving the problem with mathematical reasoning
5. **Step-by-Step Solution**: Show detailed solution steps
6. **Code Implementation**: If applicable, provide Python code with explanations
7. **Verification**: Show how to verify the solution
8. **Learning Notes**: Highlight key concepts the student should understand

When including mathematical expressions:
- Use $...$ for inline math
- Use $$...$$ for equation blocks

Be proactive and solution-oriented. If specific details are missing, make reasonable assumptions and explain your approach.
"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Prepare context
        context_text = "\n\n".join(context_docs[:5])  # Use top 5 context documents
        related_lectures_text = ", ".join(related_lectures) if related_lectures else "General numerical analysis materials"
        
        # Generate solution
        chain = prompt | self.llm | StrOutputParser()
        
        solution = chain.invoke({
            "query": query,
            "conversation_history": conversation_history if conversation_history else "No previous conversation.",
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

class ConversationalEnhancedRAGFusion:
    """Enhanced RAG Fusion with conversation awareness."""
    
    def __init__(self, question: str, llm: ChatOpenAI, query_analyzer, conversation_history: str = ""):
        self.question = question
        self.llm = llm
        self.query_analyzer = query_analyzer
        self.conversation_history = conversation_history

    def analyze_query(self) -> QueryAnalysis:
        """Analyze the query to understand user intent."""
        return self.query_analyzer.invoke({"question": self.question})

    def generate_enhanced_queries(self, analysis: QueryAnalysis):
        """Generate queries based on analysis and conversation context."""
        
        keywords_text = ", ".join(analysis.keywords) if analysis.keywords else "none"
        document_hints_text = ", ".join(analysis.document_hints) if analysis.document_hints else "none"
        
        # Include conversation context in query generation
        context_prompt = ""
        if self.conversation_history:
            context_prompt = f"\n\nPrevious conversation context:\n{self.conversation_history[-500:]}"  # Last 500 chars
        
        if analysis.search_intent == "exact_match":
            template = """The user is looking for EXACT information. Generate 3 search queries that preserve the specific terms and keywords.
            
            Original query: {question}
            Keywords to preserve: {keywords}
            Document hints: {document_hints}{context_prompt}
            
            IMPORTANT: Always include the exact keywords in each query variation.
            Consider the conversation context when generating queries.
            
            Generate 3 queries that:
            1. Keep exact technical terms unchanged
            2. Add context around the keywords
            3. Vary the phrasing while preserving specificity
            
            Output (3 queries):"""
            
        elif analysis.search_intent == "conceptual":
            template = """The user wants conceptual understanding. Generate 4 related queries for broader context.
            
            Original query: {question}
            Topic area: {topic_area}
            Document hints: {document_hints}{context_prompt}
            
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
            Document hints: {document_hints}{context_prompt}
            
            Generate 4 queries:
            1. Exact keyword match query
            2. Conceptual understanding query
            3. Implementation/application query
            4. Related context query
            
            Always preserve exact keywords in appropriate queries.
            Consider the conversation context when relevant.
            
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
            "topic_area": analysis.topic_area or "numerical analysis",
            "context_prompt": context_prompt
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
        """Create the final enhanced RAG chain with conversation awareness."""
        # Analyze query
        analysis = self.analyze_query()
        
        # Get enhanced retrieval results
        ranked_docs = self.enhanced_retrieval_chain(retriever, analysis)
        
        # Extract top documents for context
        context_docs = [doc for doc, score in ranked_docs[:6]]  # Use top 6 docs
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Enhanced template based on search intent with conversation awareness
        if analysis.search_intent == "exact_match":
            template = """You are a helpful numerical analysis assistant. Answer the question with focus on the EXACT terms and specific information requested.

            Previous conversation:
            {conversation_history}

            Context:
            {context}

            Question: {question}
            Keywords to address: {keywords}

            Based on our previous conversation (if any) and the current context, provide specific, precise information. If exact terms like function names or specific concepts are mentioned, include them directly in your response.

            When including mathematical expressions:
            - Use $...$ for inline math
            - Use $$...$$ for equation blocks

            If the specific information is not found, clearly state what is missing."""
            
        else:
            template = """You are a helpful numerical analysis assistant. Answer the following question based on the context and our previous conversation.

            Previous conversation:
            {conversation_history}

            Context:
            {context}

            Question: {question}

            When including mathematical expressions:
            - Use $...$ for inline math
            - Use $$...$$ for equation blocks

            IMPORTANT INSTRUCTIONS:
            1. If the user asks to solve specific questions from a document (like "Q1-2 from midterm"), first check if you can find the actual questions in the context.
            2. If you find the questions but not explicit answers, ATTEMPT TO SOLVE THEM using:
               - Your knowledge of numerical analysis concepts
               - Related information from the context
               - Standard mathematical approaches
            3. If you find questions in the context, DO NOT say "the information is not found" - instead, provide your best attempt at solving them.
            4. Only if the actual questions themselves are completely missing from the context, then mention that the specific questions are not available.
            5. If this is a follow-up question, reference relevant parts of our previous conversation.
            
            Provide helpful, solution-oriented responses rather than just stating limitations."""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Generate the answer
        if analysis.search_intent == "exact_match":
            answer = prompt.invoke({
                "context": context_text, 
                "question": self.question,
                "keywords": ", ".join(analysis.keywords),
                "conversation_history": self.conversation_history if self.conversation_history else "No previous conversation."
            })
        else:
            answer = prompt.invoke({
                "context": context_text, 
                "question": self.question,
                "conversation_history": self.conversation_history if self.conversation_history else "No previous conversation."
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
            "analysis": analysis,
            "context_docs": context_docs  # Keep for source filtering
        }

class SourceRelevanceFilter:
    """Filter and score sources based on topic relevance and query analysis."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # Topic mapping for keyword-to-topic matching
        self.topic_keywords = {
            "floating_point": ["floating point", "machine epsilon", "precision", "ieee 754", "binary representation"],
            "numerical_errors": ["error", "absolute error", "relative error", "rounding", "truncation"],
            "linear_equations": ["linear system", "gaussian elimination", "lu decomposition", "matrix", "linear algebra"],
            "condition_number": ["condition number", "stability", "ill-conditioned", "well-conditioned"],
            "norms": ["norm", "vector norm", "matrix norm", "euclidean", "manhattan", "infinity norm"],
            "nonlinear_equations": ["newton method", "newton's method", "nonlinear", "root finding", "fixed point"],
            "iterative_methods": ["jacobi", "gauss-seidel", "conjugate gradient", "iterative"],
            "programming": ["python", "programming", "implementation", "code", "algorithm"],
            "introduction": ["numerical analysis", "course", "overview", "basics"],
            "exercises": ["exercise", "problem", "homework", "solution"]
        }
    
    def extract_query_topics(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Extract relevant topics from query and analysis."""
        query_lower = query.lower()
        extracted_topics = []
        
        # Use analysis topic area if available
        if analysis.topic_area:
            extracted_topics.append(analysis.topic_area.lower().replace(" ", "_"))
        
        # Match keywords to topics
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                extracted_topics.append(topic)
        
        # Extract from analysis keywords
        for keyword in analysis.keywords:
            keyword_lower = keyword.lower()
            for topic, topic_keywords in self.topic_keywords.items():
                if any(tk in keyword_lower for tk in topic_keywords):
                    if topic not in extracted_topics:
                        extracted_topics.append(topic)
        
        return list(set(extracted_topics))  # Remove duplicates
    
    def calculate_source_relevance_score(self, doc, query_topics: List[str]) -> float:
        """Calculate relevance score for a document based on query topics."""
        doc_topics_str = doc.metadata.get('topics', '')
        doc_topics = doc_topics_str.split(',') if doc_topics_str else []
        doc_type = doc.metadata.get('type', '')
        source = doc.metadata.get('source', '').lower()
        
        if not query_topics:
            return 0.5  # Neutral score if no topics identified
        
        score = 0.0
        
        # Direct topic match scoring
        topic_matches = len(set(query_topics) & set(doc_topics))
        if topic_matches > 0:
            score += topic_matches / len(query_topics) * 0.8  # High weight for topic matches
        
        # Document type relevance
        if any(qt in ['exercises', 'exercise'] for qt in query_topics):
            if doc_type == 'exercise':
                score += 0.3
        elif any(qt in ['programming', 'python', 'implementation'] for qt in query_topics):
            if 'exercise1' in source or 'python' in source:
                score += 0.4
        
        # Penalize completely unrelated sources
        if score == 0.0 and doc_topics:
            # Check if there's any semantic overlap
            for qt in query_topics:
                for dt in doc_topics:
                    if qt in dt or dt in qt:
                        score += 0.1
                        break
        
        return min(score, 1.0)  # Cap at 1.0
    
    def filter_sources_by_relevance(self, context_docs: List[Document], query: str, 
                                   analysis: QueryAnalysis, threshold: float = 0.2) -> List[Document]:
        """Filter documents by relevance score and return top relevant sources."""
        query_topics = self.extract_query_topics(query, analysis)
        
        # Score all documents
        scored_docs = []
        for doc in context_docs:
            score = self.calculate_source_relevance_score(doc, query_topics)
            if score >= threshold:  # Only keep docs above threshold
                scored_docs.append((doc, score))
        
        # Sort by score (descending) and return top documents
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 4 most relevant documents
        filtered_docs = [doc for doc, score in scored_docs[:4]]
        
        # If no docs pass threshold, return top 2 from original
        if not filtered_docs and context_docs:
            logger.warning(f"No sources met relevance threshold {threshold}. Using top 2 sources.")
            return context_docs[:2]
        
        return filtered_docs

class WebSearch:
    def __init__(self, client: OpenAI):
        self.client = client

    def web_search(self, query: str, conversation_history: str = ""):
        """Perform web search using OpenAI's search capabilities with conversation context."""
        try:
            # Include conversation context in web search if relevant
            enhanced_query = query
            if conversation_history:
                enhanced_query = f"Previous context: {conversation_history[-200:]}\n\nCurrent question: {query}"
            
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini-search-preview",
                messages=[
                    {
                        "role": "user",
                        "content": enhanced_query
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

class ConversationalAgenticRetrieval:
    """
    Enhanced Conversational Agentic Retrieval system with conversation memory and session management.
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
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(self.llm)
        
        # Initialize source relevance filter
        self.source_filter = SourceRelevanceFilter(self.llm)
        
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
        
        Route to exercise_solver if the query:
        - Asks to SOLVE specific questions from any document (like "solve Q1-2 from midterm", "how to solve midterm question 1")
        - Asks to solve a specific exercise (like "solve Exercise 5.1")
        - Wants step-by-step solutions for homework problems
        - Uses action words like "solve", "calculate", "find the answer to"
        
        Route to local_documents if the query:
        - Asks ABOUT midterm/exam content without requesting solutions (like "what is covered in midterm")
        - Asks about specific content from lecture notes or course materials
        - Wants to understand concepts from the textbook or course materials
        - Is asking for information retrieval rather than problem solving
        
        Route to web_search if the query:
        - Asks about things not covered in the course materials
        - Requires current information from the internet
        
        Key distinction: Use exercise_solver for SOLVING/CALCULATING, use local_documents for FINDING/UNDERSTANDING."""
        
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

                doc = pdfium.PdfDocument(pdf_path, password=password)
                for i, page in enumerate(doc):
                    text_page = page.get_textpage()
                    text = text_page.get_text_range()
                    
                    # Add topic tags based on PDF filename
                    pdf_name = os.path.basename(pdf_path).lower()
                    pdf_metadata = {
                        "source": os.path.basename(pdf_path), 
                        "page": i,
                        "type": "lecture_pdf",
                        "topics": ""
                    }
                    
                    if "1st.pdf" in pdf_name:
                        pdf_metadata["topics"] = "introduction,course_overview,numerical_analysis_basics"
                    elif "2nd.pdf" in pdf_name:
                        pdf_metadata["topics"] = "floating_point,numerical_errors,computer_arithmetic,error_analysis"
                    elif "3rd.pdf" in pdf_name:
                        pdf_metadata["topics"] = "floating_point,machine_epsilon,precision,rounding_errors"
                    elif "4th.pdf" in pdf_name:
                        pdf_metadata["topics"] = "exercises,ai_guidelines,academic_integrity"
                    elif "5th.pdf" in pdf_name:
                        pdf_metadata["topics"] = "linear_equations,gaussian_elimination,lu_decomposition,iterative_methods,linear_algebra"
                    elif "midterm" in pdf_name:
                        pdf_metadata["topics"] = "midterm,exam,comprehensive_review"
                        pdf_metadata["type"] = "exam"
                    
                    documents.append(Document(page_content=text, metadata=pdf_metadata))
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
                    'has_math': False,
                    'topics': ""  # Will be populated based on filename and content
                }
                
                # Detect type and add topic tags based on filename
                notebook_name = os.path.basename(notebook_path).lower()
                if 'exercise' in notebook_name:
                    notebook_metadata['type'] = 'exercise'
                    # Extract exercise number for topic mapping
                    if 'exercise1' in notebook_name:
                        notebook_metadata['topics'] = 'programming,python_basics,introduction'
                    elif 'exercise2' in notebook_name:
                        notebook_metadata['topics'] = 'floating_point,numerical_errors,error_analysis'
                    elif 'exercise3' in notebook_name:
                        notebook_metadata['topics'] = 'floating_point,machine_epsilon,precision'
                    elif 'exercise4' in notebook_name:
                        notebook_metadata['topics'] = 'norms,vector_norms,matrix_norms'
                    elif 'exercise5' in notebook_name:
                        notebook_metadata['topics'] = 'linear_equations,condition_number,linear_algebra'
                    elif 'exercise6' in notebook_name:
                        notebook_metadata['topics'] = 'linear_equations,iterative_methods'
                    elif 'exercise7' in notebook_name:
                        notebook_metadata['topics'] = 'nonlinear_equations,newton_method,root_finding'
                elif 'lecture' in notebook_name:
                    notebook_metadata['type'] = 'lecture_note'
                    # Map lecture notes to topics
                    if '4th' in notebook_name or 'norms' in notebook_name:
                        notebook_metadata['topics'] = 'norms,vector_norms,matrix_norms,linear_algebra'
                    elif '5th' in notebook_name or '6th' in notebook_name or 'lineareq' in notebook_name:
                        notebook_metadata['topics'] = 'linear_equations,condition_number,iterative_methods,conjugate_gradient,linear_algebra'
                    elif '7th' in notebook_name or '8th' in notebook_name or 'nonlineareq' in notebook_name:
                        notebook_metadata['topics'] = 'nonlinear_equations,newton_method,fixed_point,root_finding'
                elif 'python_guidelines' in notebook_name:
                    notebook_metadata['topics'] = 'programming,python_guidelines,coding_standards'
                
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

        # Split documents with smaller, more focused chunks for better topic precision
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for more focused content
            chunk_overlap=50,  # Maintain overlap for context
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]  # Better separators
        )
        chunks = text_splitter.split_documents(documents)
        
        # Remove duplicate chunks
        unique_chunks = []
        seen_content = set()
        for chunk in chunks:
            # Use first 200 chars as deduplication key
            content_key = chunk.page_content[:200].strip()
            if content_key not in seen_content and len(content_key) > 20:  # Skip very short chunks
                seen_content.add(content_key)
                unique_chunks.append(chunk)
        
        logger.info(f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
        chunks = unique_chunks
        
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
    
    def choose_route(self, question: str, result: RouteQuery, session: ConversationSession) -> Dict[str, Any]:
        """Choose the appropriate route based on the routing decision with conversation awareness."""
        datasource = result.datasource
        conversation_history = session.get_conversation_history()
        
        if "local_documents" in datasource.lower():
            if self.retriever is None:
                return {
                    "answer": "Sorry, the document retrieval system is not available.",
                    "sources": []
                }
            
            # Use conversational enhanced RAG fusion
            conversational_ragfusion = ConversationalEnhancedRAGFusion(
                question, self.llm, self.query_analyzer, conversation_history
            )
            rag_result = conversational_ragfusion.final_rag_chain(self.retriever)
            
            # Apply source relevance filtering
            if "context_docs" in rag_result and "analysis" in rag_result:
                filtered_docs = self.source_filter.filter_sources_by_relevance(
                    rag_result["context_docs"], 
                    question, 
                    rag_result["analysis"],
                    threshold=0.2
                )
                
                # Extract filtered source names
                filtered_sources = []
                for doc in filtered_docs:
                    source_info = doc.metadata.get('source', 'Unknown source')
                    if '/' in source_info:
                        source_info = source_info.split('/')[-1]
                    if source_info not in filtered_sources:
                        filtered_sources.append(source_info)
                
                # Log filtering results for debugging
                logger.info(f"Filtered sources from {len(rag_result['sources'])} to {len(filtered_sources)}: {filtered_sources}")
                
                return {
                    "answer": rag_result["answer"],
                    "sources": filtered_sources,
                    "query_analysis": rag_result.get("analysis"),
                    "source_filtering_applied": True
                }
            else:
                return {
                    "answer": rag_result["answer"],
                    "sources": rag_result["sources"],
                    "query_analysis": rag_result.get("analysis"),
                    "source_filtering_applied": False
                }
        
        elif "web_search" in datasource.lower():
            return {
                "answer": self.web_search.web_search(question, conversation_history),
                "sources": ["web_search"]
            }
        
        elif "exercise_solver" in datasource.lower():
            exercise_solver = ExerciseSolver(self.llm, self.retriever, self.pdf_folder)
            return exercise_solver.solve_exercise(question, conversation_history)
        
        else:
            return {
                "answer": "This information cannot be found in the textbook nor the internet. Please try again!",
                "sources": []
            }
    
    def invoke(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process a question with conversation awareness.
        
        Args:
            question: The user's question
            session_id: Unique identifier for the conversation session (e.g., Mattermost thread ID)
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Get or create conversation session
            session = self.conversation_manager.get_or_create_session(session_id)
            
            # Route the query
            result = self.router.invoke({"question": question})
            
            # Update session context with query analysis
            analysis = self.query_analyzer.invoke({"question": question})
            session.update_context(analysis, result.datasource)
            
            # Get the answer and sources
            response = self.choose_route(question, result, session)
            
            # Add this exchange to conversation memory
            session.add_exchange(question, response["answer"])
            
            # Determine if web search was used
            web_search_used = "web_search" in result.datasource.lower()
            
            # Base response structure
            response_data = {
                "response": response["answer"],  # Use 'response' for compatibility with existing API
                "datasource": result.datasource,
                "web_search_used": web_search_used,
                "sources": response["sources"],
                "session_id": session_id,
                "conversation_turns": len(session.memory.chat_memory.messages) // 2  # Rough estimate of turns
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
                "sources": [],
                "session_id": session_id
            }
    
    async def ainvoke(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """Async version of invoke."""
        return self.invoke(question, session_id)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about active conversation sessions."""
        return self.conversation_manager.get_session_info()
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a specific conversation session."""
        if session_id in self.conversation_manager.sessions:
            del self.conversation_manager.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    def clear_all_sessions(self):
        """Clear all conversation sessions."""
        self.conversation_manager.sessions.clear()
        logger.info("Cleared all conversation sessions")

# Example usage
if __name__ == "__main__":
    print("🤖 Conversational Numerical Analysis Knowledge Agent")
    print("=" * 60)
    print("Initializing the system...")
    
    try:
        # Initialize the system
        rag = ConversationalAgenticRetrieval(
            pdf_folder="./data/",
            persist_directory="./chroma_db"
        )
        print("✅ System initialized successfully!")
        print("💡 You can ask questions about Numerical Analysis topics.")
        print("💡 The system will remember our conversation context.")
        print("💡 Type 'quit', 'exit', or 'bye' to stop the program.")
        print("💡 Type 'new session' to start a fresh conversation.")
        print("💡 Type 'session info' to see conversation statistics.")
        print("=" * 60)
        
        current_session_id = "interactive_session"
        
        # Interactive chat loop
        while True:
            try:
                # Get user input
                question = input(f"\n🧑 Your question [Session: {current_session_id}]: ").strip()
                
                # Check for special commands
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Goodbye! Thanks for using the Conversational NA Knowledge Agent!")
                    break
                elif question.lower() == 'new session':
                    current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    print(f"🔄 Started new conversation session: {current_session_id}")
                    continue
                elif question.lower() == 'session info':
                    session_info = rag.get_session_info()
                    print(f"📊 Active sessions: {session_info['active_sessions']}")
                    for sid, info in session_info['sessions'].items():
                        print(f"  - {sid}: {info['topic_area'] or 'General'} (Last: {info['last_accessed'][:16]})")
                    continue
                
                # Skip empty questions
                if not question:
                    print("❌ Please enter a question.")
                    continue
                
                # Process the question with conversation awareness
                print("\n🤔 Thinking...")
                result = rag.invoke(question, current_session_id)
                
                # Display the result
                print(f"\n🤖 Answer:")
                print(f"{result['response']}")
                print(f"\n📊 Source: {result['datasource']} | Session: {result['session_id']} | Turns: {result['conversation_turns']}")
                
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
                print("\n\n👋 Goodbye! Thanks for using the Conversational NA Knowledge Agent!")
                break
            except Exception as e:
                print(f"\n❌ Error processing your question: {str(e)}")
                print("Please try again with a different question.")
                
    except Exception as e:
        print(f"❌ Failed to initialize the system: {str(e)}")
        print("Please check your .env file and make sure all dependencies are installed.")
        exit(1) 