import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import logging
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    A semantic cache that stores and retrieves similar queries and their responses.
    Uses embeddings to find semantically similar queries and their cached responses.
    Includes version tracking to invalidate cache when vectorstore is updated.
    """
    
    def __init__(
        self,
        cache_file: str = "semantic_cache.json",
        similarity_threshold: float = 0.9,
        max_cache_size: int = 1000,
        embedding_model: str = "text-embedding-3-small",
        vectorstore_path: str = None
    ):
        self.cache_file = cache_file
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.cache: Dict[str, Dict] = {}
        self.query_embeddings: Dict[str, np.ndarray] = {}
        self.vectorstore_path = vectorstore_path
        self.current_vectorstore_version = self._compute_vectorstore_version()
        self.load_cache()
    
    def _compute_vectorstore_version(self) -> str:
        """Compute a version hash of the vectorstore to detect changes"""
        if not self.vectorstore_path or not os.path.exists(self.vectorstore_path):
            return "initial"
        
        try:
            # Get all files in the vectorstore directory
            files = []
            for root, _, filenames in os.walk(self.vectorstore_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    files.append(file_path)
            
            # Sort files to ensure consistent hashing
            files.sort()
            
            # Compute hash of all file contents
            hasher = hashlib.sha256()
            for file_path in files:
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
            
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error computing vectorstore version: {e}")
            return "error"
    
    def _check_vectorstore_version(self) -> bool:
        """Check if vectorstore has been updated since last cache load"""
        new_version = self._compute_vectorstore_version()
        if new_version != self.current_vectorstore_version:
            logger.info(f"Vectorstore version changed from {self.current_vectorstore_version[:8]} to {new_version[:8]}")
            self.current_vectorstore_version = new_version
            return True
        return False
    
    def load_cache(self) -> None:
        """Load the cache from disk if it exists"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.cache = data.get('cache', {})
                    # Convert string embeddings back to numpy arrays
                    self.query_embeddings = {
                        query: np.array(embedding)
                        for query, embedding in data.get('embeddings', {}).items()
                    }
                    # Load vectorstore version
                    self.current_vectorstore_version = data.get('vectorstore_version', 'initial')
                
                # Check if vectorstore has been updated
                if self._check_vectorstore_version():
                    logger.info("Vectorstore updated, clearing cache")
                    self.clear()
                else:
                    logger.info(f"Loaded cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.cache = {}
            self.query_embeddings = {}
    
    def save_cache(self) -> None:
        """Save the cache to disk"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            embeddings_data = {
                query: embedding.tolist()
                for query, embedding in self.query_embeddings.items()
            }
            data = {
                'cache': self.cache,
                'embeddings': embeddings_data,
                'vectorstore_version': self.current_vectorstore_version
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def compute_similarity(self, query1: str, query2: str) -> float:
        """Compute cosine similarity between two queries"""
        if query1 not in self.query_embeddings:
            self.query_embeddings[query1] = np.array(self.embeddings.embed_query(query1))
        if query2 not in self.query_embeddings:
            self.query_embeddings[query2] = np.array(self.embeddings.embed_query(query2))
        
        vec1 = self.query_embeddings[query1]
        vec2 = self.query_embeddings[query2]
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def find_similar_query(self, query: str) -> Optional[Tuple[str, float]]:
        """Find the most similar query in the cache"""
        if not self.cache:
            return None
        
        # Check if vectorstore has been updated
        if self._check_vectorstore_version():
            logger.info("Vectorstore updated, clearing cache")
            self.clear()
            return None
        
        best_similarity = 0.0
        best_query = None
        
        for cached_query in self.cache.keys():
            similarity = self.compute_similarity(query, cached_query)
            if similarity > best_similarity:
                best_similarity = similarity
                best_query = cached_query
        
        if best_similarity >= self.similarity_threshold:
            return best_query, best_similarity
        return None
    
    def get(self, query: str) -> Optional[Dict]:
        """Get a cached response for a query if a similar one exists"""
        similar = self.find_similar_query(query)
        if similar:
            cached_query, similarity = similar
            logger.info(f"Found similar cached query with similarity {similarity:.2f}")
            return self.cache[cached_query]
        return None
    
    def put(self, query: str, response: Dict) -> None:
        """Add a query and its response to the cache"""
        # Check if vectorstore has been updated
        if self._check_vectorstore_version():
            logger.info("Vectorstore updated, clearing cache before adding new entry")
            self.clear()
        
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.max_cache_size:
            oldest_query = next(iter(self.cache))
            del self.cache[oldest_query]
            del self.query_embeddings[oldest_query]
        
        # Add new entry
        self.cache[query] = response
        # Compute and store embedding
        self.query_embeddings[query] = np.array(self.embeddings.embed_query(query))
        self.save_cache()
        logger.info(f"Cached response for query: {query}")
    
    def clear(self) -> None:
        """Clear the cache"""
        self.cache = {}
        self.query_embeddings = {}
        self.save_cache()
        logger.info("Cleared cache") 