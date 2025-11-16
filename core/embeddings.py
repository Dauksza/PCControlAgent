"""
Embeddings API for RAG and semantic search
"""
import numpy as np
from typing import List, Dict, Any, Optional
from mistralai import Mistral
from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

class EmbeddingsManager:
    """
    Generate embeddings using mistral-embed model
    for RAG, semantic search, and clustering
    """
    
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-embed"
        self.embedding_cache = {}
    
    async def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        Supports batching for large datasets
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    inputs=batch
                )
                
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Embedding generation failed for batch: {e}")
                raise
        
        return all_embeddings
    
    def cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    async def semantic_search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search over documents
        
        Args:
            query: Search query
            documents: List of dicts with 'text' and optional 'embedding' keys
            top_k: Number of results to return
        """
        # Generate query embedding
        query_embeddings = await self.generate_embeddings([query])
        query_embedding = query_embeddings[0]
        
        # Calculate similarities
        similarities = []
        for doc in documents:
            if "embedding" not in doc:
                # Generate embedding if not present
                doc_embeddings = await self.generate_embeddings([doc["text"]])
                doc["embedding"] = doc_embeddings[0]
            
            similarity = self.cosine_similarity(query_embedding, doc["embedding"])
            similarities.append({
                **doc,
                "similarity": similarity
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    async def cluster_documents(
        self,
        documents: List[str],
        num_clusters: int = 5
    ) -> Dict[int, List[int]]:
        """
        Cluster documents using K-means on embeddings
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
            raise
        
        # Generate embeddings
        embeddings = await self.generate_embeddings(documents)
        embeddings_array = np.array(embeddings)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings_array)
        
        # Group documents by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            label_int = int(label)
            if label_int not in clusters:
                clusters[label_int] = []
            clusters[label_int].append(idx)
        
        return clusters
