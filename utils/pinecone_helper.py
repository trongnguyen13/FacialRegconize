"""
Pinecone vector database helper functions.
"""

from typing import Dict, List, Optional, Tuple
import os
from pinecone import Pinecone, ServerlessSpec
import time


class PineconeHelper:
    """Helper class for Pinecone vector database operations."""
    
    def __init__(self, api_key: str, index_name: str = "face-recognition-index", dimension: int = 512):
        """
        Initialize Pinecone helper.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            dimension: Dimension of the embeddings (must match the model)
        """
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.pc = None
        self.index = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone connection and create/connect to index."""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists, if not create it
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                # Create new index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                # Wait for index to be ready
                time.sleep(1)
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            raise Exception(f"Failed to initialize Pinecone: {str(e)}")
    
    def register_face(
        self,
        embedding: List[float],
        face_id: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Register a face embedding in Pinecone.
        
        Args:
            embedding: Face embedding vector
            face_id: Unique identifier for the face
            metadata: Optional metadata (name, date, etc.)
            
        Returns:
            True if successful
        """
        try:
            if metadata is None:
                metadata = {}
            
            # Upsert the embedding
            self.index.upsert(
                vectors=[
                    {
                        "id": face_id,
                        "values": embedding,
                        "metadata": metadata
                    }
                ]
            )
            return True
            
        except Exception as e:
            raise Exception(f"Failed to register face: {str(e)}")
    
    def search_faces(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar faces in Pinecone.
        
        Args:
            query_embedding: Query face embedding
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matches with id, score, and metadata
        """
        try:
            # Query the index
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Filter by threshold and format results
            matches = []
            for match in results.matches:
                if match.score >= score_threshold:
                    matches.append({
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata
                    })
            
            return matches
            
        except Exception as e:
            raise Exception(f"Failed to search faces: {str(e)}")
    
    def delete_face(self, face_id: str) -> bool:
        """
        Delete a face from Pinecone.
        
        Args:
            face_id: ID of the face to delete
            
        Returns:
            True if successful
        """
        try:
            self.index.delete(ids=[face_id])
            return True
        except Exception as e:
            raise Exception(f"Failed to delete face: {str(e)}")
    
    def get_stats(self) -> Dict:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index stats
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension
            }
        except Exception as e:
            raise Exception(f"Failed to get stats: {str(e)}")
    
    def list_all_faces(self) -> List[Dict]:
        """
        List all face entries in the index with their metadata.

        Returns:
            List of dicts with id and metadata for each entry
        """
        try:
            all_ids = []
            for id_list in self.index.list():
                all_ids.extend(id_list)

            if not all_ids:
                return []

            # Fetch metadata in batches of 100
            results = []
            for i in range(0, len(all_ids), 100):
                batch = all_ids[i:i + 100]
                fetch_response = self.index.fetch(ids=batch)
                for vid, vector_data in fetch_response.vectors.items():
                    results.append({
                        "id": vid,
                        "metadata": vector_data.metadata or {}
                    })

            return results

        except Exception as e:
            raise Exception(f"Failed to list faces: {str(e)}")


def initialize_pinecone_from_env() -> Optional[PineconeHelper]:
    """
    Initialize Pinecone from environment variables.
    
    Returns:
        PineconeHelper instance or None if credentials not found
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "face-recognition-index")
    
    if not api_key or api_key == "your_pinecone_api_key_here":
        return None
    
    try:
        # ArcFace uses 512-dimensional embeddings
        return PineconeHelper(api_key=api_key, index_name=index_name, dimension=512)
    except Exception as e:
        raise Exception(f"Failed to initialize Pinecone: {str(e)}")
