
"""
Embedder Module
---------------
Responsible for converting text chunks into vector representations (embeddings).
These vectors are used for semantic search in the vector database.

Model: 'sentence-transformers/all-MiniLM-L6-v2'
- Dimensions: 384
- Speed: Very Fast
- Quality: High for general purpose retrieval
"""

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:
    """
    Wrapper around SentenceTransformers via LangChain.

    Attributes:
        embedding_model (HuggingFaceEmbeddings): The underlying LangChain embedding model instance.

    Why this is useful:
      - We need to turn text into numbers (vectors) to perform semantic search.
      - Using a local model (`all-MiniLM-L6-v2`) is fast and requires no API cost.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the embedding model.

        Args:
            model_name (str): HuggingFace model ID (default: all-MiniLM-L6-v2).
            device (str): Computation device ('cpu' or 'cuda'/'mps').

        Why this is useful:
          - Loads the heavy model weights into memory once.
          - Configures execution device (CPU/GPU) for performance.
        """
        print(f"Loading embedding model: {model_name} on {device}...")
        
        # Initialize HuggingFaceEmbeddings
        # 'normalize_embeddings': True is crucial for cosine similarity
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True} 
        )

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Returns the LangChain embedding object.

        Returns:
            HuggingFaceEmbeddings: The configured embedding model.

        Why this is useful:
          - The `WeaviateVectorStore` needs a strictly typed `Embeddings` object to perform internal vectorization of queries.
        """
        return self.embedding_model
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single text query.

        Args:
            text (str): The text string to embed.

        Returns:
            List[float]: The 384-dimensional vector representation.

        Why this is useful:
          - For testing or standalone usage where we just want the vector for a string.
          - Used internally by manual ingestion scripts or debugging.
        """
        return self.embedding_model.embed_query(text)

