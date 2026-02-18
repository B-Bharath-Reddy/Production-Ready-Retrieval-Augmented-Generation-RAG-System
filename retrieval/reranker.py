
"""
Reranker Module
----------------
Responsible for refining the initial retrieval results.
Uses a Cross-Encoder model (slower but more accurate than bi-encoders) to strictly
score the relevance of each [Query, Document] pair and re-order the list.
"""

from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Re-ranks documents using a Cross-Encoder.

    Why this is useful:
      - Vector search is fast but approximate (Bi-Encoder).
      - Cross-Encoders are slow but very precise because they attend to query and document tokens simultaneously.
      - A "Retrieve-then-Rerank" pipeline gives us the best of both worlds (speed + accuracy).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the Cross-Encoder model.

        Args:
            model_name (str): The HuggingFace model ID for the cross-encoder.

        Why this is useful:
          - Loads the model weights into memory.
        """
        print(f"Loading Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """
        Score and re-order the documents based on relevance to the query.

        Args:
            query (str): The search query.
            documents (List[Document]): The list of documents retrieved from Weaviate.
            top_n (int): The number of top documents to keep after re-ranking.

        Returns:
            List[Document]: The re-ordered list of documents.

        Why this is useful:
          - Filters out irrelevant documents that might confuse the LLM.
          - Prioritizes the most critical information by placing it at the top of the context window.
        """
        if not documents:
            return []

        # Prepare pairs for the model: [[query, doc_text], [query, doc_text], ...]
        # Cross-encoders expect pairs of text to score their similarity.
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Predict scores (higher is better)
        scores = self.model.predict(pairs)
        
        # Attach scores to documents for debugging/verification
        for doc, score in zip(documents, scores):
            doc.metadata["relevance_score"] = float(score)
            
        # Sort by score descending
        # Zip docs and scores, sort, then unzip
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        # Select top N to return to the generator
        top_docs = [doc for doc, score in scored_docs[:top_n]]
        
        print(f"Reranked {len(documents)} docs -> Top {top_n} selected.")
        return top_docs
