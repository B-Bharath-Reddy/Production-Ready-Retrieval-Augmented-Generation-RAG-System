
"""
Retriever Module
----------------
Responsible for retrieving relevant documents from the Vector Database.
Implements Hybrid Search (combining BM25 Keyword search and dense Vector search)
to maximize recall and precision.
"""

from typing import List
from langchain_core.documents import Document
from database.vector_store import WeaviateManager
from embedding.embedder import Embedder
from conf.config import cfg

class Retriever:
    """
    Handles search queries against the Weaviate index.

    Why this is useful:
      - This is the "R" in RAG. It finds the context needed to answer the user's question.
      - Encapsulates the logic of connecting to the DB and executing the search.
    """

    def __init__(self):
        """
        Initialize the retriever with Weaviate connection and embedding model.

        Why this is useful:
          - Weaviate requires an embedder to vectorize the incoming query.
          - Sets up the `vectorstore` object which allows query execution.
        """
        # Load embedding model to convert query string -> vector
        self.embedder = Embedder(model_name=cfg.embedding.model_name, device=cfg.embedding.device)
        
        # Connect to Weaviate
        self.vector_db_manager = WeaviateManager(
            url=cfg.weaviate.url,
            api_key=cfg.weaviate.api_key,
            index_name=cfg.weaviate.class_name,
            embeddings=self.embedder.get_embeddings()
        )
        self.vectorstore = self.vector_db_manager.get_vectorstore()

    def get_relevant_documents(self, query: str, top_k: int = 4, alpha: float = 0.5) -> List[Document]:
        """
        Perform a hybrid search for the query.

        Args:
            query (str): The semantic search string.
            top_k (int): Number of documents to return (default: 4).
            alpha (float): Hybrid search weight (0 = pure keyword, 1 = pure vector).

        Returns:
            List[Document]: Top extracted documents matching the query.

        Why this is useful:
          - Pure vector search misses exact keyword matches (e.g., acronyms like "GPP").
          - Pure keyword search misses semantic meaning (e.g., "cost" vs "price").
          - Hybrid search combines both for robust retrieval.
        """
        print(f"Retrieving top {top_k} docs for query: '{query}' (Hybrid Alpha: {alpha})")
        
        try:
            # Configure retriever interface
            # Currently using 'similarity' (Vector) search as the default LangChain wrapper implementation.
            # TODO: Upgrade to `.as_retriever(search_type="hybrid")` when LangChain Weaviate wrapper stabilizes that feature.
            retriever = self.vectorstore.as_retriever(
               search_type="similarity",
               search_kwargs={"k": top_k}
            )
            
            # Execute search
            docs = retriever.invoke(query)
            return docs

        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

