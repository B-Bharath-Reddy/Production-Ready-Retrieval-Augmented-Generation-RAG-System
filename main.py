"""
Main Application Entry Point (CLI)
----------------------------------
Phase 2: Interactive Usage.
Runs the Chat Interface for the RAG system.
Integrates all modules (Rewriter, Retriever, Reranker, Generator) into a loop.
"""

# PRODUCTION-READY: Standard logging setup
import logging
import sys
import os

# PRODUCTION-READY: Load .env file FIRST before setting LangSmith variables
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf', '.env')
load_dotenv(env_path)

# PRODUCTION-READY: LangSmith tracing environment variables (set AFTER loading .env)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "YOUR_LANGCHAIN_API_KEY_HERE")
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "rag-project")

from conf.config import cfg
from retrieval.query_rewriter import QueryRewriter
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.generator import Generator

# PRODUCTION-READY: Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    RAG CLI Loop.

    Initializes the stack and runs an infinite loop to accept user input.
    Commands:
      - Type 'exit' or 'quit' to stop.
    """
    logger.info("--- Enterprise RAG Assistant ---")
    logger.info("Initializing components...")
    
    # Initialize components
    # 1. Rewriter: Optimizes queries
    rewriter = QueryRewriter()
    # 2. Retriever: Fetches from Weaviate
    retriever = Retriever()
    # 3. Reranker: Filters results
    reranker = Reranker(model_name=cfg.reranking.model_name)
    # 4. Generator: Produces answers
    generator = Generator()
    
    logger.info("Ready! Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        logger.info("Assistant: Thinking...")
        
        try:
            # Step 1: Rewrite Query
            search_query = rewriter.rewrite(user_input)
            
            # Step 2: Retrieve Docs
            docs = retriever.get_relevant_documents(search_query, top_k=cfg.retrieval.top_k)
            
            # Step 3: Rerank Results
            if cfg.reranking.enabled:
                docs = reranker.rerank(search_query, docs, top_n=cfg.reranking.top_n)
            else:
                docs = docs[:cfg.reranking.top_n]
                
            # Step 4: Generate Answer
            context_text = "\n\n".join([d.page_content for d in docs])
            answer = generator.generate_answer(user_input, context_text)
            
            logger.info(f"Assistant: {answer}\n")
            
            # Optional: Print citations/sources if needed
            # sourcelist = [d.metadata.get('filename') for d in docs]
            # logger.info(f"[Sources: {', '.join(sourcelist)}]\n")
            
        except Exception as e:
            logger.error(f"Error: {e}\n")

if __name__ == "__main__":
    main()