
"""
Main Application Entry Point (CLI)
----------------------------------
Phase 2: Interactive Usage.
Runs the Chat Interface for the RAG system.
Integrates all modules (Rewriter, Retriever, Reranker, Generator) into a loop.
"""

import sys
from conf.config import cfg
from retrieval.query_rewriter import QueryRewriter
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.generator import Generator

def main():
    """
    RAG CLI Loop.

    Initializes the stack and runs an infinite loop to accept user input.
    Commands:
      - Type 'exit' or 'quit' to stop.
    """
    print("--- Enterprise RAG Assistant ---")
    print("Initializing components...")
    
    # Initialize components
    # 1. Rewriter: Optimizes queries
    rewriter = QueryRewriter()
    # 2. Retriever: Fetches from Weaviate
    retriever = Retriever()
    # 3. Reranker: Filters results
    reranker = Reranker(model_name=cfg.reranking.model_name)
    # 4. Generator: Produces answers
    generator = Generator()
    
    print("Ready! Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        print("Assistant: Thinking...", end="\r")
        
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
            
            print(f"Assistant: {answer}\n")
            
            # Optional: Print citations/sources if needed
            # sourcelist = [d.metadata.get('filename') for d in docs]
            # print(f"[Sources: {', '.join(sourcelist)}]\n")
            
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
