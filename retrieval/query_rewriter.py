
"""
Query Rewriter Module
---------------------
Uses an LLM (Groq) to rewrite the user's query before retrieval.
This helps resolve ambiguity and optimizes the query for vector search.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from generation.prompts import REWRITE_TEMPLATE
from conf.config import cfg

class QueryRewriter:
    """
    Rewrites user queries to be more search-friendly.

    Why this is useful:
      - Users often ask vague or conversational questions (e.g., "how much?").
      - Vector databases prefer specific, noun-rich queries (e.g., "enterprise license pricing 2024").
      - This intermediate step bridges the gap between user intent and database precision.
    """
    
    def __init__(self):
        """
        Initialize the Rewriter with Groq.

        Why this is useful:
          - Sets up the LLM chain with a specific prompt designed for query expansion.
          - We use a slightly higher temperature (0.3) to allow for some creative expansion of terms.
        """
        self.llm = ChatGroq(
            temperature=0.3, # Creative for rewriting
            model_name=cfg.generation.model_name,
            groq_api_key=cfg.groq_api_key
        )
        self.prompt = PromptTemplate.from_template(REWRITE_TEMPLATE)
        # Construct the LCEL chain (Prompt -> LLM -> String)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def rewrite(self, query: str) -> str:
        """
        Rewrites the input query.

        Args:
            query (str): The original user question.

        Returns:
            str: The machine-optimized search query.

        Why this is useful:
          - Transforms the raw input into a search-optimized query.
          - Removes conversational filler and resolves ambiguities if possible.
        """
        try:
            # Invoke the chain
            new_query = self.chain.invoke({"question": query})
            
            # Clean up artifacts: some models repeat the prompt instructions
            cleaned_query = new_query.replace("Rewritten Query:", "").strip()
            
            print(f"Query Rewrite: '{query}' -> '{cleaned_query}'")
            return cleaned_query
        except Exception as e:
            # Fallback: if rewrite fails, safe to use original query
            print(f"Query Rewrite failed: {e}. Using original query.")
            return query
