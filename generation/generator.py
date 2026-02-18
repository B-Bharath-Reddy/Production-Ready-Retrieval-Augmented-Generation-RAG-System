
"""
Generator Module
----------------
Responsible for generating the final answer using the Groq API.
Constructs the LLM chain with prompts and context.
"""

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from generation.prompts import get_rag_prompt
from conf.config import cfg

class Generator:
    """
    Wrapper for Groq LLM generation.

    Attributes:
        llm (ChatGroq): The configured Groq Chat Model.
        chain (Runnable): The LCEL chain combining Prompt -> LLM -> Parser.

    Why this is useful:
      - This is the "G" in RAG. It synthesizes the retrieved information into a human-readable answer.
      - Manages the API connection to Groq and applies the specific prompt templates.
    """

    def __init__(self):
        """
        Initialize the Groq Chat Model.

        Why this is useful:
          - Sets up the LLM with the correct model (`llama3...`) and temperature.
          - Compiles the LCEL (LangChain Expression Language) chain: Prompt -> LLM -> String Parser.
        """
        print(f"Initializing Groq LLM: {cfg.generation.model_name}")
        
        # Initialize the Chat Model
        # Temperature is set via config (usually 0 for factual RAG)
        self.llm = ChatGroq(
            temperature=cfg.generation.temperature,
            model_name=cfg.generation.model_name,
            groq_api_key=cfg.groq_api_key,
            max_tokens=cfg.generation.max_tokens
        )
        
        # Get the standard RAG prompt template
        self.prompt = get_rag_prompt()
        
        # Build the chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer given the query and retrieved context.

        Args:
            query (str): The user's original question.
            context (str): The concatenated text from retrieved documents.

        Returns:
            str: The final generated answer from the LLM.

        Why this is useful:
          - The core function that produces the value for the user.
          - Takes the raw relevant text chunks and the user's question, and uses the LLM to write the answer.
        """
        if not context:
            return "No relevant context found to answer the question."

        try:
            # Invoke the chain with the required inputs
            response = self.chain.invoke({
                "context": context,
                "question": query
            })
            return response
        except Exception as e:
            return f"Error interacting with LLM: {e}"

