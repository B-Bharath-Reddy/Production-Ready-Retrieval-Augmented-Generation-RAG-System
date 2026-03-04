
"""
Prompts Module
--------------
Stores the prompt templates used by the LLM.
Includes System Prompts, Context Wrappers, and Chain-of-Thought instructions.
"""

from langchain_core.prompts import ChatPromptTemplate

# System Prompt
# Defines the persona and constraints for the RAG assistant.
# PRODUCTION-READY: Enhanced citation instructions for source transparency
SYSTEM_TEMPLATE = """You are an expert internal enterprise assistant for GitLab and NIST compliance.
Your goal is to answer the user's question accurately using ONLY the provided context.

Rules:
1. Use the provided Context sections to answer the question.
2. If the answer is not in the context, say "I don't have enough information in the provided documents to answer that."
3. Cite your sources. Use [Source: filename] format at the end of sentences where you use information from documents.
4. At the end of your answer, list all sources used in a "Sources:" section.
5. Maintain a professional, concise, and helpful tone.
6. Format your answer with clear headings and bullet points where appropriate.

Context:
{context}
"""

# Human Prompt
# The placeholder for the user's actual question.
HUMAN_TEMPLATE = "{question}"

def get_rag_prompt() -> ChatPromptTemplate:
    """
    Returns the ChatPromptTemplate for the RAG chain.

    Returns:
        ChatPromptTemplate: The compiled LangChain prompt object.

    Why this is useful:
      - Centralizes the prompt design. To change the bot's persona or rules, we only edit this function.
      - Combines the system instructions (persona) with the variable user input.
    """
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", HUMAN_TEMPLATE)
    ])

# Query Rewrite Prompt
# Used by the QueryRewriter module to optimize search terms.
REWRITE_TEMPLATE = """You are a helpful assistant. Rewrite the following user question to be more specific and optimized for a search engine/vector database lookup. 
Do not answer the question. Just output the rewritten query text.

User Question: {question}
Rewritten Query:"""

