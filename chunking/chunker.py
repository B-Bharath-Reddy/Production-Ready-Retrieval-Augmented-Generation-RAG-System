
"""
Text Chunker Module
-------------------
Responsible for splitting large documents into smaller, semantically meaningful chunks.
This is crucial for RAG to ensure retrieved context fits within the LLM's context window
and is focused on specific topics.

Key Strategies:
- Recursive Splitting: Tries to split by paragraphs, then sentences, then words.
- Structure-Aware Splitting: For Markdown, it respects headers (#, ##) to keep sections together.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextChunker:
    """
    Splits documents into chunks using recursive character splitting.

    Attributes:
        chunk_size (int): The maximum number of characters per chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Why this is useful:
      - LLMs have finite context windows; we cannot feed entire books at once.
      - Small chunks allow precise retrieval of only the relevant information.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the splitting strategies.

        Args:
            chunk_size (int): Max characters per chunk (default: 500).
            chunk_overlap (int): Overlap characters to preserve context at boundaries (default: 50).
        
        Why this is useful:
          - Configures the size and overlap of chunks.
          - We use different splitters for Markdown (structure-aware) vs. plain text (character-aware).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Splitter for general text/PDFs
        # The 'separators' list is ordered by priority:
        # 1. Double newline (paragraph break)
        # 2. Single newline (line break)
        # 3. Period (sentence break)
        # 4. Space (word break)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Splitter specifically optimized for Markdown structure (headers, code blocks)
        # It attempts to keep headers with their content logic
        self.md_splitter = RecursiveCharacterTextSplitter.from_language(
            language="markdown",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process a list of documents and split them into chunks.

        Args:
            documents (List[Document]): The raw documents to split.

        Returns:
            List[Document]: The list of split document chunks.

        Why this is useful:
          - Applies the correct splitting logic based on the document type (Markdown vs. PDF/Text).
          - Flattens the hierarchy into a list of chunks ready for embedding.
        """
        chunked_docs = []
        
        for doc in documents:
            # Check metadata to decide which splitter to use
            if doc.metadata.get("type") == "markdown":
                chunks = self.md_splitter.split_documents([doc])
            else:
                chunks = self.text_splitter.split_documents([doc])
            
            chunked_docs.extend(chunks)
            
        print(f"Split {len(documents)} documents into {len(chunked_docs)} chunks.")
        return chunked_docs

