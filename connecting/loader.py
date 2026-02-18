
"""
Data Loader Module
------------------
Responsible for loading raw text from various file formats (Markdown, PDF).
This module scans the `data/raw` directory and yields document objects.

Key Features:
- Recursive file scanning: Finds files in nested subdirectories.
- Format support: Handles .md (Markdown) and .pdf (standard PDF).
- Metadata extraction: Captures filename and path for citation.
"""

import os
import glob
from typing import List, Generator
import pypdf
from langchain_core.documents import Document

class DataLoader:
    """
    Handles loading of documents from the file system.

    Attributes:
        data_dir (str): The absolute path to the root directory containing raw data.

    Why this is useful:
      - Abstraction layer over file I/O allows us to easily add more formats later.
      - Centralizes logic for traversing directories and handling file encoding.

    What it would tell you:
      - Provides methods to get `Document` objects from the raw data directory.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the DataLoader.

        Args:
            data_dir (str): The root directory to scan for files.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        self.data_dir = data_dir
        # Validate input directory to fail fast if configuration is wrong
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_markdown(self) -> Generator[Document, None, None]:
        """
        Recursively finds and reads all .md files in the data directory.

        Yields:
            Document: A LangChain Document object with 'page_content' and 'metadata'.

        Why this is useful:
          - The primary dataset (GitLab Handbook) consists of nested Markdown files.
          - We need to preserve the directory structure or filename as metadata for citations.
        """
        # Find all .md files recursively using glob
        # recursive=True allows us to match files in subfolders (e.g., data/raw/handbook/engineering/...)
        md_files = glob.glob(os.path.join(self.data_dir, "**", "*.md"), recursive=True)
        print(f"Found {len(md_files)} Markdown files.")

        for file_path in md_files:
            try:
                # Open with utf-8 encoding to handle special characters/emojis in handbook text
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Create metadata dictionary
                # 'source': Full path, useful for debugging
                # 'filename': Short name, useful for user-facing citations
                # 'type': Helps the chunker decide which splitter to use
                metadata = {
                    "source": file_path,
                    "filename": os.path.basename(file_path),
                    "type": "markdown"
                }
                
                yield Document(page_content=content, metadata=metadata)
            except Exception as e:
                # Catch specific file read errors so one bad file doesn't crash the whole pipeline
                print(f"Error reading Markdown file {file_path}: {e}")

    def load_pdf(self) -> Generator[Document, None, None]:
        """
        Finds and extracts text from all .pdf files in the data directory.

        Yields:
            Document: A LangChain Document object containing the full text of the PDF.

        Why this is useful:
          - The secondary dataset (NIST Frames) are PDF documents.
          - We need to extract raw text from these binary files to be chunked.
        """
        # Find all .pdf files recursively
        pdf_files = glob.glob(os.path.join(self.data_dir, "**", "*.pdf"), recursive=True)
        print(f"Found {len(pdf_files)} PDF files.")

        for file_path in pdf_files:
            try:
                # Use pypdf for robust text extraction
                reader = pypdf.PdfReader(file_path)
                text = ""
                
                # Iterate over every page and concatenate text
                # We add a newline character to ensure separation between pages
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                metadata = {
                    "source": file_path,
                    "filename": os.path.basename(file_path),
                    "type": "pdf"
                }

                yield Document(page_content=text, metadata=metadata)
            except Exception as e:
                print(f"Error reading PDF file {file_path}: {e}")

    def load_all(self) -> List[Document]:
        """
        Loads all supported file types (Markdown and PDF).

        Returns:
            List[Document]: A consolidated list of all documents found.

        Why this is useful:
          - Single entry point to get *all* data for the ingestion pipeline.
          - Abstracts away the specific format handlers from the caller.
        """
        documents = []
        # Chain the generators to build a single list
        documents.extend(list(self.load_markdown()))
        documents.extend(list(self.load_pdf()))
        return documents

