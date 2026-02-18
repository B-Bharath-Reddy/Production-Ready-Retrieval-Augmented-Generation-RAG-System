
"""
Ingestion Orchestrator
----------------------
Phase 1: Data Preparation & Indexing.
Orchestrates the full data ingestion process.
Connects Loader -> Chunker -> Embedder -> VectorStore.
"""

from conf.config import cfg
from connecting.loader import DataLoader
from chunking.chunker import TextChunker
from embedding.embedder import Embedder
from database.vector_store import WeaviateManager
import os

def run_ingestion():
    """
    Main function to run the ingestion job.

    Orchestrates the 4-step pipeline:
    1. LOAD: Read raw .md and .pdf files.
    2. CHUNK: Split into smaller semantic pieces.
    3. EMBED: Setup the embedding model.
    4. INDEX: Push chunks + vectors to Weaviate.

    Why this is useful:
      - This is the "One Click" entry point to prepare the database.
      - Coordinates all the separate steps: Loading files -> Splitting them -> creating Vectors -> Saving to DB.
    """
    print("--- Starting Ingestion Pipeline ---")
    
    # 0. Check data path
    # Path calc adjusted to find 'data/raw' relative to this script location
    raw_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    raw_data_path = os.path.abspath(raw_data_path)
    print(f"Scanning for data in: {raw_data_path}")
    
    # 1. Load Data
    loader = DataLoader(raw_data_path)
    documents = loader.load_all()
    if not documents:
        print("No documents found to ingest. Please check 'data/raw' folder.")
        return

    # 2. Chunk Data
    # Initialize chunker with config values from config.yaml
    chunker = TextChunker(
        chunk_size=cfg.chunking.chunk_size, 
        chunk_overlap=cfg.chunking.chunk_overlap
    )
    chunks = chunker.chunk_documents(documents)

    # 3. Setup Embedding
    # Initialize model (cpu/cuda)
    embedder = Embedder(
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device
    )
    
    # 4. Connect to Weaviate & Ingest
    # Validate configuration
    if not cfg.weaviate.url:
        print("ERROR: Weaviate URL is not set in config.yaml. Please set it to run ingestion.")
        return

    # Initialize Manager
    vector_db = WeaviateManager(
        url=cfg.weaviate.url,
        api_key=cfg.weaviate.api_key,
        index_name=cfg.weaviate.class_name,
        embeddings=embedder.get_embeddings()
    )
    
    # Create/Reset schema (Caution: force_recreate=True deletes old data)
    # We enable force_recreate to ensure a clean state for this run
    vector_db.create_schema(force_recreate=True)
    
    # Upload data
    vector_db.ingest_chunks(chunks)
    
    # Close connection
    vector_db.close()

if __name__ == "__main__":
    if not cfg:
        print("Configuration failed to load. Exiting.")
    else:
        run_ingestion()
