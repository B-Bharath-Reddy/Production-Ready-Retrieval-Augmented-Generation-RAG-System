
"""
Vector Store Module
-------------------
Responsible for interacting with the Weaviate Vector Database.
Handles connection, schema management, and indexing of document chunks.
Supports weaviate-client v4.
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Optional

class WeaviateManager:
    """
    Manages the Weaviate client and VectorStore operations using v4 client.

    Attributes:
        client (weaviate.WeaviateClient): The active connection to Weaviate.
        index_name (str): The name of the collection (i.e., Class) in Weaviate.

    Why this is useful:
      - Encapsulates all database-specific logic (connecting, schema creation, insertion).
      - Abstracts the difference between Weaviate Cloud (WCS) and local instances.
    """

    def __init__(self, url: str, api_key: Optional[str], index_name: str, embeddings: Embeddings):
        """
        Initialize Weaviate connection (v4).

        Args:
            url (str): The URL of the Weaviate instance.
            api_key (Optional[str]): API key for authentication (if required).
            index_name (str): The name of the collection/index to use.
            embeddings (Embeddings): The LangChain embedding object for vectorizing data.

        Raises:
            Exception: If connection fails.
        """
        self.index_name = index_name
        self.embeddings = embeddings
        self.url = url
        self.api_key = api_key
        
        print(f"Connecting to Weaviate at {url} (v4 client)...")
        try:
            # V4 Connection Handling
            # Configures auth if a key is provided
            auth_config =  weaviate.auth.AuthApiKey(api_key) if api_key else None
            
            # Use 'connect_to_wcs' for cloud instances or 'connect_to_local' for docker
            # We assume WCS/Remote URL here based on standard user config
            self.client = weaviate.connect_to_wcs(
                cluster_url=url,
                auth_credentials=auth_config,
                headers={} 
            )
            
            # Simple readiness check
            if not self.client.is_ready():
                 print("WARNING: Client reported not ready.")
                 
        except Exception as e:
            print(f"Failed to connect to Weaviate: {e}")
            raise e

    def create_schema(self, force_recreate: bool = False):
        """
        Create the collection in Weaviate (v4).

        Args:
            force_recreate (bool): If True, deletes the existing collection before creating (DESTRUCTIVE).

        Why this is useful:
          - The database needs a defined schema (Class/Collection) to store data.
          - Useful for resetting the database (`force_recreate=True`) during testing or fresh ingestion.
        """
        try:
            # Check if collection exists
            if self.client.collections.exists(self.index_name):
                if force_recreate:
                    print(f"Deleting existing collection: {self.index_name}")
                    self.client.collections.delete(self.index_name)
                else:
                    return

            # V4 Collection Creation
            print(f"Creating collection: {self.index_name}")
            self.client.collections.create(
                name=self.index_name,
                # Define specific properties to optimize storage and search
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="type", data_type=DataType.TEXT),
                ]
            )
            print("Schema created.")
            
        except Exception as e:
            print(f"Schema creation failed: {e}")

    def ingest_chunks(self, chunks: List[Document]):
        """
        Uploads document chunks to Weaviate.

        Args:
            chunks (List[Document]): The list of chunked documents to ingest.

        Why this is useful:
          - Pushes the processed data (text + metadata) into the vector database.
          - Handles the vectorization (via the passed embedding model) and batch uploading.
        """
        print(f"Ingesting {len(chunks)} chunks...")
        
        try:
            # Use LangChain's wrapper for convenience
            vectorstore = WeaviateVectorStore(
                client=self.client,
                index_name=self.index_name,
                text_key="text",
                embedding=self.embeddings
            )
            vectorstore.add_documents(chunks)
            print("Ingestion complete.")
            
        except Exception as e:
            # Fallback to manual batching if LangChain wrapper has issues
            print(f"Ingestion failed via LangChain: {e}")
            self._manual_ingest(chunks)

    def _manual_ingest(self, chunks: List[Document]):
        """
        Manual ingestion fallback using Weaviate v4 internal batching.

        Args:
            chunks (List[Document]): The list of chunked documents to ingest.

        Why this is useful:
          - Sometimes high-level wrappers fail or have version mismatches.
          - This ensures we can still load data using the native client if LangChain fails.
        """
        print("Attempting manual ingestion via v4 batching...")
        collection = self.client.collections.get(self.index_name)
        
        # Use context manager for automatic batch flushing
        with collection.batch.dynamic() as batch:
            for doc in chunks:
                # Compute vector manually
                vector = self.embeddings.embed_query(doc.page_content)
                
                batch.add_object(
                    properties={
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "type": doc.metadata.get("type", "")
                    },
                    vector=vector
                )
        print("Manual ingestion complete.")

    def get_vectorstore(self) -> WeaviateVectorStore:
        """
        Returns the LangChain vectorstore interface.

        Returns:
            WeaviateVectorStore: The initialized vector store object.

        Why this is useful:
          - The `Retriever` module expects a standardized LangChain interface for querying.
          - Decouples the low-level client from the high-level application logic.
        """
        return WeaviateVectorStore(
            client=self.client,
            index_name=self.index_name,
            text_key="text",
            embedding=self.embeddings
        )
        
    def close(self):
        """
        Closes the Weaviate client connection.

        Why this is useful:
          - Frees up network resources and sockets when the application shuts down.
        """
        self.client.close()

