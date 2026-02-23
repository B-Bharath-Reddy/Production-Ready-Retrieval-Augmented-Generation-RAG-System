
"""
Configuration Module
--------------------
Loads the YAML configuration file and validates it using Pydantic.
This ensures type safety and fail-fast behavior for missing settings.
"""

import os
import yaml
from typing import Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings
# PRODUCTION-READY: Load environment variables from .env file
from dotenv import load_dotenv
# Load .env from conf/ directory
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

# --- Pydantic Models for Validation ---

class WeaviateConfig(BaseModel):
    url: str
    class_name: str
    api_key: Optional[str] = None

class ChunkingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int

class EmbeddingConfig(BaseModel):
    model_name: str
    device: str

class RetrievalConfig(BaseModel):
    hybrid_alpha: float
    top_k: int

class RerankingConfig(BaseModel):
    enabled: bool
    model_name: str
    top_n: int

class GenerationConfig(BaseModel):
    model_name: str
    temperature: float
    max_tokens: int

class AppConfig(BaseSettings):
    groq_api_key: str
    weaviate: WeaviateConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    generation: GenerationConfig

# --- Loader Function ---

def load_config(config_path: str = "conf/config.yaml") -> AppConfig:
    """
    Load configuration from a YAML file.

    Why this is useful:
      - Separates code from configuration.
      - Allows changing settings (like model names or chunk sizes) without touching Python code.
    
    What it would tell you:
      - Returns a validated `AppConfig` object.
      - Crashes immediately if the config file is missing or invalid.
    """
    # Determine absolute path relative to project root
    # Since this file is now in conf/, we go up one level to root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(abs_config_path):
         # Fallback for when running from root
         if os.path.exists(config_path):
             abs_config_path = config_path
         else:
            raise FileNotFoundError(f"Config file not found at {abs_config_path}")

    with open(abs_config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Inject env vars if needed (e.g. from .env file or system env)
    if "GROQ_API_KEY" in os.environ:
        raw_config["groq_api_key"] = os.environ["GROQ_API_KEY"]
        
    return AppConfig(**raw_config)

# Global Config Object
try:
    cfg = load_config()
except Exception as e:
    print(f"Failed to load config: {e}")
    cfg = None
