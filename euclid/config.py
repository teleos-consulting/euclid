import os
from pathlib import Path
from typing import Dict, Any, Optional

import dotenv
from pydantic import BaseModel

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Default configuration
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"

class EuclidConfig(BaseModel):
    """Configuration for the Euclid application."""
    # Ollama settings
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
    ollama_model: str = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    
    # Application settings
    history_file: Path = Path.home() / ".euclid_history"
    max_history_length: int = 100
    max_token_limit: int = 4096
    temperature: float = 0.7

# Global configuration instance
config = EuclidConfig()
