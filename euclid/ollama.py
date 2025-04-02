import json
from typing import Dict, List, Optional, Any, Generator, Union
import requests
from pydantic import BaseModel

from euclid.config import config

class Message(BaseModel):
    """Represents a message in a conversation."""
    role: str
    content: str

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API. Defaults to config.ollama_base_url.
            model: Model to use. Defaults to config.ollama_model.
        """
        self.base_url = base_url or config.ollama_base_url
        self.model = model or config.ollama_model
    
    def chat_completion(self, messages: List[Message], stream: bool = True, **kwargs) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Generate a chat completion.
        
        Args:
            messages: List of messages in the conversation.
            stream: Whether to stream the response.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            If stream=True, a generator yielding response chunks.
            If stream=False, the complete response.
        """
        url = f"{self.base_url}/api/chat"
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
            "temperature": kwargs.get("temperature", config.temperature),
        }
        
        # Add other parameters if provided
        for key, value in kwargs.items():
            if key not in payload and value is not None:
                payload[key] = value
        
        if stream:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            return self._process_stream(response)
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    def _process_stream(self, response: requests.Response) -> Generator[str, None, None]:
        """Process a streaming response from Ollama.
        
        Args:
            response: Streaming response from Ollama API.
            
        Yields:
            Text chunks from the response.
        """
        for line in response.iter_lines():
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]
            except json.JSONDecodeError:
                continue
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get a list of available models from Ollama.
        
        Returns:
            List of model information dictionaries.
        """
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("models", [])
