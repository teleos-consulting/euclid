import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from euclid.config import config
from euclid.ollama import Message

class Conversation(BaseModel):
    """Represents a conversation with message history."""
    id: str
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def formatted_messages(self) -> List[Dict[str, str]]:
        """Get messages formatted for Ollama API."""
        return [message.model_dump() for message in self.messages]
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation.
        
        Args:
            role: The role of the message sender (user/assistant/system).
            content: The content of the message.
        """
        self.messages.append(Message(role=role, content=content))
    
    def save(self, file_path: Optional[Path] = None) -> None:
        """Save the conversation to a file.
        
        Args:
            file_path: Path to save the conversation to. Defaults to config.history_file.
        """
        file_path = file_path or config.history_file
        
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing history if file exists
        history = []
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                # If file is corrupted, start with empty history
                history = []
        
        # Add or update this conversation
        for i, conv in enumerate(history):
            if conv.get("id") == self.id:
                history[i] = self.model_dump()
                break
        else:
            history.append(self.model_dump())
        
        # Limit history length
        if len(history) > config.max_history_length:
            history = history[-config.max_history_length:]
        
        # Write history back to file
        with open(file_path, "w") as f:
            json.dump(history, f, indent=2)
    
    @classmethod
    def load(cls, conversation_id: str, file_path: Optional[Path] = None) -> 'Conversation':
        """Load a conversation from history.
        
        Args:
            conversation_id: ID of the conversation to load.
            file_path: Path to load the conversation from. Defaults to config.history_file.
            
        Returns:
            The loaded conversation, or a new one if not found.
        """
        file_path = file_path or config.history_file
        
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    history = json.load(f)
                
                for conv in history:
                    if conv.get("id") == conversation_id:
                        return cls.model_validate(conv)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # If conversation not found or error, return a new one
        return cls(id=conversation_id)
    
    @classmethod
    def list_conversations(cls, file_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """List all saved conversations.
        
        Args:
            file_path: Path to load conversations from. Defaults to config.history_file.
            
        Returns:
            List of conversation metadata.
        """
        file_path = file_path or config.history_file
        
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    history = json.load(f)
                return history
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        
        return []
