"""Retrieval-Augmented Generation (RAG) capabilities for Euclid."""

import os
import re
import json
import uuid
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from functools import lru_cache

import numpy as np
from pydantic import BaseModel, Field
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from euclid.functions import register_function
from euclid.formatting import console, create_spinner, EnhancedMarkdown

# Check if sentence_transformers is available
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

# Default paths
DEFAULT_DB_PATH = Path.home() / ".euclid_vectordb"
DEFAULT_COLLECTIONS_PATH = DEFAULT_DB_PATH / "collections"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class Document(BaseModel):
    """A document with content and metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class DocumentChunk(BaseModel):
    """A chunk of a document."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    document_id: str


class Collection(BaseModel):
    """A collection of documents."""
    id: str
    name: str
    description: Optional[str] = None
    documents: Dict[str, Document] = Field(default_factory=dict)
    chunks: Dict[str, DocumentChunk] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorDB:
    """Simple vector database for document storage and retrieval."""
    
    def __init__(
        self, 
        db_path: Optional[Union[str, Path]] = None, 
        embedding_model: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """Initialize the vector database.
        
        Args:
            db_path: Path to the database directory.
            embedding_model: Name of the sentence transformers model to use.
            chunk_size: Size of document chunks in characters.
            chunk_overlap: Overlap between chunks in characters.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.collections_path = self.db_path / "collections"
        self.collections_path.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model if available
        self.embedding_model_name = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.embedding_model = None
        
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                console.print(f"[warning]Warning: Could not load embedding model: {str(e)}[/warning]")
                console.print("[warning]RAG functionality will be limited. Install sentence-transformers for full functionality.[/warning]")
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all available collections.
        
        Returns:
            List of collection metadata.
        """
        collections = []
        for file_path in self.collections_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    metadata = json.load(f)
                    collections.append({
                        "id": metadata.get("id", file_path.stem),
                        "name": metadata.get("name", file_path.stem),
                        "description": metadata.get("description", ""),
                        "doc_count": metadata.get("doc_count", 0),
                        "chunk_count": metadata.get("chunk_count", 0),
                    })
            except Exception:
                # Skip invalid files
                pass
        
        return collections
    
    def create_collection(self, name: str, description: Optional[str] = None) -> str:
        """Create a new collection.
        
        Args:
            name: Name of the collection.
            description: Description of the collection.
            
        Returns:
            Collection ID.
        """
        collection_id = str(uuid.uuid4())
        collection = Collection(
            id=collection_id,
            name=name,
            description=description
        )
        
        # Save metadata
        metadata_path = self.collections_path / f"{collection_id}.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "id": collection_id,
                "name": name,
                "description": description,
                "doc_count": 0,
                "chunk_count": 0,
            }, f)
        
        # Save empty collection
        self._save_collection(collection)
        
        return collection_id
    
    def get_collection(self, collection_id: str) -> Optional[Collection]:
        """Get a collection by ID.
        
        Args:
            collection_id: ID of the collection.
            
        Returns:
            Collection or None if not found.
        """
        collection_path = self.collections_path / f"{collection_id}.pkl"
        
        if not collection_path.exists():
            return None
        
        try:
            with open(collection_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def _save_collection(self, collection: Collection) -> None:
        """Save a collection to disk.
        
        Args:
            collection: Collection to save.
        """
        collection_path = self.collections_path / f"{collection.id}.pkl"
        
        with open(collection_path, "wb") as f:
            pickle.dump(collection, f)
        
        # Update metadata
        metadata_path = self.collections_path / f"{collection.id}.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "id": collection.id,
                "name": collection.name,
                "description": collection.description,
                "doc_count": len(collection.documents),
                "chunk_count": len(collection.chunks),
            }, f)
    
    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection.
        
        Args:
            collection_id: ID of the collection to delete.
            
        Returns:
            True if deleted, False otherwise.
        """
        collection_path = self.collections_path / f"{collection_id}.pkl"
        metadata_path = self.collections_path / f"{collection_id}.json"
        
        if not collection_path.exists():
            return False
        
        try:
            collection_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            return True
        except Exception:
            return False
    
    def add_document(
        self, 
        collection_id: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True
    ) -> Optional[str]:
        """Add a document to a collection.
        
        Args:
            collection_id: ID of the collection.
            content: Document content.
            metadata: Document metadata.
            chunk: Whether to chunk the document.
            
        Returns:
            Document ID if added, None otherwise.
        """
        collection = self.get_collection(collection_id)
        if not collection:
            return None
        
        # Create document
        doc_id = str(uuid.uuid4())
        document = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {}
        )
        
        # Create embedding for the document
        if self.embedding_model:
            try:
                document.embedding = self.embedding_model.encode(content).tolist()
            except Exception:
                # Continue without embedding
                pass
        
        # Add document to collection
        collection.documents[doc_id] = document
        
        # Chunk document if requested
        if chunk:
            chunks = self._chunk_document(document)
            for chunk in chunks:
                collection.chunks[chunk.id] = chunk
        
        # Save collection
        self._save_collection(collection)
        
        return doc_id
    
    def _chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a document into chunks.
        
        Args:
            document: Document to chunk.
            
        Returns:
            List of document chunks.
        """
        content = document.content
        chunks = []
        
        # Simple character-based chunking
        start = 0
        while start < len(content):
            end = start + self.chunk_size
            if end >= len(content):
                chunk_content = content[start:]
            else:
                # Try to end at a period or newline
                cutoff = min(end + 100, len(content))
                last_period = content[end:cutoff].find(". ")
                last_newline = content[end:cutoff].find("\n")
                
                if last_period != -1 and (last_newline == -1 or last_period < last_newline):
                    end = end + last_period + 2  # Include the period and space
                elif last_newline != -1:
                    end = end + last_newline + 1  # Include the newline
            
            chunk_content = content[start:end]
            
            # Create chunk
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=document.metadata.copy(),
                document_id=document.id
            )
            
            # Create embedding for the chunk
            if self.embedding_model:
                try:
                    chunk.embedding = self.embedding_model.encode(chunk_content).tolist()
                except Exception:
                    # Continue without embedding
                    pass
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def search(
        self, 
        collection_id: str, 
        query: str, 
        top_k: int = 5,
        use_chunks: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for documents or chunks in a collection.
        
        Args:
            collection_id: ID of the collection.
            query: Search query.
            top_k: Number of results to return.
            use_chunks: Whether to search chunks instead of documents.
            
        Returns:
            List of search results with content and metadata.
        """
        collection = self.get_collection(collection_id)
        if not collection:
            return []
        
        # If no embedding model, fall back to keyword search
        if not self.embedding_model:
            return self._keyword_search(collection, query, top_k, use_chunks)
        
        # Encode query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in either chunks or documents
        items = collection.chunks if use_chunks else collection.documents
        results = []
        
        for item_id, item in items.items():
            if not item.embedding:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, item.embedding)
            
            results.append({
                "id": item_id,
                "content": item.content,
                "metadata": item.metadata,
                "similarity": similarity,
                "document_id": getattr(item, "document_id", item_id)
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_k]
    
    def _keyword_search(
        self, 
        collection: Collection, 
        query: str, 
        top_k: int,
        use_chunks: bool
    ) -> List[Dict[str, Any]]:
        """Perform keyword search when no embedding model is available.
        
        Args:
            collection: Collection to search.
            query: Search query.
            top_k: Number of results to return.
            use_chunks: Whether to search chunks instead of documents.
            
        Returns:
            List of search results.
        """
        # Simple keyword matching
        keywords = query.lower().split()
        items = collection.chunks if use_chunks else collection.documents
        results = []
        
        for item_id, item in items.items():
            content_lower = item.content.lower()
            
            # Count keyword matches
            match_count = sum(1 for kw in keywords if kw in content_lower)
            
            if match_count > 0:
                results.append({
                    "id": item_id,
                    "content": item.content,
                    "metadata": item.metadata,
                    "similarity": match_count / len(keywords),  # Approx. relevance score
                    "document_id": getattr(item, "document_id", item_id)
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector.
            vec2: Second vector.
            
        Returns:
            Cosine similarity.
        """
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get_document(self, collection_id: str, document_id: str) -> Optional[Document]:
        """Get a document by ID.
        
        Args:
            collection_id: ID of the collection.
            document_id: ID of the document.
            
        Returns:
            Document or None if not found.
        """
        collection = self.get_collection(collection_id)
        if not collection or document_id not in collection.documents:
            return None
        
        return collection.documents[document_id]


# Initialize global VectorDB instance
vectordb = VectorDB()


@register_function(
    name="CreateCollection",
    description="Create a new collection in the vector database for RAG."
)
def create_collection(name: str, description: Optional[str] = None) -> str:
    """Create a new collection for document storage.
    
    Args:
        name: Name of the collection.
        description: Description of the collection.
        
    Returns:
        Result message with collection ID.
    """
    collection_id = vectordb.create_collection(name, description)
    return f"Collection created: **{name}** (ID: {collection_id})"


@register_function(
    name="ListCollections",
    description="List all available collections in the vector database."
)
def list_collections() -> str:
    """List all available collections.
    
    Returns:
        Formatted list of collections.
    """
    collections = vectordb.list_collections()
    
    if not collections:
        return "No collections found. Create one with CreateCollection function."
    
    result = "# Available Collections\n\n"
    for i, collection in enumerate(collections):
        result += f"## {i+1}. {collection['name']} (ID: {collection['id']})\n\n"
        if collection['description']:
            result += f"{collection['description']}\n\n"
        result += f"Documents: {collection['doc_count']} | Chunks: {collection['chunk_count']}\n\n"
    
    return result


@register_function(
    name="AddDocument",
    description="Add a document to a collection in the vector database."
)
def add_document(
    collection_id: str, 
    content: str, 
    title: Optional[str] = None,
    source: Optional[str] = None,
    chunk: bool = True
) -> str:
    """Add a document to a collection.
    
    Args:
        collection_id: ID of the collection.
        content: Document content.
        title: Document title.
        source: Document source.
        chunk: Whether to chunk the document.
        
    Returns:
        Result message.
    """
    metadata = {
        "title": title or "Untitled Document",
        "source": source or "Unknown",
        "added_at": str(datetime.datetime.now())
    }
    
    doc_id = vectordb.add_document(collection_id, content, metadata, chunk)
    
    if not doc_id:
        return f"Error: Collection with ID '{collection_id}' not found."
    
    return f"Document added to collection (ID: {doc_id})"


@register_function(
    name="QueryCollection",
    description="Search for similar documents in a collection."
)
def query_collection(
    collection_id: str,
    query: str,
    top_k: int = 3,
    use_chunks: bool = True
) -> str:
    """Search for documents or chunks in a collection.
    
    Args:
        collection_id: ID of the collection.
        query: Search query.
        top_k: Number of results to return.
        use_chunks: Whether to search chunks instead of documents.
        
    Returns:
        Search results formatted as markdown.
    """
    results = vectordb.search(collection_id, query, top_k, use_chunks)
    
    if not results:
        return f"No results found for query: '{query}' in collection ID: {collection_id}"
    
    result_text = f"# Search Results for: '{query}'\n\n"
    
    for i, result in enumerate(results):
        similarity = result["similarity"]
        content = result["content"]
        
        # Truncate content if too long
        if len(content) > 500:
            content = content[:497] + "..."
        
        result_text += f"## Result {i+1} (Relevance: {similarity:.2f})\n\n"
        
        # Add metadata if available
        metadata = result.get("metadata", {})
        if metadata:
            if "title" in metadata:
                result_text += f"**Title**: {metadata['title']}  \n"
            if "source" in metadata:
                result_text += f"**Source**: {metadata['source']}  \n"
        
        result_text += f"\n{content}\n\n"
        result_text += "---\n\n"
    
    return result_text


@register_function(
    name="DeleteCollection",
    description="Delete a collection from the vector database."
)
def delete_collection(collection_id: str) -> str:
    """Delete a collection.
    
    Args:
        collection_id: ID of the collection to delete.
        
    Returns:
        Result message.
    """
    if vectordb.delete_collection(collection_id):
        return f"Collection deleted: {collection_id}"
    else:
        return f"Error: Collection with ID '{collection_id}' not found or could not be deleted."


# Fix missing datetime import
import datetime