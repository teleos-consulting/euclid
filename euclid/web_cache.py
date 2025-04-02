"""Web content caching system for Euclid."""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import tempfile
import shutil
import base64
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebCache:
    """Cache for web content to reduce repeated requests."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_age: int = 86400):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files (default: ~/.euclid/web_cache)
            max_age: Maximum age of cache entries in seconds (default: 24 hours)
        """
        self.max_age = max_age
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Default to ~/.euclid/web_cache
            self.cache_dir = Path.home() / ".euclid" / "web_cache"
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Keep track of cache hits/misses
        self.hits = 0
        self.misses = 0
    
    def _get_cache_key(self, url: str) -> str:
        """Generate a safe cache key for a URL.
        
        Args:
            url: URL to generate key for
            
        Returns:
            Cache key (hashed URL)
        """
        # Use SHA-256 hash of the URL as the cache key
        return hashlib.sha256(url.encode()).hexdigest()
    
    def _get_cache_file_path(self, url: str) -> Path:
        """Get the cache file path for a URL.
        
        Args:
            url: URL to get cache file for
            
        Returns:
            Path to cache file
        """
        key = self._get_cache_key(url)
        # Store metadata and content separately
        return self.cache_dir / f"{key}.json"
    
    def _get_content_file_path(self, url: str) -> Path:
        """Get the content file path for a URL.
        
        Args:
            url: URL to get content file for
            
        Returns:
            Path to content file
        """
        key = self._get_cache_key(url)
        # Use .bin extension for binary content
        return self.cache_dir / f"{key}.bin"
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached response for a URL.
        
        Args:
            url: URL to get cached response for
            
        Returns:
            Cached response dict or None if not in cache or expired
        """
        cache_file = self._get_cache_file_path(url)
        content_file = self._get_content_file_path(url)
        
        # Check if cache file exists
        if not cache_file.exists() or not content_file.exists():
            self.misses += 1
            return None
        
        try:
            # Load metadata
            with open(cache_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache entry is expired
            cached_time = metadata.get('timestamp', 0)
            if time.time() - cached_time > self.max_age:
                logger.debug(f"Cache entry for {url} has expired")
                self.misses += 1
                return None
            
            # Load content
            with open(content_file, 'rb') as f:
                content = f.read()
            
            # Decode content if it's text
            content_type = metadata.get('content_type', '')
            if any(t in content_type for t in ['text/', 'application/json', 'application/xml']):
                content = content.decode('utf-8')
            
            # Construct response
            response = {
                'url': url,
                'status_code': metadata.get('status_code', 200),
                'headers': metadata.get('headers', {}),
                'content': content,
                'cached': True,
                'timestamp': cached_time
            }
            
            self.hits += 1
            return response
        
        except Exception as e:
            logger.error(f"Error reading cache for {url}: {str(e)}")
            self.misses += 1
            return None
    
    def put(self, url: str, status_code: int, headers: Dict[str, str], content: Any) -> bool:
        """Store response in cache.
        
        Args:
            url: URL to store response for
            status_code: HTTP status code
            headers: HTTP headers
            content: Response content (bytes or string)
            
        Returns:
            True if successfully cached, False otherwise
        """
        # Don't cache error responses
        if status_code >= 400:
            return False
        
        cache_file = self._get_cache_file_path(url)
        content_file = self._get_content_file_path(url)
        
        try:
            # Create metadata entry
            metadata = {
                'url': url,
                'status_code': status_code,
                'headers': dict(headers),
                'timestamp': time.time(),
                'content_type': headers.get('Content-Type', '')
            }
            
            # Write metadata to cache file
            with open(cache_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Write content to content file
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            with open(content_file, 'wb') as f:
                f.write(content)
            
            return True
        
        except Exception as e:
            logger.error(f"Error caching response for {url}: {str(e)}")
            # Clean up partially written files
            if cache_file.exists():
                cache_file.unlink()
            if content_file.exists():
                content_file.unlink()
            return False
    
    def clear(self, url: Optional[str] = None) -> int:
        """Clear cache entries.
        
        Args:
            url: Specific URL to clear (optional, if None, clears entire cache)
            
        Returns:
            Number of entries cleared
        """
        if url:
            # Clear specific URL
            cache_file = self._get_cache_file_path(url)
            content_file = self._get_content_file_path(url)
            deleted = 0
            
            if cache_file.exists():
                cache_file.unlink()
                deleted += 1
            
            if content_file.exists():
                content_file.unlink()
                deleted += 1
            
            return deleted // 2  # Count as one entry deleted
        else:
            # Clear entire cache
            count = 0
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
                count += 1
                
            for f in self.cache_dir.glob("*.bin"):
                f.unlink()
            
            return count // 2  # Each entry has two files
    
    def purge_expired(self) -> int:
        """Remove expired cache entries.
        
        Returns:
            Number of expired entries removed
        """
        count = 0
        now = time.time()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    metadata = json.load(f)
                
                cached_time = metadata.get('timestamp', 0)
                if now - cached_time > self.max_age:
                    # Remove expired entry
                    url = metadata.get('url', '')
                    if url:
                        self.clear(url)
                        count += 1
            except Exception as e:
                logger.error(f"Error checking cache entry {cache_file}: {str(e)}")
                # Remove corrupted entry
                cache_file.unlink(missing_ok=True)
                # Try to find and remove the corresponding content file
                content_file = self.cache_dir / f"{cache_file.stem}.bin"
                content_file.unlink(missing_ok=True)
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # Count cache entries
        cache_files = list(self.cache_dir.glob("*.json"))
        entry_count = len(cache_files)
        
        # Calculate cache size
        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.*"))
        
        # Calculate cache age
        now = time.time()
        ages = []
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    metadata = json.load(f)
                
                cached_time = metadata.get('timestamp', 0)
                age = now - cached_time
                ages.append(age)
            except Exception:
                pass
        
        avg_age = sum(ages) / len(ages) if ages else 0
        
        return {
            'entries': entry_count,
            'size_bytes': cache_size,
            'size_human': self._format_size(cache_size),
            'avg_age_seconds': avg_age,
            'avg_age_human': self._format_duration(avg_age),
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable form.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Human-readable size string
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable form.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Human-readable duration string
        """
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            return f"{int(seconds / 60)} minutes"
        elif seconds < 86400:
            return f"{int(seconds / 3600)} hours"
        else:
            return f"{int(seconds / 86400)} days"


# Singleton instance
_cache = None

def get_cache() -> WebCache:
    """Get or create the singleton cache instance.
    
    Returns:
        The global WebCache instance
    """
    global _cache
    if _cache is None:
        _cache = WebCache()
    return _cache