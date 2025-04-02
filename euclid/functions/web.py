"""Web browsing functions for Euclid."""

import re
import requests
from typing import Optional, Dict, Any, List, Tuple, Union
import json
import logging
import time
from datetime import datetime
import base64
from urllib.parse import urlparse
import os

# Import web tools implementation if available
try:
    from bs4 import BeautifulSoup
    from euclid.tools.web import (
        is_valid_url, 
        html_to_markdown, 
        truncate_content,
        summarize_with_model,
        fetch_image,
        get_search_results
    )
    from euclid.web_cache import get_cache
    HAVE_WEB_TOOLS = True
except ImportError:
    HAVE_WEB_TOOLS = False

from euclid.functions import register_function

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@register_function(
    name="web_fetch", 
    description="Fetch content from a specified URL and process it"
)
def web_fetch(url: str, prompt: Optional[str] = None, include_images: bool = True) -> Dict[str, Any]:
    """Fetch content from a specified URL and process it.
    
    Args:
        url: URL to fetch content from
        prompt: Instruction for processing the content (optional)
        include_images: Whether to include image data (default: True)
        
    Returns:
        Dict with content and metadata
    """
    if not HAVE_WEB_TOOLS:
        return {
            "error": True,
            "message": "Web browsing dependencies not installed. Install beautifulsoup4 and html2text."
        }
    
    # Validate URL
    if not is_valid_url(url):
        return {
            "error": True,
            "message": f"Invalid or unsafe URL: {url}"
        }
    
    # Default prompt if not provided
    if not prompt:
        prompt = "Summarize the main content of this web page concisely."
    
    try:
        # Get the web cache
        cache = get_cache()
        
        # Check if we have a cached version
        cached = cache.get(url)
        response_from_cache = False
        content = None
        headers = {}
        status_code = None
        
        if cached:
            logger.info(f"Using cached version of {url}")
            content = cached.get('content')
            headers = cached.get('headers', {})
            status_code = cached.get('status_code')
            response_from_cache = True
        
        # If not in cache, fetch the content
        if content is None:
            # Add user agent to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Make the request
            response = requests.get(url, headers=headers, timeout=10)
            status_code = response.status_code
            headers = response.headers
            
            # Check for successful response
            if response.status_code != 200:
                return {
                    "error": True,
                    "message": f"Failed to fetch URL (status code: {response.status_code})"
                }
            
            # Get the content
            content = response.text
            
            # Cache the response if successful
            cache.put(url, response.status_code, dict(response.headers), content)
        
        # Process the content
        content_type = headers.get('Content-Type', '').lower()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        metadata = {
            "url": url,
            "status_code": status_code,
            "content_type": content_type,
            "headers": dict(headers),
            "cached": response_from_cache,
            "timestamp": timestamp
        }
        
        if 'text/html' in content_type:
            # Convert HTML to markdown and extract images
            markdown_content, images = html_to_markdown(content, url)
            
            # Summarize the content
            processed_content = summarize_with_model(markdown_content, prompt)
            
            result = {
                "error": False,
                "metadata": metadata,
                "content": processed_content,
                "raw_content": truncate_content(markdown_content, 2000)  # Include first part of raw content
            }
            
            # Include images if requested
            if include_images and images:
                # Limit to first 3 images to reduce size
                image_data = []
                for i, img in enumerate(images[:3]):
                    image_bytes = fetch_image(img['src'])
                    if image_bytes:
                        # Include base64 encoded image
                        base64_img = base64.b64encode(image_bytes).decode('utf-8')
                        alt = img['alt'] or f"Image {i+1}"
                        
                        image_data.append({
                            "src": img['src'],
                            "alt": alt,
                            "type": "base64",
                            "data": base64_img,
                            "context": img.get('context', '')
                        })
                
                result["images"] = image_data
                result["total_images"] = len(images)
            
            return result
        
        elif 'application/json' in content_type:
            # Process JSON content
            try:
                if isinstance(content, str):
                    json_data = json.loads(content)
                else:
                    json_data = content
                
                formatted_json = json.dumps(json_data, indent=2)
                
                # If JSON is too large, summarize it
                if len(formatted_json) > 10000:
                    processed_content = summarize_with_model(formatted_json, prompt)
                else:
                    processed_content = formatted_json
                
                return {
                    "error": False,
                    "metadata": metadata,
                    "content": processed_content,
                    "raw_content": truncate_content(formatted_json, 2000)
                }
            except Exception as e:
                return {
                    "error": True,
                    "message": f"Error parsing JSON: {str(e)}",
                    "metadata": metadata
                }
        
        elif 'image/' in content_type:
            # Handle image content
            try:
                # Convert string content to bytes if needed
                if isinstance(content, str) and hasattr(content, 'encode'):
                    image_data = content.encode('latin1')  # Convert back to bytes
                else:
                    image_data = content
                
                # Encode as base64
                base64_img = base64.b64encode(image_data).decode('utf-8')
                
                return {
                    "error": False,
                    "metadata": metadata,
                    "content": "Image content",
                    "image": {
                        "type": "base64",
                        "data": base64_img,
                        "src": url
                    }
                }
            except Exception as e:
                return {
                    "error": True,
                    "message": f"Error processing image: {str(e)}",
                    "metadata": metadata
                }
        
        elif any(t in content_type for t in ['text/plain', 'text/markdown', 'text/csv']):
            # Process text content
            if len(content) > 10000:
                processed_content = summarize_with_model(content, prompt)
            else:
                processed_content = content
            
            return {
                "error": False,
                "metadata": metadata,
                "content": processed_content,
                "raw_content": truncate_content(content, 2000)
            }
        
        else:
            # Unsupported content type
            return {
                "error": True,
                "message": f"Unsupported content type: {content_type}",
                "metadata": metadata
            }
    
    except requests.exceptions.Timeout:
        return {
            "error": True,
            "message": "Request timed out. The website may be down or too slow."
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "error": True,
            "message": f"Error fetching URL: {str(e)}"
        }
    
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return {
            "error": True,
            "message": f"Error processing content: {str(e)}"
        }

@register_function(
    name="search_web", 
    description="Search the web for information"
)
def search_web(query: str, num_results: int = 5) -> Dict[str, Any]:
    """Search the web for information.
    
    Args:
        query: Search query
        num_results: Number of results to return (max 10)
        
    Returns:
        Dict with search results
    """
    if not HAVE_WEB_TOOLS:
        return {
            "error": True,
            "message": "Web browsing dependencies not installed."
        }
    
    # Limit num_results to a reasonable range
    num_results = max(1, min(10, num_results))
    
    try:
        # Get search results
        results = get_search_results(query, num_results)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "position": result.get("position", 0)
            })
        
        # Build response
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = {
            "error": False,
            "query": query,
            "num_results": len(formatted_results),
            "results": formatted_results,
            "timestamp": timestamp
        }
        
        # Add note if using mock results
        if not os.environ.get('SERPAPI_API_KEY'):
            response["using_mock_results"] = True
            response["note"] = "Using demo search results. For real search results, set the SERPAPI_API_KEY environment variable."
        
        return response
    
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return {
            "error": True,
            "message": f"Error performing search: {str(e)}"
        }

@register_function(
    name="web_cache_stats", 
    description="Get statistics about the web content cache"
)
def web_cache_stats() -> Dict[str, Any]:
    """Get statistics about the web content cache.
    
    Returns:
        Dict with cache statistics
    """
    if not HAVE_WEB_TOOLS:
        return {
            "error": True,
            "message": "Web browsing dependencies not installed."
        }
    
    try:
        cache = get_cache()
        stats = cache.get_stats()
        
        return {
            "error": False,
            "entries": stats['entries'],
            "size_bytes": stats['size_bytes'],
            "size_human": stats['size_human'],
            "avg_age_seconds": stats['avg_age_seconds'],
            "avg_age_human": stats['avg_age_human'],
            "hits": stats['hits'],
            "misses": stats['misses'],
            "hit_ratio": stats['hit_ratio']
        }
    
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return {
            "error": True,
            "message": f"Error getting cache statistics: {str(e)}"
        }