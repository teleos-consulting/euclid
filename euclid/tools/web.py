"""Web browsing tools for Euclid."""

import re
import requests
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import tempfile
import json
import logging
import time
import os
import io
from datetime import datetime
from bs4 import BeautifulSoup
import urllib.parse
from urllib.parse import urlparse
import base64

from euclid.tools.registry import register_tool
from euclid.ollama import OllamaClient
from euclid.models import ModelRegistry
from euclid.web_cache import get_cache, WebCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_valid_url(url: str) -> bool:
    """Check if the URL is valid and safe.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is valid and safe, False otherwise
    """
    try:
        result = urlparse(url)
        # Check if the URL has a scheme and netloc
        if not all([result.scheme, result.netloc]):
            return False
        
        # Check for allowed schemes
        if result.scheme not in ['http', 'https']:
            return False
        
        # Additional security checks could be added here
        return True
    except Exception:
        return False

def extract_images(soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
    """Extract images from HTML.
    
    Args:
        soup: BeautifulSoup parsed HTML
        base_url: Base URL for resolving relative URLs
        
    Returns:
        List of image information dictionaries
    """
    images = []
    
    for img in soup.find_all('img', src=True):
        src = img['src']
        
        # Skip data URLs, empty sources, and icons
        if not src or src.startswith('data:') or 'icon' in src.lower():
            continue
        
        # Convert relative URLs to absolute
        if not src.startswith(('http://', 'https://')):
            src = urllib.parse.urljoin(base_url, src)
        
        # Get alt text and dimensions
        alt = img.get('alt', '')
        width = img.get('width', '')
        height = img.get('height', '')
        
        # Extract nearby text for context
        parent = img.parent
        context = parent.get_text().strip() if parent else ''
        
        images.append({
            'src': src,
            'alt': alt,
            'width': width,
            'height': height,
            'context': context[:100] + '...' if len(context) > 100 else context
        })
    
    return images

def fetch_image(url: str) -> Optional[bytes]:
    """Fetch an image from a URL.
    
    Args:
        url: Image URL
        
    Returns:
        Image data as bytes, or None if failed
    """
    try:
        # Check cache first
        cache = get_cache()
        cached = cache.get(url)
        
        if cached and cached.get('status_code') == 200:
            logger.info(f"Using cached image: {url}")
            return cached.get('content')
        
        # Add user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the image
        response = requests.get(url, headers=headers, timeout=5, stream=True)
        
        if response.status_code == 200:
            # Check if it's actually an image
            content_type = response.headers.get('Content-Type', '').lower()
            if not content_type.startswith('image/'):
                logger.warning(f"URL does not point to an image: {url} (Content-Type: {content_type})")
                return None
            
            # Get image data
            image_data = response.content
            
            # Cache the image
            cache.put(url, response.status_code, response.headers, image_data)
            
            return image_data
        else:
            logger.warning(f"Failed to fetch image: {url} (Status: {response.status_code})")
            return None
    
    except Exception as e:
        logger.error(f"Error fetching image {url}: {str(e)}")
        return None

def format_image_for_terminal(image_data: bytes, max_width: int = 100) -> str:
    """Format an image for display in the terminal.
    
    Args:
        image_data: Image data as bytes
        max_width: Maximum display width
        
    Returns:
        Formatted image for terminal display
    """
    try:
        # Convert to base64 for terminal-image or for markdown rendering
        base64_img = base64.b64encode(image_data).decode('utf-8')
        
        # For Markdown output, we can use an HTML img tag with base64 data
        return f'<img src="data:image/png;base64,{base64_img}" style="max-width: {max_width}px" />'
    
    except Exception as e:
        logger.error(f"Error formatting image for terminal: {str(e)}")
        return "[Error displaying image]"

def html_to_markdown(html: str, url: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Convert HTML to markdown.
    
    Args:
        html: HTML content
        url: Source URL for resolving relative links
        
    Returns:
        Tuple of (markdown text, list of image information)
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style tags
    for script in soup(["script", "style"]):
        script.extract()
    
    # Extract images before further processing
    images = extract_images(soup, url)
    
    # Get the text
    text = soup.get_text()
    
    # Process text to remove excessive newlines and spaces
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    # Try to preserve headings
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        heading_text = heading.get_text().strip()
        level = int(heading.name[1])
        text = text.replace(heading_text, f"\n{'#' * level} {heading_text}\n")
    
    # Extract links
    links = []
    for link in soup.find_all('a', href=True):
        link_text = link.get_text().strip()
        href = link['href']
        if link_text and href and 'javascript:' not in href:
            # Convert relative URLs to absolute
            if not href.startswith(('http://', 'https://')):
                href = urllib.parse.urljoin(url, href)
            links.append(f"- [{link_text}]({href})")
    
    if links:
        text += "\n\n## Links\n" + "\n".join(links)
    
    # Add a section for images if there are any
    if images:
        text += "\n\n## Images\n"
        for i, img in enumerate(images):
            alt = img['alt'] or f"Image {i+1}"
            text += f"- Image {i+1}: {alt}\n"
    
    return text, images

def truncate_content(content: str, max_length: int = 8000) -> str:
    """Truncate content to a maximum length while preserving structure.
    
    Args:
        content: Content to truncate
        max_length: Maximum length
        
    Returns:
        Truncated content
    """
    if len(content) <= max_length:
        return content
    
    # Try to truncate at paragraph breaks
    paragraphs = content.split("\n\n")
    result = ""
    
    for paragraph in paragraphs:
        if len(result) + len(paragraph) + 2 <= max_length - 100:  # Leave room for ellipsis
            result += paragraph + "\n\n"
        else:
            remaining = max_length - len(result) - 100
            if remaining > 0:
                result += paragraph[:remaining] + "...\n\n"
            break
    
    result += "\n\n[Content truncated due to length limits]"
    return result

def summarize_with_model(content: str, prompt: str, model: Optional[str] = None) -> str:
    """Summarize content using a model.
    
    Args:
        content: Content to summarize
        prompt: Instruction for summarization
        model: Model to use (optional)
        
    Returns:
        Summarized content
    """
    # Truncate content if it's too long
    content = truncate_content(content)
    
    # Get the default model if not specified
    if not model:
        model_registry = ModelRegistry()
        available_models = [m.name for m in model_registry.get_available_models()]
        model = available_models[0] if available_models else "mistral"
    
    # Initialize client with the model
    client = OllamaClient(model=model)
    
    # Prepare messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts information from web content."},
        {"role": "user", "content": f"{prompt}\n\nHere is the content to analyze:\n\n{content}"}
    ]
    
    try:
        # Generate non-streaming response
        response = client.chat_completion(messages, stream=False)
        
        if "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        else:
            return "Error: Invalid response format from the model."
    except Exception as e:
        logger.error(f"Error summarizing content: {str(e)}")
        return f"Error summarizing content: {str(e)}"

@register_tool("web")
def web_tool(url: str, prompt: Optional[str] = None, extract_images: bool = True) -> str:
    """Fetch content from a specified URL and analyze it.
    
    Args:
        url: URL to fetch content from
        prompt: Specific instructions for analyzing the content (optional)
        extract_images: Whether to extract and display images (default: True)
    """
    # Validate URL
    if not is_valid_url(url):
        return f"Error: Invalid or unsafe URL: {url}"
    
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
                return f"Error: Failed to fetch URL (status code: {response.status_code})"
            
            # Get the content
            content = response.text
            
            # Cache the response if successful
            cache.put(url, response.status_code, dict(response.headers), content)
        
        # Process the content
        content_type = headers.get('Content-Type', '').lower()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cached_text = " (cached)" if response_from_cache else ""
        metadata = f"Fetched at: {timestamp}{cached_text}"
        
        if 'text/html' in content_type:
            # Convert HTML to markdown and extract images
            markdown_content, images = html_to_markdown(content, url)
            
            # Build result text
            result_text = f"# Web Content Analysis: {url}\n\n"
            
            # Summarize the content
            summary = summarize_with_model(markdown_content, prompt)
            result_text += summary
            
            # Add metadata
            result_text += f"\n\n---\n{metadata}\n"
            
            # Include image previews if requested
            if extract_images and images:
                # Limit to first 3 images to avoid too much clutter
                for i, img in enumerate(images[:3]):
                    if i >= 3:  # Limit to 3 images
                        break
                    
                    # Fetch the image
                    image_data = fetch_image(img['src'])
                    if image_data:
                        # Add image caption
                        img_caption = img['alt'] or f"Image {i+1}"
                        result_text += f"\n\n### {img_caption}\n"
                        
                        # Add image context if available
                        if img['context']:
                            result_text += f"Context: {img['context']}\n\n"
                        
                        # Format image for display
                        result_text += format_image_for_terminal(image_data)
                
                # Add note if there are more images
                if len(images) > 3:
                    result_text += f"\n\n*...and {len(images) - 3} more images*"
            
            return result_text
        
        elif 'application/json' in content_type:
            # Pretty print JSON
            try:
                if isinstance(content, str):
                    json_data = json.loads(content)
                else:
                    json_data = content
                    
                formatted_json = json.dumps(json_data, indent=2)
                
                result_text = f"# JSON Content from {url}\n\n"
                
                # If JSON is too large, summarize it
                if len(formatted_json) > 10000:
                    result_text += summarize_with_model(formatted_json, prompt)
                else:
                    result_text += f"```json\n{formatted_json[:10000]}\n```"
                    if len(formatted_json) > 10000:
                        result_text += "\n\n*Content truncated due to length...*"
                
                # Add metadata
                result_text += f"\n\n---\n{metadata}\n"
                
                return result_text
            except Exception as e:
                return f"Error parsing JSON: {str(e)}"
        
        elif 'image/' in content_type:
            # Handle image content
            result_text = f"# Image Content from {url}\n\n"
            
            # If content is string, it might be a cached binary that was converted to string
            if isinstance(content, str) and hasattr(content, 'encode'):
                image_data = content.encode('latin1')  # Convert back to bytes
            else:
                image_data = content
                
            # Format image for display
            image_display = format_image_for_terminal(image_data)
            result_text += image_display
            
            # Add metadata
            result_text += f"\n\n---\n{metadata}\n"
            
            return result_text
        
        elif any(t in content_type for t in ['text/plain', 'text/markdown', 'text/csv']):
            # Text content, return as is or summarize if long
            result_text = f"# Text Content from {url}\n\n"
            
            if len(content) > 10000:
                result_text += summarize_with_model(content, prompt)
            else:
                result_text += f"```\n{content[:10000]}\n```"
                if len(content) > 10000:
                    result_text += "\n\n*Content truncated due to length...*"
            
            # Add metadata
            result_text += f"\n\n---\n{metadata}\n"
            
            return result_text
        
        else:
            # Unsupported content type
            return f"Error: Unsupported content type: {content_type}"
    
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The website may be down or too slow."
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return f"Error processing content: {str(e)}"

def get_search_results(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Get search results for a query (with API if available, otherwise fake data).
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of search result dictionaries
    """
    # Check for SERPAPI_API_KEY in environment
    import os
    serpapi_key = os.environ.get('SERPAPI_API_KEY')
    
    # If we have SerpAPI, use it
    if serpapi_key:
        try:
            cache = get_cache()
            cache_key = f"search_{query}_{num_results}"
            cached = cache.get(cache_key)
            
            if cached:
                logger.info(f"Using cached search results for: {query}")
                return cached.get('content')
            
            import json
            from urllib.parse import urlencode
            
            # Build parameters
            params = {
                "engine": "google",
                "q": query,
                "api_key": serpapi_key,
                "num": min(num_results, 10)  # SerpAPI limit
            }
            
            # Make the request
            url = f"https://serpapi.com/search?{urlencode(params)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            logger.info(f"Making search request to SerpAPI for: {query}")
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"SerpAPI error: {response.status_code} - {response.text}")
                return _generate_mock_search_results(query, num_results)
            
            # Parse the response
            data = response.json()
            
            # Get organic results
            organic_results = data.get('organic_results', [])
            
            # Format the results
            results = []
            for result in organic_results[:num_results]:
                results.append({
                    "title": result.get('title', ''),
                    "link": result.get('link', ''),
                    "snippet": result.get('snippet', ''),
                    "position": result.get('position', 0)
                })
            
            # Cache the results
            cache.put(cache_key, 200, {"Content-Type": "application/json"}, results)
            
            return results
        
        except Exception as e:
            logger.error(f"Error using SerpAPI: {str(e)}")
            return _generate_mock_search_results(query, num_results)
    else:
        # No API key, use mock results
        return _generate_mock_search_results(query, num_results)

def _generate_mock_search_results(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Generate mock search results for demo purposes.
    
    Args:
        query: Search query
        num_results: Number of results to generate
        
    Returns:
        List of mock search result dictionaries
    """
    results = []
    
    # Create some fake domains and snippets based on the query
    domains = ["example.com", "infosite.org", "encyclopedia.net", "techinfo.dev", "research.edu"]
    
    for i in range(min(num_results, len(domains))):
        # Clean query for URL
        clean_query = query.lower().replace(" ", "-")
        
        results.append({
            "title": f"{query.title()} - Information and Resources",
            "link": f"https://{domains[i]}/{clean_query}",
            "snippet": f"Learn about {query} with detailed explanations and examples. Find the most relevant information about {query} and related topics.",
            "position": i + 1
        })
    
    logger.info(f"Generated mock search results for: {query}")
    return results

@register_tool("search")
def search_tool(query: str, num_results: int = 5) -> str:
    """Search the web for information.
    
    Args:
        query: Search query
        num_results: Number of results to return (max 10)
    """
    # Limit num_results to a reasonable range
    num_results = max(1, min(10, num_results))
    
    try:
        # Get search results
        results = get_search_results(query, num_results)
        
        # Format results as markdown
        output = f"# Search Results for: {query}\n\n"
        
        if not results:
            output += "No results found."
            return output
        
        # Add results
        for i, result in enumerate(results):
            position = result.get('position', i + 1)
            title = result.get('title', 'Untitled')
            link = result.get('link', '')
            snippet = result.get('snippet', 'No description available.')
            
            output += f"## {position}. {title}\n"
            output += f"[{link}]({link})\n\n"
            output += f"{snippet}\n\n"
        
        # Add a note if using mock results
        if not os.environ.get('SERPAPI_API_KEY'):
            output += "\n\n*Note: Using demo search results. For real search results, set the SERPAPI_API_KEY environment variable.*"
        
        return output
    
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return f"Error performing search: {str(e)}"

@register_tool("web_cache")
def web_cache_tool(command: str, url: Optional[str] = None) -> str:
    """Manage the web content cache.
    
    Args:
        command: Cache command to execute (stats, clear, purge)
        url: URL to clear from cache (for clear command only)
    """
    cache = get_cache()
    
    if command == "stats":
        # Get cache statistics
        stats = cache.get_stats()
        
        output = "# Web Cache Statistics\n\n"
        output += f"Entries: {stats['entries']}\n"
        output += f"Size: {stats['size_human']}\n"
        output += f"Average age: {stats['avg_age_human']}\n"
        output += f"Hit ratio: {stats['hit_ratio']:.2%} ({stats['hits']} hits, {stats['misses']} misses)\n"
        
        return output
    
    elif command == "clear":
        # Clear cache entries
        if url:
            # Clear specific URL
            count = cache.clear(url)
            if count > 0:
                return f"Cleared cache for {url}"
            else:
                return f"URL not found in cache: {url}"
        else:
            # Clear entire cache
            count = cache.clear()
            return f"Cleared {count} entries from cache"
    
    elif command == "purge":
        # Purge expired entries
        count = cache.purge_expired()
        return f"Purged {count} expired entries from cache"
    
    else:
        return f"Error: Unknown command '{command}'. Valid commands are: stats, clear, purge"