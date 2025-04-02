# Web Browsing Examples

This directory contains examples demonstrating the web browsing functionality in Euclid.

## Prerequisites

Make sure you have the web browsing dependencies installed:

```bash
pip install beautifulsoup4 html2text
```

## Examples

### Python Script Examples

The `test_web_fetch.py` script provides a simple interface to test the web browsing functionality:

```bash
# Fetch and analyze a web page
python examples/web/test_web_fetch.py fetch https://example.com --prompt "Summarize this page"

# Search the web (limited implementation)
python examples/web/test_web_fetch.py search "quantum computing basics"
```

### CLI Examples

You can also use the built-in CLI commands:

```bash
# Fetch and analyze a web page
euclid-cli web fetch https://example.com --prompt "Summarize this page"

# Search the web (limited implementation)
euclid-cli web search "quantum computing basics"
```

### API Server Examples

When running the Euclid API server, you can use the following endpoints:

```bash
# Start the server
euclid-cli server start

# Fetch a web page
curl -X POST http://localhost:8000/v1/web/fetch -H "Content-Type: application/json" -d '{
  "url": "https://example.com",
  "prompt": "Summarize this page"
}'

# Search the web
curl -X POST http://localhost:8000/v1/web/search -H "Content-Type: application/json" -d '{
  "query": "quantum computing basics",
  "num_results": 5
}'
```

## Implementation Details

The web browsing functionality in Euclid consists of:

1. **Web Tool**: A slash-command tool for interactive CLI usage
2. **Function Interface**: JSON-schema validated function calls for structured outputs
3. **API Endpoints**: REST API for programmatic access

The implementation handles:

- URL validation and safety checks
- HTML to markdown conversion
- Content summarization using local Ollama models
- JSON processing for structured data
- Content truncation for large pages