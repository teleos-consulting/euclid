# Euclid

A CLI tool for interacting with local Ollama models, inspired by Claude Code but with significant enhancements.

## Features

- **Beautiful CLI Experience**
  - Interactive CLI interface with rich markdown rendering
  - Code syntax highlighting for all languages
  - Terminal image rendering
  - Progress spinners and animations

- **Full-Featured TUI Mode**
  - Split-screen view showing thinking and results
  - Real-time function call visualization
  - Model switching interface
  - Multi-line editing with syntax highlighting

- **Model Management**
  - Download, list, and remove models directly
  - Automatic model downloading when needed
  - Model performance benchmarking
  - Model details and parameter inspection

- **Advanced Capabilities**
  - Structured function calling with JSON schema validation
  - Parallel execution with BatchTool
  - Autonomous agent functionality
  - RAG (Retrieval Augmented Generation) with vector database
  - Web browsing and content summarization
  - Thinking mode display

- **Developer Features**
  - Conversation history management
  - Custom system prompts
  - Tool ecosystem with file operations
  - Local embedding and semantic search

## Installation

### Method 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/euclid.git
cd euclid

# Install with all dependencies (recommended for full functionality)
pip install -e ".[all]"

# OR install with minimal dependencies
pip install -e .

# OR install specific optional features:
# For RAG (Retrieval Augmented Generation) functionality
pip install -e ".[rag]"

# For web browsing capabilities
pip install -e ".[web]"

# For development and testing
pip install -e ".[dev,test]"
```

For detailed installation instructions, see [Installation Guide](INSTALL.md).

### Method 2: Docker Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/euclid.git
cd euclid

# Start the containers
docker-compose -f docker/docker-compose.yml up -d

# Access Euclid
docker exec -it euclid-app bash
euclid chat
```

For more details on Docker setup, see [Docker README](docker/README.md).

## Usage

### Interactive Chat

```bash
# Start the advanced terminal UI (default)
euclid

# Use the CLI interface instead
euclid-cli chat

# Use a specific model
euclid --model mistral

# Show model's thinking process
euclid --thinking
```

### Single Prompt

```bash
# Run a single prompt with the CLI interface
euclid-cli run "Your prompt here"

# With a specific model
euclid-cli run "Write a Python function to calculate Fibonacci numbers" --model codellama
```

### Model Management

```bash
# List available models
euclid-cli models list

# Pull a new model
euclid-cli models pull llama3

# Get model details
euclid-cli models details llama3

# Benchmark model performance
euclid-cli models benchmark llama3 --iterations 5
```

### RAG Functionality (if sentence-transformers is installed)

```bash
# Create a new collection
euclid-cli rag create "My Knowledge Base" --description "General knowledge"

# Add a document from a file
euclid-cli rag add my-collection-id --file data.txt --title "Important Data"

# Query the collection
euclid-cli rag query my-collection-id "How does photosynthesis work?"
```

### Web Browsing

```bash
# Fetch and analyze a web page
euclid-cli web fetch https://example.com --prompt "Summarize the main points"

# Fetch with images disabled (for faster results)
euclid-cli web fetch https://example.com --no-images

# Search the web (uses mock data by default)
euclid-cli web search "quantum computing basics"

# Search the web with SerpAPI (set SERPAPI_API_KEY env var first)
SERPAPI_API_KEY=your_key euclid-cli web search "quantum computing basics"

# View web cache statistics
euclid-cli web_cache stats

# Clear web cache
euclid-cli web_cache clear

# Purge expired entries from web cache
euclid-cli web_cache purge
```

### Additional Commands

```bash
# View available functions
euclid-cli functions

# View conversation history
euclid-cli history

# Launch autonomous agent
euclid-cli agent "Analyze the project structure and summarize it"

# Start MCP-compatible API server
euclid-cli server start --port 8000
```

## Functions and Tools

Euclid supports the following functions:

- **File Operations**: View, LS, GlobTool, GrepTool, Edit, Replace
- **Model Management**: ListModels, PullModel, RemoveModel, ModelDetails, BenchmarkModel
- **RAG Operations**: CreateCollection, ListCollections, AddDocument, QueryCollection
- **Web Operations**: web_fetch, search_web
- **Meta-Functions**: BatchTool, dispatch_agent

## Configuration

### Environment Variables

Euclid uses environment variables for configuration:

- `OLLAMA_BASE_URL`: Base URL for your Ollama server (default: http://localhost:11434)
- `OLLAMA_MODEL`: Default model to use (default: llama3)

You can set these in a `.env` file in your project directory.

### EUCLID.md Files

For project-specific configuration, Euclid supports a special `EUCLID.md` file in your working directory. When present, Euclid automatically incorporates the contents of this file into the system prompt, providing project-specific context and instructions to the AI.

The `EUCLID.md` file can contain:
- Project overview and purpose
- Codebase structure and organization
- Development guidelines and best practices
- Shortcuts and common commands
- Instructions for specific tasks

Example `EUCLID.md` structure:
```markdown
# EUCLID.md - Project-specific Instructions

## Project Overview
Brief description of the project...

## Development Guidelines
Guidelines for code style, testing, etc...

## Shortcuts and Commands
Common commands and shortcuts...

## Common Tasks
Instructions for specific tasks...
```

You can also define a complete system prompt by including a "System Prompt" section:

```markdown
# System Prompt

You are an AI assistant specialized in helping with this specific project...
[rest of custom system prompt]

# Project Overview
...
```

## Advantages Over Claude Code

- **Works Offline**: Uses your local Ollama models, no internet required
- **No API Costs**: Free to use with any Ollama model
- **Model Flexibility**: Switch between any Ollama model instantly
- **RAG Capability**: Built-in vector database for knowledge retrieval
- **Advanced TUI**: Split-screen interface showing both thinking and responses
- **Model Insights**: Benchmarking and detailed model information
- **Privacy**: All processing happens locally on your machine
- **API Server**: MCP-compatible API for integration with other tools
- **EUCLID.md Support**: Similar to Claude's CLAUDE.md file, provide project-specific instructions

## API Server

Euclid includes a Model Control Protocol (MCP) compatible API server for programmatic access:

```bash
# Start the API server
euclid-cli server start --port 8000
```

The server provides the following endpoints:

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Generate chat completions
- `GET /v1/functions` - List available functions
- `POST /v1/functions/{function_name}/execute` - Execute a specific function
- `POST /v1/web/fetch` - Fetch and analyze content from a URL
- `POST /v1/web/search` - Search the web for information

The API is compatible with the OpenAI API format, allowing you to use Euclid with tools designed for commercial LLMs.

For detailed documentation on the API server, see [API Documentation](docs/api/README.md).

## Testing

For information on how to run tests and contribute to testing, see [Test Plan](tests/TEST_PLAN.md).

## Contributing

Contributions are welcome! Please see [Contributing Guide](CONTRIBUTING.md) for more information.

## Documentation

- [Installation Guide](INSTALL.md): Detailed installation instructions
- [Docker Setup](docker/README.md): Running with Docker
- [Contributing](CONTRIBUTING.md): How to contribute to the project
- [Test Plan](tests/TEST_PLAN.md): Testing strategy and process