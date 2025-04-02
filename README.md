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
  - Thinking mode display

- **Developer Features**
  - Conversation history management
  - Custom system prompts
  - Tool ecosystem with file operations
  - Local embedding and semantic search

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/euclid.git
cd euclid

# Install dependencies
pip install -e .

# Optional: Install sentence-transformers for RAG functionality
pip install sentence-transformers
```

## Usage

### Interactive Chat

```bash
# Start basic chat interface
euclid chat

# Launch advanced terminal UI
euclid tui

# Use a specific model
euclid chat --model llama3

# Show model's thinking process
euclid chat --thinking
```

### Single Prompt

```bash
# Run a single prompt
euclid run "Your prompt here"

# With a specific model
euclid run "Write a Python function to calculate Fibonacci numbers" --model codellama
```

### Model Management

```bash
# List available models
euclid models list

# Pull a new model
euclid models pull llama3

# Get model details
euclid models details llama3

# Benchmark model performance
euclid models benchmark llama3 --iterations 5
```

### RAG Functionality (if sentence-transformers is installed)

```bash
# Create a new collection
euclid rag create "My Knowledge Base" --description "General knowledge"

# Add a document from a file
euclid rag add my-collection-id --file data.txt --title "Important Data"

# Query the collection
euclid rag query my-collection-id "How does photosynthesis work?"
```

### Additional Commands

```bash
# View available functions
euclid functions

# View conversation history
euclid history

# Launch autonomous agent
euclid agent "Analyze the project structure and summarize it"
```

## Functions and Tools

Euclid supports the following functions:

- **File Operations**: View, LS, GlobTool, GrepTool, Edit, Replace
- **Model Management**: ListModels, PullModel, RemoveModel, ModelDetails, BenchmarkModel
- **RAG Operations**: CreateCollection, ListCollections, AddDocument, QueryCollection
- **Meta-Functions**: BatchTool, dispatch_agent

## Configuration

Euclid uses environment variables for configuration:

- `OLLAMA_BASE_URL`: Base URL for your Ollama server (default: http://localhost:11434)
- `OLLAMA_MODEL`: Default model to use (default: llama3)

You can set these in a `.env` file in your project directory.

## Advantages Over Claude Code

- **Works Offline**: Uses your local Ollama models, no internet required
- **No API Costs**: Free to use with any Ollama model
- **Model Flexibility**: Switch between any Ollama model instantly
- **RAG Capability**: Built-in vector database for knowledge retrieval
- **Advanced TUI**: Split-screen interface showing both thinking and responses
- **Model Insights**: Benchmarking and detailed model information
- **Privacy**: All processing happens locally on your machine

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.