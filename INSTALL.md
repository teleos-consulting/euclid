# Installation and Usage Guide

## Prerequisites

1. **Python 3.8+** - Euclid requires Python 3.8 or newer
2. **Ollama** - You need to have Ollama installed and running locally

## Installing Ollama

If you don't have Ollama installed yet, follow these instructions:

### macOS/Linux
```bash
curl https://ollama.ai/install.sh | sh
```

### Windows
Download and install from: https://ollama.ai/download/windows

## Running Ollama

After installation, make sure Ollama is running in the background:

```bash
# Start the Ollama server
ollama serve
```

In a separate terminal, you can pull a model to use with Euclid:

```bash
# Pull a model (choose one that suits your needs)
ollama pull llama3  # Best general model
ollama pull mistral  # Alternative model
ollama pull phi  # Smaller model for lower-resource machines
```

## Installing Euclid

### Method 1: Standard Installation

1. Clone the repository or download the files

2. Install in development mode with all dependencies:
   ```bash
   cd euclid
   pip install -e ".[all]"
   ```

3. Or install with minimal dependencies:
   ```bash
   cd euclid
   pip install -e .
   ```

4. If you want specific optional features, install the corresponding extras:
   ```bash
   # For RAG (Retrieval Augmented Generation) functionality
   pip install -e ".[rag]"
   
   # For web browsing capabilities
   pip install -e ".[web]"
   
   # For server functionality
   pip install -e ".[server]"
   
   # For development and testing
   pip install -e ".[dev,test]"
   ```

5. Alternatively, run without installing (requires dependencies to be installed manually):
   ```bash
   cd euclid
   pip install -r requirements.txt
   python main.py
   ```

### Method 2: Docker Installation

For a containerized setup that includes both Euclid and Ollama:

1. Make sure you have Docker and Docker Compose installed

2. Clone the repository and navigate to it:
   ```bash
   git clone https://github.com/yourusername/euclid.git
   cd euclid
   ```

3. Start the containers:
   ```bash
   docker-compose up -d
   ```

4. Access Euclid through the container:
   ```bash
   docker exec -it euclid-app bash
   euclid chat
   ```

## Configuration

You can configure Euclid using environment variables or by creating a `.env` file in the project directory:

```
# .env file example
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

When using Docker, modify the environment variables in the `docker-compose.yml` file.

## Usage

### Interactive Chat

```bash
# Start interactive chat
euclid chat

# Use a specific model
euclid chat --model mistral

# Use a system prompt from a file
euclid chat --system "$(cat prompts/system_prompt.txt)"

# Continue a previous conversation
euclid chat --conversation <conversation_id>
```

### Single Prompt

```bash
# Run a single prompt
euclid run "What is the capital of France?"

# With a specific model
euclid run "Write a Python function to calculate Fibonacci numbers" --model codellama
```

### Advanced Terminal UI

```bash
# Launch the split-screen TUI
euclid tui

# With a specific model
euclid tui --model mistral

# Without showing thinking
euclid tui --no-thinking
```

### Other Commands

```bash
# List available models
euclid models

# View conversation history
euclid history
```

## Using Tools

In the chat interface, you can use various tools by typing slash commands:

```
/help           - List available tools
/ls [directory] - List files in a directory
/cat <file>     - Display file contents
/pwd            - Show current directory
/system         - Show system information
/models         - List available Ollama models
/search <pattern> [directory] - Search for files containing a pattern
/find <pattern> [directory]   - Find files matching a pattern
/exec <command> - Execute a shell command
/edit <file>    - Open a file in your default editor
/wget <url> [output_file]     - Download a file from the web
```

## Optional Features

### RAG Functionality

To enable RAG (Retrieval Augmented Generation), install the additional dependencies:

```bash
pip install sentence-transformers scikit-learn
```

When using Docker, these are included automatically.

### Image Rendering

For terminal image rendering support, install:

```bash
pip install terminal-image
```