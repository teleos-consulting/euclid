# Euclid

A CLI tool for interacting with local Ollama models, inspired by Claude Code.

## Features

- Interactive CLI interface with beautiful markdown rendering
- Structured function calling for file operations and system tasks
- Parallel execution with BatchTool
- Autonomous agent capabilities
- Conversation history management
- Streaming responses with thinking mode
- Code syntax highlighting
- Support for image rendering in terminals

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/euclid.git
cd euclid

# Install dependencies
pip install -e .
```

## Usage

```bash
# Start interactive chat session
euclid chat

# Run a single prompt
euclid run "Your prompt here"

# List available models
euclid models

# View conversation history
euclid history

# Launch an autonomous agent
euclid agent "Analyze the project structure and summarize it"

# List available functions
euclid functions
```

## Advanced Options

### Chat Mode

```bash
# Use a specific model
euclid chat --model llama3

# Use a custom system prompt
euclid chat --system "You are a helpful assistant"

# Use a system prompt from a file
euclid chat --system-file prompts/my_prompt.txt

# Continue a previous conversation
euclid chat --conversation <conversation_id>

# Show the model's thinking process
euclid chat --thinking

# Disable function calling
euclid chat --no-functions
```

### Run Mode

```bash
# Run with a specific model
euclid run "Write a Python function to calculate Fibonacci numbers" --model codellama

# Run with a custom system prompt
euclid run "Explain quantum computing" --system "You are a quantum physics expert"

# Disable streaming output
euclid run "Tell me a joke" --no-stream

# Show the model's thinking process
euclid run "Solve this math problem: 3x + 5 = 14" --thinking
```

## Functions

Euclid supports the following functions:

- **View**: Read files with syntax highlighting
- **LS**: List directory contents
- **GlobTool**: Find files matching patterns
- **GrepTool**: Search file contents
- **Edit**: Modify file contents
- **Replace**: Create or overwrite files
- **BatchTool**: Run multiple functions in parallel
- **dispatch_agent**: Launch an autonomous agent

## Configuration

Euclid uses environment variables for configuration:

- `OLLAMA_BASE_URL`: Base URL for your Ollama server (default: http://localhost:11434)
- `OLLAMA_MODEL`: Default model to use (default: llama3)

You can set these in a `.env` file in your project directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.