# Euclid Project - Development Guidelines and Structure

## Project Overview

Euclid is a command-line interface (CLI) tool designed to interact with local Ollama models, inspired by Claude Code but with enhanced features. The project aims to provide a feature-rich, offline alternative to cloud-based AI coding assistants while prioritizing privacy, flexibility, and extensibility.

## Project Structure

```
euclid/
├── euclid/                  # Main package directory
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # Command-line interface using Typer
│   ├── config.py            # Configuration management
│   ├── conversation.py      # Conversation history management
│   ├── formatting.py        # Terminal output formatting
│   ├── functions.py         # Function registration and execution
│   ├── functions/           # Specialized function implementations
│   │   ├── __init__.py
│   │   └── web.py           # Web-related functions
│   ├── models.py            # Model management utilities
│   ├── ollama.py            # Ollama API client
│   ├── rag.py               # Retrieval Augmented Generation
│   ├── semantic_cache.py    # Semantic caching for responses
│   ├── server.py            # MCP-compatible API server
│   ├── streaming.py         # Streaming response handling
│   ├── tools/               # Tool implementations
│   │   ├── __init__.py
│   │   ├── advanced.py      # Advanced tools
│   │   ├── agent.py         # Agent functionality
│   │   ├── basic.py         # Basic tools
│   │   ├── batch.py         # Batch processing tools
│   │   ├── context.py       # Context management tools
│   │   ├── file_operations.py # File operation tools
│   │   ├── git.py           # Git-related tools
│   │   ├── registry.py      # Tool registry
│   │   └── web.py           # Web-related tools
│   ├── tui.py               # Terminal user interface
│   └── web_cache.py         # Web content caching
├── tests/                   # Test directory
│   ├── integration/         # Integration tests
│   ├── unit/                # Unit tests
│   ├── test_*.py            # Test modules
│   └── ...
├── prompts/                 # System prompts
│   └── system_prompt.txt    # Default system prompt
├── docker/                  # Docker configuration
├── main.py                  # Entry point
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies
└── ...                      # Documentation and configuration files
```

## Python Best Practices

### Code Style and Formatting

1. **PEP 8 Compliance**: Follow [PEP 8](https://peps.python.org/pep-0008/) style guide for Python code.
   - 4 spaces for indentation
   - Maximum line length of 100 characters
   - Appropriate whitespace in expressions and statements
   - Meaningful variable and function names

2. **Type Hints**: Use type hints consistently to improve code readability and enable static type checking.
   ```python
   def process_data(input_text: str, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
       # Function implementation
   ```

3. **Docstrings**: Include docstrings for all modules, classes, and functions using the Google style format.
   ```python
   def function_name(param1: str, param2: int) -> bool:
       """Short description of function.
       
       Longer description explaining details.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Description of return value
           
       Raises:
           ExceptionType: When and why this exception is raised
       """
       # Function implementation
   ```

### Project Organization

1. **Modular Architecture**: Keep modules focused on specific functionality.
   - Use dependency injection where appropriate
   - Follow separation of concerns principle

2. **Import Organization**: Organize imports in this order:
   - Standard library imports
   - Third-party library imports
   - Local application imports
   - Each group separated by a blank line

3. **Unit Testing**: Maintain high test coverage (target: >90%).
   - Write tests for all new functionality
   - Run tests before committing changes
   - Use mocks for external dependencies

### Performance Considerations

1. **Asynchronous Code**: Use async/await for I/O-bound operations.
   - Network requests
   - File operations
   - Model inference when appropriate

2. **Caching**: Implement appropriate caching strategies.
   - Cache model responses
   - Cache web content
   - Store embedding vectors

3. **Resource Management**: Be mindful of resource usage.
   - Close file handles properly
   - Release resources in exception handlers
   - Use context managers (`with` statements)

## Common Development Tasks

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/euclid.git
cd euclid

# Set up development environment with all dependencies
pip install -e ".[all]"

# Set up pre-commit hooks
./setup-pre-commit.sh
```

### Running Tests

```bash
# Run all tests
python -m unittest discover -s tests

# Run tests with coverage
./tests/run_coverage.sh

# Run only unit tests
./tests/run_unit_tests.sh
```

### Adding New Features

1. **New Function/Tool**:
   - Add implementation in appropriate module
   - Register function with decorator:
     ```python
     @register_function(
         name="FunctionName",
         description="Description of what the function does"
     )
     def function_name(param1: str, param2: int = 0) -> str:
         # Implementation
     ```
   - Add unit tests in corresponding test module
   - Update documentation if necessary

2. **New Command**:
   - Add new command in cli.py using Typer
   - Implement logic or connect to existing function
   - Add unit tests
   - Update help text and README.md

### Dependency Management

1. When adding a new dependency:
   - Add to requirements.txt for core dependencies
   - For optional features, add to the appropriate extras_require section in setup.py
   - Document the dependency in INSTALL.md if it requires special handling

2. Keep dependencies up to date:
   - Regularly review and update dependencies
   - Test thoroughly after updates

## System Requirements

- Python 3.8 or newer
- Ollama installed and running locally
- Sufficient disk space for models and cache

## Debugging

1. Set the appropriate log level for more detailed output:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Use the `--thinking` flag to see the model's thought process.

3. For server debugging, run with `--log-level debug`:
   ```bash
   euclid-cli server start --log-level debug
   ```

## Contribution Workflow

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Ensure all tests pass and coverage is maintained
5. Submit a pull request with clear description of changes

By following these guidelines, we can maintain a high-quality, consistent codebase that is easy to understand and extend.