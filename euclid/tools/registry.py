"""Tool registry for classic slash-command style tools."""

from typing import Dict, Callable, Optional, List, Any

# Tool registry
_TOOLS: Dict[str, Callable] = {}

def register_tool(name: str):
    """Decorator to register a tool function.
    
    Args:
        name: Name of the tool.
    """
    def decorator(func: Callable):
        _TOOLS[name] = func
        return func
    return decorator

def get_available_tools() -> List[str]:
    """Get a list of available tool names.
    
    Returns:
        List of tool names.
    """
    return list(_TOOLS.keys())

def run_tool(name: str, input_text: str) -> str:
    """Run a tool with the given input.
    
    Args:
        name: Name of the tool to run.
        input_text: Input text for the tool.
        
    Returns:
        Result of the tool execution.
    """
    if name not in _TOOLS:
        return f"Error: Tool '{name}' not found."
    
    try:
        # Parse arguments from input text
        # Format: /tool_name arg1 arg2 arg3...
        args = input_text.split()[1:]
        return _TOOLS[name](*args)
    except Exception as e:
        return f"Error running tool '{name}': {str(e)}"

# Import tool modules
from euclid.tools import basic
from euclid.tools import advanced
