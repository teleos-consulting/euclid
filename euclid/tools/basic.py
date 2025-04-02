import os
import sys
from typing import Optional, List
from pathlib import Path
import platform
import json

from euclid.tools.registry import register_tool

@register_tool("help")
def help_tool() -> str:
    """Display help information about available tools."""
    from euclid.tools.registry import get_available_tools
    
    tools = get_available_tools()
    
    help_text = "# Available Tools\n\n"
    help_text += "Use tools by typing `/tool_name arguments`\n\n"
    
    for tool in sorted(tools):
        help_text += f"- `/{tool}`: "
        # Get the docstring for the tool
        tool_func = sys.modules[__name__].__dict__.get(f"{tool}_tool")
        if tool_func and tool_func.__doc__:
            help_text += tool_func.__doc__.strip()
        else:
            help_text += "No description available."
        help_text += "\n"
    
    return help_text

@register_tool("ls")
def ls_tool(directory: Optional[str] = ".") -> str:
    """List files in a directory."""
    try:
        path = Path(directory).expanduser().resolve()
        if not path.is_dir():
            return f"Error: '{directory}' is not a directory."
        
        files = list(path.iterdir())
        files.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
        
        result = f"# Directory listing for `{path}`\n\n"
        
        for file in files:
            icon = "📁 " if file.is_dir() else "📄 "
            result += f"{icon} {file.name}"    
            if file.is_dir():
                result += "/"
            result += "\n"
        
        return result
    
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@register_tool("cat")
def cat_tool(file_path: str) -> str:
    """Display contents of a file."""
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            return f"Error: '{file_path}' is not a file or does not exist."
        
        content = path.read_text()
        
        file_ext = path.suffix.lower()
        if file_ext in (".py", ".js", ".ts", ".html", ".css", ".json", ".md"):
            return f"```{file_ext[1:]}\n{content}\n```"
        else:
            return f"```\n{content}\n```"
    
    except Exception as e:
        return f"Error reading file: {str(e)}"

@register_tool("pwd")
def pwd_tool() -> str:
    """Print working directory."""
    return f"Current working directory: `{os.getcwd()}`"

@register_tool("system")
def system_tool() -> str:
    """Display system information."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor() or "Unknown",
        "system": platform.system(),
        "release": platform.release(),
    }
    
    result = "# System Information\n\n"
    for key, value in info.items():
        result += f"- **{key.replace('_', ' ').title()}**: {value}\n"
    
    return result

@register_tool("models")
def models_tool() -> str:
    """List available models from Ollama."""
    from euclid.ollama import OllamaClient
    
    client = OllamaClient()
    try:
        models = client.get_available_models()
        
        if not models:
            return "No models found. Make sure Ollama is running."
        
        result = "# Available Ollama Models\n\n"
        for model in models:
            result += f"- **{model['name']}**"    
            if "size" in model:
                result += f" ({model['size']})"
            result += "\n"
        
        return result
    
    except Exception as e:
        return f"Error getting models: {str(e)}\n\nMake sure Ollama is running at {client.base_url}"
