import os
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Optional, List, Dict, Any

from euclid.tools.registry import register_tool

@register_tool("search")
def search_tool(pattern: str, directory: Optional[str] = ".") -> str:
    """Search for files containing a pattern."""
    try:
        path = Path(directory).expanduser().resolve()
        if not path.is_dir():
            return f"Error: '{directory}' is not a directory."
        
        result = "# Search Results\n\n"
        
        # Use grep or similar to search for the pattern
        try:
            output = subprocess.check_output(
                ["grep", "-r", "-l", pattern, str(path)],
                stderr=subprocess.STDOUT,
                text=True
            )
            
            if not output.strip():
                return f"No files containing '{pattern}' found in {path}."
            
            result += f"Files containing '{pattern}':\n\n"
            for line in output.strip().split("\n"):
                file_path = line.strip()
                result += f"- `{file_path}`\n"
        
        except subprocess.CalledProcessError:
            result += f"No files containing '{pattern}' found in {path}.\n"
        
        return result
    
    except Exception as e:
        return f"Error searching for pattern: {str(e)}"

@register_tool("find")
def find_tool(pattern: str, directory: Optional[str] = ".") -> str:
    """Find files matching a pattern."""
    try:
        path = Path(directory).expanduser().resolve()
        if not path.is_dir():
            return f"Error: '{directory}' is not a directory."
        
        result = "# Find Results\n\n"
        
        # Use find command to search for files
        try:
            output = subprocess.check_output(
                ["find", str(path), "-name", pattern],
                stderr=subprocess.STDOUT,
                text=True
            )
            
            if not output.strip():
                return f"No files matching '{pattern}' found in {path}."
            
            result += f"Files matching '{pattern}':\n\n"
            for line in output.strip().split("\n"):
                file_path = line.strip()
                result += f"- `{file_path}`\n"
        
        except subprocess.CalledProcessError as e:
            result += f"Error during find: {e.output}\n"
        
        return result
    
    except Exception as e:
        return f"Error finding files: {str(e)}"

@register_tool("exec")
def exec_tool(command: str) -> str:
    """Execute a command and return its output. Use with caution."""
    try:
        result = "# Command Execution\n\n"
        result += f"Executing: `{command}`\n\n"
        
        # Execute the command and capture output
        process = subprocess.run(
            command,
            shell=True,  # Use shell to support pipes and redirects
            text=True,
            capture_output=True,
            timeout=10  # Timeout to prevent hanging
        )
        
        if process.stdout:
            result += "```\n" + process.stdout + "\n```\n"
        
        if process.stderr:
            result += "**Error output:**\n\n"
            result += "```\n" + process.stderr + "\n```\n"
        
        if process.returncode != 0:
            result += f"\n**Command exited with code {process.returncode}**\n"
        
        return result
    
    except subprocess.TimeoutExpired:
        return "Command timed out after 10 seconds.\n"
    except Exception as e:
        return f"Error executing command: {str(e)}"

@register_tool("edit")
def edit_tool(file_path: str) -> str:
    """Open file in a text editor."""
    try:
        path = Path(file_path).expanduser().resolve()
        
        if not path.exists():
            # Create a new file if it doesn't exist
            path.touch()
        
        # Determine editor to use
        editor = os.environ.get("EDITOR", "nano")
        
        try:
            # Open the editor
            result = "Opening editor... Press Ctrl+X to exit nano, or use your editor's exit command.\n"
            subprocess.run([editor, str(path)])
            result += f"\nFile '{file_path}' has been edited."
            return result
        except subprocess.CalledProcessError as e:
            return f"Error opening editor: {e.output}"
    
    except Exception as e:
        return f"Error editing file: {str(e)}"

@register_tool("wget")
def wget_tool(url: str, output_file: Optional[str] = None) -> str:
    """Download a file from the web."""
    try:
        import urllib.request
        import urllib.error
        
        result = "# File Download\n\n"
        
        # Determine output filename if not provided
        if not output_file:
            output_file = url.split("/")[-1]
            if not output_file or output_file.endswith("/"):
                output_file = "downloaded_file"
        
        # Download the file
        try:
            urllib.request.urlretrieve(url, output_file)
            output_path = Path(output_file).resolve()
            result += f"Downloaded {url} to `{output_path}`\n"
            
            # Get file size
            size_bytes = output_path.stat().st_size
            if size_bytes < 1024:
                size_str = f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes/1024:.1f} KB"
            else:
                size_str = f"{size_bytes/(1024*1024):.1f} MB"
            
            result += f"File size: {size_str}\n"
            
            return result
        
        except urllib.error.URLError as e:
            return f"Error downloading file: {str(e)}"
    
    except Exception as e:
        return f"Error downloading file: {str(e)}"
