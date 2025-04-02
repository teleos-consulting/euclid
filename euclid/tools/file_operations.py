"""File operations tools similar to Claude Code's file tools."""

import os
import re
import subprocess
from pathlib import Path
import glob as glob_module
from typing import List, Optional, Dict, Any, Union

from euclid.functions import register_function
from euclid.formatting import console, EnhancedMarkdown

@register_function(
    name="View",
    description="Reads a file from the local filesystem. The file_path parameter must be an absolute path, not a relative path."
)
def view_file(file_path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> str:
    """Read a file from the filesystem with optional offset and limit.
    
    Args:
        file_path: The absolute path to the file to read.
        offset: The line number to start reading from (0-indexed).
        limit: The number of lines to read.
        
    Returns:
        The contents of the file, possibly truncated to the given offset and limit.
    """
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            return f"Error: File '{file_path}' does not exist or is not a file."
        
        # Handle binary files (images)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
            return f"Image file: {file_path}\n\n![Image]({file_path})"
        
        # Read file with line numbers, respecting offset and limit
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        # Apply offset and limit if provided
        if offset is not None:
            lines = lines[offset:]
        if limit is not None:
            lines = lines[:limit]
        
        # Format with line numbers (starting from offset + 1 if offset is provided)
        start_line = 1 if offset is None else offset + 1
        numbered_lines = [f"{i+start_line:4d} | {line}" for i, line in enumerate(lines)]
        
        # Determine what syntax highlighting to use based on file extension
        file_ext = path.suffix.lower()
        if file_ext in {".py", ".js", ".ts", ".html", ".css", ".json", ".md"}:
            return f"```{file_ext[1:]}\n{''.join(numbered_lines)}\n```"
        else:
            return f"```\n{''.join(numbered_lines)}\n```"
    
    except Exception as e:
        return f"Error reading file: {str(e)}"

@register_function(
    name="GlobTool",
    description="Fast file pattern matching tool that works with any codebase size. Supports glob patterns like '**/*.js' or 'src/**/*.ts'. Returns matching file paths sorted by modification time."
)
def glob_tool(pattern: str, path: Optional[str] = ".") -> str:
    """Find files matching a glob pattern.
    
    Args:
        pattern: The glob pattern to match files against.
        path: The directory to search in. Defaults to the current working directory.
        
    Returns:
        A list of matching file paths.
    """
    try:
        base_path = Path(path).expanduser().resolve()
        if not base_path.is_dir():
            return f"Error: '{path}' is not a directory."
        
        # Use glob to find matching files
        matches = list(base_path.glob(pattern))
        
        # Sort by modification time, newest first
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not matches:
            return f"No files matching '{pattern}' found in {path}."
        
        result = f"# Files matching '{pattern}' in {path}\n\n"
        for i, file_path in enumerate(matches):
            result += f"{i+1}. `{file_path}`\n"
        
        return result
    
    except Exception as e:
        return f"Error during glob search: {str(e)}"

@register_function(
    name="GrepTool",
    description="Fast content search tool that works with any codebase size. Searches file contents using regular expressions. Filter files by pattern with the include parameter."
)
def grep_tool(pattern: str, path: Optional[str] = ".", include: Optional[str] = None) -> str:
    """Search for files containing a pattern.
    
    Args:
        pattern: The regular expression pattern to search for in file contents.
        path: The directory to search in. Defaults to the current working directory.
        include: File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}").
        
    Returns:
        A list of matching file paths and snippets.
    """
    try:
        base_path = Path(path).expanduser().resolve()
        if not base_path.is_dir():
            return f"Error: '{path}' is not a directory."
        
        # Prepare grep command
        cmd = ["grep", "-r", "-n", "-I", pattern]
        
        # Add include pattern if provided
        if include:
            cmd.extend(["--include", include])
        
        # Add path to search
        cmd.append(str(base_path))
        
        # Execute grep command
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:  # grep returns 1 when no matches
                return f"No files containing '{pattern}' found in {path}."
            else:
                return f"Error during grep: {e.output}"
        
        if not output.strip():
            return f"No files containing '{pattern}' found in {path}."
        
        # Process the output for a cleaner display
        result = f"# Files containing '{pattern}' in {path}\n\n"
        
        current_file = None
        file_matches = []
        
        for line in output.strip().split("\n"):
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            
            file_path, line_num, content = parts
            
            # If we've moved to a new file
            if file_path != current_file:
                current_file = file_path
                file_matches.append({"file": file_path, "matches": []})
            
            # Add this match
            file_matches[-1]["matches"].append({
                "line": int(line_num),
                "content": content.strip()
            })
        
        # Format the results
        for file_match in file_matches:
            file_path = file_match["file"]
            result += f"## `{file_path}`\n\n"
            
            for match in file_match["matches"][:5]:  # Limit to first 5 matches per file
                result += f"Line {match['line']}: `{match['content']}`\n"
            
            if len(file_match["matches"]) > 5:
                result += f"...and {len(file_match['matches']) - 5} more matches\n"
            
            result += "\n"
        
        return result
    
    except Exception as e:
        return f"Error during grep search: {str(e)}"

@register_function(
    name="LS",
    description="Lists files and directories in a given path. The path parameter must be an absolute path, not a relative path."
)
def ls_tool(path: str, ignore: Optional[List[str]] = None) -> str:
    """List files and directories in a path.
    
    Args:
        path: The absolute path to the directory to list.
        ignore: List of glob patterns to ignore.
        
    Returns:
        A formatted listing of files and directories.
    """
    try:
        base_path = Path(path).expanduser().resolve()
        if not base_path.is_dir():
            return f"Error: '{path}' is not a directory."
        
        # Get all files and directories
        entries = sorted(base_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        
        # Filter out ignored patterns if provided
        if ignore:
            filtered_entries = []
            for entry in entries:
                skip = False
                for pattern in ignore:
                    if glob_module.fnmatch.fnmatch(entry.name, pattern):
                        skip = True
                        break
                if not skip:
                    filtered_entries.append(entry)
            entries = filtered_entries
        
        if not entries:
            return f"No files or directories found in {path} (after applying ignore patterns)."
        
        result = f"# Contents of {path}\n\n"
        
        # Group by directories and files
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        
        if dirs:
            result += "## Directories\n\n"
            for dir_path in dirs:
                result += f"📁 `{dir_path.name}/`\n"
            result += "\n"
        
        if files:
            result += "## Files\n\n"
            for file_path in files:
                # Add file size
                size_bytes = file_path.stat().st_size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} bytes"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes/1024:.1f} KB"
                else:
                    size_str = f"{size_bytes/(1024*1024):.1f} MB"
                
                result += f"📄 `{file_path.name}` ({size_str})\n"
        
        return result
    
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@register_function(
    name="Edit",
    description="Edit the contents of a file. For new files, provide an empty old_string. For modifications, the old_string must exactly match a portion of the file content, including whitespace."
)
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing old_string with new_string.
    
    Args:
        file_path: The absolute path to the file to modify.
        old_string: The text to replace. Empty string for new files.
        new_string: The text to replace it with.
        
    Returns:
        A success message or error message.
    """
    try:
        path = Path(file_path).expanduser().resolve()
        
        # Creating a new file
        if not path.exists() and old_string == "":
            # Ensure the parent directory exists
            if not path.parent.exists():
                return f"Error: Parent directory for '{file_path}' does not exist."
            
            # Create the file with the new content
            path.write_text(new_string)
            return f"File created successfully at: {file_path}"
        
        # Modifying an existing file
        if not path.is_file():
            return f"Error: '{file_path}' does not exist or is not a file."
        
        # Read the current content
        current_content = path.read_text()
        
        # For existing files with empty old_string, it means overwrite the entire file
        if old_string == "":
            path.write_text(new_string)
            return f"File '{file_path}' completely overwritten."
        
        # Count occurrences of old_string
        occurrences = current_content.count(old_string)
        
        if occurrences == 0:
            return f"Error: The specified text to replace was not found in '{file_path}'."
        
        if occurrences > 1:
            return f"Error: The specified text to replace appears {occurrences} times in '{file_path}'. It must be unique."
        
        # Replace the text
        new_content = current_content.replace(old_string, new_string, 1)
        path.write_text(new_content)
        
        return f"The file '{file_path}' has been updated. Here's the result of running `cat -n` on a snippet of the edited file:\n" + view_file(file_path, max(0, new_content.find(new_string) - 50), 10)
    
    except Exception as e:
        return f"Error editing file: {str(e)}"

@register_function(
    name="Replace",
    description="Write a file to the local filesystem. Overwrites the existing file if there is one."
)
def replace_file(file_path: str, content: str) -> str:
    """Write content to a file, overwriting any existing content.
    
    Args:
        file_path: The absolute path to the file to write.
        content: The content to write to the file.
        
    Returns:
        A success message or error message.
    """
    try:
        path = Path(file_path).expanduser().resolve()
        
        # Ensure the parent directory exists
        if not path.parent.exists():
            return f"Error: Parent directory for '{file_path}' does not exist."
        
        # Write the content to the file
        path.write_text(content)
        
        return f"File {'created' if not path.exists() else 'updated'} successfully at: {file_path}"
    
    except Exception as e:
        return f"Error writing file: {str(e)}"
