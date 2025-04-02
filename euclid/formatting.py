"""Module for terminal formatting and UI elements."""

import io
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from rich.console import Console, ConsoleOptions, RenderResult, RenderGroup
from rich.highlighter import ReprHighlighter
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.box import ROUNDED

from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.html import HtmlLexer
from pygments.lexers.shell import BashLexer

try:
    import terminal_image
    TERMINAL_IMAGE_AVAILABLE = True
except ImportError:
    TERMINAL_IMAGE_AVAILABLE = False

# Custom theme
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "error": "bold red",
    "user": "bold blue",
    "assistant": "bold green",
    "system": "bold yellow",
    "tool": "bold magenta",
    "function_name": "bold cyan",
    "parameter": "bold yellow",
    "progress.spinner": "cyan",
    "progress.description": "dim cyan",
})

console = Console(theme=custom_theme, highlight=True)

# Common code file extensions and their lexers
LEXER_MAPPING = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".md": "markdown",
    ".sh": "bash",
    ".bash": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".c": "c",
    ".cpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".sql": "sql",
}

class EnhancedMarkdown(Markdown):
    """Enhanced Markdown renderer with code block syntax highlighting and image support."""
    
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.highlighter = ReprHighlighter()
    
    def _render_code_block(self, code: str, lexer_name: Optional[str] = None) -> Syntax:
        """Render a code block with proper syntax highlighting."""
        try:
            if lexer_name is None or lexer_name == "":
                # Try to guess the lexer
                try:
                    lexer = guess_lexer(code)
                    lexer_name = lexer.name.lower()
                except Exception:
                    lexer_name = "text"
            
            return Syntax(code, lexer_name, theme="monokai", line_numbers=True, word_wrap=True)
        except Exception:
            # Fallback to plain text if lexer fails
            return Syntax(code, "text", theme="monokai", line_numbers=True, word_wrap=True)
    
    def _render_image(self, path_or_url: str, alt_text: str = "") -> Union[Text, RenderGroup]:
        """Render an image if possible, otherwise return a text representation."""
        if not TERMINAL_IMAGE_AVAILABLE:
            return Text(f"[Image: {alt_text or path_or_url}]")
        
        # Check if it's a file path and if it exists
        if os.path.exists(path_or_url):
            try:
                img_path = Path(path_or_url)
                if img_path.suffix.lower() in self.IMAGE_EXTS:
                    # Render the image
                    try:
                        return terminal_image.TerminalImage(img_path)
                    except Exception:
                        return Text(f"[Image: {alt_text or path_or_url}]")
            except Exception:
                pass
        
        return Text(f"[Image: {alt_text or path_or_url}]")

def format_user_message(message: str) -> str:
    """Format a user message for display."""
    return f"[user]{message}[/user]"

def format_assistant_message(message: str) -> str:
    """Format an assistant message for display."""
    return EnhancedMarkdown(message)

def format_system_message(message: str) -> str:
    """Format a system message for display."""
    return f"[system]{message}[/system]"

def format_tool_message(tool_name: str, content: str) -> str:
    """Format a tool message for display."""
    return f"[tool]{tool_name}:[/tool] {content}"

def format_function_call(function_name: str, parameters: Dict[str, Any]) -> Panel:
    """Format a function call for display."""
    title = f"[function_name]{function_name}[/function_name]"
    table = Table(show_header=False, box=ROUNDED, padding=0)
    table.add_column("Parameter", style="parameter")
    table.add_column("Value")
    
    for param, value in parameters.items():
        table.add_row(param, repr(value))
    
    return Panel(table, title=title, title_align="left", border_style="function_name")

def create_spinner(description: str = "Working", color: str = "cyan") -> Progress:
    """Create a spinner for showing progress."""
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[progress.description]{description}[/progress.description]"),
        transient=True
    )

def multi_line_prompt(prompt_text: str, completer: Optional[WordCompleter] = None) -> str:
    """Show a multi-line input prompt."""
    console.print(prompt_text, end="")
    
    # Get console dimensions
    width = console.width
    
    # Create a custom style
    style = Style.from_dict({
        'prompt': 'bold blue',
        # You can add more styles here
    })
    
    user_input = pt_prompt(
        "\n", 
        multiline=True,
        lexer=PygmentsLexer(PythonLexer),
        completer=completer,
        style=style,
        prompt_continuation=lambda width, line_number, is_soft_wrap: "... "
    )
    
    return user_input

def display_thinking(thinking_text: str) -> None:
    """Display the assistant's thinking process."""
    panel = Panel(
        EnhancedMarkdown(thinking_text),
        title="Thinking",
        title_align="left",
        border_style="dim cyan",
        padding=(1, 2)
    )
    console.print(panel)

def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')
