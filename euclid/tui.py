"""Terminal User Interface for Euclid."""

import sys
import os
import threading
import time
import uuid
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path

from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window, FormattedTextControl
from prompt_toolkit.layout.containers import WindowAlign, Float, FloatContainer, ConditionalContainer
from prompt_toolkit.layout.dimension import D
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.widgets import TextArea, Box, Frame, Label, Button
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.box import ROUNDED
from rich.console import RenderableType
from rich.spinner import Spinner
from rich.live import Live

from euclid.config import config
from euclid.ollama import OllamaClient, Message
from euclid.conversation import Conversation
from euclid.models import ModelRegistry
from euclid.functions import (
    process_text_with_function_calls, 
    get_available_functions,
    parse_function_calls
)
from euclid.formatting import console as rich_console
from euclid.tools.registry import get_available_tools, run_tool


# Styling
STYLE = Style.from_dict({
    "status": "reverse",
    "status.position": "#ffffff",
    "status.key": "#ffaa00",
    "status.bar": "#00aa00",
    "message_input": "#00bbbb",
    "message_output": "#cccccc",
    "sidebar": "#003333",
    "sidebar.title": "#ffffff bold",
    "sidebar.item": "#aaaaaa",
    "sidebar.selected": "#ffffff",
    "function_name": "#00ffff bold",
    "thinking": "#888888 italic",
    "button": "#000000 on #888888",
    "button.focused": "#ffffff on #008888",
    "frame.border": "#ffffff",
    "frame.title": "#ffff00",
})

# Keybindings
kb = KeyBindings()


class Message:
    """A chat message."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class EuclidTUI:
    """Terminal UI for Euclid."""
    
    def __init__(
        self, 
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        conversation_id: Optional[str] = None,
        show_thinking: bool = True
    ):
        """Initialize the TUI.
        
        Args:
            model: Model to use.
            system_prompt: System prompt to use.
            conversation_id: Conversation ID to continue.
            show_thinking: Whether to show thinking process.
        """
        self.model = model or config.ollama_model
        self.system_prompt = system_prompt
        self.show_thinking = show_thinking
        
        # Set up client
        self.client = OllamaClient(model=self.model)
        
        # Create or load conversation
        self.conv_id = conversation_id or str(uuid.uuid4())
        self.conversation = Conversation.load(self.conv_id)
        
        # Add system prompt if none exists
        if not self.conversation.messages:
            default_system_file = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
            if default_system_file.exists():
                try:
                    system_content = default_system_file.read_text()
                except Exception:
                    system_content = "You are a helpful assistant."
            else:
                system_content = "You are a helpful assistant."
            
            self.conversation.add_message("system", system_content)
        
        # Initialize session state
        self.messages: List[Message] = []
        self.thinking_text = ""
        self.loading = False
        self.function_calls_active = False
        self.available_models = []
        self.loaded_models = False
        self.available_tools = get_available_tools()
        
        # Convert conversation to messages
        for msg in self.conversation.messages:
            if msg.role != "system":  # Skip system message
                self.messages.append(Message(msg.role, msg.content))
        
        # Create UI components
        self.create_layout()
        
        # Function calls components
        self.function_output = ""
        self.function_name = ""
        self.function_params = {}
        
        # Create application
        self.app = Application(
            layout=self.layout,
            key_bindings=self.create_keybindings(),
            style=STYLE,
            mouse_support=True,
            full_screen=True
        )
        
        # Tool completions
        self.tool_completer = WordCompleter(
            self.available_tools,
            meta_dict={tool: f"Tool: {tool}" for tool in self.available_tools},
            ignore_case=True
        )
    
    def create_layout(self):
        """Create the UI layout."""
        # Create message input area
        self.input_field = TextArea(
            prompt="You: ",
            multiline=True,
            wrap_lines=True,
            style="class:message_input",
            height=D(min=3, max=10),
            completer=self.tool_completer
        )
        
        # Create chat history area
        self.chat_window = TextArea(
            style="class:message_output",
            read_only=True,
            scrollbar=True,
            wrap_lines=True
        )
        
        # Create thinking area
        self.thinking_window = TextArea(
            style="class:thinking",
            read_only=True,
            scrollbar=True,
            wrap_lines=True,
            height=D(min=1, max=10)
        )
        
        # Function call output window
        self.function_window = TextArea(
            style="class:function_name",
            read_only=True,
            scrollbar=True,
            wrap_lines=True,
            height=D(min=1, max=10)
        )
        
        # Status bar
        self.status_bar = Window(
            FormattedTextControl(self.get_status_text),
            height=1,
            style="class:status"
        )
        
        # Create sidebar for model selection
        self.sidebar = self.create_sidebar()
        
        # Main chat area
        self.main_container = HSplit([
            # Chat history takes most of the space
            self.chat_window,
            
            # Thinking window (conditional)
            ConditionalContainer(
                content=Frame(
                    self.thinking_window,
                    title="Thinking",
                    style="class:thinking"
                ),
                filter=Condition(lambda: self.show_thinking and self.thinking_text.strip())
            ),
            
            # Function call window (conditional)
            ConditionalContainer(
                content=Frame(
                    self.function_window,
                    title="Function Calls",
                    style="class:function_name"
                ),
                filter=Condition(lambda: self.function_calls_active)
            ),
            
            # Input area at the bottom
            Frame(self.input_field, title="Message", height=D(min=3, max=10)),
            
            # Status bar
            self.status_bar
        ])
        
        # Combine sidebar and main area
        self.layout = Layout(
            VSplit([
                # Left sidebar (20% width)
                self.sidebar,
                # Main chat area (80% width)
                self.main_container
            ])
        )
    
    def create_sidebar(self):
        """Create the sidebar with model information and controls."""
        # Current model label
        self.model_label = Label(f"Current Model: {self.model}")
        
        # Button to load models
        self.load_models_button = Button(
            "Load Models", 
            handler=self.load_models
        )
        
        # Model list (initially empty)
        self.model_list = TextArea(
            "Loading models...",
            read_only=True,
            scrollbar=True,
            style="class:sidebar.item"
        )
        
        # Create the sidebar container
        return Frame(
            HSplit([
                # Header
                Label("Euclid", style="class:sidebar.title"),
                Label("-" * 20, style="class:sidebar.title"),
                
                # Model info
                self.model_label,
                self.load_models_button,
                self.model_list,
                
                # Help text
                Label("-" * 20, style="class:sidebar"),
                Label("Commands:", style="class:sidebar"),
                Label("- /help: Show help", style="class:sidebar"),
                Label("- /model <name>: Switch model", style="class:sidebar"),
                Label("- Ctrl+Q: Quit", style="class:sidebar"),
                Label("- Ctrl+Enter: Send message", style="class:sidebar")
            ]),
            title="Euclid Controls",
            style="class:sidebar",
            width=D(min=20, max=30)
        )
    
    def get_status_text(self):
        """Get the current status text."""
        return [
            ("class:status.bar", " Euclid "),
            ("class:status.position", f" Model: {self.model} "),
            ("class:status.position", f" Conversation: {self.conv_id[:8]} "),
            ("class:status.key", " Ctrl+Q "),
            ("class:status", "Quit "),
            ("class:status.key", " Ctrl+Enter "),
            ("class:status", "Send "),
        ]
    
    def create_keybindings(self):
        """Create keyboard bindings."""
        kb = KeyBindings()
        
        @kb.add("c-q")
        def _(event):
            event.app.exit()
        
        @kb.add("c-enter")
        def _(event):
            self.send_message()
        
        return kb
    
    def send_message(self):
        """Send the current message."""
        # Get message from input field
        user_input = self.input_field.text
        
        if not user_input.strip():
            return
        
        # Clear input field
        self.input_field.text = ""
        
        # Add message to chat
        self.add_message("user", user_input)
        
        # Check for special commands
        if user_input.startswith("/"):
            self.handle_command(user_input)
            return
        
        # Process message in background thread
        self.loading = True
        thread = threading.Thread(target=self.process_message, args=(user_input,))
        thread.daemon = True
        thread.start()
    
    def process_message(self, user_input: str):
        """Process a user message in the background."""
        # Add user message to conversation
        self.conversation.add_message("user", user_input)
        
        # Reset thinking text
        self.thinking_text = ""
        self.function_calls_active = False
        
        try:
            # Stream the response
            chunks = []
            for chunk in self.client.chat_completion(self.conversation.messages):
                chunks.append(chunk)
                self.thinking_text += chunk
                self.update_thinking_window()
                time.sleep(0.01)  # Small delay for smoother updates
            
            # Process the full response
            response_text = "".join(chunks)
            
            # Check for function calls
            if "<function_calls>" in response_text:
                self.function_calls_active = True
                self.function_window.text = "Processing function calls..."
                
                # Process function calls in separate thread to avoid blocking
                process_thread = threading.Thread(
                    target=self.process_function_calls,
                    args=(response_text,)
                )
                process_thread.daemon = True
                process_thread.start()
            else:
                # Add normal response to chat
                self.conversation.add_message("assistant", response_text)
                self.add_message("assistant", response_text)
                self.conversation.save()
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.add_message("assistant", error_msg)
        
        finally:
            self.loading = False
    
    def process_function_calls(self, response_text: str):
        """Process function calls in the assistant's response."""
        try:
            # Extract function calls
            calls = parse_function_calls(response_text)
            if not calls:
                self.function_window.text = "No valid function calls found."
                return
            
            # Display function call info
            call_info = ""
            for call in calls:
                func_name = call["function"]
                params = call["parameters"]
                
                call_info += f"Function: {func_name}\n"
                for param, value in params.items():
                    call_info += f"- {param}: {value}\n"
                call_info += "\n"
            
            self.function_window.text = call_info
            
            # Process the function calls
            result_text = process_text_with_function_calls(response_text)
            
            # Add processed response to chat
            self.conversation.add_message("assistant", result_text)
            self.add_message("assistant", result_text)
            self.conversation.save()
            
            # Clear function call state
            self.function_calls_active = False
        
        except Exception as e:
            error_msg = f"Error processing function calls: {str(e)}"
            self.add_message("assistant", error_msg)
            self.function_calls_active = False
    
    def handle_command(self, command: str):
        """Handle special commands."""
        parts = command.split()
        cmd = parts[0][1:]  # Remove the leading /
        
        if cmd == "help":
            help_text = (
                "Available commands:\n"
                "/help - Show this help\n"
                "/model <name> - Switch to a different model\n"
                "/models - List available models\n"
                "/exit or /quit - Exit the application\n"
                "\nAvailable tools:\n"
            )
            
            for tool in sorted(self.available_tools):
                help_text += f"/{tool} - Run the {tool} tool\n"
            
            self.add_message("assistant", help_text)
        
        elif cmd == "model" and len(parts) > 1:
            new_model = parts[1]
            self.switch_model(new_model)
        
        elif cmd == "models":
            self.load_models()
            models_text = "Available models:\n"
            for model in self.available_models:
                models_text += f"- {model}\n"
            self.add_message("assistant", models_text)
        
        elif cmd == "exit" or cmd == "quit":
            self.app.exit()
        
        elif cmd in self.available_tools:
            # Run the tool
            tool_result = run_tool(cmd, command)
            self.add_message("assistant", tool_result)
            
            # Add to conversation
            self.conversation.add_message("assistant", tool_result)
            self.conversation.save()
        
        else:
            self.add_message("assistant", f"Unknown command: {cmd}")
    
    def switch_model(self, new_model: str):
        """Switch to a different model."""
        # Check if model is available
        if not self.loaded_models:
            self.load_models()
        
        if new_model in self.available_models:
            self.model = new_model
            self.client = OllamaClient(model=new_model)
            self.model_label.text = f"Current Model: {new_model}"
            self.add_message("assistant", f"Switched to model: {new_model}")
        else:
            # Prompt to pull model
            self.add_message("assistant", f"Model '{new_model}' not available. Would you like to pull it? Use /pull {new_model}")
    
    def load_models(self):
        """Load available models."""
        self.model_list.text = "Loading models..."
        
        # Run in background thread
        thread = threading.Thread(target=self._load_models_thread)
        thread.daemon = True
        thread.start()
    
    def _load_models_thread(self):
        """Background thread to load models."""
        registry = ModelRegistry()
        models = registry.get_available_models()
        
        self.available_models = [m.name for m in models]
        
        models_text = "Available models:\n"
        for model in self.available_models:
            models_text += f"- {model}\n"
        
        self.model_list.text = models_text
        self.loaded_models = True
    
    def add_message(self, role: str, content: str):
        """Add a message to the chat history."""
        msg = Message(role, content)
        self.messages.append(msg)
        self.update_chat_window()
    
    def update_chat_window(self):
        """Update the chat window with current messages."""
        chat_text = ""
        
        for msg in self.messages:
            if msg.role == "user":
                chat_text += f"\n[You]:\n{msg.content}\n"
            else:
                chat_text += f"\n[Assistant]:\n{msg.content}\n"
        
        self.chat_window.text = chat_text
        
        # Scroll to bottom
        self.chat_window.buffer.cursor_position = len(self.chat_window.buffer.text)
    
    def update_thinking_window(self):
        """Update the thinking window."""
        if self.show_thinking:
            self.thinking_window.text = self.thinking_text
    
    def run(self):
        """Run the application."""
        self.app.run()
        
        # Save conversation on exit
        self.conversation.save()
        print(f"Conversation saved with ID: {self.conv_id}")


def run_tui(
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    conversation_id: Optional[str] = None,
    show_thinking: bool = True
):
    """Run the Terminal UI.
    
    Args:
        model: Model to use.
        system_prompt: System prompt to use.
        conversation_id: Conversation ID to continue.
        show_thinking: Whether to show thinking process.
    """
    tui = EuclidTUI(
        model=model,
        system_prompt=system_prompt,
        conversation_id=conversation_id,
        show_thinking=show_thinking
    )
    tui.run()


if __name__ == "__main__":
    run_tui()