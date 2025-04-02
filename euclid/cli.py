"""CLI interface for Euclid."""

import sys
import uuid
import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.box import ROUNDED
from rich.live import Live
from rich.prompt import Prompt

from euclid.config import config
from euclid.ollama import OllamaClient, Message
from euclid.conversation import Conversation
from euclid.functions import (
    process_text_with_function_calls, 
    get_available_functions,
    parse_function_calls
)
from euclid.formatting import (
    console, 
    format_user_message, 
    format_assistant_message, 
    format_system_message,
    multi_line_prompt,
    create_spinner,
    EnhancedMarkdown,
    clear_screen,
    display_thinking
)
from euclid.tools.registry import get_available_tools, run_tool

# Import all function modules to register them
from euclid.tools import file_operations

app = typer.Typer(help="A CLI tool for interacting with local Ollama models")

@app.command("chat")
def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    system_prompt: Optional[str] = typer.Option(
        None, "--system", "-s", 
        help="System prompt to use"
    ),
    system_file: Optional[str] = typer.Option(
        None, "--system-file", "-sf",
        help="File containing the system prompt"
    ),
    conversation_id: Optional[str] = typer.Option(
        None, "--conversation", "-c",
        help="Conversation ID to continue"
    ),
    show_thinking: bool = typer.Option(
        False, "--thinking", "-t",
        help="Show model's thinking process"
    ),
    function_calling: bool = typer.Option(
        True, "--functions/--no-functions",
        help="Enable or disable function calling"
    ),
):
    """Start an interactive chat session."""
    # Clear screen for a clean start
    clear_screen()
    
    # Create or load conversation
    conv_id = conversation_id or str(uuid.uuid4())
    conversation = Conversation.load(conv_id)
    
    # Set up client
    client = OllamaClient(model=model)
    
    # Get system prompt
    if system_file:
        try:
            with open(system_file, "r") as f:
                system_content = f.read()
        except Exception as e:
            console.print(f"[error]Error reading system prompt file: {str(e)}[/error]")
            console.print("Using default system prompt instead.")
            system_content = "You are a helpful assistant."
    elif system_prompt:
        system_content = system_prompt
    else:
        # Use default system prompt
        default_system_file = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
        if default_system_file.exists():
            try:
                system_content = default_system_file.read_text()
            except Exception:
                system_content = "You are a helpful assistant."
        else:
            system_content = "You are a helpful assistant."
    
    # Add system prompt if no messages exist
    if not conversation.messages:
        conversation.add_message("system", system_content)
    
    # Load function schemas for the system prompt
    if function_calling and not any(msg.content.find("function") >= 0 for msg in conversation.messages if msg.role == "system"):
        # Add function schema information to the system prompt
        functions_schema = get_available_functions()
        if functions_schema:
            function_desc = "You have access to the following functions:\n\n"
            function_desc += json.dumps({"functions": list(functions_schema.values())}, indent=2)
            function_desc += "\n\nTo use these functions, use the following format in your response:\n"
            function_desc += """
<function_calls>
<invoke name="FunctionName">
<parameter name="param1">value1</parameter>
<parameter name="param2">value2</parameter>
</invoke>
</function_calls>
"""
            # Add to existing system message or create new one
            if conversation.messages and conversation.messages[0].role == "system":
                conversation.messages[0].content += "\n\n" + function_desc
            else:
                conversation.add_message("system", system_content + "\n\n" + function_desc)
    
    # Display header
    console.print(Panel(
        "[bold green]Euclid Chat[/bold green] 🌿",
        subtitle=f"Using model: [bold]{client.model}[/bold]",
        box=ROUNDED,
        border_style="green"
    ))
    console.print("Type 'exit' or press Ctrl+D to exit. Type '/help' for available commands.\n")
    
    # Main chat loop
    try:
        while True:
            # Get user input
            user_input = multi_line_prompt("[bold blue]You:[/bold blue]")
            
            if user_input.lower() in ("exit", "quit", "bye"):
                break
            
            # Display user message
            console.print(format_user_message(user_input))
            
            # Add user message to conversation
            conversation.add_message("user", user_input)
            
            # Check for tool use with slash command
            tool_name = None
            if user_input.startswith("/"):
                tool_name = user_input.split()[0][1:]  # Extract tool name without slash
                
            if tool_name and tool_name in get_available_tools():
                # Run tool and get result
                tool_result = run_tool(tool_name, user_input)
                
                # Add tool result as assistant message
                conversation.add_message("assistant", tool_result)
                
                # Display result
                console.print("[bold green]Assistant:[/bold green]")
                console.print(EnhancedMarkdown(tool_result))
                
                # Save conversation and continue
                conversation.save()
                continue
            
            # Generate response
            console.print("[bold green]Assistant:[/bold green]")
            
            response_text = ""
            thinking_text = ""
            
            # Display spinner while generating response
            with create_spinner("Thinking") as progress:
                try:
                    # Stream the response
                    chunks = []
                    for chunk in client.chat_completion(conversation.messages):
                        chunks.append(chunk)
                        thinking_text += chunk
                except Exception as e:
                    console.print(f"[error]Error generating response: {str(e)}[/error]")
                    continue
            
            # Process the full response for function calls
            response_text = "".join(chunks)
            
            # Show thinking if enabled
            if show_thinking:
                display_thinking(thinking_text)
            
            # Check for function calls in the response
            if function_calling and "<function_calls>" in response_text:
                try:
                    # Process function calls
                    result_text = process_text_with_function_calls(response_text)
                    console.print(EnhancedMarkdown(result_text))
                    
                    # Add the assistant's final response to the conversation
                    conversation.add_message("assistant", result_text)
                except Exception as e:
                    console.print(f"[error]Error processing function calls: {str(e)}[/error]")
                    # Fall back to displaying the raw response
                    console.print(EnhancedMarkdown(response_text))
                    conversation.add_message("assistant", response_text)
            else:
                # Display the response without function processing
                console.print(EnhancedMarkdown(response_text))
                
                # Add assistant response to conversation
                conversation.add_message("assistant", response_text)
            
            # Save conversation after each exchange
            conversation.save()
    
    except (KeyboardInterrupt, EOFError):
        console.print("\nExiting chat.")
    
    console.print(f"\nConversation saved with ID: {conv_id}")

@app.command("run")
def run(
    prompt: str = typer.Argument(..., help="Prompt to send to the model"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    system_prompt: Optional[str] = typer.Option(
        None, "--system", "-s", 
        help="System prompt to use"
    ),
    no_stream: bool = typer.Option(
        False, "--no-stream", help="Disable streaming for response"
    ),
    function_calling: bool = typer.Option(
        True, "--functions/--no-functions",
        help="Enable or disable function calling"
    ),
    show_thinking: bool = typer.Option(
        False, "--thinking", "-t",
        help="Show model's thinking process"
    ),
):
    """Run a single prompt and get a response."""
    # Set up client
    client = OllamaClient(model=model)
    
    # Create messages
    messages = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    else:
        # Use default system prompt
        default_system_file = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
        if default_system_file.exists():
            try:
                system_content = default_system_file.read_text()
                messages.append(Message(role="system", content=system_content))
            except Exception:
                pass
    
    # Add function schema information to the system prompt if not already there
    if function_calling and (not messages or "function" not in messages[0].content):
        functions_schema = get_available_functions()
        if functions_schema:
            function_desc = "You have access to the following functions:\n\n"
            function_desc += json.dumps({"functions": list(functions_schema.values())}, indent=2)
            function_desc += "\n\nTo use these functions, use the following format in your response:\n"
            function_desc += """
<function_calls>
<invoke name="FunctionName">
<parameter name="param1">value1</parameter>
<parameter name="param2">value2</parameter>
</invoke>
</function_calls>
"""
            # Add to existing system message or create new one
            if messages and messages[0].role == "system":
                messages[0].content += "\n\n" + function_desc
            else:
                messages.insert(0, Message(role="system", function_desc))
    
    messages.append(Message(role="user", content=prompt))
    
    # Check for tool use with slash command
    if prompt.startswith("/"):
        tool_name = prompt.split()[0][1:]  # Extract tool name without slash
        if tool_name in get_available_tools():
            # Run tool and get result
            tool_result = run_tool(tool_name, prompt)
            
            # Display result
            console.print(EnhancedMarkdown(tool_result))
            return
    
    # Generate response
    thinking_chunks = []
    if not no_stream:
        with create_spinner("Thinking") as progress:
            # Collect all chunks first
            chunks = []
            for chunk in client.chat_completion(messages):
                chunks.append(chunk)
                thinking_chunks.append(chunk)
            
            response_text = "".join(chunks)
            
            # Show thinking if requested
            if show_thinking:
                display_thinking("".join(thinking_chunks))
            
            # Process function calls if present
            if function_calling and "<function_calls>" in response_text:
                try:
                    result_text = process_text_with_function_calls(response_text)
                    console.print(EnhancedMarkdown(result_text))
                except Exception as e:
                    console.print(f"[error]Error processing function calls: {str(e)}[/error]")
                    console.print(EnhancedMarkdown(response_text))
            else:
                console.print(EnhancedMarkdown(response_text))
    else:
        with create_spinner("Generating response"):
            response = client.chat_completion(messages, stream=False)
        
        if "message" in response and "content" in response["message"]:
            response_text = response["message"]["content"]
            
            # Process function calls if present
            if function_calling and "<function_calls>" in response_text:
                try:
                    result_text = process_text_with_function_calls(response_text)
                    console.print(EnhancedMarkdown(result_text))
                except Exception as e:
                    console.print(f"[error]Error processing function calls: {str(e)}[/error]")
                    console.print(EnhancedMarkdown(response_text))
            else:
                console.print(EnhancedMarkdown(response_text))

@app.command("models")
def list_models():
    """List available models from Ollama."""
    client = OllamaClient()
    
    with create_spinner("Fetching models"):
        try:
            models = client.get_available_models()
        except Exception as e:
            console.print(f"[error]Error: {str(e)}[/error]")
            console.print("Make sure Ollama is running at " + client.base_url)
            return
    
    if not models:
        console.print("No models found. Make sure Ollama is running and has models available.")
        return
    
    # Create a formatted table
    from rich.table import Table
    table = Table(title="Available Models", box=ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Modified", style="blue")
    
    for model in models:
        size = model.get("size", "Unknown")
        if isinstance(size, int):
            # Convert to human-readable size
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.1f} MB"
        else:
            size_str = str(size)
        
        modified = model.get("modified", "Unknown")
        
        table.add_row(model["name"], size_str, modified)
    
    console.print(table)

@app.command("history")
def history():
    """List conversation history."""
    conversations = Conversation.list_conversations()
    
    if not conversations:
        console.print("No conversation history found.")
        return
    
    # Create a formatted table
    from rich.table import Table
    table = Table(title="Conversation History", box=ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("ID", style="cyan")
    table.add_column("Preview", style="green")
    table.add_column("Messages", style="blue")
    
    for i, conv in enumerate(conversations):
        # Get first user message as summary if available
        summary = "No messages"
        for msg in conv.get("messages", []):
            if msg.get("role") == "user":
                summary = msg.get("content", "No content")
                if len(summary) > 50:
                    summary = summary[:47] + "..."
                break
        
        # Count messages by role
        message_count = len(conv.get("messages", []))
        
        table.add_row(
            str(i+1), 
            conv.get("id", "Unknown"),
            summary,
            str(message_count)
        )
    
    console.print(table)

@app.command("agent")
def agent(
    prompt: str = typer.Argument(..., help="Task for the agent to perform"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
):
    """Launch an autonomous agent to perform a task."""
    from euclid.tools.agent import dispatch_agent
    
    # Set model in config if provided
    if model:
        config.ollama_model = model
    
    # Run the agent
    result = dispatch_agent(prompt)
    
    # Display the result
    console.print(EnhancedMarkdown(result))

@app.command("functions")
def functions():
    """List available functions."""
    functions_schema = get_available_functions()
    
    if not functions_schema:
        console.print("No functions available.")
        return
    
    # Create a formatted table
    from rich.table import Table
    table = Table(title="Available Functions", box=ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Parameters", style="blue")
    
    for name, schema in functions_schema.items():
        description = schema.get("description", "No description")
        # Truncate long descriptions
        if len(description) > 100:
            description = description[:97] + "..."
        
        params = schema.get("parameters", {}).get("properties", {})
        param_str = ", ".join(params.keys())
        
        table.add_row(name, description, param_str)
    
    console.print(table)

if __name__ == "__main__":
    app()