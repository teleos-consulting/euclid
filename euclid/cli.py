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
from rich.prompt import Prompt, Confirm

from euclid.config import config
from euclid.ollama import OllamaClient, Message
from euclid.conversation import Conversation
from euclid.models import ModelRegistry
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

# Import RAG features
try:
    from euclid.rag import (
        VectorDB, 
        vectordb, 
        create_collection, 
        list_collections, 
        query_collection
    )
    HAVE_RAG = True
except ImportError:
    HAVE_RAG = False

app = typer.Typer(help="A CLI tool for interacting with local Ollama models")
models_app = typer.Typer(help="Model management commands")
rag_app = typer.Typer(help="RAG (Retrieval Augmented Generation) commands")
app.add_typer(models_app, name="models")
app.add_typer(rag_app, name="rag")

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
    advanced_ui: bool = typer.Option(
        False, "--tui", "-u",
        help="Use the advanced Terminal UI"
    ),
):
    """Start an interactive chat session."""
    # Check if using advanced TUI
    if advanced_ui:
        from euclid.tui import run_tui
        run_tui(
            model=model,
            system_prompt=system_prompt,
            conversation_id=conversation_id,
            show_thinking=show_thinking
        )
        return
    
    # Clear screen for a clean start
    clear_screen()
    
    # Create or load conversation
    conv_id = conversation_id or str(uuid.uuid4())
    conversation = Conversation.load(conv_id)
    
    # Set up client
    client = OllamaClient(model=model)
    
    # Check if model exists
    registry = ModelRegistry()
    available_models = [m.name for m in registry.get_available_models()]
    
    if client.model not in available_models:
        console.print(f"[warning]Model '{client.model}' not found locally.[/warning]")
        should_pull = Confirm.ask(f"Would you like to pull the model '{client.model}' now?")
        
        if should_pull:
            with create_spinner(f"Pulling model {client.model}"):
                try:
                    registry.pull_model(client.model)
                    console.print(f"[success]Successfully pulled model: {client.model}[/success]")
                except Exception as e:
                    console.print(f"[error]Error pulling model: {str(e)}[/error]")
                    console.print("Available models:")
                    console.print(registry.list_available_models_table())
                    return
        else:
            console.print("Available models:")
            console.print(registry.list_available_models_table())
            return
    
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
        "[bold green]Euclid Chat[/bold green] \ud83c\udf3f",
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
            
            # Check for special commands
            if user_input.startswith("/model "):
                # Model switching command
                new_model = user_input.split("/model ")[1].strip()
                if new_model in available_models:
                    client = OllamaClient(model=new_model)
                    console.print(f"[info]Switched to model: [bold]{new_model}[/bold][/info]")
                    continue
                else:
                    console.print(f"[warning]Model '{new_model}' not found locally.[/warning]")
                    should_pull = Confirm.ask(f"Would you like to pull the model '{new_model}' now?")
                    
                    if should_pull:
                        with create_spinner(f"Pulling model {new_model}"):
                            try:
                                registry.pull_model(new_model)
                                client = OllamaClient(model=new_model)
                                console.print(f"[success]Switched to model: [bold]{new_model}[/bold][/success]")
                            except Exception as e:
                                console.print(f"[error]Error pulling model: {str(e)}[/error]")
                    continue
            
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
    
    # Check if model exists
    registry = ModelRegistry()
    available_models = [m.name for m in registry.get_available_models()]
    
    if client.model not in available_models:
        console.print(f"[warning]Model '{client.model}' not found locally.[/warning]")
        should_pull = Confirm.ask(f"Would you like to pull the model '{client.model}' now?")
        
        if should_pull:
            with create_spinner(f"Pulling model {client.model}"):
                try:
                    registry.pull_model(client.model)
                    console.print(f"[success]Successfully pulled model: {client.model}[/success]")
                except Exception as e:
                    console.print(f"[error]Error pulling model: {str(e)}[/error]")
                    console.print("Available models:")
                    console.print(registry.list_available_models_table())
                    return
        else:
            console.print("Available models:")
            console.print(registry.list_available_models_table())
            return
    
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

@app.command("tui")
def tui(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    system_prompt: Optional[str] = typer.Option(
        None, "--system", "-s", 
        help="System prompt to use"
    ),
    conversation_id: Optional[str] = typer.Option(
        None, "--conversation", "-c",
        help="Conversation ID to continue"
    ),
    show_thinking: bool = typer.Option(
        True, "--thinking/--no-thinking", "-t/-nt",
        help="Show model's thinking process"
    ),
):
    """Launch the advanced terminal user interface."""
    from euclid.tui import run_tui
    run_tui(
        model=model,
        system_prompt=system_prompt,
        conversation_id=conversation_id,
        show_thinking=show_thinking
    )

@models_app.command("list")
def list_models():
    """List available models from Ollama."""
    registry = ModelRegistry()
    console.print(registry.list_available_models_table())

@models_app.command("pull")
def pull_model(
    model_name: str = typer.Argument(..., help="Model to pull from Ollama"),
):
    """Pull a model from the Ollama repository."""
    registry = ModelRegistry()
    
    try:
        model = registry.pull_model(model_name)
        console.print(f"[success]Successfully pulled model: {model_name}[/success]")
    except Exception as e:
        console.print(f"[error]Error pulling model: {str(e)}[/error]")

@models_app.command("remove")
def remove_model(
    model_name: str = typer.Argument(..., help="Model to remove"),
):
    """Remove a model from Ollama."""
    registry = ModelRegistry()
    
    if Confirm.ask(f"Are you sure you want to remove model '{model_name}'?"):
        if registry.remove_model(model_name):
            console.print(f"[success]Successfully removed model: {model_name}[/success]")
        else:
            console.print(f"[error]Failed to remove model: {model_name}[/error]")

@models_app.command("details")
def model_details(
    model_name: str = typer.Argument(..., help="Model to get details for"),
):
    """Get detailed information about a model."""
    registry = ModelRegistry()
    details = registry.get_model_details(model_name)
    
    if not details:
        console.print(f"[error]No details found for model: {model_name}[/error]")
        return
    
    # Create a formatted display
    console.print(f"[bold]Details for {model_name}[/bold]")
    
    # Format model parameters
    parameters = details.get("parameters", {})
    if parameters:
        console.print("\n[bold]Parameters:[/bold]")
        for key, value in parameters.items():
            console.print(f"  {key}: {value}")
    
    # Format model template
    template = details.get("template", "")
    if template:
        console.print("\n[bold]Template:[/bold]")
        console.print(Syntax(template, "text", theme="monokai", line_numbers=True))
    
    # Format license information
    license_info = details.get("license", "")
    if license_info:
        console.print("\n[bold]License:[/bold]")
        console.print(license_info)

@models_app.command("benchmark")
def benchmark_model(
    model_name: str = typer.Argument(..., help="Model to benchmark"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Prompt to use for benchmarking"),
    iterations: int = typer.Option(3, "--iterations", "-i", help="Number of iterations to run"),
):
    """Benchmark a model's performance with a sample prompt."""
    registry = ModelRegistry()
    
    test_prompt = prompt or "Generate a short poem about AI assistants."
    
    try:
        results = registry.benchmark_model(model_name, test_prompt, iterations)
        
        avg_time = results["avg_time"]
        avg_tokens_per_second = results["avg_tokens_per_second"]
        
        console.print(f"[bold]Benchmark Results for {model_name}[/bold]")
        console.print(f"Prompt: \"{test_prompt}\"")
        console.print(f"Iterations: {iterations}")
        console.print(f"Average Response Time: {avg_time:.2f} seconds")
        console.print(f"Average Tokens/Second: {avg_tokens_per_second:.2f}")
        
        # Add individual runs
        console.print("\n[bold]Individual Runs:[/bold]")
        for i, (time_taken, tokens_per_sec) in enumerate(zip(results["times"], results["tokens_per_second"])):
            console.print(f"Run {i+1}: {time_taken:.2f}s ({tokens_per_sec:.2f} tokens/s)")
    except Exception as e:
        console.print(f"[error]Error benchmarking model: {str(e)}[/error]")

# RAG Commands (if available)
if HAVE_RAG:
    @rag_app.command("create")
    def create_rag_collection(
        name: str = typer.Argument(..., help="Collection name"),
        description: Optional[str] = typer.Option(None, "--description", "-d", help="Collection description"),
    ):
        """Create a new RAG collection."""
        from euclid.rag import create_collection
        collection_id = create_collection(name, description)
        console.print(f"[success]Collection created: {name} (ID: {collection_id})[/success]")
    
    @rag_app.command("list")
    def list_rag_collections():
        """List all RAG collections."""
        from euclid.rag import list_collections
        console.print(EnhancedMarkdown(list_collections()))
    
    @rag_app.command("add")
    def add_to_collection(
        collection_id: str = typer.Argument(..., help="Collection ID"),
        file_path: Optional[str] = typer.Option(None, "--file", "-f", help="File to add"),
        content: Optional[str] = typer.Option(None, "--content", "-c", help="Content to add"),
        title: Optional[str] = typer.Option(None, "--title", "-t", help="Document title"),
    ):
        """Add a document to a RAG collection."""
        if not file_path and not content:
            console.print("[error]Either --file or --content must be provided[/error]")
            return
        
        if file_path:
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                
                if not title:
                    title = os.path.basename(file_path)
            except Exception as e:
                console.print(f"[error]Error reading file: {str(e)}[/error]")
                return
        
        from euclid.rag import add_document
        console.print(EnhancedMarkdown(add_document(
            collection_id=collection_id,
            content=content,
            title=title,
            source=file_path
        )))
    
    @rag_app.command("query")
    def query_rag_collection(
        collection_id: str = typer.Argument(..., help="Collection ID"),
        query: str = typer.Argument(..., help="Query text"),
        top_k: int = typer.Option(3, "--top-k", "-k", help="Number of results to return"),
        use_chunks: bool = typer.Option(True, "--chunks/--documents", help="Search chunks or documents"),
    ):
        """Query a RAG collection."""
        from euclid.rag import query_collection
        console.print(EnhancedMarkdown(query_collection(
            collection_id=collection_id,
            query=query,
            top_k=top_k,
            use_chunks=use_chunks
        )))
    
    @rag_app.command("delete")
    def delete_rag_collection(
        collection_id: str = typer.Argument(..., help="Collection ID"),
    ):
        """Delete a RAG collection."""
        from euclid.rag import delete_collection
        console.print(EnhancedMarkdown(delete_collection(collection_id)))

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