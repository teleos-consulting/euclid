"""Agent tool for autonomous task completion."""

import json
import uuid
from typing import Dict, Any, List, Optional

from euclid.functions import register_function
from euclid.formatting import console, create_spinner, display_thinking
from euclid.ollama import OllamaClient, Message
from euclid.config import config

@register_function(
    name="dispatch_agent",
    description="Launch a new agent that has access to all tools and can perform autonomous tasks. The agent will receive your prompt, perform the requested task using available tools, and return a report of its findings."
)
def dispatch_agent(prompt: str) -> str:
    """Launch an agent to perform an autonomous task.
    
    Args:
        prompt: The task for the agent to perform.
        
    Returns:
        The agent's final report.
    """
    # Add system instructions for the agent
    system_prompt = f"""
    You are an autonomous agent with access to a set of tools. 
    Your task is to help the user by performing the task they request.
    
    You have access to the following tools:
    1. View - Read the contents of a file
    2. LS - List files in a directory
    3. GrepTool - Search for patterns in files
    4. GlobTool - Find files matching a pattern
    5. SearchTool - Search for files containing specific text
    
    When using tools, use the standard function call format:
    <function_calls>
    <invoke name="ToolName">
    <parameter name="param1">value1</parameter>
    <parameter name="param2">value2</parameter>
    </invoke>
    </function_calls>
    
    Always think step by step and plan your approach before taking action.
    After completing the task, provide a concise report of your findings.
    """
    
    # Initialize the agent with the system prompt and user task
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=prompt)
    ]
    
    # Create Ollama client for the agent
    client = OllamaClient(model=config.ollama_model)
    
    # Display agent initialization
    console.print("[info]Initializing agent...[/info]")
    
    # Generate agent response
    thinking_chunks = []
    final_response = ""
    
    with create_spinner("Agent working"):
        try:
            # Stream the agent's thinking process
            for chunk in client.chat_completion(messages, stream=True):
                thinking_chunks.append(chunk)
                final_response += chunk
        except Exception as e:
            console.print(f"[error]Error during agent execution: {str(e)}[/error]")
            return f"Agent execution failed: {str(e)}"
    
    # Display the agent's thinking process if it's not too long
    thinking_text = "".join(thinking_chunks)
    if len(thinking_text) < 10000:  # Don't display if it's too long
        display_thinking(thinking_text)
    
    # Process any function calls in the response
    # This would involve implementing a function call parser and executor
    # For now, we'll just return the raw response
    
    console.print("[info]Agent completed task[/info]")
    
    return final_response
