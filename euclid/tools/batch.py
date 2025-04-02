"""Batch execution tool for running multiple tools in parallel."""

import json
from typing import List, Dict, Any, Optional

from euclid.functions import register_function, execute_function, batch_execute_functions
from euclid.formatting import console, create_spinner

@register_function(
    name="BatchTool",
    description="- Batch execution tool that runs multiple tool invocations in a single request\n- Tools are executed in parallel when possible, and otherwise serially\n- Takes a list of tool invocations (tool_name and input pairs)\n- Returns the collected results from all invocations\n- Use this tool when you need to run multiple independent tool operations at once -- it is awesome for speeding up your workflow, reducing both context usage and latency\n- Each tool will respect its own permissions and validation rules"
)
def batch_tool(description: str, invocations: List[Dict[str, Any]]) -> str:
    """Execute multiple tool invocations in parallel.
    
    Args:
        description: A short description of the batch operation.
        invocations: List of tool invocations, each with a tool_name and input.
        
    Returns:
        Combined results from all invocations.
    """
    if not invocations:
        return "No invocations provided."
    
    console.print(f"[info]Running batch operation: {description}[/info]")
    
    # Convert invocations to function calls
    calls = []
    for invoc in invocations:
        tool_name = invoc.get("tool_name")
        input_params = invoc.get("input", {})
        
        if not tool_name:
            console.print("[error]Missing tool_name in invocation[/error]")
            continue
        
        calls.append({
            "function": tool_name,
            "parameters": input_params
        })
    
    # Execute all functions in parallel
    with create_spinner(f"Executing {len(calls)} tools in parallel"):
        results = batch_execute_functions(calls)
    
    # Format results
    result_text = f"BatchTool({description}) results:\n\n"
    
    for call, result in zip(calls, results.values()):
        func_name = call["function"]
        result_text += f"{func_name}:\n{result}\n\n"
    
    return result_text.strip()
