"""Framework for structured function calls and execution."""

import inspect
import json
import re
import traceback
from typing import Dict, Any, List, Optional, Callable, Union, Type, TypeVar, get_type_hints
from threading import Thread, Event
from queue import Queue

import jsonschema
from pydantic import BaseModel, create_model, Field
from rich.panel import Panel

from euclid.formatting import console, format_function_call, create_spinner

# Type definitions
FunctionDef = Dict[str, Any]
FunctionRegistry = Dict[str, Callable]
FunctionSchema = Dict[str, Any]

# Global function registry
_FUNCTIONS: FunctionRegistry = {}

# Function call regex pattern
FUNCTION_CALL_PATTERN = re.compile(
    r"<function_calls>\s*" +
    r"<invoke name=\"([\w_]+)\">(.*?)<\/antml:invoke>" +
    r"\s*<\/antml:function_calls>",
    re.DOTALL
)

PARAMETER_PATTERN = re.compile(
    r"<parameter name=\"([\w_]+)\">(.*?)<\/antml:parameter>",
    re.DOTALL
)

def register_function(name: str, description: str = "", schema: Optional[FunctionSchema] = None):
    """Decorator to register a function in the function registry.
    
    Args:
        name: Name of the function.
        description: Description of the function.
        schema: JSON Schema for the function parameters (optional).
            If not provided, it will be generated from the function signature.
    """
    def decorator(func: Callable):
        # Generate schema from function signature if not provided
        func_schema = schema or _generate_schema_from_function(func, description)
        
        # Store function with its schema
        _FUNCTIONS[name] = func
        func._schema = func_schema
        
        return func
    return decorator

def get_available_functions() -> Dict[str, FunctionDef]:
    """Get all available functions with their schemas.
    
    Returns:
        Dictionary mapping function names to their definitions.
    """
    function_defs = {}
    for name, func in _FUNCTIONS.items():
        if hasattr(func, '_schema'):
            function_defs[name] = func._schema
    return function_defs

def parse_function_calls(text: str) -> List[Dict[str, Any]]:
    """Parse function calls from text using the antml format.
    
    Args:
        text: Text containing function calls.
        
    Returns:
        List of dictionaries with function name and parameters.
    """
    calls = []
    matches = FUNCTION_CALL_PATTERN.findall(text)
    
    for func_name, params_text in matches:
        param_matches = PARAMETER_PATTERN.findall(params_text)
        parameters = {}
        
        for param_name, param_value in param_matches:
            # Try to parse as JSON if possible
            try:
                parameters[param_name] = json.loads(param_value)
            except json.JSONDecodeError:
                parameters[param_name] = param_value
        
        calls.append({
            "function": func_name,
            "parameters": parameters
        })
    
    return calls

def validate_function_call(func_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a function call against its schema.
    
    Args:
        func_name: Name of the function to call.
        parameters: Parameters for the function.
        
    Returns:
        Validated parameters.
        
    Raises:
        ValueError: If the function doesn't exist or parameters are invalid.
    """
    if func_name not in _FUNCTIONS:
        raise ValueError(f"Function '{func_name}' not found in registry.")
    
    func = _FUNCTIONS[func_name]
    if not hasattr(func, '_schema'):
        raise ValueError(f"Function '{func_name}' has no schema.")
    
    schema = func._schema.get("parameters", {})
    
    try:
        jsonschema.validate(instance=parameters, schema=schema)
        return parameters
    except jsonschema.exceptions.ValidationError as e:
        raise ValueError(f"Invalid parameters for function '{func_name}': {str(e)}")

def execute_function(func_name: str, parameters: Dict[str, Any]) -> Any:
    """Execute a registered function with the given parameters.
    
    Args:
        func_name: Name of the function to call.
        parameters: Parameters for the function.
        
    Returns:
        Result of the function call.
        
    Raises:
        ValueError: If the function doesn't exist.
        Exception: Any exception raised by the function.
    """
    if func_name not in _FUNCTIONS:
        raise ValueError(f"Function '{func_name}' not found in registry.")
    
    # Validate parameters
    validated_params = validate_function_call(func_name, parameters)
    
    # Execute function
    try:
        return _FUNCTIONS[func_name](**validated_params)
    except Exception as e:
        error_message = f"Error executing function '{func_name}': {str(e)}\n"
        error_message += traceback.format_exc()
        raise Exception(error_message)

def batch_execute_functions(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute multiple functions in parallel where possible.
    
    Args:
        calls: List of function calls with function name and parameters.
        
    Returns:
        Dictionary mapping function names to their results.
    """
    results = {}
    threads = []
    result_queue = Queue()
    
    with create_spinner("Executing functions") as progress:
        for i, call in enumerate(calls):
            func_name = call["function"]
            parameters = call["parameters"]
            
            # Create a thread for each function call
            thread = Thread(
                target=_thread_function,
                args=(func_name, parameters, result_queue, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        while not result_queue.empty():
            idx, name, result = result_queue.get()
            results[name] = result
    
    return results

def process_text_with_function_calls(text: str) -> str:
    """Process text and execute any function calls found in it.
    
    Args:
        text: Text that may contain function calls.
        
    Returns:
        Processed text with function call results.
    """
    # Check if text contains function calls
    if "<function_calls>" not in text:
        return text
    
    # Parse and execute function calls
    calls = parse_function_calls(text)
    if not calls:
        return text
    
    # Execute functions
    results = {}
    for call in calls:
        func_name = call["function"]
        parameters = call["parameters"]
        
        # Display the function call
        console.print(format_function_call(func_name, parameters))
        
        # Execute the function
        try:
            result = execute_function(func_name, parameters)
            results[func_name] = result
        except Exception as e:
            console.print(f"[error]Error executing {func_name}:[/error] {str(e)}")
            results[func_name] = str(e)
    
    # Replace function calls in text with their results
    result_text = text
    for func_name, result in results.items():
        # Convert result to string if needed
        if not isinstance(result, str):
            result = str(result)
        
        # Find the function call block and replace it
        pattern = re.compile(
            r"<function_calls>\s*" +
            r"<invoke name=\"" + re.escape(func_name) + r"\">.*?<\/antml:invoke>" +
            r"\s*<\/antml:function_calls>",
            re.DOTALL
        )
        result_text = pattern.sub(result, result_text, count=1)
    
    return result_text

def get_function_schemas_as_json() -> str:
    """Get all function schemas as a JSON string.
    
    Returns:
        JSON string with all function schemas.
    """
    schemas = get_available_functions()
    return json.dumps({"functions": list(schemas.values())}, indent=2)

def _thread_function(func_name: str, parameters: Dict[str, Any], result_queue: Queue, idx: int):
    """Thread worker function for batch execution.
    
    Args:
        func_name: Name of the function to call.
        parameters: Parameters for the function.
        result_queue: Queue to store results.
        idx: Index for ordering results.
    """
    try:
        result = execute_function(func_name, parameters)
        result_queue.put((idx, func_name, result))
    except Exception as e:
        result_queue.put((idx, func_name, str(e)))

def _generate_schema_from_function(func: Callable, description: str = "") -> FunctionSchema:
    """Generate a JSON Schema from a function's signature.
    
    Args:
        func: Function to generate schema for.
        description: Description of the function.
        
    Returns:
        JSON Schema for the function.
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    # Build parameter properties
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        param_type = type_hints.get(name, Any)
        param_default = None if param.default is param.empty else param.default
        param_description = ""
        
        # Get description from docstring if available
        if func.__doc__:
            docstring = func.__doc__
            param_match = re.search(rf"\s+{name}:\s+(.+?)(?:\n\s+\w+:|$)", docstring, re.DOTALL)
            if param_match:
                param_description = param_match.group(1).strip()
        
        # Map Python type to JSON Schema type
        if param_type == str:
            param_schema = {"type": "string"}
        elif param_type == int:
            param_schema = {"type": "integer"}
        elif param_type == float:
            param_schema = {"type": "number"}
        elif param_type == bool:
            param_schema = {"type": "boolean"}
        elif param_type == list or getattr(param_type, "__origin__", None) is list:
            param_schema = {"type": "array"}
        elif param_type == dict or getattr(param_type, "__origin__", None) is dict:
            param_schema = {"type": "object"}
        else:
            param_schema = {}
        
        if param_description:
            param_schema["description"] = param_description
        
        properties[name] = param_schema
        
        # Add to required list if no default value
        if param.default is param.empty:
            required.append(name)
    
    # Build the complete schema
    schema = {
        "name": func.__name__,
        "description": description or (func.__doc__ or "").strip(),
        "parameters": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required
        }
    }
    
    return schema
