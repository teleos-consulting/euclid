import unittest
import tempfile
import json
import os
from pathlib import Path

from euclid.functions import (
    register_function,
    get_available_functions,
    parse_function_calls,
    validate_function_call,
    execute_function,
    _generate_schema_from_function
)


class TestFunctions(unittest.TestCase):
    def setUp(self):
        # Sample function for testing
        @register_function(name="TestFunc", description="Test function for unit tests")
        def test_func(param1: str, param2: int = 0) -> str:
            """Test function.
            
            Args:
                param1: First parameter
                param2: Second parameter
            """
            return f"Called with {param1} and {param2}"
        
        self.test_func = test_func
    
    def test_register_function(self):
        """Test that functions are properly registered."""
        functions = get_available_functions()
        self.assertIn("TestFunc", functions)
        
        schema = functions["TestFunc"]
        self.assertEqual(schema["name"], "TestFunc")
        self.assertEqual(schema["description"], "Test function for unit tests")
        
        # Check parameters
        params = schema["parameters"]["properties"]
        self.assertIn("param1", params)
        self.assertIn("param2", params)
        
        # Check required parameters
        required = schema["parameters"]["required"]
        self.assertIn("param1", required)
        self.assertNotIn("param2", required)  # param2 has a default value
    
    def test_parse_function_calls(self):
        """Test parsing function calls from text."""
        test_text = """
        This is a test text.
        
        <function_calls>
        <invoke name="TestFunc">
        <parameter name="param1">test value</parameter>
        <parameter name="param2">42</parameter>
        </invoke>
        </function_calls>
        
        More text.
        """
        
        calls = parse_function_calls(test_text)
        self.assertEqual(len(calls), 1)
        
        call = calls[0]
        self.assertEqual(call["function"], "TestFunc")
        self.assertEqual(call["parameters"]["param1"], "test value")
        self.assertEqual(call["parameters"]["param2"], 42)
    
    def test_validate_function_call(self):
        """Test function call validation."""
        # Valid parameters
        params = {"param1": "test", "param2": 42}
        validated = validate_function_call("TestFunc", params)
        self.assertEqual(validated, params)
        
        # Missing required parameter
        with self.assertRaises(ValueError):
            validate_function_call("TestFunc", {"param2": 42})
        
        # Invalid parameter type
        with self.assertRaises(ValueError):
            validate_function_call("TestFunc", {"param1": "test", "param2": "not an int"})
    
    def test_execute_function(self):
        """Test function execution."""
        result = execute_function("TestFunc", {"param1": "hello", "param2": 42})
        self.assertEqual(result, "Called with hello and 42")
        
        # With default parameter
        result = execute_function("TestFunc", {"param1": "hello"})
        self.assertEqual(result, "Called with hello and 0")


if __name__ == "__main__":
    unittest.main()
