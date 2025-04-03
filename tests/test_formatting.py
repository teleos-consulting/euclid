import unittest
from unittest.mock import patch, MagicMock
import io
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax

from euclid.formatting import (
    EnhancedMarkdown, 
    format_user_message, 
    format_assistant_message, 
    format_system_message, 
    format_tool_message,
    format_function_call,
    create_spinner,
    clear_screen
)


class TestFormatting(unittest.TestCase):
    
    def test_format_user_message(self):
        message = "Hello, world!"
        formatted = format_user_message(message)
        self.assertEqual(formatted, "[user]Hello, world![/user]")
    
    def test_format_system_message(self):
        message = "System notification"
        formatted = format_system_message(message)
        self.assertEqual(formatted, "[system]System notification[/system]")
    
    def test_format_tool_message(self):
        tool_name = "TestTool"
        content = "Tool output"
        formatted = format_tool_message(tool_name, content)
        self.assertEqual(formatted, "[tool]TestTool:[/tool] Tool output")
    
    def test_format_assistant_message(self):
        message = "Assistant response"
        formatted = format_assistant_message(message)
        self.assertIsInstance(formatted, EnhancedMarkdown)
        self.assertEqual(formatted.markup, "Assistant response")
    
    def test_format_function_call(self):
        function_name = "test_function"
        parameters = {"param1": "value1", "param2": 42}
        
        result = format_function_call(function_name, parameters)
        
        self.assertIsInstance(result, Panel)
        self.assertEqual(result.title, "[function_name]test_function[/function_name]")
        
        # Check that the table inside the panel has the correct structure
        self.assertIsInstance(result.renderable, Table)
        table = result.renderable
        self.assertEqual(len(table.columns), 2)
    
    def test_enhanced_markdown_code_block(self):
        md = EnhancedMarkdown("Test")
        code = "def hello(): print('Hello')"
        
        # Test with explicit lexer
        result = md._render_code_block(code, "python")
        self.assertIsInstance(result, Syntax)
        self.assertEqual(result.code, code)
        self.assertEqual(result.lexer_name, "python")
        
        # Test with no lexer (should guess)
        result = md._render_code_block(code)
        self.assertIsInstance(result, Syntax)
        self.assertEqual(result.code, code)
    
    @patch('euclid.formatting.TERMINAL_IMAGE_AVAILABLE', False)
    def test_render_image_unavailable(self):
        md = EnhancedMarkdown("Test")
        result = md._render_image("test.png", "test image")
        self.assertEqual(str(result), "[Image: test image]")
    
    @patch('euclid.formatting.TERMINAL_IMAGE_AVAILABLE', True)
    @patch('os.path.exists')
    @patch('euclid.formatting.terminal_image.TerminalImage')
    def test_render_image_available(self, mock_terminal_image, mock_exists):
        mock_exists.return_value = True
        mock_img = MagicMock()
        mock_terminal_image.return_value = mock_img
        
        md = EnhancedMarkdown("Test")
        result = md._render_image("test.png", "test image")
        
        mock_terminal_image.assert_called_once()
        self.assertEqual(result, mock_img)
    
    @patch('euclid.formatting.TERMINAL_IMAGE_AVAILABLE', True)
    @patch('os.path.exists')
    @patch('euclid.formatting.terminal_image.TerminalImage')
    def test_render_image_exception(self, mock_terminal_image, mock_exists):
        mock_exists.return_value = True
        mock_terminal_image.side_effect = Exception("Image error")
        
        md = EnhancedMarkdown("Test")
        result = md._render_image("test.png", "test image")
        
        self.assertEqual(str(result), "[Image: test image]")
    
    def test_create_spinner(self):
        spinner = create_spinner("Testing")
        self.assertEqual(len(spinner.columns), 2)  # SpinnerColumn and TextColumn
    
    @patch('os.system')
    def test_clear_screen_unix(self, mock_system):
        with patch('os.name', 'posix'):
            clear_screen()
            mock_system.assert_called_once_with('clear')
    
    @patch('os.system')
    def test_clear_screen_windows(self, mock_system):
        with patch('os.name', 'nt'):
            clear_screen()
            mock_system.assert_called_once_with('cls')


if __name__ == '__main__':
    unittest.main()