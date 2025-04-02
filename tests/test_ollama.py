import unittest
from unittest.mock import patch, MagicMock
import json

from euclid.ollama import OllamaClient, Message


class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient(base_url="http://test-ollama:11434", model="test-model")
        self.test_messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Hello!")
        ]
    
    @patch('requests.post')
    def test_chat_completion_non_streaming(self, mock_post):
        """Test chat completion without streaming."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Hello there! How can I help?"}
        }
        mock_post.return_value = mock_response
        
        # Call the method
        response = self.client.chat_completion(self.test_messages, stream=False)
        
        # Verify the call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['model'], "test-model")
        self.assertEqual(kwargs['json']['stream'], False)
        self.assertEqual(kwargs['json']['messages'][0]['role'], "system")
        
        # Verify the response
        self.assertEqual(response['message']['content'], "Hello there! How can I help?")
    
    @patch('requests.post')
    def test_chat_completion_streaming(self, mock_post):
        """Test chat completion with streaming."""
        # Mock the streaming response
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            json.dumps({"message": {"content": "Hello"}}).encode(),
            json.dumps({"message": {"content": " there!"}}).encode(),
            json.dumps({"message": {"content": " How can I help?"}}).encode(),
        ]
        mock_post.return_value = mock_response
        
        # Call the method and collect results
        chunks = list(self.client.chat_completion(self.test_messages, stream=True))
        
        # Verify the call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['model'], "test-model")
        self.assertEqual(kwargs['json']['stream'], True)
        
        # Verify the chunks
        self.assertEqual(chunks, ["Hello", " there!", " How can I help?"])
    
    @patch('requests.get')
    def test_get_available_models(self, mock_get):
        """Test retrieving available models."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "model1", "size": 1000000},
                {"name": "model2", "size": 2000000}
            ]
        }
        mock_get.return_value = mock_response
        
        # Call the method
        models = self.client.get_available_models()
        
        # Verify the call
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], "http://test-ollama:11434/api/tags")
        
        # Verify the models
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["name"], "model1")
        self.assertEqual(models[1]["name"], "model2")


if __name__ == "__main__":
    unittest.main()
