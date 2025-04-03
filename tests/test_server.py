import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import asyncio
from fastapi.testclient import TestClient

from euclid.server import (
    app,
    ChatMessage,
    ChatCompletionRequest,
    list_models,
    chat_completions,
    list_functions,
    execute_function
)


class TestServerEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "Euclid API Server")
        self.assertIn("version", data)
        self.assertIn("description", data)
    
    @patch('euclid.server.model_registry')
    def test_list_models(self, mock_registry):
        # Mock the model registry
        mock_model1 = MagicMock()
        mock_model1.name = "model1"
        mock_model1.modified = "2023-01-01"
        
        mock_model2 = MagicMock()
        mock_model2.name = "model2"
        mock_model2.modified = None
        
        mock_registry.get_available_models.return_value = [mock_model1, mock_model2]
        
        response = self.client.get("/v1/models")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["models"]), 2)
        self.assertEqual(data["models"][0]["id"], "model1")
        self.assertEqual(data["models"][1]["id"], "model2")
        self.assertEqual(data["models"][0]["created"], "2023-01-01")
        self.assertEqual(data["models"][1]["created"], 0)
    
    @patch('euclid.server.model_registry')
    def test_list_models_error(self, mock_registry):
        # Mock an error scenario
        mock_registry.get_available_models.side_effect = Exception("Test error")
        
        response = self.client.get("/v1/models")
        
        self.assertEqual(response.status_code, 500)
        self.assertIn("Error listing models", response.json()["detail"])
    
    @patch('euclid.server.OllamaClient')
    @patch('euclid.server.model_registry')
    def test_chat_completions(self, mock_registry, mock_client_class):
        # Mock the model registry
        mock_registry.get_available_models.return_value = [
            MagicMock(name="test-model")
        ]
        
        # Mock the Ollama client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the chat completion response
        mock_client.chat_completion.return_value = {
            "message": {
                "content": "This is a test response"
            },
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        
        # Create a chat request
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["choices"][0]["message"]["content"], "This is a test response")
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(data["usage"]["total_tokens"], 30)
        
        # Verify client was called with right parameters
        mock_client_class.assert_called_once_with(model="test-model")
        mock_client.chat_completion.assert_called_once()
        call_args = mock_client.chat_completion.call_args[0]
        self.assertEqual(len(call_args[0]), 1)  # One message
        self.assertEqual(call_args[0][0].role, "user")
        self.assertEqual(call_args[0][0].content, "Hello")
        self.assertEqual(mock_client.chat_completion.call_args[1]["temperature"], 0.7)
        self.assertEqual(mock_client.chat_completion.call_args[1]["stream"], False)
    
    @patch('euclid.server.OllamaClient')
    @patch('euclid.server.model_registry')
    def test_chat_completions_with_function_call(self, mock_registry, mock_client_class):
        # Mock the model registry
        mock_registry.get_available_models.return_value = [
            MagicMock(name="test-model")
        ]
        
        # Mock the Ollama client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the chat completion response with function call
        mock_client.chat_completion.return_value = {
            "message": {
                "content": "Here's the result <function_calls>\n<invoke name=\"test_function\">\n<parameter name=\"param1\">value1</parameter>\n</invoke>\n</function_calls>"
            }
        }
        
        # Create a chat request
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Call a function"}
            ]
        }
        
        with patch('euclid.server.parse_function_calls') as mock_parse:
            mock_parse.return_value = [{"function": "test_function", "parameters": {"param1": "value1"}}]
            
            response = self.client.post("/v1/chat/completions", json=request_data)
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("function_call", data["choices"][0]["message"])
            self.assertEqual(data["choices"][0]["message"]["function_call"]["name"], "test_function")
            self.assertEqual(json.loads(data["choices"][0]["message"]["function_call"]["arguments"]), {"param1": "value1"})
            self.assertEqual(data["choices"][0]["finish_reason"], "function_call")
    
    @patch('euclid.server.OllamaClient')
    @patch('euclid.server.model_registry')
    def test_chat_completions_model_not_found(self, mock_registry, mock_client_class):
        # Mock the model registry with no models
        mock_registry.get_available_models.return_value = []
        
        # Mock the model pull to fail
        mock_registry.pull_model.side_effect = Exception("Model not found")
        
        # Create a chat request with non-existent model
        request_data = {
            "model": "nonexistent-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found and could not be pulled", response.json()["detail"])
    
    @patch('euclid.server.get_available_functions')
    def test_list_functions(self, mock_get_functions):
        # Mock the function registry
        mock_get_functions.return_value = {
            "TestFunction1": {
                "description": "Test function 1",
                "parameters": {
                    "param1": {"type": "string"}
                }
            },
            "TestFunction2": {
                "description": "Test function 2",
                "parameters": {
                    "param2": {"type": "number"}
                }
            }
        }
        
        response = self.client.get("/v1/functions")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["functions"]), 2)
        
        # Check function details
        functions = {f["name"]: f for f in data["functions"]}
        self.assertIn("TestFunction1", functions)
        self.assertIn("TestFunction2", functions)
        self.assertEqual(functions["TestFunction1"]["description"], "Test function 1")
        self.assertEqual(functions["TestFunction2"]["description"], "Test function 2")
        self.assertEqual(functions["TestFunction1"]["parameters"]["param1"]["type"], "string")
    
    @patch('euclid.server.execute_function')
    def test_execute_function(self, mock_execute):
        # Mock the function execution
        mock_execute.return_value = "Function result"
        
        # Create a function execution request
        request_data = {
            "parameters": {
                "param1": "value1"
            }
        }
        
        response = self.client.post("/v1/functions/TestFunction/execute", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "TestFunction")
        self.assertEqual(data["result"], "Function result")
        
        # Verify function was called with right parameters
        mock_execute.assert_called_once_with("TestFunction", {"param1": "value1"})
    
    @patch('euclid.server.execute_function')
    def test_execute_function_error(self, mock_execute):
        # Mock a function execution error
        mock_execute.side_effect = Exception("Function error")
        
        response = self.client.post("/v1/functions/TestFunction/execute", json={"parameters": {}})
        
        self.assertEqual(response.status_code, 500)
        self.assertIn("Error executing function", response.json()["detail"])


@unittest.skipIf(not hasattr(app, "web_fetch_endpoint"), "Web functions not available")
class TestWebEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
    
    @patch('euclid.server.web_fetch')
    def test_web_fetch(self, mock_web_fetch):
        # Mock web fetch function
        mock_web_fetch.return_value = {
            "title": "Test Page",
            "content": "Test content",
            "summary": "Test summary"
        }
        
        # Create web fetch request
        request_data = {
            "url": "https://example.com",
            "prompt": "Summarize this page"
        }
        
        response = self.client.post("/v1/web/fetch", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["title"], "Test Page")
        self.assertEqual(data["summary"], "Test summary")
        
        # Verify function was called with right parameters
        mock_web_fetch.assert_called_once_with(url="https://example.com", prompt="Summarize this page")
    
    @patch('euclid.server.search_web')
    def test_web_search(self, mock_search_web):
        # Mock web search function
        mock_search_web.return_value = {
            "results": [
                {"title": "Result 1", "snippet": "Snippet 1", "url": "https://example.com/1"},
                {"title": "Result 2", "snippet": "Snippet 2", "url": "https://example.com/2"}
            ]
        }
        
        # Create web search request
        request_data = {
            "query": "test query",
            "num_results": 2
        }
        
        response = self.client.post("/v1/web/search", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(data["results"][0]["title"], "Result 1")
        
        # Verify function was called with right parameters
        mock_search_web.assert_called_once_with(query="test query", num_results=2)


if __name__ == '__main__':
    unittest.main()