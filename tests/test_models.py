import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import time
from pathlib import Path

from rich.table import Table

from euclid.models import (
    ModelInfo,
    ModelRegistry,
    list_models,
    pull_model,
    remove_model,
    model_details,
    benchmark_model
)


class TestModelInfo(unittest.TestCase):
    def test_model_info_creation(self):
        model = ModelInfo(name="test-model")
        self.assertEqual(model.name, "test-model")
        self.assertIsNone(model.size)
        self.assertIsNone(model.modified)
        self.assertIsNone(model.digest)
        self.assertEqual(model.details, {})
        
        model = ModelInfo(
            name="test-model", 
            size=1000, 
            modified="2023-01-01", 
            digest="abc123",
            details={"parameter": "value"}
        )
        self.assertEqual(model.name, "test-model")
        self.assertEqual(model.size, 1000)
        self.assertEqual(model.modified, "2023-01-01")
        self.assertEqual(model.digest, "abc123")
        self.assertEqual(model.details, {"parameter": "value"})


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry(base_url="http://test-ollama:11434")
        self.test_models = [
            {"name": "model1", "size": 1000, "modified": "2023-01-01"},
            {"name": "model2", "size": 2000, "modified": "2023-01-02"}
        ]
    
    @patch('euclid.models.requests.get')
    @patch('pathlib.Path.exists')
    def test_get_available_models_api(self, mock_exists, mock_get):
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": self.test_models}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        mock_exists.return_value = False
        
        # Create mock for file operations
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file):
            models = self.registry.get_available_models(refresh=True)
        
        # Verify API call
        mock_get.assert_called_once_with("http://test-ollama:11434/api/tags")
        
        # Verify results
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, "model1")
        self.assertEqual(models[1].name, "model2")
        self.assertEqual(models[0].size, 1000)
        
        # Verify cache was written
        mock_file.assert_called_once()
        mock_file().write.assert_called_once_with(json.dumps(self.test_models))
    
    @patch('euclid.models.requests.get')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_get_available_models_cache(self, mock_stat, mock_exists, mock_get):
        # Mock cache file exists and is recent
        mock_exists.return_value = True
        mock_stat_result = MagicMock()
        mock_stat_result.st_mtime = time.time() - 1800  # 30 minutes ago
        mock_stat.return_value = mock_stat_result
        
        # Set up mock open to return test models
        mock_file = mock_open(read_data=json.dumps(self.test_models))
        
        with patch('builtins.open', mock_file):
            models = self.registry.get_available_models()
        
        # Verify API was not called
        mock_get.assert_not_called()
        
        # Verify results from cache
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, "model1")
        self.assertEqual(models[1].name, "model2")
    
    @patch('euclid.models.requests.post')
    @patch('euclid.models.subprocess.run')
    def test_pull_model_with_progress(self, mock_run, mock_post):
        # Mock subprocess result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Mock get_available_models to return a test model
        self.registry.get_available_models = MagicMock(return_value=[
            ModelInfo(name="test-model", size=1000)
        ])
        
        result = self.registry.pull_model("test-model", show_progress=True)
        
        # Verify subprocess was called
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args[0][0], ["ollama", "pull", "test-model"])
        
        # Verify API was not called directly
        mock_post.assert_not_called()
        
        # Verify result
        self.assertEqual(result.name, "test-model")
        self.assertEqual(result.size, 1000)
        
        # Verify get_available_models was called to refresh
        self.registry.get_available_models.assert_called_once_with(refresh=True)
    
    @patch('euclid.models.requests.post')
    def test_pull_model_without_progress(self, mock_post):
        # Mock API response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Mock get_available_models to return a test model
        self.registry.get_available_models = MagicMock(return_value=[
            ModelInfo(name="test-model", size=1000)
        ])
        
        result = self.registry.pull_model("test-model", show_progress=False)
        
        # Verify API was called
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[1]["json"], {"name": "test-model"})
        
        # Verify result
        self.assertEqual(result.name, "test-model")
        self.assertEqual(result.size, 1000)
    
    @patch('euclid.models.requests.delete')
    def test_remove_model(self, mock_delete):
        # Mock API response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_delete.return_value = mock_response
        
        # Mock get_available_models
        self.registry.get_available_models = MagicMock()
        
        result = self.registry.remove_model("test-model")
        
        # Verify API was called
        mock_delete.assert_called_once()
        self.assertEqual(mock_delete.call_args[1]["json"], {"name": "test-model"})
        
        # Verify result
        self.assertTrue(result)
        
        # Verify get_available_models was called to refresh
        self.registry.get_available_models.assert_called_once_with(refresh=True)
    
    @patch('euclid.models.requests.post')
    def test_get_model_details(self, mock_post):
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "parameters": {"param1": "value1"},
            "template": "Template content",
            "license": "License info"
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        result = self.registry.get_model_details("test-model")
        
        # Verify API was called
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[1]["json"], {"name": "test-model"})
        
        # Verify result
        self.assertEqual(result["parameters"]["param1"], "value1")
        self.assertEqual(result["template"], "Template content")
        self.assertEqual(result["license"], "License info")
    
    @patch('euclid.models.requests.post')
    @patch('euclid.models.Progress')
    def test_benchmark_model(self, mock_progress, mock_post):
        # Mock API responses
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "eval_count": 100,
            "prompt_eval_count": 50
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Mock progress
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance
        mock_progress_instance.add_task.return_value = "task_id"
        
        result = self.registry.benchmark_model("test-model", "Test prompt", 2)
        
        # Verify API was called twice (2 iterations)
        self.assertEqual(mock_post.call_count, 2)
        
        # Verify result structure
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["prompt"], "Test prompt")
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(len(result["times"]), 2)
        self.assertEqual(len(result["tokens_per_second"]), 2)
        self.assertIsInstance(result["avg_time"], float)
        self.assertIsInstance(result["avg_tokens_per_second"], float)
    
    def test_list_available_models_table(self):
        # Mock get_available_models
        self.registry.get_available_models = MagicMock(return_value=[
            ModelInfo(name="model1", size=1024, modified="2023-01-01"),
            ModelInfo(name="model2", size=1024*1024, modified="2023-01-02"),
            ModelInfo(name="model3", size=1024*1024*1024, modified="2023-01-03")
        ])
        
        result = self.registry.list_available_models_table()
        
        # Verify table structure
        self.assertIsInstance(result, Table)
        self.assertEqual(len(result.columns), 3)


class TestModelFunctions(unittest.TestCase):
    @patch('euclid.models.ModelRegistry')
    def test_list_models(self, mock_registry_class):
        # Mock registry instance
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        
        # Mock get_available_models
        mock_registry.get_available_models.return_value = [
            ModelInfo(name="model1", size=1024*1024),
            ModelInfo(name="model2", size=1024*1024*1024)
        ]
        
        result = list_models()
        
        # Verify result format
        self.assertIn("# Available Ollama Models", result)
        self.assertIn("- **model1** (1.0 MB)", result)
        self.assertIn("- **model2** (1.0 GB)", result)
    
    @patch('euclid.models.ModelRegistry')
    def test_pull_model(self, mock_registry_class):
        # Mock registry instance
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        
        # Mock pull_model
        mock_registry.pull_model.return_value = ModelInfo(name="test-model", size=1024*1024)
        
        result = pull_model("test-model")
        
        # Verify registry method was called
        mock_registry.pull_model.assert_called_once_with("test-model")
        
        # Verify result format
        self.assertIn("Successfully pulled model: **test-model**", result)
        self.assertIn("(1.0 MB)", result)
    
    @patch('euclid.models.ModelRegistry')
    def test_remove_model_success(self, mock_registry_class):
        # Mock registry instance
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        
        # Mock remove_model
        mock_registry.remove_model.return_value = True
        
        result = remove_model("test-model")
        
        # Verify registry method was called
        mock_registry.remove_model.assert_called_once_with("test-model")
        
        # Verify result format
        self.assertIn("Successfully removed model: **test-model**", result)
    
    @patch('euclid.models.ModelRegistry')
    def test_remove_model_failure(self, mock_registry_class):
        # Mock registry instance
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        
        # Mock remove_model
        mock_registry.remove_model.return_value = False
        
        result = remove_model("test-model")
        
        # Verify result format
        self.assertIn("Failed to remove model: **test-model**", result)
    
    @patch('euclid.models.ModelRegistry')
    def test_model_details(self, mock_registry_class):
        # Mock registry instance
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        
        # Mock get_model_details
        mock_registry.get_model_details.return_value = {
            "parameters": {"param1": "value1"},
            "template": "Template content",
            "license": "License info"
        }
        
        result = model_details("test-model")
        
        # Verify registry method was called
        mock_registry.get_model_details.assert_called_once_with("test-model")
        
        # Verify result format
        self.assertIn("# Details for test-model", result)
        self.assertIn("## Parameters", result)
        self.assertIn("- **param1**: value1", result)
        self.assertIn("## Template", result)
        self.assertIn("Template content", result)
        self.assertIn("## License", result)
        self.assertIn("License info", result)
    
    @patch('euclid.models.ModelRegistry')
    def test_benchmark_model(self, mock_registry_class):
        # Mock registry instance
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        
        # Mock benchmark_model
        mock_registry.benchmark_model.return_value = {
            "model": "test-model",
            "prompt": "Test prompt",
            "iterations": 3,
            "times": [1.0, 1.1, 0.9],
            "tokens_per_second": [50, 45, 55],
            "avg_time": 1.0,
            "avg_tokens_per_second": 50.0
        }
        
        result = benchmark_model("test-model", "Test prompt", 3)
        
        # Verify registry method was called
        mock_registry.benchmark_model.assert_called_once_with("test-model", "Test prompt", 3)
        
        # Verify result format
        self.assertIn("# Benchmark Results for test-model", result)
        self.assertIn("Prompt: \"Test prompt\"", result)
        self.assertIn("Iterations: 3", result)
        self.assertIn("**Average Response Time**: 1.00 seconds", result)
        self.assertIn("**Average Tokens/Second**: 50.00", result)
        self.assertIn("## Individual Runs", result)


if __name__ == '__main__':
    unittest.main()