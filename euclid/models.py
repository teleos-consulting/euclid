"""Advanced model management for Euclid."""

import os
import json
import time
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import concurrent.futures

import requests
from pydantic import BaseModel, Field
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.table import Table
from rich.box import ROUNDED

from euclid.config import config
from euclid.formatting import console, create_spinner
from euclid.functions import register_function


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    size: Optional[int] = None
    modified: Optional[str] = None
    digest: Optional[str] = None
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ModelRegistry:
    """Registry for managing Ollama models."""
    
    def __init__(self, base_url: Optional[str] = None):
        """Initialize model registry.
        
        Args:
            base_url: Base URL for Ollama API. Defaults to config.ollama_base_url.
        """
        self.base_url = base_url or config.ollama_base_url
        self.cache_file = Path.home() / ".euclid_models.json"
    
    def get_available_models(self, refresh: bool = False) -> List[ModelInfo]:
        """Get list of available models from Ollama.
        
        Args:
            refresh: Force refresh from API instead of using cache.
            
        Returns:
            List of model information.
        """
        if not refresh and self.cache_file.exists():
            try:
                cache_time = self.cache_file.stat().st_mtime
                # Only use cache if it's less than 1 hour old
                if time.time() - cache_time < 3600:
                    with open(self.cache_file, "r") as f:
                        models_data = json.load(f)
                        return [ModelInfo(**model) for model in models_data]
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Fetch from API
        try:
            with create_spinner("Fetching available models"):
                url = f"{self.base_url}/api/tags"
                response = requests.get(url)
                response.raise_for_status()
                
                models_data = response.json().get("models", [])
                
                # Save to cache
                with open(self.cache_file, "w") as f:
                    json.dump(models_data, f)
                
                return [ModelInfo(**model) for model in models_data]
        except Exception as e:
            console.print(f"[error]Error fetching models: {str(e)}[/error]")
            return []
    
    def pull_model(self, model_name: str, show_progress: bool = True) -> ModelInfo:
        """Pull a model from Ollama.
        
        Args:
            model_name: Name of the model to pull.
            show_progress: Whether to show progress bar.
            
        Returns:
            Information about the pulled model.
        """
        if show_progress:
            console.print(f"Pulling model: [bold]{model_name}[/bold]")
            
            # Use subprocess to show progress
            try:
                result = subprocess.run(
                    ["ollama", "pull", model_name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    console.print(f"[error]Error pulling model: {result.stderr}[/error]")
                    raise Exception(result.stderr)
                
                console.print(f"[success]Successfully pulled model: {model_name}[/success]")
            except Exception as e:
                console.print(f"[error]Error pulling model: {str(e)}[/error]")
                raise
        else:
            # Use API directly
            url = f"{self.base_url}/api/pull"
            response = requests.post(url, json={"name": model_name})
            response.raise_for_status()
        
        # Refresh cache and return the pulled model
        models = self.get_available_models(refresh=True)
        for model in models:
            if model.name == model_name:
                return model
        
        # If model not found, create a basic info object
        return ModelInfo(name=model_name)
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from Ollama.
        
        Args:
            model_name: Name of the model to remove.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            with create_spinner(f"Removing model {model_name}"):
                url = f"{self.base_url}/api/delete"
                response = requests.delete(url, json={"name": model_name})
                response.raise_for_status()
                
                # Refresh cache
                self.get_available_models(refresh=True)
                
                return True
        except Exception as e:
            console.print(f"[error]Error removing model: {str(e)}[/error]")
            return False
    
    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Dictionary with model details.
        """
        try:
            with create_spinner(f"Fetching details for model {model_name}"):
                url = f"{self.base_url}/api/show"
                response = requests.post(url, json={"name": model_name})
                response.raise_for_status()
                
                return response.json()
        except Exception as e:
            console.print(f"[error]Error getting model details: {str(e)}[/error]")
            return {}
    
    def benchmark_model(self, model_name: str, prompt: str = "Hello, world!", iterations: int = 5) -> Dict[str, Any]:
        """Benchmark a model's performance.
        
        Args:
            model_name: Name of the model to benchmark.
            prompt: Prompt to use for benchmarking.
            iterations: Number of iterations to run.
            
        Returns:
            Dictionary with benchmark results.
        """
        results = {
            "model": model_name,
            "prompt": prompt,
            "iterations": iterations,
            "times": [],
            "tokens_per_second": [],
            "avg_time": 0,
            "avg_tokens_per_second": 0
        }
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"Benchmarking {model_name}", total=iterations)
            
            for i in range(iterations):
                start_time = time.time()
                
                # Make request to Ollama API
                url = f"{self.base_url}/api/generate"
                response = requests.post(url, json={
                    "model": model_name,
                    "prompt": prompt,
                })
                response.raise_for_status()
                
                elapsed = time.time() - start_time
                response_data = response.json()
                
                # Calculate metrics
                total_tokens = response_data.get("eval_count", 0) + response_data.get("prompt_eval_count", 0)
                tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0
                
                results["times"].append(elapsed)
                results["tokens_per_second"].append(tokens_per_second)
                
                progress.update(task, advance=1)
        
        # Calculate averages
        results["avg_time"] = sum(results["times"]) / iterations
        results["avg_tokens_per_second"] = sum(results["tokens_per_second"]) / iterations
        
        return results
    
    def list_available_models_table(self) -> Table:
        """List available models in a formatted table.
        
        Returns:
            Rich Table with model information.
        """
        models = self.get_available_models()
        
        table = Table(title="Available Models", box=ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="blue")
        
        for model in models:
            size = model.size or 0
            # Convert to human-readable size
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size/1024:.1f} KB"
            elif size < 1024 * 1024 * 1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            else:
                size_str = f"{size/(1024*1024*1024):.1f} GB"
            
            modified = model.modified or "Unknown"
            
            table.add_row(model.name, size_str, modified)
        
        return table


@register_function(
    name="ListModels",
    description="List all available Ollama models."
)
def list_models() -> str:
    """List all available models.
    
    Returns:
        Formatted string with model information.
    """
    registry = ModelRegistry()
    models = registry.get_available_models()
    
    if not models:
        return "No models found. Make sure Ollama is running."
    
    result = "# Available Ollama Models\n\n"
    for model in models:
        size = model.size or 0
        if size < 1024 * 1024:
            size_str = f"{size/1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            size_str = f"{size/(1024*1024):.1f} MB"
        else:
            size_str = f"{size/(1024*1024*1024):.1f} GB"
        
        result += f"- **{model.name}** ({size_str})\n"
    
    return result


@register_function(
    name="PullModel",
    description="Pull a model from the Ollama repository."
)
def pull_model(model_name: str) -> str:
    """Pull a model from Ollama.
    
    Args:
        model_name: Name of the model to pull.
        
    Returns:
        Status message.
    """
    registry = ModelRegistry()
    
    try:
        model = registry.pull_model(model_name)
        
        size = model.size or 0
        if size < 1024 * 1024:
            size_str = f"{size/1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            size_str = f"{size/(1024*1024):.1f} MB"
        else:
            size_str = f"{size/(1024*1024*1024):.1f} GB"
        
        return f"Successfully pulled model: **{model_name}** ({size_str})"
    except Exception as e:
        return f"Error pulling model: {str(e)}"


@register_function(
    name="RemoveModel",
    description="Remove a model from Ollama."
)
def remove_model(model_name: str) -> str:
    """Remove a model from Ollama.
    
    Args:
        model_name: Name of the model to remove.
        
    Returns:
        Status message.
    """
    registry = ModelRegistry()
    
    if registry.remove_model(model_name):
        return f"Successfully removed model: **{model_name}**"
    else:
        return f"Failed to remove model: **{model_name}**"


@register_function(
    name="ModelDetails",
    description="Get detailed information about a model."
)
def model_details(model_name: str) -> str:
    """Get detailed information about a model.
    
    Args:
        model_name: Name of the model.
        
    Returns:
        Formatted string with model details.
    """
    registry = ModelRegistry()
    details = registry.get_model_details(model_name)
    
    if not details:
        return f"No details found for model: **{model_name}**"
    
    result = f"# Details for {model_name}\n\n"
    
    # Format model parameters
    parameters = details.get("parameters", {})
    if parameters:
        result += "## Parameters\n\n"
        for key, value in parameters.items():
            result += f"- **{key}**: {value}\n"
        result += "\n"
    
    # Format model template
    template = details.get("template", "")
    if template:
        result += "## Template\n\n"
        result += f"```\n{template}\n```\n\n"
    
    # Format license information
    license_info = details.get("license", "")
    if license_info:
        result += f"## License\n\n{license_info}\n\n"
    
    return result


@register_function(
    name="BenchmarkModel",
    description="Benchmark a model's performance with a sample prompt."
)
def benchmark_model(model_name: str, prompt: Optional[str] = None, iterations: int = 3) -> str:
    """Benchmark a model's performance.
    
    Args:
        model_name: Name of the model to benchmark.
        prompt: Prompt to use for benchmarking. Defaults to a standard greeting.
        iterations: Number of iterations to run. Defaults to 3.
        
    Returns:
        Benchmark results.
    """
    registry = ModelRegistry()
    
    test_prompt = prompt or "Generate a short poem about AI assistants."
    
    try:
        results = registry.benchmark_model(model_name, test_prompt, iterations)
        
        avg_time = results["avg_time"]
        avg_tokens_per_second = results["avg_tokens_per_second"]
        
        result = f"# Benchmark Results for {model_name}\n\n"
        result += f"Prompt: \"{test_prompt}\"\n\n"
        result += f"Iterations: {iterations}\n\n"
        result += f"**Average Response Time**: {avg_time:.2f} seconds\n\n"
        result += f"**Average Tokens/Second**: {avg_tokens_per_second:.2f}\n\n"
        
        # Add individual runs
        result += "## Individual Runs\n\n"
        for i, (time_taken, tokens_per_sec) in enumerate(zip(results["times"], results["tokens_per_second"])):
            result += f"Run {i+1}: {time_taken:.2f}s ({tokens_per_sec:.2f} tokens/s)\n"
        
        return result
    except Exception as e:
        return f"Error benchmarking model: {str(e)}"
