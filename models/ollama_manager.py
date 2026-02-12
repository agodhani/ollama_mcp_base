"""
Ollama Model Manager

Utilities for managing Ollama models including:
- Creating custom models from Modelfiles
- Listing available models
- Pulling models from registry
- Deleting models
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaManager:
    """
    Manages Ollama models and their configurations.
    
    This class provides utilities for working with Ollama models,
    including creating custom models from Modelfiles.
    
    Example Usage:
        manager = OllamaManager()
        
        # Create a custom model
        await manager.create_model(
            name="my-agent",
            modelfile_path="./modelfiles/my-agent.modelfile"
        )
        
        # List available models
        models = await manager.list_models()
        
        # Pull a model from registry
        await manager.pull_model("llama2")
    """
    
    def __init__(self, host: str = "http://localhost:11434"):
        """
        Initialize the Ollama manager.
        
        Args:
            host: Ollama API endpoint
        """
        self.host = host
        logger.info(f"Initialized Ollama manager for {host}")
    
    async def create_model(
        self,
        name: str,
        modelfile_path: str
    ) -> Dict[str, Any]:
        """
        Create a custom model from a Modelfile.
        
        Modelfiles define custom models by:
        - Starting from a base model
        - Setting custom system prompts
        - Configuring parameters (temperature, etc.)
        - Adding custom behavior
        
        Args:
            name: Name for the new model
            modelfile_path: Path to the Modelfile
        
        Returns:
            Creation result information
        
        Example Modelfile:
            FROM llama2
            
            SYSTEM You are a helpful coding assistant specialized in Python.
            
            PARAMETER temperature 0.7
            PARAMETER top_p 0.9
        """
        logger.info(f"Creating model '{name}' from {modelfile_path}")
        
        # Read the Modelfile
        with open(modelfile_path, 'r') as f:
            modelfile_content = f.read()
        
        payload = {
            "name": name,
            "modelfile": modelfile_content
        }
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.host}/api/create",
                    json=payload
                )
                response.raise_for_status()
                
                logger.info(f"Successfully created model: {name}")
                return response.json()
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model information dicts
        """
        logger.info("Listing available models")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                
                result = response.json()
                models = result.get("models", [])
                
                logger.info(f"Found {len(models)} models")
                return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise
    
    async def pull_model(self, name: str) -> Dict[str, Any]:
        """
        Pull a model from the Ollama registry.
        
        Args:
            name: Model name (e.g., "llama2", "mistral")
        
        Returns:
            Pull result information
        """
        logger.info(f"Pulling model: {name}")
        
        payload = {"name": name}
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.host}/api/pull",
                    json=payload
                )
                response.raise_for_status()
                
                logger.info(f"Successfully pulled model: {name}")
                return response.json()
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            raise
    
    async def delete_model(self, name: str) -> Dict[str, Any]:
        """
        Delete a model.
        
        Args:
            name: Model name to delete
        
        Returns:
            Deletion result
        """
        logger.info(f"Deleting model: {name}")
        
        payload = {"name": name}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.host}/api/delete",
                    json=payload
                )
                response.raise_for_status()
                
                logger.info(f"Successfully deleted model: {name}")
                return response.json()
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            raise
    
    async def show_model_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            name: Model name
        
        Returns:
            Model information including parameters, template, etc.
        """
        logger.info(f"Getting info for model: {name}")
        
        payload = {"name": name}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.host}/api/show",
                    json=payload
                )
                response.raise_for_status()
                
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise