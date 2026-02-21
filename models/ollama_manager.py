"""
Ollama Model Manager

Utilities for managing Ollama models including:
- Checking Ollama availability
- Creating custom models from Modelfiles
- Listing available models
- Pulling models from registry
- Deleting models
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaManager:
    """
    Manages Ollama models and their configurations.

    Example Usage:
        manager = OllamaManager()

        # Check Ollama is running
        if not await manager.health_check():
            raise RuntimeError("Ollama is not running")

        # List available models
        models = await manager.list_models()

        # Pull a model from registry
        await manager.pull_model("llama3.2")

        # Create a custom model from a Modelfile
        await manager.create_model(
            name="my-agent",
            modelfile_path="./modelfiles/my-agent.modelfile"
        )
    """

    def __init__(self, host: str = "http://localhost:11434"):
        """
        Initialize the Ollama manager.

        Args:
            host: Ollama API endpoint
        """
        self.host = host
        logger.info(f"Initialized Ollama manager for {host}")

    async def health_check(self) -> bool:
        """
        Check if Ollama is reachable and running.

        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def create_model(
        self,
        name: str,
        modelfile_path: str,
    ) -> Dict[str, Any]:
        """
        Create a custom model from a Modelfile.

        Modelfiles define custom models by specifying a base model,
        system prompt, and parameters.

        Args:
            name: Name for the new model
            modelfile_path: Path to the Modelfile

        Returns:
            Creation result from the Ollama API

        Example Modelfile:
            FROM llama3.2

            SYSTEM You are a helpful coding assistant specialized in Python.

            PARAMETER temperature 0.7
            PARAMETER top_p 0.9
        """
        logger.info(f"Creating model '{name}' from {modelfile_path}")

        with open(modelfile_path, "r") as f:
            modelfile_content = f.read()

        payload = {"name": name, "modelfile": modelfile_content}

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.host}/api/create",
                    json=payload,
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
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                logger.info(f"Found {len(models)} models")
                return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise

    async def pull_model(self, name: str) -> Dict[str, Any]:
        """
        Pull a model from the Ollama registry.

        Args:
            name: Model name (e.g., "llama3.2", "mistral")

        Returns:
            Pull result from the Ollama API
        """
        logger.info(f"Pulling model: {name}")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.host}/api/pull",
                    json={"name": name},
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
            Deletion result from the Ollama API
        """
        logger.info(f"Deleting model: {name}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.host}/api/delete",
                    json={"name": name},
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

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.host}/api/show",
                    json={"name": name},
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise
