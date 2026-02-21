"""
Configuration Settings

Centralized configuration for MCP servers. All values can be overridden
via environment variables.
"""

import os
from dataclasses import dataclass


@dataclass
class OllamaConfig:
    """Ollama-specific configuration."""
    host: str = "http://localhost:11434"
    default_model: str = "llama3.2"
    timeout: int = 120
    temperature: float = 0.7


@dataclass
class ServerConfig:
    """Server configuration."""
    name: str = "base-mcp-server"
    version: str = "1.0.0"
    log_level: str = "INFO"


class Settings:
    """
    Global settings for the MCP server.

    Override via environment variables:
        OLLAMA_HOST        - Ollama API endpoint (default: http://localhost:11434)
        OLLAMA_MODEL       - Model name (default: llama3.2)
        OLLAMA_TIMEOUT     - Request timeout in seconds (default: 120)
        OLLAMA_TEMPERATURE - Sampling temperature (default: 0.7)
        SERVER_NAME        - MCP server name (default: base-mcp-server)
        SERVER_VERSION     - Server version string (default: 1.0.0)
        LOG_LEVEL          - Logging level (default: INFO)
    """

    def __init__(self):
        self.ollama = OllamaConfig(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            default_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
        )

        self.server = ServerConfig(
            name=os.getenv("SERVER_NAME", "base-mcp-server"),
            version=os.getenv("SERVER_VERSION", "1.0.0"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# Global settings instance
settings = Settings()
