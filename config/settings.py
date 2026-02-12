"""
Configuration Settings

Centralized configuration for MCP servers.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class OllamaConfig:
    """Ollama-specific configuration."""
    host: str = "http://localhost:11434"
    default_model: str = "llama3:8b"
    timeout: int = 120
    temperature: float = 0.7


@dataclass
class ServerConfig:
    """Server configuration."""
    name: str = "base-mcp-server"
    version: str = "1.0.0"
    log_level: str = "INFO"


@dataclass
class Settings:
    """
    Global settings for the MCP server.
    
    Settings can be overridden via environment variables:
    - OLLAMA_HOST
    - OLLAMA_MODEL
    - SERVER_NAME
    - LOG_LEVEL
    """
    
    def __init__(self):
        self.ollama = OllamaConfig(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            default_model=os.getenv("OLLAMA_MODEL", "llama2"),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
        )
        
        self.server = ServerConfig(
            name=os.getenv("SERVER_NAME", "base-mcp-server"),
            version=os.getenv("SERVER_VERSION", "1.0.0"),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )


# Global settings instance
settings = Settings()