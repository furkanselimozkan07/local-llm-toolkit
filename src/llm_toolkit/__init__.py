"""local-llm-toolkit: helpers for working with local LLMs through Ollama."""

from .client import ChatMessage, OllamaClient, OllamaError

__all__ = ["ChatMessage", "OllamaClient", "OllamaError"]
__version__ = "0.1.0"
