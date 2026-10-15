"""Embedding provider contracts and adapters."""

from .contracts import Embedder, EmbeddingBatch
from .fastembed import FastEmbedder, LocalEmbeddingBackend, default_query_instruction
from .gemini import GeminiEmbedding2, GoogleGenAIEmbeddingTransport

__all__ = [
    "Embedder",
    "EmbeddingBatch",
    "FastEmbedder",
    "LocalEmbeddingBackend",
    "default_query_instruction",
    "GeminiEmbedding2",
    "GoogleGenAIEmbeddingTransport",
]
