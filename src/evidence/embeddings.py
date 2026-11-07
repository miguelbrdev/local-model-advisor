"""Embedding contract and the local Ollama implementation.

The index and the retrieval code depend on the :class:`Embedder` Protocol
only; tests inject a deterministic fake, production uses
``langchain_ollama.OllamaEmbeddings`` against the same local Ollama server
that serves the criteria extractor.  No text ever leaves the machine.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

DEFAULT_EMBEDDING_MODEL = "embeddinggemma"
"""Google's 300M embedding model from the Ollama library (``ollama pull embeddinggemma``)."""


@runtime_checkable
class Embedder(Protocol):
    """Structural contract satisfied by ``OllamaEmbeddings`` and test fakes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def crear_embedder_ollama(modelo: str, base_url: str) -> Any:
    """Build the Ollama embedder lazily, so importing this module costs nothing."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=modelo, base_url=base_url, keep_alive=-1)
