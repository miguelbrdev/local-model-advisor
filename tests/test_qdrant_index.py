"""Tests for src.evidence.qdrant_index against the in-memory Qdrant client.

The embedder is a deterministic fake (letter-frequency vectors), so
similarity is real cosine similarity over real Qdrant filtering, without a
server or a model.
"""

from __future__ import annotations

import math

import pytest
from qdrant_client import QdrantClient

from src.evidence.embeddings import Embedder
from src.evidence.qdrant_index import (
    DEFAULT_COLECCION,
    Fragmento,
    IndiceEvidencia,
    IndiceNoDisponibleError,
)


class FakeEmbedder:
    """Letter-frequency vectors over a fixed alphabet; dimension configurable."""

    def __init__(self, dimension: int = 26) -> None:
        self.dimension = dimension
        self.consultas: list[str] = []

    def _vec(self, texto: str) -> list[float]:
        v = [0.0] * self.dimension
        for ch in texto.lower():
            if "a" <= ch <= "z":
                v[(ord(ch) - 97) % self.dimension] += 1.0
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.consultas.append(text)
        return self._vec(text)


class _BrokenClient:
    def collection_exists(self, name: str) -> bool:
        raise ConnectionError("qdrant down")


@pytest.fixture
def indice() -> IndiceEvidencia:
    return IndiceEvidencia(QdrantClient(":memory:"), FakeEmbedder())


class TestIndexar:
    def test_fake_satisfies_protocol(self) -> None:
        assert isinstance(FakeEmbedder(), Embedder)

    def test_index_and_search_filters_by_family(self, indice: IndiceEvidencia) -> None:
        indice.indexar_familia("coder", "snap-1", "u1", ["python code generation", "unit tests"])
        indice.indexar_familia("chatter", "snap-1", "u2", ["python code generation"])

        hits = indice.buscar("python code", "coder", k=5)
        assert [h.familia_id for h in hits] == ["coder", "coder"]
        assert hits[0].texto == "python code generation"
        assert hits[0].url == "u1" and hits[0].indice == 0
        assert isinstance(hits[0], Fragmento)
        assert hits[0].score >= hits[1].score

    def test_reindex_replaces_previous_fragments(self, indice: IndiceEvidencia) -> None:
        indice.indexar_familia("f", "snap-1", None, ["alpha", "beta", "gamma"])
        n = indice.indexar_familia("f", "snap-2", None, ["delta"])
        assert n == 1
        hits = indice.buscar("delta", "f", k=10)
        assert [h.texto for h in hits] == ["delta"]

    def test_empty_fragments_remove_family(self, indice: IndiceEvidencia) -> None:
        indice.indexar_familia("f", "snap-1", None, ["alpha"])
        assert indice.indexar_familia("f", "snap-1", None, []) == 0
        assert indice.buscar("alpha", "f") == []
        assert indice.familias_indexadas() == set()

    def test_point_ids_are_deterministic(self) -> None:
        a = IndiceEvidencia(QdrantClient(":memory:"), FakeEmbedder())
        b = IndiceEvidencia(QdrantClient(":memory:"), FakeEmbedder())
        a.indexar_familia("f", "s", None, ["one", "two"])
        b.indexar_familia("f", "s", None, ["one", "two"])
        ids_a = {p.id for p in a._client.scroll(DEFAULT_COLECCION, limit=10)[0]}
        ids_b = {p.id for p in b._client.scroll(DEFAULT_COLECCION, limit=10)[0]}
        assert ids_a == ids_b and len(ids_a) == 2

    def test_familias_indexadas(self, indice: IndiceEvidencia) -> None:
        indice.indexar_familia("a", "s", None, ["x" * 10])
        indice.indexar_familia("b", "s", None, ["y" * 10, "z" * 10])
        assert indice.familias_indexadas() == {"a", "b"}

    def test_dimension_change_recreates_collection(self) -> None:
        client = QdrantClient(":memory:")
        IndiceEvidencia(client, FakeEmbedder(26)).indexar_familia("f", "s", None, ["abc"])
        IndiceEvidencia(client, FakeEmbedder(8)).indexar_familia("g", "s", None, ["abc"])
        info = client.get_collection(DEFAULT_COLECCION)
        assert info.config.params.vectors.size == 8
        # the old 26-d family is gone with the recreated collection
        assert IndiceEvidencia(client, FakeEmbedder(8)).familias_indexadas() == {"g"}


class TestBuscar:
    def test_no_collection_returns_empty(self, indice: IndiceEvidencia) -> None:
        assert indice.buscar("anything", "f") == []

    def test_unknown_family_returns_empty(self, indice: IndiceEvidencia) -> None:
        indice.indexar_familia("f", "s", None, ["hello world"])
        assert indice.buscar("hello", "other") == []

    def test_k_limits_results(self, indice: IndiceEvidencia) -> None:
        indice.indexar_familia("f", "s", None, [f"fragment {i}" for i in range(5)])
        assert len(indice.buscar("fragment", "f", k=2)) == 2

    def test_query_text_reaches_embedder(self, indice: IndiceEvidencia) -> None:
        indice.indexar_familia("f", "s", None, ["hello"])
        indice.buscar("the query", "f")
        assert indice._embedder.consultas == ["the query"]

    def test_calentar(self, indice: IndiceEvidencia) -> None:
        assert indice.calentar() is True
        assert indice._embedder.consultas == ["warm-up"]

        class _Broken:
            def embed_documents(self, texts):
                raise ConnectionError("down")

            def embed_query(self, text):
                raise ConnectionError("down")

        assert IndiceEvidencia(QdrantClient(":memory:"), _Broken()).calentar() is False

    def test_backend_failure_is_explicit(self) -> None:
        indice = IndiceEvidencia(_BrokenClient(), FakeEmbedder())  # type: ignore[arg-type]
        with pytest.raises(IndiceNoDisponibleError, match="ConnectionError"):
            indice.buscar("x", "f")
