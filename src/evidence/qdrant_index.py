"""Qdrant-backed index of documentation fragments, one payload per family.

Design points:

* One collection (``model_docs``) with a ``familia_id`` payload index.
  Retrieval always filters by family: the question is "what does *this*
  model's card say that matters for the user", never "which model matches
  the text", because candidate selection is the deterministic pipeline's
  job (AGENTS.md: evidence explains, it never ranks).
* Point ids are ``uuid5(familia_id:index)``, so re-indexing a family
  overwrites its fragments in place and nothing depends on insertion order.
* The embedding dimension is learnt from the first vector; a collection
  created with another dimension (model switch) is dropped and recreated,
  with a warning, since its vectors are meaningless for the new model.
* ``QdrantClient(":memory:")`` runs the same code path in tests, so the
  suite exercises real filtering and scoring without a server.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from src.evidence.embeddings import Embedder

logger = logging.getLogger(__name__)

DEFAULT_COLECCION = "model_docs"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"

_NAMESPACE = uuid.UUID("2b5d9f1e-6a0f-4d0c-9b6e-3a7c0e4f1b22")


class IndiceNoDisponibleError(Exception):
    """Qdrant or the embedder could not be reached or answered with an error."""


@dataclass(frozen=True)
class Fragmento:
    """One retrieved documentation fragment."""

    familia_id: str
    texto: str
    score: float
    url: str | None
    indice: int


class IndiceEvidencia:
    """Index and query documentation fragments per family.

    Args:
        client: Qdrant client (server or ``":memory:"``).
        embedder: Anything satisfying :class:`Embedder`.
        coleccion: Collection name.
    """

    def __init__(
        self, client: QdrantClient, embedder: Embedder, coleccion: str = DEFAULT_COLECCION
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._coleccion = coleccion

    @property
    def coleccion(self) -> str:
        return self._coleccion

    # ------------------------------------------------------------------
    # Indexing (external process)
    # ------------------------------------------------------------------

    def asegurar_coleccion(self, dimension: int) -> None:
        """Create the collection for ``dimension`` or recreate it on mismatch."""
        if self._client.collection_exists(self._coleccion):
            actual = self._dimension_actual()
            if actual == dimension:
                return
            logger.warning(
                "Collection %s has dimension %s, embedder produces %s; recreating.",
                self._coleccion, actual, dimension,
            )
            self._client.delete_collection(self._coleccion)
        self._client.create_collection(
            self._coleccion,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        self._client.create_payload_index(
            self._coleccion, field_name="familia_id", field_schema=models.PayloadSchemaType.KEYWORD
        )

    def indexar_familia(
        self, familia_id: str, snapshot_id: str, url: str | None, fragmentos: list[str]
    ) -> int:
        """Replace the family's fragments with ``fragmentos``.

        Returns:
            Number of points written (0 when there is nothing to index; the
            family's previous points are still removed).
        """
        vectores = self._embedder.embed_documents(fragmentos) if fragmentos else []
        if vectores:
            self.asegurar_coleccion(len(vectores[0]))
        if self._client.collection_exists(self._coleccion):
            self._client.delete(
                self._coleccion, points_selector=models.FilterSelector(filter=_filtro(familia_id))
            )
        if not vectores:
            return 0
        puntos = [
            models.PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, f"{familia_id}:{i}")),
                vector=vector,
                payload={
                    "familia_id": familia_id,
                    "snapshot_id": snapshot_id,
                    "url": url,
                    "indice": i,
                    "texto": texto,
                },
            )
            for i, (texto, vector) in enumerate(zip(fragmentos, vectores, strict=True))
        ]
        self._client.upsert(self._coleccion, points=puntos)
        return len(puntos)

    def familias_indexadas(self) -> set[str]:
        """Family ids that have at least one fragment."""
        if not self._client.collection_exists(self._coleccion):
            return set()
        familias: set[str] = set()
        siguiente = None
        while True:
            puntos, siguiente = self._client.scroll(
                self._coleccion, limit=256, offset=siguiente, with_payload=["familia_id"],
                with_vectors=False,
            )
            familias.update(p.payload["familia_id"] for p in puntos if p.payload)
            if siguiente is None:
                return familias

    # ------------------------------------------------------------------
    # Retrieval (request time)
    # ------------------------------------------------------------------

    def buscar(self, consulta: str, familia_id: str, k: int = 2) -> list[Fragmento]:
        """Best ``k`` fragments of ``familia_id`` for ``consulta``.

        Returns an empty list when the family has no fragments.

        Raises:
            IndiceNoDisponibleError: When Qdrant or the embedder fails.
        """
        try:
            if not self._client.collection_exists(self._coleccion):
                return []
            vector = self._embedder.embed_query(consulta)
            respuesta = self._client.query_points(
                self._coleccion, query=vector, limit=k, query_filter=_filtro(familia_id),
                with_payload=True,
            )
        except Exception as exc:
            raise IndiceNoDisponibleError(f"{type(exc).__name__}: {exc}") from exc
        return [
            Fragmento(
                familia_id=familia_id,
                texto=str(p.payload.get("texto", "")),
                score=float(p.score),
                url=p.payload.get("url"),
                indice=int(p.payload.get("indice", 0)),
            )
            for p in respuesta.points
            if p.payload
        ]

    def _dimension_actual(self) -> int | None:
        info = self._client.get_collection(self._coleccion)
        params = info.config.params.vectors
        if isinstance(params, models.VectorParams):
            return params.size
        return None


def _filtro(familia_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="familia_id", match=models.MatchValue(value=familia_id))]
    )
