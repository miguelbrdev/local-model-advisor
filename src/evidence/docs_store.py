"""Local store of downloaded model documentation.

One JSON file per family under ``runtime/docs/`` (gitignored), written
atomically.  Keeping the raw text locally means indexing, re-indexing and
switching embedding models never touch Hugging Face again, and the fetch
status of every family (``ok``, ``gated``, ...) is preserved so the API can
say why a candidate has no evidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from src.evidence.hf_readme import EstadoDescarga, ReadmeDescargado

logger = logging.getLogger(__name__)

DEFAULT_DOCS_DIR = Path(__file__).resolve().parents[2] / "runtime" / "docs"


class DocumentoModelo(BaseModel):
    """Stored documentation (or the reason there is none) for one family."""

    familia_id: str = Field(min_length=1)
    url_repo: str
    url_readme: str | None = None
    estado: EstadoDescarga
    texto: str = ""
    base_model: str | None = None
    redirigido_desde: str | None = None
    detalle: str = ""
    fecha_descarga: datetime
    hash_texto: str = ""

    @classmethod
    def desde_descarga(
        cls, familia_id: str, descarga: ReadmeDescargado, *, fecha: datetime | None = None
    ) -> DocumentoModelo:
        return cls(
            familia_id=familia_id,
            url_repo=descarga.url_repo,
            url_readme=descarga.url_readme,
            estado=descarga.estado,
            texto=descarga.texto,
            base_model=descarga.base_model,
            redirigido_desde=descarga.redirigido_desde,
            detalle=descarga.detalle,
            fecha_descarga=fecha or datetime.now(UTC),
            hash_texto=hashlib.sha256(descarga.texto.encode("utf-8")).hexdigest(),
        )

    @property
    def tiene_texto(self) -> bool:
        return self.estado == "ok" and bool(self.texto.strip())


class AlmacenDocumentos:
    """Directory-backed store of :class:`DocumentoModelo` files.

    Args:
        directorio: Target directory; created on first write.  Defaults to
            ``runtime/docs/`` at the project root.
    """

    def __init__(self, directorio: Path | None = None) -> None:
        self._dir = directorio if directorio is not None else DEFAULT_DOCS_DIR

    @property
    def directorio(self) -> Path:
        return self._dir

    def _ruta(self, familia_id: str) -> Path:
        return self._dir / f"{familia_id}.json"

    def guardar(self, doc: DocumentoModelo) -> Path:
        """Write ``doc`` atomically (temp file + ``os.replace``)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        destino = self._ruta(doc.familia_id)
        temporal = destino.with_name(destino.name + ".tmp")
        try:
            temporal.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporal, destino)
        finally:
            if temporal.exists():
                temporal.unlink(missing_ok=True)
        return destino

    def cargar(self, familia_id: str) -> DocumentoModelo | None:
        """Return the stored document, or ``None`` if missing or corrupt."""
        ruta = self._ruta(familia_id)
        if not ruta.is_file():
            return None
        try:
            return DocumentoModelo.model_validate_json(ruta.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            logger.warning("Ignoring unreadable document %s: %s", ruta, exc)
            return None

    def listar(self) -> list[DocumentoModelo]:
        """Every readable document, sorted by family id."""
        if not self._dir.is_dir():
            return []
        docs = (self.cargar(p.stem) for p in sorted(self._dir.glob("*.json")))
        return [d for d in docs if d is not None]

    def resumen(self) -> dict[str, int]:
        """Count of stored documents per fetch status."""
        return dict(Counter(d.estado for d in self.listar()))


def cargar_json_seguro(ruta: Path) -> dict | None:
    """Small helper for callers that only need the raw mapping."""
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
