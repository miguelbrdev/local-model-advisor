"""Fetch model documentation and index it in Qdrant for explanatory evidence.

Runs outside the API (after a catalogue sync, or on demand):

1. **fetch** — for every family of the active catalogue snapshot without a
   stored document (or all of them with ``--refetch``), download the
   Hugging Face README and store the text or the reason it is missing in
   ``runtime/docs/``.
2. **index** — split every stored README into fragments, embed them with
   the local Ollama embedding model and upsert them into the Qdrant
   collection, one family at a time.

Exit codes: ``0`` done, ``1`` indexing failed (Qdrant or the embedder
unreachable; stored documents are kept), ``2`` no catalogue snapshot
resolves.  Fetch failures never abort the run: they are stored per family
as ``gated`` / ``not_found`` / ``error`` and summarised.

Defaults for the Qdrant URL, the embedding model and the Ollama server come
from the same ``LMA_QDRANT_URL`` / ``LMA_EMBEDDING_MODEL`` /
``LMA_OLLAMA_BASE_URL`` variables the API reads, so inside Compose a plain
``python -m scripts.index_docs --skip-fetch`` targets the right services.

Usage::

    python -m scripts.index_docs [--skip-fetch] [--skip-index] [--refetch]
        [--qdrant-url URL] [--embedding-model NAME] [--ollama-base-url URL]
        [--docs-dir DIR] [--catalog-dir DIR] [--bootstrap-path FILE] [-v]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from src.catalog.protocol import CatalogUnavailableError
from src.catalog.synced import DEFAULT_RUNTIME_CATALOG_DIR, CatalogoSincronizado
from src.chains.extraccion_criterios import DEFAULT_OLLAMA_BASE_URL
from src.evidence.chunker import trocear
from src.evidence.docs_store import DEFAULT_DOCS_DIR, AlmacenDocumentos, DocumentoModelo
from src.evidence.embeddings import DEFAULT_EMBEDDING_MODEL, crear_embedder_ollama
from src.evidence.hf_readme import ReadmeDescargado, descargar_readme
from src.evidence.qdrant_index import DEFAULT_COLECCION, DEFAULT_QDRANT_URL, IndiceEvidencia
from src.sync.bootstrap import DEFAULT_BOOTSTRAP_PATH

logger = logging.getLogger("scripts.index_docs")

EXIT_OK = 0
EXIT_INDEX_FAILED = 1
EXIT_NO_CATALOGUE = 2

USER_AGENT = "local-model-advisor/0.1 (+https://github.com/miguelbrdev/local-model-advisor)"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (kept separate so tests can inspect defaults)."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.index_docs",
        description="Download model READMEs and index them in Qdrant as explanatory evidence.",
    )
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_RUNTIME_CATALOG_DIR)
    parser.add_argument("--bootstrap-path", type=Path, default=DEFAULT_BOOTSTRAP_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR,
                        help="Where READMEs are stored (default: runtime/docs).")
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("LMA_QDRANT_URL", DEFAULT_QDRANT_URL)
    )
    parser.add_argument("--coleccion", default=DEFAULT_COLECCION)
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("LMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("LMA_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout (seconds).")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Index what is already stored; do not contact Hugging Face.")
    parser.add_argument("--skip-index", action="store_true",
                        help="Only download and store READMEs.")
    parser.add_argument("--refetch", action="store_true",
                        help="Download every README again, ignoring stored ones.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    http_client: httpx.Client | None = None,
    embedder: Any | None = None,
    qdrant: Any | None = None,
) -> int:
    """Run fetch and/or index and return the exit code.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
        http_client: Injected HTTP client (tests use ``httpx.MockTransport``).
        embedder: Injected :class:`~src.evidence.embeddings.Embedder`.
        qdrant: Injected Qdrant client (tests pass ``QdrantClient(":memory:")``).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    catalogo = CatalogoSincronizado(
        runtime_dir=args.catalog_dir, bootstrap_path=args.bootstrap_path
    )
    try:
        familias = catalogo.cargar()
        snapshot_id = catalogo.snapshot_id or ""
    except CatalogUnavailableError as exc:
        logger.error("No catalogue snapshot available: %s", exc)
        return EXIT_NO_CATALOGUE

    almacen = AlmacenDocumentos(args.docs_dir)

    if not args.skip_fetch:
        resumen = _fetch(familias, almacen, http_client, args.timeout, args.refetch)
        print("fetch: " + " ".join(f"{k}={v}" for k, v in sorted(resumen.items())))

    if not args.skip_index:
        try:
            indexadas, fragmentos, sin_texto = _index(
                almacen, snapshot_id, args, embedder=embedder, qdrant=qdrant
            )
        except Exception as exc:  # Qdrant or embedder unreachable, model missing...
            logger.error("Indexing failed (%s: %s); stored documents are kept.",
                         type(exc).__name__, exc)
            return EXIT_INDEX_FAILED
        print(
            f"index: snapshot={snapshot_id} familias={indexadas} fragmentos={fragmentos} "
            f"sin_texto={sin_texto}"
        )

    return EXIT_OK


def _fetch(
    familias, almacen: AlmacenDocumentos, http_client, timeout: float, refetch: bool
) -> dict[str, int]:
    """Download missing (or all) READMEs into the store; returns counts by state."""
    resumen: Counter[str] = Counter()
    cliente = http_client or httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})
    try:
        for familia in familias:
            if not refetch and almacen.cargar(familia.familia_id) is not None:
                resumen["cached"] += 1
                continue
            url = familia.license_url  # the catalogue record's repository URL
            if not url:
                descarga = ReadmeDescargado("", None, "not_hf", detalle="no url in catalogue")
            else:
                descarga = descargar_readme(url, client=cliente)
            almacen.guardar(DocumentoModelo.desde_descarga(familia.familia_id, descarga))
            resumen[descarga.estado] += 1
            logger.info("%s: %s %s", familia.familia_id, descarga.estado, descarga.detalle)
    finally:
        if http_client is None:
            cliente.close()
    return dict(resumen)


def _index(almacen: AlmacenDocumentos, snapshot_id: str, args, *, embedder, qdrant):
    """Embed and upsert every stored README; returns (families, fragments, without text)."""
    if embedder is None:
        embedder = crear_embedder_ollama(args.embedding_model, args.ollama_base_url)
    if qdrant is None:
        from qdrant_client import QdrantClient

        qdrant = QdrantClient(url=args.qdrant_url)
    indice = IndiceEvidencia(qdrant, embedder, args.coleccion)

    indexadas = fragmentos = sin_texto = 0
    for doc in almacen.listar():
        if not doc.tiene_texto:
            sin_texto += 1
            continue
        trozos = trocear(doc.texto)
        n = indice.indexar_familia(doc.familia_id, snapshot_id, doc.url_readme, trozos)
        logger.info("%s: %d fragments", doc.familia_id, n)
        if n:
            indexadas += 1
            fragmentos += n
        else:
            sin_texto += 1
    return indexadas, fragmentos, sin_texto


if __name__ == "__main__":
    sys.exit(main())
