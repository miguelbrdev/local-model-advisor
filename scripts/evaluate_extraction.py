"""Measure free-text criteria extraction against the golden extraction suite.

Needs the local Ollama model (``LMA_OLLAMA_MODEL`` / ``LMA_OLLAMA_BASE_URL``
or the flags below); nothing leaves the machine.  Exit codes: ``0`` pass
rate at or above ``--min-aprobacion``, ``1`` below it or the model could
not be reached, ``2`` the suite file is unusable.

Usage::

    python -m scripts.evaluate_extraction [--suite FILE] [--model TAG]
        [--ollama-base-url URL] [--min-aprobacion 0.8] [--json]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from src.chains.extraccion_criterios import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    ExtractorCompuesto,
    ExtractorJson,
    ExtractorOllama,
)
from src.evaluation.extraccion import (
    DEFAULT_SUITE_PATH,
    InformeExtraccion,
    cargar_suite,
    evaluar_extraccion,
)

logger = logging.getLogger("scripts.evaluate_extraction")

EXIT_OK = 0
EXIT_BELOW_THRESHOLD = 1
EXIT_SUITE_INVALID = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.evaluate_extraction",
        description="Evaluate free-text criteria extraction with the local Ollama model.",
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--model", default=os.environ.get("LMA_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("LMA_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
    )
    parser.add_argument("--min-aprobacion", type=float, default=0.8,
                        help="Minimum pass rate for exit code 0 (default 0.8).")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    return parser


def main(argv: Sequence[str] | None = None, *, extractor=None) -> int:
    """Run the suite; ``extractor`` is injectable so tests stay offline."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        suite = cargar_suite(args.suite)
    except (OSError, ValidationError) as exc:
        logger.error("Extraction suite unusable (%s): %s", args.suite, exc)
        return EXIT_SUITE_INVALID

    if extractor is None:
        extractor = ExtractorCompuesto(
            ExtractorJson(), ExtractorOllama(modelo=args.model, base_url=args.ollama_base_url)
        )
    informe = evaluar_extraccion(suite, extractor)

    if args.json:
        print(informe.model_dump_json(indent=2))
    else:
        print(formatear_informe(informe, args.model))
    if informe.errores_backend == informe.total:
        return EXIT_BELOW_THRESHOLD
    return EXIT_OK if informe.tasa_aprobacion >= args.min_aprobacion else EXIT_BELOW_THRESHOLD


def formatear_informe(informe: InformeExtraccion, modelo: str) -> str:
    lineas = [f"Extraction suite {informe.version} with model {modelo}", ""]
    for r in informe.resultados:
        estado = "PASS" if r.aprobado else "FAIL"
        lineas.append(f"[{estado}] {r.id} ({r.segundos:.1f}s)")
        if r.error:
            lineas.append(f"        - error: {r.error}")
        for campo, detalle in r.campos_fallidos.items():
            lineas.append(
                f"        - {campo}: expected {detalle['esperado']!r}, got {detalle['obtenido']!r}"
            )
    lineas += [
        "",
        f"cases passed: {informe.aprobados}/{informe.total} ({informe.tasa_aprobacion:.0%})",
        "field accuracy: " + ", ".join(
            f"{c}={v:.0%}" for c, v in informe.exactitud_por_campo.items()
        ),
        f"backend errors: {informe.errores_backend}",
        f"mean latency: {informe.segundos_medio:.1f}s",
    ]
    return "\n".join(lineas)


if __name__ == "__main__":
    sys.exit(main())
