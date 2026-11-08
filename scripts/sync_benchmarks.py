"""Synchronise benchmark scores into ``runtime/benchmarks/current.json``.

Companion of ``scripts/sync_catalog.py`` for the measured capability
signal; meant for the same operating-system scheduler (every 21 days) and
never run inside the API process.

Sources:

* Epoch AI (mandatory, public, CC BY 4.0).
* Artificial Analysis (optional): used only when the environment variable
  ``ARTIFICIAL_ANALYSIS_API_KEY`` is set and ``--skip-artificial-analysis``
  is not passed.  Its data never leaves ``runtime/``.

``--promote-bootstrap`` copies the redistributable part of the new
snapshot (Epoch AI only) to ``data/benchmarks_bootstrap.json`` so a fresh
clone ranks with real scores; committing that file is a human decision.

Exit codes: ``0`` snapshot in place (new or unchanged), ``1`` synchronisation
failed (previous snapshot kept), ``2`` mapping file invalid or bootstrap
promotion failed.

Usage::

    python -m scripts.sync_benchmarks [--output FILE] [--mapping FILE]
        [--timeout SECONDS] [--skip-artificial-analysis]
        [--promote-bootstrap] [--bootstrap-path FILE] [-v]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
from pydantic import ValidationError

from src.benchmarks.artificial_analysis import VARIABLE_CLAVE, FuenteArtificialAnalysis
from src.benchmarks.epoch import FuenteEpoch
from src.benchmarks.fuentes import FuenteBenchmarks
from src.benchmarks.mapping import DEFAULT_MAPPING_PATH, MapeoInvalidoError, cargar_mapeo
from src.benchmarks.snapshot import (
    DEFAULT_BENCHMARKS_BOOTSTRAP_PATH,
    DEFAULT_RUNTIME_BENCHMARKS_PATH,
    read_snapshot,
    write_snapshot,
)
from src.benchmarks.sync import sync_benchmarks

logger = logging.getLogger("scripts.sync_benchmarks")

EXIT_OK = 0
EXIT_SYNC_FAILED = 1
EXIT_POST_STEP_FAILED = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (kept separate so tests can inspect defaults)."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.sync_benchmarks",
        description="Download benchmark scores from the approved sources into a local snapshot.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_RUNTIME_BENCHMARKS_PATH,
        help="Snapshot file to write (default: runtime/benchmarks/current.json).",
    )
    parser.add_argument(
        "--mapping", type=Path, default=DEFAULT_MAPPING_PATH,
        help="Reviewed mapping table (default: data/benchmark_mapping.yaml).",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="HTTP timeout in seconds (default: 60)."
    )
    parser.add_argument(
        "--skip-artificial-analysis", action="store_true",
        help=f"Ignore {VARIABLE_CLAVE} and use Epoch AI only.",
    )
    parser.add_argument(
        "--promote-bootstrap", action="store_true",
        help="Also write the redistributable part (Epoch AI only) to the versioned bootstrap.",
    )
    parser.add_argument(
        "--bootstrap-path", type=Path, default=DEFAULT_BENCHMARKS_BOOTSTRAP_PATH,
        help="Destination of --promote-bootstrap (default: data/benchmarks_bootstrap.json).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Log at DEBUG level instead of INFO."
    )
    return parser


def elegir_fuentes(
    *, skip_artificial_analysis: bool, env: Mapping[str, str] | None = None
) -> list[FuenteBenchmarks]:
    """Epoch AI always; Artificial Analysis only with a key and unless skipped."""
    fuentes: list[FuenteBenchmarks] = [FuenteEpoch()]
    if skip_artificial_analysis:
        logger.info("Artificial Analysis skipped by flag.")
        return fuentes
    aa = FuenteArtificialAnalysis.desde_entorno(env)
    if aa is None:
        logger.info("%s not set; using Epoch AI only.", VARIABLE_CLAVE)
    else:
        fuentes.append(aa)
    return fuentes


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run one synchronisation cycle and return the process exit code.

    Args:
        argv: Command-line arguments without the program name.
        transport: Injected HTTP transport (tests); ``None`` in production.
        env: Environment to read the API key from; defaults to ``os.environ``.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        mapeo = cargar_mapeo(args.mapping)
    except MapeoInvalidoError as exc:
        logger.error("%s", exc)
        return EXIT_POST_STEP_FAILED

    fuentes = elegir_fuentes(
        skip_artificial_analysis=args.skip_artificial_analysis,
        env=env if env is not None else os.environ,
    )
    resultado = sync_benchmarks(
        args.output, fuentes, mapeo, timeout=args.timeout, transport=transport
    )
    if not resultado.ok:
        logger.error(
            "Benchmark synchronisation failed (%s). The previous snapshot at %s is kept.",
            resultado.error, args.output,
        )
        return EXIT_SYNC_FAILED

    print(
        f"snapshot={resultado.snapshot_id} unchanged={resultado.unchanged} "
        f"sources={','.join(resultado.fuentes_ok)} "
        f"families_with_data={resultado.familias_con_datos} "
        f"unmatched={len(resultado.sin_coincidencia)}"
    )
    for fuente, motivo in resultado.fuentes_fallidas.items():
        print(f"skipped_source={fuente} reason={motivo}")
    for linea in resultado.sin_coincidencia:
        print(f"unmatched: {linea}")

    if args.promote_bootstrap:
        snapshot = read_snapshot(args.output)
        if snapshot is None:
            logger.error("Cannot promote: %s does not validate.", args.output)
            return EXIT_POST_STEP_FAILED
        try:
            redistribuible = snapshot.solo_redistribuible()
            write_snapshot(redistribuible, args.bootstrap_path)
        except (OSError, ValidationError) as exc:
            logger.error("Bootstrap promotion failed: %s", exc)
            return EXIT_POST_STEP_FAILED
        print(
            f"bootstrap={args.bootstrap_path} snapshot={redistribuible.meta.snapshot_id} "
            f"sources={','.join(f.nombre for f in redistribuible.meta.fuentes)}"
        )

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
