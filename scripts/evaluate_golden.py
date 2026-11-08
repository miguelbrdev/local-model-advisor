"""Run the golden-query suite against the local catalogue and report metrics.

Offline by construction: the catalogue comes from ``runtime/catalog/`` or
the committed bootstrap, benchmark scores come from the committed
``data/benchmarks_bootstrap.json`` only (never from ``runtime/``, whose
optional Artificial Analysis data differs per operator), and VRAM
compatibility is the deterministic parameter-count heuristic, so two runs
on the same snapshots always agree.

Exit codes: ``0`` every case passed (and, with ``--strict-snapshot``, the
catalogue is the snapshot the suite was written for), ``1`` at least one
case failed, ``2`` the suite file is missing or invalid.

Usage::

    python -m scripts.evaluate_golden [--suite FILE] [--catalog-dir DIR]
        [--bootstrap-path FILE] [--benchmarks-path FILE | --no-benchmarks]
        [--json] [--strict-snapshot]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from src.benchmarks.provider import BenchmarksLocales
from src.benchmarks.snapshot import DEFAULT_BENCHMARKS_BOOTSTRAP_PATH
from src.catalog.synced import DEFAULT_RUNTIME_CATALOG_DIR, CatalogoSincronizado
from src.compatibility.heuristica import HeuristicaParametrosProvider
from src.evaluation.golden import DEFAULT_SUITE_PATH, InformeGolden, cargar_suite, evaluar_suite
from src.sync.bootstrap import DEFAULT_BOOTSTRAP_PATH

logger = logging.getLogger("scripts.evaluate_golden")

EXIT_OK = 0
EXIT_CASES_FAILED = 1
EXIT_SUITE_INVALID = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (kept separate so tests can inspect defaults)."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.evaluate_golden",
        description="Evaluate the golden-query suite against the local catalogue snapshot.",
    )
    parser.add_argument(
        "--suite", type=Path, default=DEFAULT_SUITE_PATH,
        help="Golden suite YAML (default: data/golden_queries.yaml).",
    )
    parser.add_argument(
        "--catalog-dir", type=Path, default=DEFAULT_RUNTIME_CATALOG_DIR,
        help="Runtime catalogue directory (default: runtime/catalog).",
    )
    parser.add_argument(
        "--bootstrap-path", type=Path, default=DEFAULT_BOOTSTRAP_PATH,
        help="Bootstrap fallback (default: data/catalog_bootstrap.json).",
    )
    parser.add_argument(
        "--benchmarks-path", type=Path, default=DEFAULT_BENCHMARKS_BOOTSTRAP_PATH,
        help="Benchmark snapshot to rank with (default: data/benchmarks_bootstrap.json).",
    )
    parser.add_argument(
        "--no-benchmarks", action="store_true",
        help="Evaluate with neutral capacidad_medida (no benchmark snapshot).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full report as JSON instead of text."
    )
    parser.add_argument(
        "--strict-snapshot", action="store_true",
        help="Fail when the catalogue snapshot differs from the one the suite was written for.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the suite and return the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    try:
        suite = cargar_suite(args.suite)
    except (OSError, ValidationError) as exc:
        logger.error("Golden suite unusable (%s): %s", args.suite, exc)
        return EXIT_SUITE_INVALID

    catalogo = CatalogoSincronizado(
        runtime_dir=args.catalog_dir, bootstrap_path=args.bootstrap_path
    )
    benchmarks = (
        None
        if args.no_benchmarks
        else BenchmarksLocales(
            runtime_path=args.benchmarks_path, bootstrap_path=args.benchmarks_path
        )
    )
    informe = evaluar_suite(
        suite,
        catalogo=catalogo,
        vram_provider=HeuristicaParametrosProvider(),
        benchmarks=benchmarks,
    )

    if args.json:
        print(informe.model_dump_json(indent=2))
    else:
        print(formatear_informe(informe))

    if informe.aprobados < informe.total:
        return EXIT_CASES_FAILED
    if args.strict_snapshot and not (informe.snapshot_coincide and informe.benchmarks_coinciden):
        return EXIT_CASES_FAILED
    return EXIT_OK


def formatear_informe(informe: InformeGolden) -> str:
    """Human-readable summary: one line per case plus aggregate metrics."""
    lineas = [
        f"Golden suite {informe.version} (written for {informe.snapshot_id_suite}) "
        f"against catalogue {informe.snapshot_id_catalogo}"
        + ("" if informe.snapshot_coincide else "  [SNAPSHOT MISMATCH]"),
        f"benchmarks: {informe.benchmarks_id_usado or 'none'} "
        f"(suite written for {informe.benchmarks_id_suite or 'none'})"
        + ("" if informe.benchmarks_coinciden else "  [BENCHMARKS MISMATCH]"),
        "",
    ]
    for r in informe.resultados:
        estado = "PASS" if r.aprobado else "FAIL"
        lineas.append(f"[{estado}] {r.id}: {r.candidatos_totales} candidates, top={r.top}")
        lineas.extend(f"        - {fallo}" for fallo in r.fallos)
    lineas += [
        "",
        f"cases passed: {informe.aprobados}/{informe.total} "
        f"({informe.tasa_aprobacion:.0%})",
        f"hit rate (expected inclusions found): {_pct(informe.hit_rate)}",
        f"exclusion rate (expected exclusions respected): {_pct(informe.tasa_exclusion)}",
        f"first-result matches: {informe.aciertos_primero}/{informe.esperados_primero}",
    ]
    return "\n".join(lineas)


def _pct(valor: float | None) -> str:
    return "n/a" if valor is None else f"{valor:.0%}"


if __name__ == "__main__":
    sys.exit(main())
