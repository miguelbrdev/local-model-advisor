"""Tests for src.compatibility.cache — TTL cache of measured VRAM verdicts."""

from __future__ import annotations

import pytest

from src.compatibility.cache import CacheCompatibilidad
from src.models.schemas import CompatibilityResult


def _medido(fid: str = "m") -> CompatibilityResult:
    return CompatibilityResult(
        modelo_id=fid, variante="7B", compatible=True, fuente="canirun_ai", vram_required_gb=5.0
    )


class _Reloj:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TestCache:
    def test_hit_within_ttl(self) -> None:
        reloj = _Reloj()
        cache = CacheCompatibilidad(ttl_segundos=100, reloj=reloj)
        cache.guardar("m", 12.0, _medido())
        reloj.t = 99
        assert cache.obtener("m", 12.0) == _medido()
        assert (cache.aciertos, cache.fallos) == (1, 0)

    def test_expires_at_ttl(self) -> None:
        reloj = _Reloj()
        cache = CacheCompatibilidad(ttl_segundos=100, reloj=reloj)
        cache.guardar("m", 12.0, _medido())
        reloj.t = 100
        assert cache.obtener("m", 12.0) is None
        assert len(cache) == 0

    def test_key_rounds_vram(self) -> None:
        cache = CacheCompatibilidad(reloj=_Reloj())
        cache.guardar("m", 12.004, _medido())
        assert cache.obtener("m", 12.0) is not None
        assert cache.obtener("m", 12.5) is None
        assert cache.obtener("other", 12.0) is None

    @pytest.mark.parametrize("fuente", ["heuristica_propia", "no_disponible"])
    def test_only_measured_results_are_stored(self, fuente: str) -> None:
        cache = CacheCompatibilidad(reloj=_Reloj())
        cache.guardar("m", 12.0, _medido().model_copy(update={"fuente": fuente}))
        assert len(cache) == 0

    def test_ttl_zero_disables(self) -> None:
        cache = CacheCompatibilidad(ttl_segundos=0, reloj=_Reloj())
        cache.guardar("m", 12.0, _medido())
        assert cache.obtener("m", 12.0) is None and len(cache) == 0

    def test_eviction_is_oldest_first(self) -> None:
        cache = CacheCompatibilidad(max_entradas=2, reloj=_Reloj())
        cache.guardar("a", 1.0, _medido("a"))
        cache.guardar("b", 1.0, _medido("b"))
        cache.guardar("c", 1.0, _medido("c"))
        assert cache.obtener("a", 1.0) is None
        assert cache.obtener("b", 1.0) is not None and cache.obtener("c", 1.0) is not None

    def test_overwrite_refreshes_position(self) -> None:
        cache = CacheCompatibilidad(max_entradas=2, reloj=_Reloj())
        cache.guardar("a", 1.0, _medido("a"))
        cache.guardar("b", 1.0, _medido("b"))
        cache.guardar("a", 1.0, _medido("a2"))  # a becomes newest
        cache.guardar("c", 1.0, _medido("c"))   # evicts b
        assert cache.obtener("b", 1.0) is None
        assert cache.obtener("a", 1.0).modelo_id == "a2"

    def test_vaciar(self) -> None:
        cache = CacheCompatibilidad(reloj=_Reloj())
        cache.guardar("a", 1.0, _medido())
        cache.vaciar()
        assert len(cache) == 0

    def test_invalid_config(self) -> None:
        with pytest.raises(ValueError):
            CacheCompatibilidad(ttl_segundos=-1)
        with pytest.raises(ValueError):
            CacheCompatibilidad(max_entradas=0)
