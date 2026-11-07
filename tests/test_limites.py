"""Tests for src.api.limites — rate limiter and API-key comparison."""

from __future__ import annotations

import pytest

from src.api.limites import LimitadorTasa, api_key_valida


class _Reloj:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TestLimitadorTasa:
    def test_allows_up_to_the_limit_then_refuses(self) -> None:
        reloj = _Reloj()
        limitador = LimitadorTasa(maximo_por_ventana=3, ventana_segundos=60, reloj=reloj)
        assert [limitador.admitir("a")[0] for _ in range(3)] == [True, True, True]
        admitida, espera = limitador.admitir("a")
        assert admitida is False
        assert 0 < espera <= 60

    def test_window_resets(self) -> None:
        reloj = _Reloj()
        limitador = LimitadorTasa(maximo_por_ventana=1, ventana_segundos=60, reloj=reloj)
        assert limitador.admitir("a")[0] is True
        assert limitador.admitir("a")[0] is False
        reloj.t = 60
        assert limitador.admitir("a")[0] is True

    def test_clients_are_independent(self) -> None:
        limitador = LimitadorTasa(maximo_por_ventana=1, reloj=_Reloj())
        assert limitador.admitir("a")[0] is True
        assert limitador.admitir("b")[0] is True
        assert limitador.admitir("a")[0] is False

    def test_zero_disables(self) -> None:
        limitador = LimitadorTasa(maximo_por_ventana=0, reloj=_Reloj())
        assert limitador.activo is False
        assert all(limitador.admitir("a") == (True, 0.0) for _ in range(100))

    def test_invalid_config(self) -> None:
        with pytest.raises(ValueError):
            LimitadorTasa(maximo_por_ventana=-1)
        with pytest.raises(ValueError):
            LimitadorTasa(ventana_segundos=0)


class TestApiKey:
    def test_open_when_not_configured(self) -> None:
        assert api_key_valida(None, None) is True
        assert api_key_valida("", "anything") is True

    def test_requires_exact_match(self) -> None:
        assert api_key_valida("secret", "secret") is True
        assert api_key_valida("secret", "Secret") is False
        assert api_key_valida("secret", None) is False
        assert api_key_valida("secret", "") is False
