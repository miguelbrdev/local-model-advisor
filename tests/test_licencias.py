"""Tests for src.tools.licencias — the shared license normaliser.

Pins the spellings the synced catalogue actually uses (``"Apache 2.0"``,
``"MIT"``, ``"Gemma"``, ``"Llama 3.1 Community"``) so the filter and the
scorer can never disagree again about what counts as permissive.
"""

from __future__ import annotations

import pytest

from src.tools.licencias import (
    LICENCIAS_PERMISIVAS,
    es_licencia_permisiva,
    normalizar_licencia,
)


class TestNormalizarLicencia:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Apache 2.0", "apache-2.0"),
            ("apache-2.0", "apache-2.0"),
            ("APACHE   2.0", "apache-2.0"),
            ("  MIT ", "mit"),
            ("BSD 3 Clause", "bsd-3-clause"),
            ("Llama 3.1 Community", "llama-3.1-community"),
        ],
    )
    def test_collapses_case_and_whitespace(self, raw: str, expected: str) -> None:
        assert normalizar_licencia(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_missing_returns_none(self, raw: str | None) -> None:
        assert normalizar_licencia(raw) is None


class TestEsLicenciaPermisiva:
    @pytest.mark.parametrize(
        "lic", ["Apache 2.0", "Apache-2.0", "MIT", "mit", "BSD-2-Clause", "ISC", "MPL-2.0"]
    )
    def test_permissive(self, lic: str) -> None:
        assert es_licencia_permisiva(lic) is True

    @pytest.mark.parametrize("lic", ["Gemma", "Llama 3.1 Community", "NVIDIA Open", "Kimi"])
    def test_source_specific_licenses_are_not_permissive(self, lic: str) -> None:
        assert es_licencia_permisiva(lic) is False

    def test_none_is_not_permissive(self) -> None:
        assert es_licencia_permisiva(None) is False

    def test_catalogue_is_normalised(self) -> None:
        """Every entry must already be in normalised form, or lookups miss."""
        for entry in LICENCIAS_PERMISIVAS:
            assert normalizar_licencia(entry) == entry
