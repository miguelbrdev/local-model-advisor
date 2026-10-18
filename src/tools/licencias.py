"""Single source of truth for license normalisation and permissiveness.

Both the structured filter (``requiere_licencia_comercial``) and the
ranking scorer (``licencia`` sub-score) need to answer "is this license
permissive?".  Before this module each kept its own list and its own
normalisation, and they disagreed: the synced catalogue spells the most
common license ``"Apache 2.0"`` (space, no dash), which one normaliser
accepted and the other rejected, so 48 of the 91 bootstrap families were
scored as if they carried a restrictive license.

Data source: ``FamiliaModelo.license_declarada`` as delivered by the
canIRun.ai ``license`` field.  Observed spellings: ``"Apache 2.0"``,
``"MIT"``, ``"Gemma"``, ``"Llama 3.1 Community"``, ``"NVIDIA Open"``.
Anything not in :data:`LICENCIAS_PERMISIVAS` is treated as *not* known to
be permissive — never as forbidden; the caller decides what to do.
"""

from __future__ import annotations

LICENCIAS_PERMISIVAS: frozenset[str] = frozenset(
    {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "mpl-2.0"}
)
"""Normalised identifiers of licenses that allow commercial use without
copyleft or usage-policy restrictions.  Compared against
:func:`normalizar_licencia` output only."""


def normalizar_licencia(licencia: str | None) -> str | None:
    """Normalise a declared license string to a comparable identifier.

    Lower-cases and joins whitespace-separated tokens with ``-`` so
    ``"Apache 2.0"``, ``"apache-2.0"`` and ``"APACHE  2.0"`` all become
    ``"apache-2.0"``.  Does not attempt to resolve aliases such as
    ``"Apache License 2.0"``; the source does not use them today.

    Args:
        licencia: Raw license text, possibly ``None`` or blank.

    Returns:
        The normalised identifier, or ``None`` when no license is declared.
    """
    if licencia is None or not licencia.strip():
        return None
    return "-".join(licencia.lower().split())


def es_licencia_permisiva(licencia: str | None) -> bool:
    """Return whether the declared license is a known permissive one.

    ``None`` and unknown licenses return ``False``: absence of evidence
    is not treated as permission.
    """
    return normalizar_licencia(licencia) in LICENCIAS_PERMISIVAS
