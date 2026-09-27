"""Catalogue abstraction consumed by the recommendation pipeline.

This package isolates the pipeline from concrete catalogue storage: it
defines the ``CatalogoModelos`` contract plus a synchronized-catalogue
implementation with a deterministic fallback chain.  Callers import from
the submodules (``src.catalog.protocol``, ``src.catalog.synced``), the
same convention used by ``src.sync``.
"""
