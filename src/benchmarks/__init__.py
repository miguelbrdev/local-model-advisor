"""Measured capability signal from explicitly approved benchmark sources.

Two sources are approved (see AGENTS.md):

* **Epoch AI Benchmarking Hub** (``epoch.py``) — public ZIP of per-benchmark
  CSVs plus the Epoch Capabilities Index, CC BY 4.0, no account.  It is the
  mandatory base source and the only one whose data may be committed to the
  repository (``data/benchmarks_bootstrap.json``).
* **Artificial Analysis** (``artificial_analysis.py``) — the Intelligence
  Index from the free Data API tier.  Optional: it needs a per-operator key
  (``ARTIFICIAL_ANALYSIS_API_KEY``) and its terms forbid redistribution, so
  its scores only ever live in the gitignored ``runtime/`` snapshot.

Scores reach a catalogue family exclusively through the reviewed mapping
table ``data/benchmark_mapping.yaml`` (``mapping.py``): one exact source
name per family and source, never fuzzy matching.  ``sync.py`` turns the
downloaded data plus the mapping into an atomic snapshot (``snapshot.py``)
that the API reads through ``provider.py`` — the pipeline never downloads
anything during a request.
"""
