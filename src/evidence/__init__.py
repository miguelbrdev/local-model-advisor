"""Explanatory evidence: model documentation that explains, never scores.

Pipeline (all outside ``POST /chat`` except the final lookup):

1. ``hf_readme`` downloads each family's Hugging Face README (the only
   documentation source approved for v1: every catalogue record's ``url``
   points to a Hugging Face repository).
2. ``docs_store`` keeps the downloaded text and its fetch status under
   ``runtime/docs/`` so indexing and re-indexing never hit the network.
3. ``chunker`` turns a README into deterministic text fragments.
4. ``qdrant_index`` embeds the fragments with a local Ollama model and
   stores them in Qdrant, filtered per family.
5. At request time the API attaches the best fragments to each ranked
   candidate *after* scoring.  AGENTS.md invariant: this evidence never
   changes a score or the order of candidates.
"""
