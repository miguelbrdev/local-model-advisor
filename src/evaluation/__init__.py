"""Offline evaluation of the deterministic recommendation pipeline.

``golden`` runs a versioned suite of query cases against a catalogue and a
deterministic compatibility oracle and reports pass/fail plus aggregate
metrics.  Nothing here calls the network or an LLM.
"""
