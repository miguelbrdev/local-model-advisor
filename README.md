# Local Model Advisor

[![CI](https://github.com/miguelbrdev/local-model-advisor/actions/workflows/ci.yml/badge.svg)](https://github.com/miguelbrdev/local-model-advisor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Local Model Advisor is a local FastAPI service that recommends Hugging Face
models according to a user's task, available VRAM, language requirements,
runtime preference, and commercial-license needs.

It is designed to make local-model selection explainable and reproducible:
structured evidence determines the ranking, while qualitative model
documentation is kept separate from scoring.

## Why this project?

Choosing a local model is rarely just a benchmark comparison. A model must fit
the available GPU memory, support the intended runtime, satisfy licensing
constraints, and be appropriate for the target task.

This project turns those requirements into a deterministic recommendation
pipeline instead of a static, manually maintained list.

## Current status

Weeks 0–1 are complete: the deterministic vertical slice works, and the
catalog-synchronization layer with canIRun.ai is implemented and tested.

**Bootstrap implemented; runtime → bootstrap → loader integration pending.**

Implemented:

- FastAPI API with `POST /chat`, `GET /health`, and `GET /families`
- Structured validation with Pydantic v2
- Seed catalog with five code-model families (`data/families_seed.yaml`,
  kept temporarily as a transition until the pipeline loader migrates)
- Catalog synchronization from canIRun.ai (`src/sync/`): versioned local
  policy, normalization, SHA-256 content hash, atomic snapshot writes to
  `runtime/catalog/` (gitignored), and a `current.json` pointer
- First real validated snapshot (`snap-176d40242fd0157a`: 104 source models,
  91 included, 13 excluded — all out of scope)
- Versioned bootstrap catalog `data/catalog_bootstrap.json`, which preserves
  the original `snapshot_id` (no synthetic `seed-<hash>` ids)
- Filtering by task, VRAM, language, runtime, and commercial-license needs
- Real VRAM compatibility lookup through canirun.ai
- Deterministic multi-criteria ranking with explicit sub-scores
- SQLite-backed server-side conversation state with a 24-hour TTL
- 248 automated tests and Ruff linting

Not implemented yet:

- Pipeline loader for the synced catalog: `recommend_pipeline.py` still reads
  `families_seed.yaml`; it does not yet read `runtime/catalog/current.json`
  or fall back to `data/catalog_bootstrap.json`
- Recurring scheduler: no `scripts/sync_catalog.py` and no OS scheduled task
  yet — synchronization has been run manually once
- `snapshot_id` persistence per conversation (field exists but is not filled)
- Chat and reasoning tasks in the pipeline (the bootstrap already contains
  those models, but it is not connected yet; the seed only has code models)
- Qdrant hybrid retrieval and reranking for explanatory evidence
- LangChain-based extraction of free-form user requirements
- Golden-query evaluation metrics
- Docker Compose packaging

## Architecture

```text
User criteria
   |
   v
Structured filter
   |
   v
VRAM compatibility provider
   |
   v
Deterministic weighted scorer
   |
   v
Top-N explainable recommendations
```

The ranking only uses structured and deterministic fields. Retrieved model-card
text will be added later as explanatory evidence and will never alter a
candidate score.

## Tech stack

- Python
- FastAPI
- Pydantic v2
- SQLite with a 24-hour conversation TTL
- httpx
- PyYAML
- canirun.ai compatibility API
- pytest and Ruff

## Quick start

### Requirements

- Python 3.11 or newer (CI runs the suite on 3.11 through 3.14)
- Access to the canirun.ai public API

### Installation

```bash
git clone https://github.com/miguelbrdev/local-model-advisor.git
cd local-model-advisor

python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -e .
```

Start the API:

```bash
uvicorn src.api.main:app --reload
```

### Run tests

```bash
python -m pytest
python -m ruff check .
```

## Example

Request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "tarea": "codigo",
    "vram_gb": 11,
    "idiomas_requeridos": ["en", "es"],
    "runtime_preferido": "ollama"
  }'
```

Example result:

```json
{
  "conversation_id": "<uuid>",
  "tipo": "recomendacion",
  "candidatos": [
    {
      "familia": "qwen2.5-coder",
      "puntuacion_total": 0.77
    },
    {
      "familia": "ornith-1.0",
      "puntuacion_total": 0.72
    },
    {
      "familia": "mistral-small",
      "puntuacion_total": 0.54
    }
  ]
}
```

The exact ranking depends on the curated catalog and the compatibility response
returned by the provider.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Returns API health status |
| `GET` | `/families` | Lists the seeded model families |
| `POST` | `/chat` | Returns a clarification or recommendation response |

Once the server is running, interactive OpenAPI documentation is available at
`/docs`.

## Design principles

- **Reproducibility**: identical input and catalog snapshot produce the same ranking.
- **Explainability**: every recommendation includes its weighted sub-scores.
- **Evidence separation**: retrieved README content provides context, not ranking points.
- **Safe state**: candidates and scores live on the server; the client only holds the conversation ID.
- **Graceful degradation**: an unavailable external compatibility provider does not crash the pipeline.

## Project structure

```text
data/
├── catalog_policy.yaml       # versioned sync policy (categories, required fields)
├── families_seed.yaml        # seed catalog (transition until loader migration)
└── catalog_bootstrap.json    # versioned bootstrap catalog (preserves snapshot_id)
src/
├── api/             # FastAPI endpoints
├── chains/          # Recommendation orchestration
├── compatibility/   # Interchangeable VRAM providers
├── models/          # Pydantic contracts
├── ranking/         # Deterministic weighted scoring
├── state/           # SQLite conversation persistence
├── sync/            # Catalog sync: download, policy, normalization, snapshots, bootstrap
└── tools/           # Seed loading and structured filtering
runtime/             # gitignored: SQLite conversations + catalog snapshots
tests/
```

## Roadmap

- [x] Week 0: deterministic recommendation vertical slice
- [x] Automated tests for filters, scoring, providers, state, and API
- [x] Catalog synchronization from canIRun.ai with atomic snapshots
- [x] First real snapshot and versioned bootstrap catalog
- [ ] Connect the pipeline loader (runtime catalog → bootstrap fallback)
- [ ] Add `scripts/sync_catalog.py` with a recurring scheduled task (every 21 days)
- [ ] Persist `snapshot_id` per conversation
- [ ] Enable chat and reasoning tasks in the pipeline
- [ ] Add Qdrant hybrid retrieval for explanatory evidence
- [ ] Add reranking of retrieved evidence
- [ ] Add LangChain-based criteria extraction
- [ ] Add golden-query evaluation metrics
- [ ] Add Docker Compose packaging and GPU passthrough
- [ ] Record a demo GIF

## Notes

The project currently depends on the availability and response format of the
canirun.ai compatibility API (the observed response contract is documented in
`src/sync/canirun_schemas.py`). Provider failures are handled explicitly, and
the architecture supports replacing this integration with an internal VRAM
heuristic in a future version. The API never downloads the catalog during
`POST /chat`; it always uses the last valid local snapshot.

## License

Licensed under the [MIT License](LICENSE).
