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

The deterministic vertical slice works end to end on the synchronized
canIRun.ai catalog: the API boots from the local snapshot (or the committed
bootstrap), filters by task, license, language and VRAM, and ranks with an
explicit weighted score.

**Pipeline served by the synced catalog; `scripts/sync_catalog.py` ready for the OS scheduler.**

Implemented:

- FastAPI API with `POST /chat`, `GET /health`, and `GET /families`
- Structured validation with Pydantic v2
- Catalog abstraction (`src/catalog/`): `CatalogoModelos` protocol plus a
  synchronized adapter with a strict fallback chain
  `runtime/catalog/current.json` → `data/catalog_bootstrap.json` → loud
  startup failure (no degraded mode, no empty catalog)
- Catalog synchronization from canIRun.ai (`src/sync/`): versioned local
  policy, normalization, SHA-256 content hash, atomic snapshot writes to
  `runtime/catalog/` (gitignored), and a `current.json` pointer
- First real validated snapshot (`snap-176d40242fd0157a`: 104 source models,
  91 included, 13 excluded — all out of scope), promoted to the versioned
  bootstrap `data/catalog_bootstrap.json` with its original `snapshot_id`
- All three v1 tasks (`chat`, `codigo`, `razonamiento`), filtered on the
  full `tareas_soportadas` list of each model
- Filtering by VRAM, language and commercial-license needs, with a single
  license normalizer shared by filter and scorer (`runtime_preferido` is
  accepted but not applied yet)
- Real VRAM compatibility lookup through canirun.ai
- Deterministic multi-criteria ranking with explicit sub-scores; VRAM fit is
  measured as headroom on the user's own GPU memory
- SQLite-backed server-side conversation state with a 24-hour TTL, storing
  the catalog `snapshot_id` that produced each recommendation
- Limitations reported in every recommendation, so a short result is never
  ambiguous
- `scripts/sync_catalog.py`: the entry point for the 21-day scheduled task,
  with retention of historical snapshots (`src/sync/retention.py`) that never
  deletes the active or previous snapshot
- 336 automated tests and Ruff linting

Not implemented yet:

- Language data for synced models: the canIRun.ai source exposes no
  language field, so language requirements are not enforced on them
- Runtime preference filtering
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

Install the project dependencies (the `dev` extra provides `pytest` and
`ruff`, needed by the test and lint steps below):

```bash
pip install -e ".[dev]"
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
      "familia": "qwen2.5-coder-7b",
      "puntuacion_total": 0.86
    },
    {
      "familia": "qwen2.5-coder-1.5b",
      "puntuacion_total": 0.81
    },
    {
      "familia": "granite-4.1-3b",
      "puntuacion_total": 0.75
    }
  ],
  "limitaciones": ["..."]
}
```

The exact ranking depends on the catalog snapshot and the compatibility
response returned by the provider; the figures above are illustrative.

## Catalog synchronization

The API never downloads the catalog during a request. A separate process
refreshes `runtime/catalog/` and the API picks the new snapshot up on its
next start:

```bash
python -m scripts.sync_catalog            # fetch, validate, persist atomically, prune
python -m scripts.sync_catalog --help     # all options
```

Exit code `0` means a valid snapshot is active (new or unchanged), `1` that
the fetch or validation failed and the previous snapshot was kept, `2` that
an optional post-step failed. Snapshots not referenced by `current.json` are
deleted after 90 days (`--retain-days`). Pass `--promote-bootstrap` to copy
the active snapshot to `data/catalog_bootstrap.json`; committing that file is
a deliberate, reviewed step.

Register it every 21 days. The module must run from the project root, so the
task wraps it in a `cd`. Windows Task Scheduler, from a PowerShell prompt at
the project root:

```powershell
$root = (Get-Location).Path
schtasks /Create /TN "LocalModelAdvisor catalog sync" /SC DAILY /MO 21 /ST 03:00 /F `
  /TR "cmd /c cd /d ""$root"" && .venv\Scripts\python.exe -m scripts.sync_catalog >> runtime\sync.log 2>&1"
```

cron (Linux/macOS), running on the 1st and 22nd of each month:

```cron
0 3 1,22 * * cd /path/to/local-model-advisor && .venv/bin/python -m scripts.sync_catalog >> runtime/sync.log 2>&1
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Returns API health status |
| `GET` | `/families` | Lists every family of the mounted catalog snapshot |
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
├── data/
│   ├── catalog_policy.yaml    # versioned sync policy (categories, required fields)
│   └── catalog_bootstrap.json # versioned bootstrap catalog (preserves snapshot_id)
├── docs/
│   └── technical-design.md    # full design (local only — gitignored, not shipped)
├── scripts/                   # sync_catalog.py: scheduled catalog sync + snapshot retention
├── src/
│   ├── api/             # FastAPI endpoints and app lifespan (mounts the catalog)
│   ├── catalog/         # Catalog protocol + synchronized adapter with fallback chain
│   ├── chains/          # Recommendation orchestration
│   ├── compatibility/   # Interchangeable VRAM providers
│   ├── models/          # Pydantic contracts
│   ├── ranking/         # Deterministic weighted scoring
│   ├── state/           # SQLite conversation persistence
│   ├── sync/            # Catalog sync: download, policy, normalization, snapshots, bootstrap, retention
│   └── tools/           # Structured filtering and license normalization
├── runtime/             # gitignored: SQLite conversations + catalog snapshots
├── tests/
├── .github/workflows/   # CI: Ruff + pytest on Python 3.11-3.14
├── AGENTS.md            # agent instructions and current project state
├── LICENSE              # MIT
├── README.md
└── pyproject.toml
```

## Roadmap

- [x] Week 0: deterministic recommendation vertical slice
- [x] Automated tests for filters, scoring, providers, state, and API
- [x] Catalog synchronization from canIRun.ai with atomic snapshots
- [x] First real snapshot and versioned bootstrap catalog
- [x] Connect the pipeline loader (runtime catalog → bootstrap fallback)
- [x] Persist `snapshot_id` per conversation
- [x] Enable chat and reasoning tasks in the pipeline
- [x] Add `scripts/sync_catalog.py` for the recurring scheduled task (every 21 days)
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
