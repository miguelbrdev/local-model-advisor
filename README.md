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

**Pipeline served by the synced catalog; measured capability from Epoch AI benchmarks; `scripts/sync_catalog.py` and `scripts/sync_benchmarks.py` ready for the OS scheduler.**

Implemented:

- FastAPI API with `POST /chat`, `GET /health`, and `GET /families`
- Structured validation with Pydantic v2
- Free-text input: `mensaje` may be criteria JSON or natural language
  ("quiero programar en Python con 12 GB"), interpreted by a local Ollama
  model through LangChain structured output, re-validated with Pydantic and
  merged into the conversation's criteria; missing fields trigger a
  clarification and the follow-up completes them
- Catalog abstraction (`src/catalog/`): `CatalogoModelos` protocol plus a
  synchronized adapter with a strict fallback chain
  `runtime/catalog/current.json` → `data/catalog_bootstrap.json` → loud
  startup failure (no degraded mode, no empty catalog)
- Catalog synchronization from canIRun.ai (`src/sync/`): versioned local
  policy, normalization, SHA-256 content hash, atomic snapshot writes to
  `runtime/catalog/` (gitignored), and a `current.json` pointer
- Committed bootstrap `snap-f84e4af19dfbc6b8` (136 source models, 107
  included, 29 excluded as out of scope), promoted to the versioned
  `data/catalog_bootstrap.json` with its original `snapshot_id`
- Catalogue enrichment during synchronisation (`src/sync/enrichment.py`):
  languages from the Hugging Face model card (`cardData.language`, 40 of 107
  families declare them) and runtimes from the canIRun.ai detail endpoint
  (`ollamaId`, `lmStudioId`) plus the formats published on Hugging Face
  (GGUF, MLX, AWQ, …; 107 of 107 families). `idiomas_requeridos` and
  `runtime_preferido` filter the families that declare the data and the
  response counts the ones that could not be checked
- All three v1 tasks (`chat`, `codigo`, `razonamiento`), filtered on the
  full `tareas_soportadas` list of each model
- Filtering by VRAM, language, runtime and commercial-license needs, with a
  single license normalizer shared by filter and scorer and a single runtime
  normalizer shared by the catalogue and the filter
- Real VRAM compatibility lookup through canirun.ai, with the offline
  parameter-count heuristic (`src/compatibility/heuristica.py`) as the
  production fallback when the service is down or lacks a model; the same
  heuristic drives the offline evaluation
- Deterministic multi-criteria ranking with seven explicit sub-scores (VRAM
  safety margin, capacity, measured capability, task specialisation, recency,
  context window, licence), following "the most capable model that still
  fits comfortably"; every signal comes from the synced catalogue or the
  benchmark snapshot, none from model names
- Measured capability (`src/benchmarks/`): benchmark scores from Epoch AI's
  Benchmarking Hub (GPQA Diamond, MATH level 5, SWE-bench Verified and the
  Epoch Capabilities Index; CC BY 4.0, no account) and, optionally, the
  Artificial Analysis Intelligence Index (per-operator key), joined to the
  catalogue only through the reviewed table `data/benchmark_mapping.yaml`.
  `scripts/sync_benchmarks.py` writes an atomic snapshot; the Epoch-only part
  is committed as `data/benchmarks_bootstrap.json` so a fresh clone ranks
  with real scores, and every recommendation carries the source attribution
- SQLite-backed server-side conversation state with a 24-hour TTL, storing
  the catalog `snapshot_id` that produced each recommendation
- Explanatory evidence: README fragments per candidate from a Qdrant index
  built with a local Ollama embedding model, attached after ranking and never
  used for scoring
- Limitations and the age of the catalog data reported in every
  recommendation, so a short result is never ambiguous
- `scripts/sync_catalog.py`: the entry point for the 21-day scheduled task,
  with retention of historical snapshots (`src/sync/retention.py`) that never
  deletes the active or previous snapshot
- Golden-query evaluation (`data/golden_queries.yaml`,
  `scripts/evaluate_golden.py`): reviewed cases pin filter and ranking
  behaviour per catalog snapshot and report pass rate, hit rate and
  exclusion rate
- 723 automated tests and Ruff linting

Not implemented yet:

- Language data for every family: only the 40 families whose Hugging Face
  card declares languages are filtered by `idiomas_requeridos`; the rest are
  shown and counted in `limitaciones` (languages are never inferred from text)
- Hybrid (sparse + dense) retrieval and reranking of evidence
- Benchmark coverage for every family: 55 of the 107 catalogue families have an
  Epoch AI score today; the rest score the neutral value and the response
  counts them
- Per-quantisation VRAM from the canIRun.ai detail endpoint (`quants`) instead
  of the parameter-count heuristic

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
Benchmark scores (approved snapshot, reviewed mapping)
   |
   v
Deterministic weighted scorer
   |
   v
Top-N explainable recommendations
   |
   v
Explanatory README fragments (never alter a score)
```

The ranking only uses structured and deterministic fields: the synced
catalogue, the compatibility verdict and the benchmark snapshot. Retrieved
model-card text is attached after ranking as explanatory evidence and never
alters a candidate score.

## Tech stack

- Python
- FastAPI
- Pydantic v2
- SQLite with a 24-hour conversation TTL
- httpx
- PyYAML
- LangChain (`langchain-core`, `langchain-ollama`) with a local Ollama model
- Qdrant (`qdrant-client`) for the explanatory-evidence index
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

## Natural-language input

`POST /chat` accepts either criteria JSON (always available, no LLM involved)
or free text. Free text is interpreted by a local Ollama model: the model only
fills the criteria fields it can read from the message, the output is
validated with Pydantic, and the deterministic pipeline runs exactly as with
JSON. The model never sees the catalog and never influences the ranking.

The recommended way to run it is Docker Compose (see below), which pulls the
model for you. For a host install instead:

```bash
winget install Ollama.Ollama        # or https://ollama.com/download
ollama pull gemma4:e4b
```

| Variable | Default | Purpose |
|---|---|---|
| `LMA_EXTRACCION_LLM` | `1` | Set to `0` to accept only criteria JSON |
| `LMA_OLLAMA_MODEL` | `gemma4:e4b` | Ollama model tag |
| `LMA_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama server |
| `LMA_EVIDENCIA` | `1` | Set to `0` to skip explanatory evidence |
| `LMA_QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant server holding the evidence index |
| `LMA_EMBEDDING_MODEL` | `embeddinggemma` | Ollama embedding model |
| `LMA_CALENTAR_MODELOS` | `1` | Load both Ollama models in the background at startup; `0` to skip |
| `LMA_API_KEY` | unset | When set, `/chat` and `/families` require the same value in `X-API-Key` |
| `LMA_RATE_LIMIT_POR_MINUTO` | `60` | `/chat` requests per client address per minute; `0` disables |
| `ARTIFICIAL_ANALYSIS_API_KEY` | unset | Read only by `scripts/sync_benchmarks.py`; adds the Artificial Analysis Intelligence Index to the local benchmark snapshot |
| `LMA_TLS_CERTIFI` | unset | Outbound HTTPS verifies against the operating-system trust store; set to `1` to use the `certifi` bundle instead (which rejects canirun.ai's chain since late 2026) |

If Ollama is unreachable or returns invalid output, a free-text message gets
`503` with a hint to send JSON; JSON input keeps working. The first message
may be incomplete: `"necesito razonar"` returns a clarification, and
`"tengo 12 GB"` sent with the same `conversation_id` completes the criteria.

## Explanatory evidence

Every recommended candidate can carry `evidencia_explicativa`: up to two
fragments of the model's own Hugging Face README, chosen by similarity to the
user's message, with the source URL and a relevance score for display. The
fragments are attached **after** ranking and never influence a score or the
order (the tests pin both). The index is built outside the API:

```bash
python -m scripts.index_docs        # fetch READMEs into runtime/docs, embed, upsert into Qdrant
```

Source: the Hugging Face repository each catalog record points to, fetched
without a token. Licence-gated repositories (Meta, Google) answer 401 and are
recorded as `gated`; quantisation mirrors are followed one hop to the
original model. When Qdrant or the embedding model is unreachable, or a
candidate has no indexed documentation, `limitaciones` says so and the
recommendation is returned anyway.

## Measured capability (benchmarks)

`capacidad_medida` (weight 0.15) is the only sub-score fed by data outside
the catalogue, and it comes exclusively from two approved sources:

- **Epoch AI Benchmarking Hub** (mandatory, public, CC BY 4.0): the
  `benchmark_data.zip` archive with one CSV per benchmark plus the Epoch
  Capabilities Index. `codigo` uses SWE-bench Verified, `razonamiento` the
  mean of GPQA Diamond and MATH level 5, `chat` the Capabilities Index;
  a task without its own benchmark falls back to the index. Accuracies are
  anchored on the benchmark's random baseline and ceiling, the index on fixed
  anchors (100 → 0, 160 → 1), so a new model in the source never moves the
  others.
- **Artificial Analysis** (optional): the Intelligence Index from the free
  Data API tier. It needs a per-operator key (`ARTIFICIAL_ANALYSIS_API_KEY`,
  free account), its terms forbid redistribution, and it covers few local
  models today, so it is used only when no Epoch index exists for a family and
  its data never leaves `runtime/`.

Scores reach a family only through `data/benchmark_mapping.yaml`, a reviewed
table with the exact source name per family (reasoning-effort suffixes,
revisions and API aliases are decided by a human, never by fuzzy matching).
Families without an entry or without data score the neutral value and the
response counts them in `limitaciones`; `atribuciones` lists the sources
behind the data.

```bash
python -m scripts.sync_benchmarks                       # Epoch AI (+ Artificial Analysis if the key is set)
python -m scripts.sync_benchmarks --promote-bootstrap   # also refresh data/benchmarks_bootstrap.json (Epoch only)
python -m scripts.sync_benchmarks --skip-artificial-analysis
```

Exit codes: `0` snapshot in place, `1` Epoch AI failed (previous snapshot
kept), `2` mapping invalid or promotion failed. An Artificial Analysis
failure (no key, quota, 401) is reported and never fatal. Schedule it next to
the catalogue sync (every 21 days) and review the `unmatched:` lines it
prints after each run. The API loads `runtime/benchmarks/current.json`, else
the committed bootstrap, else runs with neutral scores and says so.

## Docker Compose

`compose.yaml` runs the API and a local Ollama server side by side, pulls the
model once into a volume, and keeps `runtime/` (SQLite conversations and
catalog snapshots) in another volume so recreating the containers loses
nothing:

```bash
docker compose up --build                                        # CPU
docker compose -f compose.yaml -f compose.gpu.yaml up --build    # NVIDIA GPU
LMA_OLLAMA_MODEL=gemma4:e2b docker compose up --build            # lighter model
docker compose run --rm sync                                     # catalog sync cycle
docker compose run --rm sync-benchmarks                          # benchmark scores (key read from .env)
docker compose run --rm index                                    # READMEs -> Qdrant evidence index
docker compose run --rm index python -m scripts.index_docs --skip-fetch   # re-index stored READMEs
```

First free-text request after start pays the model load (about 100 s for
`gemma4:e4b` on an RTX 2080 Ti); later ones take a few seconds. The model
stays loaded for `OLLAMA_KEEP_ALIVE` (default `1h`).

The API listens on `127.0.0.1:8000` and Ollama on `127.0.0.1:11434`, both bound
to localhost only. The GPU overlay needs the NVIDIA driver on the host and, on
Windows, Docker Desktop with the WSL2 backend; without it Ollama runs on CPU.
Tests and the golden evaluation still run in the local virtualenv; the image
is for running the service, not for development.

## Operating it

- **Cold start**: at startup the API loads the chat and embedding models in a
  background thread and asks Ollama to keep them resident (`keep_alive=-1`),
  so the first user request does not pay the ~100 s model load.
  `GET /health` shows `modelos_listos` once the warm-up has finished.
- **Exposure**: ports are bound to `127.0.0.1` by default. If you expose the
  API, set `LMA_API_KEY` (clients send `X-API-Key`) and keep the per-client
  rate limit on `/chat` (`429` with `Retry-After` when exceeded). `mensaje` is
  limited to 4000 characters.
- **Catalog age**: `/health` reports `catalogo_dias` and flags
  `catalogo_obsoleto` after 28 days (21-day sync cadence plus a week); every
  recommendation then carries a limitation line until the next sync.
- **State**: expired conversations (24 h TTL) are purged at startup and every
  100 chats.

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
  "limitaciones": ["..."],
  "atribuciones": ["Benchmark scores and Capabilities Index: Epoch AI, ..."]
}
```

The exact ranking depends on the catalog snapshot and the compatibility
response returned by the provider; the figures above are illustrative.

## Catalog synchronization

The API never downloads the catalog during a request. A separate process
refreshes `runtime/catalog/` and the API picks the new snapshot up on its
next start. One run fetches the canIRun.ai list, normalises it, then enriches
every included family from the approved metadata sources (canIRun.ai detail
record, Hugging Face card and quantised-derivative tree: about three requests
per family) and writes the snapshot atomically; a family whose lookups fail
keeps empty fields and the snapshot header counts the outcomes per source:

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

## Evaluation

`data/golden_queries.yaml` holds reviewed query cases written against one
catalog snapshot: the criteria a client would send, the families that must be
in the top N, the families that must be filtered out, optionally the expected
first result, and a minimum candidate count so an ambiguous empty answer is a
failure. The suite runs offline with the parameter-count VRAM heuristic, so it
pins the deterministic part of the pipeline (filter + scorer), not live
canirun.ai answers.

```bash
python -m scripts.evaluate_golden                     # text report, exit 1 on any failure
python -m scripts.evaluate_golden --json              # full report as JSON
python -m scripts.evaluate_golden --strict-snapshot   # also fail on snapshot mismatch
python -m scripts.evaluate_golden --no-benchmarks     # neutral capacidad_medida
```

The suite ranks with the committed `data/benchmarks_bootstrap.json` (never
with `runtime/`, whose optional Artificial Analysis data differs per
operator) and records the benchmark snapshot it was written for
(`benchmarks_snapshot_id`).

The committed suite also runs inside `pytest`, so a ranking change that
breaks an expectation fails CI.

Free-text extraction has its own suite, `data/golden_extraccion.yaml` (20
Spanish and English messages; a `null` expectation pins that a field must not
be invented). It needs the local model, so it runs as a script:

```bash
python -m scripts.evaluate_extraction      # exit 1 below --min-aprobacion (0.8)
```

Last measured with `gemma4:e4b`: 20/20 cases, 2.5 s per message. When the bootstrap is re-promoted, re-review
the cases and update `snapshot_id` in the suite.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Returns API health status |
| `GET` | `/families` | Lists every family of the mounted catalog snapshot |
| `POST` | `/chat` | Returns a clarification or recommendation response; `422` when free text is disabled and `mensaje` is not criteria JSON; `503` when the local LLM is unavailable |

Once the server is running, interactive OpenAPI documentation is available at
`/docs`.

## Design principles

- **Reproducibility**: identical input and catalog snapshot produce the same ranking.
- **Explainability**: every recommendation includes its weighted sub-scores.
- **Evidence separation**: retrieved README content provides context, not ranking points.
- **Safe state**: candidates and scores live on the server; the client only holds the conversation ID.
- **Graceful degradation**: when canirun.ai is unreachable or has no entry for a model, VRAM compatibility is estimated locally from the parameter count, labelled `heuristica_propia` per candidate and counted in `limitaciones`; an outage stops further calls for the rest of that request.

## Project structure

```text
├── data/
│   ├── catalog_policy.yaml    # versioned sync policy (categories, required fields)
│   ├── catalog_bootstrap.json # versioned bootstrap catalog (preserves snapshot_id)
│   ├── benchmark_mapping.yaml # reviewed family -> source model name table
│   ├── benchmarks_bootstrap.json # Epoch-only benchmark snapshot (CC BY 4.0)
│   └── golden_queries.yaml    # reviewed golden queries for the evaluation
├── docs/
│   └── technical-design.md    # full design (local only — gitignored, not shipped)
├── scripts/                   # sync_catalog.py, sync_benchmarks.py, evaluate_golden.py, index_docs.py
├── Dockerfile                 # API image (sources run from /app, deps from pyproject)
├── compose.yaml               # api + ollama + qdrant (+ model pull); sync and index profiles; compose.gpu.yaml overlay
├── src/
│   ├── api/             # FastAPI endpoints and app lifespan (mounts catalog + benchmarks)
│   ├── benchmarks/      # Approved benchmark sources, reviewed mapping, snapshot, provider
│   ├── catalog/         # Catalog protocol + synchronized adapter with fallback chain
│   ├── chains/          # Recommendation orchestration and criteria extraction (JSON / Ollama)
│   ├── compatibility/   # Interchangeable VRAM providers (canirun.ai, offline heuristic)
│   ├── evaluation/      # Golden-query evaluation and metrics
│   ├── evidence/        # Explanatory evidence: HF README fetch/store, chunking, Qdrant index
│   ├── models/          # Pydantic contracts
│   ├── ranking/         # Deterministic weighted scoring
│   ├── state/           # SQLite conversation persistence
│   ├── sync/            # Catalog sync: download, policy, normalization, snapshots, bootstrap, retention
│   └── tools/           # Structured filtering and license normalization
├── runtime/             # gitignored: SQLite conversations, catalog + benchmark snapshots
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
- [x] Add Qdrant retrieval for explanatory evidence (dense; hybrid pending)
- [ ] Add reranking of retrieved evidence
- [x] Add LangChain-based criteria extraction
- [x] Add golden-query evaluation metrics
- [x] Add Docker Compose packaging and GPU passthrough
- [x] Add measured capability from approved benchmark sources (Epoch AI, optional Artificial Analysis)
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
