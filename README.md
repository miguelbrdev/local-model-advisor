# Local Model Advisor

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

The Week 0 vertical slice is complete.

Implemented:

- FastAPI API with `POST /chat`, `GET /health`, and `GET /families`
- Structured validation with Pydantic v2
- Curated seed catalog with five code-model families
- Filtering by task, VRAM, language, runtime, and commercial-license needs
- Real VRAM compatibility lookup through canirun.ai
- Deterministic multi-criteria ranking with explicit sub-scores
- SQLite-backed server-side conversation state with a 24-hour TTL
- 113 automated tests and Ruff linting

Not implemented yet:

- Qdrant hybrid retrieval and reranking for explanatory evidence
- LangChain-based extraction of free-form user requirements
- Golden-query evaluation metrics
- Docker Compose packaging
- Support for chat, reasoning, embeddings, and reranker model catalogs

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

- Python 3.12 or newer
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
src/
├── api/             # FastAPI endpoints
├── chains/          # Recommendation orchestration
├── compatibility/   # Interchangeable VRAM providers
├── models/          # Pydantic contracts
├── ranking/         # Deterministic weighted scoring
├── state/           # SQLite conversation persistence
└── tools/           # Seed loading and structured filtering
```

## Technical design

The complete architecture, data models, ranking constraints, planned retrieval
layer, and development roadmap are documented in
[docs/technical-design.md](docs/technical-design.md).

## Roadmap

- [x] Week 0: deterministic recommendation vertical slice
- [x] Automated tests for filters, scoring, providers, state, and API
- [ ] Curate the full model catalog
- [ ] Add Qdrant hybrid retrieval for explanatory evidence
- [ ] Add reranking of retrieved evidence
- [ ] Add LangChain-based criteria extraction
- [ ] Add golden-query evaluation metrics
- [ ] Add Docker Compose packaging and GPU passthrough
- [ ] Record a demo GIF

## Notes

The project currently depends on the availability and response format of the
canirun.ai compatibility API. Provider failures are handled explicitly, and
the architecture supports replacing this integration with an internal VRAM
heuristic in a future version.

## License

This project is currently for portfolio and educational purposes. A license
will be selected before external contributions are accepted.
