"""Tests for src.api.main — FastAPI endpoints.

Uses TestClient (synchronous) for /health and /families.
Uses mocked pipeline for /chat to avoid real canirun.ai calls.

The catalogue sources are pinned to ``tmp_path`` for every test that
runs the lifespan: level 1 (``runtime/catalog/``) is absent unless a
test says otherwise, level 2 gets a one-family bootstrap file.  Nothing
here depends on the machine's real ``runtime/catalog/`` state.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.main import (
    DIAS_OBSOLESCENCIA_CATALOGO,
    LIMITACIONES_BASE,
    app,
    calentar_modelos,
    construir_buscador_evidencia,
    construir_extractor,
    describir_antiguedad,
    dias_desde,
)
from src.benchmarks import provider as benchmarks_provider
from src.benchmarks.fuentes import FuenteMeta
from src.benchmarks.snapshot import construir_snapshot, write_snapshot
from src.catalog import synced
from src.catalog.protocol import CatalogUnavailableError
from src.chains.extraccion_criterios import (
    ExtraccionNoDisponibleError,
    ExtractorCompuesto,
    ExtractorOllama,
)
from src.evidence.qdrant_index import Fragmento, IndiceEvidencia, IndiceNoDisponibleError
from src.models.schemas import (
    CandidatoRankeado,
    CompatibilityResult,
    CriteriosBusqueda,
    EvidenciaBenchmark,
    FamiliaModelo,
)
from src.state import conversation_store
from src.sync.bootstrap import DEFAULT_BOOTSTRAP_PATH
from src.sync.snapshot import CatalogSnapshot, SnapshotMeta, compute_content_hash

# Regression sentinels for the committed bootstrap (mirrors
# tests/test_catalog_loader.py): if the file is ever re-promoted these
# tests must fail loudly instead of silently shifting.
BOOTSTRAP_SNAPSHOT_ID = "snap-f84e4af19dfbc6b8"
BOOTSTRAP_FAMILIA_COUNT = 107


# ---------------------------------------------------------------------------
# Catalogue source pinning helpers (all artefacts go under tmp_path)
# ---------------------------------------------------------------------------


def _fam(fid: str) -> FamiliaModelo:
    return FamiliaModelo(
        familia_id=fid,
        variantes=["7B"],
        pipeline_tag="chat",
        canirun_id=fid,
        tareas_soportadas=["chat"],
        nombre_mostrado=fid.title(),
    )


def _write_bootstrap(path: Path, familias: list[FamiliaModelo]) -> None:
    """Materialise a valid single-file bootstrap (level-2 source shape)."""
    hashcontenido = compute_content_hash(familias, "1.0.0")
    meta = SnapshotMeta(
        schema_version="1.0",
        snapshot_id=f"snap-{hashcontenido[:16]}",
        fecha_sincronizacion=datetime(2026, 1, 1, tzinfo=UTC),
        source_url="https://www.canirun.ai/api/models",
        hash_contenido=hashcontenido,
        modelos_origen=len(familias),
        modelos_incluidos=len(familias),
        motivos_exclusion=[],
        politica_version="1.0.0",
    )
    snapshot = CatalogSnapshot(meta=meta, familias=familias)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )


def _point_benchmark_sources(
    monkeypatch: pytest.MonkeyPatch, *, runtime_path: Path, bootstrap_path: Path
) -> None:
    """Redirect the zero-argument ``BenchmarksLocales`` defaults (same seam as
    the catalogue: ``lifespan`` builds it with no arguments)."""
    monkeypatch.setattr(benchmarks_provider, "DEFAULT_RUNTIME_BENCHMARKS_PATH", runtime_path)
    monkeypatch.setattr(benchmarks_provider, "DEFAULT_BENCHMARKS_BOOTSTRAP_PATH", bootstrap_path)


def _write_benchmarks(path: Path, puntuaciones: dict) -> str:
    """Write a one-source benchmark snapshot and return its id."""
    meta = FuenteMeta(
        nombre="epoch_ai", version="2026-11-08", url="https://epoch.ai/data/benchmark_data.zip",
        licencia="CC BY 4.0", atribucion="Source: Epoch AI (CC BY 4.0)", redistribuible=True,
        registros=1, descargado_en=datetime(2026, 11, 8, tzinfo=UTC),
    )
    snapshot = construir_snapshot(
        puntuaciones, [meta], fecha=datetime(2026, 11, 8, tzinfo=UTC),
        mapeo_version="t", mapeo_snapshot_id="snap-test", familias_mapeadas=len(puntuaciones),
    )
    write_snapshot(snapshot, path)
    return snapshot.meta.snapshot_id


def _point_catalogue_sources(
    monkeypatch: pytest.MonkeyPatch, *, runtime_dir: Path, bootstrap_path: Path
) -> None:
    """Redirect the zero-argument ``CatalogoSincronizado`` defaults.

    ``lifespan`` builds the catalogue with no arguments, so the module
    constants in ``src.catalog.synced`` are the only seam that lets tests
    choose the sources before the TestClient context starts.
    """
    monkeypatch.setattr(synced, "DEFAULT_RUNTIME_CATALOG_DIR", runtime_dir)
    monkeypatch.setattr(synced, "DEFAULT_BOOTSTRAP_PATH", bootstrap_path)


class _SentinelCatalogue:
    """In-memory ``CatalogoModelos`` double with unmistakable content.

    ``/families`` must read ``app.state.catalogo`` itself.  A handler
    that constructs its own ``CatalogoSincronizado`` would serve the
    pinned ``tmp_path`` bootstrap instead of whatever instance is
    mounted, so swapping the mounted catalogue for this sentinel (whose
    ``familia_id`` appears in no fixture source) is what tells the two
    implementations apart.
    """

    def __init__(self, familias: list[FamiliaModelo]) -> None:
        self._familias = list(familias)

    @property
    def snapshot_id(self) -> str | None:
        return "snap-sentinel"

    @property
    def fecha_sincronizacion(self) -> datetime | None:
        return datetime(2026, 1, 1, tzinfo=UTC)

    def cargar(self) -> list[FamiliaModelo]:
        return list(self._familias)


@pytest.fixture(autouse=True)
def _conversaciones_en_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every conversation written by these tests out of ``runtime/`` and
    never let a startup warm-up thread reach a real Ollama."""
    monkeypatch.setattr(conversation_store, "_DB_PATH", tmp_path / "conversations.db")
    monkeypatch.setenv("LMA_CALENTAR_MODELOS", "0")
    monkeypatch.delenv("LMA_API_KEY", raising=False)
    monkeypatch.delenv("LMA_RATE_LIMIT_POR_MINUTO", raising=False)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient with the lifespan active and sources pinned to tmp_path.

    Level 1 is absent (fresh-clone shape) and level 2 is a one-family
    bootstrap, so startup is deterministic and independent of the real
    ``runtime/catalog/`` directory.  The lifespan runs on ``__enter__``;
    ``app.state.catalogo`` therefore only exists inside the ``with``
    block, which is why every request test must go through this fixture.
    """
    bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
    _write_bootstrap(bootstrap, [_fam("test-model")])
    _point_catalogue_sources(
        monkeypatch,
        runtime_dir=tmp_path / "runtime" / "catalog",
        bootstrap_path=bootstrap,
    )
    # No benchmark snapshot by default: ``capacidad_medida`` is neutral and
    # the response carries the "no benchmark snapshot" limitation.
    _point_benchmark_sources(
        monkeypatch,
        runtime_path=tmp_path / "runtime" / "benchmarks" / "current.json",
        bootstrap_path=tmp_path / "data" / "benchmarks_bootstrap.json",
    )
    # Free-text extraction and explanatory evidence off by default: no test
    # may reach Ollama or Qdrant.  Tests of those paths mount fakes on
    # ``app.state.extractor`` / ``app.state.evidencia``.
    monkeypatch.setenv("LMA_EXTRACCION_LLM", "0")
    monkeypatch.setenv("LMA_EVIDENCIA", "0")
    with TestClient(app) as test_client:
        yield test_client


EVIDENCIA_OFF = "Explanatory evidence is disabled (LMA_EVIDENCIA=0)."
OBSOLETO_PREFIJO = "Catalogue snapshot is "
SIN_BENCHMARKS = (
    "No benchmark snapshot loaded (run python -m scripts.sync_benchmarks); "
    "capacidad_medida is neutral for every candidate."
)


def _extras(limitaciones: list[str]) -> list[str]:
    """Request-specific limitations: strip the base list, the "no benchmark
    snapshot" line (the fixture mounts none) and the staleness line (the
    fixture's 2026-01-01 snapshot is always stale)."""
    assert limitaciones[: len(LIMITACIONES_BASE)] == LIMITACIONES_BASE
    resto = limitaciones[len(LIMITACIONES_BASE):]
    assert resto and resto[0] == SIN_BENCHMARKS
    assert resto[1].startswith(OBSOLETO_PREFIJO)
    return resto[2:]


class _FakeBuscador:
    """Scripted ``BuscadorEvidencia``: fragments per family, or a failure."""

    def __init__(self, por_familia: dict[str, list[Fragmento]] | None = None,
                 error: Exception | None = None) -> None:
        self._por_familia = por_familia or {}
        self._error = error
        self.consultas: list[str] = []

    def buscar(self, consulta: str, familia_id: str, k: int = 2) -> list[Fragmento]:
        self.consultas.append(consulta)
        if self._error is not None:
            raise self._error
        return self._por_familia.get(familia_id, [])


class _FakeExtractor:
    """Scripted ``ExtractorCriterios``: returns the next prepared criteria,
    merged over ``base`` like the real extractors do, or raises."""

    def __init__(self, *salidas: CriteriosBusqueda, error: Exception | None = None) -> None:
        self._salidas = list(salidas)
        self._error = error
        self.mensajes: list[str] = []

    @property
    def acepta_texto_libre(self) -> bool:
        return True

    def extraer(self, mensaje: str, base: CriteriosBusqueda | None = None) -> CriteriosBusqueda:
        self.mensajes.append(mensaje)
        if self._error is not None:
            raise self._error
        siguiente = self._salidas.pop(0)
        previo = base.model_dump() if base is not None else {}
        nuevo = {k: v for k, v in siguiente.model_dump().items() if v is not None and v != []}
        return CriteriosBusqueda.model_validate({**previo, **nuevo})


def _mock_pipeline_result() -> list[CandidatoRankeado]:
    return [
        CandidatoRankeado(
            familia=FamiliaModelo(
                familia_id="mock-model",
                variantes=["7B"],
                pipeline_tag="code",
                license_declarada="MIT",
                idiomas_detectados=["en"],
                formatos_disponibles=["gguf"],
            ),
            variante_recomendada="7B",
            compatibilidad_vram=CompatibilityResult(
                modelo_id="mock-model",
                variante="7B",
                compatible=True,
                grado="comfortable",
                fuente="canirun_ai",
                tokens_per_second=40.0,
                vram_required_gb=4.0,
            ),
            subpuntuaciones=[],
            puntuacion_total=0.8,
        )
    ]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_reports_what_the_process_serves(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["snapshot_id"] == app.state.catalogo.snapshot_id
        assert data["catalogo_sincronizado"].startswith("2026-01-01")
        assert data["texto_libre"] is False  # fixture disables the LLM
        assert data["evidencia"] is False    # fixture disables the index
        assert data["catalogo_obsoleto"] is True  # pinned 2026-01-01 snapshot is old
        assert data["catalogo_dias"] > DIAS_OBSOLESCENCIA_CATALOGO
        assert data["api_key_requerida"] is False
        assert data["modelos_listos"] is None  # warm-up disabled by the fixture
        assert data["benchmarks_snapshot_id"] is None  # fixture mounts no snapshot
        assert data["benchmarks_fuentes"] == []

    def test_health_needs_no_api_key(self, client: TestClient, monkeypatch) -> None:
        app.state.api_key = "s3cret"
        assert client.get("/health").status_code == 200

    def test_expired_conversations_are_removed_at_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import timedelta

        from src.models.schemas import ConversationRun

        monkeypatch.setattr(conversation_store, "_DB_PATH", tmp_path / "c.db")
        viejo = ConversationRun(
            criterios=CriteriosBusqueda(tarea="codigo", vram_gb=8),
            created_at=datetime.now(UTC) - timedelta(hours=48),
        )
        conversation_store.guardar(viejo)
        bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
        _write_bootstrap(bootstrap, [_fam("test-model")])
        _point_catalogue_sources(
            monkeypatch, runtime_dir=tmp_path / "runtime" / "catalog", bootstrap_path=bootstrap
        )
        monkeypatch.setenv("LMA_EXTRACCION_LLM", "0")
        monkeypatch.setenv("LMA_EVIDENCIA", "0")
        with TestClient(app):
            assert conversation_store.cargar(viejo.conversation_id) is None


# ---------------------------------------------------------------------------
# GET /families
# ---------------------------------------------------------------------------

class TestFamilies:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/families")
        assert response.status_code == 200

    def test_returns_list(self, client: TestClient) -> None:
        data = client.get("/families").json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_each_family_has_required_fields(self, client: TestClient) -> None:
        data = client.get("/families").json()
        for fam in data:
            assert "familia_id" in fam
            assert "variantes" in fam
            assert "pipeline_tag" in fam

    def test_serves_the_mounted_catalogue_not_the_seed(
        self, client: TestClient
    ) -> None:
        """Content must come from the lifespan-mounted catalogue (the
        pinned one-family bootstrap), not from ``load_seed`` nor from
        the committed 107-family bootstrap.
        """
        data = client.get("/families").json()
        assert [fam["familia_id"] for fam in data] == ["test-model"]

    def test_response_shape_is_familia_modelo(self, client: TestClient) -> None:
        """HTTP contract pin: the serialised shape must be exactly the
        ``FamiliaModelo`` field set — no field added, none dropped —
        regardless of which catalogue source served the content.
        """
        data = client.get("/families").json()
        expected_keys = set(FamiliaModelo.model_json_schema()["properties"])
        assert data  # the pinned bootstrap contributes at least one family
        for fam in data:
            assert set(fam) == expected_keys
        assert data == [f.model_dump() for f in app.state.catalogo.cargar()]

    def test_reads_the_mounted_app_state_catalogue(self, client: TestClient) -> None:
        """The handler must read ``app.state.catalogo``, not build its own
        catalogue: replacing the mounted instance with a sentinel whose
        family exists in no fixture source must change the response.  A
        self-constructed ``CatalogoSincronizado`` would still serve the
        pinned ``tmp_path`` bootstrap and fail this assertion.
        """
        app.state.catalogo = _SentinelCatalogue([_fam("sentinel-family")])
        data = client.get("/families").json()
        assert [fam["familia_id"] for fam in data] == ["sentinel-family"]


# ---------------------------------------------------------------------------
# POST /chat — new conversation
# ---------------------------------------------------------------------------

class TestChatNuevaConversacion:
    @patch("src.api.main.ejecutar_pipeline")
    def test_returns_200(self, mock_pipeline: MagicMock, client: TestClient) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {
            "mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200

    @patch("src.api.main.ejecutar_pipeline")
    def test_returns_conversation_id(self, mock_pipeline: MagicMock, client: TestClient) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {
            "mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})
        }
        data = client.post("/chat", json=payload).json()
        assert "conversation_id" in data
        UUID(data["conversation_id"])  # raises if invalid

    @patch("src.api.main.ejecutar_pipeline")
    def test_returns_candidatos(self, mock_pipeline: MagicMock, client: TestClient) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {
            "mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})
        }
        data = client.post("/chat", json=payload).json()
        assert data["tipo"] == "recomendacion"
        assert len(data["candidatos"]) == 1
        assert data["candidatos"][0]["familia"]["familia_id"] == "mock-model"

    @patch("src.api.main.ejecutar_pipeline")
    def test_calls_pipeline_with_criteria(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {
            "mensaje": json.dumps(
                {"tarea": "codigo", "vram_gb": 11.0, "idiomas_requeridos": ["en"]}
            )
        }
        client.post("/chat", json=payload)
        call_args = mock_pipeline.call_args
        criterios: CriteriosBusqueda = call_args[0][0]
        assert criterios.tarea == "codigo"
        assert criterios.vram_gb == 11.0
        assert criterios.idiomas_requeridos == ["en"]

    @patch("src.api.main.ejecutar_pipeline")
    def test_chat_passes_the_mounted_catalogue(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        """The route must inject the exact process-wide instance the
        lifespan mounted — the guarantee behind one snapshot_id per
        process — not a freshly built catalogue.
        """
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {
            "mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})
        }
        client.post("/chat", json=payload)
        assert mock_pipeline.call_args.kwargs["catalogo"] is app.state.catalogo

    @patch("src.api.main.ejecutar_pipeline")
    def test_reports_current_limitations(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
        data = client.post("/chat", json=payload).json()
        assert _extras(data["limitaciones"]) == [EVIDENCIA_OFF]

    @patch("src.api.main.ejecutar_pipeline")
    def test_limitations_count_locally_estimated_candidates(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        medido = _mock_pipeline_result()[0]
        estimado = medido.model_copy(deep=True)
        estimado.compatibilidad_vram = estimado.compatibilidad_vram.model_copy(
            update={"fuente": "heuristica_propia"}
        )
        mock_pipeline.return_value = [medido, estimado]
        payload = {"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
        limitaciones = client.post("/chat", json=payload).json()["limitaciones"]
        extra = _extras(limitaciones)
        assert extra[0] == EVIDENCIA_OFF
        assert len(extra) == 2
        assert extra[1].startswith("VRAM compatibility of 1 of 2 candidates was estimated locally")

    @patch("src.api.main.ejecutar_pipeline")
    def test_limitations_name_the_unapplied_criteria(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        """Languages and runtime the user asked for are reported, with their
        values, when a candidate declares neither (so the filter could not
        check it) instead of a generic disclaimer."""
        candidato = _mock_pipeline_result()[0]
        candidato.familia = candidato.familia.model_copy(
            update={"idiomas_detectados": None, "runtimes_disponibles": []}
        )
        mock_pipeline.return_value = [candidato]
        payload = {
            "mensaje": json.dumps(
                {"tarea": "codigo", "vram_gb": 11.0, "idiomas_requeridos": ["es", "en"],
                 "runtime_preferido": "ollama"}
            )
        }
        limitaciones = client.post("/chat", json=payload).json()["limitaciones"]
        assert limitaciones[: len(LIMITACIONES_BASE)] == LIMITACIONES_BASE
        extra = _extras(limitaciones)
        assert extra[0] == EVIDENCIA_OFF
        assert len(extra) == 3
        assert "['es', 'en']" in extra[1] and "1 of 1 candidates" in extra[1]
        assert "'ollama'" in extra[2] and "1 of 1 candidates" in extra[2]

    @patch("src.api.main.ejecutar_pipeline")
    def test_no_limitation_when_candidates_declare_languages_and_runtimes(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        candidato = _mock_pipeline_result()[0]
        candidato.familia = candidato.familia.model_copy(
            update={"idiomas_detectados": ["es", "en"], "runtimes_disponibles": ["ollama"]}
        )
        mock_pipeline.return_value = [candidato]
        payload = {
            "mensaje": json.dumps(
                {"tarea": "codigo", "vram_gb": 11.0, "idiomas_requeridos": ["es", "en"],
                 "runtime_preferido": "ollama"}
            )
        }
        extra = _extras(client.post("/chat", json=payload).json()["limitaciones"])
        assert extra == [EVIDENCIA_OFF]

    @patch("src.api.main.ejecutar_pipeline")
    def test_reports_age_of_the_mounted_snapshot(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        """``datos_actualizados_hace`` must derive from the mounted
        catalogue's sync date (pinned to 2026-01-01 by the fixture)."""
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
        data = client.post("/chat", json=payload).json()
        assert data["datos_actualizados_hace"].startswith("hace ")
        assert data["datos_actualizados_hace"].endswith("(sincronizado el 2026-01-01)")

    @patch("src.api.main.ejecutar_pipeline")
    def test_persists_snapshot_id_of_mounted_catalogue(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        """Reproducibility: the stored run must name the snapshot that
        produced it, taken from the process-wide catalogue."""
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
        data = client.post("/chat", json=payload).json()

        run = conversation_store.cargar(UUID(data["conversation_id"]))
        assert run is not None
        assert run.snapshot_id == app.state.catalogo.snapshot_id
        assert run.snapshot_id.startswith("snap-")

    @patch("src.api.main.ejecutar_pipeline")
    def test_unknown_conversation_id_starts_a_new_conversation(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        stale_id = str(uuid4())
        payload = {
            "conversation_id": stale_id,
            "mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0}),
        }
        data = client.post("/chat", json=payload).json()
        assert data["tipo"] == "recomendacion"
        assert data["conversation_id"] != stale_id

    @patch("src.api.main.ejecutar_pipeline")
    def test_history_accumulates_across_turns(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        """Each turn appends a user and a system entry; earlier turns are kept."""
        mock_pipeline.return_value = _mock_pipeline_result()
        first = json.dumps({"tarea": "codigo", "vram_gb": 11.0})
        conv_id = client.post("/chat", json={"mensaje": first}).json()["conversation_id"]
        client.post("/chat", json={"conversation_id": conv_id, "mensaje": "tell me more"})

        run = conversation_store.cargar(UUID(conv_id))
        assert run is not None
        assert [t.rol for t in run.historial] == ["usuario", "sistema", "usuario", "sistema"]
        assert run.historial[0].contenido == first
        assert run.historial[2].contenido == "tell me more"

    @patch("src.api.main.ejecutar_pipeline")
    def test_persists_conversation(self, mock_pipeline: MagicMock, client: TestClient) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {
            "mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})
        }
        data = client.post("/chat", json=payload).json()
        conv_id = data["conversation_id"]

        # Follow-up with same conversation_id should reuse existing criteria
        mock_pipeline.return_value = _mock_pipeline_result()
        followup = {
            "conversation_id": conv_id,
            "mensaje": "tell me more",
        }
        data2 = client.post("/chat", json=followup).json()
        assert data2["conversation_id"] == conv_id


class TestDescribirAntiguedad:
    AHORA = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)

    def test_unknown_date_is_empty_string(self) -> None:
        assert describir_antiguedad(None, self.AHORA) == ""

    def test_days_and_date(self) -> None:
        fecha = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
        assert describir_antiguedad(fecha, self.AHORA) == (
            "hace 26 días (sincronizado el 2026-09-22)"
        )

    def test_singular_day(self) -> None:
        fecha = datetime(2026, 10, 17, 11, 0, tzinfo=UTC)
        assert describir_antiguedad(fecha, self.AHORA) == "hace 1 día (sincronizado el 2026-10-17)"

    def test_same_day_and_clock_skew_are_zero_days(self) -> None:
        assert describir_antiguedad(self.AHORA, self.AHORA).startswith("hace 0 días")
        future = datetime(2026, 10, 19, tzinfo=UTC)
        assert describir_antiguedad(future, self.AHORA).startswith("hace 0 días")

    def test_defaults_to_now(self) -> None:
        texto = describir_antiguedad(datetime(2026, 1, 1, tzinfo=UTC))
        assert texto.startswith("hace ") and texto.endswith("(sincronizado el 2026-01-01)")


# ---------------------------------------------------------------------------
# POST /chat — missing fields
# ---------------------------------------------------------------------------

class TestChatCamposFaltantes:
    def test_missing_tarea_returns_clarificacion(self, client: TestClient) -> None:
        payload = {
            "mensaje": json.dumps({"vram_gb": 11.0})
        }
        data = client.post("/chat", json=payload).json()
        assert data["tipo"] == "clarificacion"
        assert data["campo_faltante"] == "tarea"
        assert data["pregunta"] == (
            "Entendido: vram_gb=11. Falta indicar la tarea (chat, codigo o razonamiento)."
        )

    def test_clarification_without_anything_understood(self, client: TestClient) -> None:
        data = client.post("/chat", json={"mensaje": "{}"}).json()
        assert data["pregunta"] == (
            "Falta indicar la tarea (chat, codigo o razonamiento) y la VRAM en GB."
        )

    def test_missing_vram_returns_clarificacion(self, client: TestClient) -> None:
        payload = {
            "mensaje": json.dumps({"tarea": "codigo"})
        }
        data = client.post("/chat", json=payload).json()
        assert data["tipo"] == "clarificacion"
        assert data["campo_faltante"] == "vram_gb"


# ---------------------------------------------------------------------------
# POST /chat — invalid JSON in mensaje
# ---------------------------------------------------------------------------

class TestGuardias:
    """API key, rate limit and message bounds."""

    def test_api_key_required_when_configured(self, client: TestClient) -> None:
        app.state.api_key = "s3cret"
        payload = {"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
        assert client.post("/chat", json=payload).status_code == 401
        assert client.get("/families").status_code == 401
        assert client.get("/families", headers={"X-API-Key": "nope"}).status_code == 401
        assert client.get("/families", headers={"X-API-Key": "s3cret"}).status_code == 200
        with patch("src.api.main.ejecutar_pipeline", return_value=_mock_pipeline_result()):
            ok = client.post("/chat", json=payload, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200

    def test_api_key_from_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
        _write_bootstrap(bootstrap, [_fam("test-model")])
        _point_catalogue_sources(
            monkeypatch, runtime_dir=tmp_path / "runtime" / "catalog", bootstrap_path=bootstrap
        )
        monkeypatch.setenv("LMA_EXTRACCION_LLM", "0")
        monkeypatch.setenv("LMA_EVIDENCIA", "0")
        monkeypatch.setenv("LMA_API_KEY", "from-env")
        with TestClient(app) as c:
            assert c.get("/health").json()["api_key_requerida"] is True
            assert c.get("/families").status_code == 401
            assert c.get("/families", headers={"X-API-Key": "from-env"}).status_code == 200

    @patch("src.api.main.ejecutar_pipeline")
    def test_rate_limit_returns_429_with_retry_after(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        from src.api.limites import LimitadorTasa

        mock_pipeline.return_value = _mock_pipeline_result()
        app.state.limitador = LimitadorTasa(maximo_por_ventana=2)
        payload = {"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
        assert client.post("/chat", json=payload).status_code == 200
        assert client.post("/chat", json=payload).status_code == 200
        bloqueada = client.post("/chat", json=payload)
        assert bloqueada.status_code == 429
        assert int(bloqueada.headers["Retry-After"]) >= 1
        assert client.get("/families").status_code == 200  # only /chat is limited

    def test_rate_limit_from_environment_zero_disables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
        _write_bootstrap(bootstrap, [_fam("test-model")])
        _point_catalogue_sources(
            monkeypatch, runtime_dir=tmp_path / "runtime" / "catalog", bootstrap_path=bootstrap
        )
        monkeypatch.setenv("LMA_EXTRACCION_LLM", "0")
        monkeypatch.setenv("LMA_EVIDENCIA", "0")
        monkeypatch.setenv("LMA_RATE_LIMIT_POR_MINUTO", "0")
        with TestClient(app):
            assert app.state.limitador.activo is False

    def test_message_length_is_bounded(self, client: TestClient) -> None:
        assert client.post("/chat", json={"mensaje": ""}).status_code == 422
        assert client.post("/chat", json={"mensaje": "x" * 4001}).status_code == 422
        with patch("src.api.main.conversation_store.guardar") as guardar:
            client.post("/chat", json={"mensaje": "x" * 4001})
        guardar.assert_not_called()


class TestCalentamiento:
    def test_warm_up_calls_both_components_and_never_raises(self) -> None:
        class _Ok:
            llamadas = 0

            def calentar(self) -> bool:
                self.llamadas += 1
                return True

        class _Boom:
            def calentar(self) -> bool:
                raise RuntimeError("ollama down")

        ok, boom = _Ok(), _Boom()
        assert calentar_modelos(ok, boom) == {"extractor": True, "evidencia": False}
        assert ok.llamadas == 1

    def test_components_without_calentar_are_skipped(self) -> None:
        assert calentar_modelos(object(), None) == {"extractor": None, "evidencia": None}

    def test_startup_runs_warm_up_in_background_when_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time as _time

        bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
        _write_bootstrap(bootstrap, [_fam("test-model")])
        _point_catalogue_sources(
            monkeypatch, runtime_dir=tmp_path / "runtime" / "catalog", bootstrap_path=bootstrap
        )
        monkeypatch.setenv("LMA_EXTRACCION_LLM", "0")
        monkeypatch.setenv("LMA_EVIDENCIA", "0")
        monkeypatch.setenv("LMA_CALENTAR_MODELOS", "1")
        with TestClient(app) as c:
            for _ in range(50):  # the thread is tiny; give it a moment
                if app.state.modelos_listos is not None:
                    break
                _time.sleep(0.02)
            assert app.state.modelos_listos == {"extractor": False, "evidencia": None}
            assert c.get("/health").json()["modelos_listos"] == {
                "extractor": False, "evidencia": None,
            }

    def test_compuesto_delegates_warm_up(self) -> None:
        from src.chains.extraccion_criterios import ExtractorCompuesto, ExtractorJson

        class _LLM:
            def calentar(self) -> bool:
                return True

            def extraer(self, mensaje, base=None):
                raise AssertionError("not used")

        assert ExtractorCompuesto(ExtractorJson(), _LLM()).calentar() is True
        assert ExtractorCompuesto(ExtractorJson(), None).calentar() is False


class TestObsolescencia:
    def test_dias_desde(self) -> None:
        class _Cat:
            fecha_sincronizacion = datetime(2026, 1, 1, tzinfo=UTC)

        assert dias_desde(_Cat(), datetime(2026, 1, 31, tzinfo=UTC)) == 30
        assert dias_desde(None) is None
        assert dias_desde(_Cat(), datetime(2025, 12, 1, tzinfo=UTC)) == 0  # clock skew

    @patch("src.api.main.ejecutar_pipeline")
    def test_stale_catalogue_is_a_limitation(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        payload = {"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
        limitaciones = client.post("/chat", json=payload).json()["limitaciones"]
        assert any(lim.startswith("Catalogue snapshot is ") for lim in limitaciones)


class TestChatInvalidMessage:
    """With free-text input disabled, malformed criteria are a client error
    (422), never a server fault (500)."""

    def test_invalid_json_returns_422(self, client: TestClient) -> None:
        response = client.post("/chat", json={"mensaje": "not valid json {"})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list) and detail
        assert detail[0]["loc"][:2] == ["body", "mensaje"]
        assert {"loc", "msg", "type"} <= set(detail[0])

    def test_wrong_field_type_returns_422_naming_the_field(self, client: TestClient) -> None:
        response = client.post(
            "/chat", json={"mensaje": json.dumps({"tarea": "codigo", "vram_gb": "many"})}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "mensaje", "vram_gb"]

    def test_unknown_task_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/chat", json={"mensaje": json.dumps({"tarea": "embeddings", "vram_gb": 8})}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "mensaje", "tarea"]

    def test_invalid_message_persists_nothing(self, client: TestClient) -> None:
        with patch("src.api.main.conversation_store.guardar") as guardar:
            client.post("/chat", json={"mensaje": "not valid json {"})
        guardar.assert_not_called()


# ---------------------------------------------------------------------------
# POST /chat — free-text input through the mounted extractor
# ---------------------------------------------------------------------------


class TestChatTextoLibre:
    @patch("src.api.main.ejecutar_pipeline")
    def test_free_text_is_extracted_and_recommended(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        fake = _FakeExtractor(CriteriosBusqueda(tarea="codigo", vram_gb=12))
        app.state.extractor = fake

        data = client.post("/chat", json={"mensaje": "quiero programar con 12 GB"}).json()

        assert data["tipo"] == "recomendacion"
        assert data["criterios_interpretados"]["tarea"] == "codigo"
        assert data["criterios_interpretados"]["vram_gb"] == 12
        assert fake.mensajes == ["quiero programar con 12 GB"]
        criterios: CriteriosBusqueda = mock_pipeline.call_args[0][0]
        assert criterios.vram_gb == 12

    def test_clarification_is_persisted_and_follow_up_completes_it(
        self, client: TestClient
    ) -> None:
        """First message lacks VRAM -> clarification; the follow-up on the
        same conversation merges the missing field and recommends."""
        fake = _FakeExtractor(
            CriteriosBusqueda(tarea="razonamiento"),
            CriteriosBusqueda(vram_gb=12),
        )
        app.state.extractor = fake

        with patch("src.api.main.ejecutar_pipeline", return_value=_mock_pipeline_result()):
            first = client.post("/chat", json={"mensaje": "necesito razonar"}).json()
            assert first["tipo"] == "clarificacion"
            assert first["campo_faltante"] == "vram_gb"
            conv_id = first["conversation_id"]

            run = conversation_store.cargar(UUID(conv_id))
            assert run is not None and run.criterios.tarea == "razonamiento"

            second = client.post(
                "/chat", json={"conversation_id": conv_id, "mensaje": "tengo 12 GB"}
            ).json()

        assert second["tipo"] == "recomendacion"
        assert second["conversation_id"] == conv_id
        assert second["criterios_interpretados"] == {
            "tarea": "razonamiento",
            "vram_gb": 12.0,
            "idiomas_requeridos": [],
            "runtime_preferido": None,
            "requiere_licencia_comercial": False,
        }
        run = conversation_store.cargar(UUID(conv_id))
        assert [t.rol for t in run.historial] == ["usuario", "sistema", "usuario", "sistema"]
        assert fake.mensajes == ["necesito razonar", "tengo 12 GB"]

    @patch("src.api.main.ejecutar_pipeline")
    def test_complete_conversation_ignores_follow_up_text(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        fake = _FakeExtractor(CriteriosBusqueda(tarea="chat", vram_gb=8))
        app.state.extractor = fake
        conv_id = client.post("/chat", json={"mensaje": "un chat con 8 GB"}).json()[
            "conversation_id"
        ]
        data = client.post(
            "/chat", json={"conversation_id": conv_id, "mensaje": "cuéntame más"}
        ).json()
        assert data["tipo"] == "recomendacion"
        assert fake.mensajes == ["un chat con 8 GB"]  # extractor not consulted again

    def test_llm_unavailable_is_503_and_persists_nothing(self, client: TestClient) -> None:
        app.state.extractor = _FakeExtractor(
            error=ExtraccionNoDisponibleError("connection refused")
        )
        with patch("src.api.main.conversation_store.guardar") as guardar:
            response = client.post("/chat", json={"mensaje": "quiero programar con 12 GB"})
        assert response.status_code == 503
        assert "Send the criteria as JSON" in response.json()["detail"]
        guardar.assert_not_called()

    def test_free_text_disabled_returns_422(self, client: TestClient) -> None:
        """The fixture disables the LLM; free text must then be a 422 with
        the validation items, not a 503."""
        response = client.post("/chat", json={"mensaje": "quiero programar con 12 GB"})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"][:2] == ["body", "mensaje"]

    def test_lifespan_mounts_a_composite_extractor(self, client: TestClient) -> None:
        assert isinstance(app.state.extractor, ExtractorCompuesto)
        assert app.state.extractor.acepta_texto_libre is False


class TestChatEvidencia:
    """Explanatory evidence is attached after ranking and reported honestly."""

    @patch("src.api.main.ejecutar_pipeline")
    def test_fragments_attached_without_changing_ranking(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        fake = _FakeBuscador({
            "mock-model": [Fragmento("mock-model", "Great at Python.", 0.83, "https://hf.co/m", 0)]
        })
        app.state.evidencia = fake
        payload = {"mensaje": "quiero programar en Python con 12 GB"}
        app.state.extractor = _FakeExtractor(CriteriosBusqueda(tarea="codigo", vram_gb=12))

        data = client.post("/chat", json=payload).json()

        candidato = data["candidatos"][0]
        assert candidato["puntuacion_total"] == 0.8  # untouched
        assert candidato["evidencia_explicativa"] == [
            {"fuente_url": "https://hf.co/m", "fragmento": "Great at Python.", "relevancia": 0.83}
        ]
        assert fake.consultas == ["quiero programar en Python con 12 GB"]
        assert not any(
            "Explanatory evidence" in lim or "documentation fragments" in lim
            for lim in data["limitaciones"]
        )

    @patch("src.api.main.ejecutar_pipeline")
    def test_candidates_without_fragments_are_counted(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        app.state.evidencia = _FakeBuscador({})
        payload = {"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
        data = client.post("/chat", json=payload).json()
        assert data["candidatos"][0]["evidencia_explicativa"] == []
        assert any(
            lim.startswith("1 of 1 candidates have no documentation fragments")
            for lim in data["limitaciones"]
        )

    @patch("src.api.main.ejecutar_pipeline")
    def test_index_failure_is_a_limitation_not_an_error(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        app.state.evidencia = _FakeBuscador(error=IndiceNoDisponibleError("qdrant down"))
        payload = {"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["candidatos"][0]["evidencia_explicativa"] == []
        assert any(
            lim.startswith("Explanatory evidence unavailable: qdrant down")
            for lim in data["limitaciones"]
        )

    @patch("src.api.main.ejecutar_pipeline")
    def test_json_message_queries_with_a_task_description(
        self, mock_pipeline: MagicMock, client: TestClient
    ) -> None:
        mock_pipeline.return_value = _mock_pipeline_result()
        fake = _FakeBuscador({})
        app.state.evidencia = fake
        client.post("/chat", json={"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})})
        assert fake.consultas == ["modelo para codigo"]

    def test_lifespan_mounts_no_index_when_disabled(self, client: TestClient) -> None:
        assert app.state.evidencia is None


class TestConstruirBuscadorEvidencia:
    def test_enabled_by_default_builds_index_without_io(self) -> None:
        buscador = construir_buscador_evidencia({"LMA_QDRANT_URL": "http://127.0.0.1:9"})
        assert isinstance(buscador, IndiceEvidencia)

    @pytest.mark.parametrize("valor", ["0", "false", "no", "off"])
    def test_can_be_disabled(self, valor: str) -> None:
        assert construir_buscador_evidencia({"LMA_EVIDENCIA": valor}) is None


class TestConstruirExtractor:
    def test_enabled_by_default_with_ollama(self) -> None:
        extractor = construir_extractor({})
        assert extractor.acepta_texto_libre is True
        llm = extractor._llm
        assert isinstance(llm, ExtractorOllama)
        assert llm.modelo == "gemma4:e4b"
        assert llm.base_url == "http://127.0.0.1:11434"

    def test_env_overrides_model_and_url(self) -> None:
        extractor = construir_extractor(
            {"LMA_OLLAMA_MODEL": "qwen3:4b", "LMA_OLLAMA_BASE_URL": "http://127.0.0.1:11500"}
        )
        assert extractor._llm.modelo == "qwen3:4b"
        assert extractor._llm.base_url == "http://127.0.0.1:11500"

    @pytest.mark.parametrize("valor", ["0", "false", "no", "off", " False "])
    def test_can_be_disabled(self, valor: str) -> None:
        assert construir_extractor({"LMA_EXTRACCION_LLM": valor}).acepta_texto_libre is False


# ---------------------------------------------------------------------------
# Catalogue wiring (lifespan + sources)
# ---------------------------------------------------------------------------


class TestCatalogueWiring:
    def test_fresh_clone_boots_and_serves_bootstrap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``runtime/catalog/`` absent (fresh clone) must still boot and
        serve ``/families`` from the committed bootstrap.
        """
        _point_catalogue_sources(
            monkeypatch,
            runtime_dir=tmp_path / "runtime" / "catalog",
            bootstrap_path=DEFAULT_BOOTSTRAP_PATH,
        )
        with TestClient(app) as client:
            response = client.get("/families")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == BOOTSTRAP_FAMILIA_COUNT
        assert app.state.catalogo.snapshot_id == BOOTSTRAP_SNAPSHOT_ID
        assert data == [f.model_dump() for f in app.state.catalogo.cargar()]

    def test_startup_constructs_one_catalogue_and_mounts_that_instance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin T3 item 5 and the lifespan memoization claim: startup
        builds exactly one ``CatalogoSincronizado`` and
        ``app.state.catalogo`` *is* the eagerly resolved instance.
        Mounting a second, freshly built one would serve identical
        content while quietly forfeiting the one-snapshot-per-process
        guarantee, which is why content assertions alone cannot catch it.

        The constructor is observed by wrapping the name the lifespan
        resolves (``src.api.main.CatalogoSincronizado``), so ``main.py``
        itself needs no test hook.
        """
        bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
        _write_bootstrap(bootstrap, [_fam("test-model")])
        _point_catalogue_sources(
            monkeypatch,
            runtime_dir=tmp_path / "runtime" / "catalog",
            bootstrap_path=bootstrap,
        )

        constructed: list[synced.CatalogoSincronizado] = []
        real_catalogue = synced.CatalogoSincronizado

        def _recording_factory(*args, **kwargs):
            instance = real_catalogue(*args, **kwargs)
            constructed.append(instance)
            return instance

        monkeypatch.setattr("src.api.main.CatalogoSincronizado", _recording_factory)

        with TestClient(app):
            assert len(constructed) == 1
            assert app.state.catalogo is constructed[0]

    def test_startup_fails_loudly_when_no_source_resolves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both sources exhausted → the lifespan must abort the boot with
        ``CatalogUnavailableError``.  Degraded/silent startup is
        explicitly rejected: there is no empty-catalogue mode.
        """
        _point_catalogue_sources(
            monkeypatch,
            runtime_dir=tmp_path / "absent" / "catalog",
            bootstrap_path=tmp_path / "absent.json",
        )

        with pytest.raises(CatalogUnavailableError), TestClient(app):
            pass  # pragma: no cover — startup must never get here

    def test_unresolvable_sources_are_not_served_as_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The typed error must name both exhausted sources, so the
        operator knows which file to fix — an empty list would be
        indistinguishable from a legitimately empty catalogue.
        """
        runtime = tmp_path / "absent" / "catalog"
        bootstrap = tmp_path / "absent.json"
        _point_catalogue_sources(
            monkeypatch, runtime_dir=runtime, bootstrap_path=bootstrap
        )

        with pytest.raises(CatalogUnavailableError) as excinfo, TestClient(app):
            pass  # pragma: no cover — startup must never get here

        message = str(excinfo.value)
        assert str(runtime) in message
        assert str(bootstrap) in message


class TestBenchmarks:
    """Benchmark snapshot mounted at startup: health, attributions, limitations."""

    @pytest.fixture
    def client_con_benchmarks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Iterator[tuple[TestClient, str]]:
        bootstrap = tmp_path / "data" / "catalog_bootstrap.json"
        _write_bootstrap(bootstrap, [_fam("test-model")])
        _point_catalogue_sources(
            monkeypatch, runtime_dir=tmp_path / "runtime" / "catalog", bootstrap_path=bootstrap
        )
        benchmarks = tmp_path / "data" / "benchmarks_bootstrap.json"
        snapshot_id = _write_benchmarks(benchmarks, {"other-id": [_evidencia()]})
        _point_benchmark_sources(
            monkeypatch,
            runtime_path=tmp_path / "runtime" / "absent.json",
            bootstrap_path=benchmarks,
        )
        monkeypatch.setenv("LMA_EXTRACCION_LLM", "0")
        monkeypatch.setenv("LMA_EVIDENCIA", "0")
        with TestClient(app) as test_client:
            yield test_client, snapshot_id

    def test_health_reports_snapshot_and_sources(self, client_con_benchmarks) -> None:
        client, snapshot_id = client_con_benchmarks
        data = client.get("/health").json()
        assert data["benchmarks_snapshot_id"] == snapshot_id
        assert data["benchmarks_fuentes"] == ["epoch_ai"]

    def test_chat_without_snapshot_has_no_attributions(self, client: TestClient) -> None:
        with patch("src.api.main.ejecutar_pipeline") as mock_pipeline:
            mock_pipeline.return_value = _mock_pipeline_result()
            data = client.post(
                "/chat", json={"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
            ).json()
        assert data["atribuciones"] == []
        assert SIN_BENCHMARKS in data["limitaciones"]
        assert mock_pipeline.call_args.kwargs["benchmarks"] is app.state.benchmarks

    def test_chat_with_snapshot_attributes_and_counts_unmeasured(
        self, client_con_benchmarks
    ) -> None:
        client, _ = client_con_benchmarks
        with patch("src.api.main.ejecutar_pipeline") as mock_pipeline:
            mock_pipeline.return_value = _mock_pipeline_result()  # candidate without scores
            data = client.post(
                "/chat", json={"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
            ).json()
        assert data["atribuciones"] == ["Source: Epoch AI (CC BY 4.0)"]
        assert SIN_BENCHMARKS not in data["limitaciones"]
        assert any(
            line.startswith("1 of 1 candidates have no benchmark data")
            for line in data["limitaciones"]
        )

    def test_chat_with_snapshot_and_measured_candidate_has_no_count_line(
        self, client_con_benchmarks
    ) -> None:
        client, _ = client_con_benchmarks
        candidato = _mock_pipeline_result()[0]
        candidato.evidencia_benchmark = [_evidencia()]
        with patch("src.api.main.ejecutar_pipeline") as mock_pipeline:
            mock_pipeline.return_value = [candidato]
            data = client.post(
                "/chat", json={"mensaje": json.dumps({"tarea": "codigo", "vram_gb": 11.0})}
            ).json()
        assert not any("no benchmark data" in line for line in data["limitaciones"])
        assert data["candidatos"][0]["evidencia_benchmark"][0]["benchmark"] == "eci"


def _evidencia() -> EvidenciaBenchmark:
    return EvidenciaBenchmark(
        fuente="epoch_ai", benchmark="eci", metrica="eci", valor=132.2,
        valor_normalizado=0.54, modelo_fuente="Qwen3-8B",
    )
