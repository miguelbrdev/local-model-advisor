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
    LIMITACIONES_ACTUALES,
    app,
    construir_extractor,
    describir_antiguedad,
)
from src.catalog import synced
from src.catalog.protocol import CatalogUnavailableError
from src.chains.extraccion_criterios import (
    ExtraccionNoDisponibleError,
    ExtractorCompuesto,
    ExtractorOllama,
)
from src.models.schemas import (
    CandidatoRankeado,
    CompatibilityResult,
    CriteriosBusqueda,
    FamiliaModelo,
)
from src.state import conversation_store
from src.sync.bootstrap import DEFAULT_BOOTSTRAP_PATH
from src.sync.snapshot import CatalogSnapshot, SnapshotMeta, compute_content_hash

# Regression sentinels for the committed bootstrap (mirrors
# tests/test_catalog_loader.py): if the file is ever re-promoted these
# tests must fail loudly instead of silently shifting.
BOOTSTRAP_SNAPSHOT_ID = "snap-176d40242fd0157a"
BOOTSTRAP_FAMILIA_COUNT = 91


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
    """Keep every conversation written by these tests out of ``runtime/``."""
    monkeypatch.setattr(conversation_store, "_DB_PATH", tmp_path / "conversations.db")


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
    # Free-text extraction off by default: no test may reach Ollama.  Tests
    # of the free-text path mount a fake on ``app.state.extractor``.
    monkeypatch.setenv("LMA_EXTRACCION_LLM", "0")
    with TestClient(app) as test_client:
        yield test_client


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
        the committed 91-family bootstrap.
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
        assert data["limitaciones"] == LIMITACIONES_ACTUALES

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
