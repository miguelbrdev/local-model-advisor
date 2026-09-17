# Local Model Advisor — Plan técnico y funcional (v4)

> El usuario describe en una frase sus necesidades (tarea, VRAM, idioma, runtime, licencia). Solo se admiten 5 tareas: chat, código, razonamiento, embeddings y rerankers (clasificación de imágenes pospuesta para proteger el alcance). Si falta información clave, el chat pregunta antes de buscar. El sistema filtra un catálogo curado de ~37 familias de modelos locales de Hugging Face, aplica un ranking transparente y multi-criterio, y devuelve 3-5 candidatos recomendados para probar primero — con evidencia citada. Tras el resultado, el usuario puede seguir preguntando sobre los candidatos.

---

## 1. Producto: conversación en dos fases + pipeline de filtro/ranking

```
FASE A — Recopilar criterios (solo si faltan datos clave)
─────────────────────────────────────────────────────────
Usuario: "Quiero programar Python localmente con Ollama, necesito español e inglés."
        │
        ▼
┌─────────────────────────────┐
│  Extracción de criterios     │  ← LangChain, actualiza CriteriosBusqueda
│  (acumula turno a turno)     │     con lo dicho hasta ahora
└─────────────────────────────┘
        │
        ▼
  ¿Faltan `tarea` o `vram_gb`?  ──sí──▶  ClarificacionNecesaria
        │ no                              "¿Cuántos GB de VRAM tienes?"
        ▼
FASE B — Ejecutar pipeline (una vez, cuando los criterios están completos)
─────────────────────────────────────────────────────────
┌─────────────────────────────┐
│  Filtro determinista          │  ← tarea, idioma, licencia, runtime, encaje VRAM
│  (sobre catálogo de ~37       │     (VRAM vía canirun.ai o TamañoDirectoProvider, ver sección 5)
│  familias + sus variantes)    │
└─────────────────────────────┘
        │  candidatos elegibles (ej. 8)
        ▼
┌─────────────────────────────┐
│  Ranking transparente         │  ← SOLO datos estructurados/deterministas:
│  (multi-criterio, explicable) │     VRAM, runtime, licencia, idioma, tarea,
│                                │     benchmark (OpenEvals/model-index),
│                                │     contexto, tamaño/cuantización, doc.
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Evidencia explicativa        │  ← Qdrant híbrido sobre README — NO puntúa,
│  (solo para dar contexto)     │     solo acompaña al resultado con citas
└─────────────────────────────┘
        │
        ▼
  RespuestaRecomendacion: top 3-5 + pool completo de elegibles guardado en el estado

FASE C — Explorar resultados (turnos posteriores)
─────────────────────────────────────────────────────────
Usuario: "¿Y de esos, cuál tiene la licencia más permisiva?"  (solo envía conversation_id + mensaje)
        │
        ▼
  El servidor carga su propio ConversationRun (SQLite) por conversation_id —
  el cliente nunca reenvía candidatos ni scores, así que no hay forma de que
  una respuesta manipulada o corrupta del cliente contamine el resultado.
  Responde filtrando/ordenando el pool de candidatos elegibles YA CALCULADO
  (compatibilidad VRAM y sub-puntuaciones estructuradas ya existen —
  normalmente no hace falta volver a llamar a Qdrant ni a canirun.ai).
  Solo si la pregunta pide algo cualitativo no evaluado antes, se invoca
  retrieval semántico de nuevo (siempre como evidencia explicativa, nunca
  para recalcular el ranking). Sin búsqueda web en v1.
```

Sin modo cloud, sin despliegue público: se ejecuta en local con `docker-compose up`, se enseña en vivo en la entrevista. **Backend con estado ligero y efímero** (revisado, ver sección 6.1): un `conversation_id` identifica la conversación completa (fases A, B y C); el cliente solo reenvía el ID y el mensaje nuevo, nunca los datos calculados (criterios, candidatos, scores, evidencia) — esos vive y se valida siempre en el servidor.

---

## 2. Stack tecnológico

| Capa | Tecnología | Por qué | Habilidad CV |
|---|---|---|---|
| API | FastAPI | Backend REST | Backend moderno |
| Validación | Pydantic v2 | Contratos de datos (`CriteriosBusqueda`, `ResultadoRanking`) | Diseño de datos |
| Catálogo | ~37 familias (YAML de seed): canirun.ai para chat/código/razonamiento, curación propia vía `huggingface_hub` para embeddings/rerankers (sección 3) | Reduce trabajo de curación reutilizando un catálogo ya deduplicado donde existe | Curación de datos, ETL dirigido |
| Metadata relacional | SQLite | Licencias, variantes por familia, tags | Persistencia proporcionada al problema |
| Vector DB | **Qdrant (modo local/embebido)** | Motor real, sin servidor, sin coste | Bases de datos vectoriales reales |
| Embeddings | **BAAI/bge-m3** (denso + disperso) | Multilingüe, para retrieval de evidencia cualitativa en README | Embeddings |
| Fusión híbrida | RRF nativo de Qdrant | Sin reimplementar fusión a mano | Hybrid search |
| Reranking | **BAAI/bge-reranker-v2-m3** | Reordena evidencia recuperada | Mejora de retrieval |
| Compatibilidad VRAM | **canirun.ai API** (chat/código/razonamiento) + **fórmula directa** (embeddings/rerankers), ambas detrás de `VRAMCompatibilityProvider` | canirun.ai no cubre modelos no generativos; fórmula directa basta para modelos pequeños | Integración de APIs externas + diseño desacoplado (dependency inversion) |
| Orquestación | **LangChain (LCEL)** | Extracción estructurada de criterios desde texto libre + orquestación del pipeline de filtro/evidencia/ranking | Framework estándar, usado con criterio real |
| Retrieval custom | `BaseRetriever` propio | Lógica de evidencia (Qdrant + reranking) auditable, no oculta en el framework | Comprensión profunda de RAG |
| Generación | LLM local vía **Ollama** + `ChatOllama` | Extracción de criterios + redacción final de la recomendación | Modelos locales, gestión VRAM |
| Contenedores | Docker multi-stage + docker-compose (api + ollama, GPU passthrough) | Un único perfil, sin dualidad cloud/local | Containerización real |
| Evidencia de tarea (estructurada) | **`OpenEvals/leaderboard-data`** (parquet único, `pd.read_parquet`) + `model-index` del propio README | Un solo fichero, sin API con auth, sin adaptadores por benchmark — la opción más barata de integrar | Uso pragmático de datasets públicos, sin sobreingeniería |
| Evaluación | Golden queries + precision@k/recall@k sobre el filtro y el ranking | Corpus acotado (~37 familias) lo hace factible | Evaluación de sistemas IR |

---

## 3. Catálogo: 5 tareas cerradas, ~37 familias

El usuario solo puede pedir una de estas 5 tareas — si pide otra cosa (incluida clasificación de imágenes, pospuesta), el sistema lo dice explícitamente en vez de forzar un encaje:

| Categoría | Familias | Fuente de curación | Proveedor de VRAM |
|---|---|---|---|
| Chat | ~10 | Catálogo ya curado de canirun.ai (deduplicado, "worth running") | `CanIRunProvider` |
| Código | ~8 | Catálogo de canirun.ai | `CanIRunProvider` |
| Razonamiento | ~6 | Catálogo de canirun.ai | `CanIRunProvider` |
| Embeddings | ~8 | Curación propia vía HF Hub (canirun.ai no las cubre) | `TamañoDirectoProvider` |
| Rerankers | ~5 | Curación propia vía HF Hub | `TamañoDirectoProvider` |
| **Total** | **~37** | | |

**Clasificación de imágenes: pospuesta**, no solo el análisis de imágenes subidas (que ya estaba descartado), sino también el listado/metadata que sí estaba en el MVP. Protege el alcance del mes sin perder nada del núcleo del producto.

Usar el catálogo de canirun.ai como base para chat/código/razonamiento ahorra trabajo de curación en la Semana 1 — ya está deduplicado y filtrado por ellos. Para las otras tres categorías, la curación sigue el método de la sección 4 original (pool amplio por `pipeline_tag` + downloads/likes + revisión manual).

---

## 4. Evidencia: ranking vs. explicación (dos conceptos separados, no se mezclan)

Corrección importante de diseño: **una mención de texto recuperada por RAG no es una medición**. Si Qdrant encuentra un fragmento del README que menciona "código", eso demuestra que existe una mención — no demuestra capacidad de código. Mezclar esto con el ranking numérico rompía la reproducibilidad del sistema (dos ejecuciones podían dar rankings ligeramente distintos según qué fragmento recuperara Qdrant ese día). Se separa en tres conceptos con roles distintos:

| Concepto | Fuentes | ¿Afecta al ranking? |
|---|---|---|
| **Evidencia de ranking** | Compatibilidad VRAM (canirun), runtime declarado, licencia declarada, idiomas declarados, tipo/tarea (pipeline_tag), benchmark (OpenEvals **por match exacto de checkpoint, sin normalización ni extrapolación entre variantes** — peso alto — o model-index — peso menor, por ser autodeclarado), contexto declarado, tamaño/cuantización, checklist mecánico de documentación (secciones presentes, no juicio de calidad) | **Sí** — es la única fuente que puntúa, 100% determinista |
| **Evidencia explicativa** | Qdrant híbrido sobre el README (limitaciones declaradas, casos de uso, menciones de benchmark en prosa) | **No** — se muestra junto al resultado para dar contexto, nunca mueve el score |
| **Información exploratoria externa** | Búsqueda web | **No existe en v1** — ver roadmap |

Con esto el ranking es puramente mecánico y reproducible: misma consulta + mismo catálogo = mismo resultado, siempre. Esto también simplifica el sistema (ya no hace falta normalizar "cuánta evidencia semántica encontré" a un número defendible, que siempre iba a ser el punto más débil del scorer).

**Corrección sobre el matching de benchmarks (importante)**: comprobé el dataset `OpenEvals/leaderboard-data` directamente — `coverage_count` (1-9) y `coverage_percent` (9.1%-81.8%) varían muchísimo entre modelos, y el dataset contiene casos reales de re-subidas de un mismo modelo por distintas organizaciones con filas casi idénticas (ej. `nvidia/...` y `RedHatAI/...` del mismo checkpoint). Un matching por nombre normalizado podía atribuir un benchmark al checkpoint equivocado (base vs. instruct, una cuantización vs. otra, o una resubida de comunidad). Se corrige así: **`model_name` en OpenEvals es literalmente el ID exacto de HF** (ej. `"Qwen/Qwen3.5-397B-A17B"`), así que durante la curación manual de cada familia (Semana 1) se anota a mano el ID exacto del checkpoint concreto que tiene benchmark, sin normalizar ni adivinar. Si la familia no tiene un checkpoint exacto en el dataset, el campo queda vacío — nunca se extrapola de un checkpoint a otro. Preferimos un falso negativo ("sin evidencia suficiente") a un falso positivo (atribuir mal un benchmark).

---

## 5. Compatibilidad de VRAM: diseño desacoplado (importante)

```python
class CompatibilityResult(BaseModel):
    modelo_id: str
    variante: str                 # ej. "Q4_K_M", "7B"
    compatible: bool | None       # None si no se pudo verificar
    grado: str | None             # ej. "runs well", tal cual lo devuelve el proveedor
    fuente: Literal["canirun_ai", "heuristica_propia", "no_disponible"]
    nota: str

class VRAMCompatibilityProvider(Protocol):
    def check(self, modelo_id: str, hardware: HardwarePerfil) -> CompatibilityResult: ...

class CanIRunProvider:
    """Para tarea en {chat, codigo, razonamiento}. Llama a la API pública de canirun.ai.
    Timeout + manejo de error → CompatibilityResult(fuente="no_disponible")
    si el servicio no responde. Requiere tabla de mapeo modelo_id -> id de canirun."""
    ...

class TamañoDirectoProvider:
    """Para tarea en {embeddings, rerankers}. canirun.ai no cubre estas categorías
    (confirmado: su catálogo son solo modelos generativos de texto/imagen/vídeo).
    Estos modelos son pequeños (cientos de MB a ~2GB) y casi nunca están limitados
    por VRAM, así que no hace falta la sofisticación de canirun (cuantización,
    ancho de banda). Fórmula directa desde el tamaño declarado en HF:
    compatible = (tamaño_modelo_gb * 1.2) <= vram_gb disponible."""
    ...

def elegir_provider(tarea: str) -> VRAMCompatibilityProvider:
    if tarea in ("chat", "codigo", "razonamiento"):
        return CanIRunProvider()
    return TamañoDirectoProvider()  # embeddings, rerankers  # embeddings, rerankers
```

Esto dispara la lógica de "no se pudo verificar" en vez de romper el pipeline si el servicio externo falla — coherente con el resto del sistema. El enrutado entre los dos proveedores evita que embeddings/rerankers se queden sistemáticamente sin verificación de VRAM, que es lo que pasaría si dependieras solo de canirun.ai para todo.

**Nota de verificación (importante)**: la API pública de canirun.ai está documentada en su README de GitHub (MIT, repositorio activo) con endpoints, ejemplos de `curl` y esquema de respuesta reales — pero no ha sido ejecutada y confirmada en vivo por falta de capacidad de hacer `POST` durante la fase de diseño. **Verificar esto es la primera tarea de la Semana 1**, antes de construir `CanIRunProvider`. Si la API no responde como se documenta, hay un plan B ya viable: canirun.ai publica en su página `/why` la fórmula completa de su heurística de VRAM (parámetros × bits/peso ÷ 8 ÷ 1024³ + overhead, con multiplicadores por cuantización) y su base de datos de GPUs, todo bajo licencia MIT — reimplementable como `HeuristicaPropiaProvider` citando la metodología como fuente.

---

## 6. Esquema de datos (núcleo)

```python
class FamiliaModelo(BaseModel):
    familia_id: str                  # ej. "qwen2.5-coder"
    variantes: list[str]             # ej. ["0.5B", "1.5B", "7B", "14B", "32B"]
    pipeline_tag: str
    license_declarada: str | None
    license_url: str | None
    idiomas_detectados: list[str] | None
    formatos_disponibles: list[str]
    fuente_readme_hash: str
    fecha_ultima_revision: datetime

class EvidenciaBenchmark(BaseModel):
    """Única fuente que puntúa en el ranking, junto al resto de campos estructurados.
    CORRECCIÓN: no se usa matching por nombre normalizado. El dataset OpenEvals indexa
    por repo_id EXACTO de HF (columna model_name), no por familia — comparar "aproximado"
    arriesga atribuir el benchmark de una variante Instruct a su Base, o de un publisher
    a otro con nombre parecido. En su lugar: tabla de alias exactos, curada a mano junto
    al resto de la ficha de cada familia (ver families_seed.yaml, sección 8)."""
    fuente: Literal["openevals", "model_index"]   # openevals pesa más: no autodeclarado
    benchmark: str
    metrica: str
    valor: float
    cobertura_percent: float | None   # de OpenEvals: cuántos benchmarks del total tiene ese checkpoint (varía 9%-82%)
    nota: str

# Tabla de alias por familia — YAML curado a mano, NO generado por matching automático:
#
# family_id: qwen3-coder-30b
#   benchmark_alias:
#     exact_model_id: "Qwen/Qwen3-Coder-30B-A3B-Instruct"   # exacto contra model_name de OpenEvals
#     source: openevals
#     applicability: exact_checkpoint    # nunca "familia entera"
#
# Reglas aplicadas al construir EvidenciaBenchmark para una familia:
#   - Sin match exacto revisado a mano  -> evidencia_benchmark = None (no se inventa)
#   - Benchmark de la variante Base     -> NO se atribuye a la variante Instruct
#   - Benchmark de una cuantización     -> NO se extrapola a otra cuantización
#   Un falso negativo (decir "sin evidencia" cuando sí la había) es preferible
#   a un falso positivo (atribuir un score al checkpoint equivocado).

class CriteriosBusqueda(BaseModel):
    # Cerrado a 6 tareas — si el usuario pide algo fuera de esta lista, el sistema
    # lo dice explícitamente en vez de intentar encajarlo a la fuerza en una categoría.
    tarea: Literal["chat", "codigo", "razonamiento", "embeddings", "rerankers"] | None  # clasificacion_imagenes pospuesta, ver roadmap
    vram_gb: float | None
    idiomas_requeridos: list[str] = []              # [] = sin restricción
    runtime_preferido: str | None = None            # None = cualquiera
    requiere_licencia_comercial: bool = False

    def campos_faltantes(self) -> list[str]:
        """tarea y vram_gb son obligatorios para que el filtro tenga sentido;
        el resto tiene valores por defecto razonables."""
        faltan = []
        if self.tarea is None: faltan.append("tarea")
        if self.vram_gb is None: faltan.append("vram_gb")
        return faltan

class SubPuntuacion(BaseModel):
    criterio: str                     # ej. "evidencia_codigo", "ajuste_vram", "documentacion"
    valor: float                      # 0-1, normalizado
    peso: float                       # documentado, no oculto
    justificacion: str

class CandidatoRankeado(BaseModel):
    familia: FamiliaModelo
    variante_recomendada: str
    compatibilidad_vram: CompatibilityResult
    evidencia_benchmark: EvidenciaBenchmark | None   # única evidencia que puntúa (sección 4)
    subpuntuaciones: list[SubPuntuacion]             # SOLO datos estructurados, ver sección 4
    puntuacion_total: float
    evidencia_explicativa: list[str]  # citas del README vía Qdrant — NUNCA puntúa, solo contexto (sección 4)

# --- Tipos de turno de conversación (unión discriminada) ---

class TurnoConversacion(BaseModel):
    rol: Literal["usuario", "sistema"]
    contenido: str

# --- 6.1 Estado en servidor: el cliente NUNCA reenvía datos calculados ---
#
# Corrección tras revisión: el diseño anterior dejaba que el cliente reenviara
# criterios_conocidos y candidatos_elegibles (scores, VRAM, benchmarks, evidencia)
# íntegros en cada turno. El servidor los aceptaba sin volver a calcularlos —
# eso rompe la propiedad de reproducibilidad que se cuidó en las secciones 3 y 4
# (el ranking debe ser 100% determinista y controlado por el servidor, no por lo
# que el cliente decida reenviar). Un `conversation_id` único cubre las 3 fases.

class ConversationRun(BaseModel):
    """Vive en SQLite (tabla `conversation_runs`), no en el cliente. TTL de 24h:
    si `created_at` tiene más de 24h al leerla, se trata como no encontrada y se
    empieza una conversación nueva. Limpieza simple: DELETE por fecha al arrancar
    el contenedor, sin necesidad de un scheduler en background."""
    conversation_id: UUID
    snapshot_id: str                                     # versión de families_seed.yaml usada, para reproducibilidad
    criterios: CriteriosBusqueda
    candidatos_elegibles: list[CandidatoRankeado] = []    # pool completo; se rellena una vez, en la Fase B
    historial: list[TurnoConversacion] = []
    created_at: datetime

class PeticionChat(BaseModel):
    conversation_id: UUID | None    # None solo en el primer mensaje de una conversación nueva
    mensaje: str

class ClarificacionNecesaria(BaseModel):
    tipo: Literal["clarificacion"] = "clarificacion"
    conversation_id: UUID           # el servidor lo genera en el primer turno y lo devuelve siempre
    pregunta: str
    campo_faltante: str

class RespuestaRecomendacion(BaseModel):
    tipo: Literal["recomendacion"] = "recomendacion"
    conversation_id: UUID
    criterios_interpretados: CriteriosBusqueda
    candidatos: list[CandidatoRankeado]   # top 3-5 mostrados (el pool completo se queda en el servidor)
    limitaciones: list[str]
    datos_actualizados_hace: str

class RespuestaSeguimiento(BaseModel):
    tipo: Literal["seguimiento"] = "seguimiento"
    conversation_id: UUID
    respuesta: str
    datos_estructurados: list[str]        # de candidatos_elegibles ya calculado en el servidor (VRAM, licencia, benchmark...)
    evidencia_explicativa: list[str]      # citas del README, solo si hace falta recalcular vía Qdrant
    confianza: Literal["alta", "media", "baja"]

RespuestaChat = ClarificacionNecesaria | RespuestaRecomendacion | RespuestaSeguimiento
```

---

## 7. Endpoints

| Método | Ruta | Función |
|---|---|---|
| `GET` | `/health` | Salud de API + índice + Ollama |
| `GET` | `/families` | Lista las ~37 familias curadas, filtrable por las 5 categorías (para depuración/exploración manual) |
| `POST` | `/chat` | `PeticionChat` (`conversation_id` opcional + `mensaje`) → `RespuestaChat` — un único endpoint para las tres fases, el servidor resuelve el `conversation_id` contra SQLite |
| `GET` | `/eval/report` | Última corrida de evaluación |

---

## 8. Estructura de carpetas

```
local-model-advisor/
├── docker/
│   ├── Dockerfile.api
│   └── docker-compose.yml        # único perfil: api + ollama, GPU passthrough
├── data/
│   └── families_seed.yaml        # las ~37 familias curadas (canirun.ai + curación propia), mantenidas a mano
├── src/
│   ├── ingestion/                 # resuelve variantes de cada familia vía HF Hub API
│   │   └── benchmark_evidence.py  # carga OpenEvals (parquet) + parseo de model-index
│   ├── retrieval/
│   │   ├── qdrant_store.py
│   │   └── hybrid_retriever.py    # BaseRetriever custom (evidencia cualitativa)
│   ├── compatibility/
│   │   ├── provider.py            # interfaz VRAMCompatibilityProvider
│   │   ├── canirun_provider.py    # chat/código/razonamiento
│   │   ├── tamano_directo_provider.py  # embeddings/rerankers
│   │   └── id_mapping.yaml        # mapeo modelo_id HF -> id canirun (solo para las 3 categorías cubiertas)
│   ├── ranking/
│   │   └── scorer.py              # suma ponderada, pesos en config, no hardcoded ocultos
│   ├── chains/
│   │   ├── criteria_extraction.py # LCEL: texto libre -> CriteriosBusqueda (Fase A)
│   │   ├── recommend_pipeline.py  # orquesta filtro -> evidencia -> ranking -> redacción (Fase B)
│   │   ├── followup_chat.py       # responde sobre candidatos_elegibles ya calculado (Fase C)
│   │   └── chat_router.py         # decide en qué fase está la conversación, según el ConversationRun cargado
│   ├── state/
│   │   └── conversation_store.py  # CRUD de ConversationRun en SQLite + limpieza por TTL (24h)
│   ├── tools/
│   │   ├── structured_filter.py   # filtra/ordena candidatos_elegibles en memoria (Fase C)
│   │   └── semantic_search.py     # evidencia explicativa, nunca puntúa (Fase B y C)
│   ├── api/
│   └── models/
├── eval/
│   ├── golden_queries.yaml
│   └── run_eval.py
├── tests/
└── README.md                      # incluye nota explícita de dependencia de canirun.ai
```

---

## 9. Plan semanal (4 semanas)

**Semana 0 — Version de prueba**
Primer vertical slice
No implementar el sistema completo por capas semanales sin antes comprobar que el núcleo funciona.

El primer hito debe ser sólo éste:

Entrada:
“Tengo 11 GB de VRAM, uso Ollama,
quiero programar Python en español e inglés.”

Sistema:
1. Extrae criterios con LangChain + Pydantic.
2. Filtra un catálogo pequeño de código.
3. Consulta compatibilidad de hardware.
4. Busca benchmark exacto si existe.
5. Calcula ranking determinista.
6. Devuelve 3 candidatos o menos.
7. Explica cada filtro y limitación.
8. Guarda conversation_id.
No incluyas todavía:

Qdrant.

Hybrid Search.

Reranker.

Seguimiento conversacional.

Embeddings.

Rerankers.

Chat casual.

Docker Compose completo.

GIF.

Evaluación extensa.

Si ese vertical slice funciona pasamos con el proyecto real.


**Semana 1 — Catálogo y datos**
- **Primer paso, antes de nada**: verificar con `curl` real los tres endpoints de canirun.ai (`/api/models`, `/api/compatibility`, `/api/recommend`) contra el esquema documentado en su README. Si no responden como se espera, activar el plan B (heurística propia, ver nota abajo) antes de construir nada sobre esa base.
- Construir la tabla de alias exactos benchmark↔checkpoint (sección 4) a la vez que se curan las ~37 familias — no es un paso aparte, es una columna más del mismo proceso manual.
- Curar `families_seed.yaml` (~37 familias: importar catálogo de canirun.ai para chat/código/razonamiento, curar a mano embeddings/rerankers)
- Resolver metadata real de cada variante vía HF Hub API
- Mapeo de IDs contra canirun.ai + primeras pruebas de su API
- Cargar `OpenEvals/leaderboard-data` + parseo de `model-index` del README (evidencia estructurada barata)

**Semana 2 — Filtro, compatibilidad y evidencia**
- Filtro determinista completo (tarea, idioma, licencia, runtime)
- `CanIRunProvider` con manejo de fallo y timeout
- Índice Qdrant híbrido sobre README + reranking — **solo para evidencia explicativa, no para el ranking**

**Semana 3 — Conversación (3 fases), ranking y pipeline**
- Chain de extracción de criterios + puerta de validación (Fase A: pregunta si faltan `tarea`/`vram_gb`)
- `conversation_store.py`: tabla `conversation_runs` en SQLite + TTL de 24h — el cliente solo maneja `conversation_id` + mensaje, nunca los datos calculados
- Scorer de ranking multi-criterio **usando solo campos estructurados** (sección 4), pesos documentados
- Pipeline completo `recommend_pipeline.py` end-to-end (Fase B): filtro → ranking estructurado → evidencia explicativa (Qdrant) → redacción
- Chat de seguimiento sobre `candidatos_elegibles` ya calculado y guardado en servidor (Fase C)
- `chat_router.py` que decide la fase cargando el `ConversationRun` por `conversation_id`
- Golden queries + evaluación del filtro y del ranking (reproducible al ser 100% estructurado y controlado por el servidor)

**Semana 4 — Empaquetado y demo**
- Docker Compose único (api + ollama)
- README con setup, GIF de demo, nota de dependencia de canirun.ai, roadmap futuro
- Ensayo de demo en vivo

---

## 10. Roadmap futuro (fuera de scope v1, documentado)

- Sustituir `CanIRunProvider` por una heurística propia de VRAM (la interfaz ya está preparada para ello)
- **v1.1 — Búsqueda web con allowlist**: reintroducir información exploratoria externa, pero restringida a dominios de confianza (huggingface.co, ollama.com, canirun.ai, repositorios oficiales, benchmarks oficiales) — nunca sin restricción de dominio, y nunca afectando al ranking, solo como evidencia explicativa adicional marcada como externa
- **Clasificación de imágenes**: pospuesta desde v1 (listado/metadata, sin inferencia) — reincorporar cuando el resto del pipeline esté sólido
- Modelos de visión con inferencia real (VLMs, OCR, análisis de imágenes subidas) — excluido de forma permanente, distinto de la clasificación de imágenes de arriba (que es solo listado)
- Ampliar el catálogo curado más allá de ~37 familias, o automatizar su actualización periódica
