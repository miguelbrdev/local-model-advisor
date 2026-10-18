# Instrucciones para agentes

## Contexto del proyecto

Local Model Advisor es una API local que recomienda modelos disponibles en canIRun.ai según tarea, VRAM, idioma, runtime y licencia.

La v1 solo admite estas tareas:

- `chat`
- `codigo`
- `razonamiento`

Embeddings, rerankers, clasificación de imágenes, generación de imagen/vídeo, búsqueda web, despliegue cloud y análisis de imágenes están fuera del alcance de la v1, salvo instrucción explícita del usuario.

## Estado actual (Semana 1–2)

> **Pipeline servido por el catálogo sincronizado. `scripts/sync_catalog.py` listo para la tarea programada del SO.**

- **Catálogo en el pipeline**: `src/catalog/` define el `Protocol CatalogoModelos` y el adaptador `CatalogoSincronizado`, con cadena de fallback `runtime/catalog/current.json` → `data/catalog_bootstrap.json` → `CatalogUnavailableError`. `main.py` monta una única instancia por proceso en `lifespan` (`app.state.catalogo`) y la inyecta en `ejecutar_pipeline(criterios, *, catalogo, top_n)`. Si ninguna fuente resuelve, el arranque falla con ruido; no hay modo degradado. El catálogo semilla (`families_seed.yaml`, `seed_loader.py`) **se ha eliminado**: no es nivel de fallback.
- **Sincronización y persistencia**: implementadas y testeadas en `src/sync/`. Primer snapshot real validado `snap-176d40242fd0157a` (104 modelos de origen, 91 incluidos, 13 excluidos por `out_of_scope`), promovido a `data/catalog_bootstrap.json` conservando su `snapshot_id`.
- **Sincronización programada (cada 21 días)**: `python -m scripts.sync_catalog` ejecuta un ciclo completo (descarga, validación, snapshot atómico, retención y, opcionalmente, promoción al bootstrap con `--promote-bootstrap`). Códigos de salida: 0 snapshot activo, 1 fallo de sincronización con el snapshot anterior conservado, 2 fallo de un paso posterior. La tarea del SO (`schtasks`/`cron`) la registra el operador; los comandos están en el README. Hasta ahora la sincronización se ha ejecutado manualmente una única vez.
- **Tareas**: `chat`, `codigo` y `razonamiento` funcionan. El filtro lee `tareas_soportadas` (lista completa de tareas del modelo), no `pipeline_tag` (solo la primera): con el bootstrap pasan 73, 51 y 59 familias respectivamente antes de la comprobación VRAM.
- **Entrada**: JSON estructurado. El usuario envía `tarea`, `vram_gb`, `idiomas_requeridos`, `runtime_preferido` como campos directos en `mensaje`. No se interpreta lenguaje natural. Un `mensaje` que no valide como `CriteriosBusqueda` devuelve `422` con los errores (`loc`/`msg`/`type`), nunca un 500. El `historial` de la conversación es acumulativo: cada turno añade la entrada del usuario y la respuesta del sistema.
- **`snapshot_id`**: se persiste en cada `ConversationRun` a partir de `app.state.catalogo.snapshot_id`.
- **Licencias**: `src/tools/licencias.py` es la única fuente de normalización (`"Apache 2.0"` → `apache-2.0`) y del conjunto permisivo; filtro y scorer la comparten. Antes discrepaban y 48 de 91 familias puntuaban como restrictivas.
- **Scorer** (`src/ranking/scorer.py`, `PESOS` suman 1.0): `ajuste_vram` = margen sobre la VRAM del usuario (`1 − requerida/disponible`); `evidencia_codigo` = 1.0 solo si `tareas_soportadas == ["codigo"]` (6 de 91), 0.3 en el resto; `tamanio` sobre escala absoluta de 32 GB; `licencia` vía `licencias.py`; `documentacion` sigue siendo un marcador de posición (licencia declarada + nº de idiomas). Valor neutro 0.5 cuando falta el dato.
- **Limitaciones comunicadas**: `POST /chat` devuelve en `limitaciones` las carencias actuales (sin benchmarks, sin evidencia explicativa, sin idiomas en la fuente, runtime no aplicado). `datos_actualizados_hace` se deja vacío hasta que el `Protocol` exponga la fecha del snapshot.
- **Mapeo de IDs**: `canirun_id` se conserva directamente desde el catálogo sincronizado; no hay tabla de mapeo manual. Las familias sincronizadas no tienen `variantes`; el proveedor reporta `parametros_texto` (p. ej. `"7B"`) como variante.

## Catálogo sincronizado

- canIRun.ai es la fuente externa de catálogo para la v1.
- El catálogo de producción no se mantiene manualmente familia por familia.
- Un proceso de sincronización obtiene el catálogo desde `GET /api/models`, filtra solo las categorías soportadas, normaliza los registros al modelo interno `FamiliaModelo`, los valida y guarda un snapshot local.
- El proceso se ejecuta externamente cada 21 días mediante `scripts/sync_catalog.py`; no implementes un scheduler recurrente dentro del proceso FastAPI. La API recoge el snapshot nuevo en su siguiente arranque (una instancia de catálogo por proceso).
- La API nunca debe descargar el catálogo completo durante `POST /chat`. Debe usar el último snapshot local válido.
- Si la sincronización falla, la API debe conservar y usar el snapshot anterior válido; nunca debe borrar o sustituir el catálogo actual antes de validar completamente el nuevo.
- Las categorías, campos y valores de use case de canIRun.ai deben verificarse contra la respuesta real de la API. No inventes ni hardcodees nombres de campos externos sin documentar el contrato observado.
- La política local versionada define qué categorías externas se incluyen, cómo se mapean a `chat`, `codigo` y `razonamiento`, y qué campos son obligatorios.
- La sincronización solo excluye: categorías fuera de alcance, registros sin identificador canónico válido, duplicados no resolubles, y registros imposibles de normalizar. **No excluye** por tamaño del modelo, licencia desconocida, idiomas desconocidos, runtime desconocido ni falta de benchmark. Estos campos se mantienen como opcionales en el snapshot y se filtran o aplican como limitaciones en tiempo de recomendación, no en tiempo de sincronización.
- **La fuente no expone idiomas**: el contrato observado de `GET /api/models` solo tiene `id`, `name`, `provider`, `family`, `params`, `paramsBillions`, `architecture`, `releaseDate`, `contextLength`, `useCase`, `url` y `license` — **no hay campo de idiomas**. Por eso `normalizer.py` fija `idiomas_detectados=None` en todos los registros sincronizados. Consecuencia: el filtro por `idiomas_requeridos` no descarta nada hoy (solo actúa si la familia declara idiomas) y la limitación se comunica en la respuesta.
- **El runtime no se filtra**: `runtime_preferido` se acepta en la petición y se pasa a `CanIRunProvider.check(...)`, pero ese método no lo lee (el endpoint de compatibilidad observado no tiene parámetro de runtime). El scorer tampoco tiene criterio de runtime. En la práctica el runtime de la petición no influye en el resultado; se comunica como limitación.
- El snapshot debe incluir identificador, hash, fecha de sincronización, número de modelos de origen, número de modelos incluidos y motivos de exclusión. **Implementado**: `SnapshotMeta` en `src/sync/snapshot.py` con escritura atómica (archivo temporal + `os.replace`) y validación round-trip testeada.
- Al promover un snapshot a `data/catalog_bootstrap.json` se conserva el `snapshot_id` original dentro del `CatalogSnapshot` (`snap-176d40242fd0157a`); **no** se genera un id `seed-<hash>` para el bootstrap.
- Cada conversación registra el identificador del snapshot de catálogo usado para preservar reproducibilidad (`ConversationRun.snapshot_id`).
- Suite actual: **340 tests** (`python -m pytest`) + `python -m ruff check .` (reglas `E F I N W UP B SIM`) en verde.

## LangChain y Ollama

- LangChain LCEL y ChatOllama solo se usan para convertir texto libre en `CriteriosBusqueda` y redactar preguntas de aclaración.
- Toda salida del LLM debe validarse con Pydantic antes de ejecutar el pipeline.
- El LLM nunca puede calcular puntuaciones, modificar pesos, seleccionar candidatos, eliminar modelos ni alterar el ranking determinista.
- Ollama debe ejecutarse localmente; no enviar prompts, criterios ni datos de conversación a proveedores cloud.

## Principios de arquitectura

- Mantén separadas la evidencia de ranking y la evidencia explicativa.
- El ranking solo puede usar datos estructurados y deterministas disponibles en el snapshot y en la comprobación de compatibilidad: tarea, compatibilidad VRAM, licencia, idiomas y tamaño/contexto si la fuente los proporciona. Concretamente hoy el scorer lee `tareas_soportadas`, `license_declarada`, `idiomas_detectados`, `vram_required_gb` y la `vram_gb` del usuario (`PESOS` en `src/ranking/scorer.py`); el subcriterio `documentacion` sigue siendo un marcador de posición, y `runtime_preferido`, `contexto_maximo` y `tokens_per_second` se aceptan pero no puntúan.
- No asignes benchmarks, idiomas, licencias, repositorios de Hugging Face ni capacidades que no estén presentes en la fuente sincronizada o en una fuente adicional explícitamente aprobada.
- Qdrant, retrieval híbrido y reranking, si se implementan en el futuro, solo aportan evidencia explicativa y nunca modifican la puntuación final.
- El estado de conversación vive en SQLite y se identifica por `conversation_id`.
- El cliente solo envía `conversation_id` y el mensaje nuevo. Nunca debe ser fuente de verdad para criterios, candidatos, scores, compatibilidad VRAM o evidencia.
- Si una dependencia externa falla, devuelve o conserva un estado controlado y explícito; no rompas el pipeline ni inventes datos.
- No añadas dependencias nuevas sin explicar antes por qué son necesarias.

## Implementación

- Usa Python, FastAPI y Pydantic v2.
- Conserva la estructura de carpetas definida en `README.md`.
- Implementa cambios pequeños, cohesionados y verificables.
- Prioriza contratos tipados, separación de responsabilidades e interfaces desacopladas mediante `Protocol` cuando haya proveedores intercambiables.
- El pipeline de recomendación depende de la abstracción `CatalogoModelos` (`src/catalog/protocol.py`), nunca de una ruta concreta. Producción usa `CatalogoSincronizado`; los tests inyectan dobles en memoria. No reintroduzcas un catálogo semilla como fuente paralela.
- La escritura de un snapshot debe ser atómica: descargar a memoria o archivo temporal, validar, filtrar, normalizar, escribir snapshot y actualizar la referencia `current` solo al final.
- Conserva snapshots históricos durante un período definido y no elimines el último snapshot válido. **Implementado** en `src/sync/retention.py`: 90 días por defecto, los snapshots `active`/`previous` del manifiesto nunca se borran y sin manifiesto legible no se borra nada.
- Usa `async` solo cuando la librería o la operación de E/S lo justifique.

## Documentación del código

- Documenta con docstrings en formato Google las clases, módulos, funciones públicas, métodos públicos y endpoints cuyo propósito no sea inequívoco.
- Cada docstring debe indicar propósito, argumentos, retorno, excepciones relevantes, efectos secundarios y decisiones no obvias.
- En código de dominio crítico, documenta invariantes, fuente de datos, supuestos, fallbacks y limitaciones.
- Mantén los docstrings actualizados cuando cambie el comportamiento.
- No añadas docstrings redundantes a getters, setters, propiedades triviales o funciones de una línea con nombres completamente claros.
- Los comentarios explican el porqué; el código y los nombres explican el qué.

## Calidad y validación

- Añade o actualiza tests para cada comportamiento nuevo o modificado.
- Los tests de sincronización deben usar respuestas HTTP mockeadas; nunca deben depender de canirun.ai, red, hora real o archivos persistentes fuera de `tmp_path`.
- Prueba como mínimo: respuesta externa válida, timeout/error HTTP, campos externos inesperados, filtrado por categoría, deduplicación, conversión a modelo interno, actualización atómica y conservación del snapshot anterior ante fallo.
- No cambies contratos públicos de API sin actualizar modelos Pydantic, tests y documentación.
- Antes de terminar una tarea, ejecuta `python -m pytest` y `python -m ruff check .`.
- Si un comando falla por una limitación del entorno, indícalo claramente junto con el error y no afirmes que la validación ha pasado.
- Resume siempre los archivos modificados, las decisiones tomadas, los comandos ejecutados y las limitaciones pendientes.
