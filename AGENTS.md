# Instrucciones para agentes

## Contexto del proyecto

Local Model Advisor es una API local que recomienda modelos disponibles en canIRun.ai según tarea, VRAM, idioma, runtime y licencia.

La v1 solo admite estas tareas:

- `chat`
- `codigo`
- `razonamiento`

Embeddings, rerankers, clasificación de imágenes, generación de imagen/vídeo, búsqueda web, despliegue cloud y análisis de imágenes están fuera del alcance de la v1, salvo instrucción explícita del usuario.

## Estado actual (Semana 0)

El vertical slice implementado tiene estas limitaciones respecto al diseño final:

- **Catálogo activo**: `families_seed.yaml` con 5 familias de código. No existe `runtime/catalog/` ni `runtime/catalog/current`. No hay sincronización automática desde canIRun.ai.
- **Entrada**: JSON estructurado. El usuario envía `tarea`, `vram_gb`, `idiomas_requeridos`, `runtime_preferido` como campos directos. No se interpreta lenguaje natural.
- **`snapshot_id`**: el campo existe en el esquema `ConversationRun` pero se envía vacío o con valor por defecto — no se persiste porque no hay sincronización que lo genere.
- **Tareas**: solo código funciona correctamente (el seed solo tiene modelos de código). Si se solicita `chat` o `razonamiento`, el sistema debe informar de que el catálogo para esas tareas aún no está sincronizado.
- **Mapeo de IDs**: `CanIRunProvider` usa el campo `canirun_id` del seed YAML. Cuando la sincronización esté implementada, el identificador canónico se conservará directamente desde el catálogo de canIRun.ai, sin tabla de mapeo manual.

## Catálogo sincronizado

- canIRun.ai es la fuente externa de catálogo para la v1.
- El catálogo de producción no se mantiene manualmente familia por familia.
- Un proceso de sincronización obtiene el catálogo desde `GET /api/models`, filtra solo las categorías soportadas, normaliza los registros al modelo interno `FamiliaModelo`, los valida y guarda un snapshot local.
- El proceso se ejecutará externamente cada 21 días; no implementes un scheduler recurrente dentro del proceso FastAPI.
- La API nunca debe descargar el catálogo completo durante `POST /chat`. Debe usar el último snapshot local válido.
- Si la sincronización falla, la API debe conservar y usar el snapshot anterior válido; nunca debe borrar o sustituir el catálogo actual antes de validar completamente el nuevo.
- Las categorías, campos y valores de use case de canIRun.ai deben verificarse contra la respuesta real de la API. No inventes ni hardcodees nombres de campos externos sin documentar el contrato observado.
- La política local versionada define qué categorías externas se incluyen, cómo se mapean a `chat`, `codigo` y `razonamiento`, y qué campos son obligatorios.
- La sincronización solo excluye: categorías fuera de alcance, registros sin identificador canónico válido, duplicados no resolubles, y registros imposibles de normalizar. **No excluye** por tamaño del modelo, licencia desconocida, idiomas desconocidos, runtime desconocido ni falta de benchmark. Estos campos se mantienen como opcionales en el snapshot y se filtran o aplican como limitaciones en tiempo de recomendación, no en tiempo de sincronización.
- El snapshot debe incluir identificador, hash, fecha de sincronización, número de modelos de origen, número de modelos incluidos y motivos de exclusión.
- Cada conversación debe registrar el identificador del snapshot de catálogo usado para preservar reproducibilidad.

## LangChain y Ollama

- LangChain LCEL y ChatOllama solo se usan para convertir texto libre en `CriteriosBusqueda` y redactar preguntas de aclaración.
- Toda salida del LLM debe validarse con Pydantic antes de ejecutar el pipeline.
- El LLM nunca puede calcular puntuaciones, modificar pesos, seleccionar candidatos, eliminar modelos ni alterar el ranking determinista.
- Ollama debe ejecutarse localmente; no enviar prompts, criterios ni datos de conversación a proveedores cloud.

## Principios de arquitectura

- Mantén separadas la evidencia de ranking y la evidencia explicativa.
- El ranking solo puede usar datos estructurados y deterministas disponibles en el snapshot y en la comprobación de compatibilidad: tarea, compatibilidad VRAM, runtime, licencia, idiomas, tamaño/contexto si la fuente los proporciona y checklist mecánico de documentación.
- No asignes benchmarks, idiomas, licencias, repositorios de Hugging Face ni capacidades que no estén presentes en la fuente sincronizada o en una fuente adicional explícitamente aprobada.
- Qdrant, retrieval híbrido y reranking, si se implementan en el futuro, solo aportan evidencia explicativa y nunca modifican la puntuación final.
- El estado de conversación vive en SQLite y se identifica por `conversation_id`.
- El cliente solo envía `conversation_id` y el mensaje nuevo. Nunca debe ser fuente de verdad para criterios, candidatos, scores, compatibilidad VRAM o evidencia.
- Si una dependencia externa falla, devuelve o conserva un estado controlado y explícito; no rompas el pipeline ni inventes datos.
- No añadas dependencias nuevas sin explicar antes por qué son necesarias.

## Implementación

- Usa Python, FastAPI y Pydantic v2.
- Conserva la estructura de carpetas definida en `README.md` y `docs/technical-design.md`.
- Implementa cambios pequeños, cohesionados y verificables.
- Prioriza contratos tipados, separación de responsabilidades e interfaces desacopladas mediante `Protocol` cuando haya proveedores intercambiables.
- El pipeline de recomendación debe depender de una abstracción de catálogo, no de una ruta YAML concreta. Las implementaciones pueden incluir un catálogo semilla para tests y un repositorio de snapshots sincronizados para producción.
- La escritura de un snapshot debe ser atómica: descargar a memoria o archivo temporal, validar, filtrar, normalizar, escribir snapshot y actualizar la referencia `current` solo al final.
- Conserva snapshots históricos durante un período definido y no elimines el último snapshot válido.
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
