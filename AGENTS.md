\# Instrucciones para agentes



\## Contexto del proyecto



Local Model Advisor es una API local que recomienda modelos de Hugging Face

según tarea, VRAM, idioma, runtime y licencia.



La v1 solo admite estas tareas:



\- chat

\- codigo

\- razonamiento

\- embeddings

\- rerankers



No implementar clasificación de imágenes, búsqueda web, despliegue cloud ni

análisis de imágenes en v1 salvo que se solicite explícitamente.



\## Principios de arquitectura



\- Mantén separadas la evidencia de ranking y la evidencia explicativa.

\- El ranking solo puede usar datos estructurados y deterministas:

&#x20; compatibilidad VRAM, runtime, licencia, idioma, tarea, benchmark exacto,

&#x20; contexto, tamaño/cuantización y checklist documental.

\- Qdrant, retrieval híbrido y reranking solo aportan evidencia explicativa;

&#x20; nunca deben modificar la puntuación final.

\- No extrapoles benchmarks entre checkpoints, variantes, cuantizaciones,

&#x20; modelos Base e Instruct ni repositorios de Hugging Face distintos.

\- Si no existe una coincidencia exacta de benchmark, representa la ausencia

&#x20; de evidencia; no infieras un valor.

\- El estado de conversación vive en SQLite y se identifica por

&#x20; `conversation\_id`.

\- El cliente solo envía `conversation\_id` y el mensaje nuevo. Nunca debe

&#x20; enviar ni ser fuente de verdad para criterios, candidatos, scores,

&#x20; compatibilidad VRAM o evidencia.

\- Si una dependencia externa falla, devuelve un resultado controlado y

&#x20; explícito; no rompas el pipeline ni inventes datos.

\- No añadas dependencias nuevas sin explicar antes por qué son necesarias.



\## Implementación



\- Usa Python, FastAPI y Pydantic v2.

\- Conserva la estructura de carpetas definida en `README.md`.

\- Implementa cambios pequeños, cohesionados y verificables.

\- No implementes funcionalidades de semanas posteriores hasta que el

&#x20; vertical slice de la Semana 0 funcione.

\- Prioriza contratos tipados, separación de responsabilidades e interfaces

&#x20; desacopladas mediante `Protocol` cuando haya proveedores intercambiables.

\- Usa `async` solo cuando la librería o la operación de E/S lo justifique.



\## Documentación del código



\- Documenta con docstrings en formato Google las clases, módulos, funciones

&#x20; públicas, métodos públicos y endpoints cuyo propósito no sea inequívoco.

\- Cada docstring debe indicar propósito, argumentos, retorno, excepciones

&#x20; relevantes, efectos secundarios y decisiones no obvias.

\- En código de dominio crítico, documenta también invariantes, fuentes de

&#x20; datos, supuestos y limitaciones.

\- Mantén los docstrings actualizados cuando cambie el comportamiento.

\- No añadas docstrings redundantes a getters, setters, propiedades triviales

&#x20; o funciones de una línea con nombres completamente claros.

\- Los comentarios explican el porqué; el código y los nombres explican el qué.



\## Calidad y validación



\- Añade o actualiza tests para cada comportamiento nuevo o modificado.

\- No cambies contratos públicos de API sin actualizar modelos Pydantic,

&#x20; tests y documentación.

\- Antes de terminar una tarea, ejecuta los comandos de formato, lint,

&#x20; comprobación de tipos y tests que estén configurados en el proyecto.

\- Si un comando falla por una limitación del entorno, indícalo claramente

&#x20; junto con el error y no afirmes que la validación ha pasado.

\- Resume siempre los archivos modificados, las decisiones tomadas, los

&#x20; comandos ejecutados y las limitaciones pendientes.

