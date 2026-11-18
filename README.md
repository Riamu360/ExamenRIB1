# Examen ICCD753 — Sistema de Recuperación de Información (NFCorpus)

Sistema de recuperación de información sobre **BeIR/nfcorpus** con modelos **TF-IDF (similitud coseno)** y **Okapi BM25**.

## Contenido del repositorio (`main`)
- `examen1bim.ipynb`: Notebook completo del examen (preprocesamiento, índice invertido, evaluación sobre las 323 consultas de `test`, gráficas comparativas y servicio FastAPI).
- `app.py`: Servicio web REST en FastAPI para despliegue continuo en Render (`GET /`, `GET /health`, `POST /search`, `GET /docs`).
- `nfcorpus_corpus.json.gz`: Corpus `BeIR/nfcorpus` (3,633 documentos) comprimido para arranque rápido y bajo consumo de RAM.
- `requirements.txt` y `render.yaml`: Configuración de dependencias y despliegue en Render.
