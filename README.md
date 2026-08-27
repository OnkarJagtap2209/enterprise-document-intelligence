# Enterprise Document Intelligence

Enterprise Document Intelligence is a Python project for grounded question
answering over Infosys financial documents.

The planned system uses Docling for structured PDF processing, ChromaDB for
vector storage, semantic and BM25 retrieval combined with Reciprocal Rank
Fusion, an optional cross-encoder reranker, and Gemini for grounded answer
generation. FastAPI and Streamlit will provide the backend and user interface.

## Current status

Phases 1 and 2 establish the Python foundation and reusable Docling PDF
ingestion. Ingestion validates one PDF, preserves Docling's structured document
model and provenance, and writes a deterministic JSON artifact under
`data/extracted/`. Chunking, retrieval, generation, evaluation, API, and UI
behavior are planned but are not implemented yet. Development proceeds one
approved phase at a time.

## Development setup

The project requires Python 3.11 or newer. Install the package in a virtual
environment:

```shell
python -m pip install -e .
```

Copy `.env.example` to `.env` and adjust local paths if needed. Run the tests
with:

```shell
python -m unittest discover -s tests
```

## Ingest one PDF

```shell
python scripts/ingest_documents.py data/documents/q1-26-2027.pdf
```

The output directory can be changed with `EXTRACTED_DIR`. Generated extraction
artifacts are intentionally excluded from Git.
