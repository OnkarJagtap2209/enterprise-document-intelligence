# Enterprise Document Intelligence

Enterprise Document Intelligence is a Python project for grounded question
answering over Infosys financial documents.

The planned system uses Docling for structured PDF processing, ChromaDB for
vector storage, semantic and BM25 retrieval combined with Reciprocal Rank
Fusion, an optional cross-encoder reranker, and Gemini for grounded answer
generation. FastAPI and Streamlit will provide the backend and user interface.

## Current status

Phases 1 through 5 establish the Python foundation, reusable Docling PDF
ingestion, deterministic structure-aware chunking, and a persistent ChromaDB
vector index with semantic retrieval. Ingestion writes Docling JSON under
`data/extracted/`; chunking preserves headings, pages, provenance, and
structured table rows under `data/chunks/`. The indexing layer embeds chunk
content in batches and upserts it with deterministic chunk IDs. Semantic
retrieval embeds a query once and returns typed results in ChromaDB cosine-
distance order. Hybrid retrieval, generation, evaluation, API, and UI behavior
are planned but are not implemented yet. Development proceeds one approved
phase at a time.

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

## Chunk one extraction artifact

```shell
python scripts/chunk_document.py data/extracted/<artifact>.docling.json
```

Chunk size, fallback overlap, and output location are configured with
`CHUNK_MAX_CHARS`, `CHUNK_OVERLAP_CHARS`, and `CHUNKS_DIR`.

## Index one chunk artifact

```shell
python scripts/rebuild_indexes.py data/chunks/<artifact>.chunks.json
```

The default `sentence-transformers/all-MiniLM-L6-v2` model is compact enough
for practical local CPU development and produces 384-dimensional embeddings.
It is a starting default, not a claim of optimal retrieval quality. Configure
it, batch size, collection name, and persistent location with
`EMBEDDING_MODEL_NAME`, `EMBEDDING_BATCH_SIZE`, `CHROMA_COLLECTION_NAME`, and
`CHROMA_DB_PATH`. Re-indexing the same artifact safely replaces records with
the same Phase 3 chunk IDs.

The semantic retriever accepts query text and an optional `top_k`, defaulting
to `SEMANTIC_TOP_K=5`. Results expose raw cosine distance, where lower values
represent nearer vectors; no score conversion or reranking is applied.
