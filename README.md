# Enterprise Document Intelligence | Hybrid RAG System

> An end-to-end, evaluation-aware Retrieval-Augmented Generation (RAG) system for grounded question answering over financial documents.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-5B21B6)](https://www.trychroma.com/)
[![RAG](https://img.shields.io/badge/AI-RAG-7C3AED)](#architecture)

## Overview

Enterprise Document Intelligence is a modular hybrid RAG system designed to answer questions from financial PDFs while keeping the answer traceable to retrieved evidence.

Instead of sending a question directly to an LLM, the system follows a controlled pipeline:

**PDF → Docling → Structure-aware chunks → Embeddings/ChromaDB → Query routing → Semantic + BM25 retrieval → RRF → Optional reranking → Context building → Gemini → Citation validation → Answer + sources**

The project was intentionally built as a reusable AI-engineering system rather than as a single notebook or one-off chatbot. API, UI, retrieval, generation, evaluation, and observability are separated into clear modules and reuse the same underlying RAG services.

## Why this project?

Financial documents contain tables, exact terminology, reporting periods, document names, and numerical facts where both semantic similarity and exact lexical matching matter.

The system therefore combines:

- **Semantic retrieval** for meaning and paraphrase matching.
- **BM25 lexical retrieval** for exact terms, financial labels, names, and identifiers.
- **Reciprocal Rank Fusion (RRF)** to combine the two ranked result sets without mixing incompatible raw scores.
- **Optional cross-encoder reranking** to refine the final candidate set.
- **Metadata-aware routing and filtering** to keep document/source/year/content-type constraints aligned with the query.
- **Grounded generation and citation validation** so generated source IDs must correspond to evidence actually supplied to the model.
- **Evaluation and observability** to measure retrieval/answer behavior and diagnose failures instead of relying only on subjective inspection.

---
## System Architecture

The following diagram shows the end-to-end architecture of the Enterprise Document Intelligence system, from document ingestion and indexing to query routing, hybrid retrieval, grounded generation, citation validation, and evaluation/observability.

![Enterprise Document Intelligence Architecture](Arch%26Results_images/Final%20Arch%20Dia.png)

### End-to-end flow

1. **Document ingestion** — PDFs are processed with Docling and converted into a structured representation.
2. **Structure-aware chunking** — text and tables are chunked while preserving section context, pages, metadata, and provenance. Chunk IDs are deterministic.
3. **Embedding and indexing** — Sentence Transformers generates normalized embeddings, which are persisted in ChromaDB.
4. **Query understanding** — a deterministic router extracts document/source, content type, reporting period/year, and other supported constraints and handles genuine clarification cases.
5. **Hybrid retrieval** — semantic vector search and BM25 lexical search produce candidates; RRF combines their rankings.
6. **Optional reranking** — a cross-encoder can rerank the hybrid candidates.
7. **Context construction** — retrieved chunks are deduplicated, ordered, bounded, and passed forward with provenance.
8. **Grounded generation** — Gemini receives the question and retrieved evidence through a controlled prompt.
9. **Citation validation** — returned source IDs are checked against the supplied context; unknown or ambiguous sources are not accepted.
10. **Response** — the system returns the grounded answer together with source information and runtime metadata.
11. **Evaluation/observability** — retrieval metrics, answer-quality checks, traces, latency, usage metadata, and evaluation artifacts are captured separately from the user-facing path.

---

## Key Features

### Document processing

- Docling-based structured PDF processing.
- Structure-aware deterministic chunking.
- Section/headings context preservation.
- Table-aware chunking with row-level structure.
- Page and source provenance.
- Deterministic SHA-256-based chunk IDs.

### Embeddings and vector storage

- Sentence Transformers embeddings.
- Default embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- 384-dimensional normalized embeddings.
- Persistent ChromaDB vector store.
- Idempotent indexing using deterministic chunk IDs.

### Hybrid retrieval

- Dense semantic retrieval.
- BM25 lexical retrieval.
- Singular/plural lexical normalization for common financial terminology variations.
- Reciprocal Rank Fusion (RRF).
- Configurable candidate depth and top-K values.
- Deduplication across retrieval results.

### Query routing

- Deterministic query parsing and resolution.
- Document/source filtering.
- Content-type filtering.
- Reporting year/period handling using authoritative metadata when available.
- Clarification handling for genuinely ambiguous document-selection queries.
- Comparison-intent handling is kept separate from simple document-year selection.

### Reranking

- Optional cross-encoder reranking.
- Default model: `cross-encoder/ms-marco-MiniLM-L6-v2`.
- Reranking operates only on retrieved candidates rather than replacing retrieval.

### Grounded generation

- Gemini-based generation through an isolated generation service.
- Deterministic bounded context construction.
- Evidence-only generation instructions.
- Configurable context limit.
- Structured generation result.
- Strict source-ID validation.
- Narrow, unambiguous alias resolution for known abbreviated citation IDs.
- Unknown or ambiguous source IDs remain invalid.

### API and UI

- FastAPI backend.
- Streamlit user interface.
- Separate API client used by the Streamlit UI.
- Document upload and processing.
- Question answering.
- Source display with document/page provenance.
- Health endpoint.
- Operation-specific client timeouts for health, query, and document processing.
- Server-side exception logging and request IDs for traceability.

### Evaluation and observability

- Retrieval metrics: Hit@K, Precision@K, Recall@K, MRR, and nDCG.
- Answer metrics: correctness, groundedness/faithfulness, expected facts, missing facts, unsupported claims, and source accuracy where scorable.
- Failure-aware evaluation separating execution failures from quality failures.
- Question-level evaluation traces and reports.
- Stage-level latency measurement.
- Request IDs, status, errors, model information, source IDs, and usage metadata.
- Streamlit evaluation dashboard.
- Runtime traces kept separate from evaluation traces.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| PDF processing | Docling |
| Chunking | Custom structure-aware chunker |
| Embeddings | Sentence Transformers |
| Vector store | ChromaDB |
| Lexical retrieval | BM25 |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-Encoder (`ms-marco-MiniLM-L6-v2`) |
| LLM | Google Gemini |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Validation | Pydantic / typed models |
| Testing | Python `unittest` test suite |
| Configuration | Environment-backed settings |

**Note:** The project does **not** use LangChain. The RAG pipeline is implemented with modular Python components to keep retrieval, ranking, context construction, generation, evaluation, and observability explicit and independently testable.

---

## Project Structure

```text
enterprise-document-intelligence/
│
├── config/
│   └── experiments/
│
├── data/
│   ├── documents/          # local input PDFs; ignored from Git
│   ├── extracted/          # Docling artifacts; ignored from Git
│   ├── chunks/             # chunk artifacts; ignored from Git
│   └── evaluation/         # local traces/reports/results as configured
│
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   └── metadata.md
│
├── scripts/
│   ├── chunk_document.py
│   ├── ingest_documents.py
│   ├── rebuild_indexes.py
│   └── run_evaluation.py
│
├── src/
│   └── enterprise_rag/
│       ├── api/
│       ├── application/
│       ├── config/
│       ├── embeddings/
│       ├── evaluation/
│       ├── generation/
│       ├── ingestion/
│       ├── models/
│       ├── observability/
│       ├── reranking/
│       ├── retrieval/
│       ├── routing/
│       ├── stores/
│       └── ui/
│
├── tests/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

### Module responsibilities

- `ingestion/` — PDF validation, Docling processing, chunking, provenance.
- `embeddings/` — reusable embedding service.
- `stores/` — ChromaDB and index boundaries.
- `retrieval/` — semantic, BM25, RRF, hybrid retrieval and deduplication.
- `routing/` — deterministic query parsing, resolution, constraints, and clarification.
- `reranking/` — optional cross-encoder reranking.
- `generation/` — context builder, prompts, Gemini service, and citation validation.
- `application/` — orchestration through `RAGService`, `LazyRAGService`, and `DocumentService`.
- `evaluation/` — datasets, relevance judgments, experiments, metrics, reporting, and failure analysis.
- `observability/` — latency, request tracing, usage, and runtime evidence.
- `api/` — FastAPI routes and request/response schemas.
- `ui/` — Streamlit application and API client.

---

## Installation

### Prerequisites

- Python 3.11 or newer
- Git
- A Google Gemini API key for live generation

### 1. Clone the repository

```bash
git clone https://github.com/OnkarJagtap2209/enterprise-document-intelligence.git
cd enterprise-document-intelligence
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

```bash
python -m pip install --upgrade pip
pip install -e .
```

The repository declares its runtime dependencies in `pyproject.toml`.

---

## Environment Configuration

Create `.env` from `.env.example`.

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Set the Gemini API key in `.env`:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemini-3.6-flash
```

Relevant retrieval configuration includes:

```env
SEMANTIC_TOP_K=5
BM25_TOP_K=10
HYBRID_TOP_K=10
HYBRID_CANDIDATE_DEPTH=10
RRF_K=60
RERANKER_ENABLED=false
RERANKER_CANDIDATE_DEPTH=10
RERANKER_TOP_K=5
CONTEXT_MAX_CHARS=12000
OBSERVABILITY_ENABLED=true
```

**Never commit `.env` or a real API key.** The repository intentionally tracks `.env.example` with an empty key placeholder.

---

## Running the Application

The application is designed as two local processes: a FastAPI backend and a Streamlit frontend. The Streamlit UI communicates with the backend through the API client rather than accessing the RAG internals directly.

### Start FastAPI

From the project root:

```bash
uvicorn enterprise_rag.api.app:app --reload --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

Health check:

```text
GET /health
```

### Start Streamlit

Open a second terminal with the virtual environment activated:

```bash
streamlit run src/enterprise_rag/ui/streamlit_app.py
```

UI:

```text
http://localhost:8501
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Backend health/status |
| `POST` | `/query` | Ask a question through the RAG pipeline |
| `POST` | `/documents` | Upload and process a document |

### Example query request

```json
{
  "query": "What was the revenue for the three months ended June 30, 2026?"
}
```

The exact request and response schemas are defined in `src/enterprise_rag/api/schemas.py`.

---

## Document Ingestion

The repository includes scripts for processing documents in stages.

### Ingest a PDF with Docling

```bash
python scripts/ingest_documents.py data/documents/q1-26-2027.pdf
```

### Chunk an extraction artifact

```bash
python scripts/chunk_document.py data/extracted/<artifact>.docling.json
```

### Build/rebuild indexes

```bash
python scripts/rebuild_indexes.py data/chunks/<artifact>.chunks.json
```

Generated document artifacts, local indexes, uploaded PDFs, and runtime outputs are intentionally excluded from source control where configured by `.gitignore`.

---

## Evaluation

Evaluation is treated as a first-class engineering subsystem rather than a hard-coded score displayed in the UI.

### Retrieval metrics

- **Hit@K** — whether at least one relevant item appears in the top K.
- **Precision@K** — fraction of the top K results judged relevant.
- **Recall@K** — fraction of judged relevant items retrieved in the top K.
- **MRR** — how early the first relevant result appears.
- **nDCG@K** — ranking quality when graded relevance is available.

### Answer-quality checks

The evaluation layer can assess:

- correctness;
- groundedness/faithfulness;
- expected facts;
- missing facts;
- unsupported claims;
- source-document accuracy;
- execution status and failure-aware scoring.

### Important evaluation policy

The project does **not** fabricate retrieval or answer-quality numbers. A retrieval metric requires explicit relevance judgments, and an answer-quality result must be supported by an actual evaluation run.

The latest active evaluation set contained 24 questions. Because Gemini free-tier quota failures interrupted generation, the full benchmark was not treated as a completed quality benchmark. One successfully evaluated question produced retrieval metrics of Hit@5 = 1.0, Precision@5 = 1.0, Recall@5 = 0.625, and MRR = 1.0; these are question-level results, not claims about overall system quality.

Run the evaluation workflow with:

```bash
python scripts/run_evaluation.py
```

Check the repository's evaluation configuration and dataset paths before running a full benchmark.

---

## Validation and Testing

The project was developed and validated incrementally with focused regression tests and end-to-end checks.

Validation included:

- unit/regression test suite;
- Python compilation checks;
- Git whitespace validation with `git diff --check`;
- FastAPI `/health` verification;
- OpenAPI/schema verification;
- real PDF ingestion and indexing;
- retrieval-ranking inspection;
- real API request validation;
- Streamlit UI verification;
- Gemini integration checks when the configured account/model was available.

A documented regression-test progression reached **52 passed, 1 skipped** during the final refinement cycle. External Gemini quota/capacity issues were kept separate from application failures.

Run the test suite with:

```bash
python -m unittest discover -s tests
```

Compile the project with:

```bash
python -m compileall -q src tests
```

Check Git whitespace errors with:

```bash
git diff --check
```

---

## Engineering Problems Solved

The project was built through iterative debugging. Some representative issues and root-cause fixes were:

### 1. Duplicate deterministic chunk IDs after re-upload

Re-uploading the same PDF could cause repeated chunk IDs during BM25 reconstruction. The reconstruction path was made deterministic and idempotent by deduplicating repeated IDs while preserving distinct document/chunk records.

### 2. Retrieval missed the correct revenue chunk

A financial table contained **“Revenues”** while the query used **“revenue”**. BM25 ranked the correct chunk too low, and the hybrid top-5 candidate set excluded it. The fix added deterministic singular/plural normalization and increased the hybrid candidate depth to 10 based on retrieval diagnostics.

### 3. Year filtering used the wrong signal

A query year could be mistaken for a document-selection year. Routing was corrected to distinguish reporting periods from document identity and to use authoritative metadata such as `period_year`/`document_year` when available.

### 4. Multi-document contamination

A query could retrieve evidence from an unintended document when the user explicitly specified a source. Filename/source constraints were propagated into both semantic and BM25 retrieval paths.

### 5. Gemini citation IDs did not always match canonical IDs

Gemini could return an abbreviated identifier. A narrow alias resolver was added only for known, unambiguous IDs; unknown or ambiguous identifiers remain invalid under strict source validation.

### 6. Streamlit timeout during document processing

A generic short client timeout was insufficient for ingestion. Operation-specific timeout budgets were introduced for health checks, queries, and longer document processing operations.

### 7. Generic API failures were difficult to diagnose

Server-side exception logging and runtime tracing were added so failures can be associated with a request ID and pipeline stage instead of being treated as unexplained HTTP errors.

These problems were handled using a repeatable engineering pattern:

**Reproduce → isolate the failing stage → inspect intermediate evidence → identify root cause → make the smallest fix → add regression coverage → run the broader test suite → validate with a real request/UI.**

---

## Design Decisions

### Why hybrid retrieval?

Semantic search handles paraphrases and meaning, while BM25 is strong for exact financial terminology, names, labels, and identifiers. RRF combines their rankings without assuming their raw scores are directly comparable.

### Why not use only an LLM?

The LLM should reason over evidence retrieved from the enterprise documents rather than relying on its pretrained knowledge for document-specific facts.

### Why not use only vector search?

Exact terms and financial labels can be important retrieval signals. BM25 provides a complementary lexical signal that dense retrieval alone may miss.

### Why preserve page and provenance metadata?

A grounded answer is more useful when the user can verify where the evidence came from, including the source document and page.

### Why validate citations?

A model-generated citation should not automatically be trusted. The system checks that returned source IDs belong to the evidence actually supplied to the model.

### Why deterministic chunk IDs?

Stable IDs improve reproducibility, indexing behavior, duplicate detection, and provenance tracking.

### Why deterministic routing instead of an LLM router?

The supported routing constraints—document, source, content type, reporting year/period, and clarification—are safer and easier to test when their behavior is deterministic. This also avoids introducing an additional LLM dependency into query selection.

### Why no LangChain?

The project intentionally implements the RAG pipeline using modular Python components. This keeps retrieval, fusion, reranking, context construction, generation, citation validation, and evaluation explicit and independently testable.

---

## Scalability and Production Path

The current project is primarily a **local, modular AI-engineering implementation**, not a claim of large-scale production deployment.

The architecture provides clear boundaries that could be scaled or replaced independently in a future production environment:

- move ChromaDB to a managed/distributed vector store;
- separate ingestion into asynchronous workers;
- introduce a persistent document/metadata registry;
- scale the FastAPI service independently from the UI;
- add distributed task processing for ingestion/evaluation;
- add authentication, authorization, and document-level access control;
- add managed tracing, metrics, and alerting;
- add model fallback/retry policies and rate-limit handling;
- deploy through containers and an orchestration platform.

These are **future production enhancements**, not capabilities claimed by the current local implementation.

---

## Limitations

- The current deployment uses local ChromaDB storage.
- The representative engineering corpus is small and should not be used to make broad retrieval-quality claims.
- Gemini availability and free-tier quota can affect live generation.
- BM25 lexical normalization is intentionally lightweight; it is not a full stemming/lemmatization system.
- Metadata filtering is strongest when authoritative document/period metadata exists.
- Cross-encoder reranking is optional and can require additional local model resources.
- The current API does not implement authentication/authorization.
- The repository does not implement distributed production infrastructure, managed observability, or enterprise identity integration.

---

## Security and Repository Hygiene

The repository is configured to avoid committing sensitive/local artifacts such as:

```text
.env
.venv/
__pycache__/
local ChromaDB data
uploaded PDFs
extracted/chunk artifacts
runtime traces/results
```

Use `.env.example` as the configuration template and never commit real API keys.

---

## Development Principles

The project follows a few core engineering principles:

- **Modularity** — each RAG stage has a clear responsibility.
- **Determinism where possible** — routing, chunk IDs, context construction, evaluation metrics, and citation validation are designed to be reproducible.
- **Evidence over assumptions** — intermediate retrieval and evaluation evidence is inspected before changing components.
- **Failure-aware evaluation** — execution failures are not silently converted into quality failures.
- **No fabricated metrics** — benchmark numbers are reported only when backed by actual evaluation evidence.
- **Shared pipeline** — API, UI, and evaluation reuse the same application/RAG services rather than maintaining separate RAG implementations.
- **Incremental validation** — changes are tested at the smallest useful boundary and then validated end-to-end.

---

## Future Improvements

### Retrieval

- Better query expansion and rewriting.
- More robust linguistic normalization.
- Adaptive candidate depth.
- Larger and more diverse relevance-judgment datasets.
- Systematic semantic vs. BM25 vs. hybrid vs. reranked comparisons.

### Generation

- Retry/backoff for transient Gemini failures.
- Model fallback.
- Streaming responses.
- Better token/usage monitoring.
- Stronger structured-output recovery.

### Enterprise security

- Authentication and authorization.
- Document-level access control.
- Tenant isolation.
- Audit logging.
- Enterprise secret management.

### Deployment

- CI/CD.
- Containerized production deployment.
- Managed vector infrastructure.
- Asynchronous ingestion workers.
- Managed monitoring and distributed tracing.

---

## Repository

**GitHub:** https://github.com/OnkarJagtap2209/enterprise-document-intelligence

---

## License

See [`LICENSE`](LICENSE) for the repository license.

---

## Author

**Onkar Jagtap**

Built as an AI Engineering project focused on understanding and implementing the core components of an end-to-end hybrid RAG system: **ingestion → retrieval → ranking → generation → citation validation → evaluation → observability**.
