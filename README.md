# Fieldnotes — Hybrid Search RAG

Search internal text and Markdown documents using vector retrieval plus BM25, merge results with reciprocal-rank fusion, rerank with Cohere, and generate cited answers with Claude. Includes a responsive web UI, FastAPI endpoints, persistent local corpus, tests, and a Docker image definition.

## Run locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 3000
```

Open **http://localhost:3000**. Click **Load sample handbook**, then ask “How do we respond to a production incident?” API documentation is at `/docs`.

The default **demo mode requires no credentials**. It combines BM25 with deterministic hashed lexical vectors, applies rank fusion, and returns evidence excerpts with citations. It does not run semantic embeddings, neural reranking, or a language model. The UI labels this mode explicitly.

## Cloud mode

1. Create a Pinecone dense index with **1024 dimensions** and cosine similarity. Copy its HTTPS host.
2. Set `RAG_MODE=cloud` in `.env`, plus `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_HOST`.
3. The defaults use Cohere `embed-english-v3.0`, Cohere `rerank-v3.5`, and Claude `claude-sonnet-4-5`. Override model names in `.env` as needed. Changing embedding models/dimensions requires a compatible fresh index/namespace and re-ingestion.
4. Restart and ingest your documents. Use a different `DATA_DIR` and `PINECONE_NAMESPACE` for each corpus/mode. Do not reuse a demo corpus in cloud mode.

Cloud requests incur provider charges. Provider integration tests use mocks; live cloud operation requires your own credentials and has not been verified against live providers here.

### Retrieval pipeline

1. Split text into 220-word chunks with 40-word overlap.
2. Embed in batches, store vectors in Pinecone, and persist the text corpus locally.
3. Retrieve the top 20 vector matches and top 20 BM25 matches from that corpus.
4. Fuse ranks using `sum(1 / (60 + rank))`.
5. Rerank fused candidates with Cohere and keep the requested top k.
6. Send evidence to Claude with instructions to cite each factual claim.
7. Return the answer, numbered source excerpts, and a citation reference check.

**Citation validation checks reference existence, not semantic entailment or complete claim coverage.** Review evidence for important decisions. Prompt instructions reduce, but cannot eliminate, document prompt injection.

## API

```bash
curl -X POST http://localhost:3000/documents \
  -H 'Content-Type: application/json' \
  -d '{"source":"ops.md","text":"Incident commanders send status updates every 30 minutes."}'

curl -X POST http://localhost:3000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How often are incident updates sent?","top_k":5}'
```

`GET /documents` lists sources and chunk count. Re-ingesting an identical source name replaces its chunks. Text/Markdown only; no PDF parsing or OCR. Requests are limited to 200,000 document characters and 4,000 question characters.

## Storage and access

`data/chunks.json` stores document text, and demo mode also stores vectors there. Keep the data directory on persistent storage. It is excluded from Git. Cloud mode additionally sends document text to Cohere and Claude, and vectors with text metadata to Pinecone.

For network exposure, set `APP_API_KEY` and enter it in the UI, or send it as the `X-API-Key` header. The UI keeps the key only in memory. Health and UI routes are public. Use TLS and a reverse proxy with request size/rate limits when hosting. This starter is single-user and must run as **one process/worker**: it uses an in-process lock and a JSON file, not a shared database. Upstream writes and local persistence are not transactional; back up the corpus and retry ingestion after failures. Pinecone query visibility can lag briefly after writes.

The project calls provider HTTP APIs directly to keep the retrieval logic inspectable; LangChain is not required.

## Tests

```bash
ruff check .
pytest -q
```

Tests cover chunk overlap, rank fusion, ingestion/replacement/persistence, authentication, validation, provider call contracts, and invalid citations.

## Docker

```bash
docker build -t hybrid-search-rag .
docker run --rm -p 3000:8000 --env-file .env -v rag-data:/srv/data hybrid-search-rag
```

## Publish to GitHub later

From this directory, after logging in:

```bash
gh auth login
gh repo create hybrid-search-rag --private --source=. --remote=origin --push
```

No API keys, document data, or `.env` files should be committed.
