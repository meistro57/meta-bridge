# meta-bridge

**Synthesis engine for consciousness literature.**

Meta Bridge extracts atomic claims from curated esoteric, channeled, and experiential sources, measures cross-tradition convergence, and builds typed bridges between esoteric and mainstream conceptual spaces. It surfaces where independent sources agree, where they diverge, and where a claim has no corroboration outside a single tradition.

Sibling project to [KAE](https://github.com/meistro57/kae). Where KAE does archaeology on broad unknown corpora (*what's buried here?*), Meta Bridge does synthesis on curated known sources (*where do these traditions converge, diverge, or translate?*).

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design — core objects, bridge taxonomy, ingest pipeline, and query primitives.

---

## Status: Wave 1

Current focus: **PDF in → JSON claims out, with epistemic scoring of bridges.**

The pipeline chunks source texts (including academic/textbook PDFs), classifies segments, runs LLM-driven claim extraction, and scores resulting bridges through a Reality Filter for epistemic rigor. Qdrant storage and full bridge detection come in Wave 2.

| Wave | Scope | Status |
|------|-------|--------|
| 1 | Schema validation — single source (*Oversoul Seven*) + academic ingest harness | **Active** |
| 2 | Doctrinal core — Seth Speaks, Nature of Personal Reality, Law of One | Planned |
| 3 | Session-structured material — Cannon's Convoluted Universe v1–5 | Planned |
| 4 | Breadth — Bashar, Pleiadian material, Cassiopaeans, ROOT ACCESS | Planned |
| 5 | Meta-layer — GPT conversation history for human-annotated bridge seeds | Planned |

---

## Requirements

- Go 1.22+
- Python 3.10+ (for reflection, scoring, and report scripts)
- `pdftotext` (poppler-utils) on PATH
- OpenRouter API key (required when using OpenRouter chat and/or embeddings)
- Qdrant instance (local or remote) — required from Wave 2 onward; used now for reflection/report passes
- Ollama (required when using `ollama:` chat models and/or `MB_EMBED_PROVIDER=ollama`)

```bash
sudo apt install poppler-utils
```

---

## Install

```bash
git clone <this repo>
cd meta-bridge
cp .env.example .env
# edit .env — set OPENROUTER_API_KEY at minimum
go mod tidy
go build -o mb ./cmd/mb
```

For the Python scripts:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # qdrant-client, requests, langgraph, etc.
```

---

## Usage

### Go pipeline — claim extraction

```bash
# Dry run: first 3 chunks only — sanity-check prompts before burning tokens
MB_MAX_CHUNKS=3 ./mb ingest /path/to/oversoul7.pdf

# OpenRouter Gemma extraction + OpenRouter embeddings (default embedding provider)
MB_MODEL=google/gemma-4-31b-it \
MB_EMBED_PROVIDER=openrouter \
MB_EMBED_MODEL=openai/text-embedding-3-small \
./mb ingest /path/to/source.pdf

# Local Gemma 4 extraction + OpenRouter embeddings
MB_MODEL=ollama:gemma4 \
MB_EMBED_PROVIDER=openrouter \
MB_EMBED_MODEL=openai/text-embedding-3-small \
./mb ingest /path/to/source.pdf

# Full run with source metadata
MB_SOURCE_ID=roberts_oversoul7 \
MB_TITLE="The Education of Oversoul Seven" \
MB_AUTHOR="Jane Roberts" \
./mb ingest /path/to/source.pdf
```

Output lands in `./output/`:

| File | Contents |
|------|----------|
| `<source_id>.source.json` | Source metadata (title, author, tradition, channel type) |
| `<source_id>.chunks.json` | All chunks with indices and type classifications |
| `<source_id>.claims.json` | Extracted claims with attributions back to chunks |

### Academic PDF ingestion

For textbooks and section-numbered PDFs, use the academic chunker:

```bash
# Go harness (writes to live Qdrant collections: mb_chunks, mb_claims, mb_sources)
go run ./cmd/mb-academic-test incoming/textbook.pdf

# Python harness (writes to _test collections: mb_chunks_test, mb_sources_test only)
python3 academic_ingest_test.py incoming/textbook.pdf
```

The academic chunker detects numbered section headers (`1.2.3 Title`), `Chapter N` headers, filters TOC lines and running page headers, and keeps equations grouped with surrounding prose.

### Python scripts — reflection and reporting

```bash
# reflect.py uses the same provider env vars as mb ingest:
#   MB_MODEL (ollama:<model> or OpenRouter model id)
#   MB_EMBED_PROVIDER (openrouter|ollama)
#   MB_EMBED_MODEL
python reflect.py                                         # full run, resumes
python reflect.py --model ollama:gemma4:latest           # local Ollama chat
python reflect.py --model google/gemma-4-31b-it          # OpenRouter chat
MB_EMBED_PROVIDER=ollama MB_EMBED_MODEL=nomic-embed-text:latest python reflect.py
python reflect.py --limit 20 --workers 3 --from-scratch
python reflect.py --source-collections mb_claims,mb_chunks
python reflect.py --pick-collections                     # interactive menu: model/workers/limit/from-scratch/collections

# reflect_loop.py — LangGraph loop runner
python reflect_loop.py --limit 20
python reflect_loop.py --loop-interval 60 --max-loops 0  # run forever on a timer

# meta_report.py — synthesis report from Qdrant results via Ollama
python meta_report.py
```

### Reality Filter — epistemic bridge scoring

Classifies bridges between knowledge clusters with epistemic rigor, assigning each a `bridge_type`, `constraint_flags`, `testability_score`, and plain-language `epistemic_status`.

```bash
# Score all bridges in a vectoreology JSON report
python scoring/reality_filter.py findings/vectoreology_2026-04-28.json

# Dry run — print prompts without calling LLM
python scoring/reality_filter.py --dry-run findings/vectoreology.json

# Override model
python scoring/reality_filter.py --model google/gemma-4-31b-it findings/vectoreology.json
```

Bridge types: `Structural | Analogical | Speculative | Testable | Contradictory`

Epistemic statuses: `Established mapping | Plausible analogy | Suggestive but ungrounded | Poetic metaphor | Needs disambiguation | Likely invalid | Contradicted by evidence`

Output is `<report>.scored.json` with per-bridge scores plus a summary block.

### Utilities

```bash
# List all source IDs currently in Qdrant (mb_sources collection)
python get_sources.py
```

### Tests

```bash
python -m pytest tests/ -v
```

---

## Layout

```
cmd/
  mb/                    CLI entry point (main pipeline)
  mb-academic-test/      Academic PDF ingestion test harness (writes to live Qdrant)
internal/
  source/                Source object (a single ingested work)
  chunker/               Text chunking — chapter/paragraph + academic section detection
  claim/                 Claim + Attribution types
  llm/                   Minimal OpenRouter client
  extractor/             LLM-driven claim extraction
  store/                 Qdrant persistence layer (Wave 2)
scoring/
  reality_filter.py      Epistemic scoring of bridges (bridge type, flags, testability)
  reality_filter_prompt.md  LLM rubric for the reality filter
tests/
  test_reality_filter.py Reality filter unit tests
mcp/                     MCP server dependencies (qdrant, redis)
reflect.py               Gemma 4 reflection pass over Qdrant chunks
reflect_loop.py          LangGraph stateful/timer reflection loop runner
meta_report.py           Synthesis report generator
academic_ingest_test.py  Python academic ingest to Qdrant _test collections
get_sources.py           List ingested source IDs from Qdrant
reflect_failures/        Failed reflection JSON artifacts (for debugging)
```

---

## Core concepts

**Claim** — an atomic proposition extracted from one or more sources. The dedup layer (Wave 2) collapses phrasings to a single node. Each claim carries attributions (source, quote, claim type, confidence) and, once bridges are built, convergence scores and cross-tradition links.

**Bridge** — a typed, scored link between two claims. Six types: Convergence, Dramatization, Translation, Evolution, Contradiction, Dependency. Bridges are the product; everything else is scaffolding.

**Independence score** — how independent are the sources supporting a claim? Two sources in the same tradition, same decade, same channel type do not count as independent corroboration. The score is a function of tradition-diversity × region-diversity × decade-span × channel-type-diversity.

**Reality Filter** — an epistemic scoring layer applied to bridges. It classifies each bridge's epistemic status and flags physics/logic constraint violations, ensuring the bridge catalog distinguishes structural mappings from speculative analogies.

The full object schemas, bridge taxonomy, and query primitives are in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Required for OpenRouter extraction and/or embeddings. |
| `OLLAMA_URL` | Ollama base URL (default: `http://localhost:11434`). |
| `QDRANT_URL` | Qdrant base URL (default: `http://localhost:6333`). |
| `QDRANT_API_KEY` | Optional Qdrant API key. |
| `MB_MODEL` | Chat model selector used by `mb ingest` and `reflect.py` (`ollama:<model>` for Ollama, otherwise OpenRouter model id). |
| `MB_FILTER_MODEL` | Override model for `scoring/reality_filter.py` (falls back to `MB_MODEL`). |
| `MB_EMBED_PROVIDER` | Embedding backend used by `mb ingest` and `reflect.py`: `openrouter` (default) or `ollama`. |
| `MB_EMBED_MODEL` | Embedding model name used by `MB_EMBED_PROVIDER` (default: `openai/text-embedding-3-small`). |
| `MB_EMBED_MAX_CHARS` | Max chars for source/chunk embedding payloads (default: `8000`). |
| `MB_HEADER_PATTERN` | Optional regex override for chapter/session/part header detection (auto-detected by default). |
| `MB_SOURCE_ID` | Stable ID for the source being ingested (e.g. `seth_speaks`). |
| `MB_TITLE` | Source title. |
| `MB_AUTHOR` | Human author/scribe. |
| `MB_MAX_CHUNKS` | Limit extraction to first N chunks. Useful for prompt tuning. |
| `MB_OUTPUT_DIR` | Output directory for `.source.json`, `.chunks.json`, `.claims.json` (default: `./output`). |
