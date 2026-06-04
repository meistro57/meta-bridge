# meta-bridge
<img width="630" height="600" alt="image" src="https://github.com/user-attachments/assets/b4dcacb3-4b64-43d8-a16b-e319ac38bc8d" />

> **Synthesis engine for consciousness literature.**

[![Go 1.22](https://img.shields.io/badge/Go-1.22-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Wave 1 Active](https://img.shields.io/badge/Wave%201-Active-22c55e)](./ARCHITECTURE.md)

Meta Bridge extracts atomic claims from curated esoteric, channeled, and experiential sources, measures cross-tradition convergence, and builds typed bridges between esoteric and mainstream conceptual spaces. It surfaces where independent sources agree, where they diverge, and where a claim has no corroboration outside a single tradition.

Sibling project to [KAE](https://github.com/meistro57/kae). Where KAE does archaeology on broad unknown corpora (*what's buried here?*), Meta Bridge does synthesis on curated known sources (*where do these traditions converge, diverge, or translate?*).

---

## What It Does

The pipeline ingests a PDF, chunks the text, classifies segments by type, and runs an LLM-driven extraction pass that produces typed, attributed atomic claims. Bridges — scored links between claims — measure convergence across traditions, flag contradictions, and translate esoteric vocabulary into scientific or philosophical equivalents. An epistemic scoring layer (the **Reality Filter**) ensures the bridge catalog distinguishes structural mappings from speculative analogies.

The output is not a search engine. It is a **bridge catalog**.

---

## Roadmap

| Wave | Scope | Status |
|------|-------|--------|
| **1** | Schema validation — *Oversoul Seven* + academic ingest harness | **Active** |
| 2 | Doctrinal core — Seth Speaks, Nature of Personal Reality, Law of One | Planned |
| 3 | Session-structured material — Cannon's Convoluted Universe v1–5 | Planned |
| 4 | Breadth — Bashar, Pleiadian material, Cassiopaeans, ROOT ACCESS | Planned |
| 5 | Meta-layer — GPT conversation history for human-annotated bridge seeds | Planned |

---

## What's New — Qdrant Hybrid Search Upgrade

- Added idempotent payload indexing utility: `add_payload_indexes.py`.
  - `mb_claims`: `attributions[].source_id`, `tags[]`, `entity_type`, `editorial_status`, `chapter`
  - `mb_chunks`: `source_id`, `chapter`
  - `mb_sources`: `source_id`, `tradition`
- Added full-text payload index on `mb_claims.canonical_statement` with `TokenizerType.WORD` for exact-term fallback retrieval.
- Added reusable hybrid retrieval module: `hybrid_search.py`.
  - Dense semantic search (OpenRouter embeddings, `google/gemini-embedding-001` by default)
  - Sparse keyword fallback (`MatchText` over `canonical_statement`)
  - Reciprocal Rank Fusion (RRF, `k=60`) to combine dense + sparse rankings
  - Optional `source_filter` support to constrain by source IDs
  - Returns top-N results with canonical statement, attribution source IDs/surface quotes, and tags

---

## Quick Start

```bash
git clone https://github.com/meistro57/meta-bridge
cd meta-bridge
cp .env.example .env          # set OPENROUTER_API_KEY at minimum
go mod tidy
go build -o mb ./cmd/mb

# Dry run — first 3 chunks, no tokens burned
MB_MAX_CHUNKS=3 ./mb ingest /path/to/source.pdf
```

---

## Requirements

- **Go 1.22+** — main ingest pipeline
- **Python 3.10+** — reflection, scoring, and report scripts
- **`pdftotext`** (poppler-utils) on PATH
- **OpenRouter API key** — required for LLM extraction and/or embeddings
- **Qdrant** — local or remote instance; required from Wave 2 onward, used now for reflection/report passes
- **Ollama** — only required when using `ollama:` model selectors

```bash
sudo apt install poppler-utils
```

---

## Installation

### Go pipeline

```bash
go mod tidy
go build -o mb ./cmd/mb
```

### Python scripts

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # qdrant-client, requests, langgraph, etc.
```

---

## Usage

### Claim extraction

```bash
# Dry run — first 3 chunks only
MB_MAX_CHUNKS=3 ./mb ingest /path/to/source.pdf

# OpenRouter extraction + OpenRouter embeddings
MB_MODEL=google/gemma-4-31b-it \
MB_EMBED_PROVIDER=openrouter \
MB_EMBED_MODEL=openai/text-embedding-3-small \
./mb ingest /path/to/source.pdf

# Local Ollama extraction + OpenRouter embeddings
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
| `<source_id>.source.json` | Source metadata — title, author, tradition, channel type |
| `<source_id>.chunks.json` | All chunks with indices and type classifications |
| `<source_id>.claims.json` | Extracted claims with attributions back to chunks |

---

### Academic PDF ingestion

For textbooks and section-numbered PDFs:

```bash
# Go harness — writes to live Qdrant collections (mb_chunks, mb_claims, mb_sources)
go run ./cmd/mb-academic-test incoming/textbook.pdf

# Python harness — writes to _test collections (mb_chunks_test, mb_sources_test)
python3 academic_ingest_test.py incoming/textbook.pdf
```

The academic chunker detects numbered section headers (`1.2.3 Title`), `Chapter N` headers, filters TOC lines and running page headers, and keeps equations grouped with surrounding prose.

---

### Hybrid search utilities

```bash
# Ensure Qdrant payload/text indexes exist (safe to re-run)
python add_payload_indexes.py

# Run bundled CLI test query and print top 5 fused results
python hybrid_search.py
```

Programmatic usage:

```python
from hybrid_search import hybrid_search

results = hybrid_search(
    "volunteer souls struggling with third density linear reality",
    top_k=10,
    source_filter=["root_access", "seth_speaks"],
)
```

---

### Reflection and reporting

```bash
# Full reflection pass, resumes via source-hash checkpoint dedup
python reflect.py

# Target a specific model
python reflect.py --model ollama:gemma4:latest          # local Ollama
python reflect.py --model google/gemma-4-31b-it         # OpenRouter

# Override embedding provider
MB_EMBED_PROVIDER=ollama MB_EMBED_MODEL=nomic-embed-text:latest python reflect.py

# Partial run options
python reflect.py --limit 20 --workers 3 --from-scratch
python reflect.py --source-collections mb_claims,mb_chunks
MB_REFLECT_SOURCE_COLLECTIONS=mb_chunks,mb_claims python reflect.py

# Interactive menu — model / workers / limit / from-scratch / collections
python reflect.py --pick-collections

# LangGraph loop runner
python reflect_loop.py --limit 20
python reflect_loop.py --loop-interval 60 --max-loops 0   # run forever

# Synthesis report from Qdrant results
python meta_report.py
```

---

### Reality Filter — epistemic bridge scoring

Classifies every bridge with a `bridge_type`, `constraint_flags`, `testability_score`, and plain-language `epistemic_status`.

```bash
# Score all bridges in a report file
python scoring/reality_filter.py findings/vectoreology_2026-04-28.json

# Dry run — print prompts, no LLM calls
python scoring/reality_filter.py --dry-run findings/vectoreology.json

# Override model
python scoring/reality_filter.py --model google/gemma-4-31b-it findings/vectoreology.json
```

**Bridge types:** `Structural` · `Analogical` · `Speculative` · `Testable` · `Contradictory`

**Epistemic statuses:** `Established mapping` · `Plausible analogy` · `Suggestive but ungrounded` · `Poetic metaphor` · `Needs disambiguation` · `Likely invalid` · `Contradicted by evidence`

Output: `<report>.scored.json` — per-bridge scores plus a summary block.

---

### Corpus audit and cleanup

```bash
# Audit only — no changes
python corpus_cleanup.py

# Fix everything non-destructive in one shot (reconcile counts + delete Zone.Identifier files + orphans)
python corpus_cleanup.py --fix-all

# Individual fix modes
python corpus_cleanup.py --fix-counts    # reconcile source chunk/claim counts
python corpus_cleanup.py --fix-zones     # delete Zone.Identifier ghost files
python corpus_cleanup.py --fix-orphans   # delete orphaned chunks

# Write full JSON report
python corpus_cleanup.py --report
```

---

### Utilities

```bash
# Force-rebuild all binaries
./rebuild.sh

# List all source IDs currently in Qdrant
python get_sources.py

# Run tests
python -m pytest tests/ -v
```

---

## Core Concepts

**Claim** — an atomic proposition extracted from one or more sources. Each claim carries attributions (source, quote, claim type, confidence) and, once bridges are built, convergence scores and cross-tradition links. The dedup layer (Wave 2) collapses phrasings to a single node.

**Bridge** — a typed, scored link between two claims. Six types: `Convergence`, `Dramatization`, `Translation`, `Evolution`, `Contradiction`, `Dependency`. Bridges are the product; everything else is scaffolding.

**Independence score** — how independent are the sources supporting a claim? Two sources in the same tradition, same decade, and same channel type do not count as independent corroboration. The score is a function of tradition-diversity × region-diversity × decade-span × channel-type-diversity.

**Reality Filter** — an epistemic scoring layer applied to bridges. It classifies each bridge's epistemic status and flags physics/logic constraint violations, ensuring the bridge catalog distinguishes structural mappings from speculative analogies.

Full object schemas, bridge taxonomy, and query primitives: [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Graphability Scoring

Meta Bridge uses a **Graphability Index** to predict which chunks are worth sending to the LLM for claim extraction — and which are boilerplate noise. The approach is adapted from the [Proxy-Pointer RAG technique](https://towardsdatascience.com/proxy-pointer-rag-eliminating-wasteful-entity-relations-extraction-in-knowledge-graphs/) for Knowledge Graph ingestion cost reduction.

The index scores each chunk by *relational density* — how many actionable semantic edges the section is likely to produce — not raw entity count. A chapter packed with character names scores lower than one containing cosmological law statements.

### Score tiers

| Score | Action | Examples |
|-------|--------|----------|
| `very_high` | Always extract | Session transcripts, Law statements, Soul origin passages, Cosmological hierarchy, Density/octave descriptions |
| `high` | Extract | Practitioner commentary, Entity taxonomy, Higher-self mechanics, Sacred site descriptions |
| `medium` | Extract | Author reflections, Case narratives, Thematic summaries, Historical context |
| `low` | **Skip** | Author bio notes, Session setup, Procedural instructions, Personal anecdotes |
| `very_low` | **Skip** | Copyright pages, Dedications, Table of contents, Bibliography, Publisher forewords |
| `unknown` | Mandatory scan + flag | Section not in index — coverage gap, reported to `misfit_reports` |

### Corpus anchors

Certain section types are **always** `very_high` regardless of apparent density — they define the ontological inheritance structure for everything else. These include Law of One session openings, Seth chapter headers, Dolores Cannon chapter openings (form-feed boundaries), Urantia Papers, Upanishads, Book of the Dead spells, and species profile entries from the alien races encyclopedia.

### Usage

Enabled by default. The scorer loads `graphability_index.json` from the project root on startup.

```bash
# Normal run — graphability scoring active (default: skip low + very_low)
./mb ingest /path/to/source.pdf

# Raise the floor — only extract high + very_high
MB_GRAPHABILITY_MIN=high ./mb ingest /path/to/source.pdf

# Disable entirely — scan everything (old behaviour)
MB_SKIP_GRAPHABILITY=1 ./mb ingest /path/to/source.pdf

# Use a custom index for a specific corpus
MB_GRAPHABILITY_INDEX=/path/to/custom_index.json ./mb ingest /path/to/source.pdf
```

At the end of each ingest run, the summary line reports savings:

```
graphability:  12 skipped / 35 extracted / 4 gaps (25.5% token saving)
```

### Coverage gaps

Chunks whose chapter label doesn't match any index entry are logged as `[graphability=GAP]` and flagged in the `mb_chunks` Qdrant payload with `graphability_gap=true`. After a few ingest runs, review the gap log and add recurring generic sections to `graphability_index.json`. The index stabilizes quickly — target under 20% gaps after ~10 books.

The index file lives at `graphability_index.json` in the project root and is plain JSON — edit it directly to tune coverage for your corpus.

---

## Project Layout

```
cmd/
  mb/                       CLI entry point — main ingest pipeline
  mb-academic-test/         Academic PDF ingest harness (writes to live Qdrant)
internal/
  source/                   Source object (a single ingested work)
  chunker/                  Text chunking — chapter/paragraph + academic section detection
  claim/                    Claim + Attribution types
  llm/                      Minimal OpenRouter client
  extractor/                LLM-driven claim extraction
  graphability/             Graphability scorer — label + content keyword scoring
  store/                    Qdrant persistence layer (Wave 2)
graphability_index.json     Consciousness corpus graphability index (edit to tune coverage)
corpus_cleanup.py           Corpus audit and repair tool — count reconciliation, orphan detection, Zone.Identifier cleanup
rebuild.sh                  Force-rebuild all binaries (runs go build -a)
scoring/
  reality_filter.py         Epistemic bridge scoring (type, flags, testability)
  reality_filter_prompt.md  LLM rubric for the Reality Filter
tests/
  test_reality_filter.py    Reality Filter unit tests
mcp/                        MCP server dependencies (Qdrant, Redis)
reflect.py                  Reflection pass over Qdrant chunks
reflect_loop.py             LangGraph stateful/timer reflection loop runner
meta_report.py              Synthesis report generator
add_payload_indexes.py      Idempotent payload/text index provisioning for Qdrant
hybrid_search.py            Dense+sparse hybrid claim retrieval with RRF fusion
academic_ingest_test.py     Python academic ingest to Qdrant _test collections
get_sources.py              List ingested source IDs from Qdrant
reflect_failures/           Failed reflection JSON artifacts (for debugging)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | Required for OpenRouter extraction and/or embeddings |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant base URL |
| `QDRANT_API_KEY` | — | Optional Qdrant API key |
| `MB_MODEL` | — | Chat model — `ollama:<model>` for Ollama, otherwise an OpenRouter model ID |
| `MB_FILTER_MODEL` | `MB_MODEL` | Override model for `reality_filter.py` |
| `MB_EMBED_PROVIDER` | `openrouter` | Embedding backend — `openrouter` or `ollama` |
| `MB_EMBED_MODEL` | `openai/text-embedding-3-small` | Embedding model name |
| `EMBEDDING_MODEL` | `google/gemini-embedding-001` | Embedding model for `hybrid_search.py` |
| `MB_EMBED_MAX_CHARS` | `8000` | Max chars per embedding payload |
| `MB_HEADER_PATTERN` | auto | Regex override for chapter/session/part header detection |
| `MB_SOURCE_ID` | — | Stable ID for the source being ingested (e.g. `seth_speaks`) |
| `MB_TITLE` | — | Source title |
| `MB_AUTHOR` | — | Human author or scribe |
| `MB_MAX_CHUNKS` | — | Limit extraction to first N chunks — useful for prompt tuning |
| `MB_REFLECT_SOURCE_COLLECTIONS` | `mb_chunks,mb_claims` | Default source collections for `reflect.py` when no CLI override is provided |
| `MB_REFLECT_SOURCE_FILTER` | — | Optional comma-separated source partition filter for `reflect.py` workers |
| `MB_OUTPUT_DIR` | `./output` | Output directory for `.source.json`, `.chunks.json`, `.claims.json` |
| `MB_GRAPHABILITY_INDEX` | `./graphability_index.json` | Path to graphability index JSON |
| `MB_GRAPHABILITY_MIN` | `medium` | Minimum score tier to extract — `very_high`, `high`, or `medium` |
| `MB_SKIP_GRAPHABILITY` | — | Set to `1` to disable scoring and extract all chunks |
