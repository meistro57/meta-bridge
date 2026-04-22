# meta-bridge

**Synthesis engine for consciousness literature.**

Meta Bridge extracts atomic claims from curated esoteric, channeled, and experiential sources, measures cross-tradition convergence, and builds typed bridges between esoteric and mainstream conceptual spaces. It surfaces where independent sources agree, where they diverge, and where a claim has no corroboration outside a single tradition.

Sibling project to [KAE](https://github.com/meistro57/kae). Where KAE does archaeology on broad unknown corpora (*what's buried here?*), Meta Bridge does synthesis on curated known sources (*where do these traditions converge, diverge, or translate?*).

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design — core objects, bridge taxonomy, ingest pipeline, and query primitives.

---

## Status: Wave 1

Current focus: **PDF in → JSON claims out.**

The pipeline chunks source texts, classifies segments, and runs LLM-driven claim extraction. Output is human-readable JSON you can review before committing to the full schema. Qdrant storage, dedup, and bridge detection come in Wave 2.

| Wave | Scope | Status |
|------|-------|--------|
| 1 | Schema validation — single source (*Oversoul Seven*) | **Active** |
| 2 | Doctrinal core — Seth Speaks, Nature of Personal Reality, Law of One | Planned |
| 3 | Session-structured material — Cannon's Convoluted Universe v1–5 | Planned |
| 4 | Breadth — Bashar, Pleiadian material, Cassiopaeans, ROOT ACCESS | Planned |
| 5 | Meta-layer — GPT conversation history for human-annotated bridge seeds | Planned |

---

## Requirements

- Go 1.22+
- Python 3.10+ (for reflection and report scripts)
- `pdftotext` (poppler-utils) on PATH
- OpenRouter API key
- Qdrant instance (local or remote) — required from Wave 2 onward
- Ollama with `gemma4` and `nomic-embed-text` models — for Python reflection passes

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
pip install -r requirements.txt   # qdrant-client, requests, etc.
```

---

## Usage

### Go pipeline — claim extraction

```bash
# Dry run: first 3 chunks only — sanity-check prompts before burning tokens
MB_MAX_CHUNKS=3 ./mb ingest /path/to/oversoul7.pdf

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

### Python scripts — reflection and reporting

```bash
# reflect.py — Gemma 4 passes over chunks in Qdrant and emits structured reflections
# (summary, concepts, claims, tone, questions, echoes per chunk)
python reflect.py                  # full run, resumes from last position
python reflect.py --limit 20       # process first 20 new chunks only
python reflect.py --workers 3      # concurrent Gemma calls (default 2)
python reflect.py --from-scratch   # wipe meta_reflections and start over

# meta_report.py — synthesis report from Qdrant results via Ollama
python meta_report.py
```

---

## Layout

```
cmd/mb/              CLI entry point
internal/
  source/            Source object (a single ingested work)
  chunker/           Text chunking — chapter/paragraph boundary detection
  claim/             Claim + Attribution types
  llm/               Minimal OpenRouter client
  extractor/         LLM-driven claim extraction
  store/             Qdrant persistence layer (Wave 2)
reflect.py           Gemma 4 reflection pass over Qdrant chunks
meta_report.py       Synthesis report generator
```

---

## Core concepts

**Claim** — an atomic proposition extracted from one or more sources. The dedup layer (Wave 2) collapses phrasings to a single node. Each claim carries attributions (source, quote, claim type, confidence) and, once bridges are built, convergence scores and cross-tradition links.

**Bridge** — a typed, scored link between two claims. Six types: Convergence, Dramatization, Translation, Evolution, Contradiction, Dependency. Bridges are the product; everything else is scaffolding.

**Independence score** — how independent are the sources supporting a claim? Two sources in the same tradition, same decade, same channel type do not count as independent corroboration. The score is a function of tradition-diversity × region-diversity × decade-span × channel-type-diversity.

The full object schemas, bridge taxonomy, and query primitives are in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Required. Used for LLM extraction passes. |
| `MB_SOURCE_ID` | Stable ID for the source being ingested (e.g. `seth_speaks`). |
| `MB_TITLE` | Source title. |
| `MB_AUTHOR` | Human author/scribe. |
| `MB_MAX_CHUNKS` | Limit extraction to first N chunks. Useful for prompt tuning. |
