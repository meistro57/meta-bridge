# meta-bridge

Synthesis engine for consciousness literature. Sibling project to [kae](https://github.com/meistro57/kae).

Where KAE does archaeology on broad unknown corpora (what's buried here?), Meta Bridge does synthesis on curated known sources (where do these traditions converge, diverge, translate?).

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design.

## Status: Wave 1 scaffold

Minimum pipeline: **PDF in → JSON claims out.** No Qdrant, no dedup, no bridges yet. The goal at this stage is to produce claim extractions we can eyeball and judge for quality before committing to schema.

## Requirements

- Go 1.22+
- `pdftotext` (poppler-utils) on PATH — for PDF ingestion
- OpenRouter API key

```bash
sudo apt install poppler-utils   # if not already installed
```

## Install

```bash
git clone <this repo>
cd meta-bridge
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
go mod tidy
go build -o mb ./cmd/mb
```

## Usage

```bash
# Dry run: first 3 chunks only, so you can sanity-check prompts before burning tokens
MB_MAX_CHUNKS=3 ./mb ingest /path/to/oversoul7.pdf

# Full run with metadata
MB_SOURCE_ID=roberts_oversoul7 \
MB_TITLE="The Education of Oversoul Seven" \
MB_AUTHOR="Jane Roberts" \
./mb ingest /path/to/oversoul7.pdf
```

Output lands in `./output/`:
- `<source_id>.source.json` — source metadata
- `<source_id>.chunks.json` — all chunks with indices
- `<source_id>.claims.json` — extracted claims with attributions back to chunks

## Layout

```
cmd/mb/              CLI entry point
internal/
  source/            Source object (a single ingested work)
  chunker/           Text chunking (chapter/paragraph boundary)
  claim/             Claim + Attribution objects
  llm/               Minimal OpenRouter client
  extractor/         LLM-driven claim extraction
```

## Ingestion waves

Current: **Wave 1** — schema validation on a single source (Oversoul Seven).

Next: **Wave 2** — doctrinal core (Seth Speaks, Nature of Personal Reality, Law of One). Adds dedup, cross-source Attribution merging, first convergence bridges.

See `ARCHITECTURE.md` for the full roadmap.
