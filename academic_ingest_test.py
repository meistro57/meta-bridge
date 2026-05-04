#!/usr/bin/env python3
"""
academic_ingest_test.py — Test academic PDF chunking into Qdrant.

Creates _test collections (mb_chunks_test, mb_sources_test) and ingests
a textbook PDF using section-aware chunking. Does NOT touch production
collections. Does NOT run claim extraction (that's the Go pipeline's job).

Usage:
    python3 academic_ingest_test.py incoming/02110tpnews_11232020.pdf

Requires:
    pip install requests
    pdftotext (poppler-utils) on PATH

Reads .env for OPENROUTER_API_KEY, QDRANT_URL, MB_EMBED_MODEL.
"""

import json
import hashlib
import os
import re
import subprocess
import sys
import time
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_env(path=".env"):
    """Minimal .env loader."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if not os.environ.get(key):
                os.environ[key] = val


load_env()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
EMBED_MODEL = os.environ.get("MB_EMBED_MODEL", "openai/text-embedding-3-small")
EMBED_MAX_CHARS = int(os.environ.get("MB_EMBED_MAX_CHARS", "8000"))

# Test collection names — mirrors production but with _test suffix
COL_CHUNKS = "mb_chunks_test"
COL_SOURCES = "mb_sources_test"

# Chunker settings
TARGET_TOKENS = 600
MAX_TOKENS = 1200

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        result = subprocess.run(
            ["pdftotext", "-layout", path, "-"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftotext failed: {result.stderr}")
        return result.stdout
    elif ext in (".txt", ".md", ""):
        with open(path) as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported extension: {ext}")


# ---------------------------------------------------------------------------
# Academic chunker
# ---------------------------------------------------------------------------

# Patterns
_toc_re = re.compile(r"\.\s+\.(\s+\.)+\s*\d+\s*$")
_page_header_re = re.compile(r"^\s*\d+\s+CHAPTER\s+\d+\.\s+[A-Z\s\-]+\s*$")
_chapter_re = re.compile(r"^Chapter\s+(\d+)\s*$", re.IGNORECASE)
_section_re = re.compile(r"^(\d+(?:\.\d+)+)\s+([A-Z][A-Za-z].{0,120}?)$")
_roman_re = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)


def _est_tokens(s: str) -> int:
    return len(s) // 4


def _clean(line: str) -> str:
    return line.strip().lstrip("\f").strip()


def chunk_academic(text: str, target=TARGET_TOKENS, maxtok=MAX_TOKENS,
                   fallback="Front Matter", skip_sections=None):
    """
    Section-aware chunker for academic textbooks.

    Returns list of dicts: {index, chapter, text, token_est}
    """
    skip_lower = {s.lower() for s in (skip_sections or [])}
    lines = text.replace("\r\n", "\n").split("\n")

    chunks = []
    cur_lines = []  # accumulated paragraph texts
    cur_tokens = 0
    current_section = fallback
    cur_section = fallback
    idx = 0

    def flush():
        nonlocal cur_lines, cur_tokens, idx, cur_section
        if not cur_lines:
            return
        joined = "\n\n".join(cur_lines).strip()
        tok = _est_tokens(joined)
        if tok < 10:
            cur_lines = []
            cur_tokens = 0
            return
        chunks.append({
            "index": idx,
            "chapter": cur_section,
            "text": joined,
            "token_est": tok,
        })
        idx += 1
        cur_lines = []
        cur_tokens = 0

    def append_para(para_text):
        nonlocal cur_lines, cur_tokens, cur_section
        ptok = _est_tokens(para_text)

        # Oversized single paragraph -> own chunk
        if ptok > maxtok and not cur_lines:
            cur_section = current_section
            cur_lines.append(para_text)
            cur_tokens = ptok
            flush()
            return

        # Would exceed target -> flush first
        if cur_tokens + ptok > target and cur_lines:
            flush()

        if not cur_lines:
            cur_section = current_section
        cur_lines.append(para_text)
        cur_tokens += ptok

    i = 0
    while i < len(lines):
        line = lines[i]
        clean = _clean(line)

        if not clean:
            i += 1
            continue

        # Skip TOC, page headers, roman numerals
        if _toc_re.search(line):
            i += 1
            continue
        if _page_header_re.match(line) or _page_header_re.match(clean):
            i += 1
            continue
        if _roman_re.match(clean) and len(clean) < 8:
            i += 1
            continue

        # Chapter start
        m = _chapter_re.match(clean)
        if m:
            flush()
            # Peek for title
            title = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                peek = _clean(lines[j])
                if peek and not _section_re.match(peek) and len(peek) < 80:
                    title = peek
            current_section = f"Chapter {m.group(1)}: {title}" if title else f"Chapter {m.group(1)}"
            cur_section = current_section
            i += 1
            continue

        # Numbered section header
        m = _section_re.match(clean)
        if m and len(clean) < 120 and not _toc_re.search(line):
            sec_title = m.group(2).strip()
            # Check skip list
            if any(s in sec_title.lower() for s in skip_lower):
                i += 1
                while i < len(lines):
                    pk = _clean(lines[i])
                    if _section_re.match(pk) or _chapter_re.match(pk):
                        break
                    i += 1
                continue
            flush()
            current_section = f"{m.group(1)} {sec_title}"
            cur_section = current_section
            i += 1
            continue

        # Accumulate paragraph
        para = []
        while i < len(lines):
            cl = _clean(lines[i])
            if not cl:
                break
            if _chapter_re.match(cl):
                break
            if _section_re.match(cl) and len(cl) < 120 and not _toc_re.search(lines[i]):
                break
            if _page_header_re.match(lines[i]) or _toc_re.search(lines[i]):
                i += 1
                continue
            para.append(cl)
            i += 1

        if para:
            append_para("\n".join(para))

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def qdrant_req(method, endpoint, body=None):
    url = QDRANT_URL + endpoint
    headers = {"Content-Type": "application/json"}
    resp = requests.request(method, url, json=body, headers=headers, timeout=30)
    return resp.status_code, resp.json() if resp.content else {}


def ensure_collection(name, vector_size):
    status, _ = qdrant_req("GET", f"/collections/{name}")
    if status == 200:
        print(f"  collection {name} already exists")
        return
    body = {"vectors": {"size": vector_size, "distance": "Cosine"}}
    status, resp = qdrant_req("PUT", f"/collections/{name}", body)
    if status >= 400:
        raise RuntimeError(f"Failed to create {name}: {status} {resp}")
    print(f"  created collection {name}")


def upsert_point(collection, point_id, vector, payload):
    body = {"points": [{"id": point_id, "vector": vector, "payload": payload}]}
    status, resp = qdrant_req("PUT", f"/collections/{collection}/points?wait=true", body)
    if status >= 400:
        raise RuntimeError(f"Upsert failed in {collection}: {status} {resp}")


# ---------------------------------------------------------------------------
# Embedding via OpenRouter
# ---------------------------------------------------------------------------

def embed(text: str) -> list:
    """Get embedding via OpenRouter."""
    resp = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": EMBED_MODEL, "input": text[:EMBED_MAX_CHARS]},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding failed: {resp.status_code} {resp.text[:200]}")
    data = resp.json()
    return data["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Point ID generation (matches Go pipeline's FNV-1a hash)
# ---------------------------------------------------------------------------

def fnv1a_64(s: str) -> int:
    """FNV-1a 64-bit hash, matching Go's hash/fnv.New64a."""
    h = 0xcbf29ce484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def point_id_for_key(key: str) -> int:
    return fnv1a_64(key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 academic_ingest_test.py <path-to-pdf-or-txt>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Source ID from filename
    base = os.path.splitext(os.path.basename(path))[0]
    source_id = re.sub(r"[^a-z0-9_]+", "_", base.lower()).strip("_") or "source"

    _mc = os.environ.get("MB_MAX_CHUNKS", "").strip()
    max_chunks = int(_mc) if _mc else None

    print(f"[1/4] Extracting text from {path}")
    text = extract_text(path)
    print(f"       {len(text)} chars (~{len(text)//4} tokens)")

    print(f"[2/4] Chunking (academic mode)")
    chunks = chunk_academic(text)
    print(f"       {len(chunks)} chunks")

    if max_chunks:
        print(f"       MB_MAX_CHUNKS={max_chunks}, limiting")
        chunks = chunks[:max_chunks]

    # Quick stats
    toks = [c["token_est"] for c in chunks]
    sections = set(c["chapter"] for c in chunks)
    print(f"       {len(sections)} unique sections")
    print(f"       tokens: min={min(toks)} max={max(toks)} avg={sum(toks)//len(toks)}")

    print(f"[3/4] Creating test collections and embedding")
    # Get vector size from a test embedding
    test_vec = embed("test")
    vec_size = len(test_vec)
    print(f"       vector size: {vec_size}")

    ensure_collection(COL_CHUNKS, vec_size)
    ensure_collection(COL_SOURCES, vec_size)

    # Upsert source
    source_embed_text = f"{source_id}\n\n{text[:EMBED_MAX_CHARS]}"
    source_vec = embed(source_embed_text)
    source_payload = {
        "id": source_id,
        "title": base,
        "author": "Nouredine Zettili",
        "entity_type": "source",
        "source_path": path,
        "chunk_count": len(chunks),
        "claim_count": 0,
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "medium": "",
        "channel_type": "",
        "tradition": "",
        "region": "",
        "year": 2009,
    }
    upsert_point(COL_SOURCES, point_id_for_key(f"source:{source_id}"),
                 source_vec, source_payload)
    print(f"       source upserted: {source_id}")

    # Upsert chunks
    print(f"[4/4] Upserting {len(chunks)} chunks")
    start = time.time()
    errors = 0
    for i, ch in enumerate(chunks):
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"       chunk {i+1}/{len(chunks)}  "
                  f"section={ch['chapter'][:50]!r}  "
                  f"~{ch['token_est']} tok  "
                  f"({rate:.1f} chunks/sec)")

        try:
            vec = embed(ch["text"][:EMBED_MAX_CHARS])
        except Exception as e:
            print(f"       ! embed error chunk {i}: {e}")
            errors += 1
            continue

        payload = {
            "entity_type": "chunk",
            "source_id": source_id,
            "index": ch["index"],
            "chapter": ch["chapter"],
            "text": ch["text"],
            "token_est": ch["token_est"],
        }

        key = f"{source_id}_chunk_{ch['index']:04d}"
        try:
            upsert_point(COL_CHUNKS, point_id_for_key(key), vec, payload)
        except Exception as e:
            print(f"       ! upsert error chunk {i}: {e}")
            errors += 1

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  {len(chunks)} chunks processed ({errors} errors)")
    print(f"  collections: {COL_CHUNKS}, {COL_SOURCES}")
    print(f"  source_id: {source_id}")


if __name__ == "__main__":
    main()
