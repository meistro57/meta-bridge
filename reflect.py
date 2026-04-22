#!/usr/bin/env python3
"""
reflect.py — Gemma 4 reads every chunk in meta_test and produces structured
reflections into a new collection (meta_reflections).

This is an exploratory pass. The point is to let Gemma reason over the canon
and see what falls out. Schema is deliberately open-ended — adjust the prompt
freely and re-run; stable IDs mean re-runs overwrite rather than duplicate.

Fields per reflection:
  summary           one sentence, in Gemma's own words
  concepts          list of key concepts this passage touches
  claims            list of atomic propositions asserted or dramatized
  tone              rhetorical mode: didactic / narrative / dialogue / etc.
  questions         questions the passage implicitly raises
  echoes            other traditions or thinkers the passage resonates with

Runs against:
  Qdrant:  http://localhost:6333
  Ollama:  http://localhost:11434
  Source:  meta_test (8869 chunks)
  Target:  meta_reflections (created on first run)

Usage:
  python reflect.py                 # run it, resumes automatically
  python reflect.py --limit 20      # only process first N new chunks (sanity test)
  python reflect.py --workers 3     # concurrent Gemma calls (default 2)
  python reflect.py --model gemma4:latest
  python reflect.py --from-scratch  # wipe meta_reflections and start over
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests

# ------------------------------------------------------------------ config ----

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
SOURCE_COLLECTION = "meta_test"
TARGET_COLLECTION = "meta_reflections"
EMBED_MODEL = "nomic-embed-text:latest"
DEFAULT_MODEL = "gemma4:latest"
EMBED_DIM = 768  # nomic-embed-text produces 768-dim vectors
SCROLL_BATCH = 100

PROMPT = """You are a careful reader of consciousness literature — channeled texts,
regression transcripts, esoteric works, and contemplative writing.

For the passage below, return a single JSON object with these exact fields:

{
  "summary":   "one sentence, your own words, what this passage is actually doing",
  "concepts":  ["key", "concepts", "touched"],
  "claims":    ["atomic propositions the passage asserts, dramatizes, or implies"],
  "tone":      "didactic | narrative | dialogue | instructional | contemplative | biographical | other",
  "questions": ["open questions this passage raises for a thoughtful reader"],
  "echoes":    ["other traditions, thinkers, frameworks this resonates with"]
}

Rules:
- If the passage contains no metaphysical or contemplative content (e.g. table of
  contents, index, copyright page, pure plot mechanics), return all fields empty
  or as empty arrays. Do not invent substance.
- Claims should be rephrased abstractly — not quoted verbatim, not tied to
  specific characters. State the underlying proposition.
- Concepts: 2-6 items, short noun phrases.
- Echoes: genuine resonances only. If nothing obvious comes to mind, leave empty.
- Return ONLY the JSON object. No prose, no markdown fences."""

# --------------------------------------------------------------- qdrant io ----

def qdrant(method: str, endpoint: str, body: Any = None) -> dict:
    url = QDRANT_URL.rstrip("/") + endpoint
    resp = requests.request(method, url, json=body, timeout=30)
    if resp.status_code >= 400 and resp.status_code != 404:
        raise RuntimeError(f"qdrant {method} {endpoint}: {resp.status_code} {resp.text}")
    if resp.status_code == 404:
        return {"status": 404}
    return resp.json()


def ensure_target_collection(from_scratch: bool) -> None:
    if from_scratch:
        print(f"[init] wiping {TARGET_COLLECTION}")
        qdrant("DELETE", f"/collections/{TARGET_COLLECTION}")

    info = qdrant("GET", f"/collections/{TARGET_COLLECTION}")
    if info.get("status") == 404 or info.get("status") != "ok" and "result" not in info:
        print(f"[init] creating {TARGET_COLLECTION}")
        qdrant("PUT", f"/collections/{TARGET_COLLECTION}", {
            "vectors": {"size": EMBED_DIM, "distance": "Cosine"}
        })
    else:
        count = qdrant("POST", f"/collections/{TARGET_COLLECTION}/points/count",
                       {"exact": True}).get("result", {}).get("count", 0)
        print(f"[init] {TARGET_COLLECTION} exists with {count} points (resume mode)")


def scroll_source(offset: str | None = None):
    """Yield chunks from meta_test in pages."""
    body = {"limit": SCROLL_BATCH, "with_payload": True, "with_vector": False}
    if offset is not None:
        body["offset"] = offset
    result = qdrant("POST", f"/collections/{SOURCE_COLLECTION}/points/scroll", body)
    return result.get("result", {})


def existing_reflection_ids() -> set[str]:
    """Return set of source chunk IDs that already have reflections."""
    done: set[str] = set()
    offset = None
    while True:
        body = {
            "limit": 1000,
            "with_payload": {"include": ["source_point_id"]},
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        r = qdrant("POST", f"/collections/{TARGET_COLLECTION}/points/scroll", body).get("result", {})
        for pt in r.get("points", []):
            src_id = pt.get("payload", {}).get("source_point_id")
            if src_id:
                done.add(src_id)
        offset = r.get("next_page_offset")
        if not offset:
            break
    return done

# ---------------------------------------------------------------- ollama ------

def ollama_chat(model: str, prompt: str, system: str) -> str:
    url = OLLAMA_URL.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3},
    }
    resp = requests.post(url, json=body, timeout=180)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def ollama_embed(text: str) -> list[float]:
    url = OLLAMA_URL.rstrip("/") + "/api/embeddings"
    body = {"model": EMBED_MODEL, "prompt": text}
    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()["embedding"]

# --------------------------------------------------------------- reflect -----

@dataclass
class Chunk:
    point_id: str
    source_file: str
    page: int
    chunk_index: int
    text: str


def parse_gemma(raw: str) -> dict:
    """Gemma with format:json is usually clean but sometimes wraps or truncates."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # last-resort: find first { and last } and retry
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def reflect_on_chunk(chunk: Chunk, model: str) -> dict:
    """Run Gemma on one chunk, return the parsed reflection."""
    raw = ollama_chat(model, chunk.text, PROMPT)
    data = parse_gemma(raw)

    # Normalize shape — Gemma sometimes returns singular strings where we want arrays.
    def as_list(v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        return []

    return {
        "summary":   str(data.get("summary") or "").strip(),
        "concepts":  as_list(data.get("concepts")),
        "claims":    as_list(data.get("claims")),
        "tone":      str(data.get("tone") or "").strip(),
        "questions": as_list(data.get("questions")),
        "echoes":    as_list(data.get("echoes")),
    }


def embed_reflection(r: dict) -> list[float]:
    """Embed the reflection as a single text blob (summary + claims + concepts).
    This makes the vector searchable by meaning rather than by the raw passage."""
    parts = [r.get("summary", "")]
    if r.get("claims"):
        parts.append("Claims: " + "; ".join(r["claims"]))
    if r.get("concepts"):
        parts.append("Concepts: " + ", ".join(r["concepts"]))
    text = "\n".join(p for p in parts if p.strip())
    if not text.strip():
        text = "(empty reflection)"
    return ollama_embed(text)


def upsert_reflection(chunk: Chunk, reflection: dict, vector: list[float]) -> None:
    # Stable UUID v5 so reruns overwrite rather than duplicate.
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"reflection:{chunk.point_id}"))
    payload = {
        "source_point_id": chunk.point_id,
        "source_file":     chunk.source_file,
        "page":            chunk.page,
        "chunk_index":     chunk.chunk_index,
        "model":           CURRENT_MODEL,
        "reflected_at":    int(time.time()),
        **reflection,
    }
    qdrant("PUT", f"/collections/{TARGET_COLLECTION}/points?wait=false", {
        "points": [{"id": point_id, "vector": vector, "payload": payload}]
    })

# ---------------------------------------------------------------- runner -----

CURRENT_MODEL = DEFAULT_MODEL
STOP = False


def handle_sigint(signum, frame):
    global STOP
    if STOP:
        print("\n[!] second ctrl-c; exiting hard")
        sys.exit(130)
    STOP = True
    print("\n[!] ctrl-c caught, finishing in-flight chunks then stopping...")


def process_one(chunk: Chunk, model: str) -> tuple[Chunk, str | None]:
    try:
        reflection = reflect_on_chunk(chunk, model)
        vector = embed_reflection(reflection)
        upsert_reflection(chunk, reflection, vector)
        return chunk, None
    except Exception as e:
        return chunk, f"{type(e).__name__}: {e}"


def iter_chunks(skip: set[str]):
    """Stream chunks from meta_test, skipping ones we've already reflected on."""
    offset = None
    while True:
        page = scroll_source(offset)
        for pt in page.get("points", []):
            pid = pt.get("id")
            if pid in skip:
                continue
            pl = pt.get("payload", {})
            text = (pl.get("text") or "").strip()
            if not text:
                continue
            yield Chunk(
                point_id=str(pid),
                source_file=pl.get("source_file", "unknown"),
                page=int(pl.get("page", 0)),
                chunk_index=int(pl.get("chunk_index", 0)),
                text=text,
            )
        offset = page.get("next_page_offset")
        if not offset:
            break


def main() -> int:
    global CURRENT_MODEL, STOP

    parser = argparse.ArgumentParser(description="Reflect on meta_test chunks with Gemma")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument("--limit", type=int, default=0, help="process at most N new chunks")
    parser.add_argument("--workers", type=int, default=2, help="concurrent Gemma calls")
    parser.add_argument("--from-scratch", action="store_true", help="wipe target and restart")
    args = parser.parse_args()

    CURRENT_MODEL = args.model
    signal.signal(signal.SIGINT, handle_sigint)

    print(f"[config] model={CURRENT_MODEL} workers={args.workers} qdrant={QDRANT_URL}")

    ensure_target_collection(args.from_scratch)

    # How much is already done?
    skip = existing_reflection_ids()
    print(f"[resume] {len(skip)} chunks already reflected — will skip")

    # Get total source count for progress display.
    total = qdrant("POST", f"/collections/{SOURCE_COLLECTION}/points/count",
                   {"exact": True}).get("result", {}).get("count", 0)
    remaining = total - len(skip)
    print(f"[plan]   {total} chunks total in {SOURCE_COLLECTION}, {remaining} remaining")

    if args.limit > 0:
        print(f"[plan]   --limit {args.limit} applied")
        remaining = min(remaining, args.limit)

    processed = 0
    errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        in_flight = {}
        chunks = iter_chunks(skip)

        def submit_next() -> bool:
            try:
                ch = next(chunks)
            except StopIteration:
                return False
            fut = pool.submit(process_one, ch, CURRENT_MODEL)
            in_flight[fut] = ch
            return True

        # prime the pool
        for _ in range(args.workers):
            if not submit_next():
                break

        while in_flight:
            if STOP:
                break
            done_futs = []
            for fut in as_completed(list(in_flight.keys()), timeout=None):
                done_futs.append(fut)
                break  # handle one at a time so we can submit+print cleanly

            for fut in done_futs:
                ch = in_flight.pop(fut)
                _, err = fut.result()
                processed += 1
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (remaining - processed) / rate if rate > 0 else 0
                if err:
                    errors += 1
                    print(f"[{processed}/{remaining}] ✗ {ch.source_file} p{ch.page} c{ch.chunk_index}  {err}")
                else:
                    print(f"[{processed}/{remaining}] ✓ {ch.source_file} p{ch.page} c{ch.chunk_index}  "
                          f"({rate:.1f}/s, eta {eta/60:.0f}m)")

                if args.limit > 0 and processed >= args.limit:
                    STOP = True
                    break

                if not STOP:
                    submit_next()

    elapsed = time.time() - t0
    print(f"\n[done] {processed} processed ({errors} errors) in {elapsed/60:.1f}m")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
