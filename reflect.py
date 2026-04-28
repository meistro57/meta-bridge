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
  python reflect.py                           # run it, resumes automatically
  python reflect.py --limit 20                # only process first N new chunks (sanity test)
  python reflect.py --workers 3               # concurrent model calls (default 2)
  python reflect.py --model google/gemma-4-31b-it
  python reflect.py --model ollama:gemma4:latest
  python reflect.py --from-scratch            # wipe meta_reflections and start over
"""

from __future__ import annotations

import argparse
import hashlib
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


def load_env_file() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

# ------------------------------------------------------------------ config ----

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
DEFAULT_SOURCE_COLLECTIONS = ("mb_chunks", "mb_claims", "meta_test")
SOURCE_COLLECTIONS = tuple(
    c.strip()
    for c in os.environ.get("MB_REFLECT_SOURCE_COLLECTIONS", "").split(",")
    if c.strip()
) or DEFAULT_SOURCE_COLLECTIONS
TARGET_COLLECTION = "meta_reflections"
EMBED_PROVIDER = os.environ.get("MB_EMBED_PROVIDER", "openrouter").strip().lower()
EMBED_MODEL = os.environ.get("MB_EMBED_MODEL", "nomic-embed-text:latest")
DEFAULT_MODEL = "google/gemma-4-31b-it"
SCHEMA_VERSION = "2"
PROMPT_VERSION = "1"
SUMMARY_VECTOR_NAME = "summary_vec"
CLAIMS_VECTOR_NAME = "claims_vec"
EMBED_DIM = 0
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
    headers = {"api-key": QDRANT_API_KEY} if QDRANT_API_KEY else None
    resp = requests.request(method, url, json=body, headers=headers, timeout=30)
    if resp.status_code >= 400 and resp.status_code != 404:
        raise RuntimeError(f"qdrant {method} {endpoint}: {resp.status_code} {resp.text}")
    if resp.status_code == 404:
        return {"status": 404}
    return resp.json()


def infer_embed_dim() -> int:
    global EMBED_DIM
    if EMBED_DIM > 0:
        return EMBED_DIM
    probe = embed("reflection dimension probe")
    EMBED_DIM = len(probe)
    if EMBED_DIM <= 0:
        raise RuntimeError("embedding probe returned empty vector")
    return EMBED_DIM


def resolve_source_collections() -> tuple[str, ...]:
    configured = tuple(dict.fromkeys(SOURCE_COLLECTIONS))
    response = qdrant("GET", "/collections")
    if response.get("status") != "ok":
        raise RuntimeError(f"unable to list qdrant collections: {response}")

    available = {
        item.get("name")
        for item in response.get("result", {}).get("collections", [])
        if item.get("name")
    }
    selected = tuple(name for name in configured if name in available)
    if selected:
        return selected

    available_text = ", ".join(sorted(available)) or "(none)"
    configured_text = ", ".join(configured)
    raise RuntimeError(
        f"none of configured source collections exist ({configured_text}); available={available_text}"
    )


def ensure_target_collection(from_scratch: bool) -> None:
    global USE_NAMED_VECTORS

    embed_dim = infer_embed_dim()

    if from_scratch:
        print(f"[init] wiping {TARGET_COLLECTION}")
        qdrant("DELETE", f"/collections/{TARGET_COLLECTION}")

    info = qdrant("GET", f"/collections/{TARGET_COLLECTION}")
    if info.get("status") == 404 or info.get("status") != "ok" and "result" not in info:
        print(f"[init] creating {TARGET_COLLECTION}")
        qdrant("PUT", f"/collections/{TARGET_COLLECTION}", {
            "vectors": {
                SUMMARY_VECTOR_NAME: {"size": embed_dim, "distance": "Cosine"},
                CLAIMS_VECTOR_NAME: {"size": embed_dim, "distance": "Cosine"},
            }
        })
        USE_NAMED_VECTORS = True
    else:
        vectors_cfg = info.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
        USE_NAMED_VECTORS = not (isinstance(vectors_cfg, dict) and "size" in vectors_cfg)
        if USE_NAMED_VECTORS:
            summary_size = as_int(vectors_cfg.get(SUMMARY_VECTOR_NAME, {}).get("size"), 0)
            claims_size = as_int(vectors_cfg.get(CLAIMS_VECTOR_NAME, {}).get("size"), 0)
            if (summary_size and summary_size != embed_dim) or (claims_size and claims_size != embed_dim):
                raise RuntimeError(
                    f"{TARGET_COLLECTION} vector size mismatch: summary={summary_size}, claims={claims_size}, embed={embed_dim}; use --from-scratch or set MB_EMBED_MODEL to match"
                )
        else:
            vector_size = as_int(vectors_cfg.get("size"), 0)
            if vector_size and vector_size != embed_dim:
                raise RuntimeError(
                    f"{TARGET_COLLECTION} vector size mismatch: collection={vector_size}, embed={embed_dim}; use --from-scratch or set MB_EMBED_MODEL to match"
                )
        mode = "named-vectors" if USE_NAMED_VECTORS else "single-vector"
        count = qdrant("POST", f"/collections/{TARGET_COLLECTION}/points/count",
                       {"exact": True}).get("result", {}).get("count", 0)
        print(f"[init] {TARGET_COLLECTION} exists with {count} points (resume mode, {mode}, dim={embed_dim})")


def ensure_target_indexes() -> None:
    indexed_fields = {
        "source_point_id": "keyword",
        "source_collection": "keyword",
        "source_file": "keyword",
        "page": "integer",
        "chunk_index": "integer",
        "tone": "keyword",
        "model": "keyword",
        "reflected_at": "integer",
        "schema_version": "keyword",
        "prompt_version": "keyword",
        "token_count": "integer",
        "reflection_confidence": "float",
        "is_empty_reflection": "bool",
        "concepts_norm": "keyword",
        "claims_norm": "keyword",
    }
    for field_name, field_schema in indexed_fields.items():
        try:
            qdrant("PUT", f"/collections/{TARGET_COLLECTION}/index", {
                "field_name": field_name,
                "field_schema": field_schema,
            })
        except RuntimeError as e:
            if "already exists" not in str(e).lower():
                raise


def scroll_source(collection: str, offset: str | None = None):
    body = {"limit": SCROLL_BATCH, "with_payload": True, "with_vector": False}
    if offset is not None:
        body["offset"] = offset
    result = qdrant("POST", f"/collections/{collection}/points/scroll", body)
    return result.get("result", {})


def existing_reflection_ids(source_collections: tuple[str, ...]) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    offset = None
    while True:
        body = {
            "limit": 1000,
            "with_payload": {"include": ["source_collection", "source_point_id"]},
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        r = qdrant("POST", f"/collections/{TARGET_COLLECTION}/points/scroll", body).get("result", {})
        for pt in r.get("points", []):
            payload = pt.get("payload", {})
            src_id = payload.get("source_point_id")
            src_collection = payload.get("source_collection") or source_collections[0]
            if src_id:
                done.add((str(src_collection), str(src_id)))
        offset = r.get("next_page_offset")
        if not offset:
            break
    return done

# ----------------------------------------------------------------- llm --------

def complete(model: str, prompt: str, system: str) -> str:
    if model.startswith("ollama:"):
        return ollama_chat(model[len("ollama:"):], prompt, system)
    return openrouter_chat(model, prompt, system)


def openrouter_chat(model: str, prompt: str, system: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter models")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/meistro57/meta-bridge",
        "X-Title": "Meta Bridge",
    }
    resp = requests.post(OPENROUTER_CHAT_URL, json=body, headers=headers, timeout=180)
    if resp.status_code >= 400:
        raise RuntimeError(f"openrouter chat {resp.status_code}: {resp.text}")
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"openrouter error: {data['error'].get('message', 'unknown error')}")
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"openrouter returned no choices: {data}")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"openrouter returned empty message: {data}")
    return content


def ollama_chat(model: str, prompt: str, system: str) -> str:
    if not model.strip():
        raise RuntimeError("ollama model cannot be empty (use ollama:<model>)")

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


def embed(text: str) -> list[float]:
    if EMBED_PROVIDER == "openrouter":
        return openrouter_embed(text)
    if EMBED_PROVIDER == "ollama":
        return ollama_embed(text)
    raise RuntimeError("MB_EMBED_PROVIDER must be 'openrouter' or 'ollama'")


def openrouter_embed(text: str) -> list[float]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required for openrouter embeddings")

    body = {"model": EMBED_MODEL, "input": text}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/meistro57/meta-bridge",
        "X-Title": "Meta Bridge",
    }
    resp = requests.post(OPENROUTER_EMBEDDINGS_URL, json=body, headers=headers, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"openrouter embeddings {resp.status_code}: {resp.text}")
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"openrouter error: {data['error'].get('message', 'unknown error')}")
    vectors = data.get("data") or []
    if not vectors or not vectors[0].get("embedding"):
        raise RuntimeError("openrouter returned empty embedding")
    return vectors[0]["embedding"]


def ollama_embed(text: str) -> list[float]:
    url = OLLAMA_URL.rstrip("/") + "/api/embeddings"
    body = {"model": EMBED_MODEL, "prompt": text}
    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    embedding = data.get("embedding") or []
    if not embedding:
        raise RuntimeError("ollama returned empty embedding")
    return embedding

# --------------------------------------------------------------- reflect -----

@dataclass
class Chunk:
    source_collection: str
    point_id: str
    source_file: str
    page: int
    chunk_index: int
    text: str


def parse_gemma(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def normalize_text_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for item in items:
        value = " ".join(item.lower().split())
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def reflection_confidence(reflection: dict) -> float:
    score = 0.0
    if reflection.get("summary"):
        score += 0.35
    if reflection.get("claims"):
        score += min(0.25, 0.06 * len(reflection["claims"]))
    if reflection.get("concepts"):
        score += min(0.2, 0.05 * len(reflection["concepts"]))
    if reflection.get("questions"):
        score += min(0.1, 0.04 * len(reflection["questions"]))
    if reflection.get("echoes"):
        score += min(0.1, 0.05 * len(reflection["echoes"]))
    return round(min(1.0, score), 3)


def reflect_on_chunk(chunk: Chunk, model: str) -> dict:
    raw = complete(model, chunk.text, PROMPT)
    data = parse_gemma(raw)

    def as_list(v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        return []

    reflection = {
        "summary": str(data.get("summary") or "").strip(),
        "concepts": as_list(data.get("concepts")),
        "claims": as_list(data.get("claims")),
        "tone": str(data.get("tone") or "").strip(),
        "questions": as_list(data.get("questions")),
        "echoes": as_list(data.get("echoes")),
    }
    reflection["concepts_norm"] = normalize_text_list(reflection["concepts"])
    reflection["claims_norm"] = normalize_text_list(reflection["claims"])
    reflection["is_empty_reflection"] = not any([
        reflection["summary"],
        reflection["concepts"],
        reflection["claims"],
        reflection["tone"],
        reflection["questions"],
        reflection["echoes"],
    ])
    reflection["reflection_confidence"] = reflection_confidence(reflection)
    return reflection


def reflection_vectors(r: dict) -> dict[str, list[float]] | list[float]:
    summary_text = r.get("summary", "").strip()
    if not summary_text:
        summary_text = "(empty reflection summary)"

    claims_parts: list[str] = []
    if r.get("claims"):
        claims_parts.append("Claims: " + "; ".join(r["claims"]))
    if r.get("concepts"):
        claims_parts.append("Concepts: " + ", ".join(r["concepts"]))
    claims_text = "\n".join(part for part in claims_parts if part.strip())
    if not claims_text:
        claims_text = "(empty reflection claims)"

    if USE_NAMED_VECTORS:
        return {
            SUMMARY_VECTOR_NAME: embed(summary_text),
            CLAIMS_VECTOR_NAME: embed(claims_text),
        }

    legacy_text = "\n".join([summary_text, claims_text])
    return embed(legacy_text)


def upsert_reflection(chunk: Chunk, reflection: dict, vectors: dict[str, list[float]] | list[float]) -> None:
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"reflection:{chunk.source_collection}:{chunk.point_id}"))
    payload = {
        "source_point_id": chunk.point_id,
        "source_collection": chunk.source_collection,
        "source_file": chunk.source_file,
        "source_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "token_count": max(1, len(chunk.text.split())),
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": CURRENT_MODEL,
        "embed_provider": EMBED_PROVIDER,
        "embed_model": EMBED_MODEL,
        "reflected_at": int(time.time()),
        **reflection,
    }
    qdrant("PUT", f"/collections/{TARGET_COLLECTION}/points?wait=false", {
        "points": [{"id": point_id, "vector": vectors, "payload": payload}]
    })

# ---------------------------------------------------------------- runner -----

CURRENT_MODEL = DEFAULT_MODEL
USE_NAMED_VECTORS = True
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
        vectors = reflection_vectors(reflection)
        upsert_reflection(chunk, reflection, vectors)
        return chunk, None
    except Exception as e:
        return chunk, f"{type(e).__name__}: {e}"


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def first_attr(payload: dict[str, Any]) -> dict[str, Any]:
    attributions = payload.get("attributions")
    if isinstance(attributions, list) and attributions:
        first = attributions[0]
        if isinstance(first, dict):
            return first
    return {}


def chunk_text(payload: dict[str, Any]) -> str:
    text = payload.get("text") or payload.get("canonical_statement") or ""
    return str(text).strip()


def source_file_name(payload: dict[str, Any], attr: dict[str, Any]) -> str:
    value = (
        payload.get("source_file")
        or payload.get("source_id")
        or payload.get("document_id")
        or payload.get("title")
        or payload.get("chapter")
        or attr.get("source_id")
        or attr.get("chapter")
        or "unknown"
    )
    return str(value)


def iter_chunks(source_collections: tuple[str, ...], skip: set[tuple[str, str]]):
    for source_collection in source_collections:
        offset = None
        while True:
            page = scroll_source(source_collection, offset)
            for pt in page.get("points", []):
                pid = str(pt.get("id"))
                if (source_collection, pid) in skip:
                    continue
                pl = pt.get("payload", {})
                attr = first_attr(pl)
                text = chunk_text(pl)
                if not text:
                    continue
                yield Chunk(
                    source_collection=source_collection,
                    point_id=pid,
                    source_file=source_file_name(pl, attr),
                    page=as_int(pl.get("page", pl.get("page_number", attr.get("page", 0)))),
                    chunk_index=as_int(pl.get("chunk_index", pl.get("index", attr.get("chunk_index", 0)))),
                    text=text,
                )
            offset = page.get("next_page_offset")
            if not offset:
                break


def main() -> int:
    global CURRENT_MODEL, STOP

    parser = argparse.ArgumentParser(description="Reflect on source chunks and claims")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name (default: OpenRouter google/gemma-4-31b-it; override with ollama:<model>)")
    parser.add_argument("--limit", type=int, default=0, help="process at most N new chunks")
    parser.add_argument("--workers", type=int, default=2, help="concurrent model calls")
    parser.add_argument("--from-scratch", action="store_true", help="wipe target and restart")
    args = parser.parse_args()

    CURRENT_MODEL = args.model.strip()
    if not CURRENT_MODEL:
        raise RuntimeError("--model cannot be empty")

    if EMBED_PROVIDER not in {"openrouter", "ollama"}:
        raise RuntimeError("MB_EMBED_PROVIDER must be 'openrouter' or 'ollama'")

    uses_openrouter_model = not CURRENT_MODEL.startswith("ollama:")
    requires_openrouter = uses_openrouter_model or EMBED_PROVIDER == "openrouter"
    if requires_openrouter and not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required for current model/provider configuration")

    signal.signal(signal.SIGINT, handle_sigint)

    source_collections = resolve_source_collections()
    embed_dim = infer_embed_dim()

    print(
        f"[config] model={CURRENT_MODEL} workers={args.workers} qdrant={QDRANT_URL} "
        f"embed_provider={EMBED_PROVIDER} embed_model={EMBED_MODEL} embed_dim={embed_dim} "
        f"sources={','.join(source_collections)}"
    )

    ensure_target_collection(args.from_scratch)
    ensure_target_indexes()

    skip = existing_reflection_ids(source_collections)
    print(f"[resume] {len(skip)} chunks already reflected — will skip")

    collection_totals: dict[str, int] = {}
    for source_collection in source_collections:
        count = qdrant("POST", f"/collections/{source_collection}/points/count", {"exact": True}).get("result", {}).get("count", 0)
        collection_totals[source_collection] = count
    total = sum(collection_totals.values())
    remaining = max(0, total - len(skip))
    totals_summary = ", ".join(f"{name}={count}" for name, count in collection_totals.items())
    print(f"[plan]   total={total} ({totals_summary}), {remaining} remaining")

    if args.limit > 0:
        print(f"[plan]   --limit {args.limit} applied")
        remaining = min(remaining, args.limit)

    processed = 0
    errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        in_flight = {}
        chunks = iter_chunks(source_collections, skip)

        def submit_next() -> bool:
            try:
                ch = next(chunks)
            except StopIteration:
                return False
            fut = pool.submit(process_one, ch, CURRENT_MODEL)
            in_flight[fut] = ch
            return True

        for _ in range(args.workers):
            if not submit_next():
                break

        while in_flight:
            if STOP:
                break
            done_futs = []
            for fut in as_completed(list(in_flight.keys()), timeout=None):
                done_futs.append(fut)
                break

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
