#!/usr/bin/env python3
# reflect_loop.py
"""
Reflection loop for Meta-Bridge — plain-loop edition.

History:
    v1 wrapped reflect.py in a LangGraph StateGraph. Two structural problems:
      1. LangGraph's default recursion_limit (25 super-steps) killed runs after
         ~5 chunks (5 nodes per chunk) regardless of --limit.
      2. State hoarded full reflection dicts in history/interesting/
         contradictions lists on every step — unbounded memory for data
         already persisted to Qdrant via loop flags.
    v2 (this file) is a straight loop in the FrontPocket reflection_loop.py
    shape: counters-only state, cheap-skip guard, graceful Ctrl-C, per-tone
    and per-source stats, rate/ETA, and optional concurrency. Same CLI, same
    loop_* flag persistence, same evaluate/decide semantics. langgraph is no
    longer required.

Reuses reflect.py for:
    .env loading, OpenRouter API keys, Qdrant config and IO, source collection
    discovery, reflection prompt + parsing, embedding, and upsert.

Default target collection:
    meta_reflections

Run:
    python reflect_loop.py --limit 20
    python reflect_loop.py --limit 200 --workers 3
    python reflect_loop.py --model google/gemini-3.1-flash-lite
    python reflect_loop.py --goal "hunt contradictions across consciousness claims"
    python reflect_loop.py --target-collection meta_reflection_loop_test
    python reflect_loop.py --loop-interval 60 --max-loops 0
    python reflect_loop.py --from-scratch
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, Optional, Tuple

import reflect as rf


DEFAULT_TARGET_COLLECTION = "meta_reflections"
STABLE_TARGET_COLLECTION = "meta_reflections"

DEFAULT_MIN_TEXT_LEN = 20

# Payload fields written back to the target collection per processed chunk.
FLAG_INTERESTING   = "loop_interesting"
FLAG_CONTRADICTION = "loop_contradiction"
FLAG_DECISION      = "loop_decision"
FLAG_FLAGGED_AT    = "loop_flagged_at"

FLAG_INDEXES: dict[str, str] = {
    FLAG_INTERESTING:   "bool",
    FLAG_CONTRADICTION: "bool",
    FLAG_DECISION:      "keyword",
    FLAG_FLAGGED_AT:    "integer",
}

DECISION_ICONS = {
    "store_interesting":   "★",
    "track_contradiction": "⚡",
    "continue_scan":       "·",
}

# ── graceful shutdown ─────────────────────────────────────────────────────────

STOP = False


def handle_sigint(signum, frame):
    global STOP
    if STOP:
        print("\n[!] second ctrl-c; exiting hard")
        sys.exit(130)
    STOP = True
    print("\n[!] ctrl-c caught — finishing in-flight chunks, then stopping...")


# ── validation ────────────────────────────────────────────────────────────────

def validate_remote_config(model: str) -> None:
    """
    Enforce the non-local setup.

    Prevents accidentally running --model ollama:... or MB_EMBED_PROVIDER=ollama.
    """
    if model.startswith("ollama:"):
        raise RuntimeError(
            "Local Ollama model requested, but this runner is configured for remote use. "
            "Use an OpenRouter model like: google/gemini-3.1-flash-lite"
        )

    if not rf.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

    if rf.EMBED_PROVIDER != "openrouter":
        raise RuntimeError(
            f"MB_EMBED_PROVIDER is currently '{rf.EMBED_PROVIDER}'. "
            "Set this in .env:\n\n"
            "MB_EMBED_PROVIDER=openrouter"
        )


def validate_target_collection(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise RuntimeError("--target-collection cannot be empty")
    if cleaned == STABLE_TARGET_COLLECTION:
        print(
            f"[warn] writing into stable production collection '{STABLE_TARGET_COLLECTION}'. "
            f"Pass --target-collection <other> for isolated test runs."
        )
    return cleaned


# ── flag persistence ──────────────────────────────────────────────────────────

def ensure_flag_indexes(target_collection: str) -> None:
    """Create payload indexes for loop flag fields. Idempotent."""
    for field_name, field_schema in FLAG_INDEXES.items():
        try:
            rf.qdrant(
                "PUT",
                f"/collections/{target_collection}/index",
                {"field_name": field_name, "field_schema": field_schema},
            )
        except RuntimeError as e:
            if "already exists" not in str(e).lower():
                raise


def persist_flags(point_id: str, decision: str, target_collection: str) -> None:
    """
    Merge loop decision flags onto an existing reflection point.
    Uses set_payload so no existing fields are touched. Non-blocking.
    """
    rf.qdrant(
        "POST",
        f"/collections/{target_collection}/points/payload?wait=false",
        {
            "payload": {
                FLAG_INTERESTING:   decision == "store_interesting",
                FLAG_CONTRADICTION: decision == "track_contradiction",
                FLAG_DECISION:      decision,
                FLAG_FLAGGED_AT:    int(time.time()),
            },
            "points": [point_id],
        },
    )


# ── evaluate / decide ─────────────────────────────────────────────────────────

def detect_possible_contradiction(reflection: Dict[str, Any]) -> bool:
    """
    First-pass contradiction sniffing.

    Intentionally simple. Later this should compare claims against previous
    reflections in Qdrant, not just inspect one reflection.
    """
    claims = reflection.get("claims") or []

    contradiction_markers = [
        "not ", "never", "cannot", "opposite", "contradict",
        "conflict", "but ", "however", "rather than", "instead",
        "unlike", "deny", "denies", "reject", "rejects", "inconsistent",
    ]

    for claim in claims:
        text = str(claim).lower()
        if any(marker in text for marker in contradiction_markers):
            return True

    return False


def evaluate_reflection(reflection: Dict[str, Any]) -> Dict[str, Any]:
    """Decide whether the reflection has enough signal to matter."""
    confidence = float(reflection.get("reflection_confidence") or 0.0)
    concepts   = reflection.get("concepts") or []
    claims     = reflection.get("claims") or []
    questions  = reflection.get("questions") or []
    echoes     = reflection.get("echoes") or []

    is_interesting = (
        confidence >= 0.60
        or len(claims) >= 3
        or len(concepts) >= 4
        or len(questions) >= 2
        or len(echoes) >= 2
    )

    return {
        "confidence": confidence,
        "concept_count": len(concepts),
        "claim_count": len(claims),
        "question_count": len(questions),
        "echo_count": len(echoes),
        "is_interesting": is_interesting,
        "has_possible_contradiction": detect_possible_contradiction(reflection),
    }


def decide_next_action(evaluation: Dict[str, Any]) -> str:
    """Convert evaluation into an action label."""
    if evaluation.get("has_possible_contradiction"):
        return "track_contradiction"
    if evaluation.get("is_interesting"):
        return "store_interesting"
    return "continue_scan"


# ── worker ────────────────────────────────────────────────────────────────────

def reflection_point_id(chunk: rf.Chunk) -> str:
    """Derive the reflection point ID the same way rf.upsert_reflection does."""
    return str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"reflection:{chunk.source_collection}:{chunk.point_id}",
    ))


def process_one(
    chunk: rf.Chunk,
    model: str,
) -> Tuple[rf.Chunk, Optional[Dict[str, Any]], str, Optional[str]]:
    """
    Reflect on one chunk and upsert the result.
    Returns (chunk, reflection|None, point_id, error|None).
    Runs on worker threads; keep it side-effect-contained.
    """
    try:
        reflection = rf.reflect_on_chunk(chunk, model)
        vectors    = rf.reflection_vectors(reflection)
        rf.upsert_reflection(chunk, reflection, vectors)
        return chunk, reflection, reflection_point_id(chunk), None
    except Exception as exc:
        return chunk, None, "", f"{type(exc).__name__}: {exc}"


# ── reporting ─────────────────────────────────────────────────────────────────

def print_step(
    stats: Dict[str, Any],
    chunk: rf.Chunk,
    evaluation: Dict[str, Any],
    decision: str,
    remaining: int,
    t0: float,
) -> None:
    processed = stats["processed"]
    elapsed   = time.time() - t0
    rate      = processed / elapsed if elapsed > 0 else 0.0
    eta_min   = ((remaining - processed) / rate / 60) if rate > 0 else 0.0

    icon = DECISION_ICONS.get(decision, "?")
    print(
        f"[{processed}/{remaining}] ✓ {chunk.source_file} "
        f"p{chunk.page} c{chunk.chunk_index} | "
        f"{icon} {decision} | "
        f"conf={evaluation['confidence']:.3f} "
        f"claims={evaluation['claim_count']} "
        f"concepts={evaluation['concept_count']} | "
        f"{rate:.1f}/s eta {eta_min:.0f}m"
    )


def print_summary(stats: Dict[str, Any], target_collection: str, elapsed: float) -> None:
    print("\n[done]")
    print(f"  target:          {target_collection}")
    print(f"  processed:       {stats['processed']}")
    print(f"  skipped:         {stats['skipped']}")
    print(f"  errors:          {stats['errors']}")
    print(f"  interesting:     {stats['interesting']}")
    print(f"  contradictions:  {stats['contradictions']}")
    print(f"  flags_written:   {stats['flags_written']}")
    print(f"  elapsed:         {elapsed / 60:.1f}m")

    by_tone: Dict[str, int] = stats["by_tone"]
    if by_tone:
        tones = ", ".join(f"{k or '(none)'}={v}" for k, v in
                          sorted(by_tone.items(), key=lambda kv: -kv[1]))
        print(f"  by_tone:         {tones}")

    by_source: Dict[str, int] = stats["by_source"]
    if by_source:
        top = sorted(by_source.items(), key=lambda kv: -kv[1])[:8]
        srcs = ", ".join(f"{k}={v}" for k, v in top)
        more = len(by_source) - len(top)
        suffix = f" (+{more} more)" if more > 0 else ""
        print(f"  by_source:       {srcs}{suffix}")

    if stats["interesting"] > 0:
        print(f"\n[flags] query interesting reflections:")
        print(f'  filter: {{"must": [{{"key": "{FLAG_INTERESTING}", "match": {{"value": true}}}}]}}')
    if stats["contradictions"] > 0:
        print(f"\n[flags] query contradiction-flagged reflections:")
        print(f'  filter: {{"must": [{{"key": "{FLAG_CONTRADICTION}", "match": {{"value": true}}}}]}}')


# ── reflag existing reflections ──────────────────────────────────────────────

def reflag_existing(
    target_collection: str,
    do_persist_flags: bool,
    quiet: bool,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """
    Scroll meta_reflections and backfill loop_* flags on points that are
    missing them (pre-date the flag persistence work).

    No LLM calls — evaluates the existing payload fields only.
    Uses the same evaluate_reflection / decide_next_action logic as the
    main loop so flags are consistent.
    """
    print(f"\n[reflag] scanning {target_collection} for unflagged points...")

    stats: Dict[str, Any] = {
        "total": 0,
        "already_done": 0,
        "interesting": 0,
        "contradictions": 0,
        "continue_scan": 0,
        "written": 0,
        "errors": 0,
    }

    offset = None
    batch = []  # list of (point_id, decision)

    def flush_batch() -> None:
        if not batch or not do_persist_flags:
            stats["written"] += len(batch)
            batch.clear()
            return
        for point_id, decision in batch:
            try:
                persist_flags(point_id, decision, target_collection)
                stats["written"] += 1
            except Exception as exc:
                stats["errors"] += 1
                if not quiet:
                    print(f"  [flag] write failed for {point_id}: {exc}")
        batch.clear()

    while True:
        body: Dict[str, Any] = {
            "limit": 250,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset

        result = rf.qdrant(
            "POST",
            f"/collections/{target_collection}/points/scroll",
            body,
        ).get("result", {})

        points = result.get("points", [])
        offset = result.get("next_page_offset")

        for pt in points:
            stats["total"] += 1
            pl = pt.get("payload") or {}

            # already flagged — skip
            if "loop_decision" in pl:
                stats["already_done"] += 1
                continue

            # reconstruct a minimal reflection dict from stored payload
            reflection = {
                "reflection_confidence": pl.get("reflection_confidence", 0.0),
                "concepts":  pl.get("concepts")  or [],
                "claims":    pl.get("claims")    or [],
                "questions": pl.get("questions") or [],
                "echoes":    pl.get("echoes")    or [],
            }

            evaluation = evaluate_reflection(reflection)
            decision   = decide_next_action(evaluation)

            if decision == "store_interesting":
                stats["interesting"] += 1
            elif decision == "track_contradiction":
                stats["contradictions"] += 1
            else:
                stats["continue_scan"] += 1

            point_id = str(pt["id"])
            batch.append((point_id, decision))

            if len(batch) >= batch_size:
                flush_batch()
                if not quiet:
                    print(
                        f"  ↳ flagged {stats['written']} so far "
                        f"(interesting={stats['interesting']} "
                        f"contradiction={stats['contradictions']} "
                        f"scan={stats['continue_scan']} "
                        f"errors={stats['errors']})"
                    )

        if not offset:
            break

    flush_batch()

    print(f"\n[reflag] complete")
    print(f"  total scanned:    {stats['total']}")
    print(f"  already flagged:  {stats['already_done']}")
    print(f"  newly flagged:    {stats['written']}")
    print(f"    interesting:    {stats['interesting']}")
    print(f"    contradiction:  {stats['contradictions']}")
    print(f"    continue_scan:  {stats['continue_scan']}")
    print(f"  errors:           {stats['errors']}")
    return stats


# ── main loop ─────────────────────────────────────────────────────────────────

def run_once(
    args: argparse.Namespace,
    model: str,
    target_collection: str,
    from_scratch: bool,
    do_persist_flags: bool,
) -> Dict[str, Any]:
    global STOP
    STOP = False

    rf.CURRENT_MODEL     = model
    rf.TARGET_COLLECTION = target_collection

    print("\n[config]")
    print(f"  model:          {model}")
    print(f"  qdrant:         {rf.QDRANT_URL}")
    print(f"  embed_provider: {rf.EMBED_PROVIDER}")
    print(f"  embed_model:    {rf.EMBED_MODEL}")
    print(f"  target:         {target_collection}")
    print(f"  workers:        {args.workers}")
    print(f"  min_text:       {args.min_text}")
    print(f"  goal:           {args.goal}")
    print(f"  persist_flags:  {do_persist_flags}")

    rf.ensure_target_collection(from_scratch)
    rf.ensure_target_indexes()

    if do_persist_flags:
        print("[init] ensuring loop flag indexes...")
        ensure_flag_indexes(target_collection)

    source_collections = rf.resolve_source_collections()
    skip = rf.existing_reflection_ids(source_collections)
    print(f"[resume] {len(skip)} chunks already reflected — will skip")

    total = 0
    for source_collection in source_collections:
        count = rf.qdrant(
            "POST",
            f"/collections/{source_collection}/points/count",
            {"exact": True},
        ).get("result", {}).get("count", 0)
        total += int(count or 0)
    remaining = max(0, total - len(skip))
    if args.limit > 0:
        remaining = min(remaining, args.limit)
    print(f"[plan]   total={total}, targeting up to {remaining} this run")

    stats: Dict[str, Any] = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "interesting": 0,
        "contradictions": 0,
        "flags_written": 0,
        "by_tone": {},
        "by_source": {},
        "last_error": "",
    }

    source_ids = rf.load_source_id_map()
    chunks = rf.iter_chunks(source_collections, skip, source_ids)

    def next_chunk() -> Optional[rf.Chunk]:
        """Pull the next chunk, applying the cheap-skip guard."""
        while True:
            try:
                chunk = next(chunks)
            except StopIteration:
                return None
            if len(chunk.text.strip()) < args.min_text:
                stats["skipped"] += 1
                continue
            return chunk

    print("\n[loop] starting\n")
    t0 = time.time()
    submitted = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        in_flight: dict = {}

        def submit_next() -> bool:
            nonlocal submitted
            if STOP:
                return False
            if args.limit > 0 and submitted >= args.limit:
                return False
            chunk = next_chunk()
            if chunk is None:
                return False
            fut = pool.submit(process_one, chunk, model)
            in_flight[fut] = chunk
            submitted += 1
            return True

        for _ in range(max(1, args.workers)):
            if not submit_next():
                break

        while in_flight:
            done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)

            for fut in done:
                in_flight.pop(fut, None)
                chunk, reflection, point_id, err = fut.result()
                stats["processed"] += 1

                if err:
                    stats["errors"] += 1
                    stats["last_error"] = err
                    if not args.quiet:
                        print(
                            f"[{stats['processed']}/{remaining}] ✗ {chunk.source_file} "
                            f"p{chunk.page} c{chunk.chunk_index}  {err}"
                        )
                    continue

                evaluation = evaluate_reflection(reflection or {})
                decision   = decide_next_action(evaluation)

                if decision == "store_interesting":
                    stats["interesting"] += 1
                elif decision == "track_contradiction":
                    stats["contradictions"] += 1

                tone = str((reflection or {}).get("tone") or "")
                stats["by_tone"][tone] = stats["by_tone"].get(tone, 0) + 1
                src = chunk.source_id or chunk.source_file or "unknown"
                stats["by_source"][src] = stats["by_source"].get(src, 0) + 1

                # Persist loop decision flags for ALL decisions so downstream
                # can filter by loop_decision. Best-effort — a flag-write
                # failure never kills the loop.
                if do_persist_flags and point_id:
                    try:
                        persist_flags(point_id, decision, target_collection)
                        stats["flags_written"] += 1
                    except Exception as exc:
                        print(f"    [flag] write failed for {point_id}: {exc}")

                if not args.quiet:
                    print_step(stats, chunk, evaluation, decision, remaining, t0)

            while len(in_flight) < args.workers:
                if not submit_next():
                    break

    elapsed = time.time() - t0
    print_summary(stats, target_collection, elapsed)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plain reflection loop for Meta-Bridge (evaluate/decide/flag)"
    )
    parser.add_argument(
        "--model",
        default=rf.DEFAULT_MODEL,
        help=f"OpenRouter model to use. Default: {rf.DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--goal",
        default="Explore conceptual structure and surface interesting metaphysical claims",
        help="High-level research goal for this loop run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of chunks to process (0 = no limit)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent reflect+embed workers. Default: 2",
    )
    parser.add_argument(
        "--min-text",
        type=int,
        default=DEFAULT_MIN_TEXT_LEN,
        help=f"Skip chunks with fewer characters than this. Default: {DEFAULT_MIN_TEXT_LEN}",
    )
    parser.add_argument(
        "--target-collection",
        default=DEFAULT_TARGET_COLLECTION,
        help=f"Qdrant collection to write reflections into. Default: {DEFAULT_TARGET_COLLECTION}",
    )
    parser.add_argument(
        "--loop-interval",
        type=float,
        default=0.0,
        help="Seconds between repeated runs. 0 disables timer loop.",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=1,
        help="How many runs to execute when --loop-interval > 0 (0 = infinite).",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Wipe the target collection and rebuild from scratch",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step output",
    )
    parser.add_argument(
        "--no-persist-flags",
        action="store_true",
        help="Disable writing loop decision flags back to Qdrant",
    )
    parser.add_argument(
        "--reflag-unflagged",
        action="store_true",
        help=(
            "Scan target collection and backfill loop_* flags on points that "
            "don't have them yet. No LLM calls — evaluates existing payload. "
            "Runs once and exits."
        ),
    )
    parser.add_argument(
        "--reflag-batch",
        type=int,
        default=500,
        help="Batch size for --reflag-unflagged set_payload calls. Default: 500",
    )

    args = parser.parse_args()

    model = args.model.strip()
    if not model:
        raise RuntimeError("--model cannot be empty")
    if args.limit < 0:
        raise RuntimeError("--limit must be >= 0")
    if args.workers < 1:
        raise RuntimeError("--workers must be >= 1")
    if args.min_text < 0:
        raise RuntimeError("--min-text must be >= 0")
    if args.loop_interval < 0:
        raise RuntimeError("--loop-interval must be >= 0")
    if args.max_loops < 0:
        raise RuntimeError("--max-loops must be >= 0")

    target_collection = validate_target_collection(args.target_collection)
    validate_remote_config(model)

    do_persist_flags = not args.no_persist_flags

    signal.signal(signal.SIGINT, handle_sigint)

    # --reflag-unflagged: backfill mode, no LLM calls, runs once and exits
    if args.reflag_unflagged:
        if do_persist_flags:
            print("[init] ensuring loop flag indexes...")
            ensure_flag_indexes(args.target_collection.strip())
        stats = reflag_existing(
            target_collection=args.target_collection.strip(),
            do_persist_flags=do_persist_flags,
            quiet=args.quiet,
            batch_size=args.reflag_batch,
        )
        return 0 if stats["errors"] == 0 else 1

    if args.loop_interval <= 0:
        stats = run_once(args, model, target_collection, args.from_scratch, do_persist_flags)
        return 0 if stats["errors"] == 0 else 1

    run_count    = 0
    total_errors = 0

    while True:
        run_count += 1
        print(f"\n[timer] run {run_count} starting")
        stats = run_once(
            args, model, target_collection,
            args.from_scratch if run_count == 1 else False,
            do_persist_flags,
        )
        total_errors += int(stats["errors"] or 0)

        if STOP:
            break
        if args.max_loops > 0 and run_count >= args.max_loops:
            break

        print(f"\n[timer] sleeping {args.loop_interval:.1f}s before next run")
        time.sleep(args.loop_interval)

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
