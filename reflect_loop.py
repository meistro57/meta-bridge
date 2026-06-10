#!/usr/bin/env python3
# reflect_loop.py
"""
LangGraph-style reflection loop for Meta-Bridge.

This does NOT use local Ollama by default.

It reuses reflect.py for:
- .env loading
- OpenRouter API keys
- Qdrant config
- source collection discovery
- reflection prompt
- reflection parsing
- embedding
- Qdrant upsert

Purpose:
    Turn the current batch reflector into a stateful research loop without
    tarnishing the stable meta_reflections collection.

Default target collection:
    meta_reflection_loop_test

Run:
    python reflect_loop.py --limit 20

Optional:
    python reflect_loop.py --model google/gemini-3.1-flash-lite
    python reflect_loop.py --limit 50 --goal "hunt contradictions across consciousness claims"
    python reflect_loop.py --target-collection meta_reflections
    python reflect_loop.py --loop-interval 60 --max-loops 0
    python reflect_loop.py --from-scratch
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    print(
        "\n[missing] langgraph is not installed.\n\n"
        "Install it with:\n"
        "  pip install langgraph\n"
    )
    sys.exit(1)

import reflect as rf


DEFAULT_TARGET_COLLECTION = "meta_reflections"
STABLE_TARGET_COLLECTION  = "meta_reflections"

# Payload fields written back to meta_reflections per processed chunk.
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


class LoopState(TypedDict, total=False):
    goal:             str
    model:            str
    current_chunk:    Optional[rf.Chunk]
    current_point_id: Optional[str]       # uuid of the upserted reflection point
    reflection:       Dict[str, Any]
    evaluation:       Dict[str, Any]
    decision:         str
    history:          List[Dict[str, Any]]
    interesting:      List[Dict[str, Any]]
    contradictions:   List[Dict[str, Any]]
    processed:        int
    errors:           int
    flags_written:    int                  # count of payload flag writes
    limit:            int
    done:             bool
    last_error:       str
    persist_flags:    bool                 # runtime toggle


def validate_remote_config(model: str) -> None:
    """
    Enforce the non-local setup.

    This prevents accidentally running:
        --model ollama:...
    or:
        MB_EMBED_PROVIDER=ollama
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


# ──────────────────────────────────────── flag persistence ───────────────────

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
    Merge loop decision flags onto an existing meta_reflections point.
    Uses set_payload so no existing fields are touched.
    Non-blocking (?wait=false).
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


def detect_possible_contradiction(reflection: Dict[str, Any]) -> bool:
    """
    First-pass contradiction sniffing.

    This is intentionally simple for now. Later this should compare claims
    against previous reflections in Qdrant, not just inspect one reflection.
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
    """
    Decide whether the reflection has enough signal to matter.
    """

    confidence = float(reflection.get("reflection_confidence") or 0.0)
    concepts = reflection.get("concepts") or []
    claims = reflection.get("claims") or []
    questions = reflection.get("questions") or []
    echoes = reflection.get("echoes") or []

    concept_count = len(concepts)
    claim_count = len(claims)
    question_count = len(questions)
    echo_count = len(echoes)

    has_possible_contradiction = detect_possible_contradiction(reflection)

    is_interesting = (
        confidence >= 0.60
        or claim_count >= 3
        or concept_count >= 4
        or question_count >= 2
        or echo_count >= 2
    )

    return {
        "confidence": confidence,
        "concept_count": concept_count,
        "claim_count": claim_count,
        "question_count": question_count,
        "echo_count": echo_count,
        "is_interesting": is_interesting,
        "has_possible_contradiction": has_possible_contradiction,
    }


def decide_next_action(evaluation: Dict[str, Any]) -> str:
    """
    Convert evaluation into an action label.
    """

    if evaluation.get("has_possible_contradiction"):
        return "track_contradiction"

    if evaluation.get("is_interesting"):
        return "store_interesting"

    return "continue_scan"


class ReflectLoopRuntime:
    """
    Holds runtime-only objects that do not belong directly in the LangGraph state,
    such as the chunk iterator.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.source_collections = rf.resolve_source_collections()
        self.skip = rf.existing_reflection_ids(self.source_collections)
        self.chunk_iter = rf.iter_chunks(self.source_collections, self.skip)

    def get_next_chunk(self) -> Optional[rf.Chunk]:
        try:
            return next(self.chunk_iter)
        except StopIteration:
            return None


def build_graph(runtime: ReflectLoopRuntime, target_collection: str):
    """
    Build the LangGraph loop:

        get_chunk -> reflect -> evaluate -> decide -> act -> get_chunk/end

    Important fix:
        If get_chunk marks the run as done, route directly to END instead of
        walking through reflect/evaluate/act with an empty or stale chunk.
    """

    graph = StateGraph(LoopState)

    def get_chunk_node(state: LoopState) -> LoopState:
        state["reflection"]       = {}
        state["evaluation"]       = {}
        state["decision"]         = ""
        state["last_error"]       = ""
        state["current_point_id"] = None

        if state.get("processed", 0) >= state.get("limit", runtime.limit):
            state["done"]          = True
            state["current_chunk"] = None
            return state

        chunk = runtime.get_next_chunk()
        if chunk is None:
            state["done"]          = True
            state["current_chunk"] = None
            return state

        state["current_chunk"] = chunk
        state["done"]          = False
        return state

    def route_after_get_chunk(state: LoopState) -> str:
        return "end" if state.get("done") else "reflect"

    def reflect_node(state: LoopState) -> LoopState:
        chunk = state.get("current_chunk")
        model = state.get("model") or rf.DEFAULT_MODEL

        if chunk is None:
            state["done"] = True
            return state

        try:
            reflection = rf.reflect_on_chunk(chunk, model)
            vectors    = rf.reflection_vectors(reflection)
            rf.upsert_reflection(chunk, reflection, vectors)

            # Derive point ID the same way upsert_reflection does — no round-trip needed.
            import uuid
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"reflection:{chunk.source_collection}:{chunk.point_id}",
            ))

            state["reflection"]       = reflection
            state["current_point_id"] = point_id
            state["last_error"]       = ""

        except Exception as exc:
            state["errors"]           = state.get("errors", 0) + 1
            state["last_error"]       = f"{type(exc).__name__}: {exc}"
            state["reflection"]       = {}
            state["current_point_id"] = None

        return state

    def evaluate_node(state: LoopState) -> LoopState:
        reflection = state.get("reflection") or {}

        if not reflection:
            state["evaluation"] = {
                "confidence": 0.0,
                "concept_count": 0,
                "claim_count": 0,
                "question_count": 0,
                "echo_count": 0,
                "is_interesting": False,
                "has_possible_contradiction": False,
            }
            return state

        state["evaluation"] = evaluate_reflection(reflection)
        return state

    def decide_node(state: LoopState) -> LoopState:
        evaluation = state.get("evaluation") or {}
        state["decision"] = decide_next_action(evaluation)
        return state

    def act_node(state: LoopState) -> LoopState:
        reflection = state.get("reflection") or {}
        decision   = state.get("decision", "continue_scan")
        point_id   = state.get("current_point_id")

        state.setdefault("history",        [])
        state.setdefault("interesting",    [])
        state.setdefault("contradictions", [])
        state.setdefault("flags_written",  0)

        if reflection:
            state["history"].append(reflection)
        if decision == "store_interesting" and reflection:
            state["interesting"].append(reflection)
        if decision == "track_contradiction" and reflection:
            state["contradictions"].append(reflection)

        # ── persist flags back to the reflection point in Qdrant ──────────────
        # Written for ALL decisions so downstream can filter by loop_decision.
        # Best-effort — a write failure never kills the loop.
        if state.get("persist_flags", True) and point_id and reflection:
            try:
                persist_flags(point_id, decision, target_collection)
                state["flags_written"] = state["flags_written"] + 1
            except Exception as exc:
                print(f"    [flag] write failed for {point_id}: {exc}")

        state["processed"] = state.get("processed", 0) + 1
        return state

    def route_after_act(state: LoopState) -> str:
        if state.get("processed", 0) >= state.get("limit", runtime.limit):
            return "end"
        return "continue"

    graph.add_node("get_chunk", get_chunk_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("decide", decide_node)
    graph.add_node("act", act_node)

    graph.set_entry_point("get_chunk")

    graph.add_conditional_edges(
        "get_chunk",
        route_after_get_chunk,
        {
            "reflect": "reflect",
            "end": END,
        },
    )
    graph.add_edge("reflect", "evaluate")
    graph.add_edge("evaluate", "decide")
    graph.add_edge("decide", "act")
    graph.add_conditional_edges(
        "act",
        route_after_act,
        {
            "continue": "get_chunk",
            "end": END,
        },
    )

    return graph.compile()


def print_step_summary(state: LoopState) -> None:
    chunk = state.get("current_chunk")
    evaluation = state.get("evaluation") or {}
    decision = state.get("decision", "unknown")
    processed = state.get("processed", 0)
    errors = state.get("errors", 0)
    last_error = state.get("last_error", "")

    if chunk is None:
        return

    source = getattr(chunk, "source_file", "unknown")
    page = getattr(chunk, "page", 0)
    chunk_index = getattr(chunk, "chunk_index", 0)

    confidence = float(evaluation.get("confidence", 0.0) or 0.0)
    claim_count = int(evaluation.get("claim_count", 0) or 0)
    concept_count = int(evaluation.get("concept_count", 0) or 0)

    decision_icon = {
        "store_interesting":   "★",
        "track_contradiction": "⚡",
        "continue_scan":       "·",
    }.get(decision, "?")

    status = "✗" if last_error else "✓"

    print(
        f"[{processed}] {status} {source} "
        f"p{page} c{chunk_index} | "
        f"{decision_icon} {decision} | "
        f"conf={confidence:.3f} claims={claim_count} concepts={concept_count} | "
        f"errs={errors}"
    )

    if last_error:
        print(f"    error: {last_error}")


def run_once(
    args: argparse.Namespace,
    model: str,
    target_collection: str,
    from_scratch: bool,
    persist_flags: bool,
) -> tuple[LoopState, float]:

    rf.CURRENT_MODEL     = model
    rf.TARGET_COLLECTION = target_collection

    print("\n[config]")
    print(f"  model:          {model}")
    print(f"  qdrant:         {rf.QDRANT_URL}")
    print(f"  embed_provider: {rf.EMBED_PROVIDER}")
    print(f"  embed_model:    {rf.EMBED_MODEL}")
    print(f"  target:         {rf.TARGET_COLLECTION}")
    print(f"  goal:           {args.goal}")
    print(f"  persist_flags:  {persist_flags}")

    rf.ensure_target_collection(from_scratch)
    rf.ensure_target_indexes()

    if persist_flags:
        print("[init] ensuring loop flag indexes...")
        ensure_flag_indexes(target_collection)

    runtime = ReflectLoopRuntime(limit=args.limit)
    app     = build_graph(runtime, target_collection)

    initial_state: LoopState = {
        "goal":             args.goal,
        "model":            model,
        "current_chunk":    None,
        "current_point_id": None,
        "reflection":       {},
        "evaluation":       {},
        "decision":         "",
        "history":          [],
        "interesting":      [],
        "contradictions":   [],
        "processed":        0,
        "errors":           0,
        "flags_written":    0,
        "limit":            args.limit,
        "done":             False,
        "last_error":       "",
        "persist_flags":    persist_flags,
    }

    print("\n[loop] starting\n")
    t0           = time.time()
    final_state: LoopState = initial_state

    for state_update in app.stream(initial_state):
        for node_name, state in state_update.items():
            final_state = state
            if not args.quiet and node_name == "act":
                print_step_summary(final_state)

    elapsed = time.time() - t0

    interesting_count   = len(final_state.get("interesting",   []))
    contradiction_count = len(final_state.get("contradictions",[]))
    flags_written       = final_state.get("flags_written", 0)

    print("\n[done]")
    print(f"  target:          {rf.TARGET_COLLECTION}")
    print(f"  processed:       {final_state.get('processed', 0)}")
    print(f"  errors:          {final_state.get('errors', 0)}")
    print(f"  interesting:     {interesting_count}")
    print(f"  contradictions:  {contradiction_count}")
    print(f"  flags_written:   {flags_written}")
    print(f"  elapsed:         {elapsed / 60:.1f}m")

    if interesting_count > 0:
        print(f"\n[flags] query interesting reflections:")
        print(f'  filter: {{"must": [{{"key": "loop_interesting", "match": {{"value": true}}}}]}}')
    if contradiction_count > 0:
        print(f"\n[flags] query contradiction-flagged reflections:")
        print(f'  filter: {{"must": [{{"key": "loop_contradiction", "match": {{"value": true}}}}]}}')

    return final_state, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stateful LangGraph reflection loop for Meta-Bridge"
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
        help="Maximum number of chunks to process",
    )

    parser.add_argument(
        "--target-collection",
        default=DEFAULT_TARGET_COLLECTION,
        help=(
            "Qdrant collection to write reflections into. "
            f"Default: {DEFAULT_TARGET_COLLECTION}"
        ),
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

    args = parser.parse_args()

    model = args.model.strip()

    if not model:
        raise RuntimeError("--model cannot be empty")

    if args.limit < 0:
        raise RuntimeError("--limit must be >= 0")

    if args.loop_interval < 0:
        raise RuntimeError("--loop-interval must be >= 0")

    if args.max_loops < 0:
        raise RuntimeError("--max-loops must be >= 0")

    target_collection = validate_target_collection(args.target_collection)
    validate_remote_config(model)

    persist_flags = not args.no_persist_flags

    if args.loop_interval <= 0:
        final_state, _ = run_once(args, model, target_collection, args.from_scratch, persist_flags)
        return 0 if final_state.get("errors", 0) == 0 else 1

    run_count    = 0
    total_errors = 0

    while True:
        run_count += 1
        print(f"\n[timer] run {run_count} starting")
        final_state, _ = run_once(
            args, model, target_collection,
            args.from_scratch if run_count == 1 else False,
            persist_flags,
        )
        total_errors += int(final_state.get("errors", 0) or 0)

        if args.max_loops > 0 and run_count >= args.max_loops:
            break

        print(f"\n[timer] sleeping {args.loop_interval:.1f}s before next run")
        time.sleep(args.loop_interval)

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
