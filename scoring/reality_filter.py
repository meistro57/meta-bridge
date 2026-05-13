#!/usr/bin/env python3
"""
reality_filter.py — Reality Filter v0 for Meta Bridge semantic bridges.

Classifies bridges between knowledge clusters with epistemic rigor:
  - bridge_type: Structural | Analogical | Speculative | Testable | Contradictory
  - constraint_flags: physics constraint violations detected
  - testability_score: 0-5
  - epistemic_status: plain-language honest label

Usage:
    # Score bridges from a vectoreology JSON report
    python scoring/reality_filter.py findings/vectoreology_2026-04-28.json

    # Score a single bridge interactively
    python scoring/reality_filter.py --interactive

    # Dry-run (print prompts, don't call LLM)
    python scoring/reality_filter.py --dry-run findings/vectoreology_2026-04-28.json

Reads .env for OPENROUTER_API_KEY, MB_MODEL.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key and key not in os.environ:
                os.environ[key] = val


load_env()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it"
MODEL = os.environ.get("MB_FILTER_MODEL", "") or os.environ.get("MB_MODEL", "") or DEFAULT_MODEL

PROMPT_PATH = Path(__file__).parent / "reality_filter_prompt.md"

# Valid enum values for validation
VALID_BRIDGE_TYPES = {"Structural", "Analogical", "Speculative", "Testable", "Contradictory"}
VALID_CONSTRAINT_FLAGS = {
    "no-communication-theorem",
    "decoherence-scale-gap",
    "measurement-problem-conflation",
    "energy-conservation-violation",
    "unfalsifiable",
    "equivocation",
    "category-error",
}
VALID_EPISTEMIC_STATUSES = {
    "Established mapping",
    "Plausible analogy",
    "Suggestive but ungrounded",
    "Poetic metaphor",
    "Needs disambiguation",
    "Likely invalid",
    "Contradicted by evidence",
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def load_rubric() -> str:
    """Load the reality filter rubric prompt."""
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Rubric prompt not found at {PROMPT_PATH}")


def build_bridge_input(bridge: dict[str, Any], clusters: dict[int, dict]) -> str:
    """Build the user prompt for a single bridge."""
    cluster_a = clusters.get(bridge.get("cluster_a", -1), {})
    cluster_b = clusters.get(bridge.get("cluster_b", -1), {})

    parts = [
        f"Bridge strength: {bridge.get('strength', 'unknown')}",
        f"Link type: {bridge.get('link_type', 'unknown')}",
        f"Cluster A (ID {bridge.get('cluster_a', '?')}): {cluster_a.get('label', 'unlabeled')}",
        f"  Size: {cluster_a.get('size', '?')} chunks, Coherence: {cluster_a.get('coherence', '?')}",
        f"Cluster B (ID {bridge.get('cluster_b', '?')}): {cluster_b.get('label', 'unlabeled')}",
        f"  Size: {cluster_b.get('size', '?')} chunks, Coherence: {cluster_b.get('coherence', '?')}",
    ]

    # Include sample links if available
    sample_links = bridge.get("sample_links", [])
    if sample_links:
        parts.append("\nSample cross-cluster chunk pairs:")
        for i, link in enumerate(sample_links[:3]):
            parts.append(f"  Pair {i+1}: chunk {link.get('chunk_a_id', '?')} <-> chunk {link.get('chunk_b_id', '?')} (sim={link.get('similarity', '?'):.3f})")

    # Include bridge label/description if the reasoner already produced one
    if bridge.get("label"):
        parts.append(f"\nBridge description (from prior analysis): {bridge['label']}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call OpenRouter for bridge scoring."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/meistro57/meta-bridge",
        "X-Title": "Meta Bridge Reality Filter",
    }

    resp = requests.post(OPENROUTER_CHAT_URL, json=body, headers=headers, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"openrouter {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"openrouter error: {data['error'].get('message', 'unknown')}")

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"openrouter returned no choices")

    content = (choices[0].get("message", {}).get("content", "")).strip()
    if not content:
        raise RuntimeError("openrouter returned empty message")

    return content


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def parse_score(raw: str) -> dict[str, Any]:
    """Parse and validate the LLM's scoring response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0].strip()

    # Extract JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end+1]

    score = json.loads(raw)

    # Validate and normalize
    if score.get("bridge_type") not in VALID_BRIDGE_TYPES:
        score["bridge_type"] = "Speculative"  # safe default

    if not isinstance(score.get("constraint_flags"), list):
        score["constraint_flags"] = []
    # Allow flags not in our predefined set (LLM might identify new ones)

    ts = score.get("testability_score", 0)
    if not isinstance(ts, (int, float)) or ts < 0 or ts > 5:
        score["testability_score"] = 0
    else:
        score["testability_score"] = int(ts)

    if score.get("epistemic_status") not in VALID_EPISTEMIC_STATUSES:
        # Keep it if it's a reasonable string, just flag it
        if not isinstance(score.get("epistemic_status"), str) or not score["epistemic_status"].strip():
            score["epistemic_status"] = "Needs disambiguation"

    if not isinstance(score.get("reasoning"), str):
        score["reasoning"] = ""

    return score


def score_bridge(bridge: dict[str, Any], clusters: dict[int, dict],
                 rubric: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """
    Score a single bridge with the reality filter.

    Returns dict with: bridge_type, constraint_flags, testability_score,
    epistemic_status, reasoning
    """
    if rubric is None:
        rubric = load_rubric()

    user_prompt = build_bridge_input(bridge, clusters)

    if dry_run:
        return {
            "bridge_type": "Speculative",
            "constraint_flags": [],
            "testability_score": 0,
            "epistemic_status": "Needs disambiguation",
            "reasoning": f"[DRY RUN] Would score bridge {bridge.get('cluster_a')}->{bridge.get('cluster_b')}",
            "_prompt": user_prompt,
        }

    raw = call_llm(rubric, user_prompt)
    score = parse_score(raw)
    return score


# ---------------------------------------------------------------------------
# Batch processing from vectoreology JSON
# ---------------------------------------------------------------------------

def load_vectoreology_report(path: str) -> dict[str, Any]:
    """Load a vectoreology JSON report (from Vectoreologist)."""
    with open(path) as f:
        return json.load(f)


def score_all_bridges(report: dict[str, Any], dry_run: bool = False) -> list[dict[str, Any]]:
    """Score all bridges in a vectoreology report."""
    rubric = load_rubric()

    # Build cluster lookup
    clusters_raw = report.get("clusters", [])
    clusters = {}
    for c in clusters_raw:
        cid = c.get("id", c.get("ID"))
        clusters[cid] = c

    bridges = report.get("bridges", [])
    scored = []

    for i, bridge in enumerate(bridges):
        print(f"  [{i+1}/{len(bridges)}] Scoring bridge "
              f"{bridge.get('cluster_a', '?')}->{bridge.get('cluster_b', '?')} "
              f"(strength={bridge.get('strength', '?')})")

        try:
            score = score_bridge(bridge, clusters, rubric=rubric, dry_run=dry_run)
            result = {**bridge, **score}
            scored.append(result)
            print(f"    -> {score['bridge_type']} | {score['epistemic_status']} | "
                  f"testability={score['testability_score']} | "
                  f"flags={score['constraint_flags']}")
        except Exception as e:
            print(f"    ! error: {e}")
            scored.append({**bridge, "error": str(e)})

    return scored


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_epistemic_header(scored_bridges: list[dict]) -> str:
    """Render an Epistemic Status summary block for the top of a report."""
    lines = ["## Epistemic Status Summary\n"]

    # Count by type
    type_counts: dict[str, int] = {}
    for b in scored_bridges:
        bt = b.get("bridge_type", "unknown")
        type_counts[bt] = type_counts.get(bt, 0) + 1

    lines.append("| Bridge Type | Count |")
    lines.append("|-------------|-------|")
    for bt in ["Structural", "Analogical", "Speculative", "Testable", "Contradictory"]:
        if bt in type_counts:
            lines.append(f"| {bt} | {type_counts[bt]} |")

    # Flagged bridges
    flagged = [b for b in scored_bridges if b.get("constraint_flags")]
    if flagged:
        lines.append(f"\n**{len(flagged)} bridge(s) flagged** with physics constraint concerns.")

    # Testable bridges
    testable = [b for b in scored_bridges if b.get("testability_score", 0) >= 3]
    if testable:
        lines.append(f"\n**{len(testable)} bridge(s)** with testability score >= 3 (actionable predictions).")

    lines.append("")
    return "\n".join(lines)


def render_bridge_table(scored_bridges: list[dict]) -> str:
    """Render scored bridges as a markdown table."""
    lines = [
        "## Scored Bridges\n",
        "| A | B | Strength | Type | Epistemic Status | Testability | Flags |",
        "|---|---|----------|------|------------------|-------------|-------|",
    ]
    for b in scored_bridges:
        flags = ", ".join(b.get("constraint_flags", [])) or "—"
        lines.append(
            f"| {b.get('cluster_a', '?')} "
            f"| {b.get('cluster_b', '?')} "
            f"| {b.get('strength', 0):.2f} "
            f"| {b.get('bridge_type', '?')} "
            f"| {b.get('epistemic_status', '?')} "
            f"| {b.get('testability_score', 0)} "
            f"| {flags} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MODEL
    import argparse

    parser = argparse.ArgumentParser(description="Reality Filter v0 for Meta Bridge bridges")
    parser.add_argument("report", nargs="?", help="Path to vectoreology JSON report")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling LLM")
    parser.add_argument("--output", default=None, help="Output path for scored JSON (default: <report>.scored.json)")
    parser.add_argument("--model", default=None, help=f"Override scoring model (default: {MODEL})")
    args = parser.parse_args()

    if args.model:
        MODEL = args.model

    if not args.report:
        print("Usage: python scoring/reality_filter.py <vectoreology_report.json>")
        print("       python scoring/reality_filter.py --dry-run <report.json>")
        sys.exit(1)

    if not os.path.exists(args.report):
        print(f"File not found: {args.report}")
        sys.exit(1)

    print(f"[reality-filter] Loading {args.report}")
    report = load_vectoreology_report(args.report)

    bridges = report.get("bridges", [])
    print(f"[reality-filter] Found {len(bridges)} bridges to score (model={MODEL})")

    if not bridges:
        print("[reality-filter] No bridges found. Nothing to score.")
        sys.exit(0)

    t0 = time.time()
    scored = score_all_bridges(report, dry_run=args.dry_run)
    elapsed = time.time() - t0

    # Write scored output
    output_path = args.output or args.report.replace(".json", ".scored.json")
    output = {
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL,
        "source_report": args.report,
        "bridges": scored,
        "summary": {
            "total": len(scored),
            "by_type": {},
            "flagged": len([b for b in scored if b.get("constraint_flags")]),
            "testable": len([b for b in scored if b.get("testability_score", 0) >= 3]),
        },
    }
    for b in scored:
        bt = b.get("bridge_type", "unknown")
        output["summary"]["by_type"][bt] = output["summary"]["by_type"].get(bt, 0) + 1

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    header = render_epistemic_header(scored)
    print(f"\n{header}")
    print(f"[reality-filter] Done in {elapsed:.1f}s")
    print(f"[reality-filter] Scored output: {output_path}")


if __name__ == "__main__":
    main()
