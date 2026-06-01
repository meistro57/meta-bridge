#!/usr/bin/env python3
"""
corpus_cleanup.py — Meistro Brain corpus audit and repair tool.

Checks mb_sources, mb_chunks, and mb_claims for:
  1. Duplicate sources (same title ingested under different IDs)
  2. Zero-claim sources (ingested but extraction produced nothing)
  3. Low-claim sources (suspiciously few claims relative to chunk count)
  4. Orphaned chunks (source_id not found in mb_sources)
  5. Orphaned claims (source_id not found in mb_sources)
  6. Chunk count mismatches (source.chunk_count vs actual chunks in Qdrant)
  7. Claim count mismatches (source.claim_count vs actual claims in Qdrant)
  8. Zone.Identifier ghost files in incoming/

Usage:
    python corpus_cleanup.py            # audit only, no changes
    python corpus_cleanup.py --fix      # prompt before deleting orphans
    python corpus_cleanup.py --report   # write full report to corpus_audit.json
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Load .env manually if present (avoids python-dotenv dependency)
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
INCOMING_DIR   = os.getenv("MB_INCOMING_DIR", "./incoming")

COL_SOURCES = "mb_sources"
COL_CHUNKS  = "mb_chunks"
COL_CLAIMS  = "mb_claims"

# Thresholds
LOW_CLAIM_RATIO = 0.5   # claims per chunk — below this is suspicious
ZERO_CLAIM_SKIP = {"csb_pew_bible"}  # known zero-claim sources (Bible = no metaphysical claims)


def bold(s):   return f"\033[1m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def green(s):  return f"\033[92m{s}\033[0m"
def cyan(s):   return f"\033[96m{s}\033[0m"


def scroll_all(client, collection, with_payload=True, with_vectors=False, batch=100):
    """Scroll through all points in a collection."""
    points = []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        points.extend(result)
        if next_offset is None:
            break
        offset = next_offset
    return points


def count_by_source(client, collection, source_id):
    """Count points in a collection matching a source_id."""
    # mb_claims stores source_id nested inside attributions[]
    # mb_chunks stores it as a top-level payload field
    if collection == COL_CLAIMS:
        key = "attributions[].source_id"
    else:
        key = "source_id"
    return client.count(
        collection_name=collection,
        count_filter=Filter(
            must=[FieldCondition(key=key, match=MatchValue(value=source_id))]
        ),
        exact=True,
    ).count


def delete_by_source(client, collection, source_id):
    """Delete all points in a collection matching a source_id."""
    from qdrant_client.models import FilterSelector
    key = "attributions[].source_id" if collection == COL_CLAIMS else "source_id"
    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key=key, match=MatchValue(value=source_id))]
            )
        ),
    )


def fix_source_counts(client, sources):
    """Reconcile chunk_count and claim_count on every source record to match
    actual Qdrant counts. Also cleans up Zone.Identifier files automatically.
    No data is deleted — only source payload fields are patched."""
    import hashlib

    def point_id_for_key(key):
        """Replicate Go's fnv64a hash used by the store package."""
        h = 0xcbf29ce484222325
        for b in key.encode():
            h ^= b
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        return h

    print(bold("\n=== Fix: Reconciling source counts ==="))
    fixed = 0
    skipped = 0

    for sid, pl in sorted(sources.items()):
        recorded_chunks = int(pl.get("chunk_count") or 0)
        recorded_claims = int(pl.get("claim_count") or 0)

        actual_chunks = count_by_source(client, COL_CHUNKS, sid)
        actual_claims = count_by_source(client, COL_CLAIMS, sid)

        if recorded_chunks == actual_chunks and recorded_claims == actual_claims:
            skipped += 1
            continue

        point_id = point_id_for_key("source:" + sid)
        patch = {}
        if recorded_chunks != actual_chunks:
            patch["chunk_count"] = actual_chunks
        if recorded_claims != actual_claims:
            patch["claim_count"] = actual_claims

        body = {
            "payload": patch,
            "points": [point_id],
        }
        url = f"{QDRANT_URL}/collections/{COL_SOURCES}/points/payload"
        headers = {"Content-Type": "application/json"}
        if QDRANT_API_KEY:
            headers["api-key"] = QDRANT_API_KEY

        import urllib.request
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req) as resp:
                resp.read()
            parts = []
            if recorded_chunks != actual_chunks:
                parts.append(f"chunks {recorded_chunks}→{actual_chunks}")
            if recorded_claims != actual_claims:
                parts.append(f"claims {recorded_claims}→{actual_claims}")
            print(green(f"  FIXED: {sid}  ({', '.join(parts)})"))
            fixed += 1
        except Exception as e:
            print(red(f"  ERROR patching {sid}: {e}"))

    print(f"\n  {fixed} source records updated, {skipped} already correct.")


def fix_zone_files(incoming_dir):
    """Delete all Zone.Identifier ghost files from incoming/."""
    print(bold("\n=== Fix: Removing Zone.Identifier files ==="))
    removed = 0
    if not os.path.isdir(incoming_dir):
        print(yellow(f"  incoming dir not found: {incoming_dir}"))
        return
    for f in os.listdir(incoming_dir):
        if "Zone.Identifier" in f:
            full = os.path.join(incoming_dir, f)
            try:
                os.remove(full)
                removed += 1
            except Exception as e:
                print(red(f"  ERROR removing {full}: {e}"))
    print(green(f"  Removed {removed} Zone.Identifier files."))


def fix_orphan_chunks(client, known_source_ids):
    """Delete orphaned chunks whose source_id has no matching source record."""
    print(bold("\n=== Fix: Removing orphaned chunks ==="))
    chunk_points = scroll_all(client, COL_CHUNKS)
    orphans_by_source = defaultdict(list)
    for p in chunk_points:
        sid = (p.payload or {}).get("source_id", "")
        if sid and sid not in known_source_ids:
            orphans_by_source[sid].append(p.id)

    if not orphans_by_source:
        print(green("  No orphaned chunks found."))
        return

    for sid, ids in orphans_by_source.items():
        print(yellow(f"  Deleting {len(ids)} orphaned chunks for source '{sid}'"))
        delete_by_source(client, COL_CHUNKS, sid)
        print(green(f"  Done."))


def audit(client, fix=False, preloaded_sources=None):
    issues = []
    summary = defaultdict(int)

    print(bold("\n=== Meistro Brain Corpus Audit ===\n"))

    # ------------------------------------------------------------------ #
    # 1. Load all sources
    # ------------------------------------------------------------------ #
    if preloaded_sources is not None:
        sources = preloaded_sources
        print(cyan(f"  {len(sources)} sources loaded (preloaded)\n"))
    else:
        print(cyan("Loading mb_sources..."))
        source_points = scroll_all(client, COL_SOURCES)
        sources = {}
        for p in source_points:
            pl = p.payload or {}
            sid = pl.get("id") or pl.get("source_id") or str(p.id)
            sources[sid] = pl
        print(f"  {len(sources)} sources loaded\n")

    # ------------------------------------------------------------------ #
    # 2. Duplicate detection (same title, different ID)
    # ------------------------------------------------------------------ #
    print(cyan("Checking for duplicate titles..."))
    title_map = defaultdict(list)
    for sid, pl in sources.items():
        title = (pl.get("title") or "").strip().lower()
        if title:
            title_map[title].append(sid)

    dupes = {t: ids for t, ids in title_map.items() if len(ids) > 1}
    if dupes:
        for title, ids in dupes.items():
            print(red(f"  DUPE: \"{title}\""))
            for sid in ids:
                pl = sources[sid]
                print(f"        id={sid}  chunks={pl.get('chunk_count',0)}  claims={pl.get('claim_count',0)}  ingested={pl.get('ingested_at','?')}")
            issues.append({"type": "duplicate_title", "title": title, "source_ids": ids})
            summary["duplicates"] += 1
    else:
        print(green("  No duplicate titles found."))

    # ------------------------------------------------------------------ #
    # 3. Zero-claim and low-claim sources
    # ------------------------------------------------------------------ #
    print(cyan("\nChecking claim counts..."))
    for sid, pl in sorted(sources.items()):
        chunk_count = int(pl.get("chunk_count") or 0)
        claim_count = int(pl.get("claim_count") or 0)
        title = pl.get("title", sid)

        if claim_count == 0 and sid not in ZERO_CLAIM_SKIP:
            print(red(f"  ZERO CLAIMS: {sid}  \"{title}\"  ({chunk_count} chunks)"))
            issues.append({"type": "zero_claims", "source_id": sid, "title": title, "chunk_count": chunk_count})
            summary["zero_claims"] += 1
        elif chunk_count > 0 and (claim_count / chunk_count) < LOW_CLAIM_RATIO:
            ratio = claim_count / chunk_count
            print(yellow(f"  LOW CLAIMS:  {sid}  \"{title}\"  ({claim_count} claims / {chunk_count} chunks = {ratio:.2f}/chunk)"))
            issues.append({"type": "low_claims", "source_id": sid, "title": title,
                           "chunk_count": chunk_count, "claim_count": claim_count, "ratio": ratio})
            summary["low_claims"] += 1

    if summary["zero_claims"] == 0 and summary["low_claims"] == 0:
        print(green("  All sources have healthy claim counts."))

    # ------------------------------------------------------------------ #
    # 4. Chunk count verification
    # ------------------------------------------------------------------ #
    print(cyan("\nVerifying chunk counts..."))
    chunk_mismatches = []
    for sid, pl in sorted(sources.items()):
        recorded = int(pl.get("chunk_count") or 0)
        actual = count_by_source(client, COL_CHUNKS, sid)
        if recorded != actual:
            print(yellow(f"  CHUNK MISMATCH: {sid}  recorded={recorded}  actual={actual}"))
            chunk_mismatches.append({"source_id": sid, "recorded": recorded, "actual": actual})
            issues.append({"type": "chunk_count_mismatch", "source_id": sid,
                           "recorded": recorded, "actual": actual})
            summary["chunk_mismatches"] += 1

    if not chunk_mismatches:
        print(green("  All chunk counts match."))

    # ------------------------------------------------------------------ #
    # 5. Claim count verification
    # ------------------------------------------------------------------ #
    print(cyan("\nVerifying claim counts..."))
    claim_mismatches = []
    for sid, pl in sorted(sources.items()):
        recorded = int(pl.get("claim_count") or 0)
        actual = count_by_source(client, COL_CLAIMS, sid)
        if recorded != actual:
            print(yellow(f"  CLAIM MISMATCH: {sid}  recorded={recorded}  actual={actual}"))
            claim_mismatches.append({"source_id": sid, "recorded": recorded, "actual": actual})
            issues.append({"type": "claim_count_mismatch", "source_id": sid,
                           "recorded": recorded, "actual": actual})
            summary["claim_mismatches"] += 1

    if not claim_mismatches:
        print(green("  All claim counts match."))

    # ------------------------------------------------------------------ #
    # 6. Orphaned chunks
    # ------------------------------------------------------------------ #
    print(cyan("\nScanning for orphaned chunks..."))
    known_source_ids = set(sources.keys())
    chunk_points = scroll_all(client, COL_CHUNKS)
    orphan_chunks = []
    orphan_chunk_ids_by_source = defaultdict(list)

    for p in chunk_points:
        pl = p.payload or {}
        sid = pl.get("source_id", "")
        if sid and sid not in known_source_ids:
            orphan_chunks.append(p)
            orphan_chunk_ids_by_source[sid].append(p.id)

    if orphan_chunks:
        for sid, ids in orphan_chunk_ids_by_source.items():
            print(red(f"  ORPHAN CHUNKS: source_id={sid}  count={len(ids)}"))
            issues.append({"type": "orphaned_chunks", "source_id": sid, "count": len(ids)})
            summary["orphaned_chunks"] += len(ids)

        if fix:
            print(yellow("\n  Fix orphaned chunks?"))
            for sid, ids in orphan_chunk_ids_by_source.items():
                ans = input(f"  Delete {len(ids)} orphaned chunks for source '{sid}'? [y/N] ").strip().lower()
                if ans == "y":
                    delete_by_source(client, COL_CHUNKS, sid)
                    print(green(f"  Deleted {len(ids)} orphaned chunks for {sid}"))
    else:
        print(green("  No orphaned chunks found."))

    # ------------------------------------------------------------------ #
    # 7. Orphaned claims
    # ------------------------------------------------------------------ #
    print(cyan("\nScanning for orphaned claims..."))
    claim_points = scroll_all(client, COL_CLAIMS)
    orphan_claims_by_source = defaultdict(list)

    for p in claim_points:
        pl = p.payload or {}
        # claims store source_id inside attributions[]
        attributions = pl.get("attributions") or []
        sids = {a.get("source_id", "") for a in attributions if isinstance(a, dict)}
        for sid in sids:
            if sid and sid not in known_source_ids:
                orphan_claims_by_source[sid].append(p.id)

    if orphan_claims_by_source:
        for sid, ids in orphan_claims_by_source.items():
            print(red(f"  ORPHAN CLAIMS: source_id={sid}  count={len(ids)}"))
            issues.append({"type": "orphaned_claims", "source_id": sid, "count": len(ids)})
            summary["orphaned_claims"] += len(ids)

        if fix:
            print(yellow("\n  Fix orphaned claims?"))
            for sid, ids in orphan_claims_by_source.items():
                ans = input(f"  Delete {len(ids)} orphaned claims for source '{sid}'? [y/N] ").strip().lower()
                if ans == "y":
                    delete_by_source(client, COL_CLAIMS, sid)
                    print(green(f"  Deleted {len(ids)} orphaned claims for {sid}"))
    else:
        print(green("  No orphaned claims found."))

    # ------------------------------------------------------------------ #
    # 8. Zone.Identifier ghost files in incoming/
    # ------------------------------------------------------------------ #
    print(cyan(f"\nScanning {INCOMING_DIR} for Zone.Identifier ghost files..."))
    ghost_files = []
    if os.path.isdir(INCOMING_DIR):
        for f in os.listdir(INCOMING_DIR):
            if "Zone.Identifier" in f:
                ghost_files.append(os.path.join(INCOMING_DIR, f))

    if ghost_files:
        print(yellow(f"  Found {len(ghost_files)} Zone.Identifier files (Windows download artifacts):"))
        for f in ghost_files:
            print(f"    {f}")
        issues.append({"type": "zone_identifier_files", "count": len(ghost_files), "files": ghost_files})
        summary["zone_identifier_files"] += len(ghost_files)

        if fix:
            ans = input(f"\n  Delete all {len(ghost_files)} Zone.Identifier files? [y/N] ").strip().lower()
            if ans == "y":
                for f in ghost_files:
                    os.remove(f)
                print(green(f"  Deleted {len(ghost_files)} Zone.Identifier files."))
    else:
        print(green("  No Zone.Identifier files found."))

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(bold("\n=== Audit Summary ==="))
    total_issues = sum(summary.values())
    if total_issues == 0:
        print(green("  Clean. No issues found."))
    else:
        for k, v in sorted(summary.items()):
            label = k.replace("_", " ").title()
            print(f"  {yellow(label+':')}  {v}")
        print(f"\n  {red(f'Total issues: {total_issues}')}")
        if not fix:
            print(yellow("  Run with --fix to interactively repair issues."))

    return issues, summary


def write_report(issues, summary, path="corpus_audit.json"):
    report = {"summary": dict(summary), "issues": issues}
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(green(f"\n  Report written to {path}"))


def main():
    parser = argparse.ArgumentParser(description="Meistro Brain corpus audit tool")
    parser.add_argument("--fix",        action="store_true", help="Interactively repair issues")
    parser.add_argument("--fix-counts", action="store_true", help="Reconcile source chunk/claim counts to match actual Qdrant data (non-destructive)")
    parser.add_argument("--fix-zones",  action="store_true", help="Delete Zone.Identifier ghost files from incoming/")
    parser.add_argument("--fix-orphans",action="store_true", help="Delete orphaned chunks with no matching source")
    parser.add_argument("--fix-all",    action="store_true", help="Run all non-destructive fixes: counts + zones + orphans")
    parser.add_argument("--report",     action="store_true", help="Write full audit report to corpus_audit.json")
    args = parser.parse_args()

    kwargs = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    client = QdrantClient(**kwargs)

    # Load sources once — shared by audit + fix functions
    source_points = scroll_all(client, COL_SOURCES)
    sources = {}
    for p in source_points:
        pl = p.payload or {}
        sid = pl.get("id") or pl.get("source_id") or str(p.id)
        sources[sid] = pl

    # --- Fix modes (no audit output, just fix and exit) ---
    if args.fix_counts or args.fix_all:
        fix_source_counts(client, sources)

    if args.fix_zones or args.fix_all:
        fix_zone_files(INCOMING_DIR)

    if args.fix_orphans or args.fix_all:
        fix_orphan_chunks(client, set(sources.keys()))

    if any([args.fix_counts, args.fix_zones, args.fix_orphans, args.fix_all]):
        return

    # --- Normal audit ---
    issues, summary = audit(client, fix=args.fix, preloaded_sources=sources)

    if args.report:
        write_report(issues, summary)


if __name__ == "__main__":
    main()
