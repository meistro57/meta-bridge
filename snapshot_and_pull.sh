#!/usr/bin/env bash
# snapshot_and_pull.sh
#
# Creates fresh Qdrant snapshots for the core meta-bridge collections and
# downloads the actual .snapshot files locally, ready to move/copy anywhere
# (a drive, another machine, cloud storage, etc.) instead of leaving them
# stuck inside Qdrant's own managed storage.
#
# Requires: curl, jq
#
# Usage:
#   ./snapshot_and_pull.sh                # snapshots the default collection set
#   ./snapshot_and_pull.sh coll1 coll2     # snapshots a custom set instead

set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
OUT_DIR="${OUT_DIR:-$(pwd)/snapshots}"

COLLECTIONS=("$@")
if [ ${#COLLECTIONS[@]} -eq 0 ]; then
  COLLECTIONS=("mb_claims" "mb_chunks" "mb_sources" "meta_reflections" "vectoreology_findings" "misfit_reports")
fi

mkdir -p "$OUT_DIR"

echo "Snapshotting to: $OUT_DIR"
echo "Qdrant instance:  $QDRANT_URL"
echo ""

for collection in "${COLLECTIONS[@]}"; do
  echo "== $collection =="

  # 1. Create a fresh snapshot
  response=$(curl -sf -X POST "$QDRANT_URL/collections/$collection/snapshots")
  snapshot_name=$(echo "$response" | jq -r '.result.name')

  if [ -z "$snapshot_name" ] || [ "$snapshot_name" = "null" ]; then
    echo "  FAILED to create snapshot for $collection — response: $response"
    continue
  fi

  size_bytes=$(echo "$response" | jq -r '.result.size')
  echo "  created: $snapshot_name (${size_bytes} bytes)"

  # 2. Download it locally
  dest="$OUT_DIR/$snapshot_name"
  echo "  downloading -> $dest"
  curl -sf -o "$dest" \
    "$QDRANT_URL/collections/$collection/snapshots/$snapshot_name"

  actual_size=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")
  if [ "$actual_size" != "$size_bytes" ]; then
    echo "  WARNING: downloaded size ($actual_size) doesn't match reported size ($size_bytes) — verify before trusting this file"
  else
    echo "  OK: size matches ($actual_size bytes)"
  fi
  echo ""
done

echo "Done. Files are in: $OUT_DIR"
ls -lh "$OUT_DIR"
