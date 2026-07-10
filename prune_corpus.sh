#!/usr/bin/env bash
# prune_corpus.sh
# Removes low-signal and noise sources from all meta-bridge Qdrant collections
# Targets: literary fiction (3), Norse history (2), failed ingests (1)
#
# Collections pruned: mb_sources, mb_chunks, mb_claims, meta_reflections, misfit_reports
# Run from ~/meta-bridge/

set -euo pipefail

VENV_DIR="$HOME/meta-bridge/.venv"
if [[ -d "$VENV_DIR" ]]; then source "$VENV_DIR/bin/activate"; fi

GREEN='\033[0;32m'; AMBER='\033[0;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; RESET='\033[0m'
log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${RESET} $*"; }
ok()   { echo -e "${GREEN}  ✓${RESET} $*"; }
warn() { echo -e "${AMBER}  ⚠${RESET} $*"; }
err()  { echo -e "${RED}  ✗${RESET} $*"; }

# ── Sources to remove ─────────────────────────────────────────────────────────
# Format: "source_id|qdrant_point_id|title|reason"
declare -a PRUNE_TARGETS=(
  "ulysses_by_james_joyce||Ulysses (James Joyce)|literary fiction — contaminates consciousness clusters with literary metaphor"
  "the_satanic_verses|3276022453355149300|The Satanic Verses (Rushdie)|literary fiction — magical realism noise"
  "we_2||We (Zamiatin)|literary fiction — dystopian narrative, zero esoteric signal"
  "history_of_the_norwegian_people|6197215308673960000|History of the Norwegian People|0 claims from 699 chunks — pure historical narrative"
  "historyofnorwayf00boye||History of Norway (Boyesen)|Norse political history, 193 claims from 498 chunks — low signal"
  "gullveigarbok_by_vexior|678389920759700400|Gullveigarbok (Vexior)|failed ingest — 1 chunk, 0 claims"
)

echo ""
log "meta-bridge corpus prune — $(date '+%Y-%m-%d %H:%M')"
echo ""
warn "The following sources will be removed from ALL collections:"
echo ""
for entry in "${PRUNE_TARGETS[@]}"; do
  IFS='|' read -r sid qid title reason <<< "$entry"
  echo "  • $title"
  echo "    source_id: $sid"
  echo "    reason:    $reason"
  echo ""
done

read -rp "Proceed with pruning? [y/N] " confirm
if [[ "${confirm,,}" != "y" ]]; then
  echo "Aborted."
  exit 0
fi

echo ""

# ── Python prune script ───────────────────────────────────────────────────────
python3 << 'PYEOF'
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
client = QdrantClient(url=QDRANT_URL)

# Collections that store source_file in payload
COLLECTIONS_WITH_SOURCE = [
    "mb_chunks",
    "mb_claims",
    "meta_reflections",
    "misfit_reports",
]

# mb_sources uses the 'id' field in payload (not source_file)
MB_SOURCES_COLLECTION = "mb_sources"

SOURCES_TO_REMOVE = [
    "ulysses_by_james_joyce",
    "the_satanic_verses",
    "we_2",
    "history_of_the_norwegian_people",
    "historyofnorwayf00boye",
    "gullveigarbok_by_vexior",
]

print(f"\nConnected to Qdrant at {QDRANT_URL}")
print(f"Pruning {len(SOURCES_TO_REMOVE)} sources from {len(COLLECTIONS_WITH_SOURCE)+1} collections\n")

total_deleted = 0

# 1. Remove from mb_sources (keyed on payload.id)
print(f"── mb_sources ──")
for source_id in SOURCES_TO_REMOVE:
    result = client.delete(
        collection_name=MB_SOURCES_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="id", match=MatchValue(value=source_id))]
        ),
    )
    op = result.operation_id if hasattr(result, 'operation_id') else 'done'
    print(f"  deleted '{source_id}' from mb_sources [{op}]")
    total_deleted += 1

print()

# 2. Remove from all other collections (keyed on payload.source_file)
for collection in COLLECTIONS_WITH_SOURCE:
    print(f"── {collection} ──")
    col_deleted = 0
    for source_id in SOURCES_TO_REMOVE:
        # Count first
        try:
            count_result = client.count(
                collection_name=collection,
                count_filter=Filter(
                    must=[FieldCondition(key="source_file", match=MatchValue(value=source_id))]
                ),
                exact=True,
            )
            n = count_result.count
        except Exception:
            n = "?"

        result = client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_id))]
            ),
        )
        print(f"  deleted ~{n} pts for '{source_id}'")
        col_deleted += 1

    print(f"  {col_deleted} sources cleared from {collection}")
    total_deleted += col_deleted
    print()

print(f"✓ Prune complete — {total_deleted} delete operations across all collections")
print("  Run the Vectoreologist next to re-cluster with clean corpus.")
PYEOF

echo ""
ok "Prune complete."
echo ""
log "Next steps:"
echo "  1. Verify counts:  mb check-counts"
echo "  2. Re-run Vectoreologist to re-cluster without noise"
echo "  3. Consider Cannon/Bashar/Plato chunk weighting before next Vectoreologist run"
echo ""
