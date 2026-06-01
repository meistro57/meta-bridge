#!/usr/bin/env bash
# rebuild.sh — force-rebuild all meta-bridge binaries and verify the result.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;91m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}=== meta-bridge rebuild ===${NC}"
echo "  repo: $REPO_DIR"
echo "  go:   $(go version)"
echo ""

# ── 1. Tidy modules ──────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/3] Tidying modules...${NC}"
go mod tidy
echo -e "${GREEN}      ok${NC}"

# ── 2. Force-rebuild all binaries ────────────────────────────────────────────
echo -e "${YELLOW}[2/3] Building binaries (forced)...${NC}"

BINARIES=(
    "mb:./cmd/mb"
    "mb-academic-test:./cmd/mb-academic-test"
)

for entry in "${BINARIES[@]}"; do
    name="${entry%%:*}"
    pkg="${entry##*:}"
    echo -n "      $name ... "
    if go build -a -o "$REPO_DIR/$name" "$pkg" 2>&1; then
        size=$(du -h "$REPO_DIR/$name" | cut -f1)
        echo -e "${GREEN}ok${NC} (${size})"
    else
        echo -e "${RED}FAILED${NC}"
        exit 1
    fi
done

# ── 3. Verify graphability index is present ───────────────────────────────────
echo -e "${YELLOW}[3/3] Checking assets...${NC}"

INDEX="$REPO_DIR/graphability_index.json"
if [[ -f "$INDEX" ]]; then
    size=$(du -h "$INDEX" | cut -f1)
    echo -e "${GREEN}      graphability_index.json found${NC} (${size})"
else
    echo -e "${RED}      WARNING: graphability_index.json not found at $INDEX${NC}"
    echo "      The scorer will fall back to scanning all chunks."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Build complete.${NC}"
echo ""
echo "  Binaries:"
for entry in "${BINARIES[@]}"; do
    name="${entry%%:*}"
    if [[ -f "$REPO_DIR/$name" ]]; then
        built=$(date -r "$REPO_DIR/$name" '+%Y-%m-%d %H:%M:%S')
        echo "    $REPO_DIR/$name  ($built)"
    fi
done
echo ""
