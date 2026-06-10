#!/usr/bin/env bash
# gitpush.sh — commit & push vectoreologist, MisfitCrew, meta-bridge
#
# Usage:
#   ./gitpush.sh                        # prompts for a shared commit message
#   ./gitpush.sh "your commit message"  # skips the prompt

set -uo pipefail

REPOS=(
    "/home/mark/vectoreologist"
    "/home/mark/MisfitCrew"
    "/home/mark/meta-bridge"
)

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
BOLD="\033[1m"
RESET="\033[0m"

# ── get commit message ────────────────────────────────────────────────────────
if [[ $# -ge 1 && -n "$1" ]]; then
    MSG="$1"
else
    echo -e "${BOLD}Commit message:${RESET} "
    read -r MSG
    [[ -z "$MSG" ]] && { echo -e "${RED}[!] Empty message — aborting.${RESET}"; exit 1; }
fi

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  Message:${RESET} $MSG"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

OK=0
SKIPPED=0
FAILED=0

for repo in "${REPOS[@]}"; do
    name=$(basename "$repo")
    echo -e "${BOLD}▶ $name${RESET}"

    if [[ ! -d "$repo/.git" ]]; then
        echo -e "  ${YELLOW}⚠ not a git repo — skipping${RESET}"
        ((SKIPPED++))
        echo ""
        continue
    fi

    cd "$repo"

    # show short status
    status=$(git status --short 2>&1)
    if [[ -z "$status" ]]; then
        echo -e "  ${YELLOW}✓ nothing to commit — skipping${RESET}"
        ((SKIPPED++))
        echo ""
        continue
    fi

    echo "$status" | sed 's/^/  /'
    echo ""

    # stage all, commit, push
    if git add -A \
    && git commit -m "$MSG" \
    && git push; then
        echo -e "  ${GREEN}✓ pushed${RESET}"
        ((OK++))
    else
        echo -e "  ${RED}✗ failed — check output above${RESET}"
        ((FAILED++))
    fi

    echo ""
done

# ── summary ───────────────────────────────────────────────────────────────────
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  ${GREEN}pushed: $OK${RESET}   ${YELLOW}skipped: $SKIPPED${RESET}   ${RED}failed: $FAILED${RESET}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

[[ $FAILED -gt 0 ]] && exit 1
exit 0
