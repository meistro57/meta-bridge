#!/usr/bin/env bash
# filename: fetch_reddit_requested_sources.sh
# Downloads the public-domain / freely-available sources requested in the
# r/OperationNewEarth thread. Idempotent — skips files that already exist.
#
# Usage:
#   cd ~/meta-bridge
#   chmod +x fetch_reddit_requested_sources.sh
#   ./fetch_reddit_requested_sources.sh

set -e

INCOMING="${INCOMING:-$HOME/meta-bridge/incoming}"
mkdir -p "$INCOMING"
cd "$INCOMING"

UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

fetch() {
    local name="$1"
    local url="$2"
    local out="$3"

    if [ -f "$out" ]; then
        echo "✓ skip   $out (already present)"
        return 0
    fi

    echo "→ fetch  $name"
    echo "         $url"
    if curl -L -A "$UA" --fail --silent --show-error -o "$out.part" "$url"; then
        mv "$out.part" "$out"
        local size
        size=$(du -h "$out" | cut -f1)
        echo "✓ saved  $out  ($size)"
    else
        rm -f "$out.part"
        echo "✗ FAILED $name — fetch manually from $url"
    fi
    echo
}

echo "============================================================"
echo "Fetching Reddit-requested sources into $INCOMING"
echo "============================================================"
echo

# --- Tao Te Ching (Legge 1891, public domain) ---
# Requested by u/PepperMinimum5460. Sacred Books of the East vol 39.
fetch "Tao Te Ching (Legge 1891)" \
    "https://archive.org/download/wg939/wg939.pdf" \
    "tao_te_ching_legge.pdf"

# --- The Perennial Philosophy (Huxley 1945) ---
# Requested by u/ihateyouguys. Archive.org open scan.
fetch "The Perennial Philosophy (Huxley)" \
    "https://archive.org/download/perennialphilosp035505mbp/perennialphilosp035505mbp.pdf" \
    "perennial_philosophy_huxley.pdf"

# --- The Book of the Law (Crowley 1909, public domain in US) ---
# Implicit request via u/wwarr (Crowley/ritual magic). Liber AL vel Legis.
fetch "The Book of the Law (Crowley)" \
    "https://archive.org/download/CrowleyTheBookOfTheLaw/Crowley%20-%20The%20Book%20of%20the%20Law.pdf" \
    "crowley_book_of_the_law.pdf"

# --- Sefer Yetzirah (Westcott 1887, public domain) ---
# REPLACEMENT for the broken sefer_yetzirah in your corpus (1 chunk, 0 claims).
# Will need to be re-ingested over the old broken record.
fetch "Sepher Yetzirah (Westcott)" \
    "https://archive.org/download/WestcottWWSepherYetzirah/Westcott%20W%20W%20-%20Sepher%20Yetzirah.pdf" \
    "sefer_yetzirah_westcott.pdf"

# --- Nag Hammadi Library (Robinson 1988 — Brill copyright, on Archive lending) ---
# This one's tricky. Robinson's edition is still under copyright.
# Best free path: gnosis.org has all the texts as individual HTML files
# (the actual Robinson translations, hosted with permission from the
# Coptic Gnostic Library project).
# We'll grab them and concatenate.
NAGHAMMADI_OUT="nag_hammadi_library_robinson.txt"
if [ -f "$NAGHAMMADI_OUT" ]; then
    echo "✓ skip   $NAGHAMMADI_OUT (already present)"
else
    echo "→ fetch  Nag Hammadi Library (gnosis.org — Robinson translations)"

    # Index of all Nag Hammadi tractates on gnosis.org
    NAGHAMMADI_TEXTS=(
        "prayp.html:Prayer-of-the-Apostle-Paul"
        "apocjames.html:Apocryphon-of-James"
        "got.html:Gospel-of-Truth"
        "treatres.html:Treatise-on-Resurrection"
        "tripart.html:Tripartite-Tractate"
        "apocjn.html:Apocryphon-of-John"
        "gthomas.html:Gospel-of-Thomas"
        "gop.html:Gospel-of-Philip"
        "hypostas.html:Hypostasis-of-the-Archons"
        "origin.html:On-the-Origin-of-the-World"
        "exegesis.html:Exegesis-on-the-Soul"
        "thomas.html:Book-of-Thomas-the-Contender"
        "gospegyp.html:Gospel-of-the-Egyptians"
        "eugnostos.html:Eugnostos-the-Blessed"
        "sjc.html:Sophia-of-Jesus-Christ"
        "dialsav.html:Dialogue-of-the-Savior"
        "apocpaul.html:Apocalypse-of-Paul"
        "1apocjas.html:First-Apocalypse-of-James"
        "2apocjas.html:Second-Apocalypse-of-James"
        "apocadam.html:Apocalypse-of-Adam"
        "actpet12.html:Acts-of-Peter-and-the-Twelve-Apostles"
        "thunder.html:Thunder-Perfect-Mind"
        "authoritative.html:Authoritative-Teaching"
        "greatsec.html:Great-Power"
        "trimorph.html:Trimorphic-Protennoia"
        "marsanes.html:Marsanes"
        "interp.html:Interpretation-of-Knowledge"
        "valex.html:Valentinian-Exposition"
        "allogenes.html:Allogenes"
        "hypsiph.html:Hypsiphrone"
        "sent-sextus.html:Sentences-of-Sextus"
        "trimegistus.html:Discourse-on-the-Eighth-and-Ninth"
        "asclepius.html:Asclepius"
        "gosmary.html:Gospel-of-Mary"
        "acts-peter.html:Acts-of-Peter"
        "paraphshem.html:Paraphrase-of-Shem"
        "secondsethtreat.html:Second-Treatise-of-the-Great-Seth"
        "apocpetcoptic.html:Apocalypse-of-Peter"
        "teachsilv.html:Teachings-of-Silvanus"
        "sethseth.html:Three-Steles-of-Seth"
        "zostr.html:Zostrianos"
        "epistpeterphilip.html:Letter-of-Peter-to-Philip"
        "melchiz.html:Melchizedek"
        "thoughtnorea.html:Thought-of-Norea"
        "testtruth.html:Testimony-of-Truth"
    )

    > "$NAGHAMMADI_OUT.part"
    {
        echo "=================================================================="
        echo "THE NAG HAMMADI LIBRARY"
        echo "Robinson translations, from gnosis.org / Coptic Gnostic Library"
        echo "Compiled $(date -u +%Y-%m-%d)"
        echo "=================================================================="
        echo
    } >> "$NAGHAMMADI_OUT.part"

    for entry in "${NAGHAMMADI_TEXTS[@]}"; do
        slug="${entry%%:*}"
        title="${entry##*:}"
        url="https://gnosis.org/naghamm/${slug}"

        # Try to fetch; some 404 because gnosis.org filenames vary
        if html=$(curl -L -A "$UA" --fail --silent "$url" 2>/dev/null); then
            {
                echo
                echo "=================================================================="
                echo "TRACTATE: $title"
                echo "Source: $url"
                echo "=================================================================="
                echo
                # Strip HTML — naive but works for gnosis.org's clean markup
                echo "$html" | sed -e 's/<[^>]*>//g' -e 's/&amp;/\&/g' -e 's/&lt;/</g' -e 's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#39;/'\''/g' -e 's/&nbsp;/ /g' | awk 'NF { print }'
                echo
            } >> "$NAGHAMMADI_OUT.part"
            echo "  ✓ $title"
        else
            echo "  · skip $title (not at expected URL)"
        fi
        sleep 0.3  # be polite to gnosis.org
    done

    mv "$NAGHAMMADI_OUT.part" "$NAGHAMMADI_OUT"
    size=$(du -h "$NAGHAMMADI_OUT" | cut -f1)
    echo "✓ saved  $NAGHAMMADI_OUT  ($size)"
    echo
fi

# --- The Magic Bag (Mark Probert, 1963) ---
# Requested by u/throwrathewholething. Channeled text from the Inner Circle.
# Not in clean public domain — but widely circulated as orphaned work.
# Skipping automated fetch — Mark can grab manually if he wants:
#   https://pdfroom.com/books/the-magic-bag-a-manuscript-dictated-clairaudiently-to-mark-probert-by-members-of-the-inner-circle/p0q2JoL5xEw
echo "ℹ  The Magic Bag (Probert 1963): copyright unclear, grab manually if wanted:"
echo "   https://pdfroom.com/books/the-magic-bag-a-manuscript-dictated-clairaudiently-to-mark-probert-by-members-of-the-inner-circle/p0q2JoL5xEw"
echo

echo "============================================================"
echo "Done. Contents of $INCOMING:"
echo "============================================================"
ls -lh "$INCOMING" | grep -E '\.(pdf|txt)$' || true
echo
echo "Next steps:"
echo "  1. Re-ingest sefer_yetzirah_westcott.pdf — overwrites the broken record"
echo "     (the current sefer_yetzirah in mb_sources has 1 chunk, 0 claims)"
echo "  2. Ingest the rest through your usual meta-bridge pipeline"
echo "  3. Run vectoreologist to see if Tao Te Ching shifts the Taoism cluster"
echo "     and if Perennial Philosophy reinforces the cross-tradition attractor"
