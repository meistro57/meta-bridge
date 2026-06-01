#!/usr/bin/env bash
# filename: fetch_reddit_requested_sources_v2.sh
# Fixes the broken URLs from v1 and uses verified archive.org / gnosis.org paths.

set -e

INCOMING="${INCOMING:-$HOME/meta-bridge/incoming}"
mkdir -p "$INCOMING"
cd "$INCOMING"

UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

fetch() {
    local name="$1"
    local url="$2"
    local out="$3"

    if [ -f "$out" ] && [ -s "$out" ]; then
        local size
        size=$(du -h "$out" | cut -f1)
        echo "✓ skip   $out ($size, already present)"
        return 0
    fi

    echo "→ fetch  $name"
    if curl -L -A "$UA" --fail --silent --show-error -o "$out.part" "$url"; then
        mv "$out.part" "$out"
        local size
        size=$(du -h "$out" | cut -f1)
        echo "✓ saved  $out  ($size)"
    else
        rm -f "$out.part"
        echo "✗ FAILED $name — manual fetch from $url"
    fi
    echo
}

echo "============================================================"
echo "Fix-up fetcher v2 — verified URLs"
echo "============================================================"
echo

# Tao Te Ching (Legge, via Standard Ebooks scan)
fetch "Tao Te Ching (Legge)" \
    "https://archive.org/download/laozi_tao-te-ching/laozi_tao-te-ching_james-legge.pdf" \
    "tao_te_ching_legge.pdf"

# Book of the Law (Crowley 1904, public domain)
fetch "Book of the Law (Crowley)" \
    "https://archive.org/download/CrowleyTheBookOfTheLaw/Crowley%2C%20Aleister%20-%201904%20-%20The%20Book%20of%20the%20Law.pdf" \
    "crowley_book_of_the_law.pdf"

# Clear out the broken stub from v1 before re-running NHL scrape
rm -f nag_hammadi_library_robinson.txt
echo "→ cleared stale nag_hammadi_library_robinson.txt stub"
echo

# Nag Hammadi Library — verified gnosis.org filenames from the actual index page
NAGHAMMADI_OUT="nag_hammadi_library_gnosis.txt"
echo "→ fetch  Nag Hammadi Library (gnosis.org — verified paths)"

# Format: filename:Title
NAGHAMMADI_TEXTS=(
    "prayp-meyer.html:Prayer of the Apostle Paul"
    "jam-meyer.html:Apocryphon of James"
    "got-barnstone.html:Gospel of Truth (Barnstone)"
    "resurrection-barnstone.html:Treatise on the Resurrection"
    "tripart.htm:Tripartite Tractate"
    "Hypostas-Barnstone.html:Hypostasis of the Archons"
    "origin.html:On the Origin of the World"
    "exegesis-barnstone.html:Exegesis on the Soul"
    "bookt-jdt.html:Book of Thomas the Contender"
    "gosthom-meyer.html:Gospel of Thomas (Meyer)"
    "GPhilip-Meyer.html:Gospel of Philip (Meyer)"
    "GPhilip-Barnstone.html:Gospel of Philip (Barnstone)"
    "eugn.html:Eugnostos the Blessed"
    "sjc.html:Sophia of Jesus Christ"
    "apopet.html:Apocalypse of Peter"
    "letpet-meyer.html:Letter of Peter to Philip"
    "actp.html:Acts of Peter and the Twelve Apostles"
    "1ja.html:First Apocalypse of James"
    "2ja.html:Second Apocalypse of James"
    "ascp.html:Apocalypse of Paul"
    "adam-barnstone.html:Apocalypse of Adam"
    "thunder-barnstone.html:Thunder, Perfect Mind"
    "nore.html:Thought of Norea"
    "silvanus.html:Teachings of Silvanus"
    "testruth.html:Testimony of Truth"
    "2seth-barnstone.html:Second Treatise of the Great Seth"
    "para_shem-barnstone.html:Paraphrase of Shem"
    "steles-meyer.html:Three Steles of Seth"
    "discourse-meyer.html:Discourse on the Eighth and Ninth"
    "prat.html:Prayer of Thanksgiving"
    "valex.html:Valentinian Exposition"
)

> "$NAGHAMMADI_OUT.part"
{
    echo "=================================================================="
    echo "THE NAG HAMMADI LIBRARY"
    echo "Translations from gnosis.org / Coptic Gnostic Library Project"
    echo "Translators: Marvin Meyer, Willis Barnstone, John Turner, Stevan Davies"
    echo "Compiled $(date -u +%Y-%m-%d)"
    echo "=================================================================="
    echo
} >> "$NAGHAMMADI_OUT.part"

got=0
missed=0
for entry in "${NAGHAMMADI_TEXTS[@]}"; do
    slug="${entry%%:*}"
    title="${entry##*:}"
    url="http://www.gnosis.org/naghamm/${slug}"

    if html=$(curl -L -A "$UA" --fail --silent "$url" 2>/dev/null); then
        {
            echo
            echo "=================================================================="
            echo "TRACTATE: $title"
            echo "Source: $url"
            echo "=================================================================="
            echo
            # Strip HTML tags + decode entities
            echo "$html" \
                | sed -e 's/<script[^>]*>.*<\/script>//g' \
                      -e 's/<style[^>]*>.*<\/style>//g' \
                      -e 's/<[^>]*>//g' \
                      -e 's/&amp;/\&/g' \
                      -e 's/&lt;/</g' \
                      -e 's/&gt;/>/g' \
                      -e 's/&quot;/"/g' \
                      -e 's/&#39;/'\''/g' \
                      -e 's/&nbsp;/ /g' \
                      -e 's/&mdash;/—/g' \
                      -e 's/&ndash;/–/g' \
                      -e 's/&hellip;/.../g' \
                | awk 'NF { print }'
            echo
        } >> "$NAGHAMMADI_OUT.part"
        echo "  ✓ $title"
        got=$((got + 1))
    else
        echo "  · skip $title (gnosis.org returned error)"
        missed=$((missed + 1))
    fi
    sleep 0.5  # polite delay
done

mv "$NAGHAMMADI_OUT.part" "$NAGHAMMADI_OUT"
size=$(du -h "$NAGHAMMADI_OUT" | cut -f1)
echo
echo "✓ saved  $NAGHAMMADI_OUT  ($size)  —  $got tractates fetched, $missed missing"
echo

echo "============================================================"
echo "Done."
echo "============================================================"
