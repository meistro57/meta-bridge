#!/usr/bin/env python3
# filename: codex_test.py
#
# Meta Bridge — Codex Test Runner
# Runs a 3-page vision + embed test on a codex PDF.
# Dumps full results to JSON so you can inspect what you actually got.
#
# Usage:
#   python codex_test.py <path-to-pdf> [num_pages]
#
# Env (reads from .env if present):
#   OPENROUTER_API_KEY   required
#   CODEX_EMBED_MODEL    default: google/gemini-embedding-2
#   CODEX_EMBED_DIMS     default: 3072
#   CODEX_VISION_MODEL   default: google/gemini-2.5-flash

import os, sys, json, base64, time, subprocess
from pathlib import Path

# ── load .env ────────────────────────────────────────────────────────────────
def load_dotenv(path=".env"):
    try:
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass

load_dotenv(".env")
load_dotenv(os.path.expanduser("~/meta-bridge/.env"))

# ── config ───────────────────────────────────────────────────────────────────
API_KEY      = os.environ.get("OPENROUTER_API_KEY", "")
EMBED_MODEL  = os.environ.get("CODEX_EMBED_MODEL",  "google/gemini-embedding-2")
EMBED_DIMS   = int(os.environ.get("CODEX_EMBED_DIMS", "3072"))
VISION_MODEL = os.environ.get("CODEX_VISION_MODEL", "google/gemini-2.5-flash")
OR_BASE      = "https://openrouter.ai/api/v1"
EMBED_RETRIES = 3
EMBED_RETRY_DELAY = 4  # seconds between retries

if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

# ── HTTP helper ───────────────────────────────────────────────────────────────
import urllib.request, urllib.error

def or_post(endpoint, payload, timeout=90):
    url = OR_BASE + endpoint
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {API_KEY}")
    req.add_header("HTTP-Referer", "https://github.com/meistro57/meta-bridge")
    req.add_header("X-Title", "mb-codex-test")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {body}")

# ── PDF → images ──────────────────────────────────────────────────────────────
def pdf_to_images(pdf_path, pages, dpi=200):
    results = []
    for p in pages:
        out_prefix = f"/tmp/codex_test_p{p:03d}"
        cmd = ["pdftoppm", "-r", str(dpi), "-jpeg", "-f", str(p), "-l", str(p), pdf_path, out_prefix]
        subprocess.run(cmd, check=True, capture_output=True)
        candidates = list(Path("/tmp").glob(f"codex_test_p{p:03d}*.jpg"))
        if not candidates:
            print(f"  [!] no image output for page {p}")
            continue
        img_path = candidates[0]
        img_bytes = img_path.read_bytes()
        img_path.unlink()
        results.append((p, img_bytes))
        print(f"  rasterized page {p}: {len(img_bytes)/1024:.1f} KB")
    return results

# ── Vision ────────────────────────────────────────────────────────────────────
VISION_SYSTEM = """You are an expert in Mesoamerican manuscripts, pre-Columbian iconography, and ritual calendar systems.

You are analyzing a page from an Aztec or Mixtec codex. These manuscripts encode ritual, cosmological, and calendrical knowledge through pictographic imagery.

Please provide TWO sections in your response, using EXACTLY these headers:

## SCHOLARLY DESCRIPTION
Describe what you observe with precision:
- Overall page structure and layout (grid? narrative? mixed?)
- Number and arrangement of visual cells or registers if present
- Deities or figures identified (use their Nahuatl names if recognizable: Quetzalcoatl, Tlaloc, Tezcatlipoca, Xipe Totec, etc.)
- Calendar glyphs, day signs, or numerical dots/bars visible
- Colors, cardinal directions, ritual objects (mirrors, serpents, flints, etc.)
- Any recognizable ritual scenes (sacrifice, creation, underworld journey, etc.)

## NARRATIVE PERFORMANCE
Now narrate this page as an Aztec priest-scholar would perform it aloud:
Speak as if reading the manuscript to initiates. Connect the figures, movements, and symbols into a living story or ritual instruction. Use present tense. Be specific and rich — this description will be used for semantic search and cross-tradition synthesis with texts from Egyptian, Hermetic, Gnostic, Vedic, and channeled traditions."""

def describe_page(img_bytes, codex_name, page_num):
    b64 = base64.b64encode(img_bytes).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": f"This is page {page_num} of the {codex_name}. Please describe it."}
            ]}
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
    }
    resp = or_post("/chat/completions", payload, timeout=90)
    return resp["choices"][0]["message"]["content"].strip()

def split_description(description):
    """Split the vision output into scholarly and narrative sections.
    Returns (scholarly, narrative) — falls back gracefully if headers missing."""
    scholarly = ""
    narrative = ""

    # Try splitting on the two expected headers
    lower = description.lower()
    scholar_idx  = lower.find("## scholarly description")
    narrative_idx = lower.find("## narrative performance")

    if scholar_idx != -1 and narrative_idx != -1:
        # Extract scholarly section (between the two headers)
        scholarly = description[scholar_idx + len("## scholarly description"):narrative_idx].strip()
        # Extract narrative section (after second header)
        narrative = description[narrative_idx + len("## narrative performance"):].strip()
    elif narrative_idx != -1:
        # Only narrative header found
        scholarly = description[:narrative_idx].strip()
        narrative = description[narrative_idx + len("## narrative performance"):].strip()
    elif scholar_idx != -1:
        # Only scholarly header found
        scholarly = description[scholar_idx + len("## scholarly description"):].strip()
    else:
        # No headers — treat whole thing as scholarly
        scholarly = description

    return scholarly.strip(), narrative.strip()

# ── Embed with retry ──────────────────────────────────────────────────────────
def _embed_request(payload):
    """Make an embedding request, with full response logging on unexpected shape."""
    for attempt in range(1, EMBED_RETRIES + 1):
        try:
            resp = or_post("/embeddings", payload, timeout=60)
            # Validate response shape before indexing
            if "data" not in resp:
                raise ValueError(f"Response missing 'data' key. Full response: {json.dumps(resp)[:500]}")
            if not resp["data"] or "embedding" not in resp["data"][0]:
                raise ValueError(f"Response 'data' malformed. Full response: {json.dumps(resp)[:500]}")
            return resp["data"][0]["embedding"]
        except Exception as e:
            if attempt < EMBED_RETRIES:
                print(f"    [retry {attempt}/{EMBED_RETRIES}] embed error: {e}")
                time.sleep(EMBED_RETRY_DELAY)
            else:
                raise

def embed_text(text):
    return _embed_request({
        "model": EMBED_MODEL,
        "input": text,
        "dimensions": EMBED_DIMS,
    })

def embed_image(img_bytes):
    # Pass base64 data URI as a plain string — OpenRouter embeddings endpoint
    # does NOT accept the image_url object format (that's chat completions only).
    b64 = base64.b64encode(img_bytes).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"
    return _embed_request({
        "model": EMBED_MODEL,
        "input": data_uri,
        "dimensions": EMBED_DIMS,
    })

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na  = sum(x*x for x in a)**0.5
    nb  = sum(x*x for x in b)**0.5
    return dot / (na * nb) if na and nb else 0.0

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pdf_path  = sys.argv[1] if len(sys.argv) > 1 else None
    num_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    if not pdf_path:
        print("Usage: python codex_test.py <pdf_path> [num_pages]")
        sys.exit(1)

    pdf_path   = os.path.abspath(pdf_path)
    codex_name = Path(pdf_path).stem.replace("_", " ").replace("-", " ")

    print(f"\n{'='*60}")
    print(f"  MB-CODEX TEST RUN")
    print(f"  codex:        {codex_name}")
    print(f"  pages:        {num_pages}")
    print(f"  vision model: {VISION_MODEL}")
    print(f"  embed model:  {EMBED_MODEL}")
    print(f"  embed dims:   {EMBED_DIMS}")
    print(f"{'='*60}\n")

    pages_to_test = list(range(2, 2 + num_pages))

    print(f"[1/3] Rasterizing pages {pages_to_test}...")
    images = pdf_to_images(pdf_path, pages_to_test, dpi=200)
    print(f"  got {len(images)} images\n")

    results         = []
    all_text_vecs   = []   # (page, full_text_vec)
    all_scholar_vecs = []  # (page, scholarly_vec)
    all_narr_vecs   = []   # (page, narrative_vec)
    all_image_vecs  = []   # (page, image_vec)

    for page_num, img_bytes in images:
        print(f"[PAGE {page_num}] {'─'*50}")

        # ── Vision
        print(f"  → vision ({VISION_MODEL})...")
        t0 = time.time()
        description = ""
        scholarly   = ""
        narrative   = ""
        vision_time = 0
        try:
            description = describe_page(img_bytes, codex_name, page_num)
            vision_time = time.time() - t0
            scholarly, narrative = split_description(description)
            print(f"  ✓ {len(description)} chars in {vision_time:.1f}s")
            print(f"    scholarly: {len(scholarly)} chars | narrative: {len(narrative)} chars")
            print(f"\n  --- SCHOLARLY PREVIEW ---")
            print("  " + scholarly[:300].replace("\n", "\n  "))
            print(f"\n  --- NARRATIVE PREVIEW ---")
            print("  " + narrative[:300].replace("\n", "\n  "))
            print()
        except Exception as e:
            print(f"  ✗ vision failed: {e}")
            vision_time = time.time() - t0

        # ── Embed: full description
        full_text  = f"{codex_name}\nPage {page_num}\n\n{description}"
        print(f"  → embedding full description ({EMBED_MODEL} {EMBED_DIMS}d)...")
        t0 = time.time()
        text_vec = []
        text_embed_time = 0
        try:
            text_vec = embed_text(full_text)
            text_embed_time = time.time() - t0
            print(f"  ✓ {len(text_vec)}d in {text_embed_time:.1f}s  sample: [{', '.join(f'{v:.4f}' for v in text_vec[:4])}...]")
            all_text_vecs.append((page_num, text_vec))
        except Exception as e:
            print(f"  ✗ full text embed failed: {e}")
            text_embed_time = time.time() - t0

        # ── Embed: scholarly only
        scholar_vec = []
        scholar_embed_time = 0
        if scholarly:
            print(f"  → embedding scholarly section ({EMBED_DIMS}d)...")
            t0 = time.time()
            try:
                scholar_vec = embed_text(f"{codex_name} page {page_num} — scholarly:\n{scholarly}")
                scholar_embed_time = time.time() - t0
                print(f"  ✓ {len(scholar_vec)}d in {scholar_embed_time:.1f}s")
                all_scholar_vecs.append((page_num, scholar_vec))
            except Exception as e:
                print(f"  ✗ scholarly embed failed: {e}")
                scholar_embed_time = time.time() - t0

        # ── Embed: narrative only
        narr_vec = []
        narr_embed_time = 0
        if narrative:
            print(f"  → embedding narrative section ({EMBED_DIMS}d)...")
            t0 = time.time()
            try:
                narr_vec = embed_text(f"{codex_name} page {page_num} — narrative:\n{narrative}")
                narr_embed_time = time.time() - t0
                print(f"  ✓ {len(narr_vec)}d in {narr_embed_time:.1f}s")
                all_narr_vecs.append((page_num, narr_vec))
            except Exception as e:
                print(f"  ✗ narrative embed failed: {e}")
                narr_embed_time = time.time() - t0

        # ── Embed: raw image
        image_vec = []
        image_embed_time = 0
        print(f"  → embedding raw image ({EMBED_DIMS}d)...")
        t0 = time.time()
        try:
            image_vec = embed_image(img_bytes)
            image_embed_time = time.time() - t0
            print(f"  ✓ {len(image_vec)}d in {image_embed_time:.1f}s  sample: [{', '.join(f'{v:.4f}' for v in image_vec[:4])}...]")
            all_image_vecs.append((page_num, image_vec))
        except Exception as e:
            print(f"  ✗ image embed failed: {e}")
            image_embed_time = time.time() - t0

        # ── Cross-modal sim for this page
        if text_vec and image_vec:
            print(f"\n  ★ full-text↔image:    {cosine_sim(text_vec, image_vec):.4f}")
        if scholar_vec and image_vec:
            print(f"  ★ scholarly↔image:    {cosine_sim(scholar_vec, image_vec):.4f}")
        if narr_vec and image_vec:
            print(f"  ★ narrative↔image:    {cosine_sim(narr_vec, image_vec):.4f}")
        if scholar_vec and narr_vec:
            print(f"  ★ scholarly↔narrative:{cosine_sim(scholar_vec, narr_vec):.4f}")

        print()
        results.append({
            "codex":               codex_name,
            "page":                page_num,
            "image_size_bytes":    len(img_bytes),
            "description_full":    description,
            "description_scholarly": scholarly,
            "description_narrative": narrative,
            "vision_model":        VISION_MODEL,
            "vision_time_s":       round(vision_time, 2),
            "embed_model":         EMBED_MODEL,
            "embed_dims":          EMBED_DIMS,
            "text_vector_dims":    len(text_vec),
            "scholar_vector_dims": len(scholar_vec),
            "narr_vector_dims":    len(narr_vec),
            "image_vector_dims":   len(image_vec),
            "sim_text_image":      round(cosine_sim(text_vec, image_vec), 4)    if text_vec and image_vec    else None,
            "sim_scholar_image":   round(cosine_sim(scholar_vec, image_vec), 4) if scholar_vec and image_vec else None,
            "sim_narr_image":      round(cosine_sim(narr_vec, image_vec), 4)    if narr_vec and image_vec    else None,
            "sim_scholar_narr":    round(cosine_sim(scholar_vec, narr_vec), 4)  if scholar_vec and narr_vec  else None,
            "text_vector_sample":    text_vec[:20]    if text_vec    else [],
            "scholar_vector_sample": scholar_vec[:20] if scholar_vec else [],
            "narr_vector_sample":    narr_vec[:20]    if narr_vec    else [],
            "image_vector_sample":   image_vec[:20]   if image_vec   else [],
            "text_embed_time_s":     round(text_embed_time, 2),
            "image_embed_time_s":    round(image_embed_time, 2),
        })

    # ── Cross-page similarity matrices
    def print_matrix(label, vecs):
        if len(vecs) < 2:
            return
        print(f"\n[{label}]")
        for i, (pi, vi) in enumerate(vecs):
            for j, (pj, vj) in enumerate(vecs):
                if j <= i: continue
                print(f"  page {pi} ↔ page {pj}: {cosine_sim(vi, vj):.4f}")

    print_matrix("CROSS-PAGE: full text", all_text_vecs)
    print_matrix("CROSS-PAGE: scholarly", all_scholar_vecs)
    print_matrix("CROSS-PAGE: narrative", all_narr_vecs)
    print_matrix("CROSS-PAGE: image",     all_image_vecs)

    # ── Text vs image per page
    if all_text_vecs and all_image_vecs:
        print(f"\n[SAME-PAGE TEXT↔IMAGE]")
        tv_map = dict(all_text_vecs)
        iv_map = dict(all_image_vecs)
        sv_map = dict(all_scholar_vecs)
        nv_map = dict(all_narr_vecs)
        for p in sorted(set(tv_map) & set(iv_map)):
            print(f"  page {p}  full:{cosine_sim(tv_map[p], iv_map[p]):.4f}", end="")
            if p in sv_map: print(f"  scholar:{cosine_sim(sv_map[p], iv_map[p]):.4f}", end="")
            if p in nv_map: print(f"  narr:{cosine_sim(nv_map[p], iv_map[p]):.4f}", end="")
            print()

    # ── Save
    os.makedirs("./output", exist_ok=True)
    out_name = Path(pdf_path).stem.lower().replace(" ", "_")
    out_path = f"./output/codex_test_{out_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  DONE — {len(results)} pages")
    print(f"  Results: {out_path}")
    print(f"{'='*60}\n")

    print("\n\n=== FULL DESCRIPTIONS ===\n")
    for r in results:
        print(f"── Page {r['page']} {'─'*48}")
        print(f"[SCHOLARLY]\n{r['description_scholarly']}\n")
        print(f"[NARRATIVE]\n{r['description_narrative']}\n")

if __name__ == "__main__":
    main()
