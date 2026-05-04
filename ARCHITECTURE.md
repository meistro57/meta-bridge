# meta-bridge

**A synthesis engine for consciousness literature.**

Meta Bridge extracts atomic claims from curated esoteric, channeled, and experiential sources, measures cross-tradition convergence, and builds typed bridges between esoteric and mainstream conceptual spaces. It exposes where independent experiential sources agree, where they translate into scientific vocabulary, and where genuinely orphaned claims sit.

It is the synthesis counterpart to [KAE](../kae), which does archaeology on broad unknown corpora. Where KAE asks *"what unexpected thing is buried here?"*, Meta Bridge asks *"where do these curated sources converge, diverge, or connect to outside frames?"*

---

## Mission

Build a structured, queryable, cross-referenced map of post-1970s channeled and experiential metaphysical literature, with formal bridges to mainstream scientific and philosophical discourse, such that:

1. Multi-source convergence is measurable, not anecdotal.
2. Idiosyncratic single-source claims are visible as such.
3. Cross-tradition translation (esoteric ↔ scientific vocabulary) is explicit and inspectable.
4. The cartographer (you) can correct, merge, split, and annotate the map by hand.

The output is not a search engine. It is a **bridge catalog**.

---

## Core Objects

Three first-class object types. Everything else is infrastructure.

### Source

A single work ingested into the system. Books, transcripts, channeling corpora, session archives.

```
Source {
  id                string       // stable, human-readable: "seth_speaks", "cannon_cu_v2"
  title             string
  author            string       // the human author/scribe
  medium            string       // the claimed non-human source, if any: "Seth", "Ra", "Essassani"
  channel_type      enum         // channeled | regression | dictated | authored | dramatized | dialogue
  tradition         string       // "Seth", "Ra/Law of One", "QHHT/Cannon", "Essassani/Bashar", ...
  year              int
  region            string       // geographic origin of the material, for independence scoring
  ingested_at       timestamp
  chunk_count       int
  claim_count       int
  metadata          map          // free-form
}
```

The `channel_type` and `tradition` and `region` fields feed the independence score. Two sources in the same tradition do not count as independent corroboration; two sources in different traditions, different regions, different decades, different channel types do.

### Claim

An atomic proposition extracted from one or more sources. The dedup layer collapses phrasings to a single claim node.

```
Claim {
  id                   string       // stable: "cl_0001_simultaneous_time"
  canonical_statement  string       // LLM-generated abstract phrasing, editable by hand
  semantic_embedding   vector
  attributions         []Attribution
  tradition_coverage   []string     // derived: traditions this claim appears in
  independence_score   float        // derived: how independent are the supporting sources?
  temporal_span        [int, int]   // [earliest_year, latest_year] across sources
  tags                 []string     // "cosmology", "identity", "time", "substrate", ...
  bridge_ids           []string     // bridges this claim participates in
  editorial_status     enum         // auto | reviewed | human_edited | merged_into:<id>
  notes                string       // human annotations
}

Attribution {
  source_id            string
  chunk_ids            []string     // pointers into the chunk collection
  surface_quote        string       // short (<15 words) or paraphrase
  claim_type           enum         // stated | dramatized | implied
  confidence           float        // extractor confidence
}
```

Key design decisions:
- **Stable IDs**. A claim keeps its ID across ingestion waves. New sources append to `attributions`, never duplicate the claim.
- **Canonical statement is editable**. The LLM's first-pass phrasing is a starting point, not ground truth. The cartographer's desk exposes this for correction.
- **Attributions carry claim_type**. "Stated" (Seth doctrinally asserts X) is different from "dramatized" (Roberts shows X through Ma-ah's experience) is different from "implied" (a claim logically entailed but not spoken). Bridge strength depends on this.

### Bridge

A typed, scored, annotated link between two claims (or between a claim and an external node — e.g., a KAE concept node).

```
Bridge {
  id            string
  type          enum          // see taxonomy below
  endpoints     [ClaimRef, ClaimRef]
  strength      float         // 0..1, type-specific metric
  independence  float         // only meaningful for convergence-type bridges
  evidence      []EvidenceItem
  auto_generated bool
  editorial_status enum       // auto | confirmed | rejected | edited
  notes         string
  created_at    timestamp
}

ClaimRef {
  claim_id     string
  corpus       enum           // "meta-bridge" | "kae" | "external"
}
```

Bridges are the product. Everything else is scaffolding for the bridges.

---

## Reality Filter

After bridge detection, every bridge passes through the Reality Filter — an LLM-scored epistemic layer that prevents "vibes convergence" from polluting the catalog.

Each bridge receives:

- `bridge_type` — `Structural` (formal isomorphism between systems), `Analogical` (useful metaphor, not claimed identity), `Speculative` (plausible but ungrounded), `Testable` (yields specific empirical predictions), or `Contradictory` (claims are mutually exclusive).
- `constraint_flags` — list of physics or logic concerns: `no-communication-theorem`, `decoherence-scale-gap`, `measurement-problem-conflation`, `energy-conservation-violation`, `unfalsifiable`, `equivocation`, `category-error`.
- `testability_score` — 0–5 integer (0 = not testable, 5 = clear empirical prediction with established methodology).
- `epistemic_status` — plain-language label for the catalog: `Established mapping`, `Plausible analogy`, `Suggestive but ungrounded`, `Poetic metaphor`, `Needs disambiguation`, `Likely invalid`, or `Contradicted by evidence`.

The Reality Filter makes it possible to navigate the catalog honestly: Structural and Testable bridges are the high-signal output; Speculative and Poetic bridges are still shown but clearly labeled.

---

## Bridge Taxonomy

Six bridge types, each with its own strength metric and its own detector.

### 1. Convergence

Independent sources making the same claim.

- **Detector:** claim dedup layer collapses sufficiently-similar phrasings; any claim with attributions from ≥2 independent sources produces a convergence bridge between those attributions.
- **Strength:** function of semantic similarity × independence score × attribution confidence.
- **Why it matters:** the core convergence signal. Seth + Cannon + Ra + Bashar agreeing on a claim is not proof, but it is a structural fact about the corpus worth surfacing.

### 2. Dramatization

A source shows narratively what another source states doctrinally.

- **Detector:** cross-reference between claims with `claim_type = stated` and claims with `claim_type = dramatized` that share high semantic similarity. Oversoul Seven's rock-building-by-sound scene ↔ Seth's "consciousness units" doctrine.
- **Strength:** semantic similarity × narrative specificity.
- **Why it matters:** distinguishes doctrine from illustration, which is useful for teaching and for testing whether a dramatization faithfully represents its source doctrine.

### 3. Translation

Same claim in different vocabularies — typically esoteric ↔ scientific.

- **Detector:** cross-corpus query from a Meta Bridge claim against KAE's `kae_nodes`, thresholded for close semantic neighbors with *divergent* surface vocabulary. "Akashic records" ↔ "holographic principle" should fire; "soul" ↔ "soul" should not.
- **Strength:** semantic similarity × vocabulary divergence (how different the surface words are for how close the meaning is).
- **Why it matters:** the bridge to consensus knowledge. Tells you where an esoteric claim *would land* in scientific vocabulary.

### 4. Evolution

A claim appearing in refined or extended form across time within a tradition.

- **Detector:** within-tradition claim chains with monotonic year progression and semantic progression.
- **Strength:** temporal span × semantic delta.
- **Why it matters:** channeling/esoteric traditions evolve. Seth in 1972 vs. Roberts' late material. Cannon's early regressions vs. *Convoluted Universe* v5. Makes tradition-internal development visible.

### 5. Contradiction

Sources explicitly disagreeing on a claim.

- **Detector:** claim pairs with high semantic similarity on the *topic* and explicit negation or incompatibility on the *assertion*. Hardest detector to build well — requires LLM judgment, not pure embedding.
- **Strength:** certainty of contradiction (low false-positive rate matters more than recall here).
- **Why it matters:** the esoteric corpus is not monolithic. Seth and the Cassiopaeans disagree about plenty. Making those disagreements legible is more valuable than papering them over.

### 6. Dependency

One claim logically requires or presupposes another.

- **Detector:** LLM-judged entailment pass over claim pairs within a tradition.
- **Strength:** entailment confidence.
- **Why it matters:** builds the logical skeleton of a tradition's worldview. "Reincarnation requires persistent identity across bodies" — if a tradition asserts reincarnation, it's implicitly asserting the prerequisite.

---

## Ingest Pipeline

Six stages. Human-in-the-loop checkpoints at stages 3 and 6.

```
  raw text
     │
     ▼
 ┌─────────────────────────────────────┐
 │ STAGE 1: STRUCTURAL CHUNKING        │
 │   - narrative PDFs: parse by        │
 │     chapter / session / scene       │
 │   - academic/textbook PDFs:         │
 │     SplitAcademic() detects         │
 │     numbered section headers        │
 │     (e.g., "1.2.3 Title"), filters  │
 │     TOC lines and running headers,  │
 │     groups equations with prose     │
 │   - classify chunks by type:        │
 │     framing | dialogue | doctrine   │
 │     | narrative | biographical      │
 │   - 500–800 token target, snap to   │
 │     paragraph boundaries            │
 └────────────────┬────────────────────┘
                  ▼
 ┌─────────────────────────────────────┐
 │ STAGE 2: CLAIM EXTRACTION           │
 │   - LLM pass over doctrine/         │
 │     narrative chunks only           │
 │   - emit (canonical_statement,      │
 │     surface_quote, claim_type,      │
 │     confidence) tuples              │
 │   - null for chunks with no         │
 │     metaphysical content            │
 └────────────────┬────────────────────┘
                  ▼
 ┌─────────────────────────────────────┐
 │ STAGE 3: CLAIM DEDUP & MERGE ⚑      │
 │   - embed canonical statements      │
 │   - cluster via similarity + LLM    │
 │     judgment                        │
 │   - merge into existing claims or   │
 │     create new ones                 │
 │   - HUMAN CHECKPOINT: review        │
 │     low-confidence merges           │
 └────────────────┬────────────────────┘
                  ▼
 ┌─────────────────────────────────────┐
 │ STAGE 4: BRIDGE DETECTION           │
 │   - run each bridge detector over   │
 │     updated claim graph             │
 │   - write new bridges; update       │
 │     existing bridges' evidence      │
 └────────────────┬────────────────────┘
                  ▼
 ┌─────────────────────────────────────┐
 │ STAGE 5: REALITY FILTER             │
 │   - epistemic scoring of each       │
 │     bridge via LLM rubric           │
 │   - assigns bridge_type:            │
 │     Structural | Analogical |       │
 │     Speculative | Testable |        │
 │     Contradictory                   │
 │   - flags physics/logic constraint  │
 │     violations (no-communication-   │
 │     theorem, decoherence-scale-gap, │
 │     energy-conservation-violation,  │
 │     equivocation, category-error,   │
 │     unfalsifiable, …)               │
 │   - assigns testability_score 0–5   │
 │   - assigns plain-language          │
 │     epistemic_status                │
 └────────────────┬────────────────────┘
                  ▼
 ┌─────────────────────────────────────┐
 │ STAGE 6: CARTOGRAPHER REVIEW ⚑      │
 │   - surface new/changed bridges     │
 │     with epistemic scores attached  │
 │   - human confirms, rejects,        │
 │     edits, or annotates             │
 │   - confirmed state persists        │
 └─────────────────────────────────────┘
```

Stage 3 is the hardest. Get dedup wrong and the whole system degrades. Over-merging collapses distinct claims ("the soul is eternal" and "consciousness survives bodily death" are related but not identical); under-merging produces 400 variants of the same claim and kills convergence signal. This is where the cartographer's judgment is most valuable, and where the UI needs to be genuinely good.

---

## Query Primitives

Five queries the system must support well. Everything else can be derived.

### 1. `convergence_query(topic, min_independence)`

Return claims on the topic with independence score ≥ threshold, ranked by multi-source support. *"What do these traditions agree on about time?"*

### 2. `divergence_query(topic)`

Return contradiction bridges on the topic, with endpoint claims and source attributions. *"Where do these traditions disagree about the structure of the soul?"*

### 3. `evolution_query(tradition, topic)`

Return evolution bridges for claims on the topic within a tradition, ordered by year. *"How did Seth's material on identity develop between 1970 and 1984?"*

### 4. `translation_query(claim_id | external_concept)`

Given a Meta Bridge claim, return translation bridges to KAE/external nodes. Given a scientific concept, return translation bridges into Meta Bridge. Bidirectional.

### 5. `orphan_query(min_specificity)`

Return claims with exactly one source attribution, filtered to those with sufficient specificity to be meaningful (not trivial statements, not near-duplicates of well-attributed claims). *"What does only Cannon say? What does only the Cassiopaeans say?"* Orphans are not dismissed — they are flagged for attention.

---

## Interface to KAE

Meta Bridge and KAE are peer tools, not layers. Each can query the other.

**Meta Bridge → KAE:**
For each high-confidence Meta Bridge claim, run a vector query against `kae_nodes` and `kae_chunks`. Close matches with divergent vocabulary become candidate Translation bridges. This is the primary KAE → Meta Bridge data flow.

**KAE → Meta Bridge:**
When KAE surfaces a high-anomaly node in its meta-graph, Meta Bridge can be queried for that node's semantic neighborhood. If the esoteric corpus has strong convergence on a concept near a KAE anomaly, that is a second-order signal — "mainstream knowledge has an orphan here that the esoteric literature has heavily populated." Interesting either way you read it.

Both directions run as scheduled cross-corpus passes, not inline queries. Results become bridges in Meta Bridge's catalog, tagged with `corpus = kae` on one endpoint.

---

## Cartographer's Desk (UI)

The UI is not a dashboard. It is a workbench.

Primary views:

- **Bridge feed** — new/changed bridges since last review, each inspectable, with confirm/reject/edit affordances.
- **Claim page** — canonical statement, all attributions with source context, all bridges this claim participates in, edit affordances (rewrite statement, merge with another claim, split into multiple claims, retag).
- **Source page** — all claims extracted from a source, chunk-level browsing, ability to re-run extraction on specific chunks after prompt tuning.
- **Tradition view** — all claims in a tradition, evolution bridges over time, convergence with other traditions.
- **Orphan list** — single-source claims sorted by specificity/interest score, a standing triage queue.

Built on the KAE Lens SSE scaffolding. Live-updating as pipeline stages complete. Keyboard-driven where possible — this is a tool you will use for hours at a stretch, not click through occasionally.

---

## Technology

Deliberately matches the KAE stack so infrastructure is shared and familiar:

- **Language:** Go (primary), with Python for specific LLM-heavy passes if library support matters.
- **Vector store:** Qdrant. Collections: `mb_sources`, `mb_chunks`, `mb_claims`, `mb_bridges`. Test collections use `_test` suffix; the reflection loop uses `meta_reflection_loop_test`.
- **LLM:** OpenRouter. DeepSeek R1 for reasoning passes (dedup judgment, entailment, contradiction detection). Gemini Flash for bulk extraction and classification. Gemma 4 for reflection and reality filter scoring.
- **Local LLM fallback:** Ollama with Qwen2.5-Coder:32b on the BOXX, for passes where API cost is an issue at scale.
- **Reflection loop:** LangGraph (`reflect_loop.py`) drives a stateful multi-iteration reflection pass. Distinct from the one-shot `reflect.py` batch pass — useful for goal-directed hunts (e.g., "find contradictions") without modifying stable collections.
- **Epistemic scoring:** `scoring/reality_filter.py` — standalone tool that classifies bridges from vectoreology JSON reports. Reads the rubric from `scoring/reality_filter_prompt.md`. Controlled by `MB_FILTER_MODEL` (falls back to `MB_MODEL`).
- **MCP integrations:** Qdrant and Redis MCP servers configured in `.crush.json` for interactive use.
- **UI:** Bubbletea/Lipgloss TUI for the headless/batch-mode cartographer's desk; separate SSE web dashboard (pattern from KAE Lens) for bridge feed and claim browsing.
- **Config:** same env-based config pattern as Chat Bridge and KAE.
- **Deployment:** Pop!_OS server, Docker, Qdrant instance potentially shared with KAE or separate (decide at Wave 2 based on load).

---

## Ingestion Waves

Schema is validated incrementally. Each wave is a commit-point for the architecture.

**Wave 1 — schema validation.** *The Education of Oversoul Seven* (Roberts, 1973). Short, well-structured, mix of narrative + doctrine + meta. Tests: scene-level chunking, narrative/doctrine classification, claim extraction from dramatized content, dramatization bridge detection.

**Wave 2 — doctrinal core.** *Seth Speaks* (Roberts, 1972). *The Nature of Personal Reality* (Roberts, 1974). *The Law of One, Book I* (Rueckert/Elkins/McCarty, 1981). Three dense high-signal sources. Tests: dedup across distinct traditions, convergence bridge detection, independence scoring with meaningful tradition diversity.

**Wave 3 — session-structured material.** Cannon's *Convoluted Universe* v1–v5. Tests: session-level chunking, the `regression` channel type, Cannon's narrative framing vs. regression dialogue vs. metaphysical claim classification. This is the hardest wave; it will expose schema problems if they exist.

**Wave 4 — breadth.** Bashar transcripts (Anka). Roberts' remaining Seth material and fiction. The Cassiopaeans. Barbara Marciniak's Pleiadian material. *ROOT ACCESS* (Hubrich/Eli). Other curated sources. Raises statistical power of convergence detection.

**Wave 5 — meta-layer.** Pass over `marks_gpt_history` extracting conversation moments where cross-references were drawn. Seeds the bridge catalog with human-annotated bridges. These become training/validation data for automated bridge detectors and provide ground truth for evaluating detector precision.

After Wave 3 the system should produce meaningful analysis. Waves 4–5 increase resolution and recall.

---

## Non-Goals

Worth stating explicitly, because the temptation to expand scope is real.

- **Not a search engine for esoteric content.** If the user wants to read Seth, they should read Seth. This tool is for the cross-source synthesis layer.
- **Not a truth arbiter.** Convergence across traditions is a structural fact about the corpus, not evidence the underlying claims are metaphysically correct. The tool measures; it does not adjudicate.
- **Not a recommender.** No "you might also like." The catalog is browsed by concept, tradition, and bridge, not by user preference modeling.
- **Not public-facing (initially).** Built as a personal research instrument first. Decisions about what, if anything, to expose publicly come after the schema is stable and the catalog has content worth exposing.

---

## Open Design Questions

These are not resolved and will be decided during Wave 1–2:

1. **Claim granularity.** How atomic is atomic? "The soul is eternal" vs. "Individual consciousness persists beyond bodily death" vs. "The continuity of personhood extends across embodiments." These might be one claim, two claims, or three depending on dedup tuning. Wave 1 will calibrate.
2. **Tradition taxonomy.** How finely to split traditions? Is "Seth" one tradition or does late Roberts material deserve a separate tag? Is Cannon's QHHT one tradition or does each book's thematic focus warrant sub-tagging? Likely resolve with hierarchical tags rather than flat labels.
3. **Independence scoring formula.** Multiplicative over tradition-different × region-different × decade-different × channel-type-different? Or learned from the Wave 5 ground-truth bridges? Defer to after Wave 5 has data.
4. **Bridge persistence across re-extraction.** If Wave 3 ingestion causes Stage 4 to fire a contradiction bridge between a Wave 2 claim and a new Wave 3 claim — and then Wave 4 adds a third source that resolves the contradiction — does the bridge get deleted, archived, or transformed? Probably transformed with full history, but mechanism TBD.
5. **KAE coupling depth.** Meta Bridge can stand alone. How much to invest in the KAE bridge vs. keep it loose? Start loose; tighten when real translation bridges are producing value.
