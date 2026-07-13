// Command mb is the Meta Bridge CLI.
//
// Usage:
//
//	mb ingest <path-to-pdf-or-txt>
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode"

	"github.com/joho/godotenv"
	"github.com/meistro57/meta-bridge/internal/chunker"
	"github.com/meistro57/meta-bridge/internal/claim"
	"github.com/meistro57/meta-bridge/internal/extractor"
	"github.com/meistro57/meta-bridge/internal/llm"
	"github.com/meistro57/meta-bridge/internal/graphability"
	"github.com/meistro57/meta-bridge/internal/source"
	"github.com/meistro57/meta-bridge/internal/store"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	_ = godotenv.Load()
	switch os.Args[1] {
	case "ingest":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "ingest: missing file path")
			usage()
			os.Exit(1)
		}
		if err := cmdIngest(os.Args[2]); err != nil {
			log.Fatalf("ingest failed: %v", err)
		}
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `meta-bridge (mb) — consciousness literature synthesis engine

Usage:
  mb ingest <path>       Extract claims from a PDF or text file

Env:
  OPENROUTER_API_KEY     required unless MB_MODEL starts with ollama:
  OLLAMA_URL             local Ollama base URL (default: http://localhost:11434)
  QDRANT_URL             Qdrant base URL (default: http://localhost:6333)
  QDRANT_API_KEY         optional Qdrant API key
  MB_MODEL               override extraction model (default: google/gemma-4-31b-it)
  MB_EMBED_PROVIDER      embedding backend: openrouter (default) or ollama
  MB_EMBED_MODEL         embedding model (default: openai/text-embedding-3-small)
  MB_HEADER_PATTERN      optional regex override for section headers
  MB_MAX_CHUNKS          limit chunks processed (useful for dry runs)
  MB_OUTPUT_DIR          output directory (default: ./output)
  MB_SOURCE_ID           override auto-generated source ID
  MB_TITLE               override title (skips catalog + LLM extraction)
  MB_AUTHOR              override author (skips catalog + LLM extraction)
  MB_CATALOG_PATH        path to sources.yaml (default: ./sources.yaml)
  MB_GRAPHABILITY_INDEX  path to graphability index JSON (default: ./graphability_index.json)
  MB_GRAPHABILITY_MIN    minimum score to extract: very_high|high|medium (default: medium)
  MB_SKIP_GRAPHABILITY   set to 1 to disable graphability scoring (scan everything)
  MB_SKIP_OCR            set to 1 to disable the OCR fallback for scanned PDFs
  MB_OCR_LANG            tesseract language(s) for OCR (default: eng)
  MB_OCR_ARGS            extra args appended to ocrmypdf`)
}

func cmdIngest(path string) error {
	extractionModel := envOr("MB_MODEL", extractor.DefaultModel)
	embedProvider := strings.ToLower(envOr("MB_EMBED_PROVIDER", "openrouter"))
	embedModel := envOr("MB_EMBED_MODEL", "openai/text-embedding-3-small")
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	requiresOpenRouter := !strings.HasPrefix(extractionModel, "ollama:") || embedProvider == "openrouter"
	if apiKey == "" && requiresOpenRouter {
		return fmt.Errorf("OPENROUTER_API_KEY not set (check .env or environment)")
	}

	outputDir := envOr("MB_OUTPUT_DIR", "./output")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// --- 0. Load source catalog ---
	cat, err := loadCatalog(catalogPath())
	if err != nil {
		log.Printf("[!] catalog load failed (%v) — falling back to LLM extraction", err)
		cat = &catalog{}
	} else {
		log.Printf("[0] catalog loaded: %d entries", len(cat.Sources))
	}

	// --- 1. Extract text ---
	log.Printf("[1/4] Extracting text from %s", path)
	text, err := extractText(path)
	if err != nil {
		return fmt.Errorf("extract text: %w", err)
	}
	log.Printf("      got %d characters (~%d tokens)", len(text), len(text)/4)

	// --- 2. Initialize LLM client ---
	client := llm.NewClient(apiKey)
	client.SetOllamaURL(envOr("OLLAMA_URL", "http://localhost:11434"))
	client.SetEmbeddingProvider(embedProvider)
	client.SetEmbeddingModel(embedModel)
	ctx := context.Background()

	// --- 3. Resolve source metadata: catalog → LLM → filename fallback ---
	log.Printf("[2/4] Resolving source metadata")
	var meta sourceMeta
	var metaSource string

	if os.Getenv("MB_TITLE") != "" && os.Getenv("MB_AUTHOR") != "" {
		meta.Title = os.Getenv("MB_TITLE")
		meta.Author = os.Getenv("MB_AUTHOR")
		metaSource = "env override"
	} else if entry, ok := cat.lookup(path); ok {
		if entry.Skip {
			return fmt.Errorf("source %q is marked skip=true in catalog: %s", path, entry.Notes)
		}
		meta = entry.toSourceMeta()
		metaSource = "catalog"
		if entry.Notes != "" {
			log.Printf("      note: %s", entry.Notes)
		}
	} else {
		meta = extractSourceMetadata(ctx, client, extractionModel, text)
		metaSource = "llm"
		if meta.Title == "" || strings.EqualFold(meta.Title, "unknown") {
			log.Printf("      ! LLM returned no title — consider adding this source to sources.yaml")
		}
	}

	log.Printf("      source=%s  title=%q  author=%q  year=%d  channel_type=%q  tradition=%q",
		metaSource, meta.Title, meta.Author, meta.Year, meta.ChannelType, meta.Tradition)

	// --- 4. Build Source record ---
	baseName := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	src := source.NewSource(
		envOr("MB_SOURCE_ID", resolveSourceID(meta, cat, path)),
		envOr("MB_TITLE", ifEmpty(meta.Title, baseName)),
		envOr("MB_AUTHOR", ifEmpty(meta.Author, "Unknown")),
	)
	src.SourcePath = path
	src.HeaderPattern = os.Getenv("MB_HEADER_PATTERN")
	if src.Year == 0 && meta.Year > 0 {
		src.Year = meta.Year
	}
	if src.Medium == "" {
		src.Medium = meta.Medium
	}
	if src.ChannelType == "" && meta.ChannelType != "" {
		src.ChannelType = source.ChannelType(meta.ChannelType)
	}
	if src.Tradition == "" {
		src.Tradition = meta.Tradition
	}
	log.Printf("      source.ID=%q  source.Title=%q", src.ID, src.Title)

	// --- 5. Chunk ---
	log.Printf("[3/4] Chunking")
	chunkOpts := chunker.DefaultOptions()
	chunkOpts.HeaderPattern = src.HeaderPattern
	chunkOpts.SourceTitle = src.Title
	chunks := chunker.Split(text, chunkOpts)
	log.Printf("      produced %d chunks", len(chunks))

	if len(chunks) == 0 {
		return fmt.Errorf("chunker produced zero chunks — check text extraction or header pattern")
	}

	for i := range chunks {
		chunks[i].BookTitle = src.Title
		chunks[i].SourceID = src.ID
	}

	maxChunks := envInt("MB_MAX_CHUNKS", 0)
	if maxChunks > 0 && maxChunks < len(chunks) {
		log.Printf("      MB_MAX_CHUNKS=%d; limiting to first %d chunks", maxChunks, maxChunks)
		chunks = chunks[:maxChunks]
	}
	src.ChunkCount = len(chunks)

	// --- 6. Extract claims chunk by chunk ---
	log.Printf("[4/4] Extracting claims and indexing (model=%s, embed_provider=%s, embed_model=%s)",
		extractionModel, embedProvider, embedModel)
	ex := extractor.New(client, extractionModel)
	qdrantClient := store.NewClient(envOr("QDRANT_URL", "http://localhost:6333"), os.Getenv("QDRANT_API_KEY"))
	var allClaims []claim.Claim
	embedMaxChars := envInt("MB_EMBED_MAX_CHARS", 8000)
	collectionsReady := false
	ensureCollections := func(vector []float64) bool {
		if collectionsReady {
			return true
		}
		if err := qdrantClient.EnsureCollections(ctx, len(vector)); err != nil {
			log.Printf("      ! qdrant collection init failed: %v", err)
			return false
		}
		collectionsReady = true
		return true
	}

	sourceVector, err := client.Embed(ctx, sourceEmbeddingText(src.Title, src.Author, text, embedMaxChars))
	if err != nil {
		log.Printf("      ! source embedding failed: %v", err)
	} else if ensureCollections(sourceVector) {
		if err := qdrantClient.UpsertSource(ctx, src, sourceVector); err != nil {
			log.Printf("      ! source upsert failed: %v", err)
		}
	}

	counter := 0
	idFn := func() string {
		counter++
		return fmt.Sprintf("cl_%s_%04d", src.ID, counter)
	}
	loggedFirstClaimPayload := false
	emptyChunks := 0

	// --- Graphability scorer ---
	skipGraphability := os.Getenv("MB_SKIP_GRAPHABILITY") == "1"
	gScorer, gErr := graphability.New("")
	if gErr != nil {
		log.Printf("      [graphability] index load failed (%v) — scoring disabled, all chunks will be extracted", gErr)
		skipGraphability = true
	} else {
		log.Printf("      [graphability] index loaded")
	}
	gMinStr := strings.ToLower(envOr("MB_GRAPHABILITY_MIN", "medium"))
	graphabilitySkipped := 0
	graphabilityGaps := 0

	start := time.Now()
	for i, ch := range chunks {
		// --- Graphability gate ---
		gScore := graphability.Unknown
		gScoreStr := "unknown"
		if !skipGraphability {
			gScore = gScorer.Score(ch.Chapter, ch.Text)
			gScoreStr = gScore.String()

			// Honour MB_GRAPHABILITY_MIN: allow user to raise the floor.
			var minScore graphability.Score
			switch gMinStr {
			case "high":
				minScore = graphability.High
			case "very_high":
				minScore = graphability.VeryHigh
			default: // "medium" and anything else
				minScore = graphability.Medium
			}

			if gScore.IsGap() {
				graphabilityGaps++
				log.Printf("      chunk %d/%d [graphability=GAP] chapter=%q — mandatory scan",
					i+1, len(chunks), ch.Chapter)
			} else if gScore < minScore {
				graphabilitySkipped++
				log.Printf("      chunk %d/%d [graphability=%s] SKIP chapter=%q",
					i+1, len(chunks), gScoreStr, ch.Chapter)
				// Still embed + upsert the chunk so it exists in mb_chunks for
				// search, but tag it skipped so the reflect loop also skips it.
				chunkVector, err := client.Embed(ctx, truncateForEmbedding(ch.Text, embedMaxChars))
				if err != nil {
					log.Printf("      ! chunk embedding error on chunk %d: %v", i, err)
				} else if ensureCollections(chunkVector) {
					if err := qdrantClient.UpsertChunk(ctx, src.ID, ch, chunkVector); err != nil {
						log.Printf("      ! chunk upsert error on chunk %d: %v", i, err)
					}
					// Tag the chunk payload with graphability metadata.
					if err := qdrantClient.SetChunkGraphability(ctx, src.ID, ch.Index, gScoreStr, false, false); err != nil {
						log.Printf("      ! graphability tag error on chunk %d: %v", i, err)
					}
				}
				continue
			} else {
				log.Printf("      chunk %d/%d [graphability=%s] chapter=%q",
					i+1, len(chunks), gScoreStr, ch.Chapter)
			}
		} else {
			log.Printf("      chunk %d/%d (chapter=%q, ~%d tokens)",
				i+1, len(chunks), ch.Chapter, ch.TokenEst)
		}

		chunkVector, err := client.Embed(ctx, truncateForEmbedding(ch.Text, embedMaxChars))
		if err != nil {
			log.Printf("      ! chunk embedding error on chunk %d: %v", i, err)
		} else if ensureCollections(chunkVector) {
			if err := qdrantClient.UpsertChunk(ctx, src.ID, ch, chunkVector); err != nil {
				log.Printf("      ! chunk upsert error on chunk %d: %v", i, err)
			}
			// Tag the chunk with graphability score.
			if !skipGraphability {
				isGap := gScore.IsGap()
				if err := qdrantClient.SetChunkGraphability(ctx, src.ID, ch.Index, gScoreStr, true, isGap); err != nil {
					log.Printf("      ! graphability tag error on chunk %d: %v", i, err)
				}
			}
		}

		claims, err := ex.ExtractChunk(ctx, src.ID, ch, idFn)
		if err != nil {
			log.Printf("      ! extraction error on chunk %d: %v", i, err)
			continue
		}
		if len(claims) == 0 {
			emptyChunks++
		}
		log.Printf("        -> %d claims", len(claims))
		allClaims = append(allClaims, claims...)

		for _, cl := range claims {
			claimVector, err := client.Embed(ctx, cl.CanonicalStatement)
			if err != nil {
				log.Printf("      ! claim embedding error (%s): %v", cl.ID, err)
				continue
			}
			if !ensureCollections(claimVector) {
				continue
			}

			payload := map[string]interface{}{
				"id":                  cl.ID,
				"canonical_statement": cl.CanonicalStatement,
				"tags":                cl.Tags,
				"attributions":        cl.Attributions,
				"editorial_status":    cl.EditorialStatus,
			}
			if cl.Notes != "" {
				payload["notes"] = cl.Notes
			}
			chapterNames := claimChapterNames(cl.Attributions)
			if len(chapterNames) > 0 {
				payload["chapter_names"] = chapterNames
				if len(chapterNames) == 1 {
					payload["chapter"] = chapterNames[0]
				}
			}
			if !loggedFirstClaimPayload {
				if payloadJSON, err := json.Marshal(payload); err == nil {
					log.Printf("      debug first claim payload: %s", string(payloadJSON))
				}
				loggedFirstClaimPayload = true
			}
			if err := qdrantClient.UpsertClaimPayload(ctx, cl.ID, payload, claimVector); err != nil {
				log.Printf("      ! claim upsert error (%s): %v", cl.ID, err)
			}
		}
	}
	elapsed := time.Since(start)
	src.ClaimCount = len(allClaims)

	if len(sourceVector) > 0 {
		if err := qdrantClient.UpsertSource(ctx, src, sourceVector); err != nil {
			log.Printf("      ! source re-upsert (final claim count) failed: %v", err)
		} else {
			log.Printf("      updated source record: %d chunks, %d claims", src.ChunkCount, src.ClaimCount)
		}
	}

	emptyRate := 0.0
	if len(chunks) > 0 {
		emptyRate = float64(emptyChunks) / float64(len(chunks)) * 100
	}
	log.Printf("done in %s", elapsed.Round(time.Second))
	log.Printf("  source:        %s (%s)", src.ID, metaSource)
	log.Printf("  chunks:        %d", len(chunks))
	log.Printf("  claims:        %d", len(allClaims))
	log.Printf("  empty chunks:  %d / %d (%.1f%%)", emptyChunks, len(chunks), emptyRate)
	if !skipGraphability {
		extracted := len(chunks) - graphabilitySkipped
		savingPct := 0.0
		if len(chunks) > 0 {
			savingPct = float64(graphabilitySkipped) / float64(len(chunks)) * 100
		}
		log.Printf("  graphability:  %d skipped / %d extracted / %d gaps (%.1f%% token saving)",
			graphabilitySkipped, extracted, graphabilityGaps, savingPct)
	}
	if emptyRate > 20 {
		log.Printf("  [!] high empty-chunk rate (%.1f%%) — review extraction model or source content", emptyRate)
	}

	log.Printf("[4/4] Writing output")
	srcPath := filepath.Join(outputDir, src.ID+".source.json")
	if err := writeJSON(srcPath, src); err != nil {
		return fmt.Errorf("write source: %w", err)
	}
	chunksPath := filepath.Join(outputDir, src.ID+".chunks.json")
	if err := writeJSON(chunksPath, chunks); err != nil {
		return fmt.Errorf("write chunks: %w", err)
	}
	claimsPath := filepath.Join(outputDir, src.ID+".claims.json")
	if err := writeJSON(claimsPath, allClaims); err != nil {
		return fmt.Errorf("write claims: %w", err)
	}
	log.Printf("  wrote: %s  %s  %s", srcPath, chunksPath, claimsPath)
	return nil
}

// resolveSourceID uses catalog entry ID when available, otherwise sanitized filename.
func resolveSourceID(meta sourceMeta, cat *catalog, path string) string {
	_ = meta
	if entry, ok := cat.lookup(path); ok {
		return entry.ID
	}
	baseName := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	return sanitizeID(baseName)
}

// extractText reads the source file, converting PDF to text if needed.
// Image-only (scanned) PDFs are detected and routed through OCR automatically.
func extractText(path string) (string, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".pdf":
		raw, err := runPdftotext(path)
		if err != nil {
			return "", err
		}
		if isScannedPDF(raw) {
			if os.Getenv("MB_SKIP_OCR") == "1" {
				return "", fmt.Errorf("no text layer in %s (scanned PDF) and MB_SKIP_OCR=1", path)
			}
			log.Printf("      no text layer detected (scanned PDF) — running OCR fallback")
			raw, err = ocrExtract(path)
			if err != nil {
				return "", fmt.Errorf("ocr fallback: %w", err)
			}
		}
		return normalizeExtractedText(raw), nil
	case ".txt", ".md", "":
		b, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		return normalizeExtractedText(string(b)), nil
	default:
		return "", fmt.Errorf("unsupported extension %q (use .pdf or .txt)", ext)
	}
}

func runPdftotext(path string) (string, error) {
	if _, err := exec.LookPath("pdftotext"); err != nil {
		return "", fmt.Errorf("pdftotext not found on PATH; install poppler-utils: %w", err)
	}
	cmd := exec.Command("pdftotext", path, "-")
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("pdftotext: %w (stderr: %s)", err, stderr.String())
	}
	return stdout.String(), nil
}

// isScannedPDF returns true when pdftotext output is effectively empty —
// the signature of an image-only (scanned) PDF. Pages are split on form
// feeds; a page "has text" if it carries at least 50 alphanumeric runes.
// If fewer than 10% of pages have text, we call it a scan.
func isScannedPDF(raw string) bool {
	pages := strings.Split(raw, "\f")
	if len(pages) == 0 {
		return false
	}
	textPages := 0
	for _, p := range pages {
		n := 0
		for _, r := range p {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				n++
				if n >= 50 {
					break
				}
			}
		}
		if n >= 50 {
			textPages++
		}
	}
	return float64(textPages) < 0.10*float64(len(pages))
}

// ocrExtract OCRs a scanned PDF via ocrmypdf and returns the raw sidecar text.
//
// Behaviour:
//   - If a previously OCR'd sibling exists (<stem>-ocr.pdf convention), reuse
//     it instead of re-running OCR.
//   - Otherwise run ocrmypdf, writing the OCR'd PDF sibling (cached for future
//     runs) plus a sidecar text file that becomes the extraction result.
func ocrExtract(path string) (string, error) {
	ext := filepath.Ext(path)
	stem := strings.TrimSuffix(path, ext)
	ocrPDF := stem + "-ocr.pdf"

	if _, err := os.Stat(ocrPDF); err == nil {
		log.Printf("      reusing existing OCR sibling: %s", ocrPDF)
		return runPdftotext(ocrPDF)
	}

	if _, err := exec.LookPath("ocrmypdf"); err != nil {
		return "", fmt.Errorf("scanned PDF but ocrmypdf not on PATH (install: sudo apt install ocrmypdf, or pipx install ocrmypdf): %w", err)
	}

	sidecar, err := os.CreateTemp("", "mb-ocr-*.txt")
	if err != nil {
		return "", fmt.Errorf("create sidecar temp: %w", err)
	}
	sidecarPath := sidecar.Name()
	sidecar.Close()
	defer os.Remove(sidecarPath)

	lang := envOr("MB_OCR_LANG", "eng")
	args := []string{"--skip-text", "--deskew", "-l", lang, "--sidecar", sidecarPath}
	if extra := os.Getenv("MB_OCR_ARGS"); extra != "" {
		args = append(args, strings.Fields(extra)...)
	}
	args = append(args, path, ocrPDF)

	log.Printf("      ocrmypdf %s", strings.Join(args, " "))
	cmd := exec.Command("ocrmypdf", args...)
	cmd.Stdout = os.Stderr // progress/noise to console; extraction comes from the sidecar
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("ocrmypdf: %w", err)
	}
	log.Printf("      OCR complete — wrote %s", ocrPDF)

	b, err := os.ReadFile(sidecarPath)
	if err != nil {
		return "", fmt.Errorf("read sidecar: %w", err)
	}
	return string(b), nil
}

// normalizeExtractedText cleans up raw pdftotext output for reliable chunking.
//
// Steps:
//  1. Convert form feeds (\f) to paragraph breaks — pdftotext page boundaries.
//  2. Isolate section headers: ensure CHAPTER N / SESSION N / PART N lines are
//     surrounded by blank lines so the chunker sees them as standalone paragraphs.
//  3. Join multi-line chapter title blocks: when a header line is immediately
//     followed by short ALL-CAPS title continuation lines, collapse them onto
//     one line so the chunker gets a single clean label.
//  4. Strip lone page-number lines (digit-only or roman-numeral-only paragraphs).
//  5. Collapse runs of 3+ newlines back to 2.
func normalizeExtractedText(text string) string {
	// 1. Page breaks → paragraph breaks.
	text = strings.ReplaceAll(text, "\f", "\n\n")

	// 2, 3, 4. Line-level pass.
	lines := strings.Split(text, "\n")
	out := make([]string, 0, len(lines)+64)

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		trimmed := strings.TrimSpace(line)

		// Isolate section headers and absorb immediately-following ALL-CAPS subtitle lines.
		if isSectionHeaderLine(trimmed) {
			// Ensure blank line before.
			if len(out) > 0 && out[len(out)-1] != "" {
				out = append(out, "")
			}
			// Absorb continuation ALL-CAPS subtitle lines onto the header.
			header := trimmed
			for i+1 < len(lines) {
				next := strings.TrimSpace(lines[i+1])
				if next == "" {
					break
				}
				if isAllCapsLine(next) && len(next) <= 80 {
					header += " " + next
					i++
				} else {
					break
				}
			}
			// Header as its own paragraph, blank line after.
			out = append(out, header, "")
			continue
		}

		// Drop lone page-number lines.
		if isPageNumberLine(trimmed) {
			continue
		}

		out = append(out, line)
	}

	result := strings.Join(out, "\n")

	// 5. Collapse 3+ newlines → 2.
	result = regexp.MustCompile(`\n{3,}`).ReplaceAllString(result, "\n\n")
	return strings.TrimSpace(result)
}

// sectionHeaderRE matches lines like "CHAPTER 1", "Session IV", "PART TWO", "Appendix A".
var sectionHeaderRE = regexp.MustCompile(
	`(?i)^(chapter|session|part|section|appendix|book)\s+[\w]+.*$`,
)

// isSectionHeaderLine returns true for lines that are section boundary markers.
func isSectionHeaderLine(s string) bool {
	if len(s) == 0 || len(s) > 80 {
		return false
	}
	return sectionHeaderRE.MatchString(s)
}

// isAllCapsLine returns true if the line has no lowercase letters —
// typical of chapter subtitle continuation lines in older/channeled books.
func isAllCapsLine(s string) bool {
	if len(s) == 0 {
		return false
	}
	for _, r := range s {
		if r >= 'a' && r <= 'z' {
			return false
		}
	}
	return true
}

// pageNumberRE matches lines that are only digits or roman numerals.
var pageNumberRE = regexp.MustCompile(`^[0-9ivxlcdmIVXLCDM]+$`)

// isPageNumberLine returns true for bare page-number lines (≤6 chars).
func isPageNumberLine(s string) bool {
	if len(s) == 0 || len(s) > 6 {
		return false
	}
	return pageNumberRE.MatchString(s)
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// --- Source metadata extraction (LLM fallback) ---

type sourceMeta struct {
	Title       string `json:"title"`
	Author      string `json:"author"`
	Year        int    `json:"year"`
	ChannelType string `json:"channel_type"`
	Tradition   string `json:"tradition"`
	Medium      string `json:"medium"`
}

const metaSystemPrompt = `You extract bibliographic metadata from the opening pages of a book.

Return ONLY a JSON object with exactly these fields:
{
  "title": "<full book title>",
  "author": "<human author name, or 'Unknown'>",
  "year": <copyright or publication year as integer, or 0 if not found>,
  "channel_type": "<one of: authored, channeled, regression, dictated, dramatized, dialogue>",
  "tradition": "<named source tradition e.g. Seth, Ra/Law-of-One, QHHT/Cannon — empty string if none>",
  "medium": "<claimed non-human source or entity e.g. Seth, Ra, Nostradamus — empty string if none>"
}

channel_type guidance:
- authored: ordinary non-fiction or fiction book by a human author
- channeled: trance-dictated from a claimed non-human source (Seth Material, Ra/Law of One)
- regression: hypnotic regression transcripts (Dolores Cannon books, QHHT sessions)
- dictated: human author dictating to a scribe or amanuensis
- dramatized: fiction that conveys metaphysical doctrine through narrative
- dialogue: recorded interviews, workshops, Q&A sessions

No prose, no markdown fences, no explanation. Return ONLY the JSON object.`

// extractSourceMetadata runs an LLM call over the opening text of the book.
// Uses a smarter preview window that skips garbage front matter.
// Retries with a wider window if the first pass returns empty/Unknown title.
func extractSourceMetadata(ctx context.Context, client *llm.Client, model, text string) sourceMeta {
	preview := smartPreview(text, 8000)
	meta, ok := runMetaLLM(ctx, client, model, preview)
	if ok && meta.Title != "" && !strings.EqualFold(meta.Title, "unknown") {
		return meta
	}
	if len(text) > 8000 {
		log.Printf("      ! first metadata pass empty; retrying with offset window")
		offset := findProseStart(text, 4000)
		end := offset + 8000
		if end > len(text) {
			end = len(text)
		}
		meta2, ok2 := runMetaLLM(ctx, client, model, text[offset:end])
		if ok2 && meta2.Title != "" && !strings.EqualFold(meta2.Title, "unknown") {
			return meta2
		}
	}
	return meta
}

func runMetaLLM(ctx context.Context, client *llm.Client, model, preview string) (sourceMeta, bool) {
	req := llm.Request{
		Model:       model,
		Temperature: 0.1,
		Messages: []llm.Message{
			{Role: "system", Content: metaSystemPrompt},
			{Role: "user", Content: "Extract metadata from this book opening:\n\n" + preview},
		},
		ResponseFormat: &llm.ResponseFormat{Type: "json_object"},
	}
	raw, err := client.Complete(ctx, req)
	if err != nil {
		log.Printf("      ! metadata LLM call failed: %v", err)
		return sourceMeta{}, false
	}
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "```") {
		if nl := strings.Index(raw, "\n"); nl != -1 {
			raw = raw[nl+1:]
		}
		raw = strings.TrimSuffix(strings.TrimSpace(raw), "```")
		raw = strings.TrimSpace(raw)
	}
	var meta sourceMeta
	if err := json.Unmarshal([]byte(raw), &meta); err != nil {
		log.Printf("      ! metadata parse failed: %v (raw: %s)", err, raw)
		return sourceMeta{}, false
	}
	return meta, true
}

// smartPreview extracts up to maxChars of meaningful text, skipping garbage front matter.
func smartPreview(text string, maxChars int) string {
	lines := strings.Split(text, "\n")
	var buf strings.Builder
	meaningfulLines := 0
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			if meaningfulLines > 0 {
				buf.WriteString("\n")
			}
			continue
		}
		if meaningfulLines == 0 && isGarbageLine(trimmed) {
			continue
		}
		buf.WriteString(line)
		buf.WriteString("\n")
		meaningfulLines++
		if buf.Len() >= maxChars {
			break
		}
	}
	result := buf.String()
	if len(result) > maxChars {
		result = result[:maxChars]
	}
	return strings.TrimSpace(result)
}

func findProseStart(text string, skipChars int) int {
	if skipChars >= len(text) {
		return 0
	}
	idx := strings.Index(text[skipChars:], "\n\n")
	if idx < 0 {
		return skipChars
	}
	return skipChars + idx + 2
}

func isGarbageLine(s string) bool {
	if len(s) < 3 {
		return true
	}
	allDigits := true
	for _, r := range s {
		if !unicode.IsDigit(r) && r != ' ' {
			allDigits = false
			break
		}
	}
	if allDigits {
		return true
	}
	letters := 0
	for _, r := range s {
		if unicode.IsLetter(r) {
			letters++
		}
	}
	if float64(letters)/float64(len([]rune(s))) < 0.30 {
		return true
	}
	return false
}

func ifEmpty(a, b string) string {
	if strings.TrimSpace(a) != "" {
		return a
	}
	return b
}

var sanitizeRE = regexp.MustCompile(`[^a-z0-9_]+`)

func sanitizeID(s string) string {
	s = strings.ToLower(s)
	s = sanitizeRE.ReplaceAllString(s, "_")
	s = strings.Trim(s, "_")
	if s == "" {
		s = "source"
	}
	return s
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	var n int
	_, err := fmt.Sscanf(v, "%d", &n)
	if err != nil {
		return fallback
	}
	return n
}

func sourceEmbeddingText(title, author, text string, maxChars int) string {
	header := fmt.Sprintf("%s\n%s\n\n", title, author)
	body := truncateForEmbedding(text, maxChars-len(header))
	return header + body
}

func truncateForEmbedding(text string, maxChars int) string {
	if maxChars <= 0 {
		return ""
	}
	if len(text) <= maxChars {
		return text
	}
	return text[:maxChars]
}

func claimChapterNames(attributions []claim.Attribution) []string {
	if len(attributions) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(attributions))
	chapters := make([]string, 0, len(attributions))
	for _, attr := range attributions {
		chapter := strings.TrimSpace(attr.Chapter)
		if chapter == "" {
			continue
		}
		if _, exists := seen[chapter]; exists {
			continue
		}
		seen[chapter] = struct{}{}
		chapters = append(chapters, chapter)
	}
	if len(chapters) == 0 {
		return nil
	}
	return chapters
}
