// Command mb-codex ingests Mesoamerican (and other) pictographic codices into
// a dedicated Qdrant collection using Google's gemini-embedding-2 multimodal
// embedding model via OpenRouter.
//
// Each page produces THREE named vectors in the same Qdrant point:
//
//	"scholarly"  — embed of the iconographic/calendrical description
//	"narrative"  — embed of the priest-scholar performance narration
//	"image"      — direct image embed (visual fingerprint of the page)
//
// Usage:
//
//	mb-codex ingest [--dry-run] <path-to-pdf>
//	mb-codex status
package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/joho/godotenv"
)

const (
	collectionCodex    = "mb_codex"
	defaultVectorSize  = 3072
	defaultEmbedModel  = "google/gemini-embedding-2"
	defaultVisionModel = "google/gemini-2.5-flash"
	openRouterBase     = "https://openrouter.ai/api/v1"
	embedRetries       = 3
	embedRetryDelay    = 4 * time.Second
)

const (
	vecScholarly = "scholarly"
	vecNarrative = "narrative"
	vecImage     = "image"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	_ = godotenv.Load()

	switch os.Args[1] {
	case "ingest":
		dryRun := false
		args := os.Args[2:]
		var filtered []string
		for _, a := range args {
			if a == "--dry-run" || a == "-n" {
				dryRun = true
			} else {
				filtered = append(filtered, a)
			}
		}
		if len(filtered) == 0 {
			fmt.Fprintln(os.Stderr, "ingest: missing path")
			usage()
			os.Exit(1)
		}
		if err := cmdIngest(filtered[0], dryRun); err != nil {
			log.Fatalf("ingest failed: %v", err)
		}
	case "status":
		if err := cmdStatus(); err != nil {
			log.Fatalf("status failed: %v", err)
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
	fmt.Fprintln(os.Stderr, `meta-bridge codex (mb-codex) — pictographic manuscript ingest engine

Usage:
  mb-codex ingest [--dry-run] <path>   Ingest a PDF, CBR/CBZ, or image directory
  mb-codex status                      Show mb_codex collection info

Each page produces 3 named vectors: scholarly | narrative | image

Env (inherits from .env):
  OPENROUTER_API_KEY       required
  QDRANT_URL               default: http://localhost:6333
  QDRANT_API_KEY           optional
  CODEX_EMBED_MODEL        default: google/gemini-embedding-2
  CODEX_EMBED_DIMENSIONS   128-3072 (default: 3072)
  CODEX_VISION_MODEL       default: google/gemini-2.5-flash
  CODEX_COLLECTION         default: mb_codex
  CODEX_SOURCE_ID          override auto source ID
  CODEX_TITLE              override codex title
  CODEX_TRADITION          e.g. "Aztec", "Mixtec", "Maya"
  CODEX_TRADITION_NOTES    free-form tradition notes
  CODEX_YEAR               approximate year of composition
  CODEX_MAX_PAGES          limit pages processed
  CODEX_SKIP_VISION        set to 1 to skip LLM description
  CODEX_OUTPUT_DIR         JSON artifact dir (default: ./output)
  CODEX_DPI                rasterization DPI (default: 200)`)
}

type pageRecord struct {
	SourceID              string  `json:"source_id"`
	Title                 string  `json:"title"`
	Tradition             string  `json:"tradition"`
	TraditionNotes        string  `json:"tradition_notes,omitempty"`
	Year                  int     `json:"year,omitempty"`
	PageIndex             int     `json:"page_index"`
	PageName              string  `json:"page_name"`
	DescriptionScholarly  string  `json:"description_scholarly"`
	DescriptionNarrative  string  `json:"description_narrative"`
	DescriptionFull       string  `json:"description_full"`
	VisionModel           string  `json:"vision_model"`
	EmbedModel            string  `json:"embed_model"`
	EmbedDims             int     `json:"embed_dims"`
	IngestedAt            string  `json:"ingested_at"`
	EntityType            string  `json:"entity_type"`
	SimScholarlyImage     float64 `json:"sim_scholarly_image,omitempty"`
	SimNarrativeImage     float64 `json:"sim_narrative_image,omitempty"`
	SimScholarlyNarrative float64 `json:"sim_scholarly_narrative,omitempty"`
}

func cmdIngest(path string, dryRun bool) error {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("OPENROUTER_API_KEY not set")
	}

	embedModel  := envOr("CODEX_EMBED_MODEL", defaultEmbedModel)
	embedDims   := envInt("CODEX_EMBED_DIMENSIONS", defaultVectorSize)
	visionModel := envOr("CODEX_VISION_MODEL", defaultVisionModel)
	collection  := envOr("CODEX_COLLECTION", collectionCodex)
	skipVision  := os.Getenv("CODEX_SKIP_VISION") == "1"
	maxPages    := envInt("CODEX_MAX_PAGES", 0)
	outputDir   := envOr("CODEX_OUTPUT_DIR", "./output")
	dpi         := envInt("CODEX_DPI", 200)

	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	sourceID  := envOr("CODEX_SOURCE_ID", sanitizeID(filepath.Base(path)))
	title     := envOr("CODEX_TITLE", humanTitle(filepath.Base(path)))
	tradition := envOr("CODEX_TRADITION", "Mesoamerican")
	tradNotes := os.Getenv("CODEX_TRADITION_NOTES")
	year      := envInt("CODEX_YEAR", 0)

	log.Printf("[mb-codex] source_id=%q  title=%q  tradition=%q", sourceID, title, tradition)
	log.Printf("[mb-codex] embed=%s  dims=%d  vision=%s  collection=%s", embedModel, embedDims, visionModel, collection)
	if dryRun {
		log.Printf("[mb-codex] DRY RUN — no data written to Qdrant")
	}

	log.Printf("[1/4] Loading images from: %s", path)
	images, err := loadImages(path, dpi)
	if err != nil {
		return fmt.Errorf("load images: %w", err)
	}
	log.Printf("      found %d pages", len(images))
	if len(images) == 0 {
		return fmt.Errorf("no images found in %q", path)
	}
	if maxPages > 0 && maxPages < len(images) {
		log.Printf("      CODEX_MAX_PAGES=%d; limiting to first %d pages", maxPages, maxPages)
		images = images[:maxPages]
	}

	or  := newORClient(apiKey)
	qdr := newQdrantClient(envOr("QDRANT_URL", "http://localhost:6333"), os.Getenv("QDRANT_API_KEY"))
	ctx := context.Background()

	collectionReady := false
	ensureCollection := func(vecSize int) error {
		if collectionReady {
			return nil
		}
		if dryRun {
			collectionReady = true
			return nil
		}
		if err := qdr.ensureNamedVectorCollection(ctx, collection, vecSize); err != nil {
			return err
		}
		collectionReady = true
		return nil
	}

	log.Printf("[2/4] Processing %d pages...", len(images))
	var records []pageRecord
	skipped := 0
	start := time.Now()

	for i, img := range images {
		log.Printf("  [%d/%d] page=%s  size=%dKB", i+1, len(images), img.name, len(img.data)/1024)

		scholarly, narrative, full := "", "", ""
		if !skipVision {
			desc, err := or.describeCodexPage(ctx, visionModel, img.data, img.mimeType, title, i+1)
			if err != nil {
				log.Printf("    ! vision: %v", err)
			} else {
				full = desc
				scholarly, narrative = splitDescription(desc)
				log.Printf("    vision: %dch (scholarly=%dch narrative=%dch)", len(full), len(scholarly), len(narrative))
			}
		}

		var scholVec []float64
		if scholarly != "" {
			scholVec, err = or.embedTextWithRetry(ctx, embedModel, embedDims,
				fmt.Sprintf("%s page %d — scholarly:\n%s", title, i+1, scholarly))
			if err != nil {
				log.Printf("    ! scholarly embed: %v", err)
			} else {
				log.Printf("    scholarly: %dd", len(scholVec))
			}
		}

		var narrVec []float64
		if narrative != "" {
			narrVec, err = or.embedTextWithRetry(ctx, embedModel, embedDims,
				fmt.Sprintf("%s page %d — narrative:\n%s", title, i+1, narrative))
			if err != nil {
				log.Printf("    ! narrative embed: %v", err)
			} else {
				log.Printf("    narrative: %dd", len(narrVec))
			}
		}

		var imageVec []float64
		imageVec, err = or.embedImageWithRetry(ctx, embedModel, embedDims, img.data, img.mimeType)
		if err != nil {
			log.Printf("    ! image embed: %v", err)
		} else {
			log.Printf("    image: %dd", len(imageVec))
		}

		if len(scholVec) == 0 && len(narrVec) == 0 && len(imageVec) == 0 {
			log.Printf("    ! all embeds failed — skipping page %d", i+1)
			skipped++
			continue
		}

		refVec := firstNonEmpty(scholVec, narrVec, imageVec)
		if err := ensureCollection(len(refVec)); err != nil {
			return fmt.Errorf("ensure collection: %w", err)
		}

		simSI := cosine(scholVec, imageVec)
		simNI := cosine(narrVec, imageVec)
		simSN := cosine(scholVec, narrVec)
		log.Printf("    sims  s↔i:%.3f  n↔i:%.3f  s↔n:%.3f", simSI, simNI, simSN)

		rec := pageRecord{
			SourceID:              sourceID,
			Title:                 title,
			Tradition:             tradition,
			TraditionNotes:        tradNotes,
			Year:                  year,
			PageIndex:             i + 1,
			PageName:              img.name,
			DescriptionScholarly:  scholarly,
			DescriptionNarrative:  narrative,
			DescriptionFull:       full,
			VisionModel:           visionModel,
			EmbedModel:            embedModel,
			EmbedDims:             len(refVec),
			IngestedAt:            time.Now().UTC().Format(time.RFC3339),
			EntityType:            "codex_page",
			SimScholarlyImage:     simSI,
			SimNarrativeImage:     simNI,
			SimScholarlyNarrative: simSN,
		}
		records = append(records, rec)

		if !dryRun {
			payload := map[string]interface{}{
				"entity_type":             "codex_page",
				"source_id":               sourceID,
				"title":                   title,
				"tradition":               tradition,
				"page_index":              i + 1,
				"page_name":               img.name,
				"description_scholarly":   scholarly,
				"description_narrative":   narrative,
				"vision_model":            visionModel,
				"embed_model":             embedModel,
				"embed_dims":              len(refVec),
				"ingested_at":             rec.IngestedAt,
				"sim_scholarly_image":     simSI,
				"sim_narrative_image":     simNI,
				"sim_scholarly_narrative": simSN,
			}
			if tradNotes != "" {
				payload["tradition_notes"] = tradNotes
			}
			if year > 0 {
				payload["year"] = year
			}

			vectors := map[string][]float64{}
			if len(scholVec) > 0 { vectors[vecScholarly] = scholVec }
			if len(narrVec) > 0  { vectors[vecNarrative] = narrVec }
			if len(imageVec) > 0 { vectors[vecImage]     = imageVec }

			pointID := pointIDForKey(fmt.Sprintf("codex:%s:page:%04d", sourceID, i+1))
			if err := qdr.upsertNamedVectorPoint(ctx, collection, pointID, vectors, payload); err != nil {
				log.Printf("    ! qdrant upsert: %v — skipping", err)
				skipped++
				records = records[:len(records)-1]
			}
		}
	}

	elapsed := time.Since(start)
	log.Printf("[3/4] Summary  source=%s  ingested=%d  skipped=%d  elapsed=%s",
		sourceID, len(records), skipped, elapsed.Round(time.Second))
	if dryRun {
		log.Printf("  [DRY RUN] nothing written to Qdrant")
	}

	log.Printf("[4/4] Writing output")
	outPath := filepath.Join(outputDir, sourceID+".codex.json")
	if !dryRun {
		if err := writeJSON(outPath, records); err != nil {
			return fmt.Errorf("write output: %w", err)
		}
		log.Printf("  wrote: %s", outPath)
	}
	return nil
}

func cmdStatus() error {
	qdr := newQdrantClient(envOr("QDRANT_URL", "http://localhost:6333"), os.Getenv("QDRANT_API_KEY"))
	info, err := qdr.collectionInfo(context.Background(), envOr("CODEX_COLLECTION", collectionCodex))
	if err != nil {
		return err
	}
	b, _ := json.MarshalIndent(info, "", "  ")
	fmt.Println(string(b))
	return nil
}

// ── Image loading ─────────────────────────────────────────────────────────────

type imgEntry struct {
	name     string
	data     []byte
	mimeType string
}

var supportedExts = map[string]string{
	".jpg": "image/jpeg", ".jpeg": "image/jpeg",
	".png": "image/png",  ".webp": "image/webp",
}

func loadImages(path string, dpi int) ([]imgEntry, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.IsDir() {
		return loadFromDir(path)
	}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".pdf":
		return loadFromPDF(path, dpi)
	case ".cbr", ".cbz", ".zip":
		return loadFromCBR(path)
	default:
		if mime, ok := supportedExts[strings.ToLower(filepath.Ext(path))]; ok {
			data, err := os.ReadFile(path)
			if err != nil {
				return nil, err
			}
			return []imgEntry{{name: filepath.Base(path), data: data, mimeType: mime}}, nil
		}
		return nil, fmt.Errorf("unsupported file type %q", filepath.Ext(path))
	}
}

func loadFromPDF(pdfPath string, dpi int) ([]imgEntry, error) {
	if _, err := exec.LookPath("pdftoppm"); err != nil {
		return nil, fmt.Errorf("pdftoppm not found; install poppler-utils")
	}
	out, err := exec.Command("pdfinfo", pdfPath).Output()
	if err != nil {
		return nil, fmt.Errorf("pdfinfo: %w", err)
	}
	pages := 0
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "Pages:") {
			fmt.Sscanf(strings.TrimSpace(strings.TrimPrefix(line, "Pages:")), "%d", &pages)
		}
	}
	if pages == 0 {
		return nil, fmt.Errorf("could not determine page count")
	}
	log.Printf("      pdf: %d pages", pages)

	slug := sanitizeID(filepath.Base(pdfPath))
	var images []imgEntry
	for p := 1; p <= pages; p++ {
		prefix := fmt.Sprintf("/tmp/mbcodex_%s_%04d", slug, p)
		cmd := exec.Command("pdftoppm", "-r", fmt.Sprintf("%d", dpi),
			"-jpeg", "-f", fmt.Sprintf("%d", p), "-l", fmt.Sprintf("%d", p), pdfPath, prefix)
		if err := cmd.Run(); err != nil {
			log.Printf("      ! pdftoppm page %d: %v", p, err)
			continue
		}
		matches, _ := filepath.Glob(prefix + "*.jpg")
		if len(matches) == 0 {
			continue
		}
		data, err := os.ReadFile(matches[0])
		os.Remove(matches[0])
		if err != nil {
			continue
		}
		images = append(images, imgEntry{
			name:     fmt.Sprintf("page_%04d.jpg", p),
			data:     data,
			mimeType: "image/jpeg",
		})
	}
	return images, nil
}

func loadFromDir(dir string) ([]imgEntry, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var images []imgEntry
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		mime, ok := supportedExts[strings.ToLower(filepath.Ext(e.Name()))]
		if !ok {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			log.Printf("  ! skip %s: %v", e.Name(), err)
			continue
		}
		images = append(images, imgEntry{name: e.Name(), data: data, mimeType: mime})
	}
	sort.Slice(images, func(i, j int) bool { return images[i].name < images[j].name })
	return images, nil
}

func loadFromCBR(path string) ([]imgEntry, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open as zip: %w (RAR .cbr? try: unrar x file.cbr && zip -r file.cbz *.jpg)", err)
	}
	defer r.Close()
	var images []imgEntry
	for _, f := range r.File {
		if f.FileInfo().IsDir() {
			continue
		}
		mime, ok := supportedExts[strings.ToLower(filepath.Ext(f.Name))]
		if !ok {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			continue
		}
		data, _ := io.ReadAll(rc)
		rc.Close()
		if len(data) > 0 {
			images = append(images, imgEntry{name: filepath.Base(f.Name), data: data, mimeType: mime})
		}
	}
	sort.Slice(images, func(i, j int) bool { return images[i].name < images[j].name })
	return images, nil
}

// ── OpenRouter client ─────────────────────────────────────────────────────────

type orClient struct {
	apiKey string
	http   *http.Client
}

func newORClient(apiKey string) *orClient {
	return &orClient{apiKey: apiKey, http: &http.Client{Timeout: 120 * time.Second}}
}

func (c *orClient) post(ctx context.Context, endpoint string, body any) ([]byte, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", openRouterBase+endpoint, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("HTTP-Referer", "https://github.com/meistro57/meta-bridge")
	req.Header.Set("X-Title", "mb-codex")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("openrouter %d: %s", resp.StatusCode, string(raw))
	}
	return raw, nil
}

const visionSystem = `You are an expert in Mesoamerican manuscripts, pre-Columbian iconography, and ritual calendar systems.

You are analyzing a page from an Aztec or Mixtec codex. These manuscripts encode ritual, cosmological, and calendrical knowledge through pictographic imagery.

Please provide TWO sections using EXACTLY these headers:

## SCHOLARLY DESCRIPTION
Describe what you observe with precision:
- Overall page structure and layout (grid? narrative? mixed?)
- Number and arrangement of visual cells or registers if present
- Deities or figures identified (use Nahuatl names: Quetzalcoatl, Tlaloc, Tezcatlipoca, Xipe Totec, etc.)
- Calendar glyphs, day signs, numerical dots/bars visible
- Colors, cardinal directions, ritual objects (mirrors, serpents, flints, etc.)
- Any recognizable ritual scenes (sacrifice, creation, underworld journey, etc.)

## NARRATIVE PERFORMANCE
Narrate this page as an Aztec priest-scholar performing it aloud to initiates.
Connect figures, movements, and symbols into a living story. Use present tense.
Be specific and rich — this will be used for semantic search and cross-tradition synthesis
with Egyptian, Hermetic, Gnostic, Vedic, Ra Material, and Seth Material texts.`

func (c *orClient) describeCodexPage(ctx context.Context, model string, imgData []byte, mimeType, codexTitle string, pageNum int) (string, error) {
	b64 := base64.StdEncoding.EncodeToString(imgData)
	dataURI := fmt.Sprintf("data:%s;base64,%s", mimeType, b64)
	payload := map[string]interface{}{
		"model": model,
		"messages": []map[string]interface{}{
			{"role": "system", "content": visionSystem},
			{"role": "user", "content": []map[string]interface{}{
				{"type": "image_url", "image_url": map[string]string{"url": dataURI}},
				{"type": "text", "text": fmt.Sprintf("This is page %d of the %s. Please describe it.", pageNum, codexTitle)},
			}},
		},
		"temperature": 0.3,
		"max_tokens":  1500,
	}
	raw, err := c.post(ctx, "/chat/completions", payload)
	if err != nil {
		return "", err
	}
	var resp struct {
		Choices []struct {
			Message struct{ Content string `json:"content"` } `json:"message"`
		} `json:"choices"`
		Error *struct{ Message string `json:"message"` } `json:"error,omitempty"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", fmt.Errorf("unmarshal vision: %w", err)
	}
	if resp.Error != nil {
		return "", fmt.Errorf("vision: %s", resp.Error.Message)
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices in vision response")
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

func (c *orClient) embedTextWithRetry(ctx context.Context, model string, dims int, text string) ([]float64, error) {
	return c.embedWithRetry(ctx, map[string]interface{}{"model": model, "input": text, "dimensions": dims})
}

func (c *orClient) embedImageWithRetry(ctx context.Context, model string, dims int, imgData []byte, mimeType string) ([]float64, error) {
	b64 := base64.StdEncoding.EncodeToString(imgData)
	return c.embedWithRetry(ctx, map[string]interface{}{
		"model":      model,
		"input":      fmt.Sprintf("data:%s;base64,%s", mimeType, b64),
		"dimensions": dims,
	})
}

func (c *orClient) embedWithRetry(ctx context.Context, payload map[string]interface{}) ([]float64, error) {
	var lastErr error
	for attempt := 1; attempt <= embedRetries; attempt++ {
		raw, err := c.post(ctx, "/embeddings", payload)
		if err != nil {
			lastErr = err
		} else {
			var resp struct {
				Data []struct {
					Embedding []float64 `json:"embedding"`
				} `json:"data"`
				Error *struct{ Message string `json:"message"` } `json:"error,omitempty"`
			}
			if err := json.Unmarshal(raw, &resp); err != nil {
				lastErr = fmt.Errorf("unmarshal embed: %w", err)
			} else if resp.Error != nil {
				lastErr = fmt.Errorf("embed API: %s", resp.Error.Message)
			} else if len(resp.Data) > 0 && len(resp.Data[0].Embedding) > 0 {
				return resp.Data[0].Embedding, nil
			} else {
				lastErr = fmt.Errorf("empty embedding in response")
			}
		}
		if attempt < embedRetries {
			log.Printf("    [embed retry %d/%d] %v", attempt, embedRetries, lastErr)
			time.Sleep(embedRetryDelay)
		}
	}
	return nil, lastErr
}

// ── Qdrant client (named vectors) ─────────────────────────────────────────────

type qdrantClient struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

func newQdrantClient(baseURL, apiKey string) *qdrantClient {
	return &qdrantClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		http:    &http.Client{Timeout: 30 * time.Second},
	}
}

func (q *qdrantClient) do(ctx context.Context, method, endpoint string, body any) ([]byte, int, error) {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, 0, err
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, q.baseURL+endpoint, r)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if q.apiKey != "" {
		req.Header.Set("api-key", q.apiKey)
	}
	resp, err := q.http.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	return raw, resp.StatusCode, err
}

func (q *qdrantClient) ensureNamedVectorCollection(ctx context.Context, name string, vecSize int) error {
	_, status, err := q.do(ctx, "GET", "/collections/"+name, nil)
	if err != nil {
		return err
	}
	if status == http.StatusOK {
		log.Printf("  [qdrant] collection %q already exists", name)
		return nil
	}
	if status != http.StatusNotFound {
		return fmt.Errorf("unexpected status %d checking collection %s", status, name)
	}
	vecConf := map[string]interface{}{"size": vecSize, "distance": "Cosine"}
	body := map[string]interface{}{
		"vectors": map[string]interface{}{
			vecScholarly: vecConf,
			vecNarrative: vecConf,
			vecImage:     vecConf,
		},
	}
	raw, status, err := q.do(ctx, "PUT", "/collections/"+name, body)
	if err != nil {
		return err
	}
	if status >= 400 {
		return fmt.Errorf("create collection %s: status %d: %s", name, status, string(raw))
	}
	log.Printf("  [qdrant] created collection %q (scholarly|narrative|image @ %dd Cosine)", name, vecSize)
	return nil
}

func (q *qdrantClient) upsertNamedVectorPoint(ctx context.Context, collection string, id uint64, vectors map[string][]float64, payload map[string]interface{}) error {
	body := map[string]interface{}{
		"points": []map[string]interface{}{
			{"id": id, "vectors": vectors, "payload": payload},
		},
	}
	raw, status, err := q.do(ctx, "PUT", "/collections/"+collection+"/points?wait=true", body)
	if err != nil {
		return err
	}
	if status >= 400 {
		return fmt.Errorf("upsert point %d: status %d: %s", id, status, string(raw))
	}
	return nil
}

func (q *qdrantClient) collectionInfo(ctx context.Context, name string) (map[string]interface{}, error) {
	raw, status, err := q.do(ctx, "GET", "/collections/"+name, nil)
	if err != nil {
		return nil, err
	}
	if status == http.StatusNotFound {
		return map[string]interface{}{"status": "not_found", "collection": name}, nil
	}
	var out map[string]interface{}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// ── Description parser ────────────────────────────────────────────────────────

func splitDescription(description string) (scholarly, narrative string) {
	lower := strings.ToLower(description)
	si := strings.Index(lower, "## scholarly description")
	ni := strings.Index(lower, "## narrative performance")
	if si != -1 && ni != -1 {
		scholarly = strings.TrimSpace(description[si+len("## scholarly description"):ni])
		narrative = strings.TrimSpace(description[ni+len("## narrative performance"):])
	} else if ni != -1 {
		scholarly = strings.TrimSpace(description[:ni])
		narrative = strings.TrimSpace(description[ni+len("## narrative performance"):])
	} else if si != -1 {
		scholarly = strings.TrimSpace(description[si+len("## scholarly description"):])
	} else {
		scholarly = strings.TrimSpace(description)
	}
	return
}

// ── Utilities ─────────────────────────────────────────────────────────────────

func pointIDForKey(key string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(key))
	return h.Sum64()
}

func cosine(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		if i >= len(b) {
			break
		}
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	sqrtf := func(x float64) float64 {
		z := x
		for i := 0; i < 50; i++ {
			z -= (z*z - x) / (2 * z)
		}
		return z
	}
	return dot / (sqrtf(na) * sqrtf(nb))
}

func firstNonEmpty(vecs ...[]float64) []float64 {
	for _, v := range vecs {
		if len(v) > 0 {
			return v
		}
	}
	return nil
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func sanitizeID(s string) string {
	s = strings.ToLower(strings.TrimSuffix(filepath.Base(s), filepath.Ext(s)))
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteRune('_')
		}
	}
	result := b.String()
	for strings.Contains(result, "__") {
		result = strings.ReplaceAll(result, "__", "_")
	}
	return strings.Trim(result, "_")
}

func humanTitle(filename string) string {
	s := strings.TrimSuffix(filename, filepath.Ext(filename))
	s = strings.ReplaceAll(s, "_", " ")
	s = strings.ReplaceAll(s, "-", " ")
	if len(s) > 0 {
		s = strings.ToUpper(s[:1]) + s[1:]
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
	fmt.Sscanf(v, "%d", &n)
	if n == 0 {
		return fallback
	}
	return n
}
