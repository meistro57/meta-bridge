// Command mb-academic-test is a test harness for academic PDF ingestion.
//
// It mirrors the full mb ingest pipeline but:
//   - Uses the academic chunker (SplitAcademic) for numbered-section PDFs
//   - Writes to live collections (mb_chunks, mb_claims, mb_sources)
//
// Usage:
//
//	go run ./cmd/mb-academic-test incoming/02110tpnews_11232020.pdf
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/meistro57/meta-bridge/internal/chunker"
	"github.com/meistro57/meta-bridge/internal/claim"
	"github.com/meistro57/meta-bridge/internal/extractor"
	"github.com/meistro57/meta-bridge/internal/llm"
	"github.com/meistro57/meta-bridge/internal/source"
	"github.com/meistro57/meta-bridge/internal/store"
)

const (
	testCollectionClaims  = "mb_claims"
	testCollectionChunks  = "mb_chunks"
	testCollectionSources = "mb_sources"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, `mb-academic-test — academic PDF ingestion

Usage:
  go run ./cmd/mb-academic-test <path-to-pdf>

Same env vars as mb ingest. Writes to mb_chunks, mb_claims, mb_sources.`)
		os.Exit(1)
	}

	_ = godotenv.Load()

	if err := run(os.Args[1]); err != nil {
		log.Fatalf("failed: %v", err)
	}
}

func run(path string) error {
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

	// --- 1. Extract text ---
	log.Printf("[1/4] Extracting text from %s", path)
	text, err := extractText(path)
	if err != nil {
		return fmt.Errorf("extract text: %w", err)
	}
	log.Printf("      got %d characters (~%d tokens)", len(text), len(text)/4)

	// --- 2. Build Source record ---
	baseName := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	src := source.NewSource(
		envOr("MB_SOURCE_ID", sanitizeID(baseName)),
		envOr("MB_TITLE", baseName),
		envOr("MB_AUTHOR", "Unknown"),
	)
	src.SourcePath = path

	// --- 3. Chunk with academic chunker ---
	log.Printf("[2/4] Chunking (academic mode)")
	opts := chunker.DefaultAcademicOptions()
	chunks := chunker.SplitAcademic(text, opts)
	log.Printf("      produced %d chunks", len(chunks))

	maxChunks := envInt("MB_MAX_CHUNKS", 0)
	if maxChunks > 0 && maxChunks < len(chunks) {
		log.Printf("      MB_MAX_CHUNKS=%d; limiting to first %d chunks", maxChunks, maxChunks)
		chunks = chunks[:maxChunks]
	}
	src.ChunkCount = len(chunks)

	// --- 4. Extract claims and index into live collections ---
	log.Printf("[3/4] Extracting claims and indexing into live collections (model=%s, embed=%s/%s)",
		extractionModel, embedProvider, embedModel)

	client := llm.NewClient(apiKey)
	client.SetOllamaURL(envOr("OLLAMA_URL", "http://localhost:11434"))
	client.SetEmbeddingProvider(embedProvider)
	client.SetEmbeddingModel(embedModel)
	ex := extractor.New(client, extractionModel)

	// Create a store client that targets the active collections.
	qdrantClient := newTestStoreClient(
		envOr("QDRANT_URL", "http://localhost:6333"),
		os.Getenv("QDRANT_API_KEY"),
	)

	ctx := context.Background()
	var allClaims []claim.Claim
	embedMaxChars := envInt("MB_EMBED_MAX_CHARS", 8000)
	collectionsReady := false

	ensureCollections := func(vector []float64) bool {
		if collectionsReady {
			return true
		}
		if err := qdrantClient.EnsureTestCollections(ctx, len(vector)); err != nil {
			log.Printf("      ! qdrant collection init failed: %v", err)
			return false
		}
		collectionsReady = true
		return true
	}

	// Upsert source.
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

	start := time.Now()
	for i, ch := range chunks {
		log.Printf("      chunk %d/%d (section=%q, ~%d tokens)",
			i+1, len(chunks), ch.Chapter, ch.TokenEst)

		// Embed and upsert chunk.
		chunkVector, err := client.Embed(ctx, truncateForEmbedding(ch.Text, embedMaxChars))
		if err != nil {
			log.Printf("      ! chunk embedding error on chunk %d: %v", i, err)
		} else if ensureCollections(chunkVector) {
			if err := qdrantClient.UpsertChunk(ctx, src.ID, ch, chunkVector); err != nil {
				log.Printf("      ! chunk upsert error on chunk %d: %v", i, err)
			}
		}

		// Extract claims.
		claims, err := ex.ExtractChunk(ctx, src.ID, ch, idFn)
		if err != nil {
			log.Printf("      ! extraction error on chunk %d: %v", i, err)
			continue
		}
		log.Printf("        -> %d claims", len(claims))
		allClaims = append(allClaims, claims...)

		// Embed and upsert each claim.
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

			if err := qdrantClient.UpsertClaimPayload(ctx, cl.ID, payload, claimVector); err != nil {
				log.Printf("      ! claim upsert error (%s): %v", cl.ID, err)
			}
		}
	}
	elapsed := time.Since(start)
	src.ClaimCount = len(allClaims)

	// --- 5. Write JSON output ---
	log.Printf("[4/4] Writing output")
	srcPath := filepath.Join(outputDir, src.ID+".academic_test.source.json")
	if err := writeJSON(srcPath, src); err != nil {
		return fmt.Errorf("write source: %w", err)
	}
	chunksPath := filepath.Join(outputDir, src.ID+".academic_test.chunks.json")
	if err := writeJSON(chunksPath, chunks); err != nil {
		return fmt.Errorf("write chunks: %w", err)
	}
	claimsPath := filepath.Join(outputDir, src.ID+".academic_test.claims.json")
	if err := writeJSON(claimsPath, allClaims); err != nil {
		return fmt.Errorf("write claims: %w", err)
	}

	log.Printf("done in %s", elapsed.Round(time.Second))
	log.Printf("  %d chunks processed", len(chunks))
	log.Printf("  %d claims extracted", len(allClaims))
	log.Printf("  target collections: %s, %s, %s", testCollectionChunks, testCollectionClaims, testCollectionSources)
	log.Printf("  wrote:")
	log.Printf("    %s", srcPath)
	log.Printf("    %s", chunksPath)
	log.Printf("    %s", claimsPath)
	return nil
}

// ---------------------------------------------------------------------------
// testStoreClient wraps store.Client and targets configured collection names.
// We can't modify store.Client (no changes to existing code), so we wrap it
// with direct Qdrant HTTP calls using the same patterns.
// ---------------------------------------------------------------------------

type testStoreClient struct {
	inner *store.Client
}

func newTestStoreClient(baseURL, apiKey string) *testStoreClient {
	return &testStoreClient{inner: store.NewClient(baseURL, apiKey)}
}

func (t *testStoreClient) EnsureTestCollections(ctx context.Context, vectorSize int) error {
	if vectorSize <= 0 {
		return fmt.Errorf("invalid vector size: %d", vectorSize)
	}
	for _, collection := range []string{testCollectionSources, testCollectionChunks, testCollectionClaims} {
		if err := t.ensureTestCollection(ctx, collection, vectorSize); err != nil {
			return err
		}
	}
	return nil
}

func (t *testStoreClient) ensureTestCollection(ctx context.Context, collection string, vectorSize int) error {
	qdrantURL := strings.TrimRight(envOr("QDRANT_URL", "http://localhost:6333"), "/")
	url := fmt.Sprintf("%s/collections/%s", qdrantURL, collection)

	checkReq, err := newHTTPRequest(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("build check request for %s: %w", collection, err)
	}
	checkResp, err := httpClient.Do(checkReq)
	if err != nil {
		return fmt.Errorf("check collection %s: %w", collection, err)
	}
	checkBody, _ := io.ReadAll(checkResp.Body)
	checkResp.Body.Close()

	if checkResp.StatusCode == http.StatusOK {
		return nil
	}
	if checkResp.StatusCode != http.StatusNotFound {
		return fmt.Errorf("check collection %s: unexpected status %d %s", collection, checkResp.StatusCode, strings.TrimSpace(string(checkBody)))
	}

	type vectorsConfig struct {
		Size     int    `json:"size"`
		Distance string `json:"distance"`
	}
	type collectionRequest struct {
		Vectors vectorsConfig `json:"vectors"`
	}

	payload := collectionRequest{Vectors: vectorsConfig{Size: vectorSize, Distance: "Cosine"}}
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal create request for %s: %w", collection, err)
	}

	createReq, err := newHTTPRequest(ctx, "PUT", url, data)
	if err != nil {
		return fmt.Errorf("build create request for %s: %w", collection, err)
	}
	createResp, err := httpClient.Do(createReq)
	if err != nil {
		return fmt.Errorf("create collection %s: %w", collection, err)
	}
	createBody, _ := io.ReadAll(createResp.Body)
	createResp.Body.Close()

	if createResp.StatusCode >= 400 {
		return fmt.Errorf("create collection %s: unexpected status %d %s", collection, createResp.StatusCode, strings.TrimSpace(string(createBody)))
	}
	return nil
}

// The store package hardcodes collection names, so we duplicate the minimal
// upsert logic here targeting configured collections. This avoids modifying
// internal/store/qdrant.go.

func (t *testStoreClient) UpsertSource(ctx context.Context, src source.Source, vector []float64) error {
	payload := mustPayload(src)
	payload["entity_type"] = "source"
	return t.upsert(ctx, testCollectionSources, pointIDForKey("source:"+src.ID), vector, payload)
}

func (t *testStoreClient) UpsertChunk(ctx context.Context, sourceID string, ch chunker.Chunk, vector []float64) error {
	payload := mustPayload(ch)
	payload["entity_type"] = "chunk"
	payload["source_id"] = sourceID
	return t.upsert(ctx, testCollectionChunks, pointIDForKey(fmt.Sprintf("%s_chunk_%04d", sourceID, ch.Index)), vector, payload)
}

func (t *testStoreClient) UpsertClaimPayload(ctx context.Context, claimID string, payload map[string]interface{}, vector []float64) error {
	if payload == nil {
		payload = map[string]interface{}{}
	}
	payload["entity_type"] = "claim"
	return t.upsert(ctx, testCollectionClaims, pointIDForKey("claim:"+claimID), vector, payload)
}

func (t *testStoreClient) upsert(ctx context.Context, collection string, id uint64, vector []float64, payload map[string]interface{}) error {
	// Direct HTTP upsert to Qdrant, same as store.Client but targeting our collection.
	type point struct {
		ID      uint64                 `json:"id"`
		Vector  []float64              `json:"vector"`
		Payload map[string]interface{} `json:"payload"`
	}
	type body struct {
		Points []point `json:"points"`
	}

	b := body{Points: []point{{ID: id, Vector: vector, Payload: payload}}}
	data, err := json.Marshal(b)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	qdrantURL := strings.TrimRight(envOr("QDRANT_URL", "http://localhost:6333"), "/")
	url := fmt.Sprintf("%s/collections/%s/points?wait=true", qdrantURL, collection)

	req, err := newHTTPRequest(ctx, "PUT", url, data)
	if err != nil {
		return err
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("upsert http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody := make([]byte, 512)
		n, _ := resp.Body.Read(respBody)
		return fmt.Errorf("upsert %s: %d %s", collection, resp.StatusCode, string(respBody[:n]))
	}
	return nil
}

// ---------------------------------------------------------------------------
// Helpers (duplicated from cmd/mb/main.go to avoid modifying it)
// ---------------------------------------------------------------------------

var httpClient = java_net_http_client()

func java_net_http_client() http.Client {
	return http.Client{Timeout: 30 * time.Second}
}

func newHTTPRequest(ctx context.Context, method, url string, body []byte) (*http.Request, error) {
	var r io.Reader
	if body != nil {
		r = strings.NewReader(string(body))
	}
	req, err := http.NewRequestWithContext(ctx, method, url, r)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	apiKey := os.Getenv("QDRANT_API_KEY")
	if apiKey != "" {
		req.Header.Set("api-key", apiKey)
	}
	return req, nil
}

func pointIDForKey(key string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(key))
	return h.Sum64()
}

func mustPayload(v any) map[string]interface{} {
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("marshal payload: %v", err))
	}
	var out map[string]interface{}
	if err := json.Unmarshal(b, &out); err != nil {
		panic(fmt.Sprintf("unmarshal payload: %v", err))
	}
	return out
}

func extractText(path string) (string, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".pdf":
		if _, err := exec.LookPath("pdftotext"); err != nil {
			return "", fmt.Errorf("pdftotext not found on PATH; install poppler-utils: %w", err)
		}
		cmd := exec.Command("pdftotext", "-layout", path, "-")
		var stdout, stderr strings.Builder
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			return "", fmt.Errorf("pdftotext: %w (stderr: %s)", err, stderr.String())
		}
		return stdout.String(), nil
	case ".txt", ".md", "":
		b, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		return string(b), nil
	default:
		return "", fmt.Errorf("unsupported extension %q", ext)
	}
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
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
