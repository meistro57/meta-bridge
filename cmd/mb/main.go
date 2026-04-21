// Command mb is the Meta Bridge CLI.
//
// Usage:
//   mb ingest <path-to-pdf-or-txt>
//
// Wave 1: extract text, chunk it, run claim extraction, write JSON to ./output/.
// No Qdrant, no dedup, no bridges. Just produce claims we can read and judge.
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

	"github.com/joho/godotenv"
	"github.com/meistro57/meta-bridge/internal/chunker"
	"github.com/meistro57/meta-bridge/internal/claim"
	"github.com/meistro57/meta-bridge/internal/extractor"
	"github.com/meistro57/meta-bridge/internal/llm"
	"github.com/meistro57/meta-bridge/internal/source"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	// Load .env if present; ignore error (env vars may be set directly).
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
  OPENROUTER_API_KEY     required
  MB_MODEL               override extraction model (default: deepseek/deepseek-r1)
  MB_MAX_CHUNKS          limit chunks processed (useful for dry runs)
  MB_OUTPUT_DIR          output directory (default: ./output)
  MB_SOURCE_ID           override auto-generated source ID
  MB_TITLE               override auto-generated title
  MB_AUTHOR              override auto-generated author`)
}

func cmdIngest(path string) error {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
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

	// --- 2. Chunk ---
	log.Printf("[2/4] Chunking")
	chunks := chunker.Chunk(text, chunker.DefaultOptions())
	log.Printf("      produced %d chunks", len(chunks))

	maxChunks := envInt("MB_MAX_CHUNKS", 0)
	if maxChunks > 0 && maxChunks < len(chunks) {
		log.Printf("      MB_MAX_CHUNKS=%d; limiting to first %d chunks", maxChunks, maxChunks)
		chunks = chunks[:maxChunks]
	}

	// --- 3. Build Source record ---
	baseName := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	src := source.NewSource(
		envOr("MB_SOURCE_ID", sanitizeID(baseName)),
		envOr("MB_TITLE", baseName),
		envOr("MB_AUTHOR", "Unknown"),
	)
	src.SourcePath = path
	src.ChunkCount = len(chunks)

	// --- 4. Extract claims chunk by chunk ---
	log.Printf("[3/4] Extracting claims (model=%s)", envOr("MB_MODEL", extractor.DefaultModel))
	client := llm.NewClient(apiKey)
	ex := extractor.New(client, envOr("MB_MODEL", ""))

	ctx := context.Background()
	var allClaims []claim.Claim

	counter := 0
	idFn := func() string {
		counter++
		return fmt.Sprintf("cl_%s_%04d", src.ID, counter)
	}

	start := time.Now()
	for i, ch := range chunks {
		log.Printf("      chunk %d/%d (chapter=%q, ~%d tokens)",
			i+1, len(chunks), ch.Chapter, ch.TokenEst)
		claims, err := ex.ExtractChunk(ctx, src.ID, ch, idFn)
		if err != nil {
			// Don't abort the whole run; log and move on. Bad chunks are a data
			// signal too (could indicate prompt issues or weird content).
			log.Printf("      ! extraction error on chunk %d: %v", i, err)
			continue
		}
		log.Printf("        -> %d claims", len(claims))
		allClaims = append(allClaims, claims...)
	}
	elapsed := time.Since(start)
	src.ClaimCount = len(allClaims)

	log.Printf("[4/4] Writing output")
	// --- Write Source JSON ---
	srcPath := filepath.Join(outputDir, src.ID+".source.json")
	if err := writeJSON(srcPath, src); err != nil {
		return fmt.Errorf("write source: %w", err)
	}

	// --- Write chunks JSON (useful for traceability from claim.chunk_index) ---
	chunksPath := filepath.Join(outputDir, src.ID+".chunks.json")
	if err := writeJSON(chunksPath, chunks); err != nil {
		return fmt.Errorf("write chunks: %w", err)
	}

	// --- Write claims JSON ---
	claimsPath := filepath.Join(outputDir, src.ID+".claims.json")
	if err := writeJSON(claimsPath, allClaims); err != nil {
		return fmt.Errorf("write claims: %w", err)
	}

	log.Printf("done in %s", elapsed.Round(time.Second))
	log.Printf("  %d chunks processed", len(chunks))
	log.Printf("  %d claims extracted", len(allClaims))
	log.Printf("  wrote:")
	log.Printf("    %s", srcPath)
	log.Printf("    %s", chunksPath)
	log.Printf("    %s", claimsPath)
	return nil
}

// extractText reads the source file, converting PDF to text if needed.
// PDFs: use pdftotext (poppler-utils) with layout preservation.
// Plain text: read directly.
func extractText(path string) (string, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".pdf":
		return runPdftotext(path)
	case ".txt", ".md", "":
		b, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		return string(b), nil
	default:
		return "", fmt.Errorf("unsupported extension %q (use .pdf or .txt)", ext)
	}
}

func runPdftotext(path string) (string, error) {
	if _, err := exec.LookPath("pdftotext"); err != nil {
		return "", fmt.Errorf("pdftotext not found on PATH; install poppler-utils: %w", err)
	}
	// -layout preserves spatial positioning; "-" writes to stdout.
	cmd := exec.Command("pdftotext", "-layout", path, "-")
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("pdftotext: %w (stderr: %s)", err, stderr.String())
	}
	return stdout.String(), nil
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
