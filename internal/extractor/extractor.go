// Package extractor runs claim extraction over chunks via an LLM.
//
// For Wave 1: one LLM call per chunk. No batching, no parallelism, no retries
// beyond simple JSON-parse recovery. Get the loop working end-to-end first;
// optimize after we can read the output.
package extractor

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/meistro57/meta-bridge/internal/chunker"
	"github.com/meistro57/meta-bridge/internal/claim"
	"github.com/meistro57/meta-bridge/internal/llm"
)

// Model is the OpenRouter model used for extraction.
// Gemma 4 31B Instruct; swap via env var if needed (see main.go).
const DefaultModel = "google/gemma-4-31b-it"

// Extractor extracts claims from chunks.
type Extractor struct {
	client *llm.Client
	model  string
}

// New constructs an Extractor.
func New(client *llm.Client, model string) *Extractor {
	if model == "" {
		model = DefaultModel
	}
	return &Extractor{client: client, model: model}
}

// rawClaim is the JSON shape the LLM returns per claim.
// Mirrors claim.Claim but flatter and extraction-time only.
type rawClaim struct {
	CanonicalStatement string   `json:"canonical_statement"`
	SurfaceQuote       string   `json:"surface_quote"`
	ClaimType          string   `json:"claim_type"` // "stated" | "dramatized" | "implied"
	Confidence         float64  `json:"confidence"`
	Tags               []string `json:"tags"`
}

type rawResponse struct {
	Claims []rawClaim `json:"claims"`
}

// systemPrompt defines what counts as a claim. This is the heart of the extractor.
// Tune carefully. First principles:
//   - Only metaphysical / cosmological / consciousness-related propositions.
//   - Plot mechanics, character feelings, scenery: NOT claims.
//   - Abstract the phrasing; surface quote stays short (<15 words).
//   - Mark HOW the claim is expressed (stated/dramatized/implied).
const systemPrompt = `You extract atomic metaphysical claims from passages of consciousness literature (channeled texts, regression transcripts, esoteric fiction, doctrinal works).

Rules:
1. Extract ONLY propositions about the nature of consciousness, reality, identity, time, existence, the soul, or the structure of the universe. Ignore plot, scenery, character emotions, and biographical detail.
2. Rephrase each claim in abstract doctrinal language suitable for cross-source comparison. Do NOT quote characters; state the proposition as a general claim.
3. Canonical statements must be a direct abstract reflection of the provided surface quote. Avoid injecting external metaphysical concepts not present in the immediate context.
4. Tag how the claim is expressed:
   - "stated": the text directly asserts the proposition (doctrine, exposition).
   - "dramatized": the text demonstrates it through narrative action or dialogue rather than asserting it.
   - "implied": the text logically presupposes it without stating it.
5. surface_quote: 15 words MAX, a brief paraphrase or short direct excerpt showing where the claim lives in the passage. Prefer paraphrase.
6. Confidence (0.0-1.0): how certain you are this is a genuine extraction, not a stretch.
7. Tags: 1-4 short lowercase tags for the conceptual domain (e.g. "time", "identity", "substrate", "afterlife", "perception").

If the passage contains NO metaphysical claims, return {"claims": []}.

Return ONLY a JSON object of shape: {"claims": [{"canonical_statement": "...", "surface_quote": "...", "claim_type": "stated", "confidence": 0.9, "tags": ["..."]}]}
No prose, no markdown, no explanation. Just the JSON object.`

// ExtractChunk runs the LLM over one chunk and returns the extracted claims,
// already shaped as claim.Claim objects with an Attribution pointing back at
// the chunk.
//
// claimIDFn mints an ID for each new claim (usually a counter closure from
// the caller).
func (e *Extractor) ExtractChunk(
	ctx context.Context,
	sourceID string,
	ch chunker.Chunk,
	claimIDFn func() string,
) ([]claim.Claim, error) {
	userPrompt := fmt.Sprintf("Chapter: %s\n\nPassage:\n%s", ch.Chapter, ch.Text)

	req := llm.Request{
		Model:       e.model,
		Temperature: 0.2, // low: we want consistent extractions, not creative ones
		Messages: []llm.Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
		ResponseFormat: &llm.ResponseFormat{Type: "json_object"},
	}

	raw, err := e.client.Complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("llm call: %w", err)
	}

	// Some models wrap JSON in ```json fences despite response_format. Strip.
	raw = stripFences(raw)

	parsed, err := parseRawResponse(raw)
	if err != nil {
		return nil, fmt.Errorf("parse claims JSON: %w\nraw response: %s", err, raw)
	}

	out := make([]claim.Claim, 0, len(parsed.Claims))
	for _, rc := range parsed.Claims {
		// Minimal sanity filter. Bad extractions are fine to include at Wave 1
		// since we're eyeballing output; cartographer's desk will reject later.
		if strings.TrimSpace(rc.CanonicalStatement) == "" {
			continue
		}
		ct := claim.ClaimType(rc.ClaimType)
		if ct != claim.ClaimStated && ct != claim.ClaimDramatized && ct != claim.ClaimImplied {
			ct = claim.ClaimStated // conservative default
		}
		c := claim.Claim{
			ID:                 claimIDFn(),
			CanonicalStatement: strings.TrimSpace(rc.CanonicalStatement),
			Attributions: []claim.Attribution{
				{
					SourceID:     sourceID,
					ChunkIndex:   ch.Index,
					Chapter:      ch.Chapter,
					SurfaceQuote: strings.TrimSpace(rc.SurfaceQuote),
					ClaimType:    ct,
					Confidence:   rc.Confidence,
				},
			},
			Tags:            rc.Tags,
			EditorialStatus: "auto",
		}
		out = append(out, c)
	}

	return out, nil
}

// stripFences removes ```json ... ``` wrappers if present.
func stripFences(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		// Drop first line (```json or ```)
		if nl := strings.Index(s, "\n"); nl != -1 {
			s = s[nl+1:]
		}
		// Drop trailing ```
		s = strings.TrimSuffix(strings.TrimSpace(s), "```")
	}
	return strings.TrimSpace(s)
}

var trailingCommaRE = regexp.MustCompile(`,\s*([\]}])`)
var misplacedObjectCloseRE = regexp.MustCompile(`}\s*,\s*"(claim_type|confidence|tags|surface_quote|canonical_statement)"\s*:`)

func parseRawResponse(raw string) (rawResponse, error) {
	var parsed rawResponse
	if err := json.Unmarshal([]byte(raw), &parsed); err == nil {
		return parsed, nil
	}

	repaired := repairJSON(raw)
	if err := json.Unmarshal([]byte(repaired), &parsed); err != nil {
		return rawResponse{}, err
	}
	return parsed, nil
}

func repairJSON(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}

	if i := strings.Index(s, "{"); i >= 0 {
		s = s[i:]
	}

	if cut := truncateAtTopLevelObjectEnd(s); cut != "" {
		s = cut
	}

	s = strings.TrimSpace(s)
	for strings.HasSuffix(s, ",") || strings.HasSuffix(s, ",\"") {
		s = strings.TrimSuffix(s, "\"")
		s = strings.TrimSuffix(s, ",")
		s = strings.TrimSpace(s)
	}

	s = trailingCommaRE.ReplaceAllString(s, "$1")
	s = misplacedObjectCloseRE.ReplaceAllString(s, `, "$1":`)
	if json.Valid([]byte(s)) {
		return s
	}

	return closeOpenDelimiters(s)
}

func truncateAtTopLevelObjectEnd(s string) string {
	inString := false
	escaped := false
	started := false
	depthObj := 0
	depthArr := 0

	for i := 0; i < len(s); i++ {
		ch := s[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}

		switch ch {
		case '"':
			inString = true
		case '{':
			started = true
			depthObj++
		case '}':
			if depthObj > 0 {
				depthObj--
			}
		case '[':
			if started {
				depthArr++
			}
		case ']':
			if depthArr > 0 {
				depthArr--
			}
		}

		if started && depthObj == 0 && depthArr == 0 {
			return s[:i+1]
		}
	}

	return ""
}

func closeOpenDelimiters(s string) string {
	stack := make([]byte, 0, 32)
	inString := false
	escaped := false

	for i := 0; i < len(s); i++ {
		ch := s[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}

		switch ch {
		case '"':
			inString = true
		case '{', '[':
			stack = append(stack, ch)
		case '}':
			if len(stack) > 0 && stack[len(stack)-1] == '{' {
				stack = stack[:len(stack)-1]
			}
		case ']':
			if len(stack) > 0 && stack[len(stack)-1] == '[' {
				stack = stack[:len(stack)-1]
			}
		}
	}

	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, ",")
	if inString {
		s += "\""
	}
	for i := len(stack) - 1; i >= 0; i-- {
		if stack[i] == '{' {
			s += "}"
		} else {
			s += "]"
		}
	}
	s = trailingCommaRE.ReplaceAllString(s, "$1")
	return strings.TrimSpace(s)
}
