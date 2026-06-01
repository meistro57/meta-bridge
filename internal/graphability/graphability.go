// Package graphability scores meta-bridge chunks against the Meistro Brain
// Graphability Index — a consciousness-literature analogue of the Proxy-Pointer
// "Graphability Indexing" technique for Knowledge Graph ingestion cost reduction.
//
// Score tiers (mirrors the article):
//
//	VeryHigh / High / Medium  → send to LLM for claim extraction (default)
//	Low / VeryLow             → skip LLM call, tag payload as graphability_skip=true
//	Unknown (coverage gap)    → mandatory scan + flag to misfit_reports
//
// The index is loaded once from a JSON file (default: ./graphability_index.json).
// Override path via MB_GRAPHABILITY_INDEX env var.
//
// Matching is fuzzy: the chunk's Chapter string is lower-cased and checked for
// substring containment against each index entry. Corpus anchors are checked
// first and always return VeryHigh when matched.
package graphability

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// Score represents the graphability tier of a chunk.
type Score int

const (
	Unknown  Score = iota // not in index — coverage gap, mandatory scan
	VeryLow               // boilerplate noise — skip
	Low                   // administrative — skip
	Medium                // worth scanning
	High                  // high relational density
	VeryHigh              // ontological core — always scan
)

// String returns the human-readable tier label.
func (s Score) String() string {
	switch s {
	case VeryHigh:
		return "very_high"
	case High:
		return "high"
	case Medium:
		return "medium"
	case Low:
		return "low"
	case VeryLow:
		return "very_low"
	default:
		return "unknown"
	}
}

// ShouldExtract returns true when the score warrants an LLM extraction call.
// Low and VeryLow are skipped. Unknown is always scanned (coverage gap policy).
func (s Score) ShouldExtract() bool {
	switch s {
	case Low, VeryLow:
		return false
	default:
		return true // VeryHigh, High, Medium, Unknown (gap — must scan)
	}
}

// IsGap returns true when the chapter was not found in the index at all.
func (s Score) IsGap() bool { return s == Unknown }

// index holds the loaded graphability index.
type index struct {
	VeryHigh       []string       `json:"very_high"`
	High           []string       `json:"high"`
	Medium         []string       `json:"medium"`
	Low            []string       `json:"low"`
	VeryLow        []string       `json:"very_low"`
	CorpusAnchors  corpusAnchors  `json:"corpus_anchors"`
}

type corpusAnchors struct {
	Patterns []string `json:"patterns"`
}

// Scorer scores chunks by chapter label against the loaded index.
type Scorer struct {
	idx index
}

// DefaultIndexPath is the default graphability index file location.
const DefaultIndexPath = "./graphability_index.json"

// New loads the graphability index from path.
// Resolution order:
//  1. Explicit path argument
//  2. MB_GRAPHABILITY_INDEX env var
//  3. Same directory as the running executable
//  4. Current working directory (fallback)
//
// Returns a no-op scorer (all Unknown) on load failure so the pipeline degrades
// gracefully rather than halting.
func New(path string) (*Scorer, error) {
	if path == "" {
		path = os.Getenv("MB_GRAPHABILITY_INDEX")
	}
	if path == "" {
		// Try next to the executable first.
		if exe, err := os.Executable(); err == nil {
			candidate := filepath.Join(filepath.Dir(exe), DefaultIndexPath)
			if _, err := os.Stat(candidate); err == nil {
				path = candidate
			}
		}
	}
	if path == "" {
		path = DefaultIndexPath
	}
	b, err := os.ReadFile(path)
	if err != nil {
		// Graceful degradation: return scorer that marks everything Unknown
		// (mandatory scan — same as current behaviour before this feature).
		return &Scorer{}, err
	}
	var idx index
	if err := json.Unmarshal(b, &idx); err != nil {
		return &Scorer{}, err
	}
	return &Scorer{idx: idx}, nil
}

// Score returns the combined graphability score for a chunk.
// It runs both label matching (chapter header) and content analysis (chunk text),
// returning the higher of the two scores. This means a book with no structured
// headers (all "General Content") still scores correctly based on what the
// text actually contains.
//
// Pass an empty text string to score by label only.
func (s *Scorer) Score(chapter, text string) Score {
	labelScore := s.ScoreLabel(chapter)
	textScore := ScoreText(text)
	if textScore > labelScore {
		return textScore
	}
	return labelScore
}

// ScoreLabel returns the graphability score for a chunk chapter label only.
// Matching is case-insensitive substring: the chapter string is searched for
// any index term as a substring. Corpus anchors are checked first.
func (s *Scorer) ScoreLabel(chapter string) Score {
	if chapter == "" {
		return Unknown
	}
	lower := strings.ToLower(strings.TrimSpace(chapter))

	// Corpus anchors → always VeryHigh.
	for _, pattern := range s.idx.CorpusAnchors.Patterns {
		if strings.Contains(lower, strings.ToLower(pattern)) {
			return VeryHigh
		}
	}

	// Walk tiers from highest to lowest — first match wins.
	if matchAny(lower, s.idx.VeryHigh) {
		return VeryHigh
	}
	if matchAny(lower, s.idx.High) {
		return High
	}
	if matchAny(lower, s.idx.Medium) {
		return Medium
	}
	if matchAny(lower, s.idx.Low) {
		return Low
	}
	if matchAny(lower, s.idx.VeryLow) {
		return VeryLow
	}

	return Unknown
}

// matchAny returns true if subject contains any of the terms as a substring.
func matchAny(subject string, terms []string) bool {
	for _, t := range terms {
		if strings.Contains(subject, strings.ToLower(t)) {
			return true
		}
	}
	return false
}
