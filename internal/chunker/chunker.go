// Package chunker splits source text into scene/chapter-boundary chunks.
//
// Wave 1 strategy (deliberately simple):
//   - Detect chapter boundaries using a heading regex.
//   - Within a chapter, group paragraphs until the chunk reaches a token target.
//   - Never split a paragraph. Snap to paragraph boundaries always.
//
// This is NOT the final chunker. ARCHITECTURE.md Stage 1 calls for scene-level
// chunking with POV-shift detection and scene_type classification. That's Wave 2+.
// For now: chapter + size-bounded paragraph groups is enough to validate the
// extraction pipeline end-to-end.
package chunker

import (
	"regexp"
	"strings"
)

// Chunk is one contiguous passage from the source, ready for claim extraction.
type Chunk struct {
	Index    int    `json:"index"`           // 0-based, in source order
	Chapter  string `json:"chapter"`         // e.g. "Chapter One" or "" if pre-TOC
	Text     string `json:"text"`
	TokenEst int    `json:"token_est"`       // rough token estimate (chars/4)
}

// chapterHeading matches things like:
//   "Chapter One"
//   "Chapter 1:"
//   "CHAPTER TWELVE"
//   "Chapter Twelve"
// It is intentionally permissive. False positives here are fine; chunks merely
// get labeled with a wrong chapter name, which the cartographer can fix later.
var chapterHeading = regexp.MustCompile(`(?m)^\s*(Chapter\s+[A-Za-z0-9-]+|CHAPTER\s+[A-Z0-9-]+)\b.*$`)

// Options tune the chunker. Defaults target ~600 tokens per chunk.
type Options struct {
	TargetTokens int // soft target; chunks won't split mid-paragraph to hit it
	MaxTokens    int // hard cap; a single paragraph larger than this becomes its own chunk
}

// DefaultOptions returns sensible Wave 1 defaults.
func DefaultOptions() Options {
	return Options{
		TargetTokens: 600,
		MaxTokens:    1200,
	}
}

// Chunk splits raw text into chunks. Paragraphs are defined by blank lines.
func Chunk(text string, opts Options) []Chunk {
	if opts.TargetTokens == 0 {
		opts = DefaultOptions()
	}

	// Normalize line endings and collapse runs of blank lines to exactly one.
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = regexp.MustCompile(`\n{3,}`).ReplaceAllString(text, "\n\n")

	paragraphs := splitParagraphs(text)

	var chunks []Chunk
	var cur strings.Builder
	var curTokens int
	currentChapter := ""
	chunkChapter := ""
	idx := 0

	flush := func() {
		if cur.Len() == 0 {
			return
		}
		chunks = append(chunks, Chunk{
			Index:    idx,
			Chapter:  chunkChapter,
			Text:     strings.TrimSpace(cur.String()),
			TokenEst: curTokens,
		})
		idx++
		cur.Reset()
		curTokens = 0
	}

	for _, p := range paragraphs {
		// Chapter detection: if the paragraph IS a chapter heading, update the
		// currentChapter and start a fresh chunk. The heading itself is not
		// included in the chunk body (it's metadata).
		if match := chapterHeading.FindString(p); match != "" && len(p) < 120 {
			flush()
			currentChapter = strings.TrimSpace(match)
			chunkChapter = currentChapter
			continue
		}

		pTokens := estimateTokens(p)

		// A single oversized paragraph becomes its own chunk.
		if pTokens > opts.MaxTokens && cur.Len() == 0 {
			chunkChapter = currentChapter
			cur.WriteString(p)
			curTokens = pTokens
			flush()
			continue
		}

		// If adding this paragraph would exceed the target AND we already have
		// content, flush first.
		if curTokens+pTokens > opts.TargetTokens && cur.Len() > 0 {
			flush()
		}

		if cur.Len() == 0 {
			chunkChapter = currentChapter
		} else {
			cur.WriteString("\n\n")
		}
		cur.WriteString(p)
		curTokens += pTokens
	}
	flush()

	return chunks
}

// splitParagraphs splits on blank lines, trimming each paragraph.
func splitParagraphs(text string) []string {
	raw := strings.Split(text, "\n\n")
	out := make([]string, 0, len(raw))
	for _, p := range raw {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// estimateTokens is a rough char/4 heuristic. Good enough for budgeting.
// We will swap to a real tokenizer when it matters. It does not matter yet.
func estimateTokens(s string) int {
	return len(s) / 4
}
