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
	Index     int    `json:"index"`      // 0-based, in source order
	Chapter   string `json:"chapter"`    // e.g. "Chapter One" or "" if pre-TOC
	BookTitle string `json:"book_title"` // human-readable title of the source
	SourceID  string `json:"source_id"`  // snake_case stable identifier
	Page      int    `json:"page"`       // page number if known, else 0
	Text      string `json:"text"`
	TokenEst  int    `json:"token_est"` // rough token estimate (chars/4)
}

// Options tune the chunker. Defaults target ~600 tokens per chunk.
type Options struct {
	TargetTokens   int    // soft target; chunks won't split mid-paragraph to hit it
	MaxTokens      int    // hard cap; a single paragraph larger than this becomes its own chunk
	HeaderPattern  string // optional regex for section headers; if empty, no header detection
	FallbackHeader string // default label before first matched header
	SourceTitle    string // optional: filters running page footers that echo the book title
}

// DefaultOptions returns sensible Wave 1 defaults.
func DefaultOptions() Options {
	return Options{
		TargetTokens:   600,
		MaxTokens:      1200,
		HeaderPattern:  `(?im)^\s*((chapter|session)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+|[ivxlcdm]+)(\b.*)?|part\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+|[ivxlcdm]+)(\b.*)?)\s*$`,
		FallbackHeader: "General Content",
	}
}

// Split splits raw text into chunks. Paragraphs are defined by blank lines.
func Split(text string, opts Options) []Chunk {
	def := DefaultOptions()
	if opts.TargetTokens == 0 {
		opts.TargetTokens = def.TargetTokens
	}
	if opts.MaxTokens == 0 {
		opts.MaxTokens = def.MaxTokens
	}
	if strings.TrimSpace(opts.FallbackHeader) == "" {
		opts.FallbackHeader = def.FallbackHeader
	}
	if strings.TrimSpace(opts.HeaderPattern) == "" {
		opts.HeaderPattern = def.HeaderPattern
	}

	var headerRE *regexp.Regexp
	if strings.TrimSpace(opts.HeaderPattern) != "" {
		compiled, err := regexp.Compile(opts.HeaderPattern)
		if err == nil {
			headerRE = compiled
		}
	}

	// Build a set of uppercase title fragments to suppress as false chapter
	// headers (e.g. "THE CUSTODIANS" appearing as a running page footer).
	titleFrags := buildTitleFragments(opts.SourceTitle)
	isFooterEcho := func(p string) bool {
		return titleFrags[strings.ToUpper(strings.TrimSpace(p))]
	}

	// Normalize line endings, form feeds, and runs of blank lines.
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\f", "\n\n") // pdftotext page breaks -> paragraph breaks
	text = regexp.MustCompile(`\n{3,}`).ReplaceAllString(text, "\n\n")

	paragraphs := splitParagraphs(text)

	var chunks []Chunk
	var cur strings.Builder
	var curTokens int
	currentChapter := opts.FallbackHeader
	chunkChapter := opts.FallbackHeader
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

	// tokensSinceHeader tracks content accumulated since the last header fired.
	// If a new header arrives before minContentBetweenHeaders tokens have been
	// seen, it's almost certainly a table-of-contents entry, not a real chapter
	// boundary — skip the label update to prevent TOC poisoning.
	const minContentBetweenHeaders = 80 // ~320 chars; enough to confirm real prose
	tokensSinceHeader := minContentBetweenHeaders // allow first header unconditionally

	acceptHeader := func(label string) bool {
		if tokensSinceHeader >= minContentBetweenHeaders {
			tokensSinceHeader = 0
			return true
		}
		// Rapid-fire header with no real content since the last one — TOC entry.
		return false
	}

	for i, p := range paragraphs {
		// -- Regex header detection -------------------------------------------
		// Matches "Chapter 1", "Session IV", "Part Two", etc.
		if headerRE != nil {
			trimmed := strings.TrimLeft(p, " \t")
			if loc := headerRE.FindStringIndex(trimmed); loc != nil && loc[0] == 0 && len(trimmed) <= 120 {
				proposed := strings.TrimSpace(trimmed[loc[0]:loc[1]])
				if acceptHeader(proposed) {
					flush()
					currentChapter = proposed
					chunkChapter = currentChapter
				}
				continue
			}
		}

		// -- Split-line chapter detection -------------------------------------
		// Dolores Cannon PDFs (and others) often render the section keyword
		// alone on one paragraph and the title on the next, e.g.:
		//   "Chapter"
		//   "Message from a Guide"
		// Detect that pattern and combine into a single chapter label.
		if isSectionKeywordOnly(p) && i+1 < len(paragraphs) {
			nextP := strings.TrimSpace(paragraphs[i+1])
			if isTitleLine(nextP) {
				proposed := strings.TrimSpace(p) + " — " + nextP
				if acceptHeader(proposed) {
					flush()
					currentChapter = proposed
					chunkChapter = currentChapter
				}
				continue
			}
		}

		// Skip paragraph already consumed as the title in split-line detection.
		if i > 0 && isSectionKeywordOnly(paragraphs[i-1]) && isTitleLine(p) {
			continue
		}

		// -- All-caps fallback header -----------------------------------------
		// Common in older public-domain texts (Project Gutenberg, sacred-texts).
		// Guard: skip if it matches a known title fragment (running page footer)
		// so that "THE CUSTODIANS" on every page footer doesn't hijack labels.
		if isAllCapsHeader(p) && !isFooterEcho(p) {
			proposed := strings.TrimSpace(p)
			if acceptHeader(proposed) {
				flush()
				currentChapter = proposed
				chunkChapter = currentChapter
			}
			continue
		}

		pTokens := estimateTokens(p)

		// A single oversized paragraph becomes its own chunk.
		if pTokens > opts.MaxTokens && cur.Len() == 0 {
			chunkChapter = currentChapter
			cur.WriteString(p)
			curTokens = pTokens
			tokensSinceHeader += pTokens
			flush()
			continue
		}

		// Flush if adding this paragraph would exceed the target.
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
		tokensSinceHeader += pTokens
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

// estimateTokens is a rough char/4 heuristic.
func estimateTokens(s string) int {
	return len(s) / 4
}

// isSectionKeywordOnly returns true if p is solely a section keyword like
// "Chapter", "Session", or "Part" — the split-line pattern common in
// Dolores Cannon PDFs where keyword and title land in separate paragraphs.
func isSectionKeywordOnly(p string) bool {
	trimmed := strings.TrimSpace(p)
	if strings.ContainsRune(trimmed, '\n') {
		return false
	}
	lower := strings.ToLower(trimmed)
	return lower == "chapter" || lower == "session" || lower == "part" ||
		lower == "section" || lower == "book" || lower == "appendix"
}

// isTitleLine returns true if p looks like a chapter title:
// single line, reasonable length, not a bare page number or footer.
func isTitleLine(p string) bool {
	trimmed := strings.TrimSpace(p)
	if strings.ContainsRune(trimmed, '\n') {
		return false
	}
	n := len(trimmed)
	if n < 3 || n > 100 {
		return false
	}
	// Reject bare page numbers / roman numerals.
	if regexp.MustCompile(`^[\divxlcdmIVXLCDM]+$`).MatchString(trimmed) {
		return false
	}
	// Reject footer patterns like "Conversations with Nostradamus (VOL. I)".
	if regexp.MustCompile(`(?i)\(vol\.?\s*[ivx\d]+\)`).MatchString(trimmed) {
		return false
	}
	return true
}

// isAllCapsHeader reports whether p looks like an all-caps section title.
// Common in older public-domain texts that don't use "Chapter N" formatting.
func isAllCapsHeader(p string) bool {
	trimmed := strings.TrimSpace(p)
	if strings.ContainsRune(trimmed, '\n') {
		return false
	}
	n := len(trimmed)
	if n < 8 || n > 120 {
		return false
	}
	// Reject author-attribution lines like "ROBERT FROST (1875-1963)".
	// These end with a parenthesized year or year range.
	if regexp.MustCompile(`\(\d{3,4}[-` + "`" + `\-]?\d{0,4}\)\s*$`).MatchString(trimmed) {
		return false
	}
	letters := 0
	spaces := 0
	for _, r := range trimmed {
		switch {
		case r >= 'A' && r <= 'Z':
			letters++
		case r >= 'a' && r <= 'z':
			return false
		case r == ' ':
			spaces++
		case r >= '0' && r <= '9':
			// allow digits
		case strings.ContainsRune(`-'".,&()!?:;`, r):
			// allow common header punctuation
		default:
			return false
		}
	}
	if spaces < 1 || letters < 6 {
		return false
	}
	return true
}

// buildTitleFragments returns an uppercase set of strings derived from the
// source title that are likely to appear as running page-footer echoes.
// e.g. "The Custodians: Beyond Abduction" produces:
//
//	{"THE CUSTODIANS", "BEYOND ABDUCTION", "THE CUSTODIANS: BEYOND ABDUCTION"}
func buildTitleFragments(title string) map[string]bool {
	frags := make(map[string]bool)
	if strings.TrimSpace(title) == "" {
		return frags
	}
	frags[strings.ToUpper(strings.TrimSpace(title))] = true
	for _, sep := range []string{": ", " - ", " \u2014 ", " \u2013 "} {
		if idx := strings.Index(title, sep); idx != -1 {
			frags[strings.ToUpper(strings.TrimSpace(title[:idx]))] = true
			frags[strings.ToUpper(strings.TrimSpace(title[idx+len(sep):]))] = true
		}
	}
	return frags
}
