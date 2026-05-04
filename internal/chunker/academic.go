// academic.go — section-aware chunker for academic textbooks.
//
// SplitAcademic detects numbered section headers (e.g., "1.2.3 Title") and
// "Chapter N" headers. It filters out TOC lines, page running headers, and
// keeps equations grouped with their surrounding prose.
//
// Same Chunk output as Split(), so the rest of the pipeline works unchanged.
package chunker

import (
	"regexp"
	"strings"
)

// AcademicOptions tunes the academic chunker.
type AcademicOptions struct {
	TargetTokens   int
	MaxTokens      int
	FallbackHeader string
	SkipSections   []string // section title substrings to skip entirely
}

// DefaultAcademicOptions returns sensible defaults for textbook ingestion.
func DefaultAcademicOptions() AcademicOptions {
	return AcademicOptions{
		TargetTokens:   600,
		MaxTokens:      1200,
		FallbackHeader: "Front Matter",
	}
}

var (
	// academicSectionRE matches "1.2.3 Compton Effect" style headers.
	academicSectionRE = regexp.MustCompile(`^(\d+(?:\.\d+)+)\s+([A-Z][A-Za-z].{0,120}?)$`)

	// academicChapterRE matches "Chapter N" on its own line (real chapter starts).
	academicChapterRE = regexp.MustCompile(`(?i)^Chapter\s+(\d+)\s*$`)

	// tocSpacedDotsRE matches TOC lines with spaced dotleaders: ". . . . 42"
	tocSpacedDotsRE = regexp.MustCompile(`\.\s+\.(\s+\.)+\s*\d+\s*$`)

	// runningHeaderRE matches page headers like "472 CHAPTER 8. IDENTICAL PARTICLES".
	runningHeaderRE = regexp.MustCompile(`^\s*\d+\s+CHAPTER\s+\d+\.\s+[A-Z\s\-]+\s*$`)

	// romanPageRE matches bare roman numeral page numbers.
	romanPageRE = regexp.MustCompile(`(?i)^[ivxlcdm]+$`)
)

// cleanLine strips form feeds and whitespace.
func cleanLine(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimLeft(s, "\f")
	return strings.TrimSpace(s)
}

// SplitAcademic splits academic/textbook text into chunks, respecting
// numbered section boundaries and filtering out TOC/page-header noise.
func SplitAcademic(text string, opts AcademicOptions) []Chunk {
	def := DefaultAcademicOptions()
	if opts.TargetTokens == 0 {
		opts.TargetTokens = def.TargetTokens
	}
	if opts.MaxTokens == 0 {
		opts.MaxTokens = def.MaxTokens
	}
	if strings.TrimSpace(opts.FallbackHeader) == "" {
		opts.FallbackHeader = def.FallbackHeader
	}

	text = strings.ReplaceAll(text, "\r\n", "\n")
	lines := strings.Split(text, "\n")
	skipSet := buildAcademicSkipSet(opts.SkipSections)

	var chunks []Chunk
	var cur strings.Builder
	var curTokens int
	currentSection := opts.FallbackHeader
	curSection := opts.FallbackHeader
	idx := 0

	flush := func() {
		if cur.Len() == 0 {
			return
		}
		t := strings.TrimSpace(cur.String())
		if estimateTokens(t) < 10 {
			cur.Reset()
			curTokens = 0
			return
		}
		chunks = append(chunks, Chunk{
			Index:    idx,
			Chapter:  curSection,
			Text:     t,
			TokenEst: estimateTokens(t),
		})
		idx++
		cur.Reset()
		curTokens = 0
	}

	appendPara := func(paraText string) {
		pTokens := estimateTokens(paraText)

		// Oversized single paragraph: own chunk.
		if pTokens > opts.MaxTokens && cur.Len() == 0 {
			curSection = currentSection
			cur.WriteString(paraText)
			curTokens = pTokens
			flush()
			return
		}

		// Would exceed target? Flush first.
		if curTokens+pTokens > opts.TargetTokens && cur.Len() > 0 {
			flush()
		}

		if cur.Len() == 0 {
			curSection = currentSection
		} else {
			cur.WriteString("\n\n")
		}
		cur.WriteString(paraText)
		curTokens += pTokens
	}

	i := 0
	for i < len(lines) {
		line := lines[i]
		clean := cleanLine(line)

		// Skip empty lines.
		if clean == "" {
			i++
			continue
		}

		// Skip TOC lines (spaced dotleaders).
		if tocSpacedDotsRE.MatchString(line) {
			i++
			continue
		}

		// Skip page running headers.
		if runningHeaderRE.MatchString(line) || runningHeaderRE.MatchString(clean) {
			i++
			continue
		}

		// Skip bare roman numerals.
		if romanPageRE.MatchString(clean) && len(clean) < 8 {
			i++
			continue
		}

		// Detect "Chapter N" start.
		if m := academicChapterRE.FindStringSubmatch(clean); m != nil {
			flush()
			// Peek for title on next non-empty line.
			title := ""
			j := i + 1
			for j < len(lines) && strings.TrimSpace(lines[j]) == "" {
				j++
			}
			if j < len(lines) {
				peek := cleanLine(lines[j])
				if peek != "" && !academicSectionRE.MatchString(peek) && len(peek) < 80 {
					title = peek
				}
			}
			if title != "" {
				currentSection = "Chapter " + m[1] + ": " + title
			} else {
				currentSection = "Chapter " + m[1]
			}
			curSection = currentSection
			i++
			continue
		}

		// Detect numbered section header.
		if m := academicSectionRE.FindStringSubmatch(clean); m != nil && len(clean) < 120 && !tocSpacedDotsRE.MatchString(line) {
			sectionTitle := strings.TrimSpace(m[2])

			// Check skip list.
			if shouldSkipAcademicSection(sectionTitle, skipSet) {
				i++
				for i < len(lines) {
					peek := cleanLine(lines[i])
					if academicSectionRE.MatchString(peek) || academicChapterRE.MatchString(peek) {
						break
					}
					i++
				}
				continue
			}

			flush()
			currentSection = m[1] + " " + sectionTitle
			curSection = currentSection
			i++
			continue
		}

		// Accumulate paragraph: consecutive non-empty, non-header lines.
		var para strings.Builder
		for i < len(lines) {
			cl := cleanLine(lines[i])
			if cl == "" {
				break
			}
			// Stop at headers.
			if academicChapterRE.MatchString(cl) {
				break
			}
			if academicSectionRE.MatchString(cl) && len(cl) < 120 && !tocSpacedDotsRE.MatchString(lines[i]) {
				break
			}
			// Skip inline page headers and TOC lines.
			if runningHeaderRE.MatchString(lines[i]) || tocSpacedDotsRE.MatchString(lines[i]) {
				i++
				continue
			}
			if para.Len() > 0 {
				para.WriteString("\n")
			}
			para.WriteString(cl)
			i++
		}

		if para.Len() > 0 {
			appendPara(para.String())
		}
	}
	flush()

	return chunks
}

func buildAcademicSkipSet(sections []string) map[string]bool {
	m := make(map[string]bool, len(sections))
	for _, s := range sections {
		m[strings.ToLower(strings.TrimSpace(s))] = true
	}
	return m
}

func shouldSkipAcademicSection(title string, skipSet map[string]bool) bool {
	lower := strings.ToLower(title)
	for skip := range skipSet {
		if strings.Contains(lower, skip) {
			return true
		}
	}
	return false
}
