// Package source defines the Source object: a single work ingested into Meta Bridge.
//
// This is a minimum scaffold. Fields map directly to ARCHITECTURE.md but many are
// optional/empty for Wave 1. The schema will evolve; keep this file small and boring.
package source

import "time"

// ChannelType classifies how the material was produced.
// See ARCHITECTURE.md for why this matters (independence scoring).
type ChannelType string

const (
	ChannelAuthored    ChannelType = "authored"    // ordinary human authorship
	ChannelChanneled   ChannelType = "channeled"   // claimed non-human source, trance-dictated (Seth, Ra)
	ChannelRegression  ChannelType = "regression"  // hypnotic regression transcripts (Cannon)
	ChannelDictated    ChannelType = "dictated"    // human author dictating to a scribe
	ChannelDramatized  ChannelType = "dramatized"  // fiction conveying doctrine (Oversoul Seven)
	ChannelDialogue    ChannelType = "dialogue"    // recorded dialogue / interview / workshop
)

// Source is a single work. One Source per book, transcript collection, or coherent corpus.
type Source struct {
	ID          string      `json:"id"`           // stable human-readable: "roberts_oversoul7"
	Title       string      `json:"title"`
	Author      string      `json:"author"`       // the human author/scribe
	Medium      string      `json:"medium"`       // the claimed non-human source, if any
	ChannelType ChannelType `json:"channel_type"`
	Tradition   string      `json:"tradition"`    // "Seth", "Ra/Law of One", "QHHT/Cannon", ...
	Year        int         `json:"year"`
	Region      string      `json:"region"`       // geographic origin of the material

	IngestedAt  time.Time `json:"ingested_at"`
	ChunkCount  int       `json:"chunk_count"`
	ClaimCount  int       `json:"claim_count"`

	// Path to the source file on disk. Not persisted long-term; useful during Wave 1.
	SourcePath string `json:"source_path,omitempty"`
}

// NewSource constructs a Source with IngestedAt set to now.
// Counts are filled in by the pipeline.
func NewSource(id, title, author string) Source {
	return Source{
		ID:         id,
		Title:      title,
		Author:     author,
		IngestedAt: time.Now().UTC(),
	}
}
