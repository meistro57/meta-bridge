// Package claim defines the Claim object: an atomic proposition extracted from
// one or more sources.
//
// Wave 1 scope: each extraction produces freshly-minted Claims with a single
// Attribution each. No dedup, no cross-source merging. That's Wave 2+.
package claim

// ClaimType distinguishes how a claim is expressed in the source.
type ClaimType string

const (
	ClaimStated      ClaimType = "stated"      // doctrinally asserted (Seth exposition)
	ClaimDramatized  ClaimType = "dramatized"  // shown through narrative action (Oversoul Seven)
	ClaimImplied     ClaimType = "implied"     // logically entailed but not spoken
)

// Attribution is a single source's support for a claim.
type Attribution struct {
	SourceID     string    `json:"source_id"`
	ChunkIndex   int       `json:"chunk_index"`       // pointer into the source's chunk list
	Chapter      string    `json:"chapter,omitempty"` // denormalized for readability
	SurfaceQuote string    `json:"surface_quote"`     // paraphrase preferred; actual quotes <15 words
	ClaimType    ClaimType `json:"claim_type"`
	Confidence   float64   `json:"confidence"`        // extractor's self-rated confidence 0..1
}

// Claim is an atomic proposition. Wave 1 produces one Claim per extraction hit;
// later waves will dedup into shared Claim objects with multiple Attributions.
type Claim struct {
	ID                  string        `json:"id"`                   // stable: "cl_<sourceid>_<nnnn>"
	CanonicalStatement  string        `json:"canonical_statement"`  // LLM abstract phrasing
	Attributions        []Attribution `json:"attributions"`
	Tags                []string      `json:"tags,omitempty"`       // "cosmology", "identity", ...
	EditorialStatus     string        `json:"editorial_status"`     // "auto" at extraction time
	Notes               string        `json:"notes,omitempty"`
}
