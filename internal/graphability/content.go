package graphability

import "strings"

// contentKeywords maps score tiers to vocabulary lists.
// Matching is case-insensitive substring against the chunk text.
// The scorer walks tiers from highest to lowest — first tier with
// enough keyword hits wins.
//
// Tune minHits per tier: very_high needs 3 distinct hits to avoid
// false positives on generic text that mentions "soul" once.
var contentTiers = []struct {
	score   Score
	minHits int
	terms   []string
}{
	{
		score:   VeryHigh,
		minHits: 3,
		terms: []string{
			// Cosmological structure
			"density", "octave", "logos", "sub-logos", "sub-sub-logos",
			"intelligent infinity", "intelligent energy", "original thought",
			"harvest", "graduation", "wanderer", "star seed", "starseed",
			"soul contract", "soul origin", "soul lineage",
			"oversoul", "higher self", "higher-self",
			"law of one", "law of free will", "law of confusion",
			"free will", "veil of forgetting", "veil of forgetfulness",
			"akashic", "akasha",
			// Channeled session markers
			"questioner:", "ra:", "seth:", "q:", "a:", "subject:",
			"in trance", "under hypnosis", "regression session",
			"past life", "between lives", "life between lives",
			// Dimensional / energetic structure
			"dimension", "plane of existence", "astral plane",
			"etheric", "causal body", "light body", "merkaba",
			"sacred geometry", "flower of life", "fibonacci",
			"activation", "kundalini", "chakra",
			// Creation cosmology
			"creation", "creator", "prime creator", "source energy",
			"infinite creator", "all that is", "one infinite creator",
			"consciousness", "awareness", "sentience",
			"reality tunnel", "reality engineering", "reality creation",
			// Soul mechanics
			"karma", "dharma", "reincarnation", "incarnation",
			"soul group", "soul family", "monad",
			"probable self", "probable reality", "parallel self",
			"simultaneous time", "no time", "eternal now",
			// Entities / beings
			"extraterrestrial", "et contact", "pleiadians", "sirians",
			"arcturians", "annunaki", "elohim", "nephilim",
			"spirit guide", "spirit guides", "angelic",
			// Specific traditions
			"sefirot", "sephirot", "ain soph", "tree of life",
			"hermetic", "as above so below",
			"tao", "wu wei", "te",
			"brahman", "atman", "maya", "prana", "shakti",
			"bardo", "tibetan", "dzogchen",
		},
	},
	{
		score:   High,
		minHits: 2,
		terms: []string{
			"vibration", "frequency", "resonance", "attunement",
			"meditation", "contemplation", "inner knowing",
			"intuition", "imagination", "imaginal",
			"energy body", "subtle body", "aura",
			"synchronicity", "synchronicities",
			"awakening", "enlightenment", "ascension",
			"cosmic", "universal law", "natural law",
			"healing", "transmutation", "transformation",
			"sacred", "divine", "infinite",
			"multiverse", "multidimensional",
			"time-space", "space-time",
			"belief system", "belief structure",
			"thought form", "thought-form",
			"emotion", "emotional body",
			"astrology", "celestial", "planetary influence",
			"mythology", "myth", "archetype",
			"initiation", "mystery school", "mystery tradition",
			"alchemy", "alchemical",
		},
	},
	{
		score:   Medium,
		minHits: 2,
		terms: []string{
			"spiritual", "metaphysical", "esoteric",
			"philosophy", "philosophical",
			"ancient", "tradition", "wisdom",
			"religion", "religious", "theology",
			"god", "goddess", "deity", "divine being",
			"prayer", "ritual", "ceremony",
			"symbol", "symbolic", "symbolism",
			"dream", "vision", "prophecy",
			"death", "afterlife", "beyond death",
			"rebirth", "cycle of life",
			"nature", "cosmos", "universe",
			"mind", "spirit", "soul",
			"light", "darkness", "duality",
		},
	},
	{
		score:   Low,
		minHits: 1,
		terms: []string{
			"published by", "all rights reserved",
			"copyright ©", "isbn", "library of congress",
			"printed in", "first edition", "second edition",
			"acknowledgment", "acknowledgements",
			"preface", "foreword by", "introduction by",
			"about the author", "the author",
			"bibliography", "references", "index",
			"glossary of terms",
		},
	},
	{
		score:   VeryLow,
		minHits: 1,
		terms: []string{
			"table of contents", "contents",
			"©", "all rights reserved",
			"for more information", "visit our website",
			"printed in the united states",
			"no part of this publication",
			"without written permission",
			"cataloging-in-publication",
		},
	},
}

// ScoreText scores a chunk by its text content using keyword density.
// Returns Unknown if no tier reaches its minHits threshold.
// This runs alongside ScoreLabel — the higher of the two wins in the
// combined scorer.
func ScoreText(text string) Score {
	lower := strings.ToLower(text)

	for _, tier := range contentTiers {
		hits := 0
		for _, term := range tier.terms {
			if strings.Contains(lower, term) {
				hits++
				if hits >= tier.minHits {
					return tier.score
				}
			}
		}
	}
	return Unknown
}
