#!/usr/bin/env python3
"""
test_reality_filter.py — Tests for Reality Filter v0.

Tests schema validation, enum labels, and scoring logic without LLM calls.
Run: python -m pytest tests/test_reality_filter.py -v
"""

import json
import sys
import os

# Add parent dir to path so we can import scoring module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.reality_filter import (
    parse_score,
    build_bridge_input,
    score_bridge,
    render_epistemic_header,
    render_bridge_table,
    VALID_BRIDGE_TYPES,
    VALID_CONSTRAINT_FLAGS,
    VALID_EPISTEMIC_STATUSES,
)


# ---------------------------------------------------------------------------
# Fixtures: 3 known bridge examples
# ---------------------------------------------------------------------------

FIXTURE_CLUSTERS = {
    0: {"label": "Quantum superposition and wave function collapse", "size": 45, "coherence": 0.82},
    1: {"label": "Consciousness as field phenomenon (Seth/Ra)", "size": 120, "coherence": 0.71},
    2: {"label": "Retrocausal influences and time symmetry", "size": 30, "coherence": 0.65},
    3: {"label": "Meditation states and brain wave patterns", "size": 88, "coherence": 0.74},
    4: {"label": "Decoherence and quantum-classical boundary", "size": 52, "coherence": 0.89},
}

# Bridge 1: Classic "observer collapses wavefunction = consciousness" — should be Analogical with flags
BRIDGE_OBSERVER = {
    "cluster_a": 0,
    "cluster_b": 1,
    "strength": 0.62,
    "link_type": "moderate_bridge",
    "label": "Observer role in measurement linked to consciousness creating reality",
    "sample_links": [
        {"chunk_a_id": 101, "chunk_b_id": 201, "similarity": 0.71},
        {"chunk_a_id": 102, "chunk_b_id": 202, "similarity": 0.65},
    ],
}

# Bridge 2: Retrocausality in QM ↔ precognition claims — should be Speculative
BRIDGE_RETROCAUSAL = {
    "cluster_a": 2,
    "cluster_b": 1,
    "strength": 0.44,
    "link_type": "weak_connection",
    "label": "Time-symmetric physics paralleled with channeled claims of seeing the future",
    "sample_links": [
        {"chunk_a_id": 301, "chunk_b_id": 201, "similarity": 0.48},
    ],
}

# Bridge 3: Decoherence ↔ meditation — should be flagged for scale gap
BRIDGE_DECOHERENCE_MEDITATION = {
    "cluster_a": 4,
    "cluster_b": 3,
    "strength": 0.38,
    "link_type": "weak_connection",
    "label": "Quantum coherence linked to meditative coherence states",
    "sample_links": [],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseScore:
    """Test JSON parsing and validation of LLM responses."""

    def test_valid_response(self):
        raw = json.dumps({
            "bridge_type": "Analogical",
            "constraint_flags": ["measurement-problem-conflation", "equivocation"],
            "testability_score": 1,
            "epistemic_status": "Suggestive but ungrounded",
            "reasoning": "Surface similarity in observer language, different mechanics.",
        })
        result = parse_score(raw)
        assert result["bridge_type"] == "Analogical"
        assert result["testability_score"] == 1
        assert "measurement-problem-conflation" in result["constraint_flags"]
        assert result["epistemic_status"] == "Suggestive but ungrounded"

    def test_invalid_bridge_type_defaults(self):
        raw = json.dumps({
            "bridge_type": "MadeUpType",
            "constraint_flags": [],
            "testability_score": 2,
            "epistemic_status": "Plausible analogy",
            "reasoning": "test",
        })
        result = parse_score(raw)
        assert result["bridge_type"] == "Speculative"  # safe default

    def test_out_of_range_testability(self):
        raw = json.dumps({
            "bridge_type": "Structural",
            "constraint_flags": [],
            "testability_score": 99,
            "epistemic_status": "Established mapping",
            "reasoning": "test",
        })
        result = parse_score(raw)
        assert result["testability_score"] == 0  # out of range resets

    def test_negative_testability(self):
        raw = json.dumps({
            "bridge_type": "Structural",
            "constraint_flags": [],
            "testability_score": -1,
            "epistemic_status": "Established mapping",
            "reasoning": "test",
        })
        result = parse_score(raw)
        assert result["testability_score"] == 0

    def test_missing_constraint_flags(self):
        raw = json.dumps({
            "bridge_type": "Speculative",
            "testability_score": 0,
            "epistemic_status": "Poetic metaphor",
            "reasoning": "test",
        })
        result = parse_score(raw)
        assert result["constraint_flags"] == []

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"bridge_type": "Testable", "constraint_flags": [], "testability_score": 3, "epistemic_status": "Plausible analogy", "reasoning": "test"}\n```'
        result = parse_score(raw)
        assert result["bridge_type"] == "Testable"
        assert result["testability_score"] == 3

    def test_custom_epistemic_status_preserved(self):
        """LLM may return a status not in our predefined list — keep it."""
        raw = json.dumps({
            "bridge_type": "Analogical",
            "constraint_flags": [],
            "testability_score": 1,
            "epistemic_status": "Interesting but requires formalization",
            "reasoning": "test",
        })
        result = parse_score(raw)
        assert result["epistemic_status"] == "Interesting but requires formalization"


class TestBuildBridgeInput:
    """Test prompt construction."""

    def test_builds_with_all_fields(self):
        prompt = build_bridge_input(BRIDGE_OBSERVER, FIXTURE_CLUSTERS)
        assert "Quantum superposition" in prompt
        assert "Consciousness as field" in prompt
        assert "0.62" in prompt or "strength" in prompt
        assert "Observer role" in prompt

    def test_handles_missing_clusters(self):
        bridge = {"cluster_a": 99, "cluster_b": 100, "strength": 0.5}
        prompt = build_bridge_input(bridge, FIXTURE_CLUSTERS)
        assert "unlabeled" in prompt

    def test_handles_empty_sample_links(self):
        prompt = build_bridge_input(BRIDGE_DECOHERENCE_MEDITATION, FIXTURE_CLUSTERS)
        assert "Pair" not in prompt  # no sample links to render


class TestScoreBridgeDryRun:
    """Test scoring in dry-run mode (no LLM calls)."""

    def test_dry_run_returns_valid_schema(self):
        result = score_bridge(BRIDGE_OBSERVER, FIXTURE_CLUSTERS, dry_run=True)
        assert result["bridge_type"] in VALID_BRIDGE_TYPES
        assert isinstance(result["constraint_flags"], list)
        assert isinstance(result["testability_score"], int)
        assert 0 <= result["testability_score"] <= 5
        assert isinstance(result["epistemic_status"], str)
        assert "[DRY RUN]" in result["reasoning"]

    def test_dry_run_includes_prompt(self):
        result = score_bridge(BRIDGE_RETROCAUSAL, FIXTURE_CLUSTERS, dry_run=True)
        assert "_prompt" in result
        assert "Retrocausal" in result["_prompt"]


class TestRendering:
    """Test markdown rendering helpers."""

    def test_epistemic_header(self):
        scored = [
            {"bridge_type": "Analogical", "constraint_flags": ["equivocation"], "testability_score": 1},
            {"bridge_type": "Speculative", "constraint_flags": [], "testability_score": 0},
            {"bridge_type": "Analogical", "constraint_flags": [], "testability_score": 4},
        ]
        header = render_epistemic_header(scored)
        assert "Epistemic Status Summary" in header
        assert "Analogical" in header
        assert "1 bridge(s) flagged" in header
        assert "1 bridge(s)" in header  # testability >= 3

    def test_bridge_table(self):
        scored = [
            {
                "cluster_a": 0, "cluster_b": 1, "strength": 0.62,
                "bridge_type": "Analogical", "epistemic_status": "Suggestive but ungrounded",
                "testability_score": 1, "constraint_flags": ["equivocation"],
            },
        ]
        table = render_bridge_table(scored)
        assert "| 0 |" in table or "| 0" in table
        assert "Analogical" in table
        assert "equivocation" in table


class TestEnumCompleteness:
    """Ensure our valid sets cover the rubric."""

    def test_bridge_types(self):
        assert len(VALID_BRIDGE_TYPES) == 5
        for bt in ["Structural", "Analogical", "Speculative", "Testable", "Contradictory"]:
            assert bt in VALID_BRIDGE_TYPES

    def test_constraint_flags(self):
        assert len(VALID_CONSTRAINT_FLAGS) >= 7
        assert "decoherence-scale-gap" in VALID_CONSTRAINT_FLAGS
        assert "no-communication-theorem" in VALID_CONSTRAINT_FLAGS

    def test_epistemic_statuses(self):
        assert len(VALID_EPISTEMIC_STATUSES) >= 7
        assert "Poetic metaphor" in VALID_EPISTEMIC_STATUSES
        assert "Established mapping" in VALID_EPISTEMIC_STATUSES
