#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reflect import parse_gemma


def test_parse_gemma_handles_code_fences():
    raw = """```json
{"summary":"ok","concepts":[],"claims":[],"tone":"","questions":[],"echoes":[]}
```"""
    parsed = parse_gemma(raw)
    assert parsed["summary"] == "ok"


def test_parse_gemma_extracts_object_from_extra_text():
    raw = "prefix\n{" \
        '"summary":"ok","concepts":[],"claims":[],"tone":"","questions":[],"echoes":[]' \
        "}\nsuffix"
    parsed = parse_gemma(raw)
    assert parsed["summary"] == "ok"


def test_parse_gemma_repairs_newlines_and_trailing_commas():
    raw = '''{
  "summary": "line one
line two",
  "concepts": ["a",],
  "claims": ["b",],
  "tone": "",
  "questions": [],
  "echoes": [],
}'''
    parsed = parse_gemma(raw)
    assert parsed["summary"] == "line one\nline two"
    assert parsed["concepts"] == ["a"]
    assert parsed["claims"] == ["b"]
