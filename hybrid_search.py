# hybrid_search.py
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip() or None
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "google/gemini-embedding-001").strip()
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
COLLECTION_NAME = "mb_claims"
RRF_K = 60


def _embed_query(query: str) -> list[float]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    response = requests.post(
        OPENROUTER_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/meistro57/meta-bridge",
            "X-Title": "Meta Bridge",
        },
        json={"model": EMBEDDING_MODEL, "input": query},
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"openrouter embeddings {response.status_code}: {response.text}")

    data = response.json()
    if data.get("error"):
        raise RuntimeError(f"openrouter error: {data['error'].get('message', 'unknown error')}")

    vectors = data.get("data") or []
    if not vectors or not vectors[0].get("embedding"):
        raise RuntimeError("openrouter returned empty embedding")

    return vectors[0]["embedding"]


def _base_source_filter(source_filter: list[str] | None) -> models.Filter | None:
    if not source_filter:
        return None
    normalized = [value.strip() for value in source_filter if value and value.strip()]
    if not normalized:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="attributions[].source_id",
                match=models.MatchAny(any=normalized),
            )
        ]
    )


def _dense_search(
    client: QdrantClient,
    query: str,
    top_k: int,
    source_filter: list[str] | None,
) -> list[models.ScoredPoint]:
    query_vector = _embed_query(query)
    dense_limit = max(top_k * 6, 60)
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=_base_source_filter(source_filter),
        limit=dense_limit,
        with_payload=["canonical_statement", "attributions", "tags"],
        with_vectors=False,
    )
    return list(result.points)


def _sparse_keyword_search(
    client: QdrantClient,
    query: str,
    top_k: int,
    source_filter: list[str] | None,
) -> list[models.Record]:
    must_conditions: list[models.Condition] = [
        models.FieldCondition(
            key="canonical_statement",
            match=models.MatchText(text=query),
        )
    ]

    source_condition = _base_source_filter(source_filter)
    if source_condition and source_condition.must:
        must_conditions.extend(source_condition.must)

    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(must=must_conditions),
        limit=max(top_k * 20, 200),
        with_payload=["canonical_statement", "attributions", "tags"],
        with_vectors=False,
    )

    query_terms = re.findall(r"[a-z0-9]+", query.lower())
    if not query_terms:
        return list(records)

    scored: list[tuple[int, models.Record]] = []
    for record in records:
        payload = record.payload or {}
        text = str(payload.get("canonical_statement") or "").lower()
        score = sum(text.count(term) for term in query_terms)
        if query.lower() in text:
            score += len(query_terms)
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored]


def _record_id(record: Any) -> str:
    return str(getattr(record, "id", ""))


def _extract_attributions(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw = payload.get("attributions")
    if not isinstance(raw, list):
        return []

    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "source_id": str(item.get("source_id") or "").strip(),
                "surface_quote": str(item.get("surface_quote") or "").strip(),
            }
        )
    return out


def hybrid_search(query: str, top_k: int = 10, source_filter: list[str] = None) -> list[dict]:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    dense_hits = _dense_search(client, query=query, top_k=top_k, source_filter=source_filter)
    sparse_hits = _sparse_keyword_search(client, query=query, top_k=top_k, source_filter=source_filter)

    fused_scores: dict[str, float] = defaultdict(float)
    payload_by_id: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        point_id = _record_id(hit)
        if not point_id:
            continue
        fused_scores[point_id] += 1.0 / (RRF_K + rank)
        payload_by_id[point_id] = dict(hit.payload or {})

    for rank, hit in enumerate(sparse_hits, start=1):
        point_id = _record_id(hit)
        if not point_id:
            continue
        fused_scores[point_id] += 1.0 / (RRF_K + rank)
        payload_by_id.setdefault(point_id, dict(hit.payload or {}))

    ranked_ids = sorted(fused_scores.keys(), key=lambda point_id: fused_scores[point_id], reverse=True)

    results: list[dict[str, Any]] = []
    for point_id in ranked_ids[:top_k]:
        payload = payload_by_id.get(point_id, {})
        attributions = _extract_attributions(payload)
        source_ids = [item["source_id"] for item in attributions if item.get("source_id")]

        results.append(
            {
                "id": point_id,
                "score": fused_scores[point_id],
                "canonical_statement": str(payload.get("canonical_statement") or "").strip(),
                "attributions": attributions,
                "tags": payload.get("tags") or [],
                "source_ids": source_ids,
            }
        )

    return results


def _run_cli_test() -> None:
    test_query = "volunteer souls struggling with third density linear reality"
    results = hybrid_search(test_query, top_k=5)

    print(f"Query: {test_query}")
    if not results:
        print("No results found.")
        return

    for index, item in enumerate(results, start=1):
        sources = ", ".join(item.get("source_ids") or []) or "unknown"
        print(f"{index}. source={sources} score={item['score']:.6f}")
        print(item.get("canonical_statement", ""))


if __name__ == "__main__":
    _run_cli_test()
