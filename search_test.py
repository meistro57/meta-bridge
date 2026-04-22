# search_test.py
from __future__ import annotations

import requests
from qdrant_client import QdrantClient


QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
MODEL = "nomic-embed-text:latest"
COLLECTION = "meta_test"
QUERY = "Do different authors agree that thoughts create reality?"
LIMIT = 5


def get_query_embedding(query_text: str) -> list[float]:
    """
    Get an embedding vector for the query using Ollama.
    """
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": MODEL,
            "input": query_text,
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings")

    if not embeddings or not isinstance(embeddings, list):
        raise RuntimeError(f"Unexpected Ollama response: {data}")

    return embeddings[0]


def main() -> None:
    query_vector = get_query_embedding(QUERY)

    client = QdrantClient(url=QDRANT_URL)

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=LIMIT,
        with_payload=True,
    )

    if not results.points:
        print("No results found.")
        return

    print(f"Query: {QUERY}")
    print(f"Collection: {COLLECTION}")
    print(f"Results returned: {len(results.points)}")

    for i, point in enumerate(results.points, start=1):
        payload = point.payload or {}
        page = payload.get("page", "?")
        chunk_index = payload.get("chunk_index", "?")
        source_file = payload.get("source_file", "?")
        text = payload.get("text", "")

        print("\n" + "=" * 80)
        print(f"Result #{i}")
        print(f"Score: {point.score}")
        print(f"Source: {source_file}")
        print(f"Page: {page}")
        print(f"Chunk: {chunk_index}")
        print("-" * 80)
        print(text[:1200])


if __name__ == "__main__":
    main()
