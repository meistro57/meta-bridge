# meta_report.py
# Generates a synthesis report from Qdrant results using Ollama

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import shutil
import subprocess

import requests
from qdrant_client import QdrantClient


# ---------------- CONFIG ----------------
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma3:latest"
COLLECTION = "meta_test"  # overridden at runtime by select_collection()
QUERY = "Do different authors agree that thoughts create reality?"
LIMIT = 8
OUTPUT_FILE = "meta_report.txt"
TIMEOUT_EMBED = 120
TIMEOUT_GENERATE = 300


# ---------------- COLLECTION SELECTION ----------------
def _embed_dim() -> int:
    return len(embed_query("test"))


def _collection_dim(client: QdrantClient, name: str) -> int | None:
    try:
        info = client.get_collection(name)
        params = info.config.params.vectors
        if hasattr(params, "size"):
            return params.size
        if isinstance(params, dict):
            first = next(iter(params.values()))
            return first.size
    except Exception:
        pass
    return None


def select_collection() -> str:
    print("Detecting embedding dimension...")
    dim = _embed_dim()

    client = QdrantClient(url=QDRANT_URL)
    all_names = [c.name for c in client.get_collections().collections]

    compatible = [n for n in all_names if _collection_dim(client, n) == dim]

    if not compatible:
        raise RuntimeError(f"No collections with vector dim={dim} found in Qdrant.")

    print(f"\nCompatible collections (dim={dim}):")
    for i, name in enumerate(compatible, 1):
        print(f"  {i}. {name}")

    while True:
        raw = input(f"\nSelect collection [1-{len(compatible)}]: ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(compatible):
                return compatible[idx]
        print(f"  Please enter a number between 1 and {len(compatible)}.")


# ---------------- EMBEDDING ----------------
def embed_query(text: str) -> list[float]:
    """
    Get an embedding vector for the query using Ollama.
    Tries /api/embed first, then falls back to the older /api/embeddings shape.
    """
    base = OLLAMA_URL.rstrip("/")

    try:
        r = requests.post(
            f"{base}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=TIMEOUT_EMBED,
        )
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings")
        if embeddings and isinstance(embeddings, list):
            return embeddings[0]
    except requests.RequestException:
        pass

    r = requests.post(
        f"{base}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=TIMEOUT_EMBED,
    )
    r.raise_for_status()
    data = r.json()

    embedding = data.get("embedding")
    if not embedding or not isinstance(embedding, list):
        raise RuntimeError(f"Unexpected Ollama embeddings response: {data}")

    return embedding


# ---------------- RETRIEVAL ----------------
def retrieve_chunks(query_vector: list[float]):
    client = QdrantClient(url=QDRANT_URL)

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=LIMIT,
        with_payload=True,
    )

    return results.points


# ---------------- SYNTHESIS ----------------
def build_prompt(chunks: list[Any]) -> str:
    context_parts: list[str] = []

    for c in chunks:
        payload = c.payload or {}
        source_file = payload.get("source_file", "unknown")
        page = payload.get("page", "?")
        text = payload.get("text", "")

        context_parts.append(
            f"[Source: {source_file} | Page: {page}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are analyzing multiple retrieved text passages from a semantic search system.

QUESTION:
{QUERY}

CONTEXT:
{context}

TASK:
1. Identify where the sources AGREE.
2. Identify any DIFFERENCES or different emphases.
3. Extract the CORE PRINCIPLE that connects the sources.
4. Write a clear synthesis explanation.
5. End with a short section called 'Sources Used' listing each source file mentioned.

RULES:
- Stay grounded in the provided context.
- Do not invent citations or claims.
- If the sources only partially support the conclusion, say so clearly.
- Write in plain, clear prose.
""".strip()

    return prompt


# ---------------- GENERATION ----------------
def generate_report_native(prompt: str) -> str:
    """
    Try Ollama's native /api/generate endpoint first.
    """
    r = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=TIMEOUT_GENERATE,
    )
    r.raise_for_status()

    data = r.json()
    response_text = data.get("response")
    if not response_text:
        raise RuntimeError(f"Unexpected Ollama generate response: {data}")

    return response_text



def generate_report_chat(prompt: str) -> str:
    """
    Fallback to Ollama's native /api/chat endpoint.
    """
    r = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful synthesis assistant. Stay grounded in the provided context.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
        },
        timeout=TIMEOUT_GENERATE,
    )
    r.raise_for_status()

    data = r.json()
    message = data.get("message", {})
    content = message.get("content")
    if not content:
        raise RuntimeError(f"Unexpected Ollama chat response: {data}")

    return content



def generate_report_cli(prompt: str) -> str:
    """
    Final fallback: call the local Ollama CLI directly.
    """
    if shutil.which("ollama") is None:
        raise RuntimeError(
            "All HTTP generation endpoints failed and the 'ollama' CLI was not found in PATH."
        )

    proc = subprocess.run(
        ["ollama", "run", LLM_MODEL, prompt],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_GENERATE,
    )

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        raise RuntimeError(
            f"Ollama CLI generation failed.\nSTDERR: {stderr}\nSTDOUT: {stdout}"
        )

    content = proc.stdout.strip()
    if not content:
        raise RuntimeError("Ollama CLI returned no output.")

    return content



def generate_report(prompt: str) -> str:
    """
    Try native generate, then native chat, then CLI fallback.
    """
    try:
        return generate_report_native(prompt)
    except Exception as exc:
        print(f"Native /api/generate failed: {exc}")

    try:
        return generate_report_chat(prompt)
    except Exception as exc:
        print(f"Native /api/chat failed: {exc}")

    print("Falling back to local 'ollama run' CLI...")
    return generate_report_cli(prompt)


# ---------------- SAVE ----------------
def save_report(text: str, chunks: list[Any]) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(OUTPUT_FILE)

    unique_sources: list[str] = []
    seen = set()
    for c in chunks:
        payload = c.payload or {}
        source = payload.get("source_file", "unknown")
        if source not in seen:
            seen.add(source)
            unique_sources.append(source)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("META ANALYSIS REPORT\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Collection: {COLLECTION}\n")
        f.write(f"Query: {QUERY}\n")
        f.write(f"Chunks used: {len(chunks)}\n")
        f.write("Sources retrieved:\n")
        for source in unique_sources:
            f.write(f"- {source}\n")
        f.write("=" * 80 + "\n\n")
        f.write(text.strip())
        f.write("\n")

    print(f"Report saved to {output_path}")


# ---------------- MAIN ----------------
def main() -> None:
    global COLLECTION
    COLLECTION = select_collection()
    print(f"\nUsing collection: {COLLECTION}")

    print("Embedding query...")
    q_vec = embed_query(QUERY)

    print("Retrieving relevant chunks...")
    chunks = retrieve_chunks(q_vec)

    if not chunks:
        print("No results found.")
        return

    print(f"Retrieved {len(chunks)} chunks")

    print("Building synthesis prompt...")
    prompt = build_prompt(chunks)

    print(f"Generating report with model: {LLM_MODEL}")
    report = generate_report(prompt)

    save_report(report, chunks)


if __name__ == "__main__":
    main()
