# meta_report.py
# Generates a synthesis report from Qdrant results using configured providers

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import os
import shutil
import subprocess

import requests
from qdrant_client import QdrantClient


def load_env_file() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()


# ---------------- CONFIG ----------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
EMBED_PROVIDER = os.environ.get("MB_EMBED_PROVIDER", "openrouter").strip().lower()
EMBED_MODEL = os.environ.get("MB_EMBED_MODEL", "openai/text-embedding-3-small")
DEFAULT_REPORT_MODEL = "deepseek/deepseek-v4-flash"
MODEL = (
    os.environ.get("MB_REPORT_MODEL", "").strip()
    or os.environ.get("MB_MODEL", "").strip()
    or DEFAULT_REPORT_MODEL
)
COLLECTION = "mb_chunks"  # overridden at runtime by select_collection()
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


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


def select_collection() -> str:
    print("Detecting embedding dimension...")
    dim = _embed_dim()

    client = qdrant_client()
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
def openrouter_embed_query(text: str) -> list[float]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required for openrouter embeddings")

    body = {"model": EMBED_MODEL, "input": text}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/meistro57/meta-bridge",
        "X-Title": "Meta Bridge",
    }
    resp = requests.post(OPENROUTER_EMBEDDINGS_URL, json=body, headers=headers, timeout=TIMEOUT_EMBED)
    if resp.status_code >= 400:
        raise RuntimeError(f"openrouter embeddings {resp.status_code}: {resp.text}")
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"openrouter error: {data['error'].get('message', 'unknown error')}")
    vectors = data.get("data") or []
    if not vectors or not vectors[0].get("embedding"):
        raise RuntimeError("openrouter returned empty embedding")
    return vectors[0]["embedding"]


def ollama_embed_query(text: str) -> list[float]:
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


def embed_query(text: str) -> list[float]:
    if EMBED_PROVIDER == "openrouter":
        return openrouter_embed_query(text)
    if EMBED_PROVIDER == "ollama":
        return ollama_embed_query(text)
    raise RuntimeError("MB_EMBED_PROVIDER must be 'openrouter' or 'ollama'")


# ---------------- RETRIEVAL ----------------
def retrieve_chunks(query_vector: list[float]):
    client = qdrant_client()

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
def openrouter_generate_report(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter models")

    body = {
        "model": MODEL,
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
        "temperature": 0.2,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/meistro57/meta-bridge",
        "X-Title": "Meta Bridge",
    }

    resp = requests.post(OPENROUTER_CHAT_URL, json=body, headers=headers, timeout=TIMEOUT_GENERATE)
    if resp.status_code >= 400:
        if resp.status_code == 400 and "not a valid model ID" in resp.text:
            raise RuntimeError(
                f"openrouter chat 400: model '{MODEL}' is invalid. "
                f"Set MB_REPORT_MODEL to a valid OpenRouter model id (for example '{DEFAULT_REPORT_MODEL}')."
            )
        raise RuntimeError(f"openrouter chat {resp.status_code}: {resp.text}")
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"openrouter error: {data['error'].get('message', 'unknown error')}")

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"openrouter returned no choices: {data}")

    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"openrouter returned empty message: {data}")
    return content


def generate_report_native(prompt: str, ollama_model: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        json={
            "model": ollama_model,
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


def generate_report_chat(prompt: str, ollama_model: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        json={
            "model": ollama_model,
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


def generate_report_cli(prompt: str, ollama_model: str) -> str:
    if shutil.which("ollama") is None:
        raise RuntimeError(
            "All HTTP generation endpoints failed and the 'ollama' CLI was not found in PATH."
        )

    proc = subprocess.run(
        ["ollama", "run", ollama_model, prompt],
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
    return openrouter_generate_report(prompt)


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

    if EMBED_PROVIDER not in {"openrouter", "ollama"}:
        raise RuntimeError("MB_EMBED_PROVIDER must be 'openrouter' or 'ollama'")

    if EMBED_PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required for openrouter embeddings")

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter chat models")

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

    print(f"Generating report with model: {MODEL}")
    report = generate_report(prompt)

    save_report(report, chunks)


if __name__ == "__main__":
    main()
