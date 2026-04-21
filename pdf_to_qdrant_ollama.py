# pdf_to_qdrant_ollama.py
# Usage:
#   python pdf_to_qdrant_ollama.py

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Dict

import requests
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# -----------------------------
# CONFIG
# -----------------------------
PDF_PATH = r"C:\path\to\your\file.pdf"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pdf_chunks"

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "nomic-embed-text:latest"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 32


# -----------------------------
# PDF EXTRACTION
# -----------------------------
def extract_pdf_pages(pdf_path: str) -> List[Dict]:
    """
    Extract text from each page of the PDF.
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = normalize_text(text)

        if text.strip():
            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                }
            )

    return pages


def normalize_text(text: str) -> str:
    """
    Light text cleanup so the chunks are less messy.
    """
    text = text.replace("\x00", " ")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = "\n".join(lines)

    # collapse repeated spaces
    text = re.sub(r"[ \t]+", " ", text)
    # collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple overlapping character-based chunker.
    Keeps things predictable and easy to debug.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = end - overlap

    return chunks


def build_chunk_records(pdf_path: str) -> List[Dict]:
    """
    Build chunk records with metadata for Qdrant payloads.
    """
    filename = Path(pdf_path).name
    pages = extract_pdf_pages(pdf_path)
    records: List[Dict] = []

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_index, chunk in enumerate(page_chunks, start=1):
            point_id = stable_point_id(filename, page_number, chunk_index, chunk)

            records.append(
                {
                    "id": point_id,
                    "text": chunk,
                    "source_file": filename,
                    "page": page_number,
                    "chunk_index": chunk_index,
                }
            )

    return records


def stable_point_id(filename: str, page: int, chunk_index: int, text: str) -> str:
    """
    Stable ID so re-ingesting the same file updates the same points instead of
    endlessly duplicating them.
    """
    raw = f"{filename}|{page}|{chunk_index}|{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


# -----------------------------
# OLLAMA EMBEDDINGS
# -----------------------------
def embed_texts_ollama(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using Ollama's /api/embed endpoint.

    Ollama supports `input` for embedding requests.
    """
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": OLLAMA_MODEL,
            "input": texts,
        },
        timeout=300,
    )
    response.raise_for_status()

    data = response.json()

    # Ollama returns embeddings as a list for batched input
    if "embeddings" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {data}")

    return data["embeddings"]


def get_vector_size() -> int:
    """
    Ask Ollama for one embedding so the collection size always matches the model.
    """
    test_vector = embed_texts_ollama(["test"])[0]
    return len(test_vector)


# -----------------------------
# QDRANT
# -----------------------------
def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """
    Create collection if it doesn't exist yet.
    """
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def upsert_records(client: QdrantClient, collection_name: str, records: List[Dict]) -> None:
    """
    Batch-embed and upsert records into Qdrant.
    """
    total = len(records)

    for start in range(0, total, BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        texts = [r["text"] for r in batch]

        vectors = embed_texts_ollama(texts)

        points = []
        for record, vector in zip(batch, vectors):
            points.append(
                PointStruct(
                    id=record["id"],
                    vector=vector,
                    payload={
                        "text": record["text"],
                        "source_file": record["source_file"],
                        "page": record["page"],
                        "chunk_index": record["chunk_index"],
                    },
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

        done = min(start + BATCH_SIZE, total)
        print(f"Upserted {done}/{total} chunks")


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    pdf_file = Path(PDF_PATH)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    print("Extracting and chunking PDF...")
    records = build_chunk_records(str(pdf_file))

    if not records:
        print("No text found in the PDF.")
        print("This often means the PDF is scanned/image-only and needs OCR first.")
        return

    print(f"Built {len(records)} chunks")

    print("Getting embedding vector size from Ollama...")
    vector_size = get_vector_size()
    print(f"Vector size: {vector_size}")

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    print(f"Ensuring collection exists: {COLLECTION_NAME}")
    ensure_collection(client, COLLECTION_NAME, vector_size)

    print("Embedding and uploading to Qdrant...")
    upsert_records(client, COLLECTION_NAME, records)

    print("Done.")


if __name__ == "__main__":
    main()
