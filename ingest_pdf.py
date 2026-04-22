# ingest_pdf.py
from __future__ import annotations

import argparse
import hashlib
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List

import requests
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# -----------------------------
# DEFAULT CONFIG
# -----------------------------
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text:latest"
DEFAULT_COLLECTION = "pdf_chunks"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_BATCH_SIZE = 32


# -----------------------------
# TEXT EXTRACTION
# -----------------------------
def normalize_text(text: str) -> str:
    """
    Light cleanup to improve chunk quality.
    """
    text = text.replace("\x00", " ")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> List[Dict]:
    """
    Extract text from each page of a PDF.
    Returns a list of dicts with page_number and text.
    """
    reader = PdfReader(str(pdf_path))
    pages: List[Dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = normalize_text(text)

        if text:
            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                }
            )

    return pages


def extract_txt_pages(txt_path: Path) -> List[Dict]:
    """
    Read a plain-text file as a single "page".
    Returns a list with one dict so callers treat it identically to PDF pages.
    """
    text = normalize_text(txt_path.read_text(encoding="utf-8", errors="replace"))
    if not text:
        return []
    return [{"page_number": 0, "text": text}]


# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple overlapping character chunker.
    Easy to debug and dependable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
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


def stable_point_id(source_file: str, page: int, chunk_index: int, text: str) -> str:
    """
    Stable UUID so re-ingesting the same content updates existing points
    instead of adding duplicates forever.

    Qdrant accepts point IDs as either unsigned integers or UUIDs.
    """
    raw = f"{source_file}|{page}|{chunk_index}|{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def build_chunk_records(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    source_label: str | None = None,
) -> List[Dict]:
    """
    Build chunk records for one PDF or TXT file.
    """
    label = source_label or pdf_path.name
    if pdf_path.suffix.lower() == ".txt":
        pages = extract_txt_pages(pdf_path)
    else:
        pages = extract_pdf_pages(pdf_path)
    records: List[Dict] = []

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]
        page_chunks = chunk_text(page_text, chunk_size, chunk_overlap)

        for chunk_index, chunk in enumerate(page_chunks, start=1):
            point_id = stable_point_id(label, page_number, chunk_index, chunk)
            records.append(
                {
                    "id": point_id,
                    "text": chunk,
                    "source_file": label,
                    "page": page_number,
                    "chunk_index": chunk_index,
                }
            )

    return records


# -----------------------------
# OLLAMA EMBEDDINGS
# -----------------------------
def embed_texts_ollama(
    texts: List[str],
    ollama_url: str,
    model_name: str,
) -> List[List[float]]:
    """
    Generate embeddings via Ollama's /api/embed endpoint.
    """
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/embed",
        json={
            "model": model_name,
            "input": texts,
        },
        timeout=300,
    )
    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings")

    if not embeddings:
        raise RuntimeError(f"Unexpected Ollama response: {data}")

    return embeddings


def get_vector_size(ollama_url: str, model_name: str) -> int:
    """
    Ask Ollama for a sample embedding so vector size always matches the model.
    """
    vector = embed_texts_ollama(["test"], ollama_url, model_name)[0]
    return len(vector)


# -----------------------------
# QDRANT
# -----------------------------
def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """
    Create the collection if it doesn't exist.
    """
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def upsert_records(
    client: QdrantClient,
    collection_name: str,
    records: List[Dict],
    ollama_url: str,
    model_name: str,
    batch_size: int,
) -> None:
    """
    Embed records in batches and upsert them into Qdrant.
    """
    total = len(records)

    for start in range(0, total, batch_size):
        batch = records[start:start + batch_size]
        texts = [record["text"] for record in batch]
        vectors = embed_texts_ollama(texts, ollama_url, model_name)

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

        client.upsert(collection_name=collection_name, points=points)

        done = min(start + batch_size, total)
        print(f"Upserted {done}/{total} chunks")


# -----------------------------
# FILE DISCOVERY
# -----------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def iter_input_files(input_path: Path, recursive: bool) -> Iterable[Path]:
    """
    Yield PDF and TXT files from a file path or folder path.
    """
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {input_path}")
        yield input_path
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    patterns = ["**/*.*"] if recursive else ["*.*"]
    seen: set[Path] = set()
    for pattern in patterns:
        for p in sorted(input_path.glob(pattern)):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and p not in seen:
                seen.add(p)
                yield p


# -----------------------------
# MAIN INGESTION LOGIC
# -----------------------------
def ingest_pdfs(
    input_path: Path,
    qdrant_url: str,
    collection_name: str,
    ollama_url: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    recursive: bool,
) -> None:
    """
    Ingest one PDF or a whole folder of PDFs into Qdrant.
    """
    input_files = list(iter_input_files(input_path, recursive=recursive))

    if not input_files:
        print("No supported files found (pdf, txt).")
        return

    print(f"Found {len(input_files)} file(s)")
    print("Checking embedding vector size from Ollama...")
    vector_size = get_vector_size(ollama_url, model_name)
    print(f"Vector size: {vector_size}")

    client = QdrantClient(url=qdrant_url)
    ensure_collection(client, collection_name, vector_size)

    total_files = len(input_files)
    total_chunks = 0

    for index, pdf_path in enumerate(input_files, start=1):
        print(f"\n[{index}/{total_files}] Processing: {pdf_path}")
        records = build_chunk_records(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_label=pdf_path.name,
        )

        if not records:
            print("  No extractable text found.")
            continue

        print(f"  Built {len(records)} chunks")
        upsert_records(
            client=client,
            collection_name=collection_name,
            records=records,
            ollama_url=ollama_url,
            model_name=model_name,
            batch_size=batch_size,
        )
        total_chunks += len(records)

    print("\nIngestion complete.")
    print(f"Collection: {collection_name}")
    print(f"Files processed: {total_files}")
    print(f"Chunks uploaded: {total_chunks}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk PDF/TXT files, embed them with Ollama, and store them in Qdrant."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to a PDF or TXT file, or a folder containing them.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name. Default: {DEFAULT_COLLECTION}",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help=f"Qdrant base URL. Default: {DEFAULT_QDRANT_URL}",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL. Default: {DEFAULT_OLLAMA_URL}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama embedding model. Default: {DEFAULT_OLLAMA_MODEL}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters. Default: {DEFAULT_CHUNK_SIZE}",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in characters. Default: {DEFAULT_CHUNK_OVERLAP}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding/upsert batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subfolders when input is a directory.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ingest_pdfs(
        input_path=Path(args.input),
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        ollama_url=args.ollama_url,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
