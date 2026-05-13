# ingest_pdf.py
from __future__ import annotations

import argparse
import os
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
# CHAPTER / SECTION DETECTION
# -----------------------------

# Matches: CHAPTER 1, CHAPTER ONE, SESSION 42, PART III, etc.
_NUMBERED_HEADER = re.compile(
    r'^(?:CHAPTER|SESSION|PART|SECTION|BOOK|VOLUME|APPENDIX)'  # keyword
    r'(?:\s+(?:[\dIVXivx]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN'  # number word
    r'|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY))?'
    r'(?:[:\s\-–—].*)?


def stable_point_id(source_file: str, page: int, chunk_index: int, text: str) -> str:
    """
    Stable UUID so re-ingesting the same content updates existing points
    instead of adding duplicates forever.

    Qdrant accepts point IDs as either unsigned integers or UUIDs.
    """
    raw = f"{source_file}|{page}|{chunk_index}|{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _source_id_from_label(label: str) -> str:
    """
    Derive a stable snake_case source_id from a filename label.
    e.g. "Seth Speaks (1972).pdf" -> "seth_speaks_1972"
    """
    name = Path(label).stem  # strip extension
    name = re.sub(r'[^a-z0-9]+', '_', name.lower())
    name = name.strip('_')
    return name


def build_chunk_records(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    source_label: Optional[str] = None,
    book_title: Optional[str] = None,
) -> List[Dict]:
    """
    Build chunk records for one PDF or TXT file.

    Each record carries:
    - text:          the raw chunk text
    - context_text:  "[Book: X | Chapter: Y]\n\n" + text  (used for embedding)
    - source_file:   original filename label
    - source_id:     snake_case stable identifier
    - book_title:    human-readable book name (from --book-title or filename)
    - chapter_title: nearest detected chapter/section header, or ""
    - page:          page number within the source
    - chunk_index:   position within the page
    """
    label = source_label or pdf_path.name
    source_id = _source_id_from_label(label)
    resolved_book_title = book_title or Path(label).stem

    if pdf_path.suffix.lower() == ".txt":
        pages = extract_txt_pages(pdf_path)
    else:
        pages = extract_pdf_pages(pdf_path)

    records: List[Dict] = []
    current_chapter: str = ""

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        # Update chapter tracking if we spot a header on this page
        detected = extract_chapter_title(page_text)
        if detected:
            current_chapter = detected

        page_chunks = chunk_text(page_text, chunk_size, chunk_overlap)

        for chunk_index, chunk in enumerate(page_chunks, start=1):
            # Build a context-rich version for embedding
            ctx_parts = []
            if resolved_book_title:
                ctx_parts.append(f"Book: {resolved_book_title}")
            if current_chapter:
                ctx_parts.append(f"Chapter: {current_chapter}")
            context_prefix = (" | ".join(ctx_parts) + "\n\n") if ctx_parts else ""
            context_text = context_prefix + chunk

            point_id = stable_point_id(label, page_number, chunk_index, chunk)
            records.append(
                {
                    "id": point_id,
                    "text": chunk,
                    "context_text": context_text,
                    "source_file": label,
                    "source_id": source_id,
                    "book_title": resolved_book_title,
                    "chapter_title": current_chapter,
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
    use_context: bool = True,
) -> List[List[float]]:
    """
    Generate embeddings via Ollama's /api/embed endpoint.
    When use_context=True (default), embed the context_text (book+chapter prefix).
    When False, embed the raw text.
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
        # Embed context_text (book+chapter prefix) if available, else raw text
        texts = [record.get("context_text") or record["text"] for record in batch]
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
                        "source_id": record.get("source_id", ""),
                        "book_title": record.get("book_title", ""),
                        "chapter_title": record.get("chapter_title", ""),
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
    book_title: Optional[str] = None,
    source_id: Optional[str] = None,
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
        # For single-file ingestion, allow explicit book_title and source_id overrides
        rec_book_title = book_title if total_files == 1 else None
        records = build_chunk_records(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_label=pdf_path.name,
            book_title=rec_book_title,
        )
        # Apply explicit source_id override if provided
        if source_id and total_files == 1:
            for r in records:
                r["source_id"] = source_id

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
    parser.add_argument(
        "--book-title",
        default=None,
        help="Human-readable book/source title to embed in chunk metadata (e.g. 'Seth Speaks'). Defaults to filename stem.",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="Explicit snake_case source identifier for Qdrant payload (e.g. 'seth_speaks'). Defaults to derived from filename.",
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
        book_title=args.book_title,
        source_id=args.source_id,
    )


if __name__ == "__main__":
    main()
,
    re.IGNORECASE,
)

# Matches lines that look like a named chapter title (ALL CAPS or Title Case, short line, no period)
_NAMED_HEADER = re.compile(
    r'^[A-Z][A-Z\s\'\'\-–—,!?]{4,60}


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

)

# Footer pattern — e.g. "Conversations with Nostradamus (VOL. I)" at bottom of page
# Used as a chapter boundary signal, not a title.
_FOOTER_PATTERN = re.compile(
    r'(?:VOL(?:UME|\.)?\s*[IVX\d]+|Book\s+[IVX\d]+)',
    re.IGNORECASE,
)

def _is_chapter_header(line: str) -> bool:
    """Return True if the line looks like a chapter or section header."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if _NUMBERED_HEADER.match(stripped):
        return True
    if _NAMED_HEADER.match(stripped):
        return True
    return False


def extract_chapter_title(text: str) -> Optional[str]:
    """
    Scan the first ~400 chars of a text block for a chapter/section header.
    Returns the header line if found, else None.
    """
    for line in text[:400].splitlines():
        if _is_chapter_header(line):
            return line.strip()
    return None


# -----------------------------
# CHUNKING
# -----------------------------

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on '.', '!', '?' followed by whitespace."""
    # Simple sentence splitter — good enough for esoteric/nonfiction prose
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Paragraph-aware chunker that never cuts mid-sentence.

    Strategy:
    1. Split into paragraphs on blank lines.
    2. Accumulate paragraphs until adding the next would exceed chunk_size.
    3. When a paragraph is larger than chunk_size on its own, split it at
       sentence boundaries.
    4. Overlap is carried forward as the last N characters of the previous
       chunk (trimmed to the nearest sentence boundary).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    def flush() -> None:
        if current_parts:
            chunks.append(" ".join(current_parts))

    for para in paragraphs:
        # If a single paragraph is larger than chunk_size, split at sentences
        if len(para) > chunk_size:
            # flush what we have first
            flush()
            current_parts = []
            current_len = 0

            sentences = _split_sentences(para)
            sent_buf: List[str] = []
            sent_len = 0
            for sent in sentences:
                if sent_len + len(sent) + 1 > chunk_size and sent_buf:
                    chunk_text_val = " ".join(sent_buf)
                    chunks.append(chunk_text_val)
                    # carry overlap: take last N chars worth of sentences
                    overlap_buf: List[str] = []
                    overlap_len = 0
                    for s in reversed(sent_buf):
                        if overlap_len + len(s) + 1 <= overlap:
                            overlap_buf.insert(0, s)
                            overlap_len += len(s) + 1
                        else:
                            break
                    sent_buf = overlap_buf + [sent]
                    sent_len = sum(len(s) + 1 for s in sent_buf)
                else:
                    sent_buf.append(sent)
                    sent_len += len(sent) + 1
            if sent_buf:
                current_parts = sent_buf
                current_len = sent_len
            continue

        # Normal paragraph: accumulate
        if current_len + len(para) + 1 > chunk_size and current_parts:
            flush()
            # carry overlap
            overlap_parts: List[str] = []
            overlap_len = 0
            for p in reversed(current_parts):
                if overlap_len + len(p) + 1 <= overlap:
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + 1
                else:
                    break
            current_parts = overlap_parts
            current_len = overlap_len

        current_parts.append(para)
        current_len += len(para) + 1

    flush()
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
