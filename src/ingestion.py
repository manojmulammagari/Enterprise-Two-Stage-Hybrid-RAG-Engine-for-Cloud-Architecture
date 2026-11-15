# =============================================================================
# src/ingestion.py  —  PDF loading, chunking, and dual-index construction
# =============================================================================
# What this file does (interview talking points):
#
#   1. Reads the PDF page-by-page with pypdf.
#   2. Detects which AWS Well-Architected pillar each page belongs to.
#   3. Applies a sliding-window character chunker with overlap so no key
#      sentence is cut at a boundary.
#   4. Loads chunks into ChromaDB (dense vector index, all-MiniLM-L6-v2).
#   5. Tokenises chunks and builds a BM25Okapi sparse keyword index.
#   6. Returns both indexes so the retriever can use them immediately.
# =============================================================================

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# Import all constants from our central config
from src.config import (
    AWS_PILLARS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION_NAME,
    DEFAULT_PILLAR,
    EMBEDDING_MODEL,
    MIN_CHUNK_LEN,
)


# ---------------------------------------------------------------------------
# Data model: a single document chunk with its metadata
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """
    A single piece of text extracted from the PDF, plus the metadata we
    extracted at parse time.  Using a dataclass keeps field access clean:
    chunk.text, chunk.page, chunk.pillar, chunk.chunk_id.
    """
    text:     str   # The actual text content
    page:     int   # 1-based page number in the source PDF
    pillar:   str   # Which AWS pillar this page belongs to
    chunk_id: str   # Unique identifier, e.g. "p42_c153"


# ---------------------------------------------------------------------------
# Pillar detection helper
# ---------------------------------------------------------------------------

def detect_pillar(page_text: str, current_pillar: str) -> str:
    """
    Checks whether this page's text introduces a new AWS Pillar section.

    Strategy: scan for the pattern "<PillarName> pillar" (case-insensitive).
    If found, switch to that pillar.  Otherwise keep the existing one so
    pillar context carries over across pages inside the same section.

    Args:
        page_text:      Cleaned text of the current PDF page.
        current_pillar: Whatever pillar we were in on the previous page.

    Returns:
        The (possibly updated) pillar string.
    """
    for pillar in AWS_PILLARS:
        # \b = word boundary; prevents "Reliability" matching "High Reliability"
        if re.search(rf"\b{pillar}\s+pillar\b", page_text, re.IGNORECASE):
            return pillar
    return current_pillar   # No new pillar found — carry forward


# ---------------------------------------------------------------------------
# Step 1: Extract and chunk the PDF
# ---------------------------------------------------------------------------

def extract_and_chunk_pdf(
    pdf_file_path: str,
    chunk_size:    int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """
    Reads a PDF and returns a list of DocumentChunk objects.

    The chunking uses a sliding-window approach:
      - Window size  = chunk_size  (default 900 chars)
      - Step size    = chunk_size - chunk_overlap  (default 750 chars)
      - Each window advances by the step, so adjacent chunks share
        'chunk_overlap' characters of context.

    Args:
        pdf_file_path: Absolute or relative path to any PDF file.
                       No hardcoded Colab /content/ paths here — works
                       on any local machine or cloud VM.
        chunk_size:    Max characters per chunk (from config.py).
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        A list of DocumentChunk dataclass instances.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
    """
    # Validate the path exists before doing any work
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(
            f"PDF not found at: '{pdf_file_path}'\n"
            "Please check the path or upload the file."
        )

    reader       = PdfReader(pdf_file_path)
    total_pages  = len(reader.pages)
    print(f"[ingestion] PDF loaded: {total_pages} pages  |  source: {pdf_file_path}")

    chunks         = []
    active_pillar  = DEFAULT_PILLAR   # Start before any named pillar section
    chunk_counter  = 0

    for page_idx, page in enumerate(reader.pages):
        page_num  = page_idx + 1   # Convert 0-based index to 1-based page number
        raw_text  = page.extract_text() or ""

        # Collapse all whitespace (newlines, tabs, multiple spaces) into single spaces.
        # This prevents the chunker from splitting on accidental line-break positions
        # that exist in the PDF layout but are meaningless in the text.
        cleaned_text = re.sub(r"\s+", " ", raw_text).strip()

        # Skip completely blank pages (cover images, blank separators, etc.)
        if not cleaned_text:
            continue

        # Check if this page opens a new pillar chapter
        active_pillar = detect_pillar(cleaned_text, active_pillar)

        # ----- Sliding-window chunking -----
        start = 0
        while start < len(cleaned_text):
            end        = start + chunk_size
            chunk_text = cleaned_text[start:end].strip()

            # Discard tiny remnants (e.g., a lone page number at the end)
            if len(chunk_text) >= MIN_CHUNK_LEN:
                chunk_id = f"p{page_num}_c{chunk_counter}"
                chunks.append(DocumentChunk(
                    text     = chunk_text,
                    page     = page_num,
                    pillar   = active_pillar,
                    chunk_id = chunk_id,
                ))
                chunk_counter += 1

            # Advance window by (chunk_size - overlap) so the next chunk
            # starts chunk_overlap characters back from where this one ended
            start += (chunk_size - chunk_overlap)

    print(f"[ingestion] Extracted {len(chunks)} chunks across all AWS pillars.")
    return chunks


# ---------------------------------------------------------------------------
# Step 2: Build ChromaDB (dense) and BM25 (sparse) indexes
# ---------------------------------------------------------------------------

def build_indexes(
    pdf_file_path: str,
) -> Tuple[chromadb.Collection, BM25Okapi, List[DocumentChunk]]:
    """
    Master function: parse the PDF → chunk it → build both indexes.

    This is the single public function you call from app.py and evaluator.py.
    It encapsulates every ingestion step, so callers just do:

        collection, bm25_index, chunks = build_indexes("path/to/file.pdf")

    Args:
        pdf_file_path: Path to the source PDF.

    Returns:
        A 3-tuple:
          - collection  : chromadb.Collection  (dense vector index)
          - bm25_index  : BM25Okapi            (sparse keyword index)
          - chunks      : List[DocumentChunk]  (keep this to map chunk_id → metadata)
    """

    # --- 1. Extract and chunk the PDF ---
    chunks = extract_and_chunk_pdf(pdf_file_path)

    # --- 2. Build the ChromaDB dense vector index ---
    print("[ingestion] Initialising ChromaDB with all-MiniLM-L6-v2 embeddings...")

    # SentenceTransformerEmbeddingFunction runs the embedding model locally —
    # no external API call, works fully offline.
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # ephemeral (in-memory) client — no disk files needed for a demo/interview
    chroma_client = chromadb.Client()

    # Delete the collection if it already exists so re-indexing is idempotent
    # (you can call build_indexes() multiple times without duplicate-ID errors)
    try:
        chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        pass   # Collection didn't exist yet — that's fine

    collection = chroma_client.create_collection(
        name             = CHROMA_COLLECTION_NAME,
        embedding_function = emb_fn,
        metadata         = {"hnsw:space": "cosine"},  # Use cosine similarity
    )

    # Insert chunks in batches of 100 to avoid ChromaDB's internal limits
    print(f"[ingestion] Indexing {len(chunks)} chunks into ChromaDB (batch_size=100)...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            documents  = [c.text     for c in batch],
            metadatas  = [{"page": c.page, "pillar": c.pillar, "chunk_id": c.chunk_id} for c in batch],
            ids        = [c.chunk_id for c in batch],
        )

    print(f"[ingestion] ChromaDB collection ready: {collection.count()} vectors stored.")

    # --- 3. Build the BM25 sparse keyword index ---
    # BM25Okapi expects a list of token lists (not raw strings).
    # We use a simple whitespace split here — good enough for exact acronym
    # matching (IAM, KMS, S3) which is the whole point of the sparse leg.
    print("[ingestion] Building BM25 sparse keyword index...")
    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25_index       = BM25Okapi(tokenized_corpus)

    print(f"[ingestion] BM25 index ready: {len(tokenized_corpus)} documents tokenised.")
    print("[ingestion] Dual-index build complete.\n")

    # Return all three so callers can do retrieval + score display
    return collection, bm25_index, chunks


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly:  python -m src.ingestion
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.config import DEFAULT_PDF_PATH

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH
    col, bm25, all_chunks = build_indexes(pdf_path)

    # Print a sample chunk to verify everything parsed correctly
    sample = all_chunks[min(10, len(all_chunks) - 1)]
    print(f"\nSample chunk:")
    print(f"  ID     : {sample.chunk_id}")
    print(f"  Page   : {sample.page}")
    print(f"  Pillar : {sample.pillar}")
    print(f"  Text   : {sample.text[:120]}...")
