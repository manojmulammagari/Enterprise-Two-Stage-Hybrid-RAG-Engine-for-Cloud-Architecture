# =============================================================================
# src/retriever.py  —  Two-stage hybrid retrieval: RRF fusion + Cross-Encoder
# =============================================================================
# What this file does (interview talking points):
#
#   Stage 1 — Hybrid Retrieval with RRF
#     a) Dense leg : ChromaDB returns top-25 by cosine similarity
#                    (semantic meaning, handles paraphrase queries)
#     b) Sparse leg: BM25 returns top-25 by TF-IDF-style token overlap
#                    (exact match, handles acronyms: IAM, KMS, SCP)
#     c) RRF fusion: merge both ranked lists into one using the formula
#                    Score(d) = Σ  1/(k + rank_m(d))   k=60
#                    This is rank-based, so it doesn't care about the
#                    different score scales of cosine vs. BM25.
#
#   Stage 2 — Cross-Encoder Re-ranking
#     Takes the top-10 RRF candidates and runs each (query, chunk) pair
#     through ms-marco-MiniLM-L-6-v2.  Unlike a bi-encoder (which embeds
#     query and chunk separately), a cross-encoder attends over BOTH texts
#     simultaneously → much higher precision, but slower.
#     We only run it on 10 candidates, so latency is acceptable.
# =============================================================================

import numpy as np
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from src.config import (
    DENSE_TOP_K,
    HYBRID_TOP_K,
    RERANKER_MODEL,
    RERANK_THRESHOLD,
    RERANK_TOP_N,
    RRF_K,
    SPARSE_TOP_K,
)


# ---------------------------------------------------------------------------
# Module-level Cross-Encoder (loaded once, reused on every query)
# ---------------------------------------------------------------------------
# Loading is slow (~2-3 s the first time).  We load at import time so the
# Gradio UI does not stall on the first user query.

print("[retriever] Loading Cross-Encoder model (runs once at startup)...")
reranker = CrossEncoder(RERANKER_MODEL)
print(f"[retriever] Cross-Encoder ready: {RERANKER_MODEL}")


# ---------------------------------------------------------------------------
# Stage 1A: Dense semantic retrieval (ChromaDB)
# ---------------------------------------------------------------------------

def dense_retrieve(
    query:      str,
    collection,               # chromadb.Collection — passed in from the caller
    top_k:      int = DENSE_TOP_K,
) -> List[str]:
    """
    Queries the ChromaDB vector store and returns the top-k chunk IDs.

    ChromaDB converts `query` to a vector using the same embedding function
    we specified at collection-creation time (all-MiniLM-L6-v2), then finds
    the k nearest neighbours by cosine similarity.

    Args:
        query:      The user's question string.
        collection: The chromadb.Collection built by ingestion.build_indexes().
        top_k:      Number of candidates to retrieve (default from config).

    Returns:
        A list of chunk_id strings, ordered by cosine similarity (best first).
    """
    results  = collection.query(query_texts=[query], n_results=top_k)
    # results["ids"] is a list-of-lists because ChromaDB supports batch queries.
    # We always query one at a time, so we take index [0].
    chunk_ids = results["ids"][0]
    return chunk_ids   # e.g. ["p42_c153", "p99_c301", ...]


# ---------------------------------------------------------------------------
# Stage 1B: Sparse lexical retrieval (BM25)
# ---------------------------------------------------------------------------

def sparse_retrieve(
    query:      str,
    bm25_index,               # BM25Okapi — passed in from the caller
    chunks:     list,         # List[DocumentChunk] — to map index → chunk_id
    top_k:      int = SPARSE_TOP_K,
) -> List[str]:
    """
    Scores every chunk against the query using BM25Okapi and returns the
    top-k chunk IDs.

    BM25 scores are based on term frequency + inverse document frequency.
    It excels at exact keyword and acronym matching that bi-encoders blur
    into semantic similarity (e.g., "KMS" and "RDS" look similar in embedding
    space because both are AWS database/security services).

    Args:
        query:      The user's question string.
        bm25_index: The BM25Okapi object built by ingestion.build_indexes().
        chunks:     The same list of DocumentChunk objects, so we can map
                    array indices back to chunk_id strings.
        top_k:      Number of top candidates to return.

    Returns:
        A list of chunk_id strings, ordered by BM25 score (best first).
    """
    # BM25 operates on token lists, not raw strings
    query_tokens      = query.lower().split()
    bm25_scores       = bm25_index.get_scores(query_tokens)   # shape: (n_chunks,)

    # argsort ascending, then reverse for descending, then slice top-k
    top_sparse_indices = np.argsort(bm25_scores)[::-1][:top_k]
    chunk_ids          = [chunks[idx].chunk_id for idx in top_sparse_indices]
    return chunk_ids


# ---------------------------------------------------------------------------
# Stage 1C: Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_ids:  List[str],
    sparse_ids: List[str],
    rrf_k:      int = RRF_K,
) -> Dict[str, float]:
    """
    Merges two ranked lists into a single fused score dictionary.

    Formula:
        RRF_score(d) = Σ_m  1 / (k + rank_m(d))

    where rank_m(d) is the 1-based rank of document d in method m.

    Why RRF instead of a weighted sum?
      - Dense cosine scores range [-1, 1]; BM25 scores are unbounded.
      - Directly adding them would let BM25 dominate on keyword-heavy queries.
      - RRF converts both to rank-based scores (all in (0, 1/k]), making them
        directly comparable regardless of their original scale.

    Args:
        dense_ids:  Chunk IDs from ChromaDB, best-first.
        sparse_ids: Chunk IDs from BM25, best-first.
        rrf_k:      Dampening constant (60 is the academic default).

    Returns:
        A dict mapping chunk_id → fused RRF score (higher = more relevant).
    """
    rrf_scores: Dict[str, float] = {}

    # --- Dense leg contribution ---
    for rank, chunk_id in enumerate(dense_ids):
        # rank is 0-based; we use (rank + 1) for 1-based ranking in the formula
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    # --- Sparse leg contribution ---
    for rank, chunk_id in enumerate(sparse_ids):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    return rrf_scores


# ---------------------------------------------------------------------------
# Public Stage-1 function: hybrid_rrf_retrieve
# ---------------------------------------------------------------------------

def hybrid_rrf_retrieve(
    query:      str,
    collection,
    bm25_index,
    chunks:     list,
    top_k:      int = HYBRID_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Executes the full Stage-1 hybrid retrieval pipeline:
      Dense (ChromaDB) + Sparse (BM25) → RRF → top-k fused candidates.

    This is the function called by app.py and evaluator.py.

    Args:
        query:      The user's question.
        collection: ChromaDB collection from build_indexes().
        bm25_index: BM25Okapi index from build_indexes().
        chunks:     List of DocumentChunk objects from build_indexes().
        top_k:      How many RRF-fused candidates to return for reranking.

    Returns:
        A list of dicts, each with keys:
            chunk_id   : str   — unique chunk identifier
            rrf_score  : float — fused rank score (higher is better)
            page       : int   — PDF page number
            pillar     : str   — AWS pillar label
            text       : str   — chunk text content
        Sorted descending by rrf_score.
    """
    # 1. Run both retrieval legs independently
    dense_ids  = dense_retrieve(query, collection)
    sparse_ids = sparse_retrieve(query, bm25_index, chunks)

    # 2. Fuse the two ranked lists via RRF
    rrf_scores = reciprocal_rank_fusion(dense_ids, sparse_ids)

    # 3. Sort by fused score and keep top_k
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    # 4. Build a lookup map so we can retrieve full chunk metadata by ID
    chunk_map = {c.chunk_id: c for c in chunks}

    # 5. Assemble the result list
    fused_results = []
    for cid in sorted_ids:
        chunk = chunk_map[cid]
        fused_results.append({
            "chunk_id"  : chunk.chunk_id,
            "rrf_score" : round(rrf_scores[cid], 6),
            "page"      : chunk.page,
            "pillar"    : chunk.pillar,
            "text"      : chunk.text,
        })

    return fused_results


# ---------------------------------------------------------------------------
# Stage 2: Cross-Encoder re-ranking
# ---------------------------------------------------------------------------

def rerank_candidates(
    query:      str,
    candidates: List[Dict[str, Any]],
    top_n:      int   = RERANK_TOP_N,
    threshold:  float = RERANK_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Re-scores each candidate chunk using the cross-encoder model and returns
    only the highest-precision chunks above the relevance threshold.

    Why cross-encoding beats bi-encoding for precision:
      - A bi-encoder (like all-MiniLM) embeds query and chunk *separately*,
        then computes cosine similarity.  This is fast but lossy.
      - A cross-encoder concatenates [query; chunk] and runs the full
        transformer attention over BOTH.  It can tell "this chunk mentions
        IAM but not in the security context the query asks about".
      - Downside: O(n) forward passes.  We mitigate this by only reranking
        the top-10 RRF candidates, not all 2,000+ chunks.

    Args:
        query:      The user's question string.
        candidates: Output from hybrid_rrf_retrieve() — list of chunk dicts.
        top_n:      Maximum chunks to return after reranking.
        threshold:  Minimum cross-encoder score to include a chunk.
                    Chunks below this are considered irrelevant and dropped.

    Returns:
        A filtered, sorted list of candidate dicts, each now also containing:
            rerank_score : float — raw cross-encoder logit (higher = more relevant)
        Sorted descending by rerank_score, capped at top_n.
    """
    if not candidates:
        return []

    # Build input pairs: [[query, chunk_text], [query, chunk_text], ...]
    # The cross-encoder takes exactly these two-string lists.
    pairs  = [[query, c["text"]] for c in candidates]

    # Predict returns a numpy array of raw logit scores (not probabilities)
    scores = reranker.predict(pairs)

    # Attach the rerank score to each candidate dict
    for i, score in enumerate(scores):
        candidates[i]["rerank_score"] = float(score)

    # Sort by rerank score descending, apply threshold filter, return top_n
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    filtered = [c for c in reranked if c["rerank_score"] >= threshold]

    # If threshold filtering removed everything, fall back to the top-1 candidate
    # (better to return something than a silent empty list)
    if not filtered:
        filtered = reranked[:1]

    return filtered[:top_n]


# ---------------------------------------------------------------------------
# Quick smoke test  — run this file directly:  python -m src.retriever
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.config import DEFAULT_PDF_PATH
    from src.ingestion import build_indexes

    print("Building indexes for smoke test...")
    col, bm25, chunks = build_indexes(DEFAULT_PDF_PATH)

    test_query = "What are the best practices for IAM root user protection?"
    print(f"\nQuery: {test_query}\n")

    candidates = hybrid_rrf_retrieve(test_query, col, bm25, chunks)
    print(f"RRF top-{len(candidates)} candidates:")
    for r in candidates:
        print(f"  [{r['chunk_id']}] RRF={r['rrf_score']}  Page={r['page']}  {r['text'][:80]}...")

    top = rerank_candidates(test_query, candidates)
    print(f"\nAfter reranking — top-{len(top)} final chunks:")
    for r in top:
        print(f"  [{r['chunk_id']}] Rerank={r['rerank_score']:.4f}  RRF={r['rrf_score']}  Page={r['page']}")
