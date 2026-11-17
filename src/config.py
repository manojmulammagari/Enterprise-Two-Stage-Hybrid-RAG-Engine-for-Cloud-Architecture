# =============================================================================
# src/config.py  —  Central configuration for every tunable constant
# =============================================================================
# Why a config file?
#   Instead of scattering magic numbers across five different files, we keep
#   ALL tunable parameters here. 
#   "Any hyperparameter tweak only touches this one file."
# =============================================================================

# ---------------------------------------------------------------------------
# 1.  PDF chunking parameters
# ---------------------------------------------------------------------------

CHUNK_SIZE    = 900   # Maximum characters per document chunk.
                      # 900 chars ≈ 150-200 tokens — a sweet spot that fits
                      # well-architected guidance paragraphs without splitting
                      # a single recommendation across two chunks.

CHUNK_OVERLAP = 150   # Characters of overlap between adjacent chunks.
                      # Overlap prevents a key sentence from being cut at a
                      # boundary and lost from both neighbours.

MIN_CHUNK_LEN = 80    # Chunks shorter than this (e.g., page headers) are
                      # discarded — they add noise without useful content.

# ---------------------------------------------------------------------------
# 2.  Embedding & retrieval model names
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Fast 384-dim bi-encoder.  Runs on CPU in <1 s per batch.
# Used by ChromaDB to convert chunks → vectors at index time, and to convert
# the query → vector at search time.

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# MS-MARCO-trained cross-encoder.  Unlike the bi-encoder above, it sees the
# query AND the candidate chunk *together*, allowing full self-attention
# between them.  Slower (runs once per candidate) but far more precise.

# ---------------------------------------------------------------------------
# 3.  ChromaDB collection name
# ---------------------------------------------------------------------------

CHROMA_COLLECTION_NAME = "aws_well_architected"
# The named collection inside the in-memory ChromaDB client.

# ---------------------------------------------------------------------------
# 4.  Retrieval hyperparameters
# ---------------------------------------------------------------------------

DENSE_TOP_K  = 25   # How many candidates ChromaDB (dense) returns before RRF.
SPARSE_TOP_K = 25   # How many candidates BM25 (sparse) returns before RRF.
RRF_K        = 60   # The k constant in the RRF formula:
                    #   Score(d) = Σ  1 / (k + rank_m(d))
                    # k=60 is the academic default; it dampens the impact of
                    # very high ranks without discarding them.

HYBRID_TOP_K = 10   # Number of RRF-fused candidates passed to the reranker.

# ---------------------------------------------------------------------------
# 5.  Reranker thresholds
# ---------------------------------------------------------------------------

RERANK_TOP_N      = 5     # How many chunks survive reranking and reach the LLM.
RERANK_THRESHOLD  = -5.0  # Cross-encoder raw logit threshold.
                           # Chunks scoring below this are considered irrelevant
                           # and are dropped before the LLM sees them.
                           # (ms-marco scores are raw logits, NOT probabilities;
                           #  -5.0 filters out near-zero relevance chunks.)

# ---------------------------------------------------------------------------
# 6.  AWS pillar taxonomy  (used by ingestion pillar-detection)
# ---------------------------------------------------------------------------

AWS_PILLARS = [
    "Operational Excellence",
    "Security",
    "Reliability",
    "Performance Efficiency",
    "Cost Optimization",
    "Sustainability",
]
# The 6 official AWS Well-Architected pillars in declaration order.
# detect_pillar() in ingestion.py scans each page for these strings.

DEFAULT_PILLAR = "General Overview"
# Pillar label assigned to pages before the first named pillar is detected
# (e.g., introduction and table-of-contents pages).

# ---------------------------------------------------------------------------
# 7.  Gemini generation settings
# ---------------------------------------------------------------------------

GEMINI_MODEL       = "gemini-2.5-flash-lite"
GEMINI_TEMPERATURE = 0.0   # Deterministic — zero randomness for factual answers.
                            # Setting temperature=0 makes the model choose the
                            # single most probable token at every step, which
                            # maximises citation accuracy.

# ---------------------------------------------------------------------------
# 8.  Default PDF path  (override at runtime via app.py or CLI)
# ---------------------------------------------------------------------------

import os

DEFAULT_PDF_PATH = (
    "data/wellarchitected-framework-2023-04-10.pdf"
    if os.path.exists("data/wellarchitected-framework-2023-04-10.pdf")
    else "Data/wellarchitected-framework-2023-04-10.pdf"
)
# If the user does NOT upload a custom PDF, the app falls back to this path.
