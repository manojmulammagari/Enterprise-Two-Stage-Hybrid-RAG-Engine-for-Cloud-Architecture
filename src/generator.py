# =============================================================================
# src/generator.py  —  Grounded answer generation via Gemini 2.5 Flash Lite
# =============================================================================
# What this file does (interview talking points):
#
#   1. Reads the GEMINI_API_KEY from the environment (never hardcoded).
#   2. Formats the reranked chunks into a structured context block with
#      explicit Pillar and Page metadata headers.
#   3. Applies a strict enterprise system prompt that:
#        • Forces every claim to end with [Pillar: X, Page: N] citations.
#        • Instructs the LLM to output "INSUFFICIENT_CONTEXT" for any query
#          whose answer is not in the retrieved excerpts (out-of-domain guard).
#   4. Calls the Gemini API with temperature=0 (deterministic generation).
#   5. Wraps everything in try/except so API errors surface clearly.
# =============================================================================

import os
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types

from src.config import GEMINI_MODEL, GEMINI_TEMPERATURE


# ---------------------------------------------------------------------------
# API client setup (runs once at module import time)
# ---------------------------------------------------------------------------

def _create_client() -> genai.Client:
    """
    Reads the API key from the environment and returns a configured client.

    Why os.getenv() instead of hardcoding?
      - Hardcoding secrets in source code is a security anti-pattern.
      - Using an environment variable means the key never appears in git history.
      - In production you'd use a secret manager; for a local demo, .env + python-dotenv works.

    Raises:
        ValueError: If GEMINI_API_KEY is not set in the environment.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Create a .env file with:  GEMINI_API_KEY=your-key-here\n"
            "Or export it in your shell:  export GEMINI_API_KEY=your-key"
        )

    # http_options api_version='v1beta' is required to use system_instruction
    # (developer-level instructions separate from the user prompt).
    client = genai.Client(
        api_key      = api_key,
        http_options = {"api_version": "v1beta"},
    )
    return client


# Create the client once when the module is first imported.
# Any import error (missing key) is caught in app.py with a friendly message.
try:
    _client = _create_client()
except ValueError as e:
    _client = None
    print(f"[generator] WARNING: {e}")


# ---------------------------------------------------------------------------
# System prompt (the "contract" the LLM must follow on every response)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """\
You are an expert AWS Solutions Architect assistant.
Your task is to answer user technical architecture queries ONLY using the \
provided AWS Well-Architected Framework document excerpts.

STRICT GROUNDING RULES:
1. Answer ONLY using direct facts present in the excerpts below. \
   Do NOT extrapolate, hallucinate, or assume anything beyond the text.
2. Every recommendation, design principle, or claim MUST end with an inline \
   citation formatted EXACTLY as: [Pillar: <Pillar Name>, Page: <Page Number>].
3. If the excerpts partially address the question, synthesise the verified \
   details that ARE present and cite each one.
4. If the excerpts contain ZERO relevant information about the topic \
   (e.g., non-AWS services, out-of-domain questions like Azure or GCP), \
   output ONLY this exact phrase and nothing else:
   INSUFFICIENT_CONTEXT: The provided whitepaper excerpts do not contain \
   sufficient evidence to answer this question.\
"""


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def _format_context(top_chunks: List[Dict[str, Any]]) -> str:
    """
    Formats a list of retrieved chunk dicts into a structured context string
    that the LLM can parse clearly.

    Each block is explicitly labelled with its metadata so the LLM can
    construct correct [Pillar: X, Page: N] citations.

    Args:
        top_chunks: Output from retriever.rerank_candidates() — each dict
                    has keys: text, page, pillar, chunk_id, rrf_score, rerank_score.

    Returns:
        A multi-line string with one labelled block per chunk.
    """
    blocks = []
    for i, chunk in enumerate(top_chunks, start=1):
        block = (
            f"--- DOCUMENT EXCERPT {i} ---\n"
            f"Pillar: {chunk['pillar']}\n"
            f"Page Number: {chunk['page']}\n"
            f"Content:\n{chunk['text']}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Public function: generate_grounded_answer
# ---------------------------------------------------------------------------

def generate_grounded_answer(
    query:      str,
    top_chunks: List[Dict[str, Any]],
) -> str:
    """
    Generates a grounded, cited answer for the given query using the
    reranked chunks as context.

    This is the single public function called from app.py and evaluator.py.

    Args:
        query:      The user's question string.
        top_chunks: Reranked chunks from retriever.rerank_candidates().

    Returns:
        A string containing either:
          - A well-structured answer with [Pillar: X, Page: N] citations, OR
          - "INSUFFICIENT_CONTEXT: ..." for out-of-domain queries, OR
          - An error message if the API call fails.
    """
    # Graceful early-exit when no chunks survived reranking
    if not top_chunks:
        return (
            "INSUFFICIENT_CONTEXT: No relevant document excerpts were found "
            "in the index to answer this query. Please ensure the PDF has been "
            "indexed and try rephrasing your question."
        )

    # Graceful early-exit when the API client failed to initialise
    if _client is None:
        return (
            "ERROR: Gemini API client is not initialised. "
            "Please set the GEMINI_API_KEY environment variable and restart the app."
        )

    # Format the retrieved chunks into a prompt-friendly context block
    context_str = _format_context(top_chunks)

    # Compose the user-facing prompt.
    # The system instruction (SYSTEM_INSTRUCTION) is passed separately so the
    # LLM treats it as a standing contract, not part of the conversation history.
    user_prompt = (
        f"CONTEXT EXCERPTS:\n{context_str}\n\n"
        f"USER QUERY: {query}\n\n"
        f"DETAILED ANSWER WITH INLINE CITATIONS:"
    )

    try:
        response = _client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = user_prompt,
            config   = types.GenerateContentConfig(
                system_instruction = SYSTEM_INSTRUCTION,
                temperature        = GEMINI_TEMPERATURE,  # 0.0 = deterministic
            ),
        )
        return response.text

    except Exception as e:
        # Catch-all for network errors, quota exhaustion, model unavailability, etc.
        # We return a human-readable error rather than crashing the UI.
        return (
            f"ERROR: Gemini API call failed.\n"
            f"Details: {type(e).__name__}: {e}\n\n"
            f"Check your GEMINI_API_KEY and network connection."
        )


# ---------------------------------------------------------------------------
# Naive RAG baseline (used by evaluator.py for A/B comparison)
# ---------------------------------------------------------------------------

def naive_rag_answer(query: str, collection) -> str:
    """
    Minimal baseline: vector-only retrieval (top-3), no BM25, no reranking,
    no strict citation prompt.

    This is the "Naive RAG" we benchmark against.  Its weaknesses:
      - Misses exact keyword matches (IAM, KMS acronyms).
      - Retrieves only 3 chunks, so context is thinner.
      - No citation enforcement → LLM often answers without page references.
      - No out-of-domain refusal → LLM may hallucinate for Azure/GCP queries.

    Args:
        query:      The user's question.
        collection: ChromaDB collection (same one used by the hybrid pipeline).

    Returns:
        Raw LLM response text (no citation enforcement).
    """
    if _client is None:
        return "ERROR: Gemini API client not initialised."

    try:
        # Simple top-3 vector search — no BM25, no reranking
        results    = collection.query(query_texts=[query], n_results=3)
        texts      = results["documents"][0]
        context    = "\n\n".join(texts)

        naive_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        response = _client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = naive_prompt,
            # NOTE: No system instruction, no citation enforcement
        )
        return response.text

    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Quick smoke test  —  python -m src.generator
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal test without a real PDF — just verify the prompt formats correctly
    dummy_chunks = [
        {
            "text"         : "You should implement least-privilege IAM policies...",
            "page"         : 42,
            "pillar"       : "Security",
            "chunk_id"     : "p42_c1",
            "rrf_score"    : 0.03125,
            "rerank_score" : 8.7,
        }
    ]
    query  = "What are IAM best practices?"
    result = generate_grounded_answer(query, dummy_chunks)
    print("--- Generator output ---")
    print(result)
