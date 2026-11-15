# =============================================================================
# app.py  —  Gradio UI for the Two-Stage Hybrid RAG Engine
# =============================================================================
# What this file does (interview talking points):
#
#   • Provides a clean web interface backed entirely by src/ modules.
#   • On startup, loads dotenv (reads GEMINI_API_KEY from .env) and
#     auto-indexes the default AWS PDF if it exists.
#   • Lets the user upload ANY custom PDF and re-index on the fly.
#   • Every query shows three things:
#       Tab 1 — Cited answer (Gemini 2.5 Flash Lite output)
#       Tab 2 — Retrieved chunks with RRF score vs. Cross-Encoder score
#       Tab 3 — Evaluation benchmark table (Naive vs. Hybrid RAG)
#
#   Run with:  python app.py
# =============================================================================

import os
import json
import gradio as gr

# Load .env file BEFORE importing generator (which reads GEMINI_API_KEY)
# python-dotenv reads key=value lines from .env into os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()   # Looks for .env in the current working directory
except ImportError:
    pass   # dotenv is optional; user may export the key in their shell instead

# Import our modular pipeline components
from src.config    import DEFAULT_PDF_PATH, HYBRID_TOP_K, RERANK_TOP_N
from src.ingestion import build_indexes
from src.retriever import hybrid_rrf_retrieve, rerank_candidates
from src.generator import generate_grounded_answer

# ---------------------------------------------------------------------------
# Global state — holds the currently indexed collection and BM25 object.
# Gradio runs in a single process, so module-level globals are safe here.
# ---------------------------------------------------------------------------

_state = {
    "collection" : None,
    "bm25_index" : None,
    "chunks"     : None,
    "pdf_name"   : None,
}


def _is_indexed() -> bool:
    """Returns True if a PDF has been successfully indexed."""
    return _state["collection"] is not None


# ---------------------------------------------------------------------------
# Index a PDF (called at startup and when user uploads a new PDF)
# ---------------------------------------------------------------------------

def index_pdf(pdf_path: str, status_box) -> str:
    """
    Builds both indexes from the given PDF path.
    Updates the global _state dict so all query functions can use them.
    Returns a status string displayed in the UI.
    """
    try:
        col, bm25, chunks = build_indexes(pdf_path)
        _state["collection"] = col
        _state["bm25_index"] = bm25
        _state["chunks"]     = chunks
        _state["pdf_name"]   = os.path.basename(pdf_path)
        return (
            f"✅ **Indexed:** `{_state['pdf_name']}`  \n"
            f"📄 **Chunks:** {len(chunks)}  |  "
            f"🗂️ **Vectors:** {col.count()}"
        )
    except FileNotFoundError as e:
        return f"❌ File not found: {e}"
    except Exception as e:
        return f"❌ Indexing error: {type(e).__name__}: {e}"


def handle_pdf_upload(uploaded_file) -> str:
    """
    Gradio callback: called when the user uploads a PDF via the UI.
    `uploaded_file` is a temporary file path string provided by Gradio.
    """
    if uploaded_file is None:
        return "⚠️ No file uploaded."
    return index_pdf(uploaded_file, None)


# ---------------------------------------------------------------------------
# Main query pipeline
# ---------------------------------------------------------------------------

def run_query(user_query: str):
    """
    Executes the full two-stage pipeline and returns outputs for three tabs.

    Returns (answer_md, retrieval_md, benchmark_md)
    """
    # Guard: make sure an index has been built first
    if not _is_indexed():
        err = "⚠️ No PDF indexed yet. Upload a PDF or ensure the default PDF exists."
        return err, err, err

    if not user_query.strip():
        return "Please type an architectural question above.", "", ""

    # ---- Stage 1: Hybrid RRF retrieval ----
    candidates = hybrid_rrf_retrieve(
        query      = user_query,
        collection = _state["collection"],
        bm25_index = _state["bm25_index"],
        chunks     = _state["chunks"],
        top_k      = HYBRID_TOP_K,
    )

    # ---- Stage 2: Cross-Encoder reranking ----
    top_chunks = rerank_candidates(
        query      = user_query,
        candidates = candidates,
        top_n      = RERANK_TOP_N,
    )

    # ---- Stage 3: Grounded generation via Gemini ----
    answer = generate_grounded_answer(user_query, top_chunks)

    # ---- Format Tab 1: Answer ----
    answer_md = f"### 💡 Grounded Answer\n\n{answer}"

    # ---- Format Tab 2: Retrieval evidence (RRF score vs. Rerank score) ----
    retrieval_md = (
        "### 🔍 Retrieved Evidence — RRF Score vs. Cross-Encoder Rerank Score\n\n"
        "> Each row shows how **RRF fusion** ranked the candidate, then how the "
        "**Cross-Encoder** re-scored it. Notice how reranking reorders items — "
        "that's precision filtering in action.\n\n"
        "| # | Chunk ID | Pillar | Page | RRF Score | Rerank Score |\n"
        "|---|----------|--------|------|-----------|--------------|\n"
    )
    # Show ALL candidates so the user can visually see reranking effect
    for rank_after, c in enumerate(top_chunks, start=1):
        retrieval_md += (
            f"| {rank_after} | `{c['chunk_id']}` | {c['pillar']} | {c['page']} "
            f"| `{c['rrf_score']:.5f}` | **`{c['rerank_score']:.4f}`** |\n"
        )

    retrieval_md += "\n---\n\n### 📄 Full Chunk Texts\n\n"
    for i, c in enumerate(top_chunks, start=1):
        retrieval_md += (
            f"**[{i}]** `{c['chunk_id']}` · **{c['pillar']}** · Page {c['page']}  \n"
            f"RRF Score: `{c['rrf_score']:.5f}` → Rerank Score: **`{c['rerank_score']:.4f}`**\n\n"
            f"> {c['text'][:350]}{'...' if len(c['text']) > 350 else ''}\n\n---\n\n"
        )

    # ---- Format Tab 3: Benchmark table (loads from file if available) ----
    benchmark_md = _load_benchmark_tab()

    return answer_md, retrieval_md, benchmark_md


# ---------------------------------------------------------------------------
# Benchmark tab helper
# ---------------------------------------------------------------------------

def _load_benchmark_tab() -> str:
    """
    Loads the benchmark results markdown file if the evaluator has been run.
    If not yet run, shows a placeholder with instructions.
    """
    results_md_path = (
        os.path.join("data", "benchmark_results.md")
        if os.path.exists(os.path.join("data", "benchmark_results.md"))
        else os.path.join("Data", "benchmark_results.md")
    )
    if os.path.exists(results_md_path):
        with open(results_md_path, "r", encoding="utf-8") as f:
            return f.read()

    # Read in-domain/OOD counts from the benchmark JSON for the placeholder
    benchmark_path = (
        os.path.join("data", "eval_benchmark.json")
        if os.path.exists(os.path.join("data", "eval_benchmark.json"))
        else os.path.join("Data", "eval_benchmark.json")
    )
    n_total, n_ood = "?", "?"
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r") as f:
            bm = json.load(f)
        n_total = len(bm["questions"])
        n_ood   = sum(1 for q in bm["questions"] if q["expected_pillar"] == "NONE_OUT_OF_DOMAIN")

    return (
        "## 📊 Benchmark Evaluation (Naive RAG vs. Two-Stage Hybrid RAG)\n\n"
        f"> Benchmark file found: **{n_total} questions** ({n_ood} out-of-domain traps)\n\n"
        "**Results file not generated yet.**  "
        "Run the evaluator to populate this tab:\n\n"
        "```bash\n"
        "# Activate your venv first, then:\n"
        "python -m src.evaluator\n"
        "```\n\n"
        "Once complete, `data/benchmark_results.md` will be loaded here automatically.\n\n"
        "---\n\n"
        "### Expected Result Format (from your notebook's A/B test)\n\n"
        "| Evaluation Metric | Naive RAG (Baseline) | Hybrid RAG (This Engine) | Δ Impact |\n"
        "|---|:---:|:---:|---|\n"
        "| Citation Compliance | 0% | **75–100%** | Enforces `[Pillar, Page]` auditability |\n"
        "| Context Grounding Score | 60 / 100 | **85 / 100** | +25 pts reduction in hallucination |\n"
        "| Out-of-Domain Refusal Rate | 0% *(hallucinated)* | **100%** | Clean `INSUFFICIENT_CONTEXT` |\n"
    )


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    """Constructs and returns the Gradio Blocks application."""

    with gr.Blocks(
        theme       = gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        title       = "AWS Well-Architected Copilot",
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.Markdown(
            "# ☁️ AWS Well-Architected Copilot\n"
            "**Two-Stage Hybrid RAG Engine** — Dense (ChromaDB) + Sparse (BM25) "
            "→ RRF Fusion → Cross-Encoder Reranking → Gemini 2.5 Flash Lite"
        )

        # ── PDF Upload + Index Status ────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown(
                    "### 📁 PDF Source\n"
                    "Uses the default AWS PDF in `Data/` if present. "
                    "Upload a **custom PDF** to index any document on the fly."
                )
                pdf_upload   = gr.File(
                    label       = "Upload a custom PDF (optional)",
                    file_types  = [".pdf"],
                    type        = "filepath",
                )
                index_btn    = gr.Button("🔄 Re-Index Uploaded PDF", variant="secondary")

            with gr.Column(scale=4):
                index_status = gr.Markdown("_Index status will appear here..._")

        gr.Markdown("---")

        # ── Query Input ─────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=6):
                query_input = gr.Textbox(
                    label       = "🔎 Your AWS Architectural Question",
                    placeholder = "e.g., What are the best practices for IAM root user protection?",
                    lines       = 3,
                )
                submit_btn  = gr.Button("⚡ Analyze Architecture", variant="primary", size="lg")

                gr.Examples(
                    examples = [
                        "What are the core design principles of the Security Pillar?",
                        "How does the Reliability pillar recommend automated recovery?",
                        "What does the Cost Optimization pillar say about right-sizing EC2 instances?",
                        "What sustainability practices does AWS recommend to reduce idle resources?",
                        "How do I configure Azure Cosmos DB multi-region replication?",  # OOD trap
                    ],
                    inputs = query_input,
                    label  = "Example queries (last one is an out-of-domain trap)",
                )

        gr.Markdown("---")

        # ── Three-tab output area ────────────────────────────────────────────
        with gr.Tabs():

            with gr.TabItem("💡 Cited Answer"):
                answer_out = gr.Markdown(
                    value = "_Submit a query to see the grounded, cited answer here._"
                )

            with gr.TabItem("🔍 Retrieval Evidence"):
                retrieval_out = gr.Markdown(
                    value = (
                        "_Submit a query to see retrieved chunks with their "
                        "**RRF score** (before reranking) vs. **Cross-Encoder score** "
                        "(after reranking). The score delta shows reranking precision gains._"
                    )
                )

            with gr.TabItem("📊 Benchmark: Naive vs. Hybrid RAG"):
                benchmark_out = gr.Markdown(value=_load_benchmark_tab())

        # ── Event wiring ────────────────────────────────────────────────────

        # Upload → re-index
        index_btn.click(
            fn      = handle_pdf_upload,
            inputs  = [pdf_upload],
            outputs = [index_status],
        )

        # Submit query → run full pipeline → populate all three tabs
        submit_btn.click(
            fn      = run_query,
            inputs  = [query_input],
            outputs = [answer_out, retrieval_out, benchmark_out],
        )

        # Also trigger on Enter key in the textbox
        query_input.submit(
            fn      = run_query,
            inputs  = [query_input],
            outputs = [answer_out, retrieval_out, benchmark_out],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  AWS Well-Architected Copilot — Hybrid RAG Engine")
    print("=" * 60)

    # Auto-index the default PDF at startup if it exists
    if os.path.exists(DEFAULT_PDF_PATH):
        print(f"[app] Auto-indexing default PDF: {DEFAULT_PDF_PATH}")
        status = index_pdf(DEFAULT_PDF_PATH, None)
        print(f"[app] {status}")
    else:
        print(
            f"[app] Default PDF not found at '{DEFAULT_PDF_PATH}'.\n"
            "      Upload a PDF via the UI, or place your PDF at that path."
        )

    # Build and launch the Gradio app
    app = build_ui()
    app.launch(
        server_name = "0.0.0.0",   # Listen on all interfaces (access from LAN)
        server_port = 7860,         # Default Gradio port
        share       = False,        # Set True to get a public tunnel URL
        debug       = False,
    )
