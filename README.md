<div align="center">

# ☁️ Enterprise Two-Stage Hybrid RAG Engine
### *for Cloud Architecture*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-22C55E?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-F59E0B?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Reranker](https://img.shields.io/badge/Reranker-Cross--Encoder-F97316?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash_Lite-8B5CF6?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)

*An enterprise-grade, citation-enforced RAG system built over the **850+ page AWS Well-Architected Framework** — engineered to eliminate vocabulary mismatch, context dilution, and hallucinated recommendations.*

</div>

---

## 📌 Problem Statement

Most tutorial-level RAG implementations rely on a **single bi-encoder dense vector index** (cosine similarity over embeddings). In mission-critical cloud architecture, this causes three severe failure modes:

| Failure Mode | Description | Impact |
|---|---|---|
| **Vocabulary Mismatch** | Dense embeddings blur exact technical acronyms (IAM, KMS, SCP) into broad concepts | Wrong passages retrieved for exact regulatory queries |
| **Context Dilution** | 10+ unranked chunks dilute LLM attention, increasing token cost and noise | Synthesized answers based on peripheral, irrelevant text |
| **Auditability Blindspots** | No page/pillar citations in generated responses | Output inadmissible for infrastructure audits & compliance reviews |

---

## 🏗️ System Architecture

```
                        User Architecture Query
                                  │
           ┌──────────────────────┴──────────────────────┐
           ▼                                             ▼
[ Dense Semantic Retrieval ]               [ Sparse Lexical Retrieval ]
  ChromaDB Vector Store                      BM25Okapi Token Index
 (all-MiniLM-L6-v2 Embedding)              (Exact Token & Acronym Match)
      Top 25 Candidates                          Top 25 Candidates
           │                                             │
           └──────────────────────┬──────────────────────┘
                                  ▼
                [ Reciprocal Rank Fusion (RRF, k=60) ]
                  Score(d) = Σ 1/(k + rank_m(d))
                          Top 10 Candidates
                                  │
                                  ▼
            [ Cross-Encoder Pairwise Re-Ranking Stage ]
           (cross-encoder/ms-marco-MiniLM-L-6-v2 Model)
                   Top 3–5 High-Precision Chunks
                                  │
                                  ▼
               [ Deterministic Grounded Generation ]
                   Gemini 2.5 Flash Lite | Temp = 0.0
            Enforced Citation: [Pillar: <Name>, Page: <Num>]
                                  │
                                  ▼
                  Grounded Answer + Live Audit Trail
```

---

## 📂 Repository Structure

```
Enterprise-Two-Stage-Hybrid-RAG-Engine/
│
├── notebooks/
│   └── AWS_Well_Architected_RAG.ipynb   # Original single-file prototype (preserved)
│
├── src/                                 # Modular Python package
│   ├── __init__.py
│   ├── config.py       # All tunable constants (CHUNK_SIZE, models, RRF_K, etc.)
│   ├── ingestion.py    # PDF loading, pillar detection, chunking, dual-index build
│   ├── retriever.py    # Dense + Sparse retrieval, RRF fusion, Cross-Encoder rerank
│   ├── generator.py    # Gemini API call, citation-enforced prompt, refusal guardrails
│   └── evaluator.py    # LLM-as-a-Judge A/B benchmark: Naive RAG vs. Hybrid RAG
│
├── data/
│   ├── eval_benchmark.json   # 20 questions: 15 in-domain + 5 OOD traps
│   ├── benchmark_results.csv # Generated after running python -m src.evaluator
│   └── benchmark_results.md  # Markdown summary table (auto-loaded in app Tab 3)
│
├── Data/
│   └── wellarchitected-framework-2023-04-10.pdf   # Default source document
│
├── app.py              # Gradio web UI — imports directly from src/
├── requirements.txt    # Pinned dependencies (install into .venv)
├── .env.example        # Copy to .env and add your GEMINI_API_KEY
├── .gitignore          # Excludes .venv/, .env, __pycache__, etc.
└── README.md
```

---

## ⚡ 3-Step Local Setup (using venv — no global installs)

### Step 1 — Clone & create a virtual environment

```bash
git clone https://github.com/manojmulammagari/Enterprise-Two-Stage-Hybrid-RAG-Engine-for-Cloud-Architecture.git
cd Enterprise-Two-Stage-Hybrid-RAG-Engine-for-Cloud-Architecture

# Create an isolated .venv inside the project folder
python -m venv .venv

# Activate it:
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Command Prompt)
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate

# Your prompt should now show (.venv) — all pip installs go here, not globally
```

### Step 2 — Install dependencies & set your API key

```bash
# Install all packages into .venv (never touches your global Python)
pip install -r requirements.txt

# Copy the example env file and add your Gemini API key
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux

# Edit .env and replace the placeholder:
# GEMINI_API_KEY=your-real-key-here
# Get a free key at: https://aistudio.google.com/
```

### Step 3 — Launch the app

```bash
python app.py
# Then open:  http://localhost:7860
```

> **To deactivate the venv when you're done:**
> ```bash
> deactivate
> ```

---

## 🔬 Running the Evaluation Benchmark

```bash
# Make sure your .venv is active, then:
python -m src.evaluator

# Results are exported to:
#   data/benchmark_results.csv   (per-question detail)
#   data/benchmark_results.md    (summary table — auto-loaded in app Tab 3)
```

---

## 📊 Benchmark Results (A/B Test: Naive vs. Hybrid RAG)

Evaluated against 20 benchmark questions: 15 in-domain technical questions requiring citations, and 5 out-of-domain trap questions designed to test refusal accuracy.

| Evaluation Metric | Naive RAG (Baseline) | **Hybrid RAG (This Engine)** | Δ Impact |
|---|:---:|:---:|---|
| Citation Compliance | 0% | **75–100%** | Enforces `[Pillar, Page]` auditability |
| Context Grounding Score | 60 / 100 | **85 / 100** | +25 pts reduction in hallucination |
| Out-of-Domain Refusal Rate | 0% *(hallucinated)* | **100%** | Clean `INSUFFICIENT_CONTEXT` trigger |

*Run `python -m src.evaluator` to reproduce these results with your API key.*

---

## 🚀 Key Engineering Highlights

### 1. Pillar-Aware Ingestion & Sliding-Window Chunking
- Parsed **851 pages** of official AWS documentation into **2,494 structured document chunks**
- **900-character sliding window** with 150-character overlap prevents multi-step guidance truncation
- Each chunk is stamped with AWS pillar metadata and exact page numbers at parse time

### 2. Two-Stage Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

Merged dense semantic search and sparse lexical search using rank-based RRF:

$$\text{Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)} \quad (k=60)$$

> This eliminates arbitrary hyperparameter weighting between disparate cosine distances and unbounded BM25 scores — both are rank-normalized.

### 3. Cross-Encoder Re-Ranking Stage
- Routes top-10 RRF candidates through `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Full query-document self-attention detects semantic false positives that bi-encoders miss
- Prunes to top-5 highest-scoring contexts with configurable `RERANK_THRESHOLD`

### 4. Grounded Synthesis & Refusal Guardrails
- Deterministic generation (`temperature=0.0`) enforcing `[Pillar: <Name>, Page: <Num>]` citations
- Strict `INSUFFICIENT_CONTEXT` refusal for out-of-corpus queries (Azure, GCP, Terraform, etc.)

### 5. Three-Tab Gradio UI
- **Tab 1** — Cited grounded answer
- **Tab 2** — Full retrieval evidence with RRF score vs. Cross-Encoder score side-by-side
- **Tab 3** — Live benchmark comparison table (Naive vs. Hybrid RAG)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language & Runtime | Python 3.10+ (isolated via `.venv`) |
| PDF Extraction | `pypdf >= 4.0.0` |
| Vector Store | `chromadb >= 0.4.24` |
| Embedding Model | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Lexical Index | `rank-bm25` — BM25Okapi |
| Re-Ranking Model | `CrossEncoder` — `ms-marco-MiniLM-L-6-v2` |
| LLM Engine | Google Gemini API — `gemini-2.5-flash-lite` via `google-genai` |
| Application Interface | `gradio >= 4.0.0` |
| Secret Management | `python-dotenv` — reads `GEMINI_API_KEY` from `.env` |

---

## 🔎 Sample Queries & Expected Behavior

| Query Type | Input Prompt | Expected Behavior |
|---|---|---|
| **Exact Keyword / Acronym** | *"What best practices are specified for AWS KMS keys and root credential protection?"* | BM25 prioritizes KMS passages; Cross-Encoder filters; cites Security pillar pages |
| **Broad Architectural Concept** | *"What are the foundational design principles of the Reliability pillar?"* | Dense search retrieves high-level principles; citations from Reliability section |
| **Hallucination / Guardrail Test** | *"How do I configure Microsoft Azure Cosmos DB multi-region replication?"* | Returns `INSUFFICIENT_CONTEXT` — no fabricated AWS-to-Azure mapping |

---

<div align="center">
  <sub>Built with ❤️ over the AWS Well-Architected Framework · <a href="https://github.com/manojmulammagari">@manojmulammagari</a></sub>
</div>
