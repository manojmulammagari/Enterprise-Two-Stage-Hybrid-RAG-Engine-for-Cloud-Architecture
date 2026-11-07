<div align="center">

# ☁️ Enterprise Two-Stage Hybrid RAG Engine
### *for Cloud Architecture*

<p>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/Vector_DB-ChromaDB-22C55E?style=for-the-badge&logo=databricks&logoColor=white" alt="ChromaDB"/></a>
  <a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"><img src="https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-F59E0B?style=for-the-badge&logo=huggingface&logoColor=white" alt="Embeddings"/></a>
  <a href="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2"><img src="https://img.shields.io/badge/Reranker-Cross--Encoder-F97316?style=for-the-badge&logo=huggingface&logoColor=white" alt="Reranker"/></a>
  <a href="https://ai.google.dev/"><img src="https://img.shields.io/badge/LLM-Gemini_2.0_Flash-8B5CF6?style=for-the-badge&logo=google&logoColor=white" alt="Gemini 2.0 Flash"/></a>
  <a href="https://gradio.app/"><img src="https://img.shields.io/badge/UI-Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio"/></a>
</p>

<p><em>An enterprise-grade, citation-enforced RAG system built over the <strong>850+ page AWS Well-Architected Framework</strong> — engineered to eliminate vocabulary mismatch, context dilution, and hallucinated recommendations.</em></p>

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
                            Top 10 Candidates
                                    │
                                    ▼
              [ Cross-Encoder Pairwise Re-Ranking Stage ]
             (cross-encoder/ms-marco-MiniLM-L-6-v2 Model)
                     Top 3–5 High-Precision Chunks
                                    │
                                    ▼
                 [ Deterministic Grounded Generation ]
                     Gemini 2.0 Flash  |  Temp = 0.0
              Enforced Citation: [Pillar: <Name>, Page: <Num>]
                                    │
                                    ▼
                    Grounded Answer + Live Audit Trail
```

---

## 🚀 Key Engineering Highlights

### 1. Pillar-Aware Ingestion & Sliding-Window Chunking
- Parsed **851 pages** of official AWS documentation into **2,494 structured document chunks**
- Implemented a **900-character sliding window** with 150-character overlap to prevent multi-step guidance truncation
- Dynamically classified and stamped chunks with AWS architectural pillar metadata and exact page numbers:
  `Operational Excellence` · `Security` · `Reliability` · `Performance Efficiency` · `Cost Optimization` · `Sustainability`

### 2. Two-Stage Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

Merged dense semantic search and sparse lexical search using rank-based RRF:

$$\text{Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)} \quad (k=60)$$

> This eliminates arbitrary hyperparameter weighting between disparate cosine distances ($[-1, 1]$) and unbounded BM25 scores.

### 3. Cross-Encoder Re-Ranking Stage
- Routes top 10 RRF candidates through `cross-encoder/ms-marco-MiniLM-L-6-v2` for **full query-document self-attention**
- Prunes to top 3–5 highest-scoring contexts, filtering out semantic false positives and resolving context fragmentation

### 4. Grounded Synthesis & Refusal Guardrails
- Deterministic generation (`temperature=0.0`) enforcing explicit citations: `[Pillar: <Name>, Page: <Num>]`
- Strict `INSUFFICIENT_CONTEXT` refusal when candidate scores indicate out-of-corpus queries (e.g., Azure/GCP questions)

### 5. Interactive Gradio UI with Auditable Proof Drawer
- Responsive web interface exposing both the generated answer and a **collapsible Audit Trail**
- Audit Trail displays: retrieved text snippets · source page numbers · RRF fusion ranks · raw cross-encoder relevance scores

---

## 📊 Benchmark Results (A/B Test)

Evaluated against a synthetic architectural benchmark for citation adherence, grounding precision, and out-of-domain hallucination resistance:

| Evaluation Metric | Baseline Naive RAG (Vector-Only) | **Enterprise Hybrid RAG (This Engine)** | Δ Impact |
|---|:---:|:---:|---|
| Citation Adherence | 0.0% | **75.0% – 100.0%** | Enforces `[Pillar, Page]` auditability |
| Context Grounding Score | 60.0 / 100 | **85.0 / 100** | +25.0 pts reduction in hallucination |
| Out-of-Domain Refusal Rate | 0.0% *(hallucinated)* | **100.0%** | Clean `INSUFFICIENT_CONTEXT` trigger |

---

## 📂 Repository Structure

```
Enterprise-Two-Stage-Hybrid-RAG-Engine-for-Cloud-Architecture/
│
├── AWS_Well_Architected_RAG.ipynb   # Full 8-step pipeline with executed outputs
├── requirements.txt                        # Pinned production dependencies
└── README.md                               # Architecture overview & documentation
```

---

## ⚡ Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/manojmulammagari/Enterprise-Two-Stage-Hybrid-RAG-Engine-for-Cloud-Architecture.git
cd Enterprise-Two-Stage-Hybrid-RAG-Engine-for-Cloud-Architecture
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```
> Get your free key at [Google AI Studio](https://aistudio.google.com/).

### 4. Run the Pipeline
Open the notebook in Jupyter or Google Colab and execute all cells:
```bash
jupyter notebook AWS_Well_Architected_RAG.ipynb
```

---

## 🔍 Sample Queries & Verification

| Query Type | Input Prompt | Expected System Behavior |
|---|---|---|
| **Exact Keyword / Acronym** | *"What best practices are specified for AWS KMS keys, secret rotation, and protecting data at rest?"* | BM25 prioritizes KMS encryption passages; cites Security pillar pages (e.g., pp. 325–329) |
| **Broad Architectural Concept** | *"What are the foundational design principles of the Reliability pillar?"* | Semantic search + Cross-Encoder retrieve high-level principles; cites automated recovery and capacity limits |
| **Hallucination / Guardrail Test** | *"How do I configure Microsoft Azure Cosmos DB multi-region replication?"* | Pipeline triggers refusal rule and returns `INSUFFICIENT_CONTEXT` |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language & Runtime | Python 3.10+ |
| PDF Extraction | `pypdf >= 4.0.0` |
| Vector Store | `chromadb >= 0.4.24` |
| Embedding Model | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Lexical Index | `rank-bm25` — BM25Okapi |
| Re-Ranking Model | `CrossEncoder` — `ms-marco-MiniLM-L-6-v2` |
| LLM Engine | Google Gemini API — `gemini-2.0-flash` via `google-genai` |
| Application Interface | `gradio >= 4.0.0` |

---

## 🔮 Roadmap

- [ ] Add **Hypothetical Document Embeddings (HyDE)** to expand short acronym-heavy questions before retrieval
- [ ] Implement **query routing with intent decomposition** for multi-part comparative questions
- [ ] Export a **containerized Docker image** with FastAPI backend endpoints for production cluster deployment

---

<div align="center">
  <sub>Built with ❤️ over the AWS Well-Architected Framework · <a href="https://github.com/manojmulammagari">@manojmulammagari</a></sub>
</div>
