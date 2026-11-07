# ☁️ Enterprise Two-Stage Hybrid RAG Engine for Cloud Architecture

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Vector DB - Chroma](https://img.shields.io/badge/Vector_DB-ChromaDB-green.svg)](https://www.trychroma.com/)
[![Embeddings - HuggingFace](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Reranker - Cross--Encoder](https://img.shields.io/badge/Reranker-ms--marco--MiniLM--L--6--v2-orange.svg)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
[![LLM - Gemini 2.0 Flash](https://img.shields.io/badge/Generation-Gemini_2.0_Flash-blueviolet.svg)](https://ai.google.dev/)
[![UI - Gradio](https://img.shields.io/badge/Demo-Gradio_Live-orange.svg)](https://gradio.app/)

An enterprise-grade, two-stage Retrieval-Augmented Generation (RAG) system engineered over the **850+ page AWS Well-Architected Framework** whitepaper. Built to solve critical enterprise information retrieval failures: dense-vector vocabulary mismatch on specialized cloud acronyms (e.g., IAM, KMS, S3, SCP), context dilution ("lost-in-the-middle"), and hallucinated recommendations lacking verifiable source citations.

---

## 📌 Problem Statement: Why Naive RAG Fails in Enterprise Cloud Ops

Most tutorial-level RAG implementations rely on a single bi-encoder dense vector index (cosine similarity over embeddings). In mission-critical cloud architecture, this leads to three severe failure modes:

1. **Vocabulary Mismatch:** Dense embeddings blur exact technical acronyms and error codes into broad concepts. A query for `AWS KMS envelope encryption` frequently retrieves generic encryption overviews rather than exact key rotation rules.
2. **Context Dilution ("Lost in the Middle"):** Supplying 10+ unranked chunks to an LLM dilutes attention, increases token latency/costs, and leads to synthesized answers based on noisy peripheral text.
3. **Auditability & Compliance Blindspots:** Standard conversational bots hallucinate architectural decisions without citing exact pages, making output inadmissible for infrastructure audits.

---

## 🛠️ System Architecture

```text
                                  User Architecture Query
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          [ Dense Semantic Retrieval ]                [ Sparse Lexical Retrieval ]
            ChromaDB Vector Store                       BM25Okapi Token Index
           (all-MiniLM-L6-v2 Embedding)                (Exact Token & Acronym Match)
                 Top 25 Candidates                           Top 25 Candidates
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             ▼
                          [ Reciprocal Rank Fusion (RRF, k=60) ]
                                    Top 10 Candidates
                                             │
                                             ▼
                       [ Cross-Encoder Pairwise Re-Ranking Stage ]
                      (cross-encoder/ms-marco-MiniLM-L-6-v2 Model)
                             Top 3-5 High-Precision Chunks
                                             │
                                             ▼
                         [ Deterministic Grounded Generation ]
                             Gemini 2.0 Flash (Temp = 0.0)
                    Enforced Prompt Contract: [Pillar: <Name>, Page: <Num>]
                                             │
                                             ▼
                           Grounded Answer + Live Audit Trail

🚀 Key Engineering Highlights1. Pillar-Aware Ingestion & Sliding-Window ChunkingParsed 851 pages of official AWS documentation into 2,494 structured document chunks.Implemented a 900-character sliding window with a 150-character overlap to prevent truncation of multi-step architectural guidance.Dynamically classified and stamped chunks with architectural pillar metadata (Operational Excellence, Security, Reliability, Performance Efficiency, Cost Optimization, Sustainability) alongside exact page numbers.2. Two-Stage Hybrid Retrieval with Reciprocal Rank Fusion (RRF)Merged dense semantic search (all-MiniLM-L6-v2 via ChromaDB) and sparse lexical search (BM25Okapi) using rank-based Reciprocal Rank Fusion:$$\text{Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)} \quad (k=60)$$Eliminates arbitrary hyperparameter weighting between disparate cosine distances ($[-1, 1]$) and unbounded BM25 scores.3. Cross-Encoder Re-Ranking StageRouted top 10 RRF candidates through cross-encoder/ms-marco-MiniLM-L-6-v2 for full query-document self-attention.Pruned candidates to the top 3–5 highest-scoring contexts, filtering out semantic false positives and resolving context fragmentation.4. Grounded Synthesis & Refusal GuardrailsConfigured deterministic generation (temperature=0.0) requiring explicit citations: [Pillar: <Name>, Page: <Num>].Implemented strict refusal logic (INSUFFICIENT_CONTEXT) when candidate cross-attention scores indicate out-of-corpus queries (e.g., Azure/GCP infrastructure questions).5. Interactive Gradio UI with Auditable Proof DrawerDeployed a responsive web interface exposing both the generated answer and a collapsible Audit Trail displaying retrieved text snippets, source page numbers, RRF fusion ranks, and raw cross-encoder relevance scores.📊 Benchmark & Evaluation Results (A/B Test)Evaluated against a synthetic architectural benchmark evaluating citation adherence, grounding precision, and out-of-domain hallucination resistance:Evaluation MetricBaseline Naive RAG (Vector-Only)Enterprise Hybrid RAG (This Engine)Delta / ImpactCitation Adherence0.0%75.0% - 100.0%Enforces [Pillar, Page] auditabilityContext Grounding Score60.0 / 10085.0 / 100+25.0 pts reduction in hallucinationOut-of-Domain Refusal Rate0.0% (hallucinated)100.0%Clean INSUFFICIENT_CONTEXT trigger📂 Repository StructurePlaintextEnterprise-Two-Stage-Hybrid-RAG-Engine-for-Cloud-Architecture/
├── notebooks/
│   └── AWS_Well_Architected_RAG.ipynb    # Full 8-step pipeline with executed outputs
├── data/                                 # Storage directory for source PDF
│   └── wellarchitected-framework-2023-04-10.pdf
├── requirements.txt                      # Pinned production dependencies
└── README.md                             # Architecture overview and documentation
⚡ Quickstart Guide1. Clone the RepositoryBashgit clone [https://github.com/manojmulammagari/Enterprise-Two-Stage-Hybrid-RAG-Engine-for-Cloud-Architecture.git](https://github.com/manojmulammagari/Enterprise-Two-Stage-Hybrid-RAG-Engine-for-Cloud-Architecture.git)
cd Enterprise-Two-Stage-Hybrid-RAG-Engine-for-Cloud-Architecture
2. Install DependenciesBashpip install -r requirements.txt
3. Configure API CredentialsSet your Google AI Studio API key in your terminal or environment:Bashexport GEMINI_API_KEY="your-gemini-api-key-here"
4. Run the PipelineOpen the notebook in Jupyter or Google Colab and run all cells:Bashjupyter notebook notebooks/AWS_Well_Architected_RAG.ipynb
🔍 Sample Queries & VerificationQuery TypeInput PromptExpected System BehaviorExact Keyword / Acronym"What best practices are specified for AWS KMS keys, secret rotation, and protecting data at rest?"BM25 prioritizes KMS encryption passages; cites Security pillar pages (e.g., pp. 325–329).Broad Architectural Concept"What are the foundational design principles of the Reliability pillar?"Semantic search + Cross-Encoder retrieve high-level principles; cites automated recovery and capacity limits.Hallucination / Guardrail Test"How do I configure Microsoft Azure Cosmos DB multi-region replication?"Pipeline triggers refusal rule and returns INSUFFICIENT_CONTEXT.🛠️ Tech Stack & DependenciesLanguage & Runtime: Python 3.10+PDF Extraction & Parsing: pypdfVector Store: chromadbEmbedding Model: sentence-transformers (all-MiniLM-L6-v2)Lexical Index: rank-bm25 (BM25Okapi)Re-Ranking Model: CrossEncoder (ms-marco-MiniLM-L-6-v2)LLM Engine: Google Gemini API (gemini-2.0-flash / google-genai SDK)Application Interface: gradio🔮 Roadmap[ ] Add Hypothetical Document Embeddings (HyDE) to expand short acronym-heavy questions before retrieval.[ ] Implement query routing with query intent decomposition for multi-part comparative questions.[ ] Export containerized Docker image with FastAPI backend endpoints for production cluster deployment.
