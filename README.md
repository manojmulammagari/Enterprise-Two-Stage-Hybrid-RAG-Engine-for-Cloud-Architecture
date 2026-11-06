# ☁️ Enterprise Two-Stage Hybrid RAG Engine for Cloud Architecture

An end-to-end, production-grade Retrieval-Augmented Generation (RAG) system engineered over the 100+ page **AWS Well-Architected Framework** whitepaper. Built to eliminate hallucinations, solve technical acronym mismatch (e.g., IAM, KMS, S3, SCP), and guarantee verifiable `[Pillar, Page]` audit trails.

---

## 📌 Problem Statement

Standard "Naive RAG" tutorials rely solely on bi-encoder dense vector similarity (cosine distance). In real enterprise engineering, this breaks down:
1. **Vocabulary Mismatch:** Pure embeddings blur exact cloud service names and acronyms into general concepts, missing critical infrastructure guidance.
2. **Lost in the Middle:** Stuffing 10+ raw chunks into an LLM context degrades generation quality and increases latency and token costs.
3. **Hallucination & Lack of Auditability:** Standard prompts allow the LLM to invent commands or extrapolate without citing exact page numbers.

This engine solves these challenges using a **Two-Stage Hybrid Search & Re-Ranking Architecture**.

---

## 🛠️ System Architecture

```text
                             User Architecture Query
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
       [ Dense Semantic Retrieval ]            [ Sparse Lexical Retrieval ]
         ChromaDB Vector Store                   BM25Okapi Token Index
        (all-MiniLM-L6-v2) Top 25                      Top 25 Candidates
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                     [ Reciprocal Rank Fusion (RRF, k=60) ]
                               Top 10 Candidates
                                        │
                                        ▼
                  [ Cross-Encoder Re-Ranking Stage ]
                  (ms-marco-MiniLM-L-6-v2 Pairwise Scoring)
                        Top 3 High-Precision Chunks
                                        │
                                        ▼
                    [ Deterministic Grounded Generation ]
                        Gemini 2.0 Flash (Temp = 0.0)
                  Strict Citation Contract: [Pillar, Page]
                                        │
                                        ▼
                      Grounded Answer + Live Audit Trail
