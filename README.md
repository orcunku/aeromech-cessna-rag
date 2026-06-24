## 🔗 Live demo
Try it: https://aeromech-cessna-rag-eapqkp7lfbvjiq3ysxab3c.streamlit.app

# aeromech-cessna-rag
Agentic RAG assistant for Cessna 172 ground-based maintenance &amp; training

# ✈️ AeroMech — Agentic RAG Assistant for the Cessna 172

An **agentic Retrieval-Augmented Generation (RAG)** system that answers questions about
Cessna 172 Skyhawk **maintenance, troubleshooting, and training** — grounded in source
documents, with citations, and a built-in safety guardrail. Built entirely on **free,
open-source tools**.

> **Scope & safety:** This is a *ground-based* maintenance and training study tool. It is
> **not** for in-flight decision-making and deliberately refuses to act as an in-flight
> emergency device. In a real emergency: *Aviate, Navigate, Communicate.*

---

## What it does

Ask a natural-language question (e.g. *"What is normal oil pressure in cruise?"*) and the
system retrieves the most relevant passages from the knowledge base, then generates a
concise, **cited** answer grounded only in those passages. If a question looks like an
in-flight emergency, a safety gate intercepts it and refuses, redirecting the user to
proper emergency procedure.

**Example:**

```
Q: What is normal oil pressure in cruise?
A: Normal oil pressure in cruise is 50 to 90 PSI [1].

Q: My engine just failed in flight right now, what do I do?
A: This assistant is for GROUND-BASED maintenance and training only...
   Aviate, Navigate, Communicate.
```

---

## Why this project

This demonstrates a complete, modern AI-engineering pipeline rather than a single-call demo:

- **Hybrid retrieval** — dense semantic search (BGE embeddings + NumPy cosine similarity)
  fused with sparse keyword search (**BM25**) via **Reciprocal Rank Fusion**, then
  **cross-encoder reranking** for precision. Hybrid retrieval handles exact tokens like
  part numbers (`IO-360-L2A`) that pure vector search often misses.
- **Agentic orchestration** with **LangGraph** — a stateful graph that routes between a
  safety gate, retrieval, and grounded generation.
- **Evaluation** — retrieval hit-rate and MRR against a gold question set. Measuring
  retrieval quality is a step most demo projects skip.
- **Safety guardrails** — explicit in-flight-emergency detection and refusal.
- **Grounded, cited answers** — the model answers only from retrieved context and cites
  its sources, reducing hallucination.

---

## Architecture

```
                 ┌──────────────┐
   question ───► │  safety gate │──(in-flight)──► refuse ──► END
                 └──────┬───────┘
                        │ (ok)
                        ▼
                  ┌───────────┐    hybrid retrieval:
                  │ retrieve  │ ◄─ dense (BGE + NumPy) + sparse (BM25)
                  └─────┬─────┘    → RRF fusion → cross-encoder rerank
                        ▼
                  ┌───────────┐    Gemini, grounded in context, with citations
                  │ generate  │
                  └─────┬─────┘
                        ▼
                       END
```

**Stack:** Google Gemini (free tier) · sentence-transformers (BGE) · NumPy · rank-bm25 ·
cross-encoder reranker · LangGraph · Streamlit

---

## Results

Retrieval evaluation on a synthetic gold question set (n = 6):

| Metric    | Score |
|-----------|-------|
| Hit-rate  | 1.00  |
| MRR       | 1.00  |

*These scores validate that the end-to-end pipeline is wired correctly. The gold set is
small and drawn from the synthetic sample document, so a perfect score is expected; the
next step is evaluating against a larger, independently-authored question set over a real
Pilot's Operating Handbook.*

---

## Run it

The project runs end-to-end in Google Colab on the free tier — no GPU or local setup needed.

1. **Get a free Gemini API key** at [Google AI Studio](https://aistudio.google.com/app/apikey)
   → *Create API key* → copy it.
2. **Open `AeroMech_Colab.ipynb` in Google Colab** (Upload, or File → Open).
3. In Colab, click the **🔑 Secrets** icon (left sidebar) → add a secret named
   `GOOGLE_API_KEY` with your key, and enable *Notebook access*.
4. **Runtime → Run all.**

To use your own data, upload Cessna 172 POH PDFs to `/content/data/raw/` and re-run.
A synthetic sample is included so it works out of the box.

---

## How it works (technical detail)

1. **Ingest & chunk** — documents (PDF/Markdown) are split into overlapping chunks with a
   recursive character splitter that respects document structure.
2. **Index** — each chunk is embedded with `BAAI/bge-small-en-v1.5` into a NumPy matrix;
   a BM25 index is built over the same chunks for keyword search.
3. **Retrieve** — a query runs through both indexes. The two ranked lists are merged with
   Reciprocal Rank Fusion, and the candidates are reranked by a cross-encoder for final
   precision.
4. **Generate** — the top passages are passed as context to Gemini with a strict prompt:
   answer only from context, cite sources, never give in-flight advice.
5. **Guard** — before retrieval, a gate checks the query for in-flight-emergency signals
   and short-circuits to a refusal if detected.

---

## Roadmap

- Evaluate against a larger, independently-written question set over a real POH.
- Add generation-quality metrics (faithfulness, answer relevancy).
- Add a web-search tool for current Airworthiness Directives.
- Deploy a permanent Streamlit demo.

---

## Disclaimer

Uses synthetic sample data. Not affiliated with Cessna or Textron Aviation. Not for
operational flight or maintenance use. For educational and portfolio purposes only.

