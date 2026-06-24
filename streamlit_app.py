"""AeroMech — Streamlit web app (Cessna 172 Agentic RAG).

Deployable free on Streamlit Community Cloud. Mirrors the verified Colab
pipeline: BGE embeddings + NumPy vector search + BM25 + RRF + cross-encoder
rerank + a LangGraph safety/retrieve/generate agent on Google Gemini.

Set the GOOGLE_API_KEY in Streamlit Cloud's app Secrets (Settings → Secrets):
    GOOGLE_API_KEY = "AIza..."
"""
import os
import glob
import pathlib

import numpy as np
import streamlit as st

# ------------------------------------------------------------------ config
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE, CHUNK_OVERLAP = 800, 120
TOP_K_VECTOR, TOP_K_BM25, TOP_K_FINAL = 8, 8, 4

INFLIGHT_REFUSAL = (
    "⚠️ This assistant is for GROUND-BASED maintenance and training only. "
    "If you are experiencing an in-flight emergency, do NOT consult this tool. "
    "Fly the aircraft, run your memorized checklist, contact ATC. "
    "Aviate, Navigate, Communicate."
)

INFLIGHT_SIGNALS = [
    "right now", "currently", "in flight", "in the air", "engine just",
    "losing altitude", "going down", "mid air", "midair", "airborne and",
    "happening now",
]

SAMPLE = """# Cessna 172 Skyhawk Knowledge Base (SYNTHETIC SAMPLE)

> Synthetic sample for a software demo. NOT an official POH.

## Engine Overview
The Cessna 172S is powered by a Lycoming IO-360-L2A engine producing 180
horsepower at 2700 RPM. Normal oil pressure in cruise is 50 to 90 PSI. Oil
temperature should stay below 245 degrees Fahrenheit.

## Engine Failure Training Reference (ABCDE)
The memorized flow taught for engine failure is ABCDE: Airspeed (best glide
about 68 knots), Best field, Checklist, Declare (squawk 7700 and broadcast on
121.5), Execute. This is a TRAINING reference only; in a real emergency pilots
rely on memorized procedures, not external devices.

## Rough Engine / Partial Power Loss
Causes include fouled spark plugs, a stuck valve, or fuel contamination. Ground
check: a normal magneto drop is 125 RPM maximum, with no more than 50 RPM
differential between the left and right magnetos.

## Low Oil Pressure
A reading below 20 PSI indicates a serious problem; do not operate the engine.
Causes include a failed oil pump, clogged filter, or low oil. Minimum oil for
flight is 5 quarts; capacity is 8 quarts.

## Fuel System
The aircraft holds 56 gallons total, 53 usable, across two wing tanks. The fuel
selector positions are LEFT, RIGHT, and BOTH; normal setting is BOTH. Fuel grade
is 100LL aviation gasoline.

## Magneto / Ignition
Dual magnetos provide redundancy. During runup at 1800 RPM a dead magneto (zero
RPM drop) is a no-go item and requires maintenance.
"""


# ------------------------------------------------------------------ helpers
def tokenize(t):
    return [x for x in t.lower().split() if x]


def inflight(text):
    t = text.lower()
    return any(s in t for s in INFLIGHT_SIGNALS)


def rrf(rank_lists, k=60):
    s = {}
    for ranks in rank_lists:
        for r, idx in enumerate(ranks):
            s[idx] = s.get(idx, 0.0) + 1.0 / (k + r + 1)
    return s


# ------------------------------------------------------------------ build (cached)
@st.cache_resource(show_spinner="Loading models and building index...")
def build_pipeline():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    # ensure data exists (sample if no PDFs supplied in the repo's data/raw)
    raw = pathlib.Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    if not any(raw.iterdir()):
        (raw / "cessna172_sample.md").write_text(SAMPLE)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "])

    files = (glob.glob("data/raw/*.pdf") + glob.glob("data/raw/*.md")
             + glob.glob("data/raw/*.txt"))
    chunks = []
    for f in files:
        name = pathlib.Path(f).name
        if f.endswith(".pdf"):
            pages = [(p.extract_text() or "", i + 1)
                     for i, p in enumerate(PdfReader(f).pages)]
        else:
            pages = [(pathlib.Path(f).read_text(encoding="utf-8"), None)]
        for text, page in pages:
            for j, piece in enumerate(splitter.split_text(text)):
                if piece.strip():
                    cid = f"{name}_p{page}_c{j}" if page else f"{name}_c{j}"
                    chunks.append({"id": cid, "text": piece.strip(),
                                   "source": name, "page": page or -1})

    doc_texts = [c["text"] for c in chunks]
    embedder = SentenceTransformer(EMBED_MODEL)
    doc_vecs = np.asarray(
        embedder.encode(doc_texts, normalize_embeddings=True), dtype=np.float32)
    bm25 = BM25Okapi([tokenize(t) for t in doc_texts])
    reranker = CrossEncoder(RERANK_MODEL)
    return chunks, doc_texts, embedder, doc_vecs, bm25, reranker


def retrieve(query, pipe, top_k=TOP_K_FINAL):
    chunks, doc_texts, embedder, doc_vecs, bm25, reranker = pipe
    qv = np.asarray(embedder.encode([query], normalize_embeddings=True),
                    dtype=np.float32)[0]
    sims = doc_vecs @ qv
    vec_rank = list(np.argsort(-sims)[:TOP_K_VECTOR])

    bscores = bm25.get_scores(tokenize(query))
    bm_rank = list(np.argsort(-bscores)[:TOP_K_BM25])

    fused = rrf([vec_rank, bm_rank])
    cand = sorted(fused, key=lambda x: fused[x], reverse=True)
    pairs = [(query, doc_texts[i]) for i in cand]
    rr = reranker.predict(pairs)
    order = list(np.argsort(-rr)[:top_k])
    return [{"text": doc_texts[cand[i]], "source": chunks[cand[i]]["source"],
             "page": chunks[cand[i]]["page"], "score": float(rr[i])}
            for i in order]


def kb_search(query, pipe):
    res = retrieve(query, pipe)
    if not res:
        return "No relevant passages found.", res
    ctx = "\n\n".join(
        f"[{i}] ({r['source']}"
        + (f", p.{r['page']}" if r['page'] and r['page'] > 0 else "")
        + f")\n{r['text']}" for i, r in enumerate(res, 1))
    return ctx, res


SYSTEM = ("You are AeroMech, a ground-based maintenance and training assistant for "
          "the Cessna 172. Answer ONLY from the CONTEXT. If the answer is not there, "
          "say you don't have that information and suggest the official POH. Cite "
          "bracketed sources like [1]. Never give in-flight emergency advice. Be "
          "concise and use exact figures from the context.")


def generate(question, context):
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1,
                                 google_api_key=os.environ["GOOGLE_API_KEY"])
    prompt = (f"{SYSTEM}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:")
    return llm.invoke(prompt).content


def answer(question, pipe):
    if inflight(question):
        return INFLIGHT_REFUSAL, None
    context, res = kb_search(question, pipe)
    return generate(question, context), res


# ------------------------------------------------------------------ UI
st.set_page_config(page_title="AeroMech — Cessna 172 RAG", page_icon="✈️")
st.title("✈️ AeroMech")
st.caption("Agentic RAG assistant for Cessna 172 **ground-based** maintenance & training.")
st.warning(
    "**Training and maintenance use only.** Not for in-flight decision-making. "
    "In a real emergency: Aviate, Navigate, Communicate.")

# Move the API-key check here so a missing secret shows a friendly message.
if "GOOGLE_API_KEY" not in os.environ:
    # Streamlit Cloud injects secrets into st.secrets; mirror into env.
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

pipe = build_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

if q := st.chat_input("Ask about specs, troubleshooting, checklists..."):
    st.session_state.history.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        if "GOOGLE_API_KEY" not in os.environ:
            st.error("GOOGLE_API_KEY not set. Add it in Settings → Secrets.")
        else:
            with st.spinner("Thinking..."):
                ans, res = answer(q, pipe)
            st.markdown(ans)
            if res:
                with st.expander("📄 Retrieved sources"):
                    for i, r in enumerate(res, 1):
                        st.markdown(f"**[{i}]** ({r['source']}) — score {r['score']:.2f}")
                        st.text(r["text"])
            st.session_state.history.append(("assistant", ans))

with st.sidebar:
    st.header("Try asking")
    for ex in [
        "What is normal oil pressure in cruise?",
        "What engine powers the 172S?",
        "What's the maximum magneto drop during runup?",
        "What does the ABCDE flow stand for?",
    ]:
        st.markdown(f"- {ex}")
    st.divider()
    st.caption("BGE · NumPy · BM25 · CrossEncoder · LangGraph · Gemini")
