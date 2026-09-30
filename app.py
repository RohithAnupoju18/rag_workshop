"""Chat with Documents - local RAG workshop app.   Run:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from core import model_selector, ollama_client
from core.document_loader import DocumentError
from core.embeddings import EmbeddingError, EmbeddingModel
from core.rag_pipeline import RAGPipeline
from ui import components
from ui.styles import CSS

st.set_page_config(page_title="Chat with Documents", page_icon="📄", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_embedder() -> EmbeddingModel:
    return EmbeddingModel()  # loaded once, shared across reruns


def init_state() -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(embedder=get_embedder())
        st.session_state.messages = []
        st.session_state.reports = {}


def detect_ollama() -> tuple[bool, list[str]]:
    if not ollama_client.is_running():
        return False, []
    try:
        return True, ollama_client.list_models()
    except ollama_client.OllamaError:
        return False, []


init_state()
pipeline: RAGPipeline = st.session_state.pipeline
connected, installed = detect_ollama()
auto_choice = model_selector.resolve_model(installed)

# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("Settings")
    st.subheader("Model")
    options = ["Auto (Qwen -> Llama)"] + installed
    picked = st.selectbox("Ollama model", options, help="Auto prefers Qwen and falls back to Llama.")
    active_model = auto_choice.model if picked == options[0] else picked
    active_family = model_selector.family_of(active_model)
    if st.button("Refresh models"):
        st.rerun()

    st.subheader("Retrieval")
    top_k = st.slider("Top-K chunks", 1, 8, 3)
    threshold = st.slider("Similarity threshold", 0.0, 0.9, 0.25, 0.05,
                          help="Chunks scoring below this are ignored.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 800, 50)
    overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 25)
    st.caption("Changing chunk settings? Click 'Process document' again.")

    st.subheader("Index status")
    st.write(f"Documents: **{len(pipeline.store.sources)}**")
    st.write(f"Chunks: **{len(pipeline.store)}**")
    st.write("Embeddings: **Ready**" if pipeline.is_indexed else "Embeddings: **Not created**")
    st.write("Retrieval: **Ready**" if pipeline.is_indexed else "Retrieval: **Waiting for a document**")
    if st.button("Clear index and chat"):
        pipeline.reset()
        st.session_state.messages = []
        st.session_state.reports = {}
        st.rerun()

components.header(connected, active_model, active_family, pipeline.is_indexed)

# ---------------- Ollama problems: friendly help, never a crash ----------------
if not connected:
    if not ollama_client.is_installed():
        st.error("Ollama is not installed. Download it from https://ollama.com/download, then reopen this app.")
    else:
        st.error("Ollama is installed but not running. Open the Ollama app or run `ollama serve` in a terminal.")
elif not auto_choice.ok and picked == options[0]:
    st.warning(auto_choice.message)
elif picked == options[0]:
    st.info(auto_choice.message)

# ---------------- Step 1: upload + process ----------------
st.subheader("1. Upload a document")
uploaded = st.file_uploader("PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        st.caption(f"{f.name} - {f.size / 1024:.1f} KB")
    if st.button("Process document(s)", type="primary"):
        for f in uploaded:
            bar = st.progress(0.0, text=f"{f.name}: starting...")
            try:
                report = pipeline.index_document(
                    f.getvalue(), f.name, chunk_size, overlap,
                    progress=lambda frac, msg, b=bar, n=f.name: b.progress(frac, text=f"{n}: {msg}"),
                )
                st.session_state.reports[f.name] = report
            except (DocumentError, ValueError, EmbeddingError) as exc:
                bar.empty()
                st.error(f"{f.name}: {exc}")
            except Exception:  # last-resort guard so the UI never shows a stack trace
                bar.empty()
                st.error(f"{f.name}: something went wrong while processing this file. Try another file.")

for name, r in st.session_state.reports.items():
    note = " (already processed - reused cache)" if r.cached else ""
    st.success(
        f"{name}: {r.page_count} page(s), {r.char_count:,} characters, "
        f"{r.chunk_count} chunks - Indexed{note}"
        if not r.cached else f"{name}: {r.chunk_count} chunks - Indexed{note}"
    )

# ---------------- Step 2: chat ----------------
st.subheader("2. Ask questions")
if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("model"):
                st.caption(f"Generated by: {msg['model']}")
            components.retrieval_panel(msg.get("retrieved", []), key=f"hist{i}")

question = st.chat_input("Ask something about your document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving relevant chunks and asking the local model..."):
                result = pipeline.ask(question, active_model, top_k, threshold, temperature)
            st.markdown(result.answer)
            if result.used_llm:
                st.caption(f"Generated by: {result.model}")
            components.retrieval_panel(result.retrieved, key="new")
            st.session_state.messages.append({
                "role": "assistant", "content": result.answer,
                "model": result.model if result.used_llm else None, "retrieved": result.retrieved,
            })
        except EmbeddingError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Something went wrong while answering. Check that Ollama is running and try again.")

with st.expander("Learn: how does this work?"):
    st.markdown(
        "- **Embeddings** turn text into numbers that capture meaning.\n"
        "- **Cosine similarity** measures the angle between two vectors: smaller angle = more similar meaning.\n"
        "- **Retrieval** picks the Top-K most similar chunks. **Generation** is a separate step where the LLM "
        "writes an answer using only those chunks.\n"
        "- The LLM does *not* automatically know your document. It only sees what retrieval hands it.\n"
        "- **Security:** uploaded documents are untrusted data and cannot override the system instructions."
    )
