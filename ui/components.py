from __future__ import annotations

import html

import streamlit as st

from core.retriever import RetrievedChunk


def badge(text: str, kind: str = "gray") -> str:
    return f'<span class="badge b-{kind}">{html.escape(text)}</span>'


def header(connected: bool, model: str | None, family: str | None, indexed: bool) -> None:
    badges = [badge("Ollama: Connected" if connected else "Ollama: Not Connected", "green" if connected else "red")]
    if model:
        label = {"qwen": "Qwen", "llama": "Llama"}.get(family or "", "Model")
        badges.append(badge(f"{label} - {model}", "blue"))
    else:
        badges.append(badge("No model", "amber"))
    badges.append(badge("Indexed" if indexed else "Not Indexed", "green" if indexed else "gray"))
    st.markdown(
        '<div class="hero"><h1>Chat with Documents</h1>'
        "<p>Local RAG workshop app - PDF/DOCX &rarr; Retrieval &rarr; Ollama (Qwen / Llama). "
        "No API keys, fully offline after setup.</p>" + "".join(badges) + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pipeline"><b>RAG flow:</b> Document &rarr; Extract &rarr; Chunk &rarr; Embed &rarr; '
        "Compare (cosine) &rarr; Retrieve Top-K &rarr; Context &rarr; LLM &rarr; Answer</div>",
        unsafe_allow_html=True,
    )


def retrieval_panel(retrieved: list[RetrievedChunk], key: str) -> None:
    """Transparency panel: exactly which chunks were used, and how similar they were."""
    if not retrieved:
        st.caption("No chunks passed the similarity threshold.")
        return
    with st.expander(f"View Retrieved Context ({len(retrieved)} chunks)"):
        st.caption(
            "Documents are untrusted data: retrieved text is reference material only and "
            "can never override the system instructions."
        )
        for rank, item in enumerate(retrieved, start=1):
            page = f"page {item.chunk.page}" if item.chunk.page is not None else "no page info"
            st.markdown(f"**#{rank}** `{item.chunk.chunk_id}` - {item.chunk.source} ({page})")
            st.progress(max(0.0, min(1.0, item.score)), text=f"Cosine similarity: {item.score:.3f}")
            st.text_area("Chunk text", item.chunk.text, height=130, disabled=True,
                         key=f"{key}-{rank}-{item.chunk.chunk_id}", label_visibility="collapsed")
