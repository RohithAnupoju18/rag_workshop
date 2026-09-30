import numpy as np

from core.chunker import Chunk
from core.rag_pipeline import NOT_FOUND_MESSAGE, RAGPipeline, build_messages
from core.retriever import VectorStore, cosine_similarity_scores
from tests.conftest import FakeEmbedder


def test_cosine_similarity_math():
    q = np.array([1.0, 0.0])
    m = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [2.0, 2.0]])
    s = cosine_similarity_scores(q, m)
    assert np.allclose(s, [1.0, 0.0, -1.0, 2 / np.sqrt(8)])  # scale-invariant, angle only


def test_matches_sklearn():
    from sklearn.metrics.pairwise import cosine_similarity
    rng = np.random.default_rng(0)
    q, m = rng.normal(size=8), rng.normal(size=(5, 8))
    assert np.allclose(cosine_similarity_scores(q, m), cosine_similarity([q], m)[0], atol=1e-6)


def _store():
    emb = FakeEmbedder()
    texts = ["Eligibility requires age 18 and a valid ID.", "The exam fee is 500 rupees.", "Campus has a library and a cafeteria."]
    chunks = [Chunk(f"d::doc::c{i}", t, "d.docx", None, i) for i, t in enumerate(texts)]
    store = VectorStore()
    store.add(chunks, emb.encode(texts))
    return emb, store


def test_top_k_ranking_and_threshold():
    emb, store = _store()
    hits = store.search(emb.encode_one("eligibility age requirement"), top_k=2)
    assert hits[0].chunk.index == 0 and len(hits) <= 2
    assert hits == sorted(hits, key=lambda h: -h.score)
    assert store.search(emb.encode_one("quantum astrophysics"), top_k=3, min_score=0.5) == []


def test_duplicate_chunks_are_filtered():
    emb = FakeEmbedder()
    texts = ["same text about fees", "same text about fees", "different topic entirely"]
    chunks = [Chunk(f"c{i}", t, "d", None, i) for i, t in enumerate(texts)]
    store = VectorStore()
    store.add(chunks, emb.encode(texts))
    hits = store.search(emb.encode_one("fees"), top_k=3)
    assert sum(1 for h in hits if h.chunk.text == "same text about fees") == 1


def test_remove_source():
    _, store = _store()
    store.remove_source("d.docx")
    assert len(store) == 0 and store.search(np.ones(256)) == []


def _pipeline(reply="Age 18 [d.docx]."):
    calls = []

    def fake_llm(model, messages, temperature):
        calls.append((model, messages))
        return reply

    p = RAGPipeline(embedder=FakeEmbedder(), llm_chat=fake_llm)
    emb, store = _store()
    p.store = store
    return p, calls


def test_no_document_indexed():
    p = RAGPipeline(embedder=FakeEmbedder(), llm_chat=lambda *a: "x")
    assert "upload" in p.ask("hi", "qwen2.5:3b").answer.lower()


def test_llm_receives_only_relevant_context_and_separated_prompt():
    p, calls = _pipeline()
    result = p.ask("What is the eligibility age?", "qwen2.5:3b", top_k=1, threshold=0.1)
    assert result.used_llm and result.answer.startswith("Age 18")
    system, user = calls[0][1]
    assert system["role"] == "system" and "untrusted DATA" in system["content"]
    assert "<retrieved_context>" in user["content"] and "<user_question>" in user["content"]
    assert "cafeteria" not in user["content"]  # unrelated chunk never reaches the LLM


def test_irrelevant_question_never_calls_llm():
    p, calls = _pipeline()
    result = p.ask("zebra quantum galaxy", "qwen2.5:3b", threshold=0.6)
    assert result.answer == NOT_FOUND_MESSAGE and not calls


def test_no_model_case_is_graceful():
    p, calls = _pipeline()
    result = p.ask("eligibility age", None, threshold=0.1)
    assert "no Ollama model" in result.answer and result.retrieved and not calls


def test_llm_failure_does_not_crash():
    from core.ollama_client import OllamaError
    p = RAGPipeline(embedder=FakeEmbedder(), llm_chat=lambda *a: (_ for _ in ()).throw(OllamaError("timeout")))
    p.store = _store()[1]
    assert "could not answer" in p.ask("eligibility age", "llama3.2", threshold=0.1).answer


def test_index_document_and_cache():
    import io, docx
    d = docx.Document()
    d.add_paragraph("Eligibility: candidates must be 18 years old. " * 10)
    buf = io.BytesIO(); d.save(buf)
    p = RAGPipeline(embedder=FakeEmbedder())
    r1 = p.index_document(buf.getvalue(), "e.docx", 300, 50)
    r2 = p.index_document(buf.getvalue(), "e.docx", 300, 50)
    assert r1.chunk_count > 0 and not r1.cached and r2.cached and len(p.store) == r1.chunk_count


def test_build_messages_wraps_page_info():
    _, store = _store()
    msgs = build_messages("q?", store.search(np.ones(256), top_k=1))
    assert "[Source 1: d.docx]" in msgs[1]["content"]
