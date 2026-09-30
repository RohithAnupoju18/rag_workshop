import numpy as np
import pytest

from tests.conftest import FakeEmbedder


def test_fake_embeddings_are_normalised():
    vecs = FakeEmbedder().encode(["hello world", "another sentence"])
    assert vecs.shape == (2, 256)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_real_sentence_transformer_model():
    """Runs only when sentence-transformers and the model are available locally."""
    pytest.importorskip("sentence_transformers")
    from core.embeddings import EmbeddingError, EmbeddingModel
    model = EmbeddingModel()
    try:
        vecs = model.encode(["eligibility criteria", "admission requirements", "banana bread recipe"])
    except EmbeddingError:
        pytest.skip("Embedding model not downloaded yet")
    assert vecs.shape[0] == 3 and np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2]  # related phrases are closer


def test_empty_input():
    from core.embeddings import EmbeddingModel
    assert EmbeddingModel().encode([]).size == 0
