"""Step 4 - Local embeddings with Sentence Transformers (no API key, runs on CPU)."""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~90 MB, 384 dimensions


class EmbeddingError(Exception):
    pass


class EmbeddingModel:
    """Lazy wrapper: the model loads on first use (download happens once, then cached on disk)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device="cpu")
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}'. The first run needs internet "
                    "once to download it; afterwards it works offline. "
                    f"Details: {exc.__class__.__name__}"
                ) from exc
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalised vectors."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        try:
            vectors = model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc.__class__.__name__}") from exc
        return vectors.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
