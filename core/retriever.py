"""Steps 5-6 - Cosine similarity + Top-K retrieval over an in-memory vector store.

Cosine similarity:  cos(a, b) = (a . b) / (|a| * |b|)
It measures the ANGLE between two vectors (same direction = same meaning), not their length.
Range: -1 (opposite) .. 0 (unrelated) .. 1 (same direction).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunker import Chunk


def cosine_similarity_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against every row of `matrix` (pure NumPy)."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return m @ q


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")
        if not chunks:
            return
        self.chunks.extend(chunks)
        self.embeddings = embeddings if self.embeddings.size == 0 else np.vstack([self.embeddings, embeddings])

    def remove_source(self, source: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        self.chunks = [self.chunks[i] for i in keep]
        self.embeddings = self.embeddings[keep] if keep else np.zeros((0, 0), dtype=np.float32)

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 3,
        min_score: float = 0.0,
        redundancy_limit: float = 0.95,
    ) -> list[RetrievedChunk]:
        """Rank all chunks by cosine similarity, drop weak/near-duplicate ones, return Top-K."""
        if not self.chunks or top_k < 1:
            return []
        scores = cosine_similarity_scores(query_vec, self.embeddings)
        results: list[RetrievedChunk] = []
        picked: list[int] = []
        for i in np.argsort(-scores):
            score = float(scores[i])
            if score < min_score:
                break  # sorted descending, everything after is weaker
            if any(float(self.embeddings[i] @ self.embeddings[j]) > redundancy_limit for j in picked):
                continue  # near-identical to a chunk we already picked
            picked.append(int(i))
            results.append(RetrievedChunk(self.chunks[i], score))
            if len(results) >= top_k:
                break
        return results
