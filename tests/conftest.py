import hashlib
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder so tests run offline and instantly."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out

    def encode_one(self, text):
        return self.encode([text])[0]
