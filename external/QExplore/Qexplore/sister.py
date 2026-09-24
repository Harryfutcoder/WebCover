"""Deterministic fallback for sister.MeanEmbedding."""

import hashlib
import numpy as np


class MeanEmbedding:
    def __init__(self, lang="en", dim=64):
        self.lang = lang
        self.dim = dim

    def __call__(self, text):
        data = str(text or "").encode("utf-8", errors="ignore")
        digest = hashlib.sha256(data).digest()
        values = [digest[i % len(digest)] / 255.0 for i in range(self.dim)]
        return np.array(values, dtype=float)
