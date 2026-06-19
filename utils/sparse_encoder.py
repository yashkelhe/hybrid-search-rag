"""
utils/sparse_encoder.py
Builds Pinecone-compatible sparse vectors using BM25 scoring.

Why custom instead of `pinecone-text`: that package wasn't in your requirements,
so this re-implements the core idea using `rank-bm25` + a hashing trick to turn
tokens into fixed-range integer indices (required format for Pinecone sparse vectors:
{"indices": [...], "values": [...]}).

IMPORTANT: the same fitted encoder (same vocabulary/idf) must be used at both
ingestion and query time, or the sparse space won't be comparable. fit() is only
called once during ingest.py, then save()/load() keep it consistent.
"""

import hashlib
import json
import math
import re

from rank_bm25 import BM25Okapi
from config import SPARSE_DIM

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_token(token: str, dim: int = SPARSE_DIM) -> int:
    """Maps a token to a stable integer index in [0, dim)."""
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


class BM25Sparse:
    def __init__(self, dim: int = SPARSE_DIM):
        self.dim = dim
        self.vocab_idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.k1: float = 1.5
        self.b: float = 0.75

    # ---------------- fitting (ingestion time only) ----------------
    def fit(self, corpus_texts: list[str]) -> None:
        tokenized = [_tokenize(t) for t in corpus_texts]
        bm25 = BM25Okapi(tokenized)
        self.vocab_idf = dict(bm25.idf)
        self.avgdl = bm25.avgdl
        self.k1 = bm25.k1
        self.b = bm25.b

    # ---------------- encoding ----------------
    def _doc_vector(self, text: str) -> dict:
        tokens = _tokenize(text)
        doc_len = len(tokens) or 1
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        weights: dict[int, float] = {}
        for term, freq in tf.items():
            idf = self.vocab_idf.get(term)
            if not idf or idf <= 0:
                continue
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
            score = idf * (freq * (self.k1 + 1)) / denom
            idx = _hash_token(term, self.dim)
            weights[idx] = weights.get(idx, 0.0) + score

        indices = list(weights.keys())
        values = [weights[i] for i in indices]
        return {"indices": indices, "values": values}

    def encode_documents(self, texts: list[str]) -> list[dict]:
        return [self._doc_vector(t) for t in texts]

    def encode_query(self, text: str) -> dict:
        tokens = _tokenize(text)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        weights: dict[int, float] = {}
        for term, freq in tf.items():
            idf = self.vocab_idf.get(term)
            if not idf or idf <= 0:
                continue
            idx = _hash_token(term, self.dim)
            weights[idx] = weights.get(idx, 0.0) + idf * freq

        indices = list(weights.keys())
        values = [weights[i] for i in indices]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        values = [v / norm for v in values]
        return {"indices": indices, "values": values}

    # ---------------- persistence ----------------
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "dim": self.dim,
                "vocab_idf": self.vocab_idf,
                "avgdl": self.avgdl,
                "k1": self.k1,
                "b": self.b,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            d = json.load(f)
        self.dim = d["dim"]
        self.vocab_idf = d["vocab_idf"]
        self.avgdl = d["avgdl"]
        self.k1 = d["k1"]
        self.b = d["b"]
