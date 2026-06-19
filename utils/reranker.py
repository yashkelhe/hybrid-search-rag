"""
utils/reranker.py
Re-ranks retrieved chunks against the query using Cohere's rerank API,
and trims the candidate list down from top-20 to top-5 (or whatever you set).
"""

import cohere
from config import COHERE_API_KEY, RERANK_MODEL

_co = cohere.Client(COHERE_API_KEY)


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    if not chunks:
        return []
    print(f"[reranker] Re-ranking {len(chunks)} candidates...")
    docs = [c["text"] for c in chunks]
    response = _co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=min(top_n, len(docs)),
    )

    reranked = []

    
    for result in response.results:
        chunk = chunks[result.index]
        chunk["rerank_score"] = result.relevance_score
        reranked.append(chunk)
    return reranked
