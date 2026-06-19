"""
query.py
Run this to ask a single question against the ingested PDFs.

Flow:
query -> dense embed + sparse encode -> Pinecone hybrid search (top 20)
      -> Cohere rerank (top 5) -> build prompt -> Gemini -> final answer

Usage:
    python query.py
"""

from utils.retriever import retrieve
from utils.reranker import rerank
from utils.prompt_builder import build_prompt
from utils.llm import generate_answer
from config import TOP_K_RERANK


def answer_query(query: str):
    candidates = retrieve(query)
    if not candidates:
        return "No relevant chunks found. Did you run ingest.py first?", []

    top_chunks = rerank(query, candidates, top_n=TOP_K_RERANK)
    prompt = build_prompt(query, top_chunks)
    answer = generate_answer(prompt)
    return answer, top_chunks


if __name__ == "__main__":
    q = input("Enter your question: ").strip()
    answer, chunks = answer_query(q)

    print("\n--- ANSWER ---")
    print(answer)

    print("\n--- SOURCES ---")
    for c in chunks:
        meta = c["metadata"]
        score = c.get("rerank_score", 0)
        print(f"- {meta.get('source')} | page {meta.get('page_num')} | relevance {score:.3f}")
