"""
app.py
Simple interactive command-line chat loop over the RAG pipeline.

Usage:
    python app.py
"""

from query import answer_query


def main():
    print("Hybrid Search RAG — ask questions about your ingested PDFs.")
    print("(type 'exit' or 'quit' to stop)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if not q:
            continue

        try:
            answer, chunks = answer_query(q)
            print(f"\nAssistant: {answer}\n")
            if chunks:
                print("Sources:")
                for c in chunks:
                    meta = c["metadata"]
                    score = c.get("rerank_score", 0)
                    print(f"  - {meta.get('source')} (page {meta.get('page_num')}) "
                          f"relevance={score:.3f}")
            print()
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
