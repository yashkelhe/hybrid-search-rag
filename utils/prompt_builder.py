"""
utils/prompt_builder.py
Fills the rag_prompt.txt template with retrieved context and the user's question.
"""

from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_prompt.txt"


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for c in chunks:
        meta = c.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page_num", "?")
        context_blocks.append(f"[Source: {source} | Page {page}]\n{c['text']}")

    context = "\n\n---\n\n".join(context_blocks)
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(context=context, question=query)
