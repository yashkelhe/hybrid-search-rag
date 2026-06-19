"""
utils/chunker.py
Splits page-level text into overlapping chunks using LangChain's
RecursiveCharacterTextSplitter, while preserving page/source metadata.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Input: list of {"page_num", "text", "source"} (from pdf_loader)
    Output: list of {"text", "metadata": {"source", "page_num", "chunk_id"}}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for j, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunk_id = f"{page['source']}_p{page['page_num']}_c{j}".replace(" ", "_")
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": page["source"],
                    "page_num": page["page_num"],
                    "chunk_id": chunk_id,
                },
            })
    return chunks
