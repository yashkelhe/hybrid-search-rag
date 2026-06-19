"""
utils/llm.py
Calls Gemini to generate the final answer from the RAG prompt.
"""

from google import genai
from config import GOOGLE_API_KEY, LLM_MODEL

_client = genai.Client(api_key=GOOGLE_API_KEY)


def generate_answer(prompt: str) -> str:
    response = _client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text
