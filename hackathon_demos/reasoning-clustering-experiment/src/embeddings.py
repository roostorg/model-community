"""Embedding generation via Gemini API."""
import numpy as np
from google import genai
from google.genai import types
from .config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM


_client = None

def get_client() -> genai.Client:
    """Lazy-load Gemini client."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in environment")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate normalized embeddings for texts.

    Args:
        texts: List of strings to embed

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM)
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM))

    client = get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="CLUSTERING",
            output_dimensionality=EMBEDDING_DIM
        )
    )

    embeddings = np.array([e.values for e in result.embeddings], dtype=np.float32)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-10, None)

    return embeddings


def embed_single(text: str) -> np.ndarray:
    """Embed a single text. Returns shape (EMBEDDING_DIM,)."""
    return embed_texts([text])[0]
