"""Reusable chunking utilities for Databricks preprocessing notebooks."""

from typing import List


def chunk_text_by_words(
    text: str,
    chunk_size_words: int = 350,
    overlap_words: int = 70,
) -> List[str]:
    """
    Chunk text by words with overlap.

    - If text has fewer than `chunk_size_words`, returns a single chunk.
    - Uses normalized whitespace to reduce noisy token boundaries.
    """
    if text is None:
        return []

    normalized = " ".join(text.split())
    if not normalized:
        return []

    words = normalized.split(" ")
    if len(words) <= chunk_size_words:
        return [normalized]

    chunks = []
    step = max(chunk_size_words - overlap_words, 1)
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        slice_words = words[start:end]
        if slice_words:
            chunks.append(" ".join(slice_words))
        if end >= len(words):
            break
        start += step
    return chunks
