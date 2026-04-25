"""Reusable chunking utilities for Databricks preprocessing notebooks.

The public entry point is :func:`chunk_text_by_words`, kept backward
compatible so existing UDFs in the ingestion notebooks do not need to change.

Internally it uses a small recursive splitter that prefers paragraph and
sentence boundaries before falling back to a word-based slide window with
overlap. This produces noticeably more coherent chunks than naive
word-window splitting, which improves both embedding quality and the LLM's
ability to quote answers verbatim.
"""

from __future__ import annotations

import re
from typing import List

_SENTENCE_RE = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z0-9\"\'\(])")
_WS_RE = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _split_into_units(text: str) -> List[str]:
    """Split text into the smallest semantic units we'll ever join back together.

    Order of precedence: paragraphs -> sentences -> words.
    """
    # First by blank-line paragraphs, preserving order.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: List[str] = []
    for para in paragraphs:
        # Then by sentence boundary (period/?/! followed by capitalized start).
        sentences = _SENTENCE_RE.split(para)
        for sent in sentences:
            sent = _normalize_ws(sent)
            if sent:
                units.append(sent)
    return units


def _greedy_pack(
    units: List[str],
    chunk_size_words: int,
    overlap_words: int,
) -> List[str]:
    """Greedily pack semantic units into chunks <= chunk_size_words.

    When a single unit (e.g. a runaway sentence) exceeds the budget, fall back
    to word-window slicing for that unit only.
    """
    chunks: List[str] = []
    buf: List[str] = []
    buf_words = 0

    def flush() -> None:
        nonlocal buf, buf_words
        if buf:
            chunks.append(" ".join(buf))
            buf = []
            buf_words = 0

    for unit in units:
        wc = _word_count(unit)

        if wc > chunk_size_words:
            flush()
            words = unit.split()
            step = chunk_size_words - overlap_words
            for start in range(0, len(words), step):
                window = words[start : start + chunk_size_words]
                if window:
                    chunks.append(" ".join(window))
                if start + chunk_size_words >= len(words):
                    break
            continue

        if buf_words + wc <= chunk_size_words:
            buf.append(unit)
            buf_words += wc
        else:
            flush()
            buf.append(unit)
            buf_words = wc

    flush()

    if overlap_words <= 0 or len(chunks) <= 1:
        return chunks

    # Apply tail-overlap between adjacent chunks for retrieval recall.
    overlapped: List[str] = [chunks[0]]
    for prev, current in zip(chunks, chunks[1:]):
        tail = prev.split()[-overlap_words:]
        merged_words = tail + current.split()
        # Avoid blowing past the budget by trimming the front if needed.
        if len(merged_words) > chunk_size_words + overlap_words:
            merged_words = merged_words[-(chunk_size_words + overlap_words) :]
        overlapped.append(" ".join(merged_words))
    return overlapped


def chunk_text_by_words(
    text: str,
    chunk_size_words: int = 350,
    overlap_words: int = 70,
) -> List[str]:
    """Sentence-aware chunker with a word budget and overlap.

    - If the (normalized) text fits in ``chunk_size_words``, returns a single chunk.
    - Otherwise, splits on paragraph and sentence boundaries first, then greedily
      packs sentences into ``chunk_size_words``-sized buckets with ``overlap_words``
      tail-overlap between consecutive chunks.
    - Falls back to word-window slicing for any single sentence longer than the
      budget.

    Args:
        text: Source text. ``None`` and whitespace-only inputs return ``[]``.
        chunk_size_words: Target words per chunk (must be ``>= 1``).
        overlap_words: Word overlap between consecutive chunks
            (``>= 0`` and ``< chunk_size_words``).

    Raises:
        ValueError: If parameters are out of range.
    """
    if chunk_size_words < 1:
        raise ValueError("chunk_size_words must be >= 1")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be strictly less than chunk_size_words")

    if text is None:
        return []

    normalized = _normalize_ws(text)
    if not normalized:
        return []

    if _word_count(normalized) <= chunk_size_words:
        return [normalized]

    units = _split_into_units(text)
    if not units:
        return [normalized]

    return _greedy_pack(units, chunk_size_words, overlap_words)
