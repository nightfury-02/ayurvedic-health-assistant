"""Databricks Foundation Model embeddings (OpenAI-compatible client)."""

from __future__ import annotations

from typing import Sequence

from openai import OpenAI

from . import config


def _client() -> OpenAI:
    if not config.DATABRICKS_HOST or not config.DATABRICKS_TOKEN:
        raise RuntimeError("DATABRICKS_HOST and DATABRICKS_TOKEN required for embeddings.")
    return OpenAI(api_key=config.DATABRICKS_TOKEN, base_url=f"{config.DATABRICKS_HOST.rstrip('/')}/serving-endpoints")


def embed_texts(texts: Sequence[str], model: str | None = None) -> list[list[float]]:
    """Batch embed lines using a Databricks FM embedding endpoint name."""
    m = model or config.EMBEDDING_MODEL
    client = _client()
    resp = client.embeddings.create(model=m, input=list(texts))
    ordered = sorted(resp.data, key=lambda o: getattr(o, "index", 0))
    return [item.embedding for item in ordered]
