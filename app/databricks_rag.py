"""
Load the multilingual RAG pipeline on Databricks (driver / Apps) without a numeric module name.

The notebook lives at ``databricks_notebooks/rag_pipeline/05_rag_pipeline.py``. Importing it as a
normal package is awkward; ``importlib`` loads it once ``__name__`` is not ``__main__``, so the
demo block at the bottom of that file does not run.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional


def _repo_root() -> Path:
    env = (os.environ.get("AYURGENIX_REPO_ROOT") or "").strip()
    if env:
        return Path(env).resolve()
    # app/databricks_rag.py -> parent app/ -> repo root (sibling: databricks_notebooks/)
    return Path(__file__).resolve().parent.parent


def load_rag_pipeline_module() -> ModuleType:
    path = _repo_root() / "databricks_notebooks" / "rag_pipeline" / "05_rag_pipeline.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"RAG pipeline file not found at {path}. "
            "Deploy the full repo (including databricks_notebooks/) to Databricks."
        )
    name = "ayurgenix_rag_pipeline_05"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_ask_multilingual() -> Callable[..., Dict[str, Any]]:
    mod = load_rag_pipeline_module()
    fn = getattr(mod, "ask_multilingual", None)
    if fn is None or not callable(fn):
        raise AttributeError("Loaded RAG module has no callable ask_multilingual")
    return fn


def normalize_sources(sources: Any) -> list[dict[str, Any]]:
    """Vector index rows are lists; Streamlit UI expects dicts with chunk_text / source_file."""
    out: list[dict[str, Any]] = []
    for src in sources or []:
        if isinstance(src, dict):
            out.append(src)
            continue
        if isinstance(src, (list, tuple)):
            chunk = src[0] if len(src) > 0 else ""
            source_file = src[1] if len(src) > 1 else None
            page_number = src[2] if len(src) > 2 else None
            score = src[-1] if len(src) > 1 else None
            out.append(
                {
                    "chunk_text": chunk or "",
                    "source_file": source_file,
                    "page_number": page_number,
                    "score": score,
                }
            )
            continue
        out.append({"chunk_text": str(src), "source_file": None, "page_number": None})
    return out


def ask_via_http(
    backend_url: str,
    question: str,
    top_k: int,
    user_lang: str,
    user_profile: Dict[str, Any],
) -> Dict[str, Any]:
    import requests

    url = f"{backend_url.rstrip('/')}/ask"
    payload: Dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "language": user_lang,
        "user_profile": user_profile,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def resolve_backend_url() -> Optional[str]:
    """If set (e.g. Model Serving or Apps internal URL), Streamlit calls POST /ask instead of in-process RAG."""
    raw = (os.environ.get("AYURGENIX_BACKEND_URL") or "").strip()
    return raw or None
