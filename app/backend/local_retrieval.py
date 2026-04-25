"""Lexical retrieval from local AyurGenix CSV when Databricks is unavailable."""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import config


def _tokens(q: str) -> list[str]:
    return [t.lower() for t in q.replace(",", " ").split() if len(t) > 2][:8]


def search_curated_local(query: str, limit: int = 5) -> list[dict[str, Any]]:
    path = config.LOCAL_CSV_PATH
    if not path.is_file():
        return []
    df = pd.read_csv(path, dtype=str).fillna("")
    tokens = _tokens(query)
    if not tokens:
        tokens = [query.lower().strip()[:80]] if query.strip() else []

    def row_score(row) -> int:
        blob = " ".join(str(v) for v in row).lower()
        return sum(1 for t in tokens if t in blob)

    scores = df.apply(row_score, axis=1)
    top = df.assign(_score=scores).nlargest(limit, "_score")
    out: list[dict[str, Any]] = []
    for _, row in top.iterrows():
        if row["_score"] <= 0:
            continue
        out.append(row.drop(labels=["_score"]).to_dict())
    return out
