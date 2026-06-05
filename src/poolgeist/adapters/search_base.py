"""Search provider interfaces and manual CSV fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class SearchResult:
    """A normalized search result."""

    query: str
    title: str
    url: str
    source: str = ""
    snippet: str = ""
    published_date: str | None = None
    retrieved_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    provider: str = "manual"
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class SearchProvider(Protocol):
    """Protocol for optional search/news providers."""

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Search the web and return normalized results."""


class ManualCsvSearchProvider:
    """Local/manual CSV search provider that requires no API keys."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Return rows matching a query from a local CSV, or all rows if no query column exists."""

        if not self.path.exists():
            return []
        frame = pd.read_csv(self.path)
        if "query" in frame.columns:
            frame = frame[frame["query"].astype(str).str.contains(query, case=False, na=False)]
        results = []
        for _, row in frame.head(max_results).iterrows():
            results.append(
                SearchResult(
                    query=query,
                    title=str(row.get("title", row.get("source_title", "Manual signal"))),
                    url=str(row.get("url", row.get("source_url", "manual://local"))),
                    source=str(row.get("source", "manual_csv")),
                    snippet=str(row.get("snippet", row.get("evidence_summary", ""))),
                    published_date=None
                    if pd.isna(row.get("published_date"))
                    else str(row.get("published_date")),
                    provider="manual_csv",
                    raw_metadata=row.to_dict(),
                )
            )
        return results
