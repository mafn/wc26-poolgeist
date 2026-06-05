"""Search provider protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    """A normalized search result."""

    title: str
    url: str
    snippet: str = ""


class SearchProvider(Protocol):
    """Protocol for optional search/news providers."""

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Search the web and return normalized results."""
