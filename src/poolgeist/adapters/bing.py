"""Optional bing search adapter."""

from __future__ import annotations

import os

from poolgeist.adapters.search_base import SearchResult


class BingSearchProvider:
    """Gracefully degrading bing provider wrapper."""

    provider_name = "bing"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("BING_SEARCH_API_KEY")

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Return no results when credentials or optional dependencies are absent."""

        if not self.api_key:
            return []
        return [
            SearchResult(
                query=query,
                title="Search adapter placeholder",
                url="manual://configure-provider",
                snippet="Configure a concrete provider client for live search.",
                provider=self.provider_name,
            )
        ][:max_results]
