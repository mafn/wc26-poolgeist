"""Optional Brave Search adapter."""

from __future__ import annotations

import os

import requests

from poolgeist.adapters.search_base import SearchResult


class BraveSearchProvider:
    """Minimal Brave Search adapter using an environment API key."""

    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, *, api_key: str | None = None):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY", "")

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Search with Brave when configured."""

        if not self.api_key:
            raise RuntimeError("Set BRAVE_API_KEY to enable Brave Search.")
        response = requests.get(
            self.endpoint,
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": self.api_key},
            timeout=20,
        )
        response.raise_for_status()
        results = response.json().get("web", {}).get("results", [])[:max_results]
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            )
            for item in results
        ]
