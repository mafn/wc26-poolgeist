"""Optional SerpAPI search adapter."""

from __future__ import annotations

import os

import requests

from poolgeist.adapters.search_base import SearchResult


class SerpApiSearchProvider:
    """Minimal SerpAPI adapter using an environment API key."""

    endpoint = "https://serpapi.com/search.json"

    def __init__(self, *, api_key: str | None = None):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY", "")

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Search with SerpAPI when configured."""

        if not self.api_key:
            raise RuntimeError("Set SERPAPI_API_KEY to enable SerpAPI search.")
        response = requests.get(
            self.endpoint,
            params={"q": query, "api_key": self.api_key, "num": max_results},
            timeout=20,
        )
        response.raise_for_status()
        results = response.json().get("organic_results", [])[:max_results]
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            )
            for item in results
        ]
