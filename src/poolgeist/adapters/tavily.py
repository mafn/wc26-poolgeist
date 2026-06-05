"""Optional Tavily search adapter."""

from __future__ import annotations

import os

import requests

from poolgeist.adapters.search_base import SearchResult


class TavilySearchProvider:
    """Minimal Tavily adapter using an environment API key."""

    endpoint = "https://api.tavily.com/search"

    def __init__(self, *, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Search with Tavily when configured."""

        if not self.api_key:
            raise RuntimeError("Set TAVILY_API_KEY to enable Tavily search.")
        response = requests.post(
            self.endpoint,
            json={"api_key": self.api_key, "query": query, "max_results": max_results},
            timeout=20,
        )
        response.raise_for_status()
        results = response.json().get("results", [])[:max_results]
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
            for item in results
        ]
