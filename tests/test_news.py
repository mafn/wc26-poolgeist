from poolgeist.adapters.search_base import ManualCsvSearchProvider
from poolgeist.models.sentiment_llm import heuristic_extract_news_signal


def test_news_signal_parsing_heuristic():
    signal = heuristic_extract_news_signal("Neutral player injury reported", team="Neutral A")
    assert signal is not None
    assert signal.signal_type == "injury"


def test_manual_search_missing_file_graceful(tmp_path):
    provider = ManualCsvSearchProvider(tmp_path / "missing.csv")
    assert provider.search("Neutral") == []
