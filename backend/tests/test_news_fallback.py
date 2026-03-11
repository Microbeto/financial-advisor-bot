from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.news import _parse_yahoo_rss_feed, _score_title


def test_parse_yahoo_rss_feed_normalizes_items():
    xml_text = """
    <rss version="2.0">
      <channel>
        <title>Yahoo Finance - AAPL</title>
        <item>
          <title>Apple beats estimates in Q1</title>
          <link>https://finance.yahoo.com/news/apple-q1-results-123456789.html</link>
          <description>Revenue and EPS both exceeded expectations.</description>
          <pubDate>Sat, 07 Mar 2026 10:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """

    out = _parse_yahoo_rss_feed(xml_text, symbol="AAPL", max_items=5)

    assert len(out) == 1
    item = out[0]
    assert item["title"] == "Apple beats estimates in Q1"
    assert item["url"].startswith("https://finance.yahoo.com/news/")
    assert item["sourceCollection"] == "yahoo_rss"
    assert item["domain"] == "finance.yahoo.com"
    assert item["symbol"] == "AAPL"
    assert item["seendate"] == "20260307100000"


def test_parse_yahoo_rss_feed_respects_max_items():
    xml_text = """
    <rss version="2.0">
      <channel>
        <item><title>A</title><link>https://example.com/a</link></item>
        <item><title>B</title><link>https://example.com/b</link></item>
        <item><title>C</title><link>https://example.com/c</link></item>
      </channel>
    </rss>
    """

    out = _parse_yahoo_rss_feed(xml_text, symbol="MSFT", max_items=2)
    assert len(out) == 2


def test_score_title_cascade_escalates_ambiguous_to_finbert(monkeypatch: pytest.MonkeyPatch):
    class _StubFinBert:
        @staticmethod
        def score(text: str):
            del text
            return 0.42

    monkeypatch.setattr("app.services.news.NEWS_CASCADE_LEXICON_ABS_THRESHOLD", 2.0)
    monkeypatch.setattr("app.services.news._sentiment_pipeline_name", lambda: "cascade")
    monkeypatch.setattr("app.services.news._get_news_finbert", lambda: _StubFinBert())

    # Neutral/ambiguous title should escalate to FinBERT score under cascade mode.
    s = _score_title("Markets close with mixed sector moves")
    assert abs(float(s) - 0.42) < 1e-9
