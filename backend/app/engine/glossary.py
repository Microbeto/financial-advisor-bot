from __future__ import annotations

import re
from typing import Dict

_DYNAMIC_GLOSSARY: dict[str, str] = {}


def lingo_glossary() -> Dict[str, str]:
    # Build the base glossary and layer in runtime-generated terms.
    terms: Dict[str, str] = {
        "Regime": "A market state. Risk-on means investors buy risk. Risk-off means they avoid risk.",
        "Momentum": "A measure of recent price change. Positive means rising, negative means falling.",
        "Volatility": "How much price moves day to day. High volatility means bigger swings.",
        "20D Return": "Percent price change over the last 20 trading days.",
        "5D Return": "Percent price change over the last 5 trading days.",
        "Signal Score": "A combined score used to rank stocks. Higher means stronger trend.",
        "Uptrend": "Price is rising over recent weeks and days.",
        "Downtrend": "Price is falling over recent weeks and days.",
        "Drawdown": "The percent drop from a peak to a trough.",
    }
    terms.update(_DYNAMIC_GLOSSARY)
    return terms


_POSITIVE = [
    "beats",
    "beat",
    "upgrade",
    "upgraded",
    "raises",
    "raise",
    "surge",
    "record",
    "strong",
    "growth",
    "win",
    "profits",
]
_NEGATIVE = [
    "miss",
    "misses",
    "downgrade",
    "downgraded",
    "lawsuit",
    "probe",
    "weak",
    "decline",
    "cut",
    "cuts",
    "plunge",
    "layoff",
]


def score_title(title: str) -> float:
    # Score headline sentiment using simple positive/negative keyword hits.
    t = (title or "").lower()
    score = 0.0
    for w in _POSITIVE:
        if w in t:
            score += 1.0
    for w in _NEGATIVE:
        if w in t:
            score -= 1.0
    return score


def short_why_from_title(title: str) -> str:
    # Map headline themes to short plain-language market impact explanations.
    t = (title or "").strip()
    if not t:
        return "News item"
    s = t.lower()

    if any(w in s for w in ["earnings", "revenue", "profit", "guidance"]):
        return "Earnings or guidance news moved the price."
    if any(w in s for w in ["upgrade", "downgrade", "rating", "price target"]):
        return "Analyst action changed sentiment."
    if any(w in s for w in ["lawsuit", "probe", "regulator", "sec"]):
        return "Legal or regulatory risk affected the stock."
    if any(w in s for w in ["deal", "acquire", "merger", "buyout"]):
        return "Deal news changed expectations."
    if any(w in s for w in ["layoff", "cuts", "job", "restructure"]):
        return "Cost actions changed expectations."
    if any(w in s for w in ["ai", "chip", "semiconductor"]):
        return "Sector headline shifted demand expectations."
    return "Headline shifted market expectations."


def safe_text(s: str, limit: int = 280) -> str:
    # Normalize whitespace and cap output length for safe UI/API responses.
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s[:limit]


def update_glossary_from_news_summaries(items: list[dict]) -> None:
    # Auto-add common finance terms when they appear in generated news summaries.
    for it in items:
        s = (it.get("summary") or "").strip()
        if not s:
            continue

        for term in [
            "earnings",
            "guidance",
            "rate cuts",
            "inflation",
            "recession",
            "volatility",
            "momentum",
            "drawdown",
            "risk-on",
            "risk-off",
            "valuation",
            "multiple",
            "dividend",
            "buyback",
            "ETF",
            "yield",
        ]:
            # Save the first sentence as a compact glossary definition.
            if term.lower() in s.lower() and term not in _DYNAMIC_GLOSSARY:
                first_sentence = s.split(".", 1)[0].strip()
                if first_sentence:
                    _DYNAMIC_GLOSSARY[term] = first_sentence + "."


def set_glossary_term(key: str, value: str) -> None:
    # Upsert a dynamic glossary term, or remove it when value is empty.
    k = str(key or "").strip()
    v = str(value or "").strip()
    if not k:
        return
    if not v:
        _DYNAMIC_GLOSSARY.pop(k, None)
        return
    _DYNAMIC_GLOSSARY[k] = v


def remove_glossary_term(key: str) -> None:
    # Remove one term from the dynamic glossary map if it exists.
    _DYNAMIC_GLOSSARY.pop(str(key or "").strip(), None)


def list_dynamic_glossary_terms() -> Dict[str, str]:
    # Return a shallow copy so callers cannot mutate internal state directly.
    return dict(_DYNAMIC_GLOSSARY)
