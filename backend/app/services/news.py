from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .. import db
from ..models import NewsItem
from ..engine.glossary import update_glossary_from_news_summaries
from ..ml.model_router import intelligence_router

try:
    import httpx
except Exception:
    httpx = None


# Simple logging (no extra deps)
def _log(level: str, msg: str) -> None:
    # Keep logs visible under uvicorn without requiring logging config changes
    print(f"[news][{level}] {msg}")


# Tunables (env)
# Turn news on/off without code changes
NEWS_ENABLED = os.getenv("NEWS_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")

# "Symbol first" vs "market first"
# market-first is recommended so you always have some news daily
NEWS_PREFETCH_MODE = os.getenv("NEWS_PREFETCH_MODE", "market_first").strip().lower()  # market_first|symbol_first

# Cache policy
NEWS_CACHE_TTL_HOURS = int(os.getenv("NEWS_CACHE_TTL_HOURS", "12"))
NEWS_DASHBOARD_LIMIT = int(os.getenv("NEWS_DASHBOARD_LIMIT", "8"))  # frontend can show more than 3
NEWS_DB_LIMIT = int(os.getenv("NEWS_DB_LIMIT", "200"))

# Fetch policy
NEWS_FETCH_HOURS = int(os.getenv("NEWS_FETCH_HOURS", "48"))
NEWS_MAX_ITEMS_MARKET = int(os.getenv("NEWS_MAX_ITEMS_MARKET", "25"))
NEWS_MAX_ITEMS_PER_SYMBOL = int(os.getenv("NEWS_MAX_ITEMS_PER_SYMBOL", "12"))
NEWS_CONCURRENCY = int(os.getenv("NEWS_CONCURRENCY", "6"))
NEWS_HTTP_TIMEOUT = float(os.getenv("NEWS_HTTP_TIMEOUT", "15.0"))

# Fallback seed symbols if you have zero symbols available
NEWS_DEFAULT_SEED = os.getenv(
    "NEWS_DEFAULT_SEED",
    "SPY,AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,UNH,XOM,TSLA,AVGO,AMD,NFLX,GS",
).strip()


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
    "rally",
    "outperform",
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
    "slump",
]


def _utc_day_key(dt: Optional[datetime] = None) -> str:
    d = (dt or datetime.now(timezone.utc)).astimezone(timezone.utc).date()
    return d.isoformat()


def _three_line_summary(text: str) -> str:
    s = (text or "").strip().replace("\n", " ")
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    parts = [p.strip() for p in s.split(".") if p.strip()]
    if not parts:
        return ""
    return ". ".join(parts[:3]) + "."


def _score_title_lexicon(title: str) -> float:
    t = (title or "").lower()
    score = 0.0
    for w in _POSITIVE:
        if w in t:
            score += 1.0
    for w in _NEGATIVE:
        if w in t:
            score -= 1.0
    return score


def _score_title_keyword_intensity(title: str) -> float:
    """
    Lighter-weight fallback for very constrained environments.
    """
    t = (title or "").lower()
    score = 0.0
    score += 0.6 * sum(1.0 for w in _POSITIVE if w in t)
    score -= 0.6 * sum(1.0 for w in _NEGATIVE if w in t)
    return float(score)


def _sentiment_pipeline_name() -> str:
    try:
        return str(intelligence_router.get_sentiment_pipeline() or "lexicon").strip().lower()
    except Exception:
        return "lexicon"


def _score_title(title: str) -> float:
    """
    Sentiment routing entrypoint. Keeps news scoring decoupled from a single hardcoded method.
    """
    pipeline = _sentiment_pipeline_name()
    if pipeline == "finbert":
        # Until FinBERT inference is wired for article-level scoring in this service,
        # use the lexicon proxy and keep routing explicit through the router.
        return _score_title_lexicon(title)
    if pipeline in ("lexicon", "tfidf"):
        return _score_title_lexicon(title)
    return _score_title_keyword_intensity(title)


def _parse_dt(s: str) -> Optional[datetime]:
    """
    GDELT uses formats like:
    - 20251214123000 (YYYYMMDDhhmmss)
    - ISO strings sometimes appear in other modes
    """
    if not s:
        return None
    s = str(s).strip()
    try:
        if len(s) == 14 and s.isdigit():
            return datetime(
                int(s[0:4]),
                int(s[4:6]),
                int(s[6:8]),
                int(s[8:10]),
                int(s[10:12]),
                int(s[12:14]),
                tzinfo=timezone.utc,
            )
    except Exception:
        return None

    # Best-effort ISO parse
    try:
        # datetime.fromisoformat doesn't like Z; normalize
        iso = s.replace("Z", "+00:00")
        d = datetime.fromisoformat(iso)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        return None


async def _rate_limit_take(name: str) -> None:
    """
    Optional limiter integration. If you don't have services/rate_limit.py yet,
    this becomes a no-op.
    """
    try:
        from ..core.rate_limit import limiter  # type: ignore

        await limiter.acquire(name)
    except Exception:
        return


def _newsitems_to_dicts(news: List[NewsItem]) -> List[dict]:
    out: List[dict] = []
    for n in news or []:
        try:
            if hasattr(n, "model_dump"):
                out.append(n.model_dump())  # pydantic v2
            else:
                out.append(n.dict())  # pydantic v1
        except Exception:
            continue
    return out


def _ensure_summary(doc: dict) -> dict:
    """
    Backfill summary for old cached docs that don't have it.
    Keeps you compatible without needing to wipe Mongo.
    """
    if doc.get("summary"):
        return doc
    title = str(doc.get("title") or "")
    snippet = str(doc.get("snippet") or doc.get("excerpt") or doc.get("description") or "")
    doc["summary"] = _three_line_summary(snippet or title)
    return doc


def _normalize_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    # Common ticker normalization
    if s == "BRK.B":
        return "BRK-B"
    if s == "BRK/A":
        return "BRK-A"
    return s


def _seed_symbols(symbols: Optional[List[str]]) -> List[str]:
    raw = [str(s) for s in (symbols or [])]
    cleaned = [_normalize_symbol(s) for s in raw if str(s).strip()]
    # de-dup preserve order
    out = list(dict.fromkeys(cleaned))
    if out:
        return out
    # fallback env seed
    fallback = [_normalize_symbol(s) for s in NEWS_DEFAULT_SEED.split(",") if s.strip()]
    return list(dict.fromkeys(fallback))


def _dedupe_news_docs(docs: List[dict]) -> List[dict]:
    seen: set[str] = set()
    out: List[dict] = []
    for d in docs:
        nid = str(d.get("id") or "")
        url = str(d.get("url") or "")
        key = nid or url or (str(d.get("title") or "")[:120])
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


async def _fetch_gdelt(query: str, start_dt: datetime, max_items: int) -> List[Dict]:
    """
    Lowest-level GDELT fetch. Never raises. Returns [] on any failure.
    """
    if httpx is None:
        return []

    await _rate_limit_take("gdelt")

    start_str = start_dt.strftime("%Y%m%d%H%M%S")
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": int(max_items),
        "startdatetime": start_str,
        "sourcelang": "English",
        "sort": "HybridRel",
    }

    headers = {
        "User-Agent": "financial-advisor-bot/1.0 (local dev)",
        "Accept": "application/json,text/plain,*/*",
    }

    try:
        async with httpx.AsyncClient(timeout=NEWS_HTTP_TIMEOUT, headers=headers) as client:
            r = await client.get(url, params=params)

        if r.status_code != 200:
            return []

        ctype = (r.headers.get("content-type") or "").lower()
        text = (r.text or "").strip()

        if not text:
            return []
        if "json" not in ctype and not text.startswith("{"):
            return []

        try:
            data = r.json()
        except Exception:
            return []

        articles = data.get("articles") or []
        if not isinstance(articles, list):
            return []

        out: List[Dict] = []
        for a in articles:
            if isinstance(a, dict):
                out.append(a)
        return out

    except Exception:
        return []


async def fetch_gdelt_news(symbol: str, hours: int = 72, max_items: int = 15) -> List[Dict]:
    """
    Fetch news articles mentioning a symbol via GDELT DOC 2.1.
    Never raises. Returns [] on failure.
    """
    sym = _normalize_symbol(str(symbol))
    if not sym:
        return []

    query = f'"{sym}" AND (stock OR shares OR earnings OR revenue OR guidance OR analyst)'
    start_dt = datetime.now(timezone.utc) - timedelta(hours=int(hours))
    return await _fetch_gdelt(query=query, start_dt=start_dt, max_items=max_items)


async def fetch_gdelt_market_news(hours: int = 48, max_items: int = 25) -> List[Dict]:
    """
    Fetch market-wide finance news (not tied to a symbol).
    This is the key change: you will get news daily even when symbol matching fails.
    """
    # Broad query designed to return general market headlines
    query = (
        '(stock OR stocks OR "S&P 500" OR Dow OR Nasdaq OR earnings OR guidance OR inflation OR rates '
        'OR "Federal Reserve" OR recession OR rally OR selloff OR volatility) '
        'AND (market OR markets OR shares OR equity)'
    )
    start_dt = datetime.now(timezone.utc) - timedelta(hours=int(hours))
    return await _fetch_gdelt(query=query, start_dt=start_dt, max_items=max_items)


def _article_to_doc(date: str, symbol: Optional[str], a: Dict) -> Optional[dict]:
    title = str(a.get("title") or "").strip()
    url = str(a.get("url") or "").strip()
    if not title or not url:
        return None

    source = str(a.get("sourceCountry") or a.get("sourceCollection") or a.get("domain") or "").strip()
    published_at = _parse_dt(str(a.get("seendate") or ""))

    # Prefer URL as stable id
    nid = url or f"{(symbol or 'MKT')}:{title[:80]}"

    score = _score_title(title)
    snippet = str(a.get("excerpt") or a.get("description") or a.get("snippet") or "").strip()
    summary_text = _three_line_summary(snippet or title)

    doc = {
        "date": date,
        "id": nid,
        # symbol can be None for market-wide items
        "symbol": _normalize_symbol(symbol) if symbol else None,
        "title": title,
        "url": url,
        "publisher": source,
        "published_at": published_at,
        "score": float(score),
        "tags": [],
        "summary": summary_text,
    }
    return doc


async def refresh_news_for_symbols(date: str, symbols: List[str]) -> List[NewsItem]:
    """
    Fetch and cache symbol news.
    Returns flat list of NewsItem (without date field, date is stored in DB).
    """
    if not NEWS_ENABLED:
        return []

    date = str(date or "").strip() or _utc_day_key()
    col = db.require_col(db.news_cache_col, "news_cache")
    out: List[NewsItem] = []

    syms = _seed_symbols(symbols)
    if not syms:
        return out

    sem = asyncio.Semaphore(max(1, int(NEWS_CONCURRENCY)))

    async def one(sym: str) -> None:
        nonlocal out
        async with sem:
            items = await fetch_gdelt_news(sym, hours=NEWS_FETCH_HOURS, max_items=NEWS_MAX_ITEMS_PER_SYMBOL)

        for a in items:
            doc = _article_to_doc(date=date, symbol=sym, a=a)
            if not doc:
                continue

            try:
                col.update_one(
                    {"date": date, "id": doc["id"]},
                    {"$set": doc},
                    upsert=True,
                )
                out.append(NewsItem(**{k: v for k, v in doc.items() if k != "date"}))
            except Exception:
                continue

    await asyncio.gather(*(one(s) for s in syms))

    try:
        update_glossary_from_news_summaries(_newsitems_to_dicts(out))
    except Exception:
        pass

    return out


async def refresh_market_news(date: str) -> List[NewsItem]:
    """
    Fetch and cache market-wide news (symbol=None).
    This ensures you always have something to show daily.
    """
    if not NEWS_ENABLED:
        return []

    date = str(date or "").strip() or _utc_day_key()
    col = db.require_col(db.news_cache_col, "news_cache")
    out: List[NewsItem] = []

    items = await fetch_gdelt_market_news(hours=NEWS_FETCH_HOURS, max_items=NEWS_MAX_ITEMS_MARKET)
    for a in items:
        doc = _article_to_doc(date=date, symbol=None, a=a)
        if not doc:
            continue

        # Force symbol to None for market items
        doc["symbol"] = None

        try:
            col.update_one(
                {"date": date, "id": doc["id"]},
                {"$set": doc},
                upsert=True,
            )
            out.append(NewsItem(**{k: v for k, v in doc.items() if k != "date"}))
        except Exception:
            continue

    try:
        update_glossary_from_news_summaries(_newsitems_to_dicts(out))
    except Exception:
        pass

    return out


def get_news_for_date(
    date: str,
    symbols: Optional[List[str]] = None,
    limit: int = 60,
    include_market: bool = True,
) -> List[NewsItem]:
    """
    Read cached news. By default:
    - include market-wide items (symbol=None)
    - optionally include symbol-matched items if symbols provided
    """
    date = str(date or "").strip() or _utc_day_key()
    col = db.require_col(db.news_cache_col, "news_cache")

    q: Dict = {"date": date}

    if symbols:
        syms = [_normalize_symbol(str(s)) for s in symbols if str(s).strip()]
        if include_market:
            q["$or"] = [{"symbol": {"$in": syms}}, {"symbol": None}]
        else:
            q["symbol"] = {"$in": syms}
    else:
        if not include_market:
            q["symbol"] = {"$ne": None}

    docs = list(col.find(q).sort([("published_at", -1)]).limit(int(limit)))
    out: List[NewsItem] = []
    for d in docs:
        d.pop("_id", None)
        d.pop("date", None)

        try:
            d = _ensure_summary(d)
        except Exception:
            pass

        try:
            out.append(NewsItem(**d))
        except Exception:
            continue
    return out


def _cache_is_fresh(date: str, ttl_hours: int) -> bool:
    """
    Determine if we already have fresh-enough news for this date.
    Uses latest published_at from cache.
    """
    date = str(date or "").strip() or _utc_day_key()
    col = db.require_col(db.news_cache_col, "news_cache")

    doc = col.find_one({"date": date}, sort=[("published_at", -1)])
    if not doc:
        return False

    pa = doc.get("published_at")
    if not isinstance(pa, datetime):
        return True  # treat as fresh enough if it's present but non-datetime

    age = datetime.now(timezone.utc) - pa.astimezone(timezone.utc)
    return age <= timedelta(hours=int(ttl_hours))


async def refresh_news_if_needed(date: str, symbols: List[str]) -> List[NewsItem]:
    """
    Ensure there is news for the date. This is the main "daily news" behavior:
    - Always ensure market-wide news exists (so dashboard never shows empty)
    - Optionally add symbol-specific news (better relevance)
    - Respect TTL so you don't refetch constantly
    """
    if not NEWS_ENABLED:
        return []

    date = str(date or "").strip() or _utc_day_key()
    syms = _seed_symbols(symbols)

    # Fresh cache: just read
    if _cache_is_fresh(date, NEWS_CACHE_TTL_HOURS):
        return get_news_for_date(date, symbols=syms, limit=NEWS_DB_LIMIT, include_market=True)

    # Not fresh: fetch
    _log("INFO", f"refresh needed date={date} mode={NEWS_PREFETCH_MODE}")

    if NEWS_PREFETCH_MODE == "symbol_first":
        await refresh_news_for_symbols(date, syms)
        await refresh_market_news(date)
    else:
        # market_first (recommended): guarantees news daily even if symbol matching fails
        await refresh_market_news(date)
        await refresh_news_for_symbols(date, syms)

    return get_news_for_date(date, symbols=syms, limit=NEWS_DB_LIMIT, include_market=True)


async def refresh_top_news_of_day(date: str, symbols: Optional[List[str]] = None) -> List[NewsItem]:
    """
    Returns top dashboard news items for the date.

    Behavior:
    - Guarantees some items daily by ensuring market news exists
    - If symbols are provided, it will include symbol-matched items too
    - Ranking prefers: recency + magnitude of title score (simple intensity proxy)
    """
    if not NEWS_ENABLED:
        return []

    date = str(date or "").strip() or _utc_day_key()
    syms = _seed_symbols(symbols)

    # Ensure cache exists (market news at minimum)
    try:
        await refresh_news_if_needed(date, syms)
    except Exception:
        # Never break dashboard
        pass

    cached = get_news_for_date(date, symbols=syms, limit=NEWS_DB_LIMIT, include_market=True)
    if not cached:
        return []

    # Rank: recency first, then "score magnitude"
    # This produces stable daily picks without requiring sentiment models.
    def key(n: NewsItem) -> Tuple[float, float]:
        pa = n.published_at or datetime(1970, 1, 1, tzinfo=timezone.utc)
        ts = pa.astimezone(timezone.utc).timestamp()
        mag = abs(float(getattr(n, "score", 0.0) or 0.0))
        return (ts, mag)

    cached.sort(key=key, reverse=True)

    # Deduplicate by id/url/title
    docs = _newsitems_to_dicts(cached)
    docs = _dedupe_news_docs(docs)

    top_docs = docs[: max(1, int(NEWS_DASHBOARD_LIMIT))]
    top: List[NewsItem] = []
    for d in top_docs:
        try:
            top.append(NewsItem(**d))
        except Exception:
            continue

    try:
        update_glossary_from_news_summaries(_newsitems_to_dicts(top))
    except Exception:
        pass

    return top
