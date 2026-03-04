from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .. import db

try:
    import httpx
except Exception:
    httpx = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clean(s: str) -> str:
    return " ".join((s or "").strip().split())


async def _fetch_text(url: str, timeout: float = 10.0) -> str:
    if httpx is None:
        return ""

    try:
        from ..core.rate_limit import limiter  # type: ignore

        u = (url or "").lower()
        if "stooq.com" in u:
            await limiter.acquire("stooq")
        elif "wikipedia.org" in u:
            await limiter.acquire("wiki")
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers={"User-Agent": "financial-advisor-bot/1.0"})
        if r.status_code != 200:
            return ""
        return (r.text or "").strip()
    except Exception:
        return ""


async def _name_from_stooq(symbol: str) -> Optional[str]:
    """
    Stooq quote endpoint can return a 'Name' column for many symbols.
    Example:
      https://stooq.com/q/l/?s=aapl.us&f=sd2t2ohlcvn&h&e=csv
    Where 'n' includes name.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return None

    stooq_sym = sym.lower()
    if "." not in stooq_sym and stooq_sym.isalnum():
        stooq_sym = f"{stooq_sym}.us"

    url = f"https://stooq.com/q/l/?s={stooq_sym}&f=sd2t2ohlcvn&h&e=csv"
    text = await _fetch_text(url, timeout=10.0)
    if not text or "Name" not in text:
        return None

    try:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return None

        header = [h.strip() for h in lines[0].split(",")]
        row = [c.strip() for c in lines[1].split(",")]
        if "Name" not in header:
            return None

        idx = header.index("Name")
        if idx >= len(row):
            return None

        name = row[idx].strip().strip('"')
        name = _clean(name)
        return name or None
    except Exception:
        return None


async def _name_from_wikipedia(symbol: str) -> Optional[str]:
    """
    Very lightweight fallback:
    - Ask Wikipedia REST API for the 'ticker' page.
    - This will not always work, but it is free.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return None

    # Try direct symbol page, then "<symbol> (company)" is too expensive to search without an API.
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{sym}"
    text = await _fetch_text(url, timeout=10.0)
    if not text:
        return None

    # Avoid JSON parsing dependency; do minimal extraction.
    # If you prefer strict parsing, import json and parse safely.
    try:
        import json

        obj = json.loads(text)
        title = _clean(str(obj.get("title") or ""))
        if title and title.upper() != sym:
            return title
        return title or None
    except Exception:
        return None


async def resolve_symbol_name(symbol: str) -> Optional[str]:
    """
    DB cache first. If not found, try Stooq, then Wikipedia.
    Cache result with fetched_at.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return None

    col = db.require_col(db.get_db()["symbol_meta"], "symbol_meta")

    doc = col.find_one({"symbol": sym})
    if doc and doc.get("name"):
        return str(doc["name"])

    name = await _name_from_stooq(sym)
    if not name:
        name = await _name_from_wikipedia(sym)

    if name:
        col.update_one(
            {"symbol": sym},
            {"$set": {"symbol": sym, "name": name, "fetched_at": _utc_now()}},
            upsert=True,
        )
        return name

    # Cache “miss” too, so we do not hammer sources.
    col.update_one(
        {"symbol": sym},
        {"$set": {"symbol": sym, "name": None, "fetched_at": _utc_now()}},
        upsert=True,
    )
    return None
