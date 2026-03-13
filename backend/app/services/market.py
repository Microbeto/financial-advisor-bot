# backend/app/services/market.py
from __future__ import annotations

import asyncio
import csv
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None

from .. import db

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import httpx
except Exception:
    httpx = None

try:
    from pandas_datareader import data as _pdr
except Exception:
    _pdr = None

pdr = _pdr


@dataclass
class PricePoint:
    t: str  # ISO date
    c: float  # close


# Env tunables
ENABLE_YFINANCE = os.getenv("ENABLE_YFINANCE", "0").lower() in ("1", "true", "yes")
MARKET_CACHE_TTL_DAYS = int(os.getenv("MARKET_CACHE_TTL_DAYS", "7"))
MARKET_LOCAL_CACHE_ENABLED = os.getenv("MARKET_LOCAL_CACHE_ENABLED", "1").lower() in ("1", "true", "yes")
MARKET_LOCAL_CACHE_DIR = os.getenv(
    "MARKET_LOCAL_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / ".cache" / "market"),
)
MARKET_CACHE_PIT_IMMUTABLE = os.getenv("MARKET_CACHE_PIT_IMMUTABLE", "1").lower() in ("1", "true", "yes")
MARKET_CACHE_ALLOW_RESTATEMENTS = os.getenv("MARKET_CACHE_ALLOW_RESTATEMENTS", "0").lower() in ("1", "true", "yes")

# Delay between HTTP calls (helps rate limiting)
MARKET_REQUEST_DELAY_MS = int(os.getenv("MARKET_REQUEST_DELAY_MS", "250"))

# Timeout for external requests
MARKET_HTTP_TIMEOUT_SEC = float(os.getenv("MARKET_HTTP_TIMEOUT_SEC", "15"))

# Prefer source order (first working wins)
# yfinance is optional and can be last to reduce spam and blocks
PREFER_SOURCES = os.getenv(
    "MARKET_PREFER_SOURCES",
    "yahoo_chart,yahoo_csv,stooq_csv,pdr_stooq,yfinance",
).strip().lower()


_CACHE_STATS_LOCK = threading.Lock()
_CACHE_STATS: Dict[str, int] = {
    "requests": 0,
    "disk_hits": 0,
    "mongo_hits": 0,
    "misses": 0,
    "writes": 0,
    "disk_writes": 0,
}


def _inc_cache_stat(key: str, value: int = 1) -> None:
    with _CACHE_STATS_LOCK:
        _CACHE_STATS[key] = int(_CACHE_STATS.get(key, 0)) + int(value)


def get_market_cache_runtime_stats() -> Dict[str, int]:
    with _CACHE_STATS_LOCK:
        return dict(_CACHE_STATS)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_date(d: datetime) -> str:
    return d.astimezone(timezone.utc).date().isoformat()


def _cache_file_path(symbol: str, interval: str, days: int) -> Path:
    safe_symbol = "".join(ch if ch.isalnum() else "_" for ch in (symbol or "").upper())
    safe_interval = "".join(ch if ch.isalnum() else "_" for ch in (interval or "1d").lower())
    root = Path(MARKET_LOCAL_CACHE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{safe_symbol}_{safe_interval}_{int(days)}.json"


def _pit_immutable_enabled() -> bool:
    return bool(MARKET_CACHE_PIT_IMMUTABLE and not MARKET_CACHE_ALLOW_RESTATEMENTS)


def _point_date(point: Dict[str, Any]) -> str:
    return str(point.get("t") or "").strip()


def _merge_pit_points(existing_points: List[Dict[str, Any]], incoming_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not existing_points:
        return list(incoming_points)
    if not incoming_points:
        return list(existing_points)

    kept_by_date: Dict[str, Dict[str, Any]] = {}
    passthrough: List[Dict[str, Any]] = []

    for p in existing_points:
        if not isinstance(p, dict):
            continue
        d = _point_date(p)
        if d and d not in kept_by_date:
            kept_by_date[d] = dict(p)
            continue
        if not d:
            passthrough.append(dict(p))

    for p in incoming_points:
        if not isinstance(p, dict):
            continue
        d = _point_date(p)
        if d and d in kept_by_date:
            continue
        if d:
            kept_by_date[d] = dict(p)
        else:
            passthrough.append(dict(p))

    merged = list(kept_by_date.values()) + passthrough
    merged.sort(key=lambda row: str(row.get("t") or ""))
    return merged


def _is_disk_doc_fresh(doc: dict, ttl_days: int) -> bool:
    if ttl_days <= 0:
        return True
    try:
        s = str(doc.get("fetched_at") or "")
        if not s:
            return False
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        age = (_utc_now() - dt.astimezone(timezone.utc)).total_seconds()
        return age < float(ttl_days * 86400)
    except Exception:
        return False


def _read_disk_cache(symbol: str, interval: str, days: int) -> Optional[List[Dict[str, Any]]]:
    if not MARKET_LOCAL_CACHE_ENABLED:
        return None
    try:
        p = _cache_file_path(symbol, interval, days)
        if not p.exists():
            return None
        doc = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or not _is_disk_doc_fresh(doc, MARKET_CACHE_TTL_DAYS):
            return None
        pts = doc.get("points") or []
        if isinstance(pts, list) and len(pts) >= 1:
            _inc_cache_stat("disk_hits")
            return pts
        return None
    except Exception:
        return None


def _write_disk_cache(symbol: str, interval: str, days: int, points: List[Dict[str, Any]], source: str) -> None:
    if not MARKET_LOCAL_CACHE_ENABLED:
        return
    try:
        p = _cache_file_path(symbol, interval, days)
        persisted_points = list(points)
        if _pit_immutable_enabled() and p.exists():
            try:
                current_doc = json.loads(p.read_text(encoding="utf-8"))
                current_points = current_doc.get("points") if isinstance(current_doc, dict) else []
                if isinstance(current_points, list):
                    persisted_points = _merge_pit_points(current_points, persisted_points)
            except Exception:
                pass

        payload = {
            "symbol": symbol,
            "interval": interval,
            "days": int(days),
            "source": source,
            "fetched_at": _utc_now().isoformat().replace("+00:00", "Z"),
            "points": persisted_points,
        }
        p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        _inc_cache_stat("disk_writes")
    except Exception:
        return


def _parse_ohlc_csv_points(text: str, days: int) -> List[Dict[str, Any]]:
    if not text:
        return []

    out: List[Dict[str, Any]] = []
    try:
        rows = csv.DictReader(text.splitlines())
        for row in rows:
            if not isinstance(row, dict):
                continue

            d_raw = str(row.get("Date") or "").strip()
            if not d_raw:
                continue

            close_raw = row.get("Adj Close")
            if close_raw in (None, "", "null", "None", "N/A"):
                close_raw = row.get("Close")

            open_raw = row.get("Open")
            high_raw = row.get("High")
            low_raw = row.get("Low")
            vol_raw = row.get("Volume")

            try:
                close = float(close_raw)  # type: ignore[arg-type]
            except Exception:
                continue
            try:
                opn = float(open_raw) if open_raw not in (None, "", "null", "None", "N/A") else float(close)
            except Exception:
                opn = float(close)
            try:
                high = float(high_raw) if high_raw not in (None, "", "null", "None", "N/A") else max(opn, float(close))
            except Exception:
                high = max(opn, float(close))
            try:
                low = float(low_raw) if low_raw not in (None, "", "null", "None", "N/A") else min(opn, float(close))
            except Exception:
                low = min(opn, float(close))
            try:
                vol = float(vol_raw) if vol_raw not in (None, "", "null", "None", "N/A") else 0.0
            except Exception:
                vol = 0.0

            try:
                dt = datetime.fromisoformat(d_raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                try:
                    dt = datetime.strptime(d_raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except Exception:
                    continue

            out.append(
                {
                    "t": _iso_date(dt),
                    "o": float(opn),
                    "h": float(high),
                    "l": float(low),
                    "c": float(close),
                    "v": float(max(0.0, vol)),
                }
            )

        out.sort(key=lambda x: x["t"])
        if days and len(out) > int(days):
            out = out[-int(days) :]
        return out
    except Exception:
        return []


def _to_points_from_df(df: Any) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    cols = {c.lower(): c for c in df.columns}
    close_col = None
    open_col = None
    high_col = None
    low_col = None
    volume_col = None
    for cand in ["adj close", "adjclose", "close"]:
        if cand in cols:
            close_col = cols[cand]
            break
    for cand in ["open"]:
        if cand in cols:
            open_col = cols[cand]
            break
    for cand in ["high"]:
        if cand in cols:
            high_col = cols[cand]
            break
    for cand in ["low"]:
        if cand in cols:
            low_col = cols[cand]
            break
    for cand in ["volume", "vol"]:
        if cand in cols:
            volume_col = cols[cand]
            break
    if close_col is None:
        return []

    out: List[Dict[str, Any]] = []
    idx = df.index
    for i in range(len(df)):
        dt = idx[i]
        try:
            if hasattr(dt, "to_pydatetime"):
                dt = dt.to_pydatetime()
            if not isinstance(dt, datetime):
                dt = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)  # type: ignore
        except Exception:
            continue

        try:
            c = float(df.iloc[i][close_col])
        except Exception:
            continue

        try:
            opn = float(df.iloc[i][open_col]) if open_col is not None else float(c)
        except Exception:
            opn = float(c)
        try:
            high = float(df.iloc[i][high_col]) if high_col is not None else max(opn, float(c))
        except Exception:
            high = max(opn, float(c))
        try:
            low = float(df.iloc[i][low_col]) if low_col is not None else min(opn, float(c))
        except Exception:
            low = min(opn, float(c))
        try:
            vol = float(df.iloc[i][volume_col]) if volume_col is not None else 0.0
        except Exception:
            vol = 0.0

        out.append(
            {
                "t": _iso_date(dt),
                "o": float(opn),
                "h": float(high),
                "l": float(low),
                "c": float(c),
                "v": float(max(0.0, vol)),
            }
        )

    out.sort(key=lambda x: x["t"])
    return out


async def _sleep_delay() -> None:
    ms = max(0, int(MARKET_REQUEST_DELAY_MS))
    if ms > 0:
        await asyncio.sleep(ms / 1000.0)


async def _provider_limit(url: str) -> None:
    u = (url or "").lower()
    if "stooq.com" in u:
        key = "stooq"
    elif "yahoo.com" in u:
        key = "yahoo"
    else:
        return

    try:
        from ..core.rate_limit import limiter  # type: ignore

        await limiter.acquire(key)
    except Exception:
        return


async def _fetch_text(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> str:
    if httpx is None:
        return ""

    await _provider_limit(url)
    await _sleep_delay()

    try:
        tout = float(timeout or MARKET_HTTP_TIMEOUT_SEC)
        async with httpx.AsyncClient(timeout=tout, headers=headers) as client:
            r = await client.get(url, params=params)
        if r.status_code != 200:
            return ""
        return (r.text or "").strip()
    except Exception:
        return ""


async def _fetch_json(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    if httpx is None:
        return {}

    await _provider_limit(url)
    await _sleep_delay()

    try:
        tout = float(timeout or MARKET_HTTP_TIMEOUT_SEC)
        async with httpx.AsyncClient(timeout=tout, headers=headers) as client:
            r = await client.get(url, params=params)
        if r.status_code != 200:
            return {}
        return r.json() if r.content else {}
    except Exception:
        return {}


def _cache_key(symbol: str, interval: str, days: int) -> Dict[str, Any]:
    return {"symbol": symbol, "interval": interval, "days": int(days)}


def _cache_fresh(doc: dict, ttl_days: int) -> bool:
    if ttl_days <= 0:
        return True
    try:
        fa = doc.get("fetched_at")
        if not isinstance(fa, datetime):
            return False
        age = (_utc_now() - fa).total_seconds()
        return age < float(ttl_days * 86400)
    except Exception:
        return False


def _read_cache(symbol: str, interval: str, days: int) -> Optional[List[Dict[str, Any]]]:
    _inc_cache_stat("requests")

    disk = _read_disk_cache(symbol, interval, days)
    if disk is not None:
        return disk

    try:
        mp = db.require_col(db.market_prices_col, "market_prices")
        doc = mp.find_one(_cache_key(symbol, interval, days))
        if not doc:
            return None
        if not _cache_fresh(doc, MARKET_CACHE_TTL_DAYS):
            return None
        pts = doc.get("points") or []
        if isinstance(pts, list) and len(pts) >= 1:
            _inc_cache_stat("mongo_hits")
            _write_disk_cache(symbol, interval, days, pts, str(doc.get("source") or "mongo"))
            return pts
        _inc_cache_stat("misses")
        return None
    except Exception:
        _inc_cache_stat("misses")
        return None


def _write_cache(symbol: str, interval: str, days: int, points: List[Dict[str, Any]], source: str) -> None:
    _inc_cache_stat("writes")
    persisted_points = list(points)
    _write_disk_cache(symbol, interval, days, persisted_points, source)

    try:
        mp = db.require_col(db.market_prices_col, "market_prices")
        if _pit_immutable_enabled():
            existing = mp.find_one(_cache_key(symbol, interval, days), {"points": 1}) or {}
            existing_points = existing.get("points") or []
            if isinstance(existing_points, list):
                persisted_points = _merge_pit_points(existing_points, persisted_points)

        mp.update_one(
            _cache_key(symbol, interval, days),
            {
                "$set": {
                    **_cache_key(symbol, interval, days),
                    "points": persisted_points,
                    "source": source,
                    "fetched_at": _utc_now(),
                }
            },
            upsert=True,
        )
    except Exception:
        pass


def _parse_prefer_sources() -> List[str]:
    parts = [p.strip().lower() for p in (PREFER_SOURCES or "").split(",") if p.strip()]
    if not parts:
        return ["yahoo_chart", "yahoo_csv", "stooq_csv", "pdr_stooq", "yfinance"]
    return parts


def _range_for_days(days: int) -> str:
    d = int(max(5, days))
    if d <= 7:
        return "7d"
    if d <= 30:
        return "1mo"
    if d <= 90:
        return "3mo"
    if d <= 180:
        return "6mo"
    if d <= 365:
        return "1y"
    return "2y"


async def _history_from_yahoo_chart(symbol: str, days: int) -> List[Dict[str, Any]]:
    sym = (symbol or "").strip().upper()
    if not sym:
        return []

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    params = {
        "interval": "1d",
        "range": _range_for_days(days),
        "includePrePost": "false",
    }
    headers = {"User-Agent": "financial-advisor-bot/1.0", "Accept": "application/json,*/*"}

    js = await _fetch_json(url, params=params, headers=headers, timeout=20.0)
    try:
        chart = js.get("chart") or {}
        err = chart.get("error")
        if err:
            return []

        res = (chart.get("result") or [None])[0] or {}
        ts = res.get("timestamp") or []
        ind = (res.get("indicators") or {}).get("quote") or []
        quote0 = ind[0] if ind else {}
        opens = quote0.get("open") or []
        highs = quote0.get("high") or []
        lows = quote0.get("low") or []
        closes = quote0.get("close") or []
        vols = quote0.get("volume") or []

        if not ts or not closes:
            return []

        out: List[Dict[str, Any]] = []
        for idx, (t, c) in enumerate(zip(ts, closes)):
            if c is None:
                continue
            try:
                dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
                opn = opens[idx] if idx < len(opens) and opens[idx] is not None else c
                high = highs[idx] if idx < len(highs) and highs[idx] is not None else max(opn, c)
                low = lows[idx] if idx < len(lows) and lows[idx] is not None else min(opn, c)
                vol = vols[idx] if idx < len(vols) and vols[idx] is not None else 0.0
                out.append(
                    {
                        "t": _iso_date(dt),
                        "o": float(opn),
                        "h": float(high),
                        "l": float(low),
                        "c": float(c),
                        "v": float(max(0.0, float(vol))),
                    }
                )
            except Exception:
                continue

        if days and len(out) > int(days):
            out = out[-int(days) :]

        out.sort(key=lambda x: x["t"])
        return out
    except Exception:
        return []


async def _history_from_yahoo_csv(symbol: str, days: int) -> List[Dict[str, Any]]:
    sym = (symbol or "").strip().upper()
    if not sym:
        return []

    now = _utc_now()
    start = now - timedelta(days=int(max(5, days)) + 10)

    period1 = int(start.timestamp())
    period2 = int(now.timestamp())

    url = f"https://query1.finance.yahoo.com/v7/finance/download/{sym}"
    params = {
        "period1": str(period1),
        "period2": str(period2),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": "financial-advisor-bot/1.0", "Accept": "text/csv,*/*"}

    text = await _fetch_text(url, params=params, headers=headers, timeout=20.0)
    if not text or "Date,Open,High,Low,Close" not in text:
        return []

    return _parse_ohlc_csv_points(text, days)


async def _history_from_stooq_csv(symbol: str, days: int) -> List[Dict[str, Any]]:
    sym = (symbol or "").strip().lower()
    if not sym:
        return []

    if "." not in sym and sym.isalnum():
        sym = f"{sym}.us"

    url = "https://stooq.com/q/d/l/"
    params = {"s": sym, "i": "d"}
    headers = {"User-Agent": "financial-advisor-bot/1.0", "Accept": "text/csv,*/*"}

    text = await _fetch_text(url, params=params, headers=headers, timeout=20.0)
    if not text or "Date,Open,High,Low,Close" not in text:
        return []

    return _parse_ohlc_csv_points(text, days)


async def _history_from_pdr_stooq(symbol: str, days: int) -> List[Dict[str, Any]]:
    if pdr is None:
        return []

    sym = (symbol or "").strip()
    if not sym:
        return []

    end = _utc_now().date()
    start = (_utc_now() - timedelta(days=int(max(5, days)) + 10)).date()

    try:
        df = pdr.DataReader(sym, "stooq", start, end)
        if df is None or df.empty:
            return []
        df = df.sort_index()
        if days and len(df) > int(days):
            df = df.iloc[-int(days) :]

        return _to_points_from_df(df)
    except Exception:
        return []


async def _history_from_yfinance(symbol: str, days: int) -> List[Dict[str, Any]]:
    if not ENABLE_YFINANCE or yf is None or pd is None:
        return []

    sym = (symbol or "").strip().upper()
    if not sym:
        return []

    try:
        df = yf.download(
            sym,
            period=f"{int(max(5, days))}d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return _to_points_from_df(df)
        return []
    except Exception:
        return []


async def _batch_history_from_yfinance(symbols: List[str], days: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Batch download reduces rate-limit risk.
    Returns dict(symbol -> points). Missing symbols map to [].
    """
    out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in symbols}
    if not ENABLE_YFINANCE or yf is None or pd is None:
        return out

    syms = [str(s).upper().strip() for s in symbols if str(s).strip()]
    if not syms:
        return out

    try:
        # yfinance accepts space-separated tickers
        ticker_str = " ".join(sorted(set(syms)))

        df = yf.download(
            ticker_str,
            period=f"{int(max(5, days))}d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return out

        # Single ticker case: columns not multi-indexed by ticker
        if isinstance(df, pd.DataFrame) and "Close" in df.columns:
            pts = _to_points_from_df(df)
            sym0 = syms[0]
            out[sym0] = pts
            return out

        # Multi ticker: df columns become (ticker, field)
        if isinstance(df, pd.DataFrame):
            for sym in syms:
                try:
                    if sym in df.columns.get_level_values(0):
                        sub = df[sym]
                        pts = _to_points_from_df(sub)
                        out[sym] = pts
                except Exception:
                    continue

        return out
    except Exception:
        return out


async def get_price_history(symbol: str, days: int = 180) -> Dict[str, Any]:
    sym = (symbol or "").strip().upper()
    days = int(days or 180)
    interval = "1d"

    if not sym:
        return {"symbol": sym, "points": []}

    cached = _read_cache(sym, interval, days)
    if cached is not None:
        return {"symbol": sym, "points": cached}

    prefer = _parse_prefer_sources()
    source_map = {
        "yahoo_chart": _history_from_yahoo_chart,
        "yahoo_csv": _history_from_yahoo_csv,
        "stooq_csv": _history_from_stooq_csv,
        "pdr_stooq": _history_from_pdr_stooq,
        "yfinance": _history_from_yfinance,
    }

    for src in prefer:
        getter = source_map.get(src)
        if getter is None:
            continue
        pts = await getter(sym, days)
        if pts and len(pts) >= 1:
            _write_cache(sym, interval, days, pts, src)
            return {"symbol": sym, "points": pts}

    _write_cache(sym, interval, days, [], "none")
    return {"symbol": sym, "points": []}


async def get_price_histories(
    symbols: List[str],
    days: int = 180,
    concurrency: int = 6,
) -> Dict[str, Dict[str, Any]]:
    """
    Bulk version used by signals/dashboard.
    Strategy:
    1) Read cache for all symbols.
    2) For missing, try ONE yfinance batch (if enabled).
    3) For still missing, fall back per-symbol using free sources with bounded concurrency + delay.
    """
    days = int(days or 180)
    interval = "1d"

    uniq = list(dict.fromkeys([str(s).upper().strip() for s in (symbols or []) if str(s).strip()]))
    if not uniq:
        return {}

    results: Dict[str, Dict[str, Any]] = {}

    missing: List[str] = []
    for sym in uniq:
        cached = _read_cache(sym, interval, days)
        if cached is not None:
            results[sym] = {"symbol": sym, "points": cached}
        else:
            missing.append(sym)

    # Batch yfinance first (optional)
    if missing and ENABLE_YFINANCE and yf is not None:
        yb = await _batch_history_from_yfinance(missing, days)
        for sym in list(missing):
            pts = yb.get(sym) or []
            if pts and len(pts) >= 1:
                _write_cache(sym, interval, days, pts, "yfinance_batch")
                results[sym] = {"symbol": sym, "points": pts}
                missing.remove(sym)

    # Per-symbol fallback with bounded concurrency
    if missing:
        sem = asyncio.Semaphore(max(1, int(concurrency)))
        prefer = _parse_prefer_sources()

        # Remove yfinance from per-symbol fallback order if batch already attempted
        prefer2 = [p for p in prefer if p != "yfinance"]

        source_map = {
            "yahoo_chart": _history_from_yahoo_chart,
            "yahoo_csv": _history_from_yahoo_csv,
            "stooq_csv": _history_from_stooq_csv,
            "pdr_stooq": _history_from_pdr_stooq,
        }

        async def one(sym: str) -> None:
            async with sem:
                # try sources in order; first that returns data wins
                for src in prefer2:
                    getter = source_map.get(src)
                    if getter is None:
                        continue
                    pts = await getter(sym, days)
                    if pts and len(pts) >= 1:
                        _write_cache(sym, interval, days, pts, src)
                        results[sym] = {"symbol": sym, "points": pts}
                        return

                _write_cache(sym, interval, days, [], "none")
                results[sym] = {"symbol": sym, "points": []}

        await asyncio.gather(*(one(s) for s in missing))

    return results


def _news_to_markers(news_docs: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for n in news_docs or []:
        if n is None:
            continue

        if isinstance(n, dict):
            pub = n.get("published_at")
            title = n.get("title")
            url = n.get("url")
            sym = n.get("symbol")
            score = n.get("score")
        else:
            pub = getattr(n, "published_at", None)
            title = getattr(n, "title", None)
            url = getattr(n, "url", None)
            sym = getattr(n, "symbol", None)
            score = getattr(n, "score", None)

        dt = None
        if isinstance(pub, datetime):
            dt = pub
        elif isinstance(pub, str) and pub:
            try:
                dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            except Exception:
                dt = None

        out.append(
            {
                "date": _iso_date(dt) if dt else None,
                "title": str(title or ""),
                "url": str(url or ""),
                "symbol": str(sym or ""),
                "score": float(score or 0.0),
            }
        )
    return out


async def get_annotated_history(symbol: str, days: int = 180, news_docs: Optional[List[Any]] = None) -> Dict[str, Any]:
    base = await get_price_history(symbol, days=days)
    markers = _news_to_markers(news_docs or [])
    return {"symbol": base["symbol"], "points": base["points"], "news_markers": markers}
