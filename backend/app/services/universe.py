from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .. import db

try:
    import pandas as pd
except Exception:
    pd = None


DOW_30_FALLBACK = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
]

SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","TSLA","UNH","JPM",
    "XOM","V","LLY","AVGO","JNJ","WMT","PG","MA","HD","MRK",
    "ABBV","COST","PEP","ADBE","KO","CRM","CSCO","NFLX","AMD","QCOM",
]


def _scrape_table_symbols(url: str, col_name_candidates: List[str]) -> List[str]:
    if pd is None:
        return []
    tables = pd.read_html(url)
    for t in tables:
        cols = [str(c) for c in t.columns]
        for cname in col_name_candidates:
            if cname in cols:
                syms = [str(x).strip() for x in t[cname].tolist()]
                syms = [s.replace(".", "-").upper() for s in syms if s]
                return list(dict.fromkeys(syms))
    return []


def get_dow_symbols() -> List[str]:
    override = get_universe_override("dow")
    if override:
        return override

    try:
        syms = _scrape_table_symbols(
            "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
            ["Symbol", "Ticker"],
        )
        syms = [s for s in syms if len(s) <= 8]
        return syms or DOW_30_FALLBACK[:]
    except Exception:
        return DOW_30_FALLBACK[:]


def get_sp500_symbols() -> List[str]:
    override = get_universe_override("sp500")
    if override:
        return override

    try:
        syms = _scrape_table_symbols(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            ["Symbol", "Ticker symbol", "Ticker"],
        )
        return syms or SP500_FALLBACK[:]
    except Exception:
        return SP500_FALLBACK[:]


def save_universe_for_date(date: str, key: str, symbols: List[str], meta: Optional[Dict] = None) -> None:
    col = db.require_col(db.universe_cache_col, "universe_cache")
    key = str(key).lower().strip()
    symbols = list(dict.fromkeys([str(s).upper().strip() for s in (symbols or []) if str(s).strip()]))

    col.update_one(
        {"date": date, "key": key},
        {"$set": {"date": date, "key": key, "symbols": symbols, "meta": meta or {}}},
        upsert=True,
    )


def get_universe_for_date(date: str, key: str) -> List[str]:
    col = db.require_col(db.universe_cache_col, "universe_cache")
    key = str(key).lower().strip()

    doc = col.find_one({"date": date, "key": key})
    if doc and isinstance(doc.get("symbols"), list) and doc["symbols"]:
        return list(dict.fromkeys([str(x).upper() for x in doc["symbols"]]))

    if key == "dow":
        return get_dow_symbols()
    return get_sp500_symbols()


def set_universe_override(key: str, symbols: List[str]) -> List[str]:
    key = str(key).lower().strip()
    vals = list(dict.fromkeys([str(s).upper().strip() for s in (symbols or []) if str(s).strip()]))
    col = db.get_db()["universe_overrides"]
    col.update_one(
        {"key": key},
        {"$set": {"key": key, "symbols": vals}},
        upsert=True,
    )
    return vals


def get_universe_override(key: str) -> List[str]:
    key = str(key).lower().strip()
    try:
        col = db.get_db()["universe_overrides"]
        doc = col.find_one({"key": key}) or {}
        vals = doc.get("symbols") or []
        return [str(s).upper().strip() for s in vals if str(s).strip()]
    except Exception:
        return []


def set_user_custom_universe(user_id: str, symbols: List[str]) -> List[str]:
    vals = list(dict.fromkeys([str(s).upper().strip() for s in (symbols or []) if str(s).strip()]))
    col = db.get_db()["user_universe"]
    col.update_one(
        {"user_id": user_id},
        {"$set": {"user_id": user_id, "symbols": vals}},
        upsert=True,
    )
    return vals


def get_user_custom_universe(user_id: str) -> List[str]:
    try:
        col = db.get_db()["user_universe"]
        doc = col.find_one({"user_id": user_id}) or {}
        vals = doc.get("symbols") or []
        return [str(s).upper().strip() for s in vals if str(s).strip()]
    except Exception:
        return []
