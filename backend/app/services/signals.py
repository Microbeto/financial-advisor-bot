from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from .. import db
from ..models import DashboardResponse, NewsItem, Regime, RiskProfile, TrendItem
from ..core.utils import iso_date_utc
from ..engine.glossary import lingo_glossary
from .news import refresh_top_news_of_day, get_news_for_date
from .symbols import resolve_symbol_name
from .universe import get_universe_for_date, get_user_custom_universe, save_universe_for_date
from .risk import ensure_risk_profile_for_user, get_risk_profile_for_user
from .users import get_user_role
from .market import get_price_history
from .policy import PolicyEngine

try:
    import pandas as pd
except Exception:
    pd = None


DASH_MAX_SYMBOLS_PUBLIC = int(os.getenv("DASH_MAX_SYMBOLS_PUBLIC", "35"))
DASH_MAX_SYMBOLS_PRIVATE = int(os.getenv("DASH_MAX_SYMBOLS_PRIVATE", "60"))

PRICE_DAYS_MOM = int(os.getenv("DASH_PRICE_DAYS_MOM", "120"))
PRICE_DAYS_REGIME_SPY = int(os.getenv("DASH_PRICE_DAYS_REGIME_SPY", "120"))
PRICE_DAYS_REGIME_VIX = int(os.getenv("DASH_PRICE_DAYS_REGIME_VIX", "60"))

PRICE_CONCURRENCY = int(os.getenv("DASH_PRICE_CONCURRENCY", "6"))
NEWS_WEIGHT = float(os.getenv("DASH_NEWS_WEIGHT", "0.10"))

DASH_CACHE_TTL_SEC = int(os.getenv("DASH_CACHE_TTL_SEC", "900"))
SIGNALS_CACHE_TTL_SEC = int(os.getenv("SIGNALS_CACHE_TTL_SEC", "86400"))


async def _maybe_await(res: Any) -> Any:
    if asyncio.iscoroutine(res):
        return await res
    return res


def _utc_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_doc_fresh(doc: dict, ttl_sec: int) -> bool:
    try:
        s = doc.get("cached_at") or doc.get("fetched_at")
        if not s:
            return False
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return age < float(ttl_sec)
    except Exception:
        return False


def _cap_symbols(user_role: str, symbols: List[str]) -> List[str]:
    out = list(dict.fromkeys([str(s).upper().strip() for s in (symbols or []) if str(s).strip()]))
    if user_role == "user":
        return out[: max(1, DASH_MAX_SYMBOLS_PUBLIC)]
    return out[: max(1, DASH_MAX_SYMBOLS_PRIVATE)]


def _vol20_from_closes(closes: List[float]) -> float:
    if len(closes) < 22:
        return 0.0

    if pd is not None:
        try:
            s = pd.Series(closes, dtype="float64")
            ret = s.pct_change().dropna()
            if len(ret) >= 20:
                return float(ret.tail(20).std())
            return 0.0
        except Exception:
            pass

    rets: List[float] = []
    for i in range(1, len(closes)):
        p0 = closes[i - 1]
        p1 = closes[i]
        if p0 > 0:
            rets.append((p1 / p0) - 1.0)

    last = rets[-20:] if len(rets) >= 20 else rets
    if len(last) < 2:
        return 0.0
    mu = sum(last) / len(last)
    var = sum((x - mu) ** 2 for x in last) / (len(last) - 1)
    return float(var ** 0.5)


def _momentum_from_closes(closes: List[float]) -> Tuple[float, float, float, float]:
    if len(closes) < 25:
        return (0.0, 0.0, 0.0, 0.0)

    r5 = (closes[-1] / closes[-6]) - 1.0 if len(closes) >= 6 else 0.0
    r20 = (closes[-1] / closes[-21]) - 1.0 if len(closes) >= 21 else 0.0
    vol20 = _vol20_from_closes(closes)
    score = (0.55 * float(r20)) + (0.35 * float(r5)) - (0.25 * float(vol20))
    return (float(score), float(r5), float(r20), float(vol20))


def _news_score_map(news: List[NewsItem]) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for n in news or []:
        sym = str(getattr(n, "symbol", "")).upper()
        if not sym:
            continue
        m[sym] = m.get(sym, 0.0) + float(getattr(n, "score", 0.0) or 0.0)
    return m


async def _close_series_from_history(symbol: str, days: int) -> List[float]:
    h = await get_price_history(symbol, days=days)
    pts = h.get("points") or []
    closes: List[float] = []
    for p in pts:
        try:
            closes.append(float(p["c"]))
        except Exception:
            continue
    return closes


async def _fetch_closes_many(symbols: List[str], days: int, concurrency: int) -> Dict[str, List[float]]:
    uniq = list(dict.fromkeys([str(s).upper().strip() for s in (symbols or []) if str(s).strip()]))
    if not uniq:
        return {}

    sem = asyncio.Semaphore(max(1, int(concurrency)))
    out: Dict[str, List[float]] = {}

    async def one(sym: str) -> None:
        async with sem:
            closes = await _close_series_from_history(sym, days=days)
        out[sym] = closes

    await asyncio.gather(*(one(s) for s in uniq))
    return out


async def _compute_regime_from_closes(closes_map: Dict[str, List[float]]) -> Regime:
    try:
        spy = closes_map.get("SPY") or []
        if len(spy) < 25:
            return "neutral"

        r20 = (spy[-1] / spy[-21]) - 1.0 if len(spy) >= 21 else 0.0

        vix = closes_map.get("^VIX") or []
        vix_last = float(vix[-1]) if len(vix) >= 1 else 0.0

        if r20 < 0.0 or vix_last > 20.0:
            return "risk_off"
        if r20 > 0.0 and vix_last <= 20.0:
            return "risk_on"
        return "neutral"
    except Exception:
        return "neutral"


async def _rank_trends(
    date: str,
    symbols: List[str],
    closes_map: Dict[str, List[float]],
) -> List[TrendItem]:
    symbols = list(dict.fromkeys([str(s).upper().strip() for s in (symbols or []) if str(s).strip()]))
    if not symbols:
        return []

    try:
        news: List[NewsItem] = get_news_for_date(date, symbols=symbols, limit=120)
    except Exception:
        news = []

    nmap = _news_score_map(news)

    items: List[TrendItem] = []
    for sym in symbols:
        closes = closes_map.get(sym) or []
        sc, r5, r20, vol20 = _momentum_from_closes(closes)
        if sc == 0.0 and r5 == 0.0 and r20 == 0.0 and vol20 == 0.0:
            continue

        sc2 = float(sc) + (NEWS_WEIGHT * float(nmap.get(sym, 0.0)))
        direction = "up" if sc2 >= 0 else "down"

        link = None
        for n in news:
            try:
                if str(n.symbol).upper() == sym and getattr(n, "url", None):
                    link = n.url
                    break
            except Exception:
                continue

        reason = f"20D={r20:.2%}, 5D={r5:.2%}, vol20={vol20:.3f}, news={nmap.get(sym, 0.0):.1f}"

        items.append(
            TrendItem(
                symbol=sym,
                name=None,
                direction=direction,
                score=float(sc2),
                mom_5d=float(r5),
                mom_20d=float(r20),
                vol_20d=float(vol20),
                reason=reason,
                link=link,
            )
        )

    items.sort(key=lambda x: float(x.score), reverse=True)
    return items


def _read_cached_signals(user_id: str, date: str) -> Dict[str, Any] | None:
    col = db.require_col(db.daily_signals_col, "daily_signals")
    doc = col.find_one({"user_id": user_id, "date": date})
    if not doc:
        return None
    doc.pop("_id", None)
    doc.pop("user_id", None)
    return doc


def _write_cached_signals(user_id: str, date: str, payload: Dict[str, Any]) -> None:
    col = db.require_col(db.daily_signals_col, "daily_signals")
    col.update_one(
        {"user_id": user_id, "date": date},
        {"$set": {"user_id": user_id, "date": date, **payload}},
        upsert=True,
    )


def _take_top_unique(items: List[TrendItem], want: int, used: set[str]) -> List[TrendItem]:
    out: List[TrendItem] = []
    for it in items:
        sym = (it.symbol or "").upper()
        if not sym or sym in used:
            continue
        used.add(sym)
        out.append(it)
        if len(out) >= want:
            break
    return out


async def _fill_names_for_items(items: List[TrendItem]) -> None:
    async def one(it: TrendItem) -> None:
        try:
            it.name = await resolve_symbol_name(it.symbol)
        except Exception:
            it.name = None

    await asyncio.gather(*(one(x) for x in items))


async def build_dashboard_for_user(user_id: str) -> DashboardResponse:
    today = iso_date_utc()

    dash_col = db.require_col(db.dashboard_cache_col, "dashboard_cache")
    cached = dash_col.find_one({"user_id": user_id, "date": today})
    if cached:
        cached.pop("_id", None)
        cached.pop("user_id", None)
        if _is_doc_fresh(cached, DASH_CACHE_TTL_SEC):
            return DashboardResponse(**cached)

    role = "user" if user_id == "public" else str(get_user_role(user_id))

    if user_id == "public":
        profile = RiskProfile(user_id="public", risk_tolerance="balanced", constraints=[])
    else:
        profile = get_risk_profile_for_user(user_id) or ensure_risk_profile_for_user(user_id)

    policy = PolicyEngine()

    sig = _read_cached_signals(user_id, today)
    if sig and _is_doc_fresh(sig, SIGNALS_CACHE_TTL_SEC):
        top_news = sig.get("top_news")
        if top_news is None:
            top_news = await _maybe_await(refresh_top_news_of_day(today)) or []

        resp = DashboardResponse(
            date=today,
            regime=str(sig.get("regime") or "neutral"),  # type: ignore
            sp500_up=[TrendItem(**x) for x in (sig.get("sp500_up") or [])],
            sp500_down=[TrendItem(**x) for x in (sig.get("sp500_down") or [])],
            dow_up=[TrendItem(**x) for x in (sig.get("dow_up") or [])],
            dow_down=[TrendItem(**x) for x in (sig.get("dow_down") or [])],
            top_news=[NewsItem(**x) for x in (top_news or [])],
            glossary=lingo_glossary(),
        )

        payload = resp.model_dump()
        payload["cached_at"] = _utc_iso_z(datetime.now(timezone.utc))
        dash_col.update_one(
            {"user_id": user_id, "date": today},
            {"$set": {"user_id": user_id, "date": today, **payload}},
            upsert=True,
        )
        return resp

    sp500 = get_universe_for_date(today, "sp500")
    dow = get_universe_for_date(today, "dow")
    save_universe_for_date(today, "sp500", sp500)
    save_universe_for_date(today, "dow", dow)

    base_symbols = ["SPY", "^VIX"]

    regime_placeholder: Regime = "neutral"
    _ = policy.evaluate_dashboard_access(profile, regime_placeholder)

    sp500_allowed, _ = policy.filter_symbols_for_profile(sp500, profile, regime_placeholder)
    dow_allowed, _ = policy.filter_symbols_for_profile(dow, profile, regime_placeholder)

    if role in ("premium", "admin", "manager"):
        custom = get_user_custom_universe(user_id)
        sp500_allowed = list(dict.fromkeys(sp500_allowed + custom))

    sp500_allowed = _cap_symbols(role, sp500_allowed)
    dow_allowed = _cap_symbols(role, dow_allowed)

    union_symbols = list(dict.fromkeys(base_symbols + sp500_allowed + dow_allowed))
    closes_map = await _fetch_closes_many(union_symbols, days=PRICE_DAYS_MOM, concurrency=PRICE_CONCURRENCY)

    regime = await _compute_regime_from_closes(
        {
            "SPY": closes_map.get("SPY") or await _close_series_from_history("SPY", days=PRICE_DAYS_REGIME_SPY),
            "^VIX": closes_map.get("^VIX") or await _close_series_from_history("^VIX", days=PRICE_DAYS_REGIME_VIX),
        }
    )

    _ = policy.evaluate_dashboard_access(profile, regime)
    sp500_allowed, _ = policy.filter_symbols_for_profile(sp500_allowed, profile, regime)
    dow_allowed, _ = policy.filter_symbols_for_profile(dow_allowed, profile, regime)
    sp500_allowed = _cap_symbols(role, sp500_allowed)
    dow_allowed = _cap_symbols(role, dow_allowed)

    sp_ranked = await _rank_trends(today, sp500_allowed, closes_map)
    dw_ranked = await _rank_trends(today, dow_allowed, closes_map)

    used: set[str] = set()

    sp_up = _take_top_unique([x for x in sp_ranked if x.direction == "up"], 3, used)
    dw_up = _take_top_unique([x for x in dw_ranked if x.direction == "up"], 3, used)

    sp_down_sorted = sorted([x for x in sp_ranked if x.direction == "down"], key=lambda x: float(x.score))
    dw_down_sorted = sorted([x for x in dw_ranked if x.direction == "down"], key=lambda x: float(x.score))

    sp_down = _take_top_unique(sp_down_sorted, 3, used)
    dw_down = _take_top_unique(dw_down_sorted, 3, used)

    await _fill_names_for_items(sp_up + sp_down + dw_up + dw_down)

    top_news = await _maybe_await(refresh_top_news_of_day(today))
    if top_news is None:
        top_news = []

    signals_payload: Dict[str, Any] = {
        "date": today,
        "regime": regime,
        "sp500_up": [x.model_dump() for x in sp_up],
        "sp500_down": [x.model_dump() for x in sp_down],
        "dow_up": [x.model_dump() for x in dw_up],
        "dow_down": [x.model_dump() for x in dw_down],
        "top_news": [x.model_dump() for x in top_news],
        "cached_at": _utc_iso_z(datetime.now(timezone.utc)),
    }
    _write_cached_signals(user_id, today, signals_payload)

    resp = DashboardResponse(
        date=today,
        regime=regime,
        sp500_up=sp_up,
        sp500_down=sp_down,
        dow_up=dw_up,
        dow_down=dw_down,
        top_news=top_news,
        glossary=lingo_glossary(),
    )

    payload = resp.model_dump()
    payload["cached_at"] = _utc_iso_z(datetime.now(timezone.utc))
    dash_col.update_one(
        {"user_id": user_id, "date": today},
        {"$set": {"user_id": user_id, "date": today, **payload}},
        upsert=True,
    )
    return resp


def generate_daily_signals() -> None:
    async def _run() -> None:
        users = db.require_col(db.users_col, "users")

        for u in users.find({}, {"_id": 1}):
            uid = str(u["_id"])
            try:
                await build_dashboard_for_user(uid)
            except Exception:
                continue

        try:
            await build_dashboard_for_user("public")
        except Exception:
            pass

    asyncio.run(_run())
