# backend/app/db.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional, Sequence, Tuple, Union

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

# Environment variables (keep your existing names)
_MONGO_URL = os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017")
_DB_NAME = os.getenv("MONGO_DB", "financial_advisor_bot")

# Optional cache TTLs (set to "0" to disable expiry)
_MARKET_CACHE_TTL_DAYS = int(os.getenv("MARKET_CACHE_TTL_DAYS", "7"))
_NEWS_CACHE_TTL_DAYS = int(os.getenv("NEWS_CACHE_TTL_DAYS", "30"))

_client: Optional[MongoClient] = None
_db: Optional[Database] = None

# Collections used across services
users_col: Optional[Collection] = None
risk_profiles_col: Optional[Collection] = None

# Signals + dashboard caches (daily)
daily_signals_col: Optional[Collection] = None
dashboard_cache_col: Optional[Collection] = None

# News + universe caches
news_cache_col: Optional[Collection] = None
universe_cache_col: Optional[Collection] = None

# Chosen stocks (optional)
chosen_stocks_col: Optional[Collection] = None

# Market data storage (cache + historical snapshots)
market_prices_col: Optional[Collection] = None

# Optional: store user portfolio snapshots / audit logs later
portfolio_snapshots_col: Optional[Collection] = None
audit_logs_col: Optional[Collection] = None

# ML workflow storage
ml_models_col: Optional[Collection] = None
ml_model_runs_col: Optional[Collection] = None
ml_runtime_settings_col: Optional[Collection] = None

IndexKeys = Union[str, Sequence[Tuple[str, int]]]


def init_db() -> None:
    """
    Initialise Mongo client, database, collections, and indexes.
    Safe to call multiple times.

    MongoDB creates collections on first use. Index creation here both
    ensures the collections exist and enforces uniqueness constraints.
    """
    global _client, _db
    global users_col, risk_profiles_col
    global daily_signals_col, dashboard_cache_col
    global news_cache_col, universe_cache_col, chosen_stocks_col
    global market_prices_col, portfolio_snapshots_col, audit_logs_col
    global ml_models_col, ml_model_runs_col, ml_runtime_settings_col

    if _client is None or _db is None:
        _client = MongoClient(_MONGO_URL)
        _db = _client[_DB_NAME]
        _client.admin.command("ping")

    # Core user data
    users_col = _db["users"]
    risk_profiles_col = _db["risk_profiles"]

    # Daily computed outputs (cache by date)
    daily_signals_col = _db["daily_signals"]
    dashboard_cache_col = _db["dashboard_cache"]

    # Cached external data
    news_cache_col = _db["news_cache"]
    universe_cache_col = _db["universe_cache"]

    # Optional derived data
    chosen_stocks_col = _db["chosen_stocks"]

    # Market data storage (price history points)
    market_prices_col = _db["market_prices"]

    # Optional: keep these for later features
    portfolio_snapshots_col = _db["portfolio_snapshots"]
    audit_logs_col = _db["audit_logs"]

    # ML workflow collections
    ml_models_col = _db["ml_models"]
    ml_model_runs_col = _db["ml_model_runs"]
    ml_runtime_settings_col = _db["ml_runtime_settings"]

    _ensure_indexes()


def close_db() -> None:
    """
    Close Mongo client and clear module globals.
    Safe to call multiple times.
    """
    global _client, _db
    global users_col, risk_profiles_col
    global daily_signals_col, dashboard_cache_col
    global news_cache_col, universe_cache_col, chosen_stocks_col
    global market_prices_col, portfolio_snapshots_col, audit_logs_col
    global ml_models_col, ml_model_runs_col, ml_runtime_settings_col

    try:
        if _client is not None:
            _client.close()
    finally:
        _client = None
        _db = None

        users_col = None
        risk_profiles_col = None

        daily_signals_col = None
        dashboard_cache_col = None

        news_cache_col = None
        universe_cache_col = None

        chosen_stocks_col = None

        market_prices_col = None
        portfolio_snapshots_col = None
        audit_logs_col = None
        ml_models_col = None
        ml_model_runs_col = None
        ml_runtime_settings_col = None


def require_col(col: Optional[Collection], name: str) -> Collection:
    """
    Use this in services to guarantee init_db() has run.
    """
    if col is None:
        raise RuntimeError(f"Database not initialised: {name}. Call init_db() at startup.")
    return col


def get_db() -> Database:
    """
    Returns the pymongo Database instance.
    """
    if _db is None:
        raise RuntimeError("Database not initialised. Call init_db() at startup.")
    return _db


def client() -> MongoClient:
    """
    Returns the MongoClient instance.
    """
    if _client is None:
        raise RuntimeError("Mongo client not initialised. Call init_db() at startup.")
    return _client


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_keys(keys: IndexKeys) -> Sequence[Tuple[str, int]]:
    if isinstance(keys, str):
        return [(keys, 1)]
    return list(keys)


def _ensure_index(
    col: Collection,
    keys: IndexKeys,
    *,
    name: str,
    unique: bool = False,
    expire_after_seconds: Optional[int] = None,
) -> None:
    """
    Create an index safely and idempotently.

    Fixes startup crashes caused by index option conflicts:
    - If an index with the same name exists but with different options
      (unique=True vs unique=False, TTL differences, etc), Mongo throws.
    - We detect the conflict and drop + recreate.

    If we cannot drop/create (permissions), avoid crashing startup.
    """
    keys_norm = _normalize_keys(keys)

    try:
        info = col.index_information() or {}
    except Exception:
        info = {}

    existing = info.get(name)
    if existing:
        existing_key = list(existing.get("key", []))
        existing_unique = bool(existing.get("unique", False))
        existing_expire = existing.get("expireAfterSeconds", None)

        want_unique = bool(unique)
        want_expire = expire_after_seconds

        same_key = existing_key == list(keys_norm)
        same_unique = existing_unique == want_unique
        same_expire = existing_expire == want_expire

        if same_key and same_unique and same_expire:
            return

        try:
            col.drop_index(name)
        except Exception:
            return

    kwargs = {"name": name, "unique": unique}
    if expire_after_seconds is not None:
        kwargs["expireAfterSeconds"] = int(expire_after_seconds)

    try:
        col.create_index(list(keys_norm), **kwargs)
    except Exception:
        return


def _ensure_indexes() -> None:
    """
    Create all indexes (idempotent). This effectively "creates tables"
    in the way MongoDB works.

    All indexes use explicit names to prevent auto-generated name conflicts.
    """
    u = require_col(users_col, "users")
    rp = require_col(risk_profiles_col, "risk_profiles")
    ds = require_col(daily_signals_col, "daily_signals")
    dc = require_col(dashboard_cache_col, "dashboard_cache")
    nc = require_col(news_cache_col, "news_cache")
    uc = require_col(universe_cache_col, "universe_cache")
    cs = require_col(chosen_stocks_col, "chosen_stocks")
    mp = require_col(market_prices_col, "market_prices")
    mm = require_col(ml_models_col, "ml_models")
    mr = require_col(ml_model_runs_col, "ml_model_runs")
    mrs = require_col(ml_runtime_settings_col, "ml_runtime_settings")

    # Users
    _ensure_index(u, "email", name="users_email_1", unique=True)
    _ensure_index(u, "created_at", name="users_created_at_1", unique=False)

    # Risk profiles
    _ensure_index(rp, "user_id", name="risk_profiles_user_id_1", unique=True)
    _ensure_index(rp, "updated_at", name="risk_profiles_updated_at_1", unique=False)

    # News: unique by (date, symbol, id)
    # NOTE: symbol can be None for market-wide items; Mongo unique indexes
    # treat missing/None consistently, and the 'id' (url) should be unique anyway.
    _ensure_index(nc, [("date", 1), ("symbol", 1), ("id", 1)], name="news_date_symbol_id", unique=True)
    _ensure_index(nc, "published_at", name="news_published_at_1", unique=False)

    # TTL needs a date field, so we index fetched_at. Ensure news.py sets it.
    if _NEWS_CACHE_TTL_DAYS > 0:
        _ensure_index(
            nc,
            "fetched_at",
            name="ttl_news_fetched_at",
            unique=False,
            expire_after_seconds=int(_NEWS_CACHE_TTL_DAYS * 86400),
        )

    # Universe: unique by (date, key) (and safely fixes old wrong date_1)
    _ensure_universe_unique_index(uc)

    # Daily signals: unique per user per date
    _ensure_index(ds, [("user_id", 1), ("date", 1)], name="daily_signals_user_date", unique=True)
    _ensure_index(ds, "date", name="daily_signals_date_1", unique=False)

    # Dashboard cache: unique per user per date
    _ensure_index(dc, [("user_id", 1), ("date", 1)], name="dashboard_user_date", unique=True)

    # Keep date index non-unique because multiple users share the same date.
    _ensure_index(dc, "date", name="dashboard_date_1", unique=False)

    # Chosen stocks per day (optional): one record per date
    _ensure_index(cs, "date", name="chosen_stocks_date_1", unique=True)

    # Market prices: cache key to prevent duplicate fetch payloads
    _ensure_index(mp, [("symbol", 1), ("interval", 1), ("days", 1)], name="market_prices_key", unique=True)
    _ensure_index(mp, "fetched_at", name="market_prices_fetched_at_1", unique=False)

    if _MARKET_CACHE_TTL_DAYS > 0:
        _ensure_index(
            mp,
            "fetched_at",
            name="ttl_market_fetched_at",
            unique=False,
            expire_after_seconds=int(_MARKET_CACHE_TTL_DAYS * 86400),
        )

    # ML models: registry + deployment state
    _ensure_index(mm, "model_id", name="ml_models_model_id_1", unique=True)
    _ensure_index(mm, "algorithm", name="ml_models_algorithm_1", unique=False)
    _ensure_index(mm, "nlp_pipeline", name="ml_models_nlp_pipeline_1", unique=False)
    _ensure_index(mm, "rank", name="ml_models_rank_1", unique=False)
    _ensure_index(mm, "is_selected", name="ml_models_is_selected_1", unique=False)
    _ensure_index(mm, "is_deployed", name="ml_models_is_deployed_1", unique=False)
    _ensure_index(mm, "created_at", name="ml_models_created_at_1", unique=False)

    # ML runs: async workflow executions
    _ensure_index(mr, "run_id", name="ml_runs_run_id_1", unique=True)
    _ensure_index(mr, "status", name="ml_runs_status_1", unique=False)
    _ensure_index(mr, "created_at", name="ml_runs_created_at_1", unique=False)

    # ML runtime settings: singleton key document
    _ensure_index(mrs, "key", name="ml_runtime_settings_key_1", unique=True)


def _ensure_universe_unique_index(col: Collection) -> None:
    """
    Ensures universe_cache has the correct unique index (date, key).
    If an old wrong index exists (date_1), drop it to avoid conflicts.
    """
    try:
        existing = col.index_information() or {}
        if "date_1" in existing:
            try:
                col.drop_index("date_1")
            except Exception:
                pass
    except Exception:
        pass

    _ensure_index(col, [("date", 1), ("key", 1)], name="universe_date_key", unique=True)
