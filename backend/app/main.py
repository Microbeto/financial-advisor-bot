# FastAPI application initialization with routes for authentication, signals, ML workflow, and admin management.
from __future__ import annotations

import inspect
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .models import (
    AdminUserDetailResponse,
    AdminUserExportResponse,
    AdminUserStateResetResponse,
    AdminUserStatusUpdateRequest,
    AnnotatedHistoryResponse,
    AuthResponse,
    Holding,
    DashboardResponse,
    GlossaryResponse,
    GlossaryExplainRequest,
    GlossaryExplainResponse,
    GlossaryTermUpdateRequest,
    CustomUniverseRequest,
    LoginRequest,
    MLDeployRequest,
    MLDeployResponse,
    MLModelListResponse,
    MLPredictionRequest,
    MLPredictionResponse,
    MLNlpPipelineMigrationRequest,
    MLPruneRequest,
    MLPruneResponse,
    MLRuntimeSettingsResponse,
    MLRuntimeSettingsUpdateRequest,
    MLSelectionRequest,
    MLSelectionResponse,
    MLTournamentStatsResponse,
    MLTrainResponse,
    MLTrainingRequest,
    PolicyConstraintRequest,
    PolicyConstraintResponse,
    PortfolioSnapshot,
    RateLimitUpdateRequest,
    RegisterRequest,
    RiskProfile,
    RoleUpdateRequest,
    SignalItem,
    SignalsToday,
    UniverseUpdateRequest,
    UserListResponse,
    UserPublic,
)
from .services.market import get_annotated_history
from .services.market import get_price_histories
from .services.news import refresh_news_if_needed, refresh_top_news_of_day
from .services.risk import (
    ensure_risk_profile_for_user,
    get_risk_profile_for_user,
    upsert_risk_profile_for_user,
)
from .services.auth import bootstrap_admin_if_configured, create_user, login_user, require_roles, verify_token
from .services.policy import PolicyEngine
from .services.scheduler import run_daily
from .services.signals import build_dashboard_for_user
from .services.admin.cache_admin import clear_cache, get_cache_stats, prune_stale_cache
from .services.universe import get_user_custom_universe, set_universe_override, set_user_custom_universe
from .services.users import delete_user, list_users, update_user_role
from .services.ml_workflow import (
    backfill_model_nlp_pipeline,
    deploy_neural_network,
    ensure_selected_model_exists,
    get_model_feature_importances,
    list_models,
    predict_for_basket,
    predict_series_for_symbol,
    prune_underperforming_models,
    get_ml_runtime_settings,
    get_tournament_competitor_stats,
    select_best_models,
    train_models_async,
    update_ml_runtime_settings,
)
from .core.utils import iso_date_utc
from .engine.glossary import lingo_glossary, remove_glossary_term, safe_text, set_glossary_term
from .engine.llm_service import GenerativeIntelligence
from .ml.model_router import intelligence_router


def _parse_cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).strip()

    if not raw:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]

    if raw == "*":
        return ["*"]

    return [x.strip() for x in raw.split(",") if x.strip()]


async def _maybe_await(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res


async def _acquire_limit(key: str) -> None:
    try:
        from .core.rate_limit import limiter  # type: ignore

        await limiter.acquire(key)
    except Exception:
        return


def _acquire_limit_sync(key: str) -> None:
    """
    Run async limiter acquisition from sync routes executed in FastAPI's threadpool.
    """
    try:
        import anyio

        anyio.from_thread.run(_acquire_limit, key)
    except Exception:
        return


def _extract_bearer_token(authorization: str | None) -> str | None:
    raw = (authorization or "").strip()
    if not raw:
        return None
    if raw.lower().startswith("bearer "):
        tok = raw[7:].strip()
        return tok or None
    return raw


def _claims_optional(authorization: str | None) -> dict[str, str] | None:
    token = _extract_bearer_token(authorization)
    if not token:
        return None
    return verify_token(token)


def _claims_required(authorization: str | None) -> dict[str, str]:
    claims = _claims_optional(authorization)
    if not claims:
        raise HTTPException(status_code=401, detail="Authentication required")
    return claims


def _utc_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        db.init_db()
    except Exception:
        # Degraded startup mode: API stays alive, /health will report db=false.
        pass
    try:
        app.state.db = db.get_db()
    except Exception:
        app.state.db = None

    try:
        from .core.rate_limit import limiter  # type: ignore

        limiter.configure("stooq", per_minute=int(os.getenv("STOOQ_RPM", "30")))
        limiter.configure("gdelt", per_minute=int(os.getenv("GDELT_RPM", "15")))
        limiter.configure("wiki", per_minute=int(os.getenv("WIKI_RPM", "10")))
        limiter.configure("yahoo", per_minute=int(os.getenv("YAHOO_RPM", "45")))

        limiter.configure("api_health", per_minute=int(os.getenv("API_HEALTH_RPM", "180")))
        limiter.configure("api_read", per_minute=int(os.getenv("API_READ_RPM", "120")))
        limiter.configure("api_write", per_minute=int(os.getenv("API_WRITE_RPM", "60")))
        limiter.configure("api_llm", per_minute=int(os.getenv("API_LLM_RPM", "12")))
        limiter.configure("api_ml_train", per_minute=int(os.getenv("API_ML_TRAIN_RPM", "10")))
        limiter.configure("api_ml_predict", per_minute=int(os.getenv("API_ML_PREDICT_RPM", "30")))
        limiter.configure("api_ml_admin", per_minute=int(os.getenv("API_ML_ADMIN_RPM", "20")))
    except Exception:
        pass

    try:
        _ = ensure_selected_model_exists()
    except Exception:
        pass

    try:
        import asyncio as _asyncio

        async def _bg_startup_viz() -> None:
            try:
                from .ml.visualizations import run_startup_visualizations
                from .services.ml_workflow import _model_store_dir
                await _asyncio.to_thread(run_startup_visualizations, _model_store_dir())
            except Exception:
                pass

        _asyncio.create_task(_bg_startup_viz())
    except Exception:
        pass

    try:
        bootstrap_admin_if_configured()
    except Exception:
        pass

    # Set safe defaults immediately – FinBERT / Ollama probes run in background
    # so the /health endpoint is reachable seconds after startup.
    app.state.model_router = intelligence_router
    app.state.model_router_diagnostics = {"tier": "low", "status": "pending"}
    app.state.system_capability = "low"

    try:
        import asyncio as _asyncio_diag

        async def _bg_diagnostics() -> None:
            try:
                diagnostics = await intelligence_router.run_diagnostics()
                app.state.model_router_diagnostics = diagnostics
                app.state.system_capability = diagnostics.get("tier", "low")
            except Exception:
                app.state.model_router_diagnostics = {"tier": "low", "error": "router_diagnostics_failed"}
                app.state.system_capability = "low"

        _asyncio_diag.create_task(_bg_diagnostics())
    except Exception:
        pass

    yield

    try:
        db.close_db()
    finally:
        app.state.db = None


def create_app() -> FastAPI:
    app = FastAPI(title="Financial Advisor Bot API", lifespan=lifespan)
    glossary_ai = GenerativeIntelligence()

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        msg = str(exc or "")
        if "Database not initialised" in msg:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Database unavailable. Start MongoDB and retry.",
                    "code": "database_unavailable",
                    "path": str(request.url.path),
                },
            )
        return JSONResponse(status_code=500, content={"detail": msg or "Internal server error"})

    def _normalize_term(s: str) -> str:
        return str(s or "").strip().lower()

    def _lookup_glossary_definition(term: str) -> tuple[str, str] | None:
        q = _normalize_term(term)
        if not q:
            return None
        for key, value in (lingo_glossary() or {}).items():
            if _normalize_term(key) == q:
                k = str(key or "").strip()
                v = str(value or "").strip()
                if k and v:
                    return k, v
        return None

    def _fallback_term_definition(term: str) -> str:
        t = str(term or "").strip()
        return (
            f"{t} is a finance term used to describe market behavior, valuation, risk, or portfolio impact. "
            "Exact usage can vary by strategy and timeframe."
        )

    def _obj_id(s: str):
        try:
            from bson import ObjectId

            return ObjectId(s)
        except Exception:
            return None

    def _to_dt(v: Any) -> datetime | None:
        if isinstance(v, datetime):
            return v.astimezone(timezone.utc)
        if isinstance(v, str) and v:
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                return None
        return None

    def _log_user_error(user_id: str | None, path: str, method: str, message: str) -> None:
        try:
            col = db.require_col(db.audit_logs_col, "audit_logs")
            col.insert_one(
                {
                    "user_id": user_id,
                    "level": "error",
                    "path": path,
                    "method": method,
                    "message": str(message)[:1000],
                    "created_at": db.utc_now(),
                }
            )
        except Exception:
            return

    def _price_for_buy_date(points: list[dict[str, Any]], buy_date: str) -> float:
        if not points:
            return 0.0

        exact_price = 0.0
        prior_price = 0.0
        for p in points:
            t_raw = str(p.get("t") or "")
            d = t_raw[:10]
            try:
                c = float(p.get("c", 0.0) or 0.0)
            except Exception:
                c = 0.0
            if c <= 0.0:
                continue
            if d == buy_date:
                exact_price = c
                break
            if d and d < buy_date:
                prior_price = c

        if exact_price > 0.0:
            return float(exact_price)
        if prior_price > 0.0:
            return float(prior_price)

        for p in points:
            try:
                c = float(p.get("c", 0.0) or 0.0)
            except Exception:
                c = 0.0
            if c > 0.0:
                return float(c)
        return 0.0

    async def _admin_default_portfolio() -> PortfolioSnapshot:
        candidates = [
            "SPY",
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "JPM",
            "UNH",
            "XOM",
            "TSLA",
            "AVGO",
            "AMD",
            "NFLX",
            "GS",
            "COST",
            "LLY",
            "V",
            "WMT",
            "HD",
        ]

        try:
            pred = await _maybe_await(predict_for_basket, candidates, lookback_days=30, model_id=None)
            ranked = sorted(pred.items, key=lambda x: float(x.probability_up), reverse=True)
            top = ranked[:10]
        except Exception:
            top = []

        buy_date = "2025-03-13"
        symbols = [x.symbol for x in top if getattr(x, "symbol", None)]
        histories = await get_price_histories(symbols, days=520, concurrency=8) if symbols else {}

        holdings: list[Holding] = []
        for item in top:
            sym = str(item.symbol).upper()
            pts = (histories.get(sym) or {}).get("points") or []
            buy_price = _price_for_buy_date(pts, buy_date)
            if buy_price <= 0:
                buy_price = 100.0
            holdings.append(Holding(symbol=sym, quantity=100.0, avg_price=float(buy_price)))

        return PortfolioSnapshot(cash=0.0, holdings=holdings)

    def _is_legacy_admin_seed(snapshot: PortfolioSnapshot) -> bool:
        hs = list(snapshot.holdings or [])
        if not hs:
            return False
        if len(hs) < 5:
            return False
        for h in hs:
            q = float(getattr(h, "quantity", 0.0) or 0.0)
            p = float(getattr(h, "avg_price", 0.0) or 0.0)
            if abs(q - 1.0) > 1e-9:
                return False
            if abs(p - 100.0) > 1e-9:
                return False
        return True

    def _build_ml_audit_entries() -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        try:
            runs_col = db.require_col(db.ml_model_runs_col, "ml_model_runs")
            runs = list(runs_col.find({}).sort([("created_at", -1)]).limit(200))
        except Exception:
            runs = []

        try:
            models_col = db.require_col(db.ml_models_col, "ml_models")
            models = list(models_col.find({}).sort([("created_at", -1)]).limit(300))
        except Exception:
            models = []

        idx = 1
        for r in runs:
            run_id = str(r.get("run_id") or "")
            status = str(r.get("status") or "unknown")
            stage = str(r.get("stage") or "")
            result = r.get("result") or {}
            selected_model_id = str(result.get("selected_model_id") or "")
            trained_count = int(result.get("trained_count") or 0)
            pruned_count = int(result.get("pruned_count") or 0)

            created = r.get("created_at") or db.utc_now()
            ts = created.isoformat() if hasattr(created, "isoformat") else str(created)
            description = (
                f"ML run {run_id} finished with status={status} stage={stage}. "
                f"Trained {trained_count} models, selected={selected_model_id or 'none'}, pruned={pruned_count}."
            )
            entries.append(
                {
                    "id": idx,
                    "timestamp": ts,
                    "description": description,
                    "num_recommendations": trained_count,
                    "regime": "ml_training",
                    "model_id": selected_model_id or None,
                }
            )
            idx += 1

        for m in models:
            model_id = str(m.get("model_id") or "")
            algo = str(m.get("algorithm") or "")
            rank = int(m.get("rank") or 0)
            score = float(m.get("score") or 0.0)
            selected = bool(m.get("is_selected", False))
            deployed = bool(m.get("is_deployed", False))
            metrics = m.get("metrics") or {}
            f1 = float(metrics.get("f1") or 0.0)

            created = m.get("created_at") or db.utc_now()
            ts = created.isoformat() if hasattr(created, "isoformat") else str(created)
            how_applied = []
            if selected:
                how_applied.append("used for current prediction routing")
            if deployed:
                how_applied.append("deployed neural-network candidate")
            applied_text = "; ".join(how_applied) if how_applied else "stored in model registry"

            description = (
                f"Model {model_id} ({algo}) created with rank={rank}, score={score:.3f}, f1={f1:.3f}; "
                f"application: {applied_text}."
            )
            entries.append(
                {
                    "id": idx,
                    "timestamp": ts,
                    "description": description,
                    "num_recommendations": 1,
                    "regime": "ml_model",
                    "model_id": model_id,
                }
            )
            idx += 1

        entries.sort(key=lambda x: str(x.get("timestamp") or ""), reverse=True)
        for i, e in enumerate(entries, start=1):
            e["id"] = i
        return entries[:300]

    origins = _parse_cors_origins()
    allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    if origins == ["*"]:
        allow_credentials = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def capture_user_errors(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            user_id: str | None = None
            try:
                claims = _claims_optional(request.headers.get("authorization"))
                if claims:
                    user_id = str(claims.get("uid", "") or "") or None
            except Exception:
                user_id = None
            _log_user_error(user_id, request.url.path, request.method, str(exc))
            raise

    @app.get("/health")
    def health():
        _acquire_limit_sync("api_health")
        db_ok = True
        try:
            _ = db.get_db()
        except Exception:
            db_ok = False
        return {"ok": True, "db": db_ok}

    @app.get("/glossary", response_model=GlossaryResponse)
    def glossary():
        _acquire_limit_sync("api_read")
        return GlossaryResponse(terms=lingo_glossary())

    @app.post("/glossary/explain", response_model=GlossaryExplainResponse)
    async def glossary_explain(payload: GlossaryExplainRequest):
        await _acquire_limit("api_read")
        # Strict queue guard for explicit LLM glossary requests.
        await _acquire_limit("api_llm")

        raw_term = str(payload.term or "").strip()
        if not raw_term:
            raise HTTPException(status_code=400, detail="term is required")
        if len(raw_term) > 80:
            raise HTTPException(status_code=400, detail="term is too long")

        existing = _lookup_glossary_definition(raw_term)
        if existing:
            return GlossaryExplainResponse(
                term=existing[0],
                definition=safe_text(existing[1], limit=360),
                generated=False,
                source="existing",
            )

        llm_available = bool(getattr(intelligence_router, "ollama_available", False))
        llm_text = None
        if llm_available:
            try:
                if await glossary_ai.check_health():
                    llm_text = await _maybe_await(glossary_ai.generate_term_definition, raw_term)
            except Exception:
                llm_text = None

        if llm_text:
            definition = safe_text(llm_text, limit=360)
            set_glossary_term(raw_term, definition)
            return GlossaryExplainResponse(
                term=raw_term,
                definition=definition,
                generated=True,
                source="llm",
            )

        fallback = safe_text(_fallback_term_definition(raw_term), limit=360)
        set_glossary_term(raw_term, fallback)
        return GlossaryExplainResponse(
            term=raw_term,
            definition=fallback,
            generated=True,
            source="fallback",
        )

    @app.post("/auth/register", response_model=AuthResponse)
    def register(payload: RegisterRequest):
        _acquire_limit_sync("api_write")
        user = create_user(email=payload.email, password=payload.password, role="user")
        auth = login_user(email=payload.email, password=payload.password)
        auth.user_id = user.user_id
        return auth

    @app.post("/auth/login", response_model=AuthResponse)
    def login(payload: LoginRequest):
        _acquire_limit_sync("api_write")
        return login_user(email=payload.email, password=payload.password)

    @app.get("/auth/me", response_model=UserPublic)
    def auth_me(authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_read")
        claims = _claims_required(authorization)
        return UserPublic(
            user_id=str(claims.get("uid", "")),
            email=str(claims.get("email", "")),
            role=str(claims.get("role", "user")),
            created_at=db.utc_now(),
        )

    @app.get("/auth/users", response_model=list[UserPublic])
    def auth_users(authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return list_users()

    PUBLIC_UID = "public"

    @app.get("/risk-profile", response_model=RiskProfile)
    def get_profile(authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_read")
        claims = _claims_optional(authorization)
        uid = str(claims.get("uid")) if claims else PUBLIC_UID

        prof = get_risk_profile_for_user(uid)
        if prof:
            return prof
        return ensure_risk_profile_for_user(uid)

    @app.put("/risk-profile", response_model=RiskProfile)
    def put_profile(profile: RiskProfile, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_write")
        claims = _claims_optional(authorization)
        uid = str(claims.get("uid")) if claims else PUBLIC_UID
        return upsert_risk_profile_for_user(uid, profile)

    @app.get("/portfolio", response_model=PortfolioSnapshot)
    async def portfolio_get(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_read")
        claims = _claims_required(authorization)
        uid = str(claims.get("uid", ""))
        role = str(claims.get("role", "user"))

        col = db.require_col(db.portfolio_snapshots_col, "portfolio_snapshots")
        doc = col.find_one({"user_id": uid})
        if doc:
            doc.pop("_id", None)
            doc.pop("user_id", None)
            existing = PortfolioSnapshot(**doc)

            if role == "admin" and _is_legacy_admin_seed(existing):
                regenerated = await _admin_default_portfolio()
                payload = regenerated.model_dump()
                col.update_one(
                    {"user_id": uid},
                    {"$set": {"user_id": uid, **payload, "updated_at": db.utc_now()}},
                    upsert=True,
                )
                return regenerated

            return existing

        if role == "admin":
            generated = await _admin_default_portfolio()
            payload = generated.model_dump()
            col.update_one(
                {"user_id": uid},
                {"$set": {"user_id": uid, **payload, "updated_at": db.utc_now()}},
                upsert=True,
            )
            return generated

        return PortfolioSnapshot(cash=0.0, holdings=[])

    @app.post("/portfolio", response_model=PortfolioSnapshot)
    def portfolio_post(snapshot: PortfolioSnapshot, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_write")
        claims = _claims_required(authorization)
        uid = str(claims.get("uid", ""))

        payload = snapshot.model_dump()
        col = db.require_col(db.portfolio_snapshots_col, "portfolio_snapshots")
        col.update_one(
            {"user_id": uid},
            {"$set": {"user_id": uid, **payload, "updated_at": db.utc_now()}},
            upsert=True,
        )
        return snapshot

    @app.get("/audit-log")
    def audit_log_get(authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_read")
        claims = _claims_required(authorization)
        require_roles(claims, ("premium", "admin", "manager"))
        return _build_ml_audit_entries()

    @app.post("/audit-log/clear")
    def audit_log_clear(authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_write")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))

        try:
            col = db.require_col(db.audit_logs_col, "audit_logs")
            col.delete_many({})
        except Exception:
            pass
        return {"status": "ok"}

    @app.get("/dashboard/public", response_model=DashboardResponse)
    async def dashboard_public():
        await _acquire_limit("api_read")
        # Apply stricter throttle when this route can trigger LLM summary generation.
        if bool(getattr(intelligence_router, "ollama_available", False)) and str(getattr(intelligence_router, "system_capability", "low") or "low").lower() == "high":
            await _acquire_limit("api_llm")
        return await _maybe_await(build_dashboard_for_user, PUBLIC_UID)

    @app.get("/dashboard", response_model=DashboardResponse)
    async def dashboard(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_read")
        # Apply stricter throttle when this route can trigger LLM summary generation.
        if bool(getattr(intelligence_router, "ollama_available", False)) and str(getattr(intelligence_router, "system_capability", "low") or "low").lower() == "high":
            await _acquire_limit("api_llm")
        claims = _claims_optional(authorization)
        uid = str(claims.get("uid")) if claims else PUBLIC_UID
        return await _maybe_await(build_dashboard_for_user, uid)

    @app.get("/signals/today", response_model=SignalsToday)
    async def signals_today_public(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_read")
        # Apply stricter throttle when this route can trigger LLM summary generation.
        if bool(getattr(intelligence_router, "ollama_available", False)) and str(getattr(intelligence_router, "system_capability", "low") or "low").lower() == "high":
            await _acquire_limit("api_llm")
        claims = _claims_optional(authorization)
        uid = str(claims.get("uid")) if claims else PUBLIC_UID
        dash: DashboardResponse = await _maybe_await(build_dashboard_for_user, uid)
        today = dash.date

        up_candidates = list(dash.sp500_up) + list(dash.dow_up)
        dn_candidates = list(dash.sp500_down) + list(dash.dow_down)

        up_candidates.sort(key=lambda x: float(x.score), reverse=True)
        dn_candidates.sort(key=lambda x: float(x.score))

        used: set[str] = set()

        top_up_items: list[SignalItem] = []
        for x in up_candidates:
            sym = (x.symbol or "").upper()
            if not sym or sym in used:
                continue
            used.add(sym)
            top_up_items.append(
                SignalItem(
                    symbol=x.symbol,
                    name=getattr(x, "name", None),
                    direction="up",
                    score=float(x.score),
                    reason=x.reason,
                    link=x.link,
                )
            )
            if len(top_up_items) >= 3:
                break

        top_down_items: list[SignalItem] = []
        for x in dn_candidates:
            sym = (x.symbol or "").upper()
            if not sym or sym in used:
                continue
            used.add(sym)
            top_down_items.append(
                SignalItem(
                    symbol=x.symbol,
                    name=getattr(x, "name", None),
                    direction="down",
                    score=float(x.score),
                    reason=x.reason,
                    link=x.link,
                )
            )
            if len(top_down_items) >= 3:
                break

        news = dash.top_news
        if not news:
            news = await _maybe_await(refresh_top_news_of_day, today)

        fi = get_model_feature_importances(model_id=None)

        return SignalsToday(
            date=today,
            top_up=top_up_items,
            top_down=top_down_items,
            news=news or [],
            universe=[],
            regime=dash.regime,
            system_capability=str(getattr(dash, "system_capability", "low") or "low"),  # type: ignore[arg-type]
            llm_summary_enabled=bool(getattr(dash, "llm_summary_enabled", False)),
            market_summary=getattr(dash, "market_summary", None),
            ml_model_id=fi.get("model_id"),
            ml_winner_name=fi.get("winner_name"),
            ml_feature_labels=list(fi.get("feature_labels") or []),
            ml_feature_importances=[float(x) for x in (fi.get("feature_importances") or [])],
            news_refresh_stats=getattr(dash, "news_refresh_stats", None),
        )

    @app.get("/market/annotated/{symbol}", response_model=AnnotatedHistoryResponse)
    async def annotated(symbol: str, days: int = 240):
        await _acquire_limit("api_read")
        today = iso_date_utc()

        try:
            news_items = await _maybe_await(refresh_top_news_of_day, today)
        except Exception:
            news_items = []

        news_docs = []
        for n in news_items or []:
            try:
                if hasattr(n, "model_dump"):
                    news_docs.append(n.model_dump())
                else:
                    news_docs.append(n.dict())
            except Exception:
                continue

        return await _maybe_await(
            get_annotated_history,
            symbol,
            days=days,
            news_docs=news_docs,
        )

    @app.post("/ml/train", response_model=MLTrainResponse)
    async def ml_train(req: MLTrainingRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_train")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return await _maybe_await(train_models_async, req)

    @app.get("/ml/models", response_model=MLModelListResponse)
    def ml_models_list(nlp_pipeline: str | None = None, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("premium", "admin", "manager"))
        return MLModelListResponse(items=list_models(nlp_pipeline=nlp_pipeline))

    @app.get("/ml/models/tournament-stats", response_model=MLTournamentStatsResponse)
    def ml_models_tournament_stats(model_id: str | None = None, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("premium", "admin", "manager"))
        out = get_tournament_competitor_stats(model_id=model_id)
        return MLTournamentStatsResponse(**out)

    @app.post("/ml/models/select", response_model=MLSelectionResponse)
    def ml_select(req: MLSelectionRequest, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return select_best_models(top_k=req.top_k, nlp_pipeline=req.nlp_pipeline)

    @app.get("/admin/ml/models", response_model=MLModelListResponse)
    def admin_ml_models_by_pipeline(nlp_pipeline: str, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin", "manager"))
        return MLModelListResponse(items=list_models(nlp_pipeline=nlp_pipeline))

    @app.post("/admin/ml/models/select", response_model=MLSelectionResponse)
    def admin_ml_select_by_pipeline(nlp_pipeline: str, top_k: int = 1, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return select_best_models(top_k=max(1, int(top_k)), nlp_pipeline=nlp_pipeline)

    @app.post("/admin/ml/models/migrate-nlp-pipeline")
    def admin_ml_migrate_nlp_pipeline(req: MLNlpPipelineMigrationRequest, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return backfill_model_nlp_pipeline(
            dry_run=bool(req.dry_run),
            limit=int(req.limit),
            rollout_iso=req.rollout_iso,
            default_pipeline=str(req.default_pipeline),
        )

    @app.post("/ml/models/prune", response_model=MLPruneResponse)
    def ml_prune(req: MLPruneRequest, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return prune_underperforming_models(min_f1=req.min_f1, min_accuracy=req.min_accuracy)

    @app.get("/ml/settings", response_model=MLRuntimeSettingsResponse)
    def ml_settings_get(authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        out = get_ml_runtime_settings()
        return MLRuntimeSettingsResponse(**out)

    @app.put("/ml/settings", response_model=MLRuntimeSettingsResponse)
    def ml_settings_put(req: MLRuntimeSettingsUpdateRequest, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        out = update_ml_runtime_settings(req.model_dump(exclude_none=True))
        return MLRuntimeSettingsResponse(**out)

    @app.post("/ml/deploy/neural-network", response_model=MLDeployResponse)
    async def ml_deploy_ann(req: MLDeployRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return await _maybe_await(deploy_neural_network, req.model_id)

    @app.post("/ml/predict", response_model=MLPredictionResponse)
    async def ml_predict(req: MLPredictionRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_predict")
        claims = _claims_required(authorization)
        require_roles(claims, ("premium", "admin", "manager"))
        return await _maybe_await(
            predict_for_basket,
            req.stock_basket,
            lookback_days=req.lookback_days,
            model_id=req.model_id,
        )

    @app.get("/universe/custom")
    def get_custom_universe(authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_read")
        claims = _claims_required(authorization)
        require_roles(claims, ("premium", "admin", "manager"))
        uid = str(claims.get("uid", ""))
        return {"symbols": get_user_custom_universe(uid)}

    @app.put("/universe/custom")
    def put_custom_universe(req: CustomUniverseRequest, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_write")
        claims = _claims_required(authorization)
        require_roles(claims, ("premium", "admin", "manager"))
        uid = str(claims.get("uid", ""))
        return {"symbols": set_user_custom_universe(uid, req.symbols)}

    @app.get("/admin/cache/stats")
    def admin_cache_stats(authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return get_cache_stats()

    @app.post("/admin/cache/prune")
    def admin_cache_prune(max_age_days: int | None = None, dry_run: bool = True, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return prune_stale_cache(max_age_days=max_age_days, dry_run=dry_run)

    @app.post("/admin/cache/clear")
    def admin_cache_clear(scope: str = "stale", max_age_days: int | None = None, dry_run: bool = False, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return clear_cache(scope=scope, max_age_days=max_age_days, dry_run=dry_run)

    @app.get("/admin/intelligence/status")
    async def admin_intelligence_status(refresh: bool = True, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin", "manager"))

        diagnostics: dict[str, Any] = dict(getattr(app.state, "model_router_diagnostics", {}) or {})

        if refresh:
            try:
                diagnostics = await intelligence_router.run_diagnostics()
            except Exception:
                diagnostics = {**diagnostics, "tier": "low", "error": "router_diagnostics_failed"}

        tier = str(diagnostics.get("tier") or getattr(intelligence_router, "system_capability", "low") or "low").strip().lower()
        if tier not in ("low", "medium", "high"):
            tier = "low"

        try:
            sentiment_pipeline = str(intelligence_router.get_sentiment_pipeline() or "lexicon")
        except Exception:
            sentiment_pipeline = "lexicon"

        app.state.model_router = intelligence_router
        app.state.model_router_diagnostics = diagnostics
        app.state.system_capability = tier

        return {
            "status": "ok",
            "checked_at": _utc_iso_z(datetime.now(timezone.utc)),
            "refresh": bool(refresh),
            "system_capability": tier,
            "sentiment_pipeline": sentiment_pipeline,
            "diagnostics": diagnostics,
        }

    @app.post("/admin/scheduler/run-daily")
    def admin_run_daily(authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        run_daily()
        return {"ok": True}

    @app.post("/admin/news/refresh")
    async def admin_news_refresh(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        today = iso_date_utc()
        items = await refresh_news_if_needed(today, symbols=[])
        return {"ok": True, "count": len(items)}

    @app.put("/admin/rate-limit")
    def admin_rate_limit(req: RateLimitUpdateRequest, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        from .core.rate_limit import limiter  # type: ignore

        limiter.configure(req.key, per_minute=req.per_minute)
        return {"ok": True, "key": req.key, "per_minute": req.per_minute}

    @app.get("/admin/users", response_model=UserListResponse)
    def admin_users(authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return UserListResponse(items=list_users())

    @app.get("/admin/users/{user_id}/detail", response_model=AdminUserDetailResponse)
    def admin_user_detail(user_id: str, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))

        users_col = db.require_col(db.users_col, "users")
        risk_col = db.require_col(db.risk_profiles_col, "risk_profiles")
        dash_col = db.require_col(db.dashboard_cache_col, "dashboard_cache")
        signals_col = db.require_col(db.daily_signals_col, "daily_signals")
        audit_col = db.require_col(db.audit_logs_col, "audit_logs")

        oid = _obj_id(user_id)
        if oid is None:
            raise HTTPException(status_code=404, detail="User not found")

        doc = users_col.find_one({"_id": oid})
        if not doc:
            raise HTTPException(status_code=404, detail="User not found")

        risk_doc = risk_col.find_one({"user_id": user_id}) or {}
        custom_universe = get_user_custom_universe(user_id)

        dash_latest = dash_col.find_one({"user_id": user_id}, sort=[("date", -1)]) or {}
        signal_latest = signals_col.find_one({"user_id": user_id}, sort=[("date", -1)]) or {}

        last_dashboard_at = _to_dt(dash_latest.get("cached_at"))
        if last_dashboard_at is None:
            last_dashboard_at = _to_dt(signal_latest.get("cached_at"))

        status_val = str(doc.get("account_status", "active") or "active").lower()
        failed_login_attempts = int(doc.get("failed_login_attempts", 0) or 0)
        locked_until = _to_dt(doc.get("locked_until"))
        now = datetime.now(timezone.utc)
        security_status = "suspended" if status_val == "suspended" else "active"
        if locked_until and locked_until > now and security_status != "suspended":
            security_status = "locked"

        rate_limit_state: Dict[str, Any] = {"currently_limited": False, "buckets": {}}
        try:
            from .core.rate_limit import limiter  # type: ignore

            buckets: Dict[str, Any] = {}
            currently_limited = False
            for key, bucket in getattr(limiter, "_buckets", {}).items():
                tokens = float(getattr(bucket, "tokens", 0.0))
                capacity = int(getattr(bucket, "capacity", 0) or 0)
                buckets[str(key)] = {
                    "tokens": round(tokens, 3),
                    "capacity": capacity,
                    "near_limit": bool(tokens < 1.0),
                }
                if tokens < 1.0:
                    currently_limited = True
            rate_limit_state = {"currently_limited": currently_limited, "buckets": buckets}
        except Exception:
            pass

        error_docs = list(audit_col.find({"user_id": user_id, "level": "error"}).sort([("created_at", -1)]).limit(25))
        error_logs: list[dict[str, Any]] = []
        for e in error_docs:
            error_logs.append(
                {
                    "timestamp": e.get("created_at"),
                    "path": str(e.get("path") or ""),
                    "method": str(e.get("method") or ""),
                    "message": str(e.get("message") or ""),
                }
            )

        return AdminUserDetailResponse(
            user_id=user_id,
            email=str(doc.get("email") or ""),
            role=str(doc.get("role") or "user"),
            created_at=doc.get("created_at") or db.utc_now(),
            security={
                "status": security_status,
                "failed_login_attempts": failed_login_attempts,
                "locked_until": locked_until,
            },
            financial_context={
                "risk_tolerance": str(risk_doc.get("risk_tolerance") or "balanced"),
                "horizon_years": risk_doc.get("horizon_years"),
                "max_drawdown_pct": risk_doc.get("max_drawdown_pct"),
                "constraints": list(risk_doc.get("constraints") or []),
                "custom_universe": custom_universe,
            },
            activity={
                "last_login_at": _to_dt(doc.get("last_login_at")),
                "last_dashboard_at": last_dashboard_at,
            },
            rate_limit=rate_limit_state,
            error_logs=error_logs,
        )

    @app.put("/admin/users/{user_id}/status")
    def admin_user_status(user_id: str, req: AdminUserStatusUpdateRequest, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))

        users_col = db.require_col(db.users_col, "users")
        oid = _obj_id(user_id)
        if oid is None:
            raise HTTPException(status_code=404, detail="User not found")

        now = datetime.now(timezone.utc)
        if req.status == "active":
            update = {"account_status": "active", "failed_login_attempts": 0, "locked_until": None}
        elif req.status == "suspended":
            update = {"account_status": "suspended"}
        else:
            mins = max(1, int(req.lock_minutes))
            update = {"account_status": "active", "locked_until": now + timedelta(minutes=mins)}

        res = users_col.update_one({"_id": oid}, {"$set": update})
        if res.matched_count <= 0:
            raise HTTPException(status_code=404, detail="User not found")

        return {"ok": True, "user_id": user_id, "status": req.status}

    @app.post("/admin/users/{user_id}/reset-state", response_model=AdminUserStateResetResponse)
    def admin_user_reset_state(user_id: str, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))

        signals_col = db.require_col(db.daily_signals_col, "daily_signals")
        dash_col = db.require_col(db.dashboard_cache_col, "dashboard_cache")

        r1 = signals_col.delete_many({"user_id": user_id})
        r2 = dash_col.delete_many({"user_id": user_id})
        return AdminUserStateResetResponse(
            ok=True,
            cleared_daily_signals=int(r1.deleted_count or 0),
            cleared_dashboard_cache=int(r2.deleted_count or 0),
        )

    @app.get("/admin/users/{user_id}/export", response_model=AdminUserExportResponse)
    def admin_user_export(user_id: str, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))

        users_col = db.require_col(db.users_col, "users")
        risk_col = db.require_col(db.risk_profiles_col, "risk_profiles")
        portfolio_col = db.require_col(db.portfolio_snapshots_col, "portfolio_snapshots")
        signals_col = db.require_col(db.daily_signals_col, "daily_signals")
        dash_col = db.require_col(db.dashboard_cache_col, "dashboard_cache")
        audit_col = db.require_col(db.audit_logs_col, "audit_logs")

        oid = _obj_id(user_id)
        if oid is None:
            raise HTTPException(status_code=404, detail="User not found")

        user_doc = users_col.find_one({"_id": oid})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")

        user_export = {
            "user_id": user_id,
            "email": str(user_doc.get("email") or ""),
            "role": str(user_doc.get("role") or "user"),
            "created_at": user_doc.get("created_at"),
            "account_status": str(user_doc.get("account_status") or "active"),
            "failed_login_attempts": int(user_doc.get("failed_login_attempts", 0) or 0),
            "locked_until": _to_dt(user_doc.get("locked_until")),
            "last_login_at": _to_dt(user_doc.get("last_login_at")),
        }

        risk_doc = risk_col.find_one({"user_id": user_id}) or {}
        risk_doc.pop("_id", None)
        portfolio_doc = portfolio_col.find_one({"user_id": user_id}) or {}
        portfolio_doc.pop("_id", None)
        portfolio_doc.pop("user_id", None)

        signals = list(signals_col.find({"user_id": user_id}).sort([("date", -1)]).limit(30))
        for x in signals:
            x.pop("_id", None)
            x.pop("user_id", None)

        dashboards = list(dash_col.find({"user_id": user_id}).sort([("date", -1)]).limit(30))
        for x in dashboards:
            x.pop("_id", None)
            x.pop("user_id", None)

        errs = list(audit_col.find({"user_id": user_id, "level": "error"}).sort([("created_at", -1)]).limit(50))
        for x in errs:
            x.pop("_id", None)

        return AdminUserExportResponse(
            user=user_export,
            risk_profile=risk_doc,
            portfolio=portfolio_doc,
            custom_universe=get_user_custom_universe(user_id),
            daily_signals=signals,
            dashboard_cache=dashboards,
            error_logs=errs,
        )

    @app.delete("/admin/users/{user_id}/purge")
    def admin_user_purge(user_id: str, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))

        users_col = db.require_col(db.users_col, "users")
        risk_col = db.require_col(db.risk_profiles_col, "risk_profiles")
        portfolio_col = db.require_col(db.portfolio_snapshots_col, "portfolio_snapshots")
        signals_col = db.require_col(db.daily_signals_col, "daily_signals")
        dash_col = db.require_col(db.dashboard_cache_col, "dashboard_cache")
        audit_col = db.require_col(db.audit_logs_col, "audit_logs")

        oid = _obj_id(user_id)
        if oid is None:
            raise HTTPException(status_code=404, detail="User not found")

        deleted = {
            "users": int(users_col.delete_one({"_id": oid}).deleted_count or 0),
            "risk_profiles": int(risk_col.delete_many({"user_id": user_id}).deleted_count or 0),
            "portfolio_snapshots": int(portfolio_col.delete_many({"user_id": user_id}).deleted_count or 0),
            "daily_signals": int(signals_col.delete_many({"user_id": user_id}).deleted_count or 0),
            "dashboard_cache": int(dash_col.delete_many({"user_id": user_id}).deleted_count or 0),
            "audit_logs": int(audit_col.delete_many({"user_id": user_id}).deleted_count or 0),
        }

        try:
            db.get_db()["user_universe"].delete_many({"user_id": user_id})
        except Exception:
            pass

        return {"ok": True, "deleted": deleted}

    @app.put("/admin/users/{user_id}/role")
    def admin_update_user_role(user_id: str, req: RoleUpdateRequest, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        u = update_user_role(user_id, req.role)
        if not u:
            raise HTTPException(status_code=404, detail="User not found")
        return u

    @app.delete("/admin/users/{user_id}")
    def admin_delete_user(user_id: str, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        ok = delete_user(user_id)
        if not ok:
            raise HTTPException(status_code=404, detail="User not found")
        return {"ok": True}

    @app.put("/manager/universe")
    def manager_update_universe(req: UniverseUpdateRequest, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("manager", "admin"))
        vals = set_universe_override(req.key, req.symbols)
        return {"key": req.key, "symbols": vals}

    @app.put("/manager/glossary")
    def manager_set_glossary(req: GlossaryTermUpdateRequest, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("manager", "admin"))
        set_glossary_term(req.key, req.value)
        return {"ok": True}

    @app.delete("/manager/glossary/{key}")
    def manager_delete_glossary(key: str, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("manager", "admin"))
        remove_glossary_term(key)
        return {"ok": True}

    @app.put("/manager/policy/constraints", response_model=PolicyConstraintResponse)
    def manager_policy_constraints(req: PolicyConstraintRequest, authorization: str | None = Header(default=None)):
        _acquire_limit_sync("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("manager", "admin"))
        pe = PolicyEngine()
        out = pe.update_global_constraints(add=req.add, remove=req.remove)
        return PolicyConstraintResponse(constraints=out)

    # Chart / Visualization endpoints

    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _build_model_driven_points(
        points: list[dict[str, Any]],
        prediction_series: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert raw OHLC points into an ML-driven synthetic close path using prob_up.
        This keeps chart outcomes tied to the chosen model while preserving timestamps.
        """
        if not points:
            return []

        prob_map: dict[str, float] = {}
        for row in prediction_series or []:
            t = str(row.get("t") or "")
            p = row.get("prob_up")
            if not t or p is None:
                continue
            try:
                prob_map[t] = max(0.0, min(1.0, float(p)))
            except Exception:
                continue

        out: list[dict[str, Any]] = []
        first_c = _safe_float(points[0].get("c"), 1.0)
        pred_close = max(0.01, first_c)
        for i, pt in enumerate(points):
            t = str(pt.get("t") or "")
            raw_c = max(0.01, _safe_float(pt.get("c"), pred_close))
            if i == 0:
                pred_close = raw_c
            else:
                prob = prob_map.get(t)
                if prob is None:
                    # Hold with slight decay to prevent runaway drift when probabilities are sparse.
                    pred_close = max(0.01, pred_close * 0.999)
                else:
                    # +/- 1% expected move scaled by confidence.
                    factor = 1.0 + ((prob - 0.5) * 2.0 * 0.01)
                    pred_close = max(0.01, pred_close * factor)

            scale = (pred_close / raw_c) if raw_c > 0 else 1.0
            row = dict(pt)
            row["c"] = round(pred_close, 6)
            for k in ("o", "h", "l"):
                if k in row and row[k] is not None:
                    row[k] = round(_safe_float(row[k], raw_c) * scale, 6)
            out.append(row)
        return out

    @app.get("/charts/backtest")
    async def charts_backtest(
        symbol: str = "SPY",
        benchmark: str = "SPY",
        days: int = 1825,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.001,
        slippage_bps: float = 2.0,
        model_id: str | None = None,
        save_image: bool = False,
        authorization: str | None = Header(default=None),
    ):
        """
        Model-driven backtest for *symbol* vs *benchmark* over *days*.
        PNG is saved only when save_image=true.
        """
        await _acquire_limit("api_read")
        _claims_required(authorization)

        syms_to_fetch = list({symbol.upper(), benchmark.upper()})
        histories = await get_price_histories(syms_to_fetch, days=days, concurrency=4)

        sym_pts   = (histories.get(symbol.upper()) or {}).get("points") or []
        bench_pts = (histories.get(benchmark.upper()) or {}).get("points") or []

        if not sym_pts:
            raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

        pred_ctx = await _maybe_await(
            predict_series_for_symbol,
            symbol.upper(),
            lookback_days=days,
            model_id=model_id,
        )
        pred_series = list(pred_ctx.get("series") or [])
        ml_pts = _build_model_driven_points(sym_pts, pred_series)

        from .ml.chart_engine import run_backtest_chart
        from .services.ml_workflow import _model_store_dir

        result = await __import__("asyncio").to_thread(
            run_backtest_chart,
            ml_pts or sym_pts,
            symbol.upper(),
            bench_pts or sym_pts,
            benchmark.upper(),
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_bps=slippage_bps,
            base_dir=_model_store_dir(),
            save_image=bool(save_image),
        )

        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        result["ml_model_id"] = str(pred_ctx.get("model_id") or model_id or "selected")
        result["ml_outcome_source"] = "model_driven_price_path"
        result["image_saved"] = bool(save_image)
        return result

    @app.get("/charts/technical/{symbol}")
    async def charts_technical(
        symbol: str,
        days: int = 365,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        model_id: str | None = None,
        save_image: bool = False,
        authorization: str | None = Header(default=None),
    ):
        """RSI and MACD technical indicator chart for *symbol*."""
        await _acquire_limit("api_read")
        _claims_required(authorization)

        histories = await get_price_histories([symbol.upper()], days=days, concurrency=2)
        pts = (histories.get(symbol.upper()) or {}).get("points") or []
        if not pts:
            raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

        pred_ctx = await _maybe_await(
            predict_series_for_symbol,
            symbol.upper(),
            lookback_days=days,
            model_id=model_id,
        )
        pred_series = list(pred_ctx.get("series") or [])
        ml_pts = _build_model_driven_points(pts, pred_series)

        from .ml.chart_engine import run_technical_chart
        from .services.ml_workflow import _model_store_dir

        result = await __import__("asyncio").to_thread(
            run_technical_chart,
            ml_pts or pts,
            symbol.upper(),
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            base_dir=_model_store_dir(),
            save_image=bool(save_image),
        )
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        result["ml_model_id"] = str(pred_ctx.get("model_id") or model_id or "selected")
        result["ml_outcome_source"] = "model_driven_price_path"
        result["image_saved"] = bool(save_image)
        return result

    @app.get("/charts/montecarlo/{symbol}")
    async def charts_montecarlo(
        symbol: str,
        days: int = 730,
        n_simulations: int = 500,
        horizon_days: int = 252,
        model_id: str | None = None,
        save_image: bool = False,
        authorization: str | None = Header(default=None),
    ):
        """Monte Carlo simulation + percentile breakdown for *symbol*."""
        await _acquire_limit("api_read")
        _claims_required(authorization)

        histories = await get_price_histories([symbol.upper()], days=days, concurrency=2)
        pts = (histories.get(symbol.upper()) or {}).get("points") or []
        if not pts:
            raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

        pred_ctx = await _maybe_await(
            predict_series_for_symbol,
            symbol.upper(),
            lookback_days=days,
            model_id=model_id,
        )
        pred_series = list(pred_ctx.get("series") or [])
        ml_pts = _build_model_driven_points(pts, pred_series)

        from .ml.chart_engine import run_montecarlo_chart
        from .services.ml_workflow import _model_store_dir

        result = await __import__("asyncio").to_thread(
            run_montecarlo_chart,
            ml_pts or pts,
            symbol.upper(),
            n_simulations=min(n_simulations, 2000),
            horizon_days=min(horizon_days, 504),
            base_dir=_model_store_dir(),
            save_image=bool(save_image),
        )
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        result["ml_model_id"] = str(pred_ctx.get("model_id") or model_id or "selected")
        result["ml_outcome_source"] = "model_driven_price_path"
        result["image_saved"] = bool(save_image)
        return result

    @app.get("/charts/valuation/{symbol}")
    async def charts_valuation(
        symbol: str,
        days: int = 365,
        bb_period: int = 20,
        model_id: str | None = None,
        save_image: bool = False,
        authorization: str | None = Header(default=None),
    ):
        """Valuation confidence intervals (Bollinger bands + %B) for *symbol*."""
        await _acquire_limit("api_read")
        _claims_required(authorization)

        histories = await get_price_histories([symbol.upper()], days=days, concurrency=2)
        pts = (histories.get(symbol.upper()) or {}).get("points") or []
        if not pts:
            raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

        pred_ctx = await _maybe_await(
            predict_series_for_symbol,
            symbol.upper(),
            lookback_days=days,
            model_id=model_id,
        )
        pred_series = list(pred_ctx.get("series") or [])
        ml_pts = _build_model_driven_points(pts, pred_series)

        from .ml.chart_engine import run_valuation_confidence_chart
        from .services.ml_workflow import _model_store_dir

        result = await __import__("asyncio").to_thread(
            run_valuation_confidence_chart,
            ml_pts or pts,
            symbol.upper(),
            bb_period=bb_period,
            base_dir=_model_store_dir(),
            save_image=bool(save_image),
        )
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        result["ml_model_id"] = str(pred_ctx.get("model_id") or model_id or "selected")
        result["ml_outcome_source"] = "model_driven_price_path"
        result["image_saved"] = bool(save_image)
        return result

    @app.get("/charts/price-vs-predicted/{symbol}")
    async def charts_price_vs_predicted(
        symbol: str,
        days: int = 365,
        model_id: str | None = None,
        save_image: bool = False,
        authorization: str | None = Header(default=None),
    ):
        """Historical price overlaid with ML up-probability predictions for *symbol*."""
        await _acquire_limit("api_read")
        claims = _claims_required(authorization)
        uid = str(claims.get("uid", "") or "")

        histories = await get_price_histories([symbol.upper()], days=days, concurrency=2)
        pts = (histories.get(symbol.upper()) or {}).get("points") or []
        if not pts:
            raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

        # Historical ML predictions for this symbol from selected/requested model
        pred_ctx = await _maybe_await(
            predict_series_for_symbol,
            symbol.upper(),
            lookback_days=days,
            model_id=model_id,
        )
        predictions: list[dict[str, Any]] = [
            {"t": str(r.get("t") or ""), "prob_up": float(r.get("prob_up"))}
            for r in (pred_ctx.get("series") or [])
            if r.get("prob_up") is not None
        ]

        from .ml.chart_engine import run_price_vs_predicted_chart
        from .services.ml_workflow import _model_store_dir

        result = await __import__("asyncio").to_thread(
            run_price_vs_predicted_chart,
            pts,
            symbol.upper(),
            predictions,
            base_dir=_model_store_dir(),
            save_image=bool(save_image),
        )
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        result["ml_model_id"] = str(pred_ctx.get("model_id") or model_id or "selected")
        result["ml_outcome_source"] = "historical_model_probabilities"
        result["image_saved"] = bool(save_image)
        return result

    @app.get("/charts/risk-dashboard")
    async def charts_risk_dashboard(
        model_id: str | None = None,
        save_image: bool = False,
        authorization: str | None = Header(default=None),
    ):
        """Portfolio risk dashboard: VaR, CVaR, correlation matrix, drawdown."""
        await _acquire_limit("api_read")
        claims = _claims_required(authorization)
        uid = str(claims.get("uid", "") or "")

        # Load user portfolio
        try:
            from .services.portfolio import get_portfolio_for_user  # type: ignore
            snapshot = get_portfolio_for_user(uid)
        except Exception:
            snapshot = None

        if snapshot is None:
            # Fall back to admin default portfolio
            try:
                snapshot = await _admin_default_portfolio()
            except Exception:
                raise HTTPException(status_code=422, detail="No portfolio data available")

        holdings = list(getattr(snapshot, "holdings", []) or [])
        if not holdings:
            raise HTTPException(status_code=422, detail="Portfolio is empty")

        syms = list({str(h.symbol).upper() for h in holdings if getattr(h, "symbol", None)})
        histories = await get_price_histories(syms, days=365, concurrency=8)
        price_histories_dict: dict[str, Any] = {
            sym: ((histories.get(sym) or {}).get("points") or []) for sym in syms
        }
        # Reweight exposures by current ML probability so risk reflects chosen-model outcomes.
        prob_map: dict[str, float] = {}
        model_used = model_id or "selected"
        try:
            pred_resp = await _maybe_await(
                predict_for_basket,
                syms,
                lookback_days=min(365, max(120, len(next(iter(price_histories_dict.values()), [])))),
                model_id=model_id,
            )
            model_used = str(getattr(pred_resp, "model_id", "") or model_used)
            for it in list(getattr(pred_resp, "items", []) or []):
                prob_map[str(getattr(it, "symbol", "") or "").upper()] = max(0.0, min(1.0, float(getattr(it, "probability_up", 0.5))))
        except Exception:
            prob_map = {}

        portfolio_dict = {
            "holdings": [
                {
                    "symbol":    str(h.symbol).upper(),
                    "quantity":  float(getattr(h, "quantity", 1) or 1)
                                 * max(0.05, prob_map.get(str(h.symbol).upper(), 0.5)),
                    "avg_price": float(getattr(h, "avg_price", 0) or 0),
                }
                for h in holdings
            ]
        }

        from .ml.chart_engine import run_risk_dashboard_chart
        from .services.ml_workflow import _model_store_dir

        result = await __import__("asyncio").to_thread(
            run_risk_dashboard_chart,
            portfolio_dict,
            price_histories_dict,
            base_dir=_model_store_dir(),
            save_image=bool(save_image),
        )
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        result["ml_model_id"] = model_used
        result["ml_outcome_source"] = "ml_probability_weighted_exposure"
        result["image_saved"] = bool(save_image)
        return result

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload_ = os.getenv("RELOAD", "true").lower() == "true"

    uvicorn.run("app.main:app", host=host, port=port, reload=reload_)
