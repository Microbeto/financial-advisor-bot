from __future__ import annotations

import inspect
import os
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .models import (
    AnnotatedHistoryResponse,
    AuthResponse,
    Holding,
    DashboardResponse,
    GlossaryResponse,
    GlossaryTermUpdateRequest,
    CustomUniverseRequest,
    LoginRequest,
    MLDeployRequest,
    MLDeployResponse,
    MLModelListResponse,
    MLPredictionRequest,
    MLPredictionResponse,
    MLPruneRequest,
    MLPruneResponse,
    MLRuntimeSettingsResponse,
    MLRuntimeSettingsUpdateRequest,
    MLSelectionRequest,
    MLSelectionResponse,
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
    deploy_neural_network,
    ensure_selected_model_exists,
    list_models,
    predict_for_basket,
    prune_underperforming_models,
    get_ml_runtime_settings,
    select_best_models,
    train_models_async,
    update_ml_runtime_settings,
)
from .core.utils import iso_date_utc
from .engine.glossary import lingo_glossary, remove_glossary_term, set_glossary_term


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
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
        bootstrap_admin_if_configured()
    except Exception:
        pass

    yield

    try:
        db.close_db()
    finally:
        app.state.db = None


def create_app() -> FastAPI:
    app = FastAPI(title="Financial Advisor Bot API", lifespan=lifespan)

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

        symbols = [x.symbol for x in top if getattr(x, "symbol", None)]
        histories = await get_price_histories(symbols, days=35, concurrency=8) if symbols else {}

        holdings: list[Holding] = []
        for item in top:
            sym = str(item.symbol).upper()
            pts = (histories.get(sym) or {}).get("points") or []
            last_close = 0.0
            if pts:
                try:
                    last_close = float(pts[-1].get("c", 0.0) or 0.0)
                except Exception:
                    last_close = 0.0
            if last_close <= 0:
                last_close = 100.0
            holdings.append(Holding(symbol=sym, quantity=1.0, avg_price=float(last_close)))

        return PortfolioSnapshot(cash=0.0, holdings=holdings)

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

    @app.get("/health")
    async def health():
        await _acquire_limit("api_health")
        db_ok = True
        try:
            _ = db.get_db()
        except Exception:
            db_ok = False
        return {"ok": True, "db": db_ok}

    @app.get("/glossary", response_model=GlossaryResponse)
    async def glossary():
        await _acquire_limit("api_read")
        return GlossaryResponse(terms=lingo_glossary())

    @app.post("/auth/register", response_model=AuthResponse)
    async def register(payload: RegisterRequest):
        await _acquire_limit("api_write")
        user = create_user(email=payload.email, password=payload.password, role="user")
        auth = login_user(email=payload.email, password=payload.password)
        auth.user_id = user.user_id
        return auth

    @app.post("/auth/login", response_model=AuthResponse)
    async def login(payload: LoginRequest):
        await _acquire_limit("api_write")
        return login_user(email=payload.email, password=payload.password)

    @app.get("/auth/me", response_model=UserPublic)
    async def auth_me(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_read")
        claims = _claims_required(authorization)
        return UserPublic(
            user_id=str(claims.get("uid", "")),
            email=str(claims.get("email", "")),
            role=str(claims.get("role", "user")),
            created_at=db.utc_now(),
        )

    PUBLIC_UID = "public"

    @app.get("/risk-profile", response_model=RiskProfile)
    async def get_profile(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_read")
        claims = _claims_optional(authorization)
        uid = str(claims.get("uid")) if claims else PUBLIC_UID

        prof = get_risk_profile_for_user(uid)
        if prof:
            return prof
        return ensure_risk_profile_for_user(uid)

    @app.put("/risk-profile", response_model=RiskProfile)
    async def put_profile(profile: RiskProfile, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_write")
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
            return PortfolioSnapshot(**doc)

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
    async def portfolio_post(snapshot: PortfolioSnapshot, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_write")
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
    async def audit_log_get(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_read")
        claims = _claims_required(authorization)
        require_roles(claims, ("premium", "admin", "manager"))
        return _build_ml_audit_entries()

    @app.post("/audit-log/clear")
    async def audit_log_clear(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_write")
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
        return await _maybe_await(build_dashboard_for_user, PUBLIC_UID)

    @app.get("/dashboard", response_model=DashboardResponse)
    async def dashboard(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_read")
        claims = _claims_optional(authorization)
        uid = str(claims.get("uid")) if claims else PUBLIC_UID
        return await _maybe_await(build_dashboard_for_user, uid)

    @app.get("/signals/today", response_model=SignalsToday)
    async def signals_today_public(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_read")
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

        return SignalsToday(
            date=today,
            top_up=top_up_items,
            top_down=top_down_items,
            news=news or [],
            universe=[],
            regime=dash.regime,
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
    async def ml_models_list(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("premium", "admin", "manager"))
        return MLModelListResponse(items=list_models())

    @app.post("/ml/models/select", response_model=MLSelectionResponse)
    async def ml_select(req: MLSelectionRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return select_best_models(top_k=req.top_k)

    @app.post("/ml/models/prune", response_model=MLPruneResponse)
    async def ml_prune(req: MLPruneRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return prune_underperforming_models(min_f1=req.min_f1, min_accuracy=req.min_accuracy)

    @app.get("/ml/settings", response_model=MLRuntimeSettingsResponse)
    async def ml_settings_get(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        out = get_ml_runtime_settings()
        return MLRuntimeSettingsResponse(**out)

    @app.put("/ml/settings", response_model=MLRuntimeSettingsResponse)
    async def ml_settings_put(req: MLRuntimeSettingsUpdateRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
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
    async def get_custom_universe(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_read")
        claims = _claims_required(authorization)
        require_roles(claims, ("premium", "admin", "manager"))
        uid = str(claims.get("uid", ""))
        return {"symbols": get_user_custom_universe(uid)}

    @app.put("/universe/custom")
    async def put_custom_universe(req: CustomUniverseRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_write")
        claims = _claims_required(authorization)
        require_roles(claims, ("premium", "admin", "manager"))
        uid = str(claims.get("uid", ""))
        return {"symbols": set_user_custom_universe(uid, req.symbols)}

    @app.get("/admin/cache/stats")
    async def admin_cache_stats(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return get_cache_stats()

    @app.post("/admin/cache/prune")
    async def admin_cache_prune(max_age_days: int | None = None, dry_run: bool = True, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return prune_stale_cache(max_age_days=max_age_days, dry_run=dry_run)

    @app.post("/admin/cache/clear")
    async def admin_cache_clear(scope: str = "stale", max_age_days: int | None = None, dry_run: bool = False, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return clear_cache(scope=scope, max_age_days=max_age_days, dry_run=dry_run)

    @app.post("/admin/scheduler/run-daily")
    async def admin_run_daily(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
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
    async def admin_rate_limit(req: RateLimitUpdateRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        from .core.rate_limit import limiter  # type: ignore

        limiter.configure(req.key, per_minute=req.per_minute)
        return {"ok": True, "key": req.key, "per_minute": req.per_minute}

    @app.get("/admin/users", response_model=UserListResponse)
    async def admin_users(authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        return UserListResponse(items=list_users())

    @app.put("/admin/users/{user_id}/role")
    async def admin_update_user_role(user_id: str, req: RoleUpdateRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        u = update_user_role(user_id, req.role)
        if not u:
            raise HTTPException(status_code=404, detail="User not found")
        return u

    @app.delete("/admin/users/{user_id}")
    async def admin_delete_user(user_id: str, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("admin",))
        ok = delete_user(user_id)
        if not ok:
            raise HTTPException(status_code=404, detail="User not found")
        return {"ok": True}

    @app.put("/manager/universe")
    async def manager_update_universe(req: UniverseUpdateRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("manager", "admin"))
        vals = set_universe_override(req.key, req.symbols)
        return {"key": req.key, "symbols": vals}

    @app.put("/manager/glossary")
    async def manager_set_glossary(req: GlossaryTermUpdateRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("manager", "admin"))
        set_glossary_term(req.key, req.value)
        return {"ok": True}

    @app.delete("/manager/glossary/{key}")
    async def manager_delete_glossary(key: str, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("manager", "admin"))
        remove_glossary_term(key)
        return {"ok": True}

    @app.put("/manager/policy/constraints", response_model=PolicyConstraintResponse)
    async def manager_policy_constraints(req: PolicyConstraintRequest, authorization: str | None = Header(default=None)):
        await _acquire_limit("api_ml_admin")
        claims = _claims_required(authorization)
        require_roles(claims, ("manager", "admin"))
        pe = PolicyEngine()
        out = pe.update_global_constraints(add=req.add, remove=req.remove)
        return PolicyConstraintResponse(constraints=out)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload_ = os.getenv("RELOAD", "true").lower() == "true"

    uvicorn.run("app.main:app", host=host, port=port, reload=reload_)
