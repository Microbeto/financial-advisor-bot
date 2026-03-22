# Structure: ML workflow orchestration for training, evaluation, tournament selection, and deployment.
from __future__ import annotations

import asyncio
import importlib
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .. import db
from ..ml.model_router import intelligence_router
from ..ml.registry import get_meta_competitor_factories
from ..ml.meta_interface import scalar_allocation
from ..ml.tournament import run_walk_forward_tournament
from ..core.utils import iso_date_utc
from ..models import (
    MLDeployResponse,
    MLModelInfo,
    MLModelMetric,
    MLPredictionItem,
    MLPredictionResponse,
    MLPruneResponse,
    MLSelectionResponse,
    MLTrainResponse,
    MLTrainingRequest,
)
from .market import get_price_histories
from .news import get_news_for_date, refresh_news_if_needed
from .symbols import resolve_symbol_name

try:
    import joblib
except Exception:
    joblib = None


ML_MODEL_MAX_COUNT = int(os.getenv("ML_MODEL_MAX_COUNT", "10"))
ML_SELECTED_TOP_K = int(os.getenv("ML_SELECTED_TOP_K", "3"))
WALK_FORWARD_SPLITS = int(os.getenv("ML_WALK_FORWARD_SPLITS", "5"))
WALK_FORWARD_MIN_TRAIN = int(os.getenv("ML_WALK_FORWARD_MIN_TRAIN", "120"))
WALK_FORWARD_PURGE_DAYS = int(os.getenv("ML_WALK_FORWARD_PURGE_DAYS", "0"))

TRIPLE_BARRIER_TP = float(os.getenv("ML_TP_BARRIER", "0.05"))
TRIPLE_BARRIER_SL = float(os.getenv("ML_SL_BARRIER", "0.03"))
TRIPLE_BARRIER_HOLD_DAYS = int(os.getenv("ML_TIME_BARRIER_DAYS", "20"))

FINBERT_MODEL_NAME = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
FINBERT_BATCH_SIZE = int(os.getenv("FINBERT_BATCH_SIZE", "16"))
FINBERT_RETRY_SECONDS = int(os.getenv("FINBERT_RETRY_SECONDS", "300"))
ML_ENFORCE_NEWS_COVERAGE = os.getenv("ML_ENFORCE_NEWS_COVERAGE", "1").strip().lower() in ("1", "true", "yes", "on")
ML_MAX_NEWS_MISSING_RATIO = float(os.getenv("ML_MAX_NEWS_MISSING_RATIO", "0.10"))
ML_NLP_PIPELINE = os.getenv("ML_NLP_PIPELINE", "auto").strip().lower()  # auto|lexicon|finbert|cascade
ML_CASCADE_AMBIGUOUS_ABS = float(os.getenv("ML_CASCADE_AMBIGUOUS_ABS", "0.12"))
ML_SCORING_RISK_FREE_RATE = float(os.getenv("ML_SCORING_RISK_FREE_RATE", "0.04"))


# _finbert_candidates service logic.
def _finbert_candidates() -> List[str]:
    raw = os.getenv("FINBERT_MODEL_CANDIDATES", "").strip()
    env_vals = [x.strip() for x in raw.split(",") if x.strip()] if raw else []

    defaults = [
        FINBERT_MODEL_NAME,
        "ProsusAI/finbert",
        "yiyanghkust/finbert-tone",
    ]

    out: List[str] = []
    for name in env_vals + defaults:
        if name and name not in out:
            out.append(name)
    return out

_RUNTIME_SETTINGS_DEFAULTS: Dict[str, Any] = {
    "tp_barrier": float(TRIPLE_BARRIER_TP),
    "sl_barrier": float(TRIPLE_BARRIER_SL),
    "time_barrier_days": int(TRIPLE_BARRIER_HOLD_DAYS),
    "walk_forward_splits": int(WALK_FORWARD_SPLITS),
    "walk_forward_min_train": int(WALK_FORWARD_MIN_TRAIN),
    "walk_forward_purge_days": int(WALK_FORWARD_PURGE_DAYS),
}


# Block: defines the DataValidationError class and its related behavior.
class DataValidationError(RuntimeError):
    def __init__(self, message: str, *, code: str = "data_validation_failed", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}

MACRO_TERMS: Tuple[str, ...] = (
    "rate cuts",
    "inflation",
    "recession",
    "volatility",
    "risk-on",
    "risk-off",
    "yield",
    "federal reserve",
)

BASE_FEATURE_NAMES: Tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "momentum_blend",
    "symbol_sentiment",
    "market_sentiment",
    "macro_rate_cuts",
    "macro_inflation",
    "macro_recession",
    "macro_volatility",
    "macro_risk_on",
    "macro_risk_off",
    "macro_yield",
    "macro_federal_reserve",
)


# _ml_log service logic.
def _ml_log(message: str) -> None:
    enabled = os.getenv("ML_PIPELINE_VERBOSE", "1").strip().lower() not in ("0", "false", "no", "off")
    if not enabled:
        return
    ts = _utc_now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ml][{ts}] {message}", flush=True)


# _alert_admin_ml_issue service logic.
def _alert_admin_ml_issue(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    payload = details or {}
    _ml_log(f"ADMIN ALERT: {message} | {payload}")
    try:
        col = db.require_col(db.audit_logs_col, "audit_logs")
        col.insert_one(
            {
                "user_id": "system",
                "level": "error",
                "path": "/ml/train",
                "method": "POST",
                "message": str(message)[:1000],
                "context": payload,
                "created_at": _utc_now(),
            }
        )
    except Exception:
        return


# _load_xgboost_classifier service logic.
def _load_xgboost_classifier() -> Any:
    try:
        mod = importlib.import_module("xgboost")
        return getattr(mod, "XGBClassifier", None)
    except Exception:
        return None


# _load_lightgbm_classifier service logic.
def _load_lightgbm_classifier() -> Any:
    try:
        mod = importlib.import_module("lightgbm")
        return getattr(mod, "LGBMClassifier", None)
    except Exception:
        return None


# _utc_now service logic.
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# _to_iso service logic.
def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# _iso_yesterday_utc service logic.
def _iso_yesterday_utc() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()


# _model_store_dir service logic.
def _model_store_dir() -> Path:
    root = os.getenv("MODEL_STORE_DIR", str(Path(__file__).resolve().parents[2] / "model_store"))
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


# _slug service logic.
def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in (s or "")).strip("_")


# _sentiment_score_text service logic.
def _sentiment_score_text(text: str) -> float:
    pos = [
        "beat",
        "beats",
        "upgrade",
        "growth",
        "strong",
        "rally",
        "outperform",
        "record",
        "profit",
        "bullish",
    ]
    neg = [
        "miss",
        "downgrade",
        "decline",
        "weak",
        "lawsuit",
        "selloff",
        "layoff",
        "slump",
        "bearish",
        "risk",
    ]
    t = (text or "").lower()
    score = 0.0
    for w in pos:
        if w in t:
            score += 1.0
    for w in neg:
        if w in t:
            score -= 1.0
    if score == 0.0:
        return 0.0
    return float(score / max(len(pos), len(neg)))


# _volatility service logic.
def _volatility(values: List[float], lookback: int = 20) -> float:
    if len(values) < lookback + 1:
        return 0.0
    arr = np.array(values[-(lookback + 1) :], dtype=float)
    prev = arr[:-1]
    nxt = arr[1:]
    rets = np.where(prev > 0, (nxt / prev) - 1.0, 0.0)
    if rets.size < 2:
        return 0.0
    return float(np.std(rets, ddof=1))


# _technical_features service logic.
def _technical_features(closes: List[float]) -> List[float]:
    if len(closes) < 25:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    c1 = closes[-1]
    r1 = (c1 / closes[-2] - 1.0) if closes[-2] > 0 else 0.0
    r5 = (c1 / closes[-6] - 1.0) if len(closes) >= 6 and closes[-6] > 0 else 0.0
    r20 = (c1 / closes[-21] - 1.0) if len(closes) >= 21 and closes[-21] > 0 else 0.0
    vol20 = _volatility(closes, lookback=20)
    momentum = (0.6 * r20) + (0.3 * r5) + (0.1 * r1)
    return [float(r1), float(r5), float(r20), float(vol20), float(momentum)]


# _point_date service logic.
def _point_date(p: Dict[str, Any], fallback: str) -> str:
    t_raw = p.get("t")
    if t_raw is None:
        return fallback
    try:
        t_num = float(t_raw)
        if t_num > 1e12:
            dt = datetime.fromtimestamp(t_num / 1000.0, tz=timezone.utc)
        elif t_num > 1e9:
            dt = datetime.fromtimestamp(t_num, tz=timezone.utc)
        else:
            return fallback
        return dt.date().isoformat()
    except Exception:
        return fallback


# Block: defines the _FinBertInferencer class and its related behavior.
class _FinBertInferencer:
    def __init__(self) -> None:
        self._ready = False
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._active_model_name = ""
        self._last_init_attempt_ts = 0.0
        self._last_error = ""
        self._init()

    def _init(self) -> None:
        self._last_init_attempt_ts = time.time()
        try:
            torch_mod = importlib.import_module("torch")
            tr_mod = importlib.import_module("transformers")
        except Exception as exc:
            self._ready = False
            self._tokenizer = None
            self._model = None
            self._torch = None
            self._last_error = f"import_failed:{exc}"
            _ml_log("FinBERT import failed, will use lexicon fallback unless imports become available")
            return

        cache_dir = os.getenv("FINBERT_CACHE_DIR", "").strip() or None
        candidates = _finbert_candidates()
        errors: List[str] = []

        for model_name in candidates:
            local_only_modes = (True,)
            for local_only in local_only_modes:
                mode = "cache-only" if local_only else "download"
                try:
                    kwargs: Dict[str, Any] = {"local_files_only": local_only}
                    if cache_dir:
                        kwargs["cache_dir"] = cache_dir

                    tokenizer = tr_mod.AutoTokenizer.from_pretrained(model_name, **kwargs)
                    model = tr_mod.AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
                    model.eval()

                    # GPU is optional; CPU path is valid and should not trigger fallback.
                    if bool(getattr(torch_mod, "cuda", None)) and torch_mod.cuda.is_available():
                        try:
                            model = model.to("cuda")
                        except Exception:
                            pass

                    self._torch = torch_mod
                    self._tokenizer = tokenizer
                    self._model = model
                    self._ready = True
                    self._active_model_name = str(model_name)
                    self._last_error = ""
                    _ml_log(f"FinBERT loaded: {model_name} ({mode})")
                    return
                except Exception as exc:
                    errors.append(f"{model_name}[{mode}]={exc}")

        # All attempts exhausted: lexicon is the final fallback.
            self._ready = False
            self._tokenizer = None
            self._model = None
            self._torch = None
        self._active_model_name = ""
        self._last_error = " | ".join(errors[:4]) if errors else "unknown"
        _ml_log(f"FinBERT unavailable after all attempts, using lexicon fallback sentiment | {self._last_error}")

    def score_texts(self, texts: Sequence[str]) -> List[float]:
        cleaned = [str(t or "").strip() for t in texts]
        if not cleaned:
            return []

        # Retry FinBERT initialization periodically so temporary startup/network failures
        # do not permanently lock the process into lexicon mode.
        if not self._ready and (time.time() - float(self._last_init_attempt_ts)) >= max(10, FINBERT_RETRY_SECONDS):
            self._init()

        if not self._ready or self._tokenizer is None or self._model is None or self._torch is None:
            return [_sentiment_score_text(t) for t in cleaned]

        out: List[float] = []
        torch_mod = self._torch
        for i in range(0, len(cleaned), max(1, FINBERT_BATCH_SIZE)):
            batch = cleaned[i : i + max(1, FINBERT_BATCH_SIZE)]
            try:
                enc = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
                with torch_mod.no_grad():
                    logits = self._model(**enc).logits
                    probs = torch_mod.softmax(logits, dim=1).cpu().numpy()
                for row in probs:
                    if row.shape[0] >= 3:
                        # FinBERT labels are [negative, neutral, positive]
                        score = float(row[2] - row[0])
                    else:
                        score = 0.0
                    out.append(max(-1.0, min(1.0, score)))
            except Exception:
                out.extend([_sentiment_score_text(t) for t in batch])
        return out


_FINBERT: Optional[_FinBertInferencer] = None


# _get_finbert service logic.
def _get_finbert() -> _FinBertInferencer:
    global _FINBERT
    if _FINBERT is None:
        _FINBERT = _FinBertInferencer()
    return _FINBERT


# Block: defines the _DailyNewsFeatures class and its related behavior.
@dataclass
class _DailyNewsFeatures:
    symbol_sentiment: Dict[str, float]
    market_sentiment: float
    macro_features: List[float]
    news_item_count: int = 0
    nlp_pipeline: str = "lexicon"


# _normalize_nlp_pipeline service logic.
def _normalize_nlp_pipeline(pipeline: Optional[str]) -> str:
    v = str(pipeline or "").strip().lower()
    if v in ("lexicon", "finbert", "cascade"):
        return v
    return "lexicon"


# _active_nlp_pipeline service logic.
def _active_nlp_pipeline() -> str:
    forced = _normalize_nlp_pipeline(ML_NLP_PIPELINE)
    if forced != "lexicon" or str(ML_NLP_PIPELINE).strip().lower() == "lexicon":
        return forced

    # auto mode: defer to runtime diagnostics.
    try:
        routed = _normalize_nlp_pipeline(str(intelligence_router.get_sentiment_pipeline() or "lexicon"))
        if routed == "lexicon":
            # In auto mode prefer cascade over pure lexicon so ambiguous texts can
            # still leverage FinBERT when available while keeping lexicon fallback.
            return "cascade"
        return routed
    except Exception:
        return "cascade"


# _score_texts_with_pipeline service logic.
def _score_texts_with_pipeline(texts: Sequence[str], pipeline: str) -> List[float]:
    cleaned = [str(t or "").strip() for t in texts if str(t or "").strip()]
    if not cleaned:
        return []

    mode = _normalize_nlp_pipeline(pipeline)
    if mode == "lexicon":
        return [_sentiment_score_text(t) for t in cleaned]

    if mode == "finbert":
        return _get_finbert().score_texts(cleaned)

    # Cascade mode: cheap lexicon first, escalate only ambiguous rows to FinBERT.
    lex = np.asarray([_sentiment_score_text(t) for t in cleaned], dtype=float)
    out = np.asarray(lex, dtype=float)
    amb_mask = np.abs(lex) < float(max(0.0, ML_CASCADE_AMBIGUOUS_ABS))
    if bool(np.any(amb_mask)):
        amb_idx = np.where(amb_mask)[0]
        amb_texts = [cleaned[int(i)] for i in amb_idx.tolist()]
        fb = _get_finbert().score_texts(amb_texts)
        for i, score in zip(amb_idx.tolist(), fb):
            out[int(i)] = float(score)
    return out.tolist()


# _macro_term_features service logic.
def _macro_term_features(texts: Sequence[str]) -> List[float]:
    merged = " ".join([str(t or "").lower() for t in texts])
    denom = float(max(1, len(texts)))
    vals: List[float] = []
    for term in MACRO_TERMS:
        vals.append(float(merged.count(term.lower())) / denom)
    return vals


# _news_text service logic.
def _news_text(item: Any) -> str:
    title = str(getattr(item, "title", "") or "")
    summary = str(getattr(item, "summary", "") or "")
    return f"{title}. {summary}".strip()


# _daily_news_features service logic.
def _daily_news_features(date: str, symbols: Sequence[str], nlp_pipeline: Optional[str] = None) -> _DailyNewsFeatures:
    syms = list(dict.fromkeys([str(s).upper().strip() for s in symbols if str(s).strip()]))
    active_pipeline = _normalize_nlp_pipeline(nlp_pipeline or _active_nlp_pipeline())
    items = get_news_for_date(date, symbols=syms, limit=350, include_market=True)

    symbol_texts: Dict[str, List[str]] = {s: [] for s in syms}
    market_texts: List[str] = []
    all_texts: List[str] = []

    for it in items:
        txt = _news_text(it)
        if not txt:
            continue
        all_texts.append(txt)
        sym = str(getattr(it, "symbol", "") or "").upper().strip()
        if sym and sym in symbol_texts:
            symbol_texts[sym].append(txt)
        else:
            market_texts.append(txt)

    market_scores = _score_texts_with_pipeline(market_texts or all_texts, active_pipeline)
    market_sent = float(np.mean(market_scores)) if market_scores else 0.0

    symbol_sent: Dict[str, float] = {}
    for sym in syms:
        s_scores = _score_texts_with_pipeline(symbol_texts.get(sym) or [], active_pipeline)
        if s_scores:
            symbol_sent[sym] = float(np.mean(s_scores))
        else:
            symbol_sent[sym] = market_sent

    macro = _macro_term_features(all_texts)
    return _DailyNewsFeatures(
        symbol_sentiment=symbol_sent,
        market_sentiment=market_sent,
        macro_features=macro,
        news_item_count=int(len(all_texts)),
        nlp_pipeline=active_pipeline,
    )


def _daily_news_features_compat(
    date: str,
    symbols: Sequence[str],
    nlp_pipeline: Optional[str] = None,
) -> _DailyNewsFeatures:
    # Some tests monkeypatch _daily_news_features with the legacy 2-arg signature.
    try:
        return _daily_news_features(date, symbols, nlp_pipeline=nlp_pipeline)
    except TypeError:
        return _daily_news_features(date, symbols)


# _validate_news_coverage_or_raise service logic.
def _validate_news_coverage_or_raise(daily_ctx: Dict[str, _DailyNewsFeatures], symbols: Sequence[str]) -> Dict[str, Any]:
    total_days = int(len(daily_ctx))
    missing_days = int(sum(1 for ctx in daily_ctx.values() if int(getattr(ctx, "news_item_count", 0)) <= 0))
    missing_ratio = float(missing_days / total_days) if total_days > 0 else 1.0

    report = {
        "window_days": total_days,
        "missing_days": missing_days,
        "missing_ratio": missing_ratio,
        "max_missing_ratio": float(ML_MAX_NEWS_MISSING_RATIO),
        "symbol_count": int(len(list(symbols))),
    }

    if not ML_ENFORCE_NEWS_COVERAGE:
        return report

    if total_days <= 0:
        raise DataValidationError(
            "Training aborted: no candidate training days found for news coverage validation.",
            code="news_coverage_empty_window",
            details=report,
        )

    if missing_ratio > float(ML_MAX_NEWS_MISSING_RATIO):
        raise DataValidationError(
            (
                "Training aborted: news coverage gate failed "
                f"(missing {missing_days}/{total_days} days = {missing_ratio:.1%}, "
                f"threshold {float(ML_MAX_NEWS_MISSING_RATIO):.1%})."
            ),
            code="news_coverage_threshold_exceeded",
            details=report,
        )

    return report


# _rolling_daily_volatility service logic.
def _rolling_daily_volatility(closes: Sequence[float], idx: int, lookback: int = 20) -> float:
    if idx <= 1:
        return 0.0

    start = max(1, int(idx) - max(2, int(lookback)) + 1)
    rets: List[float] = []
    for i in range(start, idx + 1):
        p0 = float(closes[i - 1])
        p1 = float(closes[i])
        if p0 > 0.0:
            rets.append((p1 / p0) - 1.0)

    if len(rets) < 2:
        return 0.0
    return float(np.std(np.asarray(rets, dtype=float), ddof=1))


# _triple_barrier_label service logic.
def _triple_barrier_label(closes: Sequence[float], idx: int, horizon: int, tp: float, sl: float) -> int:
    if idx < 0 or idx >= len(closes) - 1:
        return 0
    entry = float(closes[idx])
    if entry <= 0:
        return 0

    # Use TP/SL values as volatility multipliers so barrier distance adapts per asset risk.
    sigma_i = max(1e-6, _rolling_daily_volatility(closes, idx, lookback=20))
    tp_abs = float(max(0.0, float(tp)) * sigma_i)
    sl_abs = float(max(0.0, abs(float(sl))) * sigma_i)

    end = min(len(closes) - 1, idx + max(1, horizon))
    for j in range(idx + 1, end + 1):
        ret = (float(closes[j]) / entry) - 1.0
        if ret >= tp_abs:
            return 1
        if ret <= -sl_abs:
            return 0

    # Time barrier reached without TP/SL hit: fallback to sign of terminal return.
    terminal_ret = (float(closes[end]) / entry) - 1.0
    return 1 if terminal_ret >= 0 else 0


# build_training_matrices service logic.
async def build_training_matrices(
    stock_basket: List[str],
    lookback_days: int,
    label_horizon_days: int,
    nlp_pipeline: Optional[str] = None,
) -> Dict[str, Any]:
    symbols = list(dict.fromkeys([(s or "").upper().strip() for s in stock_basket if (s or "").strip()]))
    active_pipeline = _normalize_nlp_pipeline(nlp_pipeline or _active_nlp_pipeline())
    if not symbols:
        return {
            "X_numeric": np.zeros((0, 15), dtype=float),
            "y": np.array([], dtype=int),
            "actual_returns": np.array([], dtype=float),
            "symbols": [],
            "sample_dates": [],
            "nlp_pipeline": active_pipeline,
            "news_coverage": {
                "window_days": 0,
                "missing_days": 0,
                "missing_ratio": 0.0,
                "max_missing_ratio": float(ML_MAX_NEWS_MISSING_RATIO),
                "symbol_count": 0,
            },
        }

    histories = await get_price_histories(symbols, days=max(60, int(lookback_days)), concurrency=8)
    default_date = _iso_yesterday_utc()

    numeric_rows: List[List[float]] = []
    labels: List[int] = []
    realized_returns: List[float] = []
    sample_dates: List[str] = []

    candidate_dates: set[str] = set()
    per_symbol_points: Dict[str, List[Dict[str, Any]]] = {}
    for sym in symbols:
        pts = list((histories.get(sym) or {}).get("points") or [])
        # Exclude latest point so training stays historical-only.
        if len(pts) > 1:
            pts = pts[:-1]
        per_symbol_points[sym] = pts
        for p in pts:
            candidate_dates.add(_point_date(cast(Dict[str, Any], p), default_date))

    daily_ctx: Dict[str, _DailyNewsFeatures] = {}
    for d in sorted(candidate_dates):
        # Refresh and persist daily news before extracting features for this date.
        await refresh_news_if_needed(d, symbols)
        daily_ctx[d] = _daily_news_features_compat(d, symbols, nlp_pipeline=active_pipeline)

    news_coverage = _validate_news_coverage_or_raise(daily_ctx, symbols)

    runtime = get_ml_runtime_settings()
    horizon = max(1, int(label_horizon_days))
    tb_horizon = max(horizon, int(runtime["time_barrier_days"]))

    for sym in symbols:
        pts = per_symbol_points.get(sym) or []
        closes: List[float] = []
        pdates: List[str] = []
        for p in pts:
            try:
                closes.append(float(cast(Dict[str, Any], p).get("c", 0.0)))
                pdates.append(_point_date(cast(Dict[str, Any], p), default_date))
            except Exception:
                continue

        if len(closes) < 35 + tb_horizon:
            continue

        for i in range(25, len(closes) - tb_horizon):
            seq = closes[: i + 1]
            tech = _technical_features(seq)

            dkey = pdates[i] if i < len(pdates) else default_date
            day_ctx = daily_ctx.get(dkey) or _DailyNewsFeatures(symbol_sentiment={}, market_sentiment=0.0, macro_features=[0.0] * len(MACRO_TERMS))
            sym_sent = float(day_ctx.symbol_sentiment.get(sym, day_ctx.market_sentiment))
            mkt_sent = float(day_ctx.market_sentiment)
            macro = list(day_ctx.macro_features)

            feats = tech + [sym_sent, mkt_sent] + macro
            y = _triple_barrier_label(
                closes,
                i,
                tb_horizon,
                tp=float(runtime["tp_barrier"]),
                sl=float(runtime["sl_barrier"]),
            )
            terminal_idx = min(len(closes) - 1, i + tb_horizon)
            entry = float(closes[i]) if i < len(closes) else 0.0
            if entry > 0.0:
                realized_ret = (float(closes[terminal_idx]) / entry) - 1.0
            else:
                realized_ret = 0.0

            numeric_rows.append(feats)
            labels.append(int(y))
            realized_returns.append(float(realized_ret))
            sample_dates.append(dkey)

    if not numeric_rows:
        return {
            "X_numeric": np.zeros((0, 15), dtype=float),
            "y": np.array([], dtype=int),
            "actual_returns": np.array([], dtype=float),
            "symbols": symbols,
            "sample_dates": [],
            "nlp_pipeline": active_pipeline,
            "news_coverage": news_coverage,
        }

    return {
        "X_numeric": np.array(numeric_rows, dtype=float),
        "y": np.array(labels, dtype=int),
        "actual_returns": np.array(realized_returns, dtype=float),
        "symbols": symbols,
        "sample_dates": sample_dates,
        "nlp_pipeline": active_pipeline,
        "news_coverage": news_coverage,
    }


# _evaluate_binary service logic.
def _evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": 0.0,
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.0
    return metrics


# _weighted_score service logic.
def _weighted_score(metrics: Dict[str, float]) -> float:
    directional_score = float(
        (0.40 * metrics.get("f1", 0.0))
        + (0.30 * metrics.get("accuracy", 0.0))
        + (0.20 * metrics.get("precision", 0.0))
        + (0.10 * metrics.get("recall", 0.0))
    )

    # Institutional risk-adjusted primary block.
    # Higher Sharpe and smaller absolute drawdown should strongly dominate ranking.
    raw_sharpe = metrics.get("annualized_sharpe")
    raw_max_dd = metrics.get("max_drawdown")
    has_risk_metrics = False

    sharpe_fit = 0.0
    drawdown_fit = 0.0
    if raw_sharpe is not None and raw_max_dd is not None:
        try:
            sharpe = float(raw_sharpe)
            max_dd = float(raw_max_dd)
            # Map Sharpe to [0, 1] with diminishing returns for very large values.
            sharpe_fit = float(0.5 * (1.0 + np.tanh(sharpe / 2.0)))
            # Max drawdown is expected <= 0; use absolute loss magnitude.
            drawdown_fit = float(1.0 / (1.0 + abs(max_dd)))
            has_risk_metrics = True
        except Exception:
            has_risk_metrics = False

    # Secondary block: future price closeness factor.
    raw_price_mae = metrics.get("future_price_mae_pct")
    price_fit = 0.0
    has_price_metric = False
    if raw_price_mae is not None:
        try:
            price_mae = max(0.0, float(raw_price_mae))
            # Lower MAE -> higher score, bounded to (0, 1].
            price_fit = float(1.0 / (1.0 + price_mae))
            has_price_metric = True
        except Exception:
            has_price_metric = False

    if has_risk_metrics:
        if has_price_metric:
            return float(
                (0.45 * sharpe_fit)
                + (0.35 * drawdown_fit)
                + (0.15 * price_fit)
                + (0.05 * directional_score)
            )
        return float((0.55 * sharpe_fit) + (0.40 * drawdown_fit) + (0.05 * directional_score))

    if has_price_metric:
        return float((0.75 * price_fit) + (0.25 * directional_score))

    # Legacy fallback when only directional metrics are available.
    return directional_score


def _annualized_sharpe_ratio_from_returns(returns: np.ndarray, annual_risk_free_rate: float = ML_SCORING_RISK_FREE_RATE) -> float:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return 0.0
    daily_rf = float(annual_risk_free_rate / 252.0)
    excess_mean = float(np.mean(arr) - daily_rf)
    return float((excess_mean / std) * np.sqrt(252.0))


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + arr)
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0.0, (equity / peaks) - 1.0, 0.0)
    return float(np.min(drawdowns))


def _strategy_daily_returns_from_probs(probs: np.ndarray, realized_returns: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float).reshape(-1)
    r = np.asarray(realized_returns, dtype=float).reshape(-1)
    n = min(p.size, r.size)
    if n <= 0:
        return np.asarray([], dtype=float)
    alloc = np.clip((2.0 * p[:n]) - 1.0, -1.0, 1.0)
    return np.asarray(alloc * r[:n], dtype=float)


def _class_conditional_return_anchors(y_train: np.ndarray, returns_train: np.ndarray) -> Tuple[float, float]:
    y_arr = np.asarray(y_train, dtype=int).reshape(-1)
    r_arr = np.asarray(returns_train, dtype=float).reshape(-1)
    if y_arr.size == 0 or r_arr.size == 0 or y_arr.size != r_arr.size:
        return 0.0, 0.0

    up_mask = y_arr == 1
    dn_mask = y_arr == 0

    overall = float(np.mean(r_arr)) if r_arr.size else 0.0
    up_mean = float(np.mean(r_arr[up_mask])) if np.any(up_mask) else max(0.0, overall)
    dn_mean = float(np.mean(r_arr[dn_mask])) if np.any(dn_mask) else min(0.0, overall)

    if up_mean <= dn_mean:
        spread = float(np.std(r_arr, ddof=1)) if r_arr.size > 1 else 0.0
        spread = max(1e-6, spread)
        up_mean = float(overall + (0.5 * spread))
        dn_mean = float(overall - (0.5 * spread))

    return dn_mean, up_mean


# _data_completeness_from_news_coverage service logic.
def _data_completeness_from_news_coverage(news_coverage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    nc = dict(news_coverage or {})
    window_days = int(nc.get("window_days") or 0)
    missing_days = int(nc.get("missing_days") or 0)

    ratio_raw = nc.get("missing_ratio")
    if ratio_raw is None:
        missing_ratio = float(missing_days / window_days) if window_days > 0 else 1.0
    else:
        try:
            missing_ratio = float(ratio_raw)
        except Exception:
            missing_ratio = float(missing_days / window_days) if window_days > 0 else 1.0

    missing_ratio = max(0.0, min(1.0, float(missing_ratio)))
    coverage_ratio = max(0.0, min(1.0, 1.0 - missing_ratio))

    return {
        "news_coverage_ratio": float(coverage_ratio),
        "news_missing_ratio": float(missing_ratio),
        "news_window_days": int(window_days),
        "news_missing_days": int(missing_days),
    }


# _predict_proba_or_hard service logic.
def _predict_proba_or_hard(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x)
            if probs.ndim == 2 and probs.shape[1] > 1:
                return np.asarray(probs[:, 1], dtype=float)
        except Exception:
            pass
    preds = np.asarray(model.predict(x), dtype=int)
    return np.where(preds > 0, 1.0, 0.0)


# _family_from_algorithm service logic.
def _family_from_algorithm(algorithm: str) -> str:
    algo = (algorithm or "").lower()
    if "tournament" in algo:
        return "ensemble"
    if "stack" in algo or "meta" in algo:
        return "ensemble"
    if "ann" in algo or "mlp" in algo:
        return "neural_network"
    if "svm" in algo:
        return "svm"
    if "forest" in algo or "tree" in algo:
        return "tree_ensemble"
    if "boost" in algo or "xgboost" in algo or "lightgbm" in algo:
        return "boosting"
    return "other"


# _rank_models service logic.
def _rank_models(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for item in items:
        metrics = item.get("metrics") or {}
        if isinstance(metrics, dict) and metrics:
            item["score"] = _weighted_score(metrics)

    ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    for i, item in enumerate(ranked, start=1):
        item["rank"] = i
        item["underperforming"] = bool(item.get("metrics", {}).get("f1", 0.0) < 0.45)
    return ranked


# _artifact_path service logic.
def _artifact_path(model_id: str) -> Path:
    return _model_store_dir() / f"{_slug(model_id)}.joblib"


# _save_model_artifact service logic.
def _save_model_artifact(model_id: str, model_obj: Any) -> Optional[str]:
    if joblib is None:
        return None
    p = _artifact_path(model_id)
    joblib.dump(model_obj, p)
    return str(p)


# _delete_artifact service logic.
def _delete_artifact(artifact_path: Optional[str]) -> None:
    if not artifact_path:
        return
    try:
        p = Path(artifact_path)
        if p.exists():
            p.unlink()
    except Exception:
        return


# _registry_col service logic.
def _registry_col():
    return db.require_col(db.ml_models_col, "ml_models")


# _runs_col service logic.
def _runs_col():
    return db.require_col(db.ml_model_runs_col, "ml_model_runs")


# _runtime_settings_col service logic.
def _runtime_settings_col():
    return db.require_col(db.ml_runtime_settings_col, "ml_runtime_settings")


# get_ml_runtime_settings service logic.
def get_ml_runtime_settings() -> Dict[str, Any]:
    col = _runtime_settings_col()
    doc = col.find_one({"key": "pipeline"}) or {}

    tp = float(doc.get("tp_barrier", _RUNTIME_SETTINGS_DEFAULTS["tp_barrier"]))
    sl = float(doc.get("sl_barrier", _RUNTIME_SETTINGS_DEFAULTS["sl_barrier"]))
    tb = int(doc.get("time_barrier_days", _RUNTIME_SETTINGS_DEFAULTS["time_barrier_days"]))
    splits = int(doc.get("walk_forward_splits", _RUNTIME_SETTINGS_DEFAULTS["walk_forward_splits"]))
    min_train = int(doc.get("walk_forward_min_train", _RUNTIME_SETTINGS_DEFAULTS["walk_forward_min_train"]))
    purge_days = int(doc.get("walk_forward_purge_days", _RUNTIME_SETTINGS_DEFAULTS["walk_forward_purge_days"]))

    return {
        "tp_barrier": max(0.001, tp),
        "sl_barrier": max(0.001, sl),
        "time_barrier_days": max(1, tb),
        "walk_forward_splits": max(2, splits),
        "walk_forward_min_train": max(40, min_train),
        "walk_forward_purge_days": max(0, purge_days),
    }


# update_ml_runtime_settings service logic.
def update_ml_runtime_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    current = get_ml_runtime_settings()
    allowed = {
        "tp_barrier",
        "sl_barrier",
        "time_barrier_days",
        "walk_forward_splits",
        "walk_forward_min_train",
        "walk_forward_purge_days",
    }

    for k, v in (payload or {}).items():
        if k not in allowed or v is None:
            continue
        current[k] = v

    normalized = {
        "tp_barrier": max(0.001, float(current["tp_barrier"])),
        "sl_barrier": max(0.001, float(current["sl_barrier"])),
        "time_barrier_days": max(1, int(current["time_barrier_days"])),
        "walk_forward_splits": max(2, int(current["walk_forward_splits"])),
        "walk_forward_min_train": max(40, int(current["walk_forward_min_train"])),
        "walk_forward_purge_days": max(0, int(current.get("walk_forward_purge_days", 0))),
    }

    col = _runtime_settings_col()
    col.update_one(
        {"key": "pipeline"},
        {
            "$set": {
                "key": "pipeline",
                **normalized,
                "updated_at": _utc_now(),
            }
        },
        upsert=True,
    )
    return normalized


# _store_model_record service logic.
def _store_model_record(item: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    model_id = f"{item['algorithm']}_{uuid.uuid4().hex[:10]}"
    artifact_path = _save_model_artifact(model_id, item["model"])
    completeness = _data_completeness_from_news_coverage(item.get("news_coverage"))
    nlp_pipeline = _normalize_nlp_pipeline(str(item.get("nlp_pipeline") or "lexicon"))

    doc = {
        "model_id": model_id,
        "run_id": run_id,
        "algorithm": item["algorithm"],
        "family": item.get("family") or _family_from_algorithm(item["algorithm"]),
        "feature_type": item.get("feature_type", "numeric"),
        "metrics": item["metrics"],
        "score": float(item["score"]),
        "rank": int(item.get("rank", 0)),
        "sample_count": int(item.get("sample_count", 0)),
        "artifact_path": artifact_path,
        "is_selected": False,
        "is_deployed": False,
        "underperforming": bool(item.get("underperforming", False)),
        "nlp_pipeline": nlp_pipeline,
        # Keep top-level ratio for quick filtering in admin/debug queries.
        "news_coverage_ratio": float(completeness["news_coverage_ratio"]),
        "metadata": {
            "data_completeness": completeness,
            "nlp_pipeline": nlp_pipeline,
        },
        "created_at": _utc_now(),
    }

    col = _registry_col()
    col.update_one({"model_id": model_id}, {"$set": doc}, upsert=True)
    return doc


# _doc_to_model_info service logic.
def _doc_to_model_info(doc: Dict[str, Any]) -> MLModelInfo:
    resolved_news_ratio = _resolve_news_coverage_ratio(doc)
    raw_feature = str(doc.get("feature_type") or "numeric").lower()
    feature_type: Literal["numeric", "text"] = "text" if raw_feature == "text" else "numeric"
    return MLModelInfo(
        model_id=str(doc.get("model_id") or ""),
        algorithm=str(doc.get("algorithm") or ""),
        family=str(doc.get("family") or "other"),
        feature_type=feature_type,
        nlp_pipeline=_resolve_model_nlp_pipeline(doc),
        metrics=MLModelMetric(**(doc.get("metrics") or {})),
        rank=int(doc.get("rank") or 0),
        score=float(doc.get("score") or 0.0),
        sample_count=int(doc.get("sample_count") or 0),
        is_selected=bool(doc.get("is_selected", False)),
        is_deployed=bool(doc.get("is_deployed", False)),
        underperforming=bool(doc.get("underperforming", False)),
        news_coverage_ratio=resolved_news_ratio,
        created_at=doc.get("created_at"),
    )


# _resolve_news_coverage_ratio service logic.
def _resolve_news_coverage_ratio(doc: Dict[str, Any]) -> Optional[float]:
    top_level = doc.get("news_coverage_ratio")
    if top_level is not None:
        try:
            return float(top_level)
        except Exception:
            pass

    metadata = doc.get("metadata") or {}
    if isinstance(metadata, dict):
        data_completeness = metadata.get("data_completeness") or {}
        if isinstance(data_completeness, dict):
            nested = data_completeness.get("news_coverage_ratio")
            if nested is not None:
                try:
                    return float(nested)
                except Exception:
                    pass

        legacy_news_coverage = metadata.get("news_coverage")
        if isinstance(legacy_news_coverage, dict):
            try:
                comp = _data_completeness_from_news_coverage(legacy_news_coverage)
                return float(comp.get("news_coverage_ratio") or 0.0)
            except Exception:
                pass

    run_id = str(doc.get("run_id") or "").strip()
    if not run_id:
        return None

    try:
        run_doc = _runs_col().find_one({"run_id": run_id}, {"news_coverage": 1}) or {}
    except Exception:
        return None

    run_news = run_doc.get("news_coverage")
    if not isinstance(run_news, dict):
        return None

    try:
        comp = _data_completeness_from_news_coverage(run_news)
        return float(comp.get("news_coverage_ratio") or 0.0)
    except Exception:
        return None


# _resolve_model_nlp_pipeline service logic.
def _resolve_model_nlp_pipeline(doc: Dict[str, Any]) -> Literal["lexicon", "finbert", "cascade"]:
    top_level = str(doc.get("nlp_pipeline") or "").strip().lower()
    if top_level == "lexicon":
        return "lexicon"
    if top_level == "finbert":
        return "finbert"
    if top_level == "cascade":
        return "cascade"

    metadata = doc.get("metadata") or {}
    if isinstance(metadata, dict):
        nested = str(metadata.get("nlp_pipeline") or "").strip().lower()
        if nested == "lexicon":
            return "lexicon"
        if nested == "finbert":
            return "finbert"
        if nested == "cascade":
            return "cascade"

    # Legacy models were trained with FinBERT path (with internal lexicon fallback).
    return "finbert"


def _model_declares_nlp_pipeline(doc: Dict[str, Any]) -> bool:
    top_level = str(doc.get("nlp_pipeline") or "").strip().lower()
    if top_level in ("lexicon", "finbert", "cascade"):
        return True

    metadata = doc.get("metadata") or {}
    if isinstance(metadata, dict):
        nested = str(metadata.get("nlp_pipeline") or "").strip().lower()
        if nested in ("lexicon", "finbert", "cascade"):
            return True

    return False


# _pipeline_filter_query service logic.
def _pipeline_filter_query(nlp_pipeline: Optional[str], include_legacy_finbert: bool = True) -> Dict[str, Any]:
    if nlp_pipeline is None:
        return {}

    p = _normalize_nlp_pipeline(nlp_pipeline)
    if p == "finbert" and include_legacy_finbert:
        return {
            "$or": [
                {"nlp_pipeline": "finbert"},
                {"nlp_pipeline": {"$exists": False}},
                {"nlp_pipeline": None},
            ]
        }
    return {"nlp_pipeline": p}


# list_models service logic.
def list_models(nlp_pipeline: Optional[str] = None) -> List[MLModelInfo]:
    col = _registry_col()
    docs = list(col.find(_pipeline_filter_query(nlp_pipeline)).sort([("rank", 1), ("score", -1), ("created_at", -1)]))
    out: List[MLModelInfo] = []
    for d in docs:
        d.pop("_id", None)
        out.append(_doc_to_model_info(d))
    return out


# get_tournament_competitor_stats service logic.
def get_tournament_competitor_stats(model_id: Optional[str] = None) -> Dict[str, Any]:
    doc = _selected_or_latest_model(model_id)
    if not doc:
        return {
            "model_id": model_id,
            "algorithm": None,
            "winner_name": None,
            "competitor_stats": {},
        }

    resolved_model_id = str(doc.get("model_id") or (model_id or "")) or None
    algorithm = str(doc.get("algorithm") or "") or None
    if algorithm != "multi_armed_tournament":
        return {
            "model_id": resolved_model_id,
            "algorithm": algorithm,
            "winner_name": None,
            "competitor_stats": {},
        }

    try:
        model_obj = _load_model(doc)
    except Exception:
        return {
            "model_id": resolved_model_id,
            "algorithm": algorithm,
            "winner_name": None,
            "competitor_stats": {},
        }

    if not isinstance(model_obj, dict):
        return {
            "model_id": resolved_model_id,
            "algorithm": algorithm,
            "winner_name": None,
            "competitor_stats": {},
        }

    raw_stats = model_obj.get("competitor_stats") or {}
    stats: Dict[str, Dict[str, float]] = {}
    for name, payload in dict(raw_stats).items():
        if not isinstance(payload, dict):
            continue
        stats[str(name)] = {
            "sharpe": float(payload.get("sharpe", 0.0) or 0.0),
            "max_drawdown": float(payload.get("max_drawdown", 0.0) or 0.0),
            "mean_return": float(payload.get("mean_return", 0.0) or 0.0),
            "volatility": float(payload.get("volatility", 0.0) or 0.0),
            "sample_count": float(payload.get("sample_count", 0.0) or 0.0),
        }

    winner_name = str(model_obj.get("winner_name") or "") or None
    return {
        "model_id": resolved_model_id,
        "algorithm": algorithm,
        "winner_name": winner_name,
        "competitor_stats": stats,
    }


# get_model_feature_importances service logic.
def get_model_feature_importances(model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns feature importances for the selected/latest model.
    For tournament models, this attempts to extract importances from the winner model.
    """
    doc = _selected_or_latest_model(model_id)
    if not doc:
        return {
            "model_id": model_id,
            "winner_name": None,
            "feature_labels": [],
            "feature_importances": [],
        }

    resolved_model_id = str(doc.get("model_id") or (model_id or "")) or None
    algorithm = str(doc.get("algorithm") or "")

    try:
        model_obj = _load_model(doc)
    except Exception:
        return {
            "model_id": resolved_model_id,
            "winner_name": None,
            "feature_labels": [],
            "feature_importances": [],
        }

    # Tree-based single-model path (random_forest/boosting/xgboost/lightgbm etc).
    if hasattr(model_obj, "feature_importances_"):
        vals = np.asarray(getattr(model_obj, "feature_importances_"), dtype=float).reshape(-1)
        labels = list(BASE_FEATURE_NAMES[: len(vals)])
        if len(labels) < len(vals):
            labels.extend([f"feature_{i}" for i in range(len(labels), len(vals))])
        return {
            "model_id": resolved_model_id,
            "winner_name": None,
            "feature_labels": labels,
            "feature_importances": [float(x) for x in vals.tolist()],
        }

    # Tournament bundle path.
    if algorithm == "multi_armed_tournament" and isinstance(model_obj, dict):
        winner_name = str(model_obj.get("winner_name") or "") or None
        winner_model = model_obj.get("winner_model")
        base_names = [str(x) for x in (model_obj.get("base_model_names") or []) if str(x)]

        # MetaLabeler winner: one calibrated meta-model with explicit consensus-side features.
        vals = np.asarray(getattr(winner_model, "_meta_feature_importances", []), dtype=float).reshape(-1)
        if vals.size:
            labels: List[str] = ["consensus_side", "consensus_prob"]
            rem = int(vals.shape[0]) - len(labels)
            if rem > 0:
                labels.extend(list(BASE_FEATURE_NAMES[:rem]))
            if len(labels) < int(vals.shape[0]):
                labels.extend([f"feature_{i}" for i in range(len(labels), int(vals.shape[0]))])

            return {
                "model_id": resolved_model_id,
                "winner_name": winner_name,
                "feature_labels": labels,
                "feature_importances": [float(x) for x in vals.tolist()],
            }

        # Backward compatibility with older MetaLabeler bundle structure.
        models = getattr(winner_model, "_models", None)
        if isinstance(models, list) and models:
            rows: List[np.ndarray] = []
            for m in models:
                if hasattr(m, "feature_importances_"):
                    arr = np.asarray(getattr(m, "feature_importances_"), dtype=float).reshape(-1)
                    rows.append(arr)

            if rows:
                min_len = min(int(r.shape[0]) for r in rows)
                mat = np.vstack([r[:min_len] for r in rows])
                legacy_vals = np.mean(mat, axis=0)
                labels = [f"feature_{i}" for i in range(int(legacy_vals.shape[0]))]
                return {
                    "model_id": resolved_model_id,
                    "winner_name": winner_name,
                    "feature_labels": labels,
                    "feature_importances": [float(x) for x in legacy_vals.tolist()],
                }

        # Fallback if winner itself exposes importances.
        if hasattr(winner_model, "feature_importances_"):
            vals = np.asarray(getattr(winner_model, "feature_importances_"), dtype=float).reshape(-1)
            labels = [f"feature_{i}" for i in range(int(vals.shape[0]))]
            return {
                "model_id": resolved_model_id,
                "winner_name": winner_name,
                "feature_labels": labels,
                "feature_importances": [float(x) for x in vals.tolist()],
            }

        return {
            "model_id": resolved_model_id,
            "winner_name": winner_name,
            "feature_labels": [],
            "feature_importances": [],
        }

    return {
        "model_id": resolved_model_id,
        "winner_name": None,
        "feature_labels": [],
        "feature_importances": [],
    }


# ensure_selected_model_exists service logic.
def ensure_selected_model_exists() -> Optional[str]:
    col = _registry_col()
    selected = col.find_one({"is_selected": True}, sort=[("score", -1), ("created_at", -1)])
    if selected and selected.get("model_id"):
        return str(selected.get("model_id"))

    best = col.find_one({}, sort=[("score", -1), ("created_at", -1)])
    if not best or not best.get("model_id"):
        return None

    model_id = str(best.get("model_id"))
    col.update_many({}, {"$set": {"is_selected": False}})
    col.update_one({"model_id": model_id}, {"$set": {"is_selected": True}})
    return model_id


# select_best_models service logic.
def select_best_models(top_k: int = 1, nlp_pipeline: Optional[str] = None) -> MLSelectionResponse:
    col = _registry_col()
    filt = _pipeline_filter_query(nlp_pipeline)
    docs = list(col.find(filt).sort([("score", -1), ("created_at", -1)]))
    ranked = _rank_models(docs)

    if nlp_pipeline is None:
        col.update_many({}, {"$set": {"is_selected": False}})
    else:
        col.update_many(filt, {"$set": {"is_selected": False}})

    chosen = ranked[: max(1, int(top_k))]
    ids = [str(d.get("model_id")) for d in chosen if d.get("model_id")]
    if ids:
        col.update_many({"model_id": {"$in": ids}}, {"$set": {"is_selected": True}})

    return MLSelectionResponse(selected_model_id=(ids[0] if ids else None), ranked_model_ids=[str(d.get("model_id")) for d in ranked])


# _parse_iso_utc service logic.
def _parse_iso_utc(value: str) -> Optional[datetime]:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        return None


# backfill_model_nlp_pipeline service logic.
def backfill_model_nlp_pipeline(
    dry_run: bool = True,
    limit: int = 0,
    rollout_iso: Optional[str] = None,
    default_pipeline: str = "finbert",
) -> Dict[str, Any]:
    col = _registry_col()
    runs = _runs_col()

    docs = list(col.find({}).sort([("created_at", 1)]))
    if int(limit) > 0:
        docs = docs[: int(limit)]

    resolved_default = _normalize_nlp_pipeline(default_pipeline)
    cutoff = _parse_iso_utc(rollout_iso or os.getenv("ML_NLP_PIPELINE_ROLLOUT_AT", "2026-03-11T00:00:00+00:00"))

    inspected = 0
    updated = 0
    skipped = 0
    by_reason: Dict[str, int] = {}
    changes: List[Dict[str, Any]] = []

    for doc in docs:
        inspected += 1
        mid = str(doc.get("model_id") or "")
        if not mid:
            skipped += 1
            continue

        current = str(doc.get("nlp_pipeline") or "").strip().lower()
        if current in ("lexicon", "finbert", "cascade"):
            skipped += 1
            continue

        inferred: Optional[str] = None
        reason = ""

        metadata = doc.get("metadata") or {}
        if isinstance(metadata, dict):
            nested = str(metadata.get("nlp_pipeline") or "").strip().lower()
            if nested in ("lexicon", "finbert", "cascade"):
                inferred = nested
                reason = "metadata"

        if inferred is None:
            run_id = str(doc.get("run_id") or "").strip()
            if run_id:
                try:
                    run_doc = runs.find_one({"run_id": run_id}, {"nlp_pipeline": 1, "result": 1}) or {}
                    run_pipe = str(run_doc.get("nlp_pipeline") or "").strip().lower()
                    if run_pipe not in ("lexicon", "finbert", "cascade") and isinstance(run_doc.get("result"), dict):
                        run_pipe = str((run_doc.get("result") or {}).get("nlp_pipeline") or "").strip().lower()
                    if run_pipe in ("lexicon", "finbert", "cascade"):
                        inferred = run_pipe
                        reason = "run_metadata"
                except Exception:
                    pass

        if inferred is None:
            art = Path(str(doc.get("artifact_path") or "")) if doc.get("artifact_path") else None
            art_dt: Optional[datetime] = None
            if art is not None and art.exists():
                try:
                    art_dt = datetime.fromtimestamp(art.stat().st_mtime, tz=timezone.utc)
                except Exception:
                    art_dt = None

            if cutoff is not None and art_dt is not None:
                inferred = "cascade" if art_dt >= cutoff else "finbert"
                reason = "artifact_timestamp"

        if inferred is None:
            inferred = resolved_default
            reason = "default"

        by_reason[reason] = int(by_reason.get(reason, 0)) + 1
        changes.append({"model_id": mid, "nlp_pipeline": inferred, "reason": reason})

        if not dry_run:
            col.update_one(
                {"model_id": mid},
                {
                    "$set": {
                        "nlp_pipeline": inferred,
                        "updated_at": _utc_now(),
                        "metadata.nlp_pipeline": inferred,
                    }
                },
            )
            updated += 1

    return {
        "dry_run": bool(dry_run),
        "inspected": int(inspected),
        "updated": int(updated),
        "skipped": int(skipped),
        "reason_counts": by_reason,
        "changes": changes,
    }


# _select_prune_candidates service logic.
def _select_prune_candidates(docs: List[Dict[str, Any]], max_models: int) -> List[Dict[str, Any]]:
    if len(docs) < int(max_models):
        return []

    return [d for d in docs if not bool(d.get("is_selected", False))]


# prune_underperforming_models service logic.
def prune_underperforming_models(min_f1: float, min_accuracy: float) -> MLPruneResponse:
    col = _registry_col()
    docs = list(col.find({}).sort([("is_selected", -1), ("score", -1), ("created_at", -1)]))
    deleted: List[str] = []

    candidates = _select_prune_candidates(docs, ML_MODEL_MAX_COUNT)
    if not candidates:
        return MLPruneResponse(deleted_model_ids=[])

    for d in candidates:
        metrics = d.get("metrics") or {}
        f1 = float(metrics.get("f1") or 0.0)
        acc = float(metrics.get("accuracy") or 0.0)
        if f1 >= float(min_f1) and acc >= float(min_accuracy):
            continue

        model_id = str(d.get("model_id") or "")
        if not model_id:
            continue

        _delete_artifact(d.get("artifact_path"))
        col.delete_one({"model_id": model_id})
        deleted.append(model_id)

    docs_after = list(col.find({}).sort([("created_at", 1)]))
    while len(docs_after) >= int(ML_MODEL_MAX_COUNT):
        victim = next((d for d in docs_after if not bool(d.get("is_selected", False))), None)
        if not victim:
            break

        victim_id = str(victim.get("model_id") or "")
        if not victim_id:
            break

        _delete_artifact(victim.get("artifact_path"))
        col.delete_one({"model_id": victim_id})
        deleted.append(victim_id)
        docs_after = [d for d in docs_after if str(d.get("model_id") or "") != victim_id]

    return MLPruneResponse(deleted_model_ids=deleted)


# _selected_or_latest_model service logic.
def _selected_or_latest_model(model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    col = _registry_col()
    if model_id:
        return col.find_one({"model_id": model_id})

    return col.find_one({"is_selected": True}, sort=[("score", -1), ("created_at", -1)])


# _load_model service logic.
def _load_model(doc: Dict[str, Any]) -> Any:
    if joblib is None:
        raise RuntimeError("joblib is required for persisted model loading")

    path = doc.get("artifact_path")
    if not path:
        raise RuntimeError("Model artifact path not found")

    p = Path(str(path))
    if not p.exists():
        raise RuntimeError(f"Model artifact missing: {path}")
    return joblib.load(p)


# _algo_factories service logic.
def _algo_factories(random_seed: int) -> List[Tuple[str, Any]]:
    jobs: List[Tuple[str, Any]] = [
        ("ann_relu", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=700, random_state=random_seed)),
        ("ann_sigmoid", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), activation="logistic", max_iter=700, random_state=random_seed)),
        ("svm_rbf", lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=random_seed)),
        ("svm_poly", lambda: SVC(kernel="poly", degree=3, C=1.0, probability=True, random_state=random_seed)),
        (
            "random_forest",
            lambda: ExtraTreesClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=random_seed,
                n_jobs=-1,
            ),
        ),
        ("gradient_boosting", lambda: GradientBoostingClassifier(random_state=random_seed)),
    ]

    xgb_cls = _load_xgboost_classifier()
    if xgb_cls is not None:
        jobs.append(
            (
                "xgboost",
                lambda: xgb_cls(
                    n_estimators=180,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=random_seed,
                ),
            )
        )

    lgbm_cls = _load_lightgbm_classifier()
    if lgbm_cls is not None:
        jobs.append(
            (
                "lightgbm",
                lambda: lgbm_cls(
                    n_estimators=200,
                    learning_rate=0.07,
                    num_leaves=31,
                    random_state=random_seed,
                ),
            )
        )

    return jobs


# _safe_test_size service logic.
def _safe_test_size(n_samples: int, req_test_size: float, n_splits: int) -> int:
    candidate = max(20, int(n_samples * max(0.10, min(0.35, float(req_test_size)))))
    max_allowed = max(10, (n_samples // max(2, n_splits + 1)) - 1)
    return max(10, min(candidate, max_allowed))


# _build_walk_forward_splits service logic.
def _build_walk_forward_splits(
    n_samples: int,
    req_test_size: float,
    n_splits: int,
    min_train_samples: int,
    purge_days: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples < 120:
        return []

    n_splits = max(2, int(n_splits))
    n_splits = min(n_splits, max(2, n_samples // 25))
    test_size = _safe_test_size(n_samples, req_test_size, n_splits)

    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    purge = max(0, int(purge_days))
    for tr_idx, te_idx in splitter.split(np.arange(n_samples)):
        if te_idx.size == 0:
            continue
        te_start = int(te_idx[0])
        if purge > 0:
            cutoff = te_start - purge
            tr_idx = tr_idx[tr_idx < cutoff]

        if len(tr_idx) < int(min_train_samples):
            continue
        if len(te_idx) < 10:
            continue
        out.append((np.asarray(tr_idx, dtype=int), np.asarray(te_idx, dtype=int)))
    return out


# _fit_score_over_walk_forward service logic.
def _fit_score_over_walk_forward(
    algorithm: str,
    factory: Any,
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    actual_returns: Optional[np.ndarray] = None,
    oof_collector: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_prob_all: List[float] = []
    strategy_ret_all: List[float] = []
    abs_ret_err_all: List[float] = []
    sq_ret_err_all: List[float] = []

    returns_arr: Optional[np.ndarray]
    if actual_returns is None:
        returns_arr = None
    else:
        arr = np.asarray(actual_returns, dtype=float).reshape(-1)
        returns_arr = arr if arr.shape[0] == X.shape[0] else None

    for tr_idx, te_idx in splits:
        model = factory()
        model.fit(X[tr_idx], y[tr_idx])
        probs = _predict_proba_or_hard(model, X[te_idx])
        preds = np.where(probs >= 0.5, 1, 0)

        y_true_all.extend([int(v) for v in y[te_idx]])
        y_pred_all.extend([int(v) for v in preds])
        y_prob_all.extend([float(v) for v in probs])

        if returns_arr is not None:
            strat_rets = _strategy_daily_returns_from_probs(np.asarray(probs, dtype=float), returns_arr[te_idx])
            strategy_ret_all.extend([float(v) for v in strat_rets])
            dn_mu, up_mu = _class_conditional_return_anchors(y[tr_idx], returns_arr[tr_idx])
            pred_ret = dn_mu + (np.asarray(probs, dtype=float) * (up_mu - dn_mu))
            act_ret = np.asarray(returns_arr[te_idx], dtype=float)
            abs_err = np.abs(pred_ret - act_ret)
            sq_err = np.square(pred_ret - act_ret)
            abs_ret_err_all.extend([float(v) for v in abs_err])
            sq_ret_err_all.extend([float(v) for v in sq_err])

        if oof_collector is not None:
            oof_collector[te_idx] = probs

    if not y_true_all:
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0}
    else:
        metrics = _evaluate_binary(
            np.asarray(y_true_all, dtype=int),
            np.asarray(y_pred_all, dtype=int),
            np.asarray(y_prob_all, dtype=float),
        )

    if abs_ret_err_all:
        mae_pct = float(np.mean(np.asarray(abs_ret_err_all, dtype=float)) * 100.0)
        rmse_pct = float(np.sqrt(np.mean(np.asarray(sq_ret_err_all, dtype=float))) * 100.0)
    else:
        mae_pct = float("inf")
        rmse_pct = float("inf")

    metrics["future_price_mae_pct"] = mae_pct
    metrics["future_price_rmse_pct"] = rmse_pct
    strategy_arr = np.asarray(strategy_ret_all, dtype=float)
    metrics["annualized_sharpe"] = _annualized_sharpe_ratio_from_returns(strategy_arr)
    metrics["max_drawdown"] = _max_drawdown_from_returns(strategy_arr)

    final_model = factory()
    final_model.fit(X, y)

    return {
        "algorithm": algorithm,
        "family": _family_from_algorithm(algorithm),
        "feature_type": "numeric",
        "model": final_model,
        "metrics": metrics,
        "score": _weighted_score(metrics),
        "sample_count": int(X.shape[0]),
    }


# _train_stacking_meta service logic.
def _train_stacking_meta(
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    base_factories: List[Tuple[str, Any]],
) -> Dict[str, Any]:
    names = [n for n, _ in base_factories]
    oof = np.full((X.shape[0], len(names)), np.nan, dtype=float)
    final_base_models: Dict[str, Any] = {}

    for col_idx, (name, factory) in enumerate(base_factories):
        _ml_log(f"walk-forward base training: {name}")
        res = _fit_score_over_walk_forward(name, factory, X, y, splits, oof_collector=oof[:, col_idx])
        final_base_models[name] = res["model"]

    valid_mask = ~np.isnan(oof).any(axis=1)
    if int(np.sum(valid_mask)) < 40:
        raise RuntimeError("Insufficient OOF rows for stacking meta-learner")

    X_meta = oof[valid_mask]
    y_meta = y[valid_mask]

    meta = LogisticRegression(max_iter=700, class_weight="balanced")
    meta.fit(X_meta, y_meta)

    probs = meta.predict_proba(X_meta)[:, 1]
    preds = np.where(probs >= 0.5, 1, 0)
    metrics = _evaluate_binary(y_meta, preds, probs)

    packed_model = {
        "kind": "stacking_ensemble",
        "base_model_names": names,
        "base_models": final_base_models,
        "meta_model": meta,
    }

    return {
        "algorithm": "stacking_meta",
        "family": "ensemble",
        "feature_type": "numeric",
        "model": packed_model,
        "metrics": metrics,
        "score": _weighted_score(metrics),
        "sample_count": int(X.shape[0]),
    }


# _predict_stacking_model service logic.
def _predict_stacking_model(model_bundle: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    names = list(model_bundle.get("base_model_names") or [])
    base_models = dict(model_bundle.get("base_models") or {})
    meta = model_bundle.get("meta_model")
    if not names or not base_models or meta is None:
        raise RuntimeError("Invalid stacking model bundle")

    cols: List[np.ndarray] = []
    for name in names:
        m = base_models.get(name)
        if m is None:
            raise RuntimeError(f"Missing base model in stack: {name}")
        cols.append(_predict_proba_or_hard(m, X))

    X_meta = np.column_stack(cols)
    return _predict_proba_or_hard(meta, X_meta)


# _predict_tournament_model service logic.
def _predict_tournament_model(model_bundle: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    names = list(model_bundle.get("base_model_names") or [])
    base_models = dict(model_bundle.get("base_models") or {})
    combiner = model_bundle.get("winner_model")
    if not names or not base_models or combiner is None:
        raise RuntimeError("Invalid tournament model bundle")

    cols: List[np.ndarray] = []
    for name in names:
        m = base_models.get(name)
        if m is None:
            raise RuntimeError(f"Missing base model in tournament: {name}")
        cols.append(_predict_proba_or_hard(m, X))

    base_matrix = np.column_stack(cols)
    if bool(getattr(combiner, "supports_cross_sectional", False)):
        current_alloc = np.zeros(base_matrix.shape[0], dtype=float)
        alloc_vec = combiner.allocate(base_matrix, X, current_allocation=current_alloc)
        alloc_arr = np.asarray(alloc_vec, dtype=float).reshape(-1)
        if alloc_arr.size != base_matrix.shape[0]:
            raise RuntimeError("Tournament combiner returned invalid cross-sectional allocation shape")
        alloc_arr = np.clip(alloc_arr, -1.0, 1.0)
    else:
        allocations: List[float] = []
        current_alloc = np.zeros(1, dtype=float)
        for i in range(base_matrix.shape[0]):
            alloc_vec = combiner.allocate(base_matrix[i], X[i], current_allocation=current_alloc)
            current_alloc = np.asarray(alloc_vec, dtype=float).reshape(-1)
            alloc = scalar_allocation(alloc_vec)
            allocations.append(float(np.clip(alloc, -1.0, 1.0)))
        alloc_arr = np.asarray(allocations, dtype=float)

    return np.clip((alloc_arr + 1.0) / 2.0, 0.0, 1.0)


# _train_multi_armed_tournament service logic.
def _train_multi_armed_tournament(
    X: np.ndarray,
    y: np.ndarray,
    actual_returns: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    base_factories: List[Tuple[str, Any]],
    random_seed: int,
) -> Dict[str, Any]:
    names = [n for n, _ in base_factories]
    oof = np.full((X.shape[0], len(names)), np.nan, dtype=float)
    final_base_models: Dict[str, Any] = {}

    for col_idx, (name, factory) in enumerate(base_factories):
        _ml_log(f"walk-forward base training (tournament): {name}")
        res = _fit_score_over_walk_forward(name, factory, X, y, splits, oof_collector=oof[:, col_idx])
        final_base_models[name] = res["model"]

    valid_mask = ~np.isnan(oof).any(axis=1)
    if int(np.sum(valid_mask)) < 60:
        raise RuntimeError("Insufficient OOF rows for tournament combiner")

    valid_indices = np.where(valid_mask)[0]
    index_map = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(valid_indices.tolist())}

    filtered_splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_idx, te_idx in splits:
        tr_mapped = [index_map[int(i)] for i in tr_idx.tolist() if int(i) in index_map]
        te_mapped = [index_map[int(i)] for i in te_idx.tolist() if int(i) in index_map]
        if len(tr_mapped) < 40 or len(te_mapped) < 10:
            continue
        filtered_splits.append((np.asarray(tr_mapped, dtype=int), np.asarray(te_mapped, dtype=int)))

    X_market = X[valid_mask]
    y_valid = y[valid_mask]
    returns_valid = np.asarray(actual_returns[valid_mask], dtype=float)
    base_oof = oof[valid_mask]

    if not filtered_splits:
        n = base_oof.shape[0]
        cut = max(40, int(n * 0.7))
        if n - cut < 10:
            raise RuntimeError("Insufficient valid rows to evaluate tournament competitors")
        filtered_splits = [(np.arange(0, cut, dtype=int), np.arange(cut, n, dtype=int))]

    tournament_result = run_walk_forward_tournament(
        competitor_factories=get_meta_competitor_factories(random_seed=random_seed),
        base_predictions=base_oof,
        market_features=X_market,
        actual_returns=returns_valid,
        splits=filtered_splits,
    )

    alloc_vals: List[float] = []
    current_alloc = np.zeros(1, dtype=float)
    for i in range(base_oof.shape[0]):
        alloc_vec = tournament_result.winner_model.allocate(
            base_oof[i],
            X_market[i],
            current_allocation=current_alloc,
        )
        current_alloc = np.asarray(alloc_vec, dtype=float).reshape(-1)
        alloc_vals.append(scalar_allocation(alloc_vec))
    allocations = np.asarray(alloc_vals, dtype=float)
    probs = np.clip((np.clip(allocations, -1.0, 1.0) + 1.0) / 2.0, 0.0, 1.0)
    preds = np.where(probs >= 0.5, 1, 0)
    metrics = _evaluate_binary(y_valid, preds, probs)
    strategy_rets = _strategy_daily_returns_from_probs(probs, returns_valid)
    metrics["annualized_sharpe"] = _annualized_sharpe_ratio_from_returns(strategy_rets)
    metrics["max_drawdown"] = _max_drawdown_from_returns(strategy_rets)
    dn_mu, up_mu = _class_conditional_return_anchors(y_valid, returns_valid)
    pred_ret = dn_mu + (np.asarray(probs, dtype=float) * (up_mu - dn_mu))
    abs_err = np.abs(pred_ret - returns_valid)
    sq_err = np.square(pred_ret - returns_valid)
    metrics["future_price_mae_pct"] = float(np.mean(abs_err) * 100.0) if abs_err.size else float("inf")
    metrics["future_price_rmse_pct"] = float(np.sqrt(np.mean(sq_err)) * 100.0) if sq_err.size else float("inf")

    packed_model = {
        "kind": "multi_armed_tournament",
        "base_model_names": names,
        "base_models": final_base_models,
        "winner_name": tournament_result.winner_name,
        "winner_model": tournament_result.winner_model,
        "competitor_stats": tournament_result.competitor_stats,
    }

    return {
        "algorithm": "multi_armed_tournament",
        "family": "ensemble",
        "feature_type": "numeric",
        "model": packed_model,
        "metrics": metrics,
        "score": _weighted_score(metrics),
        "sample_count": int(X.shape[0]),
    }


# predict_for_basket service logic.
async def predict_for_basket(stock_basket: List[str], lookback_days: int = 120, model_id: Optional[str] = None) -> MLPredictionResponse:
    symbols = list(dict.fromkeys([(s or "").upper().strip() for s in stock_basket if (s or "").strip()]))
    items: List[MLPredictionItem] = []
    histories = await get_price_histories(symbols, days=max(30, int(lookback_days)), concurrency=8)
    today = iso_date_utc()
    active_pipeline = _active_nlp_pipeline()

    # Refresh and persist today's news before deriving inference-time features.
    await refresh_news_if_needed(today, symbols)
    day_ctx = _daily_news_features_compat(today, symbols, nlp_pipeline=active_pipeline)

    X_num: List[List[float]] = []
    valid_symbols: List[str] = []
    for sym in symbols:
        closes = [float(p.get("c", 0.0)) for p in (histories.get(sym) or {}).get("points") or [] if p.get("c") is not None]
        tech = _technical_features(closes)
        sym_sent = float(day_ctx.symbol_sentiment.get(sym, day_ctx.market_sentiment))
        feats = tech + [sym_sent, float(day_ctx.market_sentiment)] + list(day_ctx.macro_features)
        valid_symbols.append(sym)
        X_num.append(feats)

    if not X_num:
        return MLPredictionResponse(model_id=(model_id or "selected"), items=[])

    arr = np.array(X_num, dtype=float)

    if model_id:
        doc = _selected_or_latest_model(model_id)
        if not doc:
            raise RuntimeError(f"Requested model_id not found: {model_id}")
        model_pipeline = _resolve_model_nlp_pipeline(doc)
        if _model_declares_nlp_pipeline(doc) and model_pipeline != active_pipeline:
            raise RuntimeError(
                "Requested model was trained with incompatible NLP pipeline "
                f"('{model_pipeline}') while runtime pipeline is '{active_pipeline}'."
            )
        model = _load_model(doc)
        algo = str(doc.get("algorithm") or "")
        if algo == "stacking_meta" and isinstance(model, dict):
            probs = _predict_stacking_model(model, arr)
        elif algo == "multi_armed_tournament" and isinstance(model, dict):
            probs = _predict_tournament_model(model, arr)
        else:
            probs = _predict_proba_or_hard(model, arr)

        preds = np.where(probs >= 0.5, 1, 0)
        for i, sym in enumerate(valid_symbols):
            p_up = float(probs[i]) if i < len(probs) else (1.0 if int(preds[i]) == 1 else 0.0)
            items.append(MLPredictionItem(symbol=sym, prediction=("up" if int(preds[i]) == 1 else "down"), probability_up=max(0.0, min(1.0, p_up))))

        return MLPredictionResponse(model_id=str(doc.get("model_id")), items=items)

    col = _registry_col()
    selected_query = {"is_selected": True, **_pipeline_filter_query(active_pipeline)}
    selected_docs = list(col.find(selected_query).sort([("score", -1), ("created_at", -1)]))
    if not selected_docs:
        fallback = _selected_or_latest_model(model_id=None)
        if fallback and (not _model_declares_nlp_pipeline(fallback) or _resolve_model_nlp_pipeline(fallback) == active_pipeline):
            selected_docs = [fallback]
    selected_docs = selected_docs[: max(1, int(ML_SELECTED_TOP_K))]

    model_probs: List[np.ndarray] = []
    used_ids: List[str] = []

    for d in selected_docs:
        if _model_declares_nlp_pipeline(d) and _resolve_model_nlp_pipeline(d) != active_pipeline:
            continue
        try:
            model = _load_model(d)
            algo = str(d.get("algorithm") or "")
            if algo == "stacking_meta" and isinstance(model, dict):
                probs_i = _predict_stacking_model(model, arr)
            elif algo == "multi_armed_tournament" and isinstance(model, dict):
                probs_i = _predict_tournament_model(model, arr)
            else:
                probs_i = _predict_proba_or_hard(model, arr)
            model_probs.append(np.asarray(probs_i, dtype=float))
            used_ids.append(str(d.get("model_id") or ""))
        except Exception:
            continue

    if not model_probs:
        raise RuntimeError(
            "No selected model is available for NLP pipeline "
            f"'{active_pipeline}'. Train/select a model built with this pipeline."
        )

    probs = np.mean(np.column_stack(model_probs), axis=1)

    preds = np.where(probs >= 0.5, 1, 0)
    for i, sym in enumerate(valid_symbols):
        p_up = float(probs[i]) if i < len(probs) else (1.0 if int(preds[i]) == 1 else 0.0)
        items.append(MLPredictionItem(symbol=sym, prediction=("up" if int(preds[i]) == 1 else "down"), probability_up=max(0.0, min(1.0, p_up))))

    mix_id = ",".join([x for x in used_ids if x])
    return MLPredictionResponse(model_id=(f"mix[{mix_id}]" if mix_id else "mix[selected]"), items=items)


async def predict_series_for_symbol(
    symbol: str,
    lookback_days: int = 365,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a historical probability series for one symbol using the selected/requested model.

    Returns:
      {
        "symbol": "AAPL",
        "model_id": "...",
        "series": [{"t": "YYYY-MM-DD", "prob_up": 0.53 | None}, ...]
      }

    Notes:
      - Uses rolling technical features per date.
      - Uses neutral (0) sentiment/macro slots at inference time for deterministic history replay.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"symbol": "", "model_id": (model_id or "selected"), "series": []}

    histories = await get_price_histories([sym], days=max(60, int(lookback_days)), concurrency=2)
    points = ((histories.get(sym) or {}).get("points") or [])
    if not points:
        return {"symbol": sym, "model_id": (model_id or "selected"), "series": []}

    closes: List[float] = []
    dates: List[str] = []
    for p in points:
        c = p.get("c")
        t = str(p.get("t") or "")
        if c is None or not t:
            continue
        try:
            closes.append(float(c))
            dates.append(t)
        except Exception:
            continue

    if len(closes) < 30:
        return {
            "symbol": sym,
            "model_id": (model_id or "selected"),
            "series": [{"t": d, "prob_up": None} for d in dates],
        }

    # Build rolling features aligned to each date.
    # First 24 rows are warm-up and remain None.
    feat_rows: List[List[float]] = []
    valid_idx: List[int] = []
    zero_tail = [0.0] * (len(BASE_FEATURE_NAMES) - 5)
    for i in range(len(closes)):
        tech = _technical_features(closes[: i + 1])
        if i < 24:
            continue
        feat_rows.append(tech + zero_tail)
        valid_idx.append(i)

    if not feat_rows:
        return {
            "symbol": sym,
            "model_id": (model_id or "selected"),
            "series": [{"t": d, "prob_up": None} for d in dates],
        }

    arr = np.array(feat_rows, dtype=float)
    active_pipeline = _active_nlp_pipeline()

    used_model_id = ""
    probs: Optional[np.ndarray] = None

    if model_id:
        doc = _selected_or_latest_model(model_id)
        if not doc:
            raise RuntimeError(f"Requested model_id not found: {model_id}")
        model_pipeline = _resolve_model_nlp_pipeline(doc)
        if _model_declares_nlp_pipeline(doc) and model_pipeline != active_pipeline:
            raise RuntimeError(
                "Requested model was trained with incompatible NLP pipeline "
                f"('{model_pipeline}') while runtime pipeline is '{active_pipeline}'."
            )
        model = _load_model(doc)
        algo = str(doc.get("algorithm") or "")
        if algo == "stacking_meta" and isinstance(model, dict):
            probs = _predict_stacking_model(model, arr)
        elif algo == "multi_armed_tournament" and isinstance(model, dict):
            probs = _predict_tournament_model(model, arr)
        else:
            probs = _predict_proba_or_hard(model, arr)
        used_model_id = str(doc.get("model_id") or "")
    else:
        col = _registry_col()
        selected_query = {"is_selected": True, **_pipeline_filter_query(active_pipeline)}
        selected_docs = list(col.find(selected_query).sort([("score", -1), ("created_at", -1)]))
        if not selected_docs:
            fallback = _selected_or_latest_model(model_id=None)
            if fallback and (not _model_declares_nlp_pipeline(fallback) or _resolve_model_nlp_pipeline(fallback) == active_pipeline):
                selected_docs = [fallback]
        selected_docs = selected_docs[: max(1, int(ML_SELECTED_TOP_K))]

        model_probs: List[np.ndarray] = []
        used_ids: List[str] = []
        for d in selected_docs:
            if _model_declares_nlp_pipeline(d) and _resolve_model_nlp_pipeline(d) != active_pipeline:
                continue
            try:
                model = _load_model(d)
                algo = str(d.get("algorithm") or "")
                if algo == "stacking_meta" and isinstance(model, dict):
                    probs_i = _predict_stacking_model(model, arr)
                elif algo == "multi_armed_tournament" and isinstance(model, dict):
                    probs_i = _predict_tournament_model(model, arr)
                else:
                    probs_i = _predict_proba_or_hard(model, arr)
                model_probs.append(np.asarray(probs_i, dtype=float))
                used_ids.append(str(d.get("model_id") or ""))
            except Exception:
                continue

        if model_probs:
            probs = np.mean(np.column_stack(model_probs), axis=1)
            mix_id = ",".join([x for x in used_ids if x])
            used_model_id = f"mix[{mix_id}]" if mix_id else "mix[selected]"

    if probs is None:
        raise RuntimeError(
            "No selected model is available for NLP pipeline "
            f"'{active_pipeline}'. Train/select a compatible model first."
        )

    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    full_probs: List[Optional[float]] = [None] * len(dates)
    for j, idx in enumerate(valid_idx):
        if j < len(probs) and 0 <= idx < len(full_probs):
            full_probs[idx] = float(probs[j])

    out_series: List[Dict[str, Any]] = []
    for i in range(len(dates)):
        prob_i = full_probs[i]
        prob_up_val: Optional[float]
        if prob_i is None:
            prob_up_val = None
        else:
            prob_up_val = round(float(prob_i), 6)
        out_series.append({"t": dates[i], "prob_up": prob_up_val})

    return {"symbol": sym, "model_id": (used_model_id or model_id or "selected"), "series": out_series}


# deploy_neural_network service logic.
async def deploy_neural_network(model_id: Optional[str] = None) -> MLDeployResponse:
    col = _registry_col()
    query: Dict[str, Any]
    if model_id:
        query = {"model_id": model_id, "family": "neural_network"}
    else:
        query = {"family": "neural_network"}

    doc = col.find_one(query, sort=[("score", -1), ("created_at", -1)])
    if not doc:
        return MLDeployResponse(deployed_model_id=None)

    col.update_many({}, {"$set": {"is_deployed": False}})
    mid = str(doc.get("model_id"))
    col.update_one({"model_id": mid}, {"$set": {"is_deployed": True, "deployed_at": _utc_now()}})
    return MLDeployResponse(deployed_model_id=mid)


# train_models_async service logic.
async def train_models_async(req: MLTrainingRequest) -> MLTrainResponse:
    pipeline_t0 = time.perf_counter()
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    _ml_log(f"pipeline started | run_id={run_id} | basket={len(req.stock_basket)} symbols")

    runs = _runs_col()
    runs.update_one(
        {"run_id": run_id},
        {
            "$set": {
                "run_id": run_id,
                "status": "running",
                "stage": "initializing",
                "request": req.model_dump(),
                "created_at": _utc_now(),
            }
        },
        upsert=True,
    )

    _ml_log("stage: building training matrices")
    runs.update_one({"run_id": run_id}, {"$set": {"stage": "building_matrices"}})

    runtime = get_ml_runtime_settings()

    active_pipeline = _active_nlp_pipeline()
    _ml_log(f"active NLP pipeline for training: {active_pipeline}")

    try:
        dataset = await build_training_matrices(
            stock_basket=req.stock_basket,
            lookback_days=req.lookback_days,
            label_horizon_days=req.label_horizon_days,
            nlp_pipeline=active_pipeline,
        )
    except DataValidationError as exc:
        detail = {"code": exc.code, **dict(exc.details or {})}
        _ml_log(f"pipeline failed: {exc}")
        runs.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": "failed",
                    "stage": "validation_gate",
                    "error": str(exc),
                    "validation": detail,
                    "failed_at": _utc_now(),
                }
            },
        )
        _alert_admin_ml_issue(str(exc), {"run_id": run_id, **detail})
        return MLTrainResponse(run_id=run_id, trained_models=[], selected_model_id=None, deleted_underperforming=[])
    except Exception as exc:
        _ml_log(f"pipeline failed: build_training_matrices crashed | {exc}")
        runs.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": "failed",
                    "stage": "building_matrices",
                    "error": f"Dataset build failed: {exc}",
                    "failed_at": _utc_now(),
                }
            },
        )
        _alert_admin_ml_issue("Dataset build failed before model training.", {"run_id": run_id, "error": str(exc)})
        return MLTrainResponse(run_id=run_id, trained_models=[], selected_model_id=None, deleted_underperforming=[])

    Xn: np.ndarray = dataset["X_numeric"]
    y: np.ndarray = dataset["y"]
    raw_returns = dataset.get("actual_returns")
    if raw_returns is None:
        actual_returns: np.ndarray = np.zeros(Xn.shape[0], dtype=float)
    else:
        actual_returns = np.asarray(raw_returns, dtype=float)
    sample_dates: List[str] = list(dataset.get("sample_dates") or [])

    if sample_dates and len(sample_dates) == int(Xn.shape[0]):
        order = np.argsort(np.array(sample_dates, dtype=object), kind="stable")
        Xn = Xn[order]
        y = y[order]
        actual_returns = actual_returns[order]

    news_coverage = dict(dataset.get("news_coverage") or {})
    trained_nlp_pipeline = _normalize_nlp_pipeline(str(dataset.get("nlp_pipeline") or active_pipeline))

    _ml_log(f"stage: dataset ready | samples={Xn.shape[0]} | labels={len(y)}")
    runs.update_one(
        {"run_id": run_id},
        {
            "$set": {
                "stage": "dataset_ready",
                "dataset_samples": int(Xn.shape[0]),
                "news_coverage": news_coverage,
                "nlp_pipeline": trained_nlp_pipeline,
            }
        },
    )

    if Xn.shape[0] < 120 or len(np.unique(y)) < 2:
        _ml_log("pipeline failed: insufficient labeled samples")
        runs.update_one({"run_id": run_id}, {"$set": {"status": "failed", "error": "Insufficient labeled samples"}})
        return MLTrainResponse(run_id=run_id, trained_models=[], selected_model_id=None, deleted_underperforming=[])

    _ml_log("stage: building walk-forward splits")
    runs.update_one({"run_id": run_id}, {"$set": {"stage": "walk_forward_split"}})
    splits = _build_walk_forward_splits(
        int(Xn.shape[0]),
        req.test_size,
        n_splits=int(runtime["walk_forward_splits"]),
        min_train_samples=int(runtime["walk_forward_min_train"]),
        purge_days=max(int(runtime.get("walk_forward_purge_days", 0)), int(runtime["time_barrier_days"])),
    )
    if not splits:
        runs.update_one({"run_id": run_id}, {"$set": {"status": "failed", "error": "Could not build walk-forward folds"}})
        return MLTrainResponse(run_id=run_id, trained_models=[], selected_model_id=None, deleted_underperforming=[])

    base_jobs = _algo_factories(req.random_seed)
    _ml_log(f"stage: walk-forward training started | models={len(base_jobs)}")
    runs.update_one(
        {"run_id": run_id},
        {"$set": {"stage": "training_models", "algorithm_count": int(len(base_jobs) + 1), "fold_count": len(splits)}},
    )

    completed: List[Dict[str, Any]] = []
    failed_jobs = 0
    # OOF probability arrays per algorithm – collected for visualization (Chart 2)
    oof_probs_for_viz: Dict[str, np.ndarray] = {}

    for name, factory in base_jobs:
        t0 = time.perf_counter()
        try:
            oof_col = np.full(Xn.shape[0], np.nan, dtype=float)
            res = _fit_score_over_walk_forward(
                name,
                factory,
                Xn,
                y,
                splits,
                actual_returns=actual_returns,
                oof_collector=oof_col,
            )
            oof_probs_for_viz[name] = oof_col
            completed.append(res)
            m = res.get("metrics") or {}
            _ml_log(
                f"training finished: {name} | time={time.perf_counter() - t0:.2f}s | acc={float(m.get('accuracy', 0.0)):.3f} f1={float(m.get('f1', 0.0)):.3f}"
            )
        except Exception:
            failed_jobs += 1

    try:
        t0 = time.perf_counter()
        tournament_item = _train_multi_armed_tournament(Xn, y, actual_returns, splits, base_jobs, req.random_seed)
        completed.append(tournament_item)
        sm = tournament_item.get("metrics") or {}
        _ml_log(
            f"training finished: multi_armed_tournament | time={time.perf_counter() - t0:.2f}s | acc={float(sm.get('accuracy', 0.0)):.3f} f1={float(sm.get('f1', 0.0)):.3f}"
        )
    except Exception:
        failed_jobs += 1

    _ml_log(f"stage: training completed | success={len(completed)} failed={failed_jobs}")
    runs.update_one(
        {"run_id": run_id},
        {"$set": {"stage": "training_completed", "successful_models": int(len(completed)), "failed_models": int(failed_jobs)}},
    )

    ranked = _rank_models(completed)
    _ml_log("stage: ranking + model storage")
    runs.update_one({"run_id": run_id}, {"$set": {"stage": "ranking_and_storage"}})

    infos: List[MLModelInfo] = []
    for item in ranked:
        item["news_coverage"] = news_coverage
        item["nlp_pipeline"] = trained_nlp_pipeline
        doc = _store_model_record(item, run_id=run_id)
        infos.append(_doc_to_model_info(doc))

    _ml_log("stage: selecting best model")
    runs.update_one({"run_id": run_id}, {"$set": {"stage": "selecting_best"}})
    selection = select_best_models(top_k=max(1, int(ML_SELECTED_TOP_K)))

    _ml_log("stage: pruning underperforming models")
    runs.update_one({"run_id": run_id}, {"$set": {"stage": "pruning_underperforming"}})
    prune = prune_underperforming_models(min_f1=0.40, min_accuracy=0.40)

    runs.update_one(
        {"run_id": run_id},
        {
            "$set": {
                "status": "completed",
                "stage": "completed",
                "completed_at": _utc_now(),
                "result": {
                    "trained_count": len(infos),
                    "selected_model_id": selection.selected_model_id,
                    "pruned_count": len(prune.deleted_model_ids),
                    "nlp_pipeline": trained_nlp_pipeline,
                },
            }
        },
    )

    pipeline_elapsed = time.perf_counter() - pipeline_t0
    _ml_log(
        f"pipeline completed | run_id={run_id} | total_time={pipeline_elapsed:.2f}s | trained={len(infos)} | selected={selection.selected_model_id or 'none'} | pruned={len(prune.deleted_model_ids)}"
    )

    # ── Post-training visualizations ──────────────────────────────────────────
    try:
        from ..ml.visualizations import run_training_visualizations
        # Collect stacking-meta OOF probs from the stacking result if it was trained
        stacking_item = next(
            (d for d in completed if str(d.get("algorithm") or "") == "stacking_meta"), None
        )
        if stacking_item:
            stk_model = stacking_item.get("model") or {}
            base_names = list(stk_model.get("base_model_names") or [])
            base_models_map = dict(stk_model.get("base_models") or {})
            meta_model = stk_model.get("meta_model")
            if base_names and base_models_map and meta_model is not None:
                oof_cols = [
                    oof_probs_for_viz.get(nm, np.full(Xn.shape[0], np.nan, dtype=float))
                    for nm in base_names
                ]
                valid = ~np.isnan(np.column_stack(oof_cols)).any(axis=1)
                if np.any(valid):
                    stacking_oof = np.full(Xn.shape[0], np.nan, dtype=float)
                    X_meta = np.column_stack(oof_cols)[valid]
                    stacking_oof[valid] = meta_model.predict_proba(X_meta)[:, 1]
                    oof_probs_for_viz["stacking_meta"] = stacking_oof

        tournament_model_bundle = next(
            (d.get("model") for d in completed if str(d.get("algorithm") or "") == "multi_armed_tournament"),
            None,
        )
        vis_ctx: Dict[str, Any] = {
            "run_id": run_id,
            "Xn": Xn,
            "y": y,
            "actual_returns": actual_returns,
            "sample_dates": sample_dates,
            "oof_probs": oof_probs_for_viz,
            "competitor_stats": (
                tournament_model_bundle.get("competitor_stats") if isinstance(tournament_model_bundle, dict) else {}
            ),
            "winner_name": (
                tournament_model_bundle.get("winner_name") if isinstance(tournament_model_bundle, dict) else ""
            ),
            "tp_barrier": float(runtime.get("tp_barrier", 2.0)),
            "sl_barrier": float(runtime.get("sl_barrier", 2.0)),
            "nlp_pipeline": trained_nlp_pipeline,
        }
        run_training_visualizations(vis_ctx, _model_store_dir())
    except Exception as _viz_exc:
        _ml_log(f"visualization generation skipped: {_viz_exc}")

    return MLTrainResponse(
        run_id=run_id,
        trained_models=infos,
        selected_model_id=selection.selected_model_id,
        deleted_underperforming=prune.deleted_model_ids,
    )
