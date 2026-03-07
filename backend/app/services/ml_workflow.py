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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .. import db
from ..ml.registry import get_meta_competitor_factories
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
from .news import get_news_for_date
from .symbols import resolve_symbol_name

try:
    import joblib
except Exception:
    joblib = None


ML_MODEL_MAX_COUNT = int(os.getenv("ML_MODEL_MAX_COUNT", "10"))
ML_SELECTED_TOP_K = int(os.getenv("ML_SELECTED_TOP_K", "3"))
WALK_FORWARD_SPLITS = int(os.getenv("ML_WALK_FORWARD_SPLITS", "5"))
WALK_FORWARD_MIN_TRAIN = int(os.getenv("ML_WALK_FORWARD_MIN_TRAIN", "120"))

TRIPLE_BARRIER_TP = float(os.getenv("ML_TP_BARRIER", "0.05"))
TRIPLE_BARRIER_SL = float(os.getenv("ML_SL_BARRIER", "0.03"))
TRIPLE_BARRIER_HOLD_DAYS = int(os.getenv("ML_TIME_BARRIER_DAYS", "20"))

FINBERT_MODEL_NAME = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
FINBERT_BATCH_SIZE = int(os.getenv("FINBERT_BATCH_SIZE", "16"))
FINBERT_RETRY_SECONDS = int(os.getenv("FINBERT_RETRY_SECONDS", "300"))
ML_ENFORCE_NEWS_COVERAGE = os.getenv("ML_ENFORCE_NEWS_COVERAGE", "1").strip().lower() in ("1", "true", "yes", "on")
ML_MAX_NEWS_MISSING_RATIO = float(os.getenv("ML_MAX_NEWS_MISSING_RATIO", "0.10"))


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
}


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


def _ml_log(message: str) -> None:
    enabled = os.getenv("ML_PIPELINE_VERBOSE", "1").strip().lower() not in ("0", "false", "no", "off")
    if not enabled:
        return
    ts = _utc_now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ml][{ts}] {message}", flush=True)


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


def _load_xgboost_classifier() -> Any:
    try:
        mod = importlib.import_module("xgboost")
        return getattr(mod, "XGBClassifier", None)
    except Exception:
        return None


def _load_lightgbm_classifier() -> Any:
    try:
        mod = importlib.import_module("lightgbm")
        return getattr(mod, "LGBMClassifier", None)
    except Exception:
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _iso_yesterday_utc() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()


def _model_store_dir() -> Path:
    root = os.getenv("MODEL_STORE_DIR", str(Path(__file__).resolve().parents[2] / "model_store"))
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in (s or "")).strip("_")


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
            for local_only in (True, False):
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


def _get_finbert() -> _FinBertInferencer:
    global _FINBERT
    if _FINBERT is None:
        _FINBERT = _FinBertInferencer()
    return _FINBERT


@dataclass
class _DailyNewsFeatures:
    symbol_sentiment: Dict[str, float]
    market_sentiment: float
    macro_features: List[float]
    news_item_count: int = 0


def _macro_term_features(texts: Sequence[str]) -> List[float]:
    merged = " ".join([str(t or "").lower() for t in texts])
    denom = float(max(1, len(texts)))
    vals: List[float] = []
    for term in MACRO_TERMS:
        vals.append(float(merged.count(term.lower())) / denom)
    return vals


def _news_text(item: Any) -> str:
    title = str(getattr(item, "title", "") or "")
    summary = str(getattr(item, "summary", "") or "")
    return f"{title}. {summary}".strip()


def _daily_news_features(date: str, symbols: Sequence[str]) -> _DailyNewsFeatures:
    syms = list(dict.fromkeys([str(s).upper().strip() for s in symbols if str(s).strip()]))
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

    finbert = _get_finbert()
    market_scores = finbert.score_texts(market_texts or all_texts)
    market_sent = float(np.mean(market_scores)) if market_scores else 0.0

    symbol_sent: Dict[str, float] = {}
    for sym in syms:
        s_scores = finbert.score_texts(symbol_texts.get(sym) or [])
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
    )


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


def _triple_barrier_label(closes: Sequence[float], idx: int, horizon: int, tp: float, sl: float) -> int:
    if idx < 0 or idx >= len(closes) - 1:
        return 0
    entry = float(closes[idx])
    if entry <= 0:
        return 0

    end = min(len(closes) - 1, idx + max(1, horizon))
    for j in range(idx + 1, end + 1):
        ret = (float(closes[j]) / entry) - 1.0
        if ret >= float(tp):
            return 1
        if ret <= -abs(float(sl)):
            return 0

    # Time barrier reached without TP/SL hit: fallback to sign of terminal return.
    terminal_ret = (float(closes[end]) / entry) - 1.0
    return 1 if terminal_ret >= 0 else 0


async def build_training_matrices(stock_basket: List[str], lookback_days: int, label_horizon_days: int) -> Dict[str, Any]:
    symbols = list(dict.fromkeys([(s or "").upper().strip() for s in stock_basket if (s or "").strip()]))
    if not symbols:
        return {
            "X_numeric": np.zeros((0, 15), dtype=float),
            "y": np.array([], dtype=int),
            "actual_returns": np.array([], dtype=float),
            "symbols": [],
            "sample_dates": [],
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
        daily_ctx[d] = _daily_news_features(d, symbols)

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
            "news_coverage": news_coverage,
        }

    return {
        "X_numeric": np.array(numeric_rows, dtype=float),
        "y": np.array(labels, dtype=int),
        "actual_returns": np.array(realized_returns, dtype=float),
        "symbols": symbols,
        "sample_dates": sample_dates,
        "news_coverage": news_coverage,
    }


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


def _weighted_score(metrics: Dict[str, float]) -> float:
    return float(
        (0.40 * metrics.get("f1", 0.0))
        + (0.30 * metrics.get("accuracy", 0.0))
        + (0.20 * metrics.get("precision", 0.0))
        + (0.10 * metrics.get("recall", 0.0))
    )


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


def _rank_models(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    for i, item in enumerate(ranked, start=1):
        item["rank"] = i
        item["underperforming"] = bool(item.get("metrics", {}).get("f1", 0.0) < 0.45)
    return ranked


def _artifact_path(model_id: str) -> Path:
    return _model_store_dir() / f"{_slug(model_id)}.joblib"


def _save_model_artifact(model_id: str, model_obj: Any) -> Optional[str]:
    if joblib is None:
        return None
    p = _artifact_path(model_id)
    joblib.dump(model_obj, p)
    return str(p)


def _delete_artifact(artifact_path: Optional[str]) -> None:
    if not artifact_path:
        return
    try:
        p = Path(artifact_path)
        if p.exists():
            p.unlink()
    except Exception:
        return


def _registry_col():
    return db.require_col(db.ml_models_col, "ml_models")


def _runs_col():
    return db.require_col(db.ml_model_runs_col, "ml_model_runs")


def _runtime_settings_col():
    return db.require_col(db.ml_runtime_settings_col, "ml_runtime_settings")


def get_ml_runtime_settings() -> Dict[str, Any]:
    col = _runtime_settings_col()
    doc = col.find_one({"key": "pipeline"}) or {}

    tp = float(doc.get("tp_barrier", _RUNTIME_SETTINGS_DEFAULTS["tp_barrier"]))
    sl = float(doc.get("sl_barrier", _RUNTIME_SETTINGS_DEFAULTS["sl_barrier"]))
    tb = int(doc.get("time_barrier_days", _RUNTIME_SETTINGS_DEFAULTS["time_barrier_days"]))
    splits = int(doc.get("walk_forward_splits", _RUNTIME_SETTINGS_DEFAULTS["walk_forward_splits"]))
    min_train = int(doc.get("walk_forward_min_train", _RUNTIME_SETTINGS_DEFAULTS["walk_forward_min_train"]))

    return {
        "tp_barrier": max(0.001, tp),
        "sl_barrier": max(0.001, sl),
        "time_barrier_days": max(1, tb),
        "walk_forward_splits": max(2, splits),
        "walk_forward_min_train": max(40, min_train),
    }


def update_ml_runtime_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    current = get_ml_runtime_settings()
    allowed = {
        "tp_barrier",
        "sl_barrier",
        "time_barrier_days",
        "walk_forward_splits",
        "walk_forward_min_train",
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


def _store_model_record(item: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    model_id = f"{item['algorithm']}_{uuid.uuid4().hex[:10]}"
    artifact_path = _save_model_artifact(model_id, item["model"])
    completeness = _data_completeness_from_news_coverage(item.get("news_coverage"))

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
        # Keep top-level ratio for quick filtering in admin/debug queries.
        "news_coverage_ratio": float(completeness["news_coverage_ratio"]),
        "metadata": {
            "data_completeness": completeness,
        },
        "created_at": _utc_now(),
    }

    col = _registry_col()
    col.update_one({"model_id": model_id}, {"$set": doc}, upsert=True)
    return doc


def _doc_to_model_info(doc: Dict[str, Any]) -> MLModelInfo:
    raw_feature = str(doc.get("feature_type") or "numeric").lower()
    feature_type: Literal["numeric", "text"] = "text" if raw_feature == "text" else "numeric"
    return MLModelInfo(
        model_id=str(doc.get("model_id") or ""),
        algorithm=str(doc.get("algorithm") or ""),
        family=str(doc.get("family") or "other"),
        feature_type=feature_type,
        metrics=MLModelMetric(**(doc.get("metrics") or {})),
        rank=int(doc.get("rank") or 0),
        score=float(doc.get("score") or 0.0),
        sample_count=int(doc.get("sample_count") or 0),
        is_selected=bool(doc.get("is_selected", False)),
        is_deployed=bool(doc.get("is_deployed", False)),
        underperforming=bool(doc.get("underperforming", False)),
        news_coverage_ratio=(
            float(doc.get("news_coverage_ratio")) if doc.get("news_coverage_ratio") is not None else None
        ),
        created_at=doc.get("created_at"),
    )


def list_models() -> List[MLModelInfo]:
    col = _registry_col()
    docs = list(col.find({}).sort([("rank", 1), ("score", -1), ("created_at", -1)]))
    out: List[MLModelInfo] = []
    for d in docs:
        d.pop("_id", None)
        out.append(_doc_to_model_info(d))
    return out


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

        # MetaLabeler winner: average importances across per-base RF models.
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
                vals = np.mean(mat, axis=0)

                n_base = min(len(base_names), int(vals.shape[0]))
                labels = [f"base_pred_{name}" for name in base_names[:n_base]]
                rem = int(vals.shape[0]) - n_base
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


def select_best_models(top_k: int = 1) -> MLSelectionResponse:
    col = _registry_col()
    docs = list(col.find({}).sort([("score", -1), ("created_at", -1)]))
    ranked = _rank_models(docs)

    col.update_many({}, {"$set": {"is_selected": False}})

    chosen = ranked[: max(1, int(top_k))]
    ids = [str(d.get("model_id")) for d in chosen if d.get("model_id")]
    if ids:
        col.update_many({"model_id": {"$in": ids}}, {"$set": {"is_selected": True}})

    return MLSelectionResponse(selected_model_id=(ids[0] if ids else None), ranked_model_ids=[str(d.get("model_id")) for d in ranked])


def _select_prune_candidates(docs: List[Dict[str, Any]], max_models: int) -> List[Dict[str, Any]]:
    if len(docs) < int(max_models):
        return []

    return [d for d in docs if not bool(d.get("is_selected", False))]


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


def _selected_or_latest_model(model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    col = _registry_col()
    if model_id:
        return col.find_one({"model_id": model_id})

    return col.find_one({"is_selected": True}, sort=[("score", -1), ("created_at", -1)])


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


def _algo_factories(random_seed: int) -> List[Tuple[str, Any]]:
    jobs: List[Tuple[str, Any]] = [
        ("ann_relu", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=700, random_state=random_seed)),
        ("ann_sigmoid", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), activation="logistic", max_iter=700, random_state=random_seed)),
        ("svm_rbf", lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=random_seed)),
        ("svm_poly", lambda: SVC(kernel="poly", degree=3, C=1.0, probability=True, random_state=random_seed)),
        ("random_forest", lambda: RandomForestClassifier(n_estimators=220, max_depth=8, random_state=random_seed, n_jobs=-1)),
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


def _safe_test_size(n_samples: int, req_test_size: float, n_splits: int) -> int:
    candidate = max(20, int(n_samples * max(0.10, min(0.35, float(req_test_size)))))
    max_allowed = max(10, (n_samples // max(2, n_splits + 1)) - 1)
    return max(10, min(candidate, max_allowed))


def _build_walk_forward_splits(
    n_samples: int,
    req_test_size: float,
    n_splits: int,
    min_train_samples: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples < 120:
        return []

    n_splits = max(2, int(n_splits))
    n_splits = min(n_splits, max(2, n_samples // 25))
    test_size = _safe_test_size(n_samples, req_test_size, n_splits)

    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_idx, te_idx in splitter.split(np.arange(n_samples)):
        if len(tr_idx) < int(min_train_samples):
            continue
        if len(te_idx) < 10:
            continue
        out.append((np.asarray(tr_idx, dtype=int), np.asarray(te_idx, dtype=int)))
    return out


def _fit_score_over_walk_forward(
    algorithm: str,
    factory: Any,
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    oof_collector: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_prob_all: List[float] = []

    for tr_idx, te_idx in splits:
        model = factory()
        model.fit(X[tr_idx], y[tr_idx])
        probs = _predict_proba_or_hard(model, X[te_idx])
        preds = np.where(probs >= 0.5, 1, 0)

        y_true_all.extend([int(v) for v in y[te_idx]])
        y_pred_all.extend([int(v) for v in preds])
        y_prob_all.extend([float(v) for v in probs])

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
    allocations: List[float] = []
    for i in range(base_matrix.shape[0]):
        alloc = float(combiner.allocate(base_matrix[i], X[i]))
        allocations.append(float(np.clip(alloc, -1.0, 1.0)))

    alloc_arr = np.asarray(allocations, dtype=float)
    return np.clip((alloc_arr + 1.0) / 2.0, 0.0, 1.0)


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

    allocations = np.asarray(
        [float(tournament_result.winner_model.allocate(base_oof[i], X_market[i])) for i in range(base_oof.shape[0])],
        dtype=float,
    )
    probs = np.clip((np.clip(allocations, -1.0, 1.0) + 1.0) / 2.0, 0.0, 1.0)
    preds = np.where(probs >= 0.5, 1, 0)
    metrics = _evaluate_binary(y_valid, preds, probs)

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


async def predict_for_basket(stock_basket: List[str], lookback_days: int = 120, model_id: Optional[str] = None) -> MLPredictionResponse:
    symbols = list(dict.fromkeys([(s or "").upper().strip() for s in stock_basket if (s or "").strip()]))
    items: List[MLPredictionItem] = []
    histories = await get_price_histories(symbols, days=max(30, int(lookback_days)), concurrency=8)
    today = iso_date_utc()
    day_ctx = _daily_news_features(today, symbols)

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
    selected_docs = list(col.find({"is_selected": True}).sort([("score", -1), ("created_at", -1)]))
    if not selected_docs:
        fallback = _selected_or_latest_model(model_id=None)
        if fallback:
            selected_docs = [fallback]
    selected_docs = selected_docs[: max(1, int(ML_SELECTED_TOP_K))]

    model_probs: List[np.ndarray] = []
    used_ids: List[str] = []

    for d in selected_docs:
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
        raise RuntimeError("No selected model is available. Select the best model before prediction.")

    probs = np.mean(np.column_stack(model_probs), axis=1)

    preds = np.where(probs >= 0.5, 1, 0)
    for i, sym in enumerate(valid_symbols):
        p_up = float(probs[i]) if i < len(probs) else (1.0 if int(preds[i]) == 1 else 0.0)
        items.append(MLPredictionItem(symbol=sym, prediction=("up" if int(preds[i]) == 1 else "down"), probability_up=max(0.0, min(1.0, p_up))))

    mix_id = ",".join([x for x in used_ids if x])
    return MLPredictionResponse(model_id=(f"mix[{mix_id}]" if mix_id else "mix[selected]"), items=items)


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

    try:
        dataset = await build_training_matrices(
            stock_basket=req.stock_basket,
            lookback_days=req.lookback_days,
            label_horizon_days=req.label_horizon_days,
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

    _ml_log(f"stage: dataset ready | samples={Xn.shape[0]} | labels={len(y)}")
    runs.update_one(
        {"run_id": run_id},
        {
            "$set": {
                "stage": "dataset_ready",
                "dataset_samples": int(Xn.shape[0]),
                "news_coverage": news_coverage,
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

    for name, factory in base_jobs:
        t0 = time.perf_counter()
        try:
            res = _fit_score_over_walk_forward(name, factory, Xn, y, splits)
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
                },
            }
        },
    )

    pipeline_elapsed = time.perf_counter() - pipeline_t0
    _ml_log(
        f"pipeline completed | run_id={run_id} | total_time={pipeline_elapsed:.2f}s | trained={len(infos)} | selected={selection.selected_model_id or 'none'} | pruned={len(prune.deleted_model_ids)}"
    )

    return MLTrainResponse(
        run_id=run_id,
        trained_models=infos,
        selected_model_id=selection.selected_model_id,
        deleted_underperforming=prune.deleted_model_ids,
    )
