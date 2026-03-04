# backend/app/services/scheduler.py
from __future__ import annotations

import asyncio
import os

from app.models import MLTrainingRequest
from app.services.ml_workflow import (
    deploy_neural_network,
    prune_underperforming_models,
    select_best_models,
    train_models_async,
)
from app.services.signals import generate_daily_signals


def _env_true(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _default_ml_basket() -> list[str]:
    raw = os.getenv(
        "SCHEDULER_ML_BASKET",
        "SPY,AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,UNH,XOM,TSLA,AVGO,AMD,NFLX,GS",
    )
    return [s.upper().strip() for s in raw.split(",") if s.strip()]


def run_daily_ml(stock_basket: list[str] | None = None) -> None:
    """
    Daily ML workflow hook:
    1) train algorithms concurrently
    2) rank/select best model
    3) deploy best neural-network model
    4) prune underperforming models
    """
    basket = stock_basket or _default_ml_basket()

    req = MLTrainingRequest(
        stock_basket=basket,
        lookback_days=int(os.getenv("SCHEDULER_ML_LOOKBACK_DAYS", "240")),
        label_horizon_days=int(os.getenv("SCHEDULER_ML_LABEL_HORIZON_DAYS", "5")),
        test_size=float(os.getenv("SCHEDULER_ML_TEST_SIZE", "0.25")),
        random_seed=int(os.getenv("SCHEDULER_ML_RANDOM_SEED", "42")),
        run_async=True,
    )

    try:
        asyncio.run(train_models_async(req))
    except Exception:
        return

    try:
        select_best_models(top_k=1)
    except Exception:
        pass

    try:
        asyncio.run(deploy_neural_network(model_id=None))
    except Exception:
        pass

    try:
        prune_underperforming_models(
            min_f1=float(os.getenv("SCHEDULER_ML_MIN_F1", "0.40")),
            min_accuracy=float(os.getenv("SCHEDULER_ML_MIN_ACCURACY", "0.40")),
        )
    except Exception:
        pass


def run_daily() -> None:
    """
    Daily refresh entry point.
    Always refresh dashboard signals.
    Optionally run ML workflow when ENABLE_ML_DAILY=1.
    """
    generate_daily_signals()

    if _env_true("ENABLE_ML_DAILY", "0"):
        run_daily_ml()
