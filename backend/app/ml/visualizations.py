"""
ML training and startup visualizations.

All public functions return the saved file path (str) or None on failure.
Every function is wrapped in broad try/except so they never interrupt training
or startup.  matplotlib is imported lazily so a missing install does not break
the rest of the application.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Colour palette
_ALGO_COLOURS: Dict[str, str] = {
    "ann_relu":          "#4C9BE8",
    "ann_sigmoid":       "#2170C4",
    "svm_rbf":           "#F5A623",
    "svm_poly":          "#D4870D",
    "random_forest":     "#2ECC71",
    "gradient_boosting": "#1A8A4D",
    "xgboost":           "#E84C4C",
    "lightgbm":          "#9B59B6",
    "stacking_meta":     "#C0392B",   # bold distinct red for the meta-learner
    "logistic_regression": "#C0392B",
}

_TYPE_COLOURS: Dict[str, str] = {
    "meta_labeler":    "#4C9BE8",
    "regime_switcher": "#F5A623",
    "rl_agent":        "#E84C4C",
    "grand_ensemble":  "#2ECC71",
}

# Internal helpers 

def _charts_dir(base_dir: Path, chart_type: str) -> Path:
    p = base_dir / "visualizations" / chart_type
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _mpl():
    """Lazily import matplotlib and return (matplotlib, pyplot, patches) or (None, None, None)."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless – must come before pyplot import
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return matplotlib, plt, mpatches
    except ImportError:
        return None, None, None


def _ewma_1d(arr: np.ndarray, span: int = 10) -> np.ndarray:
    """Exponentially weighted moving average of a 1-D array."""
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty_like(arr, dtype=float)
    if arr.size == 0:
        return out
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * float(arr[i]) + (1.0 - alpha) * out[i - 1]
    return out


# Chart 1 – Triple Barrier Labeling

def save_triple_barrier_chart(
    actual_returns: np.ndarray,
    y_true: np.ndarray,
    sample_dates: List[str],
    tp_barrier_mult: float,
    sl_barrier_mult: float,
    run_id: str,
    base_dir: Path,
    *,
    max_samples: int = 100,
) -> Optional[str]:
    """
    Candlestick-style chart: synthetic cumulative-price series with dynamic
    TP / SL barrier bands and time-barrier markers.

    Labels:  green ▲ = hit TP first (label=1),  red ▼ = hit SL / time (label=0).
    """
    try:
        _, plt, mpatches = _mpl()
        if plt is None:
            return None

        out = _charts_dir(base_dir, "triple_barrier")
        n = min(max_samples, len(actual_returns))
        if n < 5:
            return None

        returns = np.asarray(actual_returns[:n], dtype=float)
        labels = np.asarray(y_true[:n], dtype=int)
        dates = sample_dates[:n]

        # Reconstruct a synthetic normalised price series
        safe_rets = np.clip(returns, -0.5, 0.5)
        prices = np.cumprod(1.0 + safe_rets)
        prices = prices / float(prices[0]) * 100.0

        # Rolling 20-day realised-volatility for dynamic barrier width
        sigma = np.full(n, 1e-4, dtype=float)
        for i in range(1, n):
            window = safe_rets[max(0, i - 20) : i]
            if len(window) >= 2:
                sigma[i] = max(1e-4, float(np.std(window, ddof=1)))
            else:
                sigma[i] = sigma[i - 1]

        tp_band = prices * (1.0 + tp_barrier_mult * sigma)
        sl_band = prices * (1.0 - sl_barrier_mult * sigma)

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(n)

        # Shaded corridor between TP and SL
        ax.fill_between(x, tp_band, sl_band, alpha=0.10, color="#4C9BE8", label="Barrier corridor")
        # Coloured edges
        ax.fill_between(x, tp_band, tp_band * 1.001, alpha=0.35, color="#2ECC71")
        ax.fill_between(x, sl_band * 0.999, sl_band, alpha=0.35, color="#E84C4C")

        ax.plot(x, prices,  color="#1A1A2E", lw=1.3, zorder=4, label="Synthetic price")
        ax.plot(x, tp_band, color="#2ECC71", lw=1.0, ls="--", label=f"TP ({tp_barrier_mult:.2g}σ)")
        ax.plot(x, sl_band, color="#E84C4C", lw=1.0, ls="--", label=f"SL ({sl_barrier_mult:.2g}σ)")

        # Per-sample label markers
        up_x   = x[labels == 1]
        down_x = x[labels == 0]
        ax.scatter(up_x,   prices[labels == 1], c="#2ECC71", marker="^", s=28, zorder=5, alpha=0.85, label="Label 1 (TP hit)")
        ax.scatter(down_x, prices[labels == 0], c="#E84C4C", marker="v", s=28, zorder=5, alpha=0.85, label="Label 0 (SL/time)")

        # X-tick date labels
        tick_step = max(1, n // 10)
        tick_idx = list(range(0, n, tick_step))
        ax.set_xticks(tick_idx)
        if dates:
            ax.set_xticklabels(
                [dates[i] if i < len(dates) else "" for i in tick_idx],
                rotation=38, ha="right", fontsize=7,
            )

        ax.set_xlabel("Sample index")
        ax.set_ylabel("Normalised price (rebased to 100)")
        ax.set_title(
            f"Triple Barrier Labeling  |  TP={tp_barrier_mult:.2g}σ   SL={sl_barrier_mult:.2g}σ   [{run_id}]"
        )
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        ax.grid(axis="y", alpha=0.25)

        fname = out / f"triple_barrier_{run_id}_{_ts()}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.debug("triple_barrier saved → %s", fname)
        return str(fname)

    except Exception:
        logger.debug("save_triple_barrier_chart failed", exc_info=True)
        return None


# Chart 2 – ROC Curves (base classifiers + stacking meta-learner)

def save_roc_curves_chart(
    oof_probs_by_algo: Dict[str, np.ndarray],
    y_true: np.ndarray,
    run_id: str,
    base_dir: Path,
) -> Optional[str]:
    """
    One faint ROC curve per base classifier.
    The stacking / logistic-regression meta-learner is plotted bold and labelled.
    """
    try:
        from sklearn.metrics import roc_curve, auc as sk_auc  # sklearn guaranteed present
    except ImportError:
        return None

    try:
        _, plt, _ = _mpl()
        if plt is None:
            return None

        out = _charts_dir(base_dir, "roc_curves")
        y = np.asarray(y_true, dtype=int)
        if y.size < 10 or len(np.unique(y)) < 2:
            return None

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.35, label="Random baseline")

        for algo, probs in sorted(oof_probs_by_algo.items()):
            p = np.asarray(probs, dtype=float)
            if p.shape[0] != y.shape[0]:
                continue
            # Fill OOF holes (samples without test-fold coverage) with prior 0.5
            p = np.where(np.isnan(p), 0.5, p)
            try:
                fpr, tpr, _ = roc_curve(y, p)
                roc_auc = float(sk_auc(fpr, tpr))
            except Exception:
                continue

            is_meta = any(kw in algo.lower() for kw in ("stacking", "meta", "logistic"))
            colour = _ALGO_COLOURS.get(algo.lower(), "#888888")

            if is_meta:
                ax.plot(fpr, tpr, lw=2.8, color=colour,
                        label=f"{algo}  AUC={roc_auc:.3f}  ★", zorder=6)
            else:
                ax.plot(fpr, tpr, lw=1.2, color=colour, alpha=0.65,
                        label=f"{algo}  AUC={roc_auc:.3f}")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves – Base Classifiers & Stacking Meta-Learner  [{run_id}]")
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.88)
        ax.grid(alpha=0.20)

        fname = out / f"roc_curves_{run_id}_{_ts()}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.debug("roc_curves saved → %s", fname)
        return str(fname)

    except Exception:
        logger.debug("save_roc_curves_chart failed", exc_info=True)
        return None


# Chart 3 – RegimeSwitcher: dual-axis volatility & softmax-weight area chart

def save_regime_switcher_chart(
    Xn: np.ndarray,
    actual_returns: np.ndarray,
    run_id: str,
    base_dir: Path,
    *,
    vol_feature_idx: int = 3,
    max_samples: int = 200,
) -> Optional[str]:
    """
    Left  Y-axis : Market volatility (z-score over rolling window).
    Right Y-axis : Softmax weights for High-vol vs Low-vol regime (0–100 %).
    Filled area shows which regime dominates at each time step.
    """
    try:
        _, plt, _ = _mpl()
        if plt is None:
            return None

        from .regime_switcher import RegimeSwitcherCombiner

        out = _charts_dir(base_dir, "regime_switcher")
        n = min(max_samples, int(Xn.shape[0]))
        if n < 10:
            return None

        Xs = np.asarray(Xn[:n], dtype=float)
        rets = np.asarray(actual_returns[:n], dtype=float)

        # Train a representative RegimeSwitcher purely for visualisation
        n_bases = 4
        dummy_preds = np.full((n, n_bases), 0.5, dtype=float)
        rs = RegimeSwitcherCombiner(vol_feature_idx=vol_feature_idx)
        rs.train(dummy_preds, Xs, rets)

        # Extract the raw volatility feature column
        v_idx = min(vol_feature_idx, Xs.shape[1] - 1)
        vols = Xs[:, v_idx].astype(float)
        roll_mean = float(rs._vol_rolling_mean)
        roll_std  = float(rs._vol_rolling_std) if float(rs._vol_rolling_std) > 1e-8 else 1.0
        z_vols = (vols - roll_mean) / roll_std

        # Compute regime scores and derive EWMA-smoothed softmax weights per step
        regime_scores = np.array(
            [rs._calculate_regime_score(float(v)) for v in vols], dtype=float
        )
        ewma_high = _ewma_1d(regime_scores, span=20)
        ewma_low  = 1.0 - ewma_high

        x = np.arange(n)

        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()

        # Left axis – volatility area
        ax1.fill_between(
            x, z_vols, 0,
            where=z_vols >= 0, alpha=0.22, color="#E84C4C",
            label="High-vol region",
        )
        ax1.fill_between(
            x, z_vols, 0,
            where=z_vols < 0, alpha=0.22, color="#4C9BE8",
            label="Low-vol region",
        )
        ax1.plot(x, z_vols, color="#555555", lw=0.9, alpha=0.5)
        ax1.axhline(0, color="#555555", lw=0.8, ls="--", alpha=0.4)
        ax1.set_ylabel("Volatility (z-score)", color="#555555", fontsize=9)
        ax1.tick_params(axis="y", labelcolor="#555555")

        # Right axis – softmax weight lines
        ax2.plot(x, ewma_high * 100, color="#E84C4C", lw=2.0, label="High-vol weight %")
        ax2.plot(x, ewma_low  * 100, color="#4C9BE8", lw=2.0, label="Low-vol weight %")
        ax2.fill_between(x, ewma_high * 100, ewma_low * 100,
                         where=ewma_high >= ewma_low, alpha=0.08, color="#E84C4C")
        ax2.fill_between(x, ewma_high * 100, ewma_low * 100,
                         where=ewma_high < ewma_low, alpha=0.08, color="#4C9BE8")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Softmax weight (%)", color="#999999", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#999999")

        ax1.set_xlabel("Sample index")
        ax1.set_title(
            f"RegimeSwitcher – Volatility Regime & Softmax Ensemble Weights  [{run_id}]"
        )
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8, framealpha=0.88)
        ax1.grid(alpha=0.18)

        fname = out / f"regime_switcher_{run_id}_{_ts()}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.debug("regime_switcher saved → %s", fname)
        return str(fname)

    except Exception:
        logger.debug("save_regime_switcher_chart failed", exc_info=True)
        return None


# Chart 4 – RLAgent: Q-value learning trajectories & reward curve

def save_rl_agent_chart(
    Xn: np.ndarray,
    actual_returns: np.ndarray,
    run_id: str,
    base_dir: Path,
    *,
    n_dummy_bases: int = 4,
    max_samples: int = 200,
) -> Optional[str]:
    """
    Top panel : per-base-model Q-value trajectories (EWMA-smoothed).
    Bottom panel : EWMA reward trajectory with green/red fill.
    """
    try:
        _, plt, _ = _mpl()
        if plt is None:
            return None

        from .rl_agent import RLAgentCombiner

        out = _charts_dir(base_dir, "rl_agent")
        n = min(max_samples, int(Xn.shape[0]))
        if n < 10:
            return None

        Xs   = np.asarray(Xn[:n], dtype=float)
        rets = np.asarray(actual_returns[:n], dtype=float)
        n_bases = n_dummy_bases

        rl = RLAgentCombiner(learning_rate=0.05, temperature=0.8)
        # Infer theta dimensions from the first state vector
        state0 = rl._state_vector(Xs[0])
        rl._ensure_theta(n_bases, state0.size)

        dummy_preds = np.full((n, n_bases), 0.5, dtype=float)
        q_history: List[np.ndarray] = []
        reward_history: List[float] = []
        prev_signal = np.zeros(n_bases, dtype=float)

        for t in range(n):
            state_t = rl._state_vector(Xs[t])
            rl._ensure_theta(n_bases, state_t.size)
            q_history.append(rl._q_values(state_t).copy())

            signal_t = (2.0 * dummy_preds[t]) - 1.0
            turnover = np.abs(signal_t - prev_signal)
            reward_t = signal_t * float(rets[t]) - rl.reward_cost_rate * turnover
            reward_history.append(float(np.mean(reward_t)))

            # TD(0) update
            if t + 1 < n:
                state_nt = rl._state_vector(Xs[t + 1])
                rl._ensure_theta(n_bases, state_nt.size)
                q_next = rl._q_values(state_nt)
                for i in range(n_bases):
                    future = float(rl.gamma) * (float(q_next[i]) if q_next.size > i else 0.0)
                    td_target = float(reward_t[i]) + future
                    td_error  = td_target - float(rl._theta[i] @ state_t)
                    rl._theta[i] += rl.learning_rate * td_error * state_t

            prev_signal = signal_t

        q_arr      = np.array(q_history, dtype=float)       # (n, n_bases)
        reward_arr = np.array(reward_history, dtype=float)

        reward_ewma = _ewma_1d(reward_arr, span=10)
        x = np.arange(n)

        COLOURS = ["#4C9BE8", "#F5A623", "#E84C4C", "#2ECC71", "#9B59B6", "#1A8A4D"]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"hspace": 0.38},
        )

        # Top: Q-value trajectories (EWMA-smoothed per base model)
        for i in range(min(n_bases, len(COLOURS))):
            q_smooth = _ewma_1d(q_arr[:, i], span=10)
            ax1.plot(x, q_smooth, color=COLOURS[i], lw=1.6, label=f"Base {i + 1}")
        ax1.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax1.set_ylabel("Q-value (EWMA-smoothed)", fontsize=9)
        ax1.set_title(f"RLAgent – Q-value Learning Trajectories  [{run_id}]")
        ax1.legend(fontsize=8, loc="upper left", framealpha=0.88)
        ax1.grid(alpha=0.22)

        # Bottom: reward trajectory
        ax2.fill_between(x, reward_ewma, 0,
                         where=reward_ewma >= 0, alpha=0.35, color="#2ECC71",
                         label="Positive reward")
        ax2.fill_between(x, reward_ewma, 0,
                         where=reward_ewma < 0, alpha=0.35, color="#E84C4C",
                         label="Negative reward")
        ax2.plot(x, reward_ewma, color="#1A1A2E", lw=1.2)
        ax2.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
        ax2.set_xlabel("Training step")
        ax2.set_ylabel("Reward (EWMA-smoothed)", fontsize=9)
        ax2.set_title("EWMA Reward Trajectory (green=profit, red=loss+cost)")
        ax2.legend(fontsize=8, loc="upper left", framealpha=0.88)
        ax2.grid(alpha=0.22)

        fname = out / f"rl_agent_{run_id}_{_ts()}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.debug("rl_agent saved → %s", fname)
        return str(fname)

    except Exception:
        logger.debug("save_rl_agent_chart failed", exc_info=True)
        return None


# Chart 5 – Tournament scatter: Risk vs Return for all 22 competitors

def save_tournament_scatter_chart(
    competitor_stats: Dict[str, Dict[str, float]],
    winner_name: str,
    run_id: str,
    base_dir: Path,
) -> Optional[str]:
    """
    Scatter plot:
      X-axis : Max Drawdown (%)
      Y-axis : Mean Return (%)
      Dot size : |Sharpe ratio|
      Colour   : competitor family (MetaLabeler=blue, RegimeSwitcher=orange,
                 RLAgent=red, GrandEnsemble=green)
      Winner marked with a gold star.
    """
    try:
        _, plt, mpatches = _mpl()
        if plt is None or mpatches is None or not competitor_stats:
            return None

        out = _charts_dir(base_dir, "tournament")

        fig, ax = plt.subplots(figsize=(11, 8))
        legend_handles = []

        for name, stats in competitor_stats.items():
            sharpe    = float(stats.get("sharpe", 0.0) or 0.0)
            drawdown  = abs(float(stats.get("max_drawdown", 0.0) or 0.0))
            mean_ret  = float(stats.get("mean_return", 0.0) or 0.0)

            nl = name.lower()
            if "meta_label" in nl or "metalabel" in nl:
                ctype = "meta_labeler"
            elif "regime" in nl:
                ctype = "regime_switcher"
            elif "rl_agent" in nl or "rlagent" in nl:
                ctype = "rl_agent"
            elif "grand" in nl or "ensemble" in nl:
                ctype = "grand_ensemble"
            else:
                ctype = "meta_labeler"

            colour   = _TYPE_COLOURS.get(ctype, "#888888")
            is_winner = name == winner_name
            dot_size  = max(45.0, min(900.0, abs(sharpe) * 220.0 + 55.0))
            marker    = "*" if is_winner else "o"
            edge_col  = "#FFD700" if is_winner else colour
            lw        = 2.2 if is_winner else 0.6

            ax.scatter(
                drawdown * 100, mean_ret * 100,
                s=dot_size, c=colour, marker=marker,
                edgecolors=edge_col, linewidths=lw,
                alpha=0.87, zorder=5,
            )
            ax.annotate(
                name[:20],
                (drawdown * 100, mean_ret * 100),
                textcoords="offset points", xytext=(4, 3),
                fontsize=5.5, alpha=0.75,
            )

        # Legend patches
        for ctype, colour in _TYPE_COLOURS.items():
            legend_handles.append(
                mpatches.Patch(color=colour, label=ctype.replace("_", " ").title())
            )
        if winner_name:
            legend_handles.append(
                mpatches.Patch(
                    facecolor="white", edgecolor="#FFD700", linewidth=2,
                    label=f"Winner ★ {winner_name}",
                )
            )

        ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.35)
        ax.axvline(0, color="gray", lw=0.7, ls="--", alpha=0.35)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.88)
        ax.set_xlabel("Max Drawdown (%)", fontsize=10)
        ax.set_ylabel("Mean Return (%)", fontsize=10)
        ax.set_title(
            f"Walk-Forward Tournament – Risk vs Return  (size ∝ |Sharpe|)  [{run_id}]"
        )
        ax.grid(alpha=0.20)

        fname = out / f"tournament_{run_id}_{_ts()}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.debug("tournament scatter saved → %s", fname)
        return str(fname)

    except Exception:
        logger.debug("save_tournament_scatter_chart failed", exc_info=True)
        return None


# Chart 6 – NLP Sentiment diverging bar chart (FinBERT vs Lexicon)

_SENTIMENT_SAMPLE_TEXTS: List[str] = [
    "Earnings beat expectations, strong revenue growth reported",
    "Company faces severe regulatory fines and legal challenges",
    "Stock market reaches all-time high amid strong economic data",
    "Federal Reserve signals interest rate cuts ahead",
    "Mass layoffs announced as firm restructures amid losses",
    "Analyst upgrades stock to buy with high price target",
    "Supply chain disruptions affect quarterly earnings outlook",
    "Record quarterly profit driven by robust consumer demand",
    "Credit rating downgraded due to mounting debt concerns",
    "Merger and acquisition deal completed at premium valuation",
    "Product recall issued amid safety concerns for customers",
    "Dividend increased, signaling confidence in future cash flows",
    "CEO resignation sparks uncertainty about strategic direction",
    "New product launch exceeds sales forecasts across regions",
    "Inflation data higher than expected, markets sell off sharply",
]


def _lexicon_score(text: str) -> float:
    """Simple word-count lexicon fallback (no external imports required)."""
    _POS = {"beat", "growth", "high", "profit", "buy", "dividend", "launch", "premium",
            "strong", "record", "upgrade", "confidence", "robust", "exceeds", "increase"}
    _NEG = {"fines", "challenges", "layoffs", "losses", "disruptions", "downgraded",
            "recall", "resignation", "concerns", "sharply", "uncertainty", "severe"}
    words = set(text.lower().split())
    pos = len(words & _POS)
    neg = len(words & _NEG)
    denom = max(1, pos + neg)
    return float(pos - neg) / denom


def save_sentiment_chart(
    run_id: str,
    base_dir: Path,
    *,
    sample_texts: Optional[List[str]] = None,
    nlp_pipeline: str = "lexicon",
) -> Optional[str]:
    """
    Diverging horizontal bar chart:
      Left panel  : Lexicon scores
      Right panel : FinBERT (or active-pipeline) scores
    Red bars extend left (negative), green bars right (positive).
    """
    try:
        _, plt, _ = _mpl()
        if plt is None:
            return None

        out = _charts_dir(base_dir, "sentiment")
        texts = list(sample_texts or _SENTIMENT_SAMPLE_TEXTS)[:20]
        n = len(texts)
        if n == 0:
            return None

        # --- Lexicon scores (always available) ---
        try:
            from ..services.ml_workflow import _score_texts_with_pipeline
            lex_scores = _score_texts_with_pipeline(texts, "lexicon")
        except Exception:
            lex_scores = [_lexicon_score(t) for t in texts]

        # --- FinBERT / active-pipeline scores ---
        try:
            from ..services.ml_workflow import _score_texts_with_pipeline
            target_pipeline = nlp_pipeline if nlp_pipeline != "lexicon" else "finbert"
            fb_scores = _score_texts_with_pipeline(texts, target_pipeline)
            fb_label = target_pipeline.capitalize()
        except Exception:
            fb_scores = list(lex_scores)
            fb_label = "Lexicon (fallback)"

        lex_arr = np.clip(np.array(lex_scores[:n], dtype=float), -1.0, 1.0)
        fb_arr  = np.clip(np.array(fb_scores[:n],  dtype=float), -1.0, 1.0)

        short_labels = [t[:40] + ("…" if len(t) > 40 else "") for t in texts]
        y = np.arange(n)

        fig, axes = plt.subplots(
            1, 2,
            figsize=(15, max(6, n * 0.44)),
            sharey=True,
        )

        for ax, scores, title in zip(
            axes,
            [lex_arr, fb_arr],
            ["Lexicon", fb_label],
        ):
            colours = ["#2ECC71" if s >= 0 else "#E84C4C" for s in scores]
            ax.barh(y, scores, color=colours, edgecolor="white",
                    linewidth=0.5, height=0.70)
            ax.axvline(0, color="#1A1A2E", lw=1.1)
            ax.set_xlim(-1.15, 1.15)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Sentiment score  (–1 = negative, +1 = positive)", fontsize=8)
            ax.grid(axis="x", alpha=0.20)

        axes[0].set_yticks(y)
        axes[0].set_yticklabels(short_labels, fontsize=7.5)

        fig.suptitle(
            f"NLP Sentiment Comparison – Lexicon vs {fb_label}  [{run_id}]",
            fontsize=11, y=1.02,
        )

        fname = out / f"sentiment_{run_id}_{_ts()}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.debug("sentiment chart saved → %s", fname)
        return str(fname)

    except Exception:
        logger.debug("save_sentiment_chart failed", exc_info=True)
        return None


# Orchestrators

def run_training_visualizations(ctx: Dict[str, Any], base_dir: Path) -> Dict[str, Optional[str]]:
    """
    Called from train_models_async() after training is complete.

    Expected keys in *ctx*:
        run_id, Xn, y, actual_returns, sample_dates,
        oof_probs  (dict[algo_name -> np.ndarray of shape (n_samples,)]),
        competitor_stats, winner_name,
        tp_barrier, sl_barrier, nlp_pipeline
    """
    results: Dict[str, Optional[str]] = {}
    run_id = str(ctx.get("run_id") or "unknown")

    try:
        results["triple_barrier"] = save_triple_barrier_chart(
            actual_returns=np.asarray(ctx.get("actual_returns") or [], dtype=float),
            y_true=np.asarray(ctx.get("y") or [], dtype=int),
            sample_dates=list(ctx.get("sample_dates") or []),
            tp_barrier_mult=float(ctx.get("tp_barrier") or 2.0),
            sl_barrier_mult=float(ctx.get("sl_barrier") or 2.0),
            run_id=run_id,
            base_dir=base_dir,
        )
    except Exception:
        results["triple_barrier"] = None

    try:
        results["roc_curves"] = save_roc_curves_chart(
            oof_probs_by_algo=dict(ctx.get("oof_probs") or {}),
            y_true=np.asarray(ctx.get("y") or [], dtype=int),
            run_id=run_id,
            base_dir=base_dir,
        )
    except Exception:
        results["roc_curves"] = None

    try:
        Xn_raw = ctx.get("Xn")
        ar_raw = ctx.get("actual_returns")
        if Xn_raw is not None and len(Xn_raw) > 10:
            results["regime_switcher"] = save_regime_switcher_chart(
                Xn=np.asarray(Xn_raw, dtype=float),
                actual_returns=np.asarray(ar_raw or [], dtype=float),
                run_id=run_id,
                base_dir=base_dir,
            )
        else:
            results["regime_switcher"] = None
    except Exception:
        results["regime_switcher"] = None

    try:
        Xn_raw = ctx.get("Xn")
        ar_raw = ctx.get("actual_returns")
        if Xn_raw is not None and len(Xn_raw) > 10:
            results["rl_agent"] = save_rl_agent_chart(
                Xn=np.asarray(Xn_raw, dtype=float),
                actual_returns=np.asarray(ar_raw or [], dtype=float),
                run_id=run_id,
                base_dir=base_dir,
            )
        else:
            results["rl_agent"] = None
    except Exception:
        results["rl_agent"] = None

    try:
        results["tournament"] = save_tournament_scatter_chart(
            competitor_stats=dict(ctx.get("competitor_stats") or {}),
            winner_name=str(ctx.get("winner_name") or ""),
            run_id=run_id,
            base_dir=base_dir,
        )
    except Exception:
        results["tournament"] = None

    try:
        results["sentiment"] = save_sentiment_chart(
            run_id=run_id,
            base_dir=base_dir,
            nlp_pipeline=str(ctx.get("nlp_pipeline") or "lexicon"),
        )
    except Exception:
        results["sentiment"] = None

    saved = [k for k, v in results.items() if v is not None]
    logger.info("Training visualizations saved: %s | run_id=%s", saved, run_id)
    return results


def run_startup_visualizations(base_dir: Path) -> Dict[str, Optional[str]]:
    """
    Called from main.py lifespan at application start-up.

    Generates the tournament scatter (from the stored selected model) and the
    NLP sentiment chart (using fixed sample texts).  Both are cheap and require
    no retraining.
    """
    results: Dict[str, Optional[str]] = {}
    run_id = f"startup_{_ts()}"

    # Tournament scatter – requires a trained tournament model in the registry
    try:
        from ..services.ml_workflow import get_tournament_competitor_stats
        stats_info = get_tournament_competitor_stats()
        competitor_stats = dict(stats_info.get("competitor_stats") or {})
        winner_name = str(stats_info.get("winner_name") or "")

        if competitor_stats:
            results["tournament"] = save_tournament_scatter_chart(
                competitor_stats=competitor_stats,
                winner_name=winner_name,
                run_id=run_id,
                base_dir=base_dir,
            )
        else:
            results["tournament"] = None
    except Exception:
        results["tournament"] = None

    # Sentiment chart – uses fixed sample sentences, always runnable
    try:
        results["sentiment"] = save_sentiment_chart(
            run_id=run_id,
            base_dir=base_dir,
        )
    except Exception:
        results["sentiment"] = None

    saved = [k for k, v in results.items() if v is not None]
    logger.info("Startup visualizations saved: %s", saved)
    return results
