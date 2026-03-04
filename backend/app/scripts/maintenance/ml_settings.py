from __future__ import annotations

import argparse
import json

from ... import db
from ...services.ml_workflow import get_ml_runtime_settings, update_ml_runtime_settings


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="View or update ML runtime settings")
    p.add_argument("--show", action="store_true", help="Print effective settings")
    p.add_argument("--tp-barrier", type=float, default=None, help="Take-profit barrier, e.g. 0.05")
    p.add_argument("--sl-barrier", type=float, default=None, help="Stop-loss barrier, e.g. 0.03")
    p.add_argument("--time-barrier-days", type=int, default=None, help="Maximum holding period in days")
    p.add_argument("--walk-forward-splits", type=int, default=None, help="Number of time-series CV splits")
    p.add_argument("--walk-forward-min-train", type=int, default=None, help="Minimum training rows per fold")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    db.init_db()
    try:
        updates = {
            "tp_barrier": args.tp_barrier,
            "sl_barrier": args.sl_barrier,
            "time_barrier_days": args.time_barrier_days,
            "walk_forward_splits": args.walk_forward_splits,
            "walk_forward_min_train": args.walk_forward_min_train,
        }
        updates = {k: v for k, v in updates.items() if v is not None}

        if updates:
            out = update_ml_runtime_settings(updates)
        else:
            out = get_ml_runtime_settings()

        print(json.dumps(out, indent=2))
    finally:
        db.close_db()


if __name__ == "__main__":
    main()
