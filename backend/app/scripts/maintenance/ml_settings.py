from __future__ import annotations

import argparse
import json

from ... import db
from ...services.ml_workflow import get_ml_runtime_settings, update_ml_runtime_settings


def _build_parser() -> argparse.ArgumentParser:
    # Define CLI flags for reading/updating ML runtime and CV settings.
    p = argparse.ArgumentParser(description="View or update ML runtime settings")
    p.add_argument("--show", action="store_true", help="Print effective settings")
    p.add_argument("--tp-barrier", type=float, default=None, help="Take-profit volatility multiplier, e.g. 2.0")
    p.add_argument("--sl-barrier", type=float, default=None, help="Stop-loss volatility multiplier, e.g. 1.5")
    p.add_argument("--time-barrier-days", type=int, default=None, help="Maximum holding period in days")
    p.add_argument("--walk-forward-splits", type=int, default=None, help="Number of time-series CV splits")
    p.add_argument("--walk-forward-min-train", type=int, default=None, help="Minimum training rows per fold")
    p.add_argument("--walk-forward-purge-days", type=int, default=None, help="Embargo/purge gap between train end and test start")
    return p


def main() -> None:
    # Parse command-line arguments provided by the operator.
    args = _build_parser().parse_args()
    # Initialize DB connection used by settings service functions.
    db.init_db()
    try:
        # Collect only explicitly provided settings for partial updates.
        updates = {
            "tp_barrier": args.tp_barrier,
            "sl_barrier": args.sl_barrier,
            "time_barrier_days": args.time_barrier_days,
            "walk_forward_splits": args.walk_forward_splits,
            "walk_forward_min_train": args.walk_forward_min_train,
            "walk_forward_purge_days": args.walk_forward_purge_days,
        }
        updates = {k: v for k, v in updates.items() if v is not None}

        if updates:
            # Persist normalized runtime settings when update flags are present.
            out = update_ml_runtime_settings(updates)
        else:
            # Read current effective runtime settings when no updates are requested.
            out = get_ml_runtime_settings()

        # Print machine-readable JSON output for scripts and operators.
        print(json.dumps(out, indent=2))
    finally:
        # Always close DB resources even if an exception occurs.
        db.close_db()


if __name__ == "__main__":
    # Execute CLI entrypoint for direct script usage.
    main()
