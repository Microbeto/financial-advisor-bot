from __future__ import annotations

import argparse
import json

from ... import db
from ...services.ml_workflow import backfill_model_nlp_pipeline


def _build_parser() -> argparse.ArgumentParser:
    # Define CLI options for dry-run/apply behavior and migration heuristics.
    p = argparse.ArgumentParser(description="Backfill nlp_pipeline for legacy ML model registry docs")
    p.add_argument("--apply", action="store_true", help="Persist updates (default is dry-run)")
    p.add_argument("--limit", type=int, default=0, help="Max number of model docs to inspect (0 = all)")
    p.add_argument("--rollout-iso", type=str, default=None, help="Cutoff ISO time for artifact timestamp inference")
    p.add_argument(
        "--default-pipeline",
        choices=["lexicon", "finbert", "cascade"],
        default="finbert",
        help="Fallback pipeline when metadata and timestamps are unavailable",
    )
    return p


def main() -> None:
    # Parse command-line arguments once at script startup.
    args = _build_parser().parse_args()

    # Open DB connection required by model registry migration service.
    db.init_db()
    try:
        # Run migration/backfill and emit a JSON summary for operators.
        out = backfill_model_nlp_pipeline(
            dry_run=not bool(args.apply),
            limit=int(args.limit),
            rollout_iso=args.rollout_iso,
            default_pipeline=str(args.default_pipeline),
        )
        print(json.dumps(out, indent=2))
    finally:
        # Always close DB resources, even if migration raises an error.
        db.close_db()


if __name__ == "__main__":
    # Execute CLI entrypoint when invoked as a script.
    main()
