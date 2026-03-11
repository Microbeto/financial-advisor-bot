from __future__ import annotations

import argparse
import json

from ... import db
from ...services.ml_workflow import backfill_model_nlp_pipeline


def _build_parser() -> argparse.ArgumentParser:
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
    args = _build_parser().parse_args()

    db.init_db()
    try:
        out = backfill_model_nlp_pipeline(
            dry_run=not bool(args.apply),
            limit=int(args.limit),
            rollout_iso=args.rollout_iso,
            default_pipeline=str(args.default_pipeline),
        )
        print(json.dumps(out, indent=2))
    finally:
        db.close_db()


if __name__ == "__main__":
    main()
