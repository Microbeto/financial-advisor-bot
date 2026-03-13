from __future__ import annotations

import argparse
import json

from app.services.admin.cache_admin import clear_cache, get_cache_stats, prune_stale_cache


def main() -> int:
    # Define CLI options for cache stats, pruning, and clear operations.
    parser = argparse.ArgumentParser(description="Market cache maintenance utility")
    parser.add_argument("--stats", action="store_true", help="Show cache stats")
    parser.add_argument("--prune-days", type=int, default=None, help="Prune files older than N days")
    parser.add_argument("--clear", choices=["stale", "all"], default=None, help="Clear cache files")
    parser.add_argument("--dry-run", action="store_true", help="List affected files without deleting")

    # Parse user-provided command-line arguments.
    args = parser.parse_args()

    if args.stats:
        # Print aggregated cache statistics and exit.
        print(json.dumps(get_cache_stats(), indent=2))
        return 0

    if args.prune_days is not None:
        # Prune stale cache entries older than the selected age threshold.
        res = prune_stale_cache(max_age_days=args.prune_days, dry_run=bool(args.dry_run))
        print(json.dumps(res, indent=2))
        return 0

    if args.clear is not None:
        # Clear cache scope (stale or all), optionally as a dry run.
        res = clear_cache(scope=args.clear, max_age_days=None, dry_run=bool(args.dry_run))
        print(json.dumps(res, indent=2))
        return 0

    # Show usage help when no actionable option is provided.
    parser.print_help()
    return 0


if __name__ == "__main__":
    # Run as a standalone script and propagate exit code.
    raise SystemExit(main())
