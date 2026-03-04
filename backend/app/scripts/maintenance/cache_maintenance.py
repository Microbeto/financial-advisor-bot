from __future__ import annotations

import argparse
import json

from app.services.admin.cache_admin import clear_cache, get_cache_stats, prune_stale_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Market cache maintenance utility")
    parser.add_argument("--stats", action="store_true", help="Show cache stats")
    parser.add_argument("--prune-days", type=int, default=None, help="Prune files older than N days")
    parser.add_argument("--clear", choices=["stale", "all"], default=None, help="Clear cache files")
    parser.add_argument("--dry-run", action="store_true", help="List affected files without deleting")

    args = parser.parse_args()

    if args.stats:
        print(json.dumps(get_cache_stats(), indent=2))
        return 0

    if args.prune_days is not None:
        res = prune_stale_cache(max_age_days=args.prune_days, dry_run=bool(args.dry_run))
        print(json.dumps(res, indent=2))
        return 0

    if args.clear is not None:
        res = clear_cache(scope=args.clear, max_age_days=None, dry_run=bool(args.dry_run))
        print(json.dumps(res, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
