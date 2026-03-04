from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ..market import (
    MARKET_CACHE_TTL_DAYS,
    MARKET_LOCAL_CACHE_DIR,
    MARKET_LOCAL_CACHE_ENABLED,
    get_market_cache_runtime_stats,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _cache_root() -> Path:
    root = Path(MARKET_LOCAL_CACHE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _iter_cache_files() -> List[Path]:
    root = _cache_root()
    if not root.exists():
        return []
    return [p for p in root.glob("*.json") if p.is_file()]


def _file_age_days(path: Path) -> float:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age_sec = (_utc_now() - mtime).total_seconds()
        return float(age_sec / 86400.0)
    except Exception:
        return 999999.0


def _bytes_human(n: int) -> str:
    size = float(max(0, int(n)))
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size >= 1024 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {units[i]}"


def get_cache_stats() -> Dict[str, Any]:
    files = _iter_cache_files()
    runtime = get_market_cache_runtime_stats()

    total_size = 0
    oldest_mtime = None
    newest_mtime = None

    for p in files:
        try:
            st = p.stat()
            total_size += int(st.st_size)
            m = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
            if oldest_mtime is None or m < oldest_mtime:
                oldest_mtime = m
            if newest_mtime is None or m > newest_mtime:
                newest_mtime = m
        except Exception:
            continue

    req = int(runtime.get("requests", 0))
    disk_hits = int(runtime.get("disk_hits", 0))
    mongo_hits = int(runtime.get("mongo_hits", 0))
    total_hits = disk_hits + mongo_hits
    hit_rate = (float(total_hits) / float(req)) if req > 0 else 0.0

    return {
        "local_cache_enabled": bool(MARKET_LOCAL_CACHE_ENABLED),
        "cache_dir": str(_cache_root()),
        "ttl_days": int(MARKET_CACHE_TTL_DAYS),
        "file_count": len(files),
        "total_size_bytes": int(total_size),
        "total_size_human": _bytes_human(total_size),
        "oldest_file_at": oldest_mtime.isoformat().replace("+00:00", "Z") if oldest_mtime else None,
        "newest_file_at": newest_mtime.isoformat().replace("+00:00", "Z") if newest_mtime else None,
        "runtime": {
            **runtime,
            "total_hits": total_hits,
            "hit_rate": hit_rate,
        },
    }


def prune_stale_cache(max_age_days: int | None = None, dry_run: bool = True) -> Dict[str, Any]:
    ttl_days = int(max_age_days if max_age_days is not None else MARKET_CACHE_TTL_DAYS)
    ttl_days = max(0, ttl_days)

    files = _iter_cache_files()
    stale: List[Path] = [p for p in files if _file_age_days(p) > float(ttl_days)]

    total_bytes = 0
    deleted = 0
    deleted_names: List[str] = []

    for p in stale:
        try:
            total_bytes += int(p.stat().st_size)
        except Exception:
            pass

        if dry_run:
            deleted_names.append(p.name)
            continue

        try:
            p.unlink(missing_ok=True)
            deleted += 1
            deleted_names.append(p.name)
        except Exception:
            continue

    return {
        "action": "prune_stale",
        "dry_run": bool(dry_run),
        "ttl_days": ttl_days,
        "matched_files": len(stale),
        "deleted_files": deleted if not dry_run else 0,
        "bytes_freed": 0 if dry_run else total_bytes,
        "bytes_freed_human": _bytes_human(0 if dry_run else total_bytes),
        "matched_names": deleted_names,
    }


def clear_cache(scope: str = "stale", max_age_days: int | None = None, dry_run: bool = False) -> Dict[str, Any]:
    scope_clean = (scope or "stale").strip().lower()
    if scope_clean == "stale":
        return prune_stale_cache(max_age_days=max_age_days, dry_run=dry_run)

    files = _iter_cache_files()
    total_bytes = 0
    deleted = 0
    names: List[str] = []

    for p in files:
        try:
            total_bytes += int(p.stat().st_size)
        except Exception:
            pass

        names.append(p.name)
        if dry_run:
            continue

        try:
            p.unlink(missing_ok=True)
            deleted += 1
        except Exception:
            continue

    return {
        "action": "clear_all",
        "dry_run": bool(dry_run),
        "matched_files": len(files),
        "deleted_files": deleted if not dry_run else 0,
        "bytes_freed": 0 if dry_run else total_bytes,
        "bytes_freed_human": _bytes_human(0 if dry_run else total_bytes),
        "matched_names": names,
    }
