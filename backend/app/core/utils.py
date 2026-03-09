from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    # Returns the current timezone-aware UTC datetime.
    return datetime.now(timezone.utc)


def iso_date_utc(dt: Optional[datetime] = None) -> str:
    # Uses the provided datetime or falls back to the current UTC time.
    d = dt or utc_now()
    # Normalizes to UTC and returns an ISO date string (YYYY-MM-DD).
    return d.astimezone(timezone.utc).date().isoformat()


def normalize_symbol(sym: str) -> str:
    # Normalizes a ticker symbol by handling empty input, uppercasing, and trimming spaces.
    return (sym or "").upper().strip()


def clamp(x: float, lo: float, hi: float) -> float:
    # Clamps values below the lower bound up to `lo`.
    if x < lo:
        return lo
    # Clamps values above the upper bound down to `hi`.
    if x > hi:
        return hi
    # Returns the original value when it is already in range.
    return x
