from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_date_utc(dt: Optional[datetime] = None) -> str:
    d = dt or utc_now()
    return d.astimezone(timezone.utc).date().isoformat()


def normalize_symbol(sym: str) -> str:
    return (sym or "").upper().strip()


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
