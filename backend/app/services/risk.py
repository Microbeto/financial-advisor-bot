from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .. import db
from ..models import RiskProfile


def get_risk_profile_for_user(user_id: str) -> Optional[RiskProfile]:
    col = db.require_col(db.risk_profiles_col, "risk_profiles")
    doc = col.find_one({"user_id": user_id})
    if not doc:
        return None
    doc.pop("_id", None)
    return RiskProfile(**doc)


def ensure_risk_profile_for_user(user_id: str) -> RiskProfile:
    col = db.require_col(db.risk_profiles_col, "risk_profiles")
    existing = col.find_one({"user_id": user_id})
    if existing:
        existing.pop("_id", None)
        return RiskProfile(**existing)

    now = datetime.now(timezone.utc)
    doc = {
        "user_id": user_id,
        "client_name": "",
        "age": None,
        "horizon_years": None,
        "risk_tolerance": "balanced",
        "max_drawdown_pct": None,
        "income_stability": "",
        "constraints": [],
        "updated_at": now,
    }
    col.insert_one(doc)
    return RiskProfile(**doc)


def upsert_risk_profile_for_user(user_id: str, profile: RiskProfile) -> RiskProfile:
    col = db.require_col(db.risk_profiles_col, "risk_profiles")
    now = datetime.now(timezone.utc)
    doc = profile.model_dump()
    doc["user_id"] = user_id
    doc["updated_at"] = now
    col.update_one({"user_id": user_id}, {"$set": doc}, upsert=True)
    return RiskProfile(**doc)
