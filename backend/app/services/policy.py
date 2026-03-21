# Policy engine module defining hard access and action constraints based on profile and regime.
# Structure: policy engine module defining hard access/action constraints from profile and regime inputs.
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .. import db
from ..models import Regime, RiskProfile


# PolicyDecision class and its related behavior.
@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reasons: List[str]


# PolicyEngine class and its related behavior.
class PolicyEngine:
    """
    Hard rules. This runs before you show signals or allow portfolio actions.
    The ML/signal layer can suggest. Policy decides.
    """

    def evaluate_dashboard_access(self, profile: RiskProfile, regime: Regime) -> PolicyDecision:
        reasons: List[str] = []

        # Example constraints you can toggle in UI later.
        constraints = set([c.strip().lower() for c in (profile.constraints or [])])
        constraints.update(set(self.get_global_constraints()))

        # If user requests "no news" you can comply by hiding links. Still allow dashboard.
        if "hide_news" in constraints:
            reasons.append("User constraint: hide_news")

        # If risk is low and regime is risk_off, you may restrict showing "downtrend short ideas"
        # and only show defensive info.
        if profile.risk_tolerance == "low" and regime == "risk_off":
            reasons.append("Low risk profile in risk-off regime: restrict aggressive ideas")

        return PolicyDecision(allowed=True, reasons=reasons)

    def filter_symbols_for_profile(self, symbols: List[str], profile: RiskProfile, regime: Regime) -> Tuple[List[str], List[str]]:
        """
        Returns (allowed_symbols, policy_notes)
        """
        notes: List[str] = []
        constraints = set([c.strip().lower() for c in (profile.constraints or [])])
        constraints.update(set(self.get_global_constraints()))

        allowed = symbols[:]

        # Example: user forbids certain sectors; you can enforce once you have sector metadata.
        # For now, keep it as a placeholder that still logs.
        if "no_crypto" in constraints:
            notes.append("Constraint no_crypto: not applicable to equities basket")

        if profile.risk_tolerance == "low" and regime == "risk_off":
            # In risk-off, low risk users only get the uptrend list (defensive).
            notes.append("Low risk in risk-off: downtrend list should be hidden on frontend")
        return allowed, notes

    def get_global_constraints(self) -> List[str]:
        try:
            col = db.get_db()["policy_settings"]
            doc = col.find_one({"key": "global_constraints"}) or {}
            vals = doc.get("constraints") or []
            return [str(x).strip().lower() for x in vals if str(x).strip()]
        except Exception:
            return []

    def update_global_constraints(self, add: List[str], remove: List[str]) -> List[str]:
        current = set(self.get_global_constraints())
        for x in add or []:
            s = str(x).strip().lower()
            if s:
                current.add(s)
        for x in remove or []:
            s = str(x).strip().lower()
            if s:
                current.discard(s)

        out = sorted(current)
        try:
            col = db.get_db()["policy_settings"]
            col.update_one(
                {"key": "global_constraints"},
                {"$set": {"key": "global_constraints", "constraints": out}},
                upsert=True,
            )
        except Exception:
            pass
        return out
