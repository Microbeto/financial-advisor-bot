from __future__ import annotations

from typing import List

from .. import db
from ..models import AdminBasketDecision


def get_admin_basket(date: str) -> AdminBasketDecision:
    col = db.require_col(db.chosen_stocks_col, "chosen_stocks")
    doc = col.find_one({"date": date}) or {}
    candidates = list(doc.get("candidates") or [])
    selected = list(doc.get("selected") or [])
    return AdminBasketDecision(date=date, candidates=candidates, selected=selected)


def set_admin_basket(date: str, selected: List[str]) -> AdminBasketDecision:
    col = db.require_col(db.chosen_stocks_col, "chosen_stocks")
    selected = [s.upper() for s in selected if s]
    doc = col.find_one({"date": date}) or {}
    candidates = list(doc.get("candidates") or [])
    col.update_one({"date": date}, {"$set": {"date": date, "candidates": candidates, "selected": selected}}, upsert=True)
    return AdminBasketDecision(date=date, candidates=candidates, selected=selected)


def set_candidates(date: str, candidates: List[str]) -> None:
    col = db.require_col(db.chosen_stocks_col, "chosen_stocks")
    candidates = [s.upper() for s in candidates if s]
    doc = col.find_one({"date": date}) or {}
    selected = list(doc.get("selected") or [])
    col.update_one({"date": date}, {"$set": {"date": date, "candidates": candidates, "selected": selected}}, upsert=True)
