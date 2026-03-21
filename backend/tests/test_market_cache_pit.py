# Unit tests for market cache point-in-time immutability and data merging logic.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.market import _merge_pit_points


def test_merge_pit_points_preserves_existing_history_values():
    existing = [
        {"t": "2024-01-02", "c": 101.0},
        {"t": "2024-01-03", "c": 102.0},
    ]
    incoming = [
        {"t": "2024-01-03", "c": 999.0},  # restated value should be ignored
        {"t": "2024-01-04", "c": 103.0},
    ]

    merged = _merge_pit_points(existing, incoming)
    by_date = {str(row.get("t")): row for row in merged}

    assert float(by_date["2024-01-03"]["c"]) == 102.0
    assert float(by_date["2024-01-04"]["c"]) == 103.0


def test_merge_pit_points_appends_new_dates_and_keeps_sorted_order():
    existing = [
        {"t": "2024-01-05", "c": 105.0},
    ]
    incoming = [
        {"t": "2024-01-04", "c": 104.0},
        {"t": "2024-01-06", "c": 106.0},
    ]

    merged = _merge_pit_points(existing, incoming)
    dates = [str(row.get("t")) for row in merged if str(row.get("t") or "")]

    assert dates == ["2024-01-04", "2024-01-05", "2024-01-06"]
