from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.engine import glossary as glossary_engine
from app.engine.llm_service import GenerativeIntelligence
from app.main import create_app
import app.main as main_module


@pytest.fixture(autouse=True)
def _reset_dynamic_glossary_state():
    glossary_engine._DYNAMIC_GLOSSARY.clear()
    yield
    glossary_engine._DYNAMIC_GLOSSARY.clear()


def _prepare_app(monkeypatch: pytest.MonkeyPatch, *, ollama_available: bool) -> TestClient:
    monkeypatch.setattr(main_module.db, "init_db", lambda: None)
    monkeypatch.setattr(main_module.db, "close_db", lambda: None)
    monkeypatch.setattr(main_module, "ensure_selected_model_exists", lambda: None)
    monkeypatch.setattr(main_module, "bootstrap_admin_if_configured", lambda: None)

    async def _fake_run_diagnostics():
        return {"tier": "high"}

    monkeypatch.setattr(main_module.intelligence_router, "run_diagnostics", _fake_run_diagnostics, raising=False)
    monkeypatch.setattr(main_module.intelligence_router, "ollama_available", ollama_available, raising=False)

    app = create_app()
    return TestClient(app)


def test_glossary_explain_existing_term_short_circuits_llm(monkeypatch: pytest.MonkeyPatch):
    async def _should_not_call_health(self):
        raise AssertionError("LLM health check should not run for existing glossary terms")

    async def _should_not_call_generate(self, term: str):
        raise AssertionError("LLM generation should not run for existing glossary terms")

    monkeypatch.setattr(GenerativeIntelligence, "check_health", _should_not_call_health)
    monkeypatch.setattr(GenerativeIntelligence, "generate_term_definition", _should_not_call_generate)

    with _prepare_app(monkeypatch, ollama_available=True) as client:
        res = client.post("/glossary/explain", json={"term": "volatility"})

    assert res.status_code == 200
    body = res.json()
    assert body["term"] == "Volatility"
    assert body["generated"] is False
    assert body["source"] == "existing"
    assert "price moves day to day" in body["definition"]


def test_glossary_explain_llm_path_with_mocked_generation(monkeypatch: pytest.MonkeyPatch):
    async def _healthy(self):
        return True

    async def _generate(self, term: str):
        assert term == "convexity"
        return "Convexity measures how a bond's duration changes as yields move. Higher convexity means larger price sensitivity shifts for the same rate change."

    monkeypatch.setattr(GenerativeIntelligence, "check_health", _healthy)
    monkeypatch.setattr(GenerativeIntelligence, "generate_term_definition", _generate)

    with _prepare_app(monkeypatch, ollama_available=True) as client:
        res = client.post("/glossary/explain", json={"term": "convexity"})

    assert res.status_code == 200
    body = res.json()
    assert body["term"] == "convexity"
    assert body["generated"] is True
    assert body["source"] == "llm"
    assert "duration" in body["definition"].lower()
    assert glossary_engine.lingo_glossary().get("convexity") == body["definition"]


def test_glossary_explain_fallback_when_llm_unavailable(monkeypatch: pytest.MonkeyPatch):
    async def _should_not_call_health(self):
        raise AssertionError("LLM health check should not run when ollama is unavailable")

    async def _should_not_call_generate(self, term: str):
        raise AssertionError("LLM generation should not run when ollama is unavailable")

    monkeypatch.setattr(GenerativeIntelligence, "check_health", _should_not_call_health)
    monkeypatch.setattr(GenerativeIntelligence, "generate_term_definition", _should_not_call_generate)

    with _prepare_app(monkeypatch, ollama_available=False) as client:
        res = client.post("/glossary/explain", json={"term": "basis risk"})

    assert res.status_code == 200
    body = res.json()
    assert body["term"] == "basis risk"
    assert body["generated"] is True
    assert body["source"] == "fallback"
    assert "market behavior" in body["definition"].lower()
    assert glossary_engine.lingo_glossary().get("basis risk") == body["definition"]
