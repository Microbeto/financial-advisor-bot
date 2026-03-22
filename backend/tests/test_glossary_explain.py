# Unit tests for glossary service including term definitions and LLM-based explanations.
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
    # Keep dynamic glossary state isolated between tests.
    glossary_engine._DYNAMIC_GLOSSARY.clear()
    yield
    # Ensure no test leaves residual dynamic terms behind.
    glossary_engine._DYNAMIC_GLOSSARY.clear()


def _prepare_app(monkeypatch: pytest.MonkeyPatch, *, ollama_available: bool) -> TestClient:
    # Avoid external startup side effects (DB bootstrap/admin bootstrap) in unit tests.
    monkeypatch.setattr(main_module.db, "init_db", lambda: None)
    monkeypatch.setattr(main_module.db, "close_db", lambda: None)
    monkeypatch.setattr(main_module, "ensure_selected_model_exists", lambda: None)
    monkeypatch.setattr(main_module, "bootstrap_admin_if_configured", lambda: None)

    async def _fake_run_diagnostics():
        # Force a stable diagnostics payload for predictable app startup.
        return {"tier": "high"}

    monkeypatch.setattr(main_module.intelligence_router, "run_diagnostics", _fake_run_diagnostics, raising=False)
    # Toggle perceived Ollama availability per test scenario.
    monkeypatch.setattr(main_module.intelligence_router, "ollama_available", ollama_available, raising=False)

    app = create_app()
    return TestClient(app)


# Test: glossary explain existing term short circuits llm.
def test_glossary_explain_existing_term_short_circuits_llm(monkeypatch: pytest.MonkeyPatch):
    async def _should_not_call_health(self):
        # Existing terms should bypass LLM health checks.
        raise AssertionError("LLM health check should not run for existing glossary terms")

    async def _should_not_call_generate(self, term: str):
        # Existing terms should bypass LLM text generation.
        raise AssertionError("LLM generation should not run for existing glossary terms")

    monkeypatch.setattr(GenerativeIntelligence, "check_health", _should_not_call_health)
    monkeypatch.setattr(GenerativeIntelligence, "generate_term_definition", _should_not_call_generate)

    with _prepare_app(monkeypatch, ollama_available=True) as client:
        # Request a known core glossary term.
        res = client.post("/glossary/explain", json={"term": "volatility"})

    # Validate existing-term response shape and source routing.
    assert res.status_code == 200
    body = res.json()
    assert body["term"] == "Volatility"
    assert body["generated"] is False
    assert body["source"] == "existing"
    assert "price moves day to day" in body["definition"]


# Test: glossary explain llm path with mocked generation.
def test_glossary_explain_llm_path_with_mocked_generation(monkeypatch: pytest.MonkeyPatch):
    async def _healthy(self):
        # Simulate healthy local inference service.
        return True

    async def _generate(self, term: str):
        # Return deterministic generated text for this unknown term.
        assert term == "convexity"
        return "Convexity measures how a bond's duration changes as yields move. Higher convexity means larger price sensitivity shifts for the same rate change."

    monkeypatch.setattr(GenerativeIntelligence, "check_health", _healthy)
    monkeypatch.setattr(GenerativeIntelligence, "generate_term_definition", _generate)

    with _prepare_app(monkeypatch, ollama_available=True) as client:
        # Request an unknown term to trigger LLM generation path.
        res = client.post("/glossary/explain", json={"term": "convexity"})

    # Validate LLM-tagged response and dynamic glossary persistence.
    assert res.status_code == 200
    body = res.json()
    assert body["term"] == "convexity"
    assert body["generated"] is True
    assert body["source"] == "llm"
    assert "duration" in body["definition"].lower()
    assert glossary_engine.lingo_glossary().get("convexity") == body["definition"]


# Test: glossary explain fallback when llm unavailable.
def test_glossary_explain_fallback_when_llm_unavailable(monkeypatch: pytest.MonkeyPatch):
    async def _should_not_call_health(self):
        # If router says unavailable, health check must never execute.
        raise AssertionError("LLM health check should not run when ollama is unavailable")

    async def _should_not_call_generate(self, term: str):
        # If router says unavailable, generation must never execute.
        raise AssertionError("LLM generation should not run when ollama is unavailable")

    monkeypatch.setattr(GenerativeIntelligence, "check_health", _should_not_call_health)
    monkeypatch.setattr(GenerativeIntelligence, "generate_term_definition", _should_not_call_generate)

    with _prepare_app(monkeypatch, ollama_available=False) as client:
        # Request an unknown term to trigger deterministic fallback path.
        res = client.post("/glossary/explain", json={"term": "basis risk"})

    # Validate fallback-tagged response and dynamic glossary persistence.
    assert res.status_code == 200
    body = res.json()
    assert body["term"] == "basis risk"
    assert body["generated"] is True
    assert body["source"] == "fallback"
    assert "market behavior" in body["definition"].lower()
    assert glossary_engine.lingo_glossary().get("basis risk") == body["definition"]
