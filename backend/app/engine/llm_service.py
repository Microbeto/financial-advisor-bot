import asyncio
import logging
import os
from typing import List, Optional

from ollama import AsyncClient

logger = logging.getLogger(__name__)

# Read strict timeout budgets from env so latency caps are configurable.
OLLAMA_HEALTH_TIMEOUT_SEC = float(os.getenv("OLLAMA_HEALTH_TIMEOUT_SEC", "1.5"))
OLLAMA_SUMMARY_TIMEOUT_SEC = float(os.getenv("OLLAMA_SUMMARY_TIMEOUT_SEC", "20.0"))
OLLAMA_GLOSSARY_TIMEOUT_SEC = float(os.getenv("OLLAMA_GLOSSARY_TIMEOUT_SEC", "8.0"))


def _ollama_host() -> str:
    direct = str(os.getenv("OLLAMA_HOST", "")).strip()
    if direct:
        return direct
    tags_url = str(os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")).strip()
    if tags_url.endswith("/api/tags"):
        return tags_url[: -len("/api/tags")]
    return tags_url.rstrip("/")


class GenerativeIntelligence:
    def __init__(self):
        # Initialize a local Ollama client using the same host convention as the model router.
        self.client = AsyncClient(host=_ollama_host())
        self.model_name = str(os.getenv("OLLAMA_MODEL", "llama3.2:1b")).strip() or "llama3.2:1b"
        self._resolved_model_name: str | None = None

    async def _available_models(self) -> list[str]:
        payload = await asyncio.wait_for(
            self.client.list(),
            timeout=max(0.2, float(OLLAMA_HEALTH_TIMEOUT_SEC)),
        )
        if isinstance(payload, dict):
            models = payload.get("models", [])
        else:
            models = getattr(payload, "models", [])
        out: list[str] = []
        for item in models:
            if isinstance(item, dict):
                names = [item.get("name"), item.get("model")]
            else:
                names = [getattr(item, "name", None), getattr(item, "model", None)]
            for raw in names:
                name = str(raw or "").strip()
                if name and name not in out:
                    out.append(name)
        return out

    async def _resolve_model_name(self) -> str | None:
        if self._resolved_model_name:
            return self._resolved_model_name

        available = await self._available_models()
        if not available:
            return None

        candidates: list[str] = []
        for candidate in [
            str(os.getenv("OLLAMA_CHAT_MODEL", "")).strip(),
            str(os.getenv("OLLAMA_MODEL", "")).strip(),
            self.model_name,
            "mistral:latest",
            "mistral",
        ]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            if candidate in available:
                self._resolved_model_name = candidate
                return candidate
            if ":" not in candidate and f"{candidate}:latest" in available:
                self._resolved_model_name = f"{candidate}:latest"
                return self._resolved_model_name

        self._resolved_model_name = available[0]
        return self._resolved_model_name

    async def check_health(self) -> bool:
        """Verifies if the local inference server is running."""
        try:
            # Keep health checks strict so slow inference is treated as unavailable.
            _ = await self._available_models()
            return True
        except Exception:
            return False

    async def generate_market_summary(self, headlines: List[str]) -> Optional[str]:
        """
        Synthesizes a human-readable market summary from raw headlines.
        """
        # Skip generation when no signal-carrying input headlines are available.
        if not headlines:
            return None

        # Format the context for the LLM
        news_context = "\n".join([f"- {h}" for h in headlines])

        # System Prompt: Defines the persona and constraints
        system_prompt = """
        You are an elite quantitative financial analyst.
        Analyze the provided news headlines and write a concise, 3-sentence summary
        of the current market regime. Focus on macroeconomic trends, risk appetite,
        and sector momentum. Do not give financial advice.
        """

        try:
            model_name = await self._resolve_model_name()
            if not model_name:
                return "Market summary unavailable because no local inference model is installed."

            # Make the asynchronous call to the local model
            response = await asyncio.wait_for(
                self.client.chat(
                    model=model_name,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': f"Here are today's top headlines:\n{news_context}"}
                    ],
                    # options allow you to tweak the generation (lower temperature = more deterministic)
                    options={'temperature': 0.2, 'num_predict': 150}
                ),
                timeout=max(0.5, float(OLLAMA_SUMMARY_TIMEOUT_SEC)),
            )
            # Return the model text as a compact display-ready summary.
            if isinstance(response, dict):
                content = str(response.get("message", {}).get("content", "") or "").strip()
            else:
                msg = getattr(response, "message", None)
                if isinstance(msg, dict):
                    content = str(msg.get("content", "") or "").strip()
                else:
                    content = str(getattr(msg, "content", "") or "").strip()
            return content or "Market summary unavailable because local inference returned empty output."

        except asyncio.TimeoutError:
            logger.warning("Local LLM market summary timed out after %.2fs", max(0.5, float(OLLAMA_SUMMARY_TIMEOUT_SEC)))
            return "Market summary unavailable because the local inference model timed out."
        except Exception as e:
            # Fail closed with a user-safe fallback message when inference errors out.
            logger.error(f"Local LLM generation failed: {e}")
            if "not found" in str(e).lower() or "model" in str(e).lower() and "pull" in str(e).lower():
                return "Market summary unavailable because the configured local inference model is missing."
            return "Market summary temporarily unavailable due to local inference error."

    async def generate_term_definition(self, term: str) -> Optional[str]:
        """
        Generates a short plain-language definition for a financial term.
        """
        # Normalize and reject empty glossary terms before inference.
        cleaned_term = str(term or "").strip()
        if not cleaned_term:
            return None

        system_prompt = """
        You are a financial glossary assistant.
        Explain one term in plain English for retail investors.
        Keep the answer to 1-2 short sentences (max 55 words), neutral and factual.
        Focus on practical meaning in markets. Do not provide investing advice.
        """

        user_prompt = f"Define this term for a dashboard glossary: {cleaned_term}"

        try:
            model_name = await self._resolve_model_name()
            if not model_name:
                return None

            # Generate a short glossary definition under a strict request timeout.
            response = await asyncio.wait_for(
                self.client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": 0.2, "num_predict": 100},
                ),
                timeout=max(0.5, float(OLLAMA_GLOSSARY_TIMEOUT_SEC)),
            )
            # Normalize the response payload and return None for empty outputs.
            if isinstance(response, dict):
                out = str(response.get("message", {}).get("content", "") or "").strip()
            else:
                msg = getattr(response, "message", None)
                if isinstance(msg, dict):
                    out = str(msg.get("content", "") or "").strip()
                else:
                    out = str(getattr(msg, "content", "") or "").strip()
            return out or None
        except asyncio.TimeoutError:
            logger.warning("Local LLM glossary generation timed out after %.2fs", max(0.5, float(OLLAMA_GLOSSARY_TIMEOUT_SEC)))
            return None
        except Exception as e:
            # Log generation failure and let caller choose fallback behavior.
            logger.error(f"Local LLM glossary generation failed: {e}")
            return None
