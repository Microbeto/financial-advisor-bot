import asyncio
import logging
import os
from typing import List, Optional

from ollama import AsyncClient

logger = logging.getLogger(__name__)

OLLAMA_HEALTH_TIMEOUT_SEC = float(os.getenv("OLLAMA_HEALTH_TIMEOUT_SEC", "1.5"))
OLLAMA_SUMMARY_TIMEOUT_SEC = float(os.getenv("OLLAMA_SUMMARY_TIMEOUT_SEC", "6.0"))
OLLAMA_GLOSSARY_TIMEOUT_SEC = float(os.getenv("OLLAMA_GLOSSARY_TIMEOUT_SEC", "4.0"))


class GenerativeIntelligence:
    def __init__(self):
        self.client = AsyncClient(host='http://localhost:11434')
        self.model_name = "mistral"  # Or "llama3"

    async def check_health(self) -> bool:
        """Verifies if the local inference server is running."""
        try:
            # Keep health checks strict so slow inference is treated as unavailable.
            await asyncio.wait_for(
                self.client.list(),
                timeout=max(0.2, float(OLLAMA_HEALTH_TIMEOUT_SEC)),
            )
            return True
        except Exception:
            return False

    async def generate_market_summary(self, headlines: List[str]) -> Optional[str]:
        """
        Synthesizes a human-readable market summary from raw headlines.
        """
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
            # Make the asynchronous call to the local model
            response = await asyncio.wait_for(
                self.client.chat(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': f"Here are today's top headlines:\n{news_context}"}
                    ],
                    # options allow you to tweak the generation (lower temperature = more deterministic)
                    options={'temperature': 0.2, 'num_predict': 150}
                ),
                timeout=max(0.5, float(OLLAMA_SUMMARY_TIMEOUT_SEC)),
            )
            return response['message']['content'].strip()

        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            return "Market summary temporarily unavailable due to inference server disconnect."

    async def generate_term_definition(self, term: str) -> Optional[str]:
        """
        Generates a short plain-language definition for a financial term.
        """
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
            response = await asyncio.wait_for(
                self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": 0.2, "num_predict": 100},
                ),
                timeout=max(0.5, float(OLLAMA_GLOSSARY_TIMEOUT_SEC)),
            )
            out = str(response.get("message", {}).get("content", "") or "").strip()
            return out or None
        except Exception as e:
            logger.error(f"Local LLM glossary generation failed: {e}")
            return None
