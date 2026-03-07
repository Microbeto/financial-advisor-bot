import asyncio
import logging
from typing import List, Optional

from ollama import AsyncClient

logger = logging.getLogger(__name__)


class GenerativeIntelligence:
    def __init__(self):
        self.client = AsyncClient(host='http://localhost:11434')
        self.model_name = "mistral"  # Or "llama3"

    async def check_health(self) -> bool:
        """Verifies if the local inference server is running."""
        try:
            # A simple fast call to check connection
            await self.client.list()
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
            response = await self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Here are today's top headlines:\n{news_context}"}
                ],
                # options allow you to tweak the generation (lower temperature = more deterministic)
                options={'temperature': 0.2, 'num_predict': 150}
            )
            return response['message']['content'].strip()

        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            return "Market summary temporarily unavailable due to inference server disconnect."
