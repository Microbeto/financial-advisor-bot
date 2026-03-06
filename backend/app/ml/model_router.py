from __future__ import annotations

import logging
import os
from typing import Any, Dict, Literal

import httpx

try:
    import psutil

    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)

Tier = Literal["low", "medium", "high"]


class IntelligenceRouter:
    """
    Startup diagnostics and lightweight routing for graceful degradation.
    """

    def __init__(self, ollama_url: str | None = None) -> None:
        self.system_capability: Tier = "low"
        self.ollama_available: bool = False
        self.total_ram_gb: float = 0.0
        self.available_ram_gb: float = 0.0
        self.gpu_available: bool = False
        self.ollama_url = (ollama_url or os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")).strip()

    def _read_ram(self) -> tuple[float, float]:
        if not PSUTIL_AVAILABLE or psutil is None:
            return (0.0, 0.0)
        try:
            vm = psutil.virtual_memory()
            total = float(vm.total) / float(1024**3)
            avail = float(vm.available) / float(1024**3)
            return (total, avail)
        except Exception:
            return (0.0, 0.0)

    def _read_gpu_available(self) -> bool:
        if not TORCH_AVAILABLE or torch is None:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    async def _ping_ollama(self) -> bool:
        try:
            timeout = httpx.Timeout(2.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(self.ollama_url)
            return response.status_code == 200
        except Exception:
            return False

    async def run_diagnostics(self) -> Dict[str, Any]:
        """
        Runs during server startup to determine capability tier.
        """
        total_ram, available_ram = self._read_ram()
        gpu_available = self._read_gpu_available()
        ollama_active = await self._ping_ollama()

        self.total_ram_gb = float(total_ram)
        self.available_ram_gb = float(available_ram)
        self.gpu_available = bool(gpu_available)
        self.ollama_available = bool(ollama_active)

        if gpu_available and available_ram > 8.0 and ollama_active:
            self.system_capability = "high"
            logger.info("Model router: high tier enabled (GPU + RAM + Ollama).")
        elif available_ram > 4.0:
            self.system_capability = "medium"
            logger.info("Model router: medium tier enabled (CPU FinBERT path available).")
        else:
            self.system_capability = "low"
            logger.warning("Model router: low tier enabled (fallback NLP path).")

        return {
            "tier": self.system_capability,
            "ram_total_gb": round(self.total_ram_gb, 2),
            "ram_available_gb": round(self.available_ram_gb, 2),
            "gpu": self.gpu_available,
            "ollama_active": self.ollama_available,
            "psutil_available": PSUTIL_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
        }

    def get_sentiment_pipeline(self) -> str:
        """
        Placeholder routing decision for sentiment engine selection.
        Replace return values with real pipeline classes as needed.
        """
        if self.system_capability in ("high", "medium"):
            return "finbert"
        return "lexicon"


intelligence_router = IntelligenceRouter()
