from __future__ import annotations

import logging
import os
import subprocess
import time
import importlib
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
        # Initialize router with system capability flags and model state
        self.system_capability: Tier = "low"
        self.sentiment_pipeline: str = "lexicon"
        self.ollama_available: bool = False
        self.total_ram_gb: float = 0.0
        self.available_ram_gb: float = 0.0
        self.gpu_available: bool = False
        self.ollama_url = (ollama_url or os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")).strip()
        self.ollama_inference_ms: float | None = None
        self.finbert_inference_ms: float | None = None

        self._finbert_ready = False
        self._finbert_tokenizer = None
        self._finbert_model = None
        self._finbert_torch = None
        self._finbert_model_name = ""

    def _read_ram(self) -> tuple[float, float]:
        # Query system RAM usage and return total and available GB
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
        # Check for GPU availability via torch CUDA or nvidia-smi fallback
        # Primary path: torch CUDA visibility.
        if TORCH_AVAILABLE and torch is not None:
            try:
                if bool(torch.cuda.is_available()):
                    return True
            except Exception:
                pass

        # Fallback path: detect NVIDIA GPU presence directly via nvidia-smi.
        # This covers environments with CPU-only torch builds (+cpu wheels).
        try:
            proc = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode == 0 and "GPU" in out:
                return True
        except Exception:
            pass

        return False

    async def _ping_ollama(self) -> bool:
        # Test if Ollama server is reachable and responsive
        try:
            timeout = httpx.Timeout(2.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(self.ollama_url)
            return response.status_code == 200
        except Exception:
            return False

    def _ensure_finbert_loaded(self) -> bool:
        # Load FinBERT model and tokenizer, attempting multiple candidate models
        if self._finbert_ready and self._finbert_tokenizer is not None and self._finbert_model is not None and self._finbert_torch is not None:
            return True

        try:
            torch_mod = importlib.import_module("torch")
            tr_mod = importlib.import_module("transformers")
        except Exception:
            return False

        cache_dir = os.getenv("FINBERT_CACHE_DIR", "").strip() or None
        candidates = [
            os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert").strip(),
            "ProsusAI/finbert",
            "yiyanghkust/finbert-tone",
        ]
        deduped = []
        for name in candidates:
            if name and name not in deduped:
                deduped.append(name)

        for model_name in deduped:
            try:
                kwargs: Dict[str, Any] = {"local_files_only": True}
                if cache_dir:
                    kwargs["cache_dir"] = cache_dir
                tokenizer = tr_mod.AutoTokenizer.from_pretrained(model_name, **kwargs)
                model = tr_mod.AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
                model.eval()
                self._finbert_tokenizer = tokenizer
                self._finbert_model = model
                self._finbert_torch = torch_mod
                self._finbert_model_name = str(model_name)
                self._finbert_ready = True
                return True
            except Exception:
                continue

        return False

    def _bench_finbert_sync(self, text: str) -> float | None:
        # Measure FinBERT inference latency in milliseconds
        if not self._ensure_finbert_loaded():
            return None
        if self._finbert_tokenizer is None or self._finbert_model is None or self._finbert_torch is None:
            return None

        t0 = time.perf_counter()
        try:
            enc = self._finbert_tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=256)
            with self._finbert_torch.no_grad():
                logits = self._finbert_model(**enc).logits
                _ = self._finbert_torch.softmax(logits, dim=1).cpu().numpy()
            return float((time.perf_counter() - t0) * 1000.0)
        except Exception:
            return None

    async def _benchmark_finbert_latency_ms(self) -> float | None:
        # Run FinBERT benchmark in a thread-safe async context
        text = "Benchmark headline: company reports mixed earnings and guides cautiously."
        try:
            import asyncio

            return await asyncio.to_thread(self._bench_finbert_sync, text)
        except Exception:
            return None

    async def _benchmark_ollama_latency_ms(self) -> float | None:
        # Measure Ollama inference latency by generating a short response
        base = self.ollama_url.rsplit("/api/tags", 1)[0] if "/api/tags" in self.ollama_url else "http://localhost:11434"
        gen_url = f"{base}/api/generate"
        bench_model = os.getenv("OLLAMA_BENCH_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2:1b")).strip()
        payload = {
            "model": bench_model,
            "prompt": "Summarize market mood in one short sentence.",
            "stream": False,
            "options": {"num_predict": 8},
        }
        try:
            timeout = httpx.Timeout(max(0.5, float(os.getenv("ROUTER_OLLAMA_BENCH_TIMEOUT_SEC", "6.0"))))
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(gen_url, json=payload)
            if resp.status_code >= 400:
                return None
            return float((time.perf_counter() - t0) * 1000.0)
        except Exception:
            return None

    async def run_diagnostics(self) -> Dict[str, Any]:
        """
        Runs during server startup to determine capability tier.
        """
        # Query hardware resources and service availability
        total_ram, available_ram = self._read_ram()
        gpu_available = self._read_gpu_available()
        ollama_active = await self._ping_ollama()

        self.total_ram_gb = float(total_ram)
        self.available_ram_gb = float(available_ram)
        self.gpu_available = bool(gpu_available)
        self.ollama_available = bool(ollama_active)
        self.ollama_inference_ms = None
        self.finbert_inference_ms = None

        # Determine system capability tier based on available resources
        if gpu_available and available_ram > 8.0 and ollama_active:
            self.system_capability = "high"
            logger.info("Model router: high tier enabled (GPU + RAM + Ollama).")
        elif available_ram > 4.0:
            self.system_capability = "medium"
            logger.info("Model router: medium tier enabled (CPU FinBERT path available).")
        else:
            self.system_capability = "low"
            logger.warning("Model router: low tier enabled (fallback NLP path).")

        # Optionally run latency benchmarks and adjust tier based on performance thresholds
        run_bench = os.getenv("ROUTER_ENABLE_MICRO_BENCH", "1").strip().lower() in ("1", "true", "yes", "on")
        max_finbert_ms = float(os.getenv("ROUTER_FINBERT_MAX_MS", "500"))
        max_ollama_ms = float(os.getenv("ROUTER_OLLAMA_MAX_MS", "6000"))

        if run_bench:
            self.ollama_inference_ms = await self._benchmark_ollama_latency_ms()
            self.finbert_inference_ms = await self._benchmark_finbert_latency_ms()

            if self.ollama_inference_ms is not None and self.ollama_inference_ms > max_ollama_ms:
                self.ollama_available = False

            if self.finbert_inference_ms is None or self.finbert_inference_ms > max_finbert_ms:
                # Enforce latency budget over raw hardware presence.
                self.system_capability = "low"

        # Select sentiment pipeline based on system capability tier
        if self.system_capability in ("high", "medium"):
            self.sentiment_pipeline = "cascade"
        else:
            self.sentiment_pipeline = "lexicon"

        return {
            "tier": self.system_capability,
            "ram_total_gb": round(self.total_ram_gb, 2),
            "ram_available_gb": round(self.available_ram_gb, 2),
            "gpu": self.gpu_available,
            "ollama_active": self.ollama_available,
            "sentiment_pipeline": self.sentiment_pipeline,
            "latency_ms": {
                "finbert": self.finbert_inference_ms,
                "ollama": self.ollama_inference_ms,
            },
            "psutil_available": PSUTIL_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
        }

    def get_sentiment_pipeline(self) -> str:
        """
        Placeholder routing decision for sentiment engine selection.
        Replace return values with real pipeline classes as needed.
        """
        # Validate and return selected sentiment pipeline based on system tier
        pipeline = str(self.sentiment_pipeline or "").strip().lower()
        if pipeline in ("lexicon", "finbert", "cascade"):
            return pipeline
        if self.system_capability in ("high", "medium"):
            return "cascade"
        return "lexicon"


intelligence_router = IntelligenceRouter()
