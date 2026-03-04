from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class Bucket:
	capacity: int
	refill_per_sec: float
	tokens: float
	last: float


class RateLimiter:
	"""
	Token bucket limiter per 'service' key.
	Example: 60 requests/min => capacity=60, refill_per_sec=1.0
	"""

	def __init__(self) -> None:
		self._buckets: Dict[str, Bucket] = {}
		self._locks: Dict[str, asyncio.Lock] = {}

	def configure(self, key: str, per_minute: int) -> None:
		per_minute = max(1, int(per_minute))
		self._buckets[key] = Bucket(
			capacity=per_minute,
			refill_per_sec=per_minute / 60.0,
			tokens=float(per_minute),
			last=time.time(),
		)
		self._locks[key] = asyncio.Lock()

	async def acquire(self, key: str, cost: float = 1.0) -> None:
		if key not in self._buckets:
			self.configure(key, per_minute=30)

		lock = self._locks[key]
		async with lock:
			b = self._buckets[key]
			now = time.time()
			elapsed = max(0.0, now - b.last)
			b.tokens = min(b.capacity, b.tokens + elapsed * b.refill_per_sec)
			b.last = now

			if b.tokens >= cost:
				b.tokens -= cost
				return

			needed = cost - b.tokens
			wait = needed / b.refill_per_sec if b.refill_per_sec > 0 else 2.0

		await asyncio.sleep(wait)
		await self.acquire(key, cost=cost)


limiter = RateLimiter()
