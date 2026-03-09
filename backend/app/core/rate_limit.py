from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict


# Stores runtime state for one token bucket.
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
		# Keeps a separate bucket per service key.
		self._buckets: Dict[str, Bucket] = {}
		# Uses one async lock per key to avoid race conditions.
		self._locks: Dict[str, asyncio.Lock] = {}

	def configure(self, key: str, per_minute: int) -> None:
		# Normalizes invalid limits to at least 1 request per minute.
		per_minute = max(1, int(per_minute))
		# Initializes a full token bucket with time-based refill settings.
		self._buckets[key] = Bucket(
			capacity=per_minute,
			refill_per_sec=per_minute / 60.0,
			tokens=float(per_minute),
			last=time.time(),
		)
		# Creates a lock for serializing updates to this bucket.
		self._locks[key] = asyncio.Lock()

	async def acquire(self, key: str, cost: float = 1.0) -> None:
		# Creates a default limiter config for unseen keys.
		if key not in self._buckets:
			self.configure(key, per_minute=30)

		# Locks per key so token math stays consistent across coroutines.
		lock = self._locks[key]
		async with lock:
			# Reads current bucket state and computes elapsed refill time.
			b = self._buckets[key]
			now = time.time()
			elapsed = max(0.0, now - b.last)
			# Refills tokens up to capacity, then updates timestamp.
			b.tokens = min(b.capacity, b.tokens + elapsed * b.refill_per_sec)
			b.last = now

			# Consumes tokens immediately when enough budget is available.
			if b.tokens >= cost:
				b.tokens -= cost
				return

			# Computes wait time needed to accumulate the missing tokens.
			needed = cost - b.tokens
			wait = needed / b.refill_per_sec if b.refill_per_sec > 0 else 2.0

		# Sleeps outside the lock and retries until tokens are available.
		await asyncio.sleep(wait)
		await self.acquire(key, cost=cost)


# Exposes a shared module-level limiter instance.
limiter = RateLimiter()
