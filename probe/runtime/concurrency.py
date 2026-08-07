"""Running many interviews at once.

At ~800 runs a serial sweep is an overnight job dominated by provider latency,
so interviews are fanned out under a bounded semaphore. Three properties are
non-negotiable and all three are tested:

* **Traces do not interleave.** Each interview owns its own traced client and
  its own ``run_id``; the store serialises writes behind a lock. A trace with
  turns from two runs mixed together would be undetectable later and would
  poison every metric.
* **Rate limits are survivable.** 429s are retried with exponential backoff
  and *jittered* delays — synchronised retries from a fan-out are what turn a
  rate limit into a retry storm.
* **One failed interview does not kill the sweep.** A run that raises is
  recorded as failed and the others continue.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from probe.runtime.llm import LLMRequest, LLMResponse

DEFAULT_CONCURRENCY = 8


class RateLimitError(RuntimeError):
    """Raised by the fault-injection wrapper; stands in for a provider 429."""


class RetryingClient:
    """Backoff wrapper around any client.

    Sits *below* the repair ladder: this layer owns transport failures, the
    ladder above owns malformed-but-delivered output. Keeping them separate
    means a rate limit never consumes a repair attempt.
    """

    def __init__(
        self,
        inner,
        *,
        max_retries: int = 5,
        base_delay: float = 0.01,
        sleep: Callable[[float], None] = time.sleep,
        seed: int = 0,
    ) -> None:
        self.inner = inner
        self.name = f"{inner.name}+retry"
        self.model = inner.model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._sleep = sleep
        self._rng = random.Random(seed)
        self.retries = 0
        self.delays: list[float] = []

    def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.inner.complete(request, **kwargs)
            except RateLimitError as exc:
                last = exc
                if attempt == self.max_retries - 1:
                    break
                self.retries += 1
                delay = self.base_delay * (2**attempt) * (0.5 + self._rng.random())
                self.delays.append(delay)
                self._sleep(delay)
        raise last if last else RuntimeError("retry loop exited without an error")


class FlakyClient:
    """Fault injection: fails a deterministic fraction of calls with a 429.

    Used by the concurrency test. Deterministic rather than random-per-run so a
    failure reproduces.
    """

    def __init__(self, inner, failure_rate: float = 0.25, seed: int = 0) -> None:
        self.inner = inner
        self.name = f"{inner.name}+flaky"
        self.model = inner.model
        self.failure_rate = failure_rate
        self._seed = seed
        self._n = 0
        self.injected = 0

    def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        self._n += 1
        rng = random.Random(f"{self._seed}:{self._n}")
        if rng.random() < self.failure_rate:
            self.injected += 1
            raise RateLimitError("429 Too Many Requests (injected)")
        return self.inner.complete(request, **kwargs)


@dataclass
class SweepOutcome:
    completed: list[Any] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    wallclock_seconds: float = 0.0

    @property
    def n_total(self) -> int:
        return len(self.completed) + len(self.failures)

    @property
    def throughput_per_minute(self) -> float:
        if self.wallclock_seconds <= 0:
            return 0.0
        return 60.0 * len(self.completed) / self.wallclock_seconds


async def _run_all(
    jobs: Sequence[tuple[str, Callable[[], Any]]], concurrency: int
) -> SweepOutcome:
    semaphore = asyncio.Semaphore(concurrency)
    outcome = SweepOutcome()

    async def one(label: str, fn: Callable[[], Any]) -> None:
        async with semaphore:
            try:
                # to_thread rather than a bare call: the work is synchronous,
                # and running it inline would serialise the whole sweep behind
                # the event loop.
                outcome.completed.append(await asyncio.to_thread(fn))
            except Exception as exc:  # noqa: BLE001 - one bad run must not end the sweep
                outcome.failures.append((label, f"{type(exc).__name__}: {exc}"))

    started = time.perf_counter()
    await asyncio.gather(*(one(label, fn) for label, fn in jobs))
    outcome.wallclock_seconds = time.perf_counter() - started
    return outcome


def run_concurrently(
    jobs: Sequence[tuple[str, Callable[[], Any]]],
    concurrency: int = DEFAULT_CONCURRENCY,
) -> SweepOutcome:
    """Run ``jobs`` with bounded parallelism. Each job is ``(label, callable)``."""
    return asyncio.run(_run_all(jobs, concurrency))
