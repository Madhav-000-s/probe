"""The provider boundary.

Everything that talks to a model goes through :class:`LLMClient`. There are
three implementations:

``fake``
    Canned responses keyed by prompt hash. Used by unit and integration tests
    so they are hermetic and instant.
``sim``
    A deterministic, seeded generative model of every role (see
    :mod:`probe.sim.llm_sim`). This is what the committed traces were produced
    with; it makes ``make eval`` reproducible at zero API cost. Its limits are
    stated first in the README.
``anthropic``
    The real thing. Activated by ``ANTHROPIC_API_KEY``; swapping to it is a
    config flag, which is what makes the small/large grader ablation cheap.

There is deliberately no agent framework anywhere near this. The control flow
*is* the interview policy and has to be explainable line by line.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from probe.models import LLMRole


def prompt_hash(role: LLMRole, prompt: str, seed: int) -> str:
    """Stable identity for a call. Two calls with the same hash must produce
    the same output under any deterministic backend — that property is what
    the reproducibility contract rests on."""
    h = hashlib.sha256()
    h.update(role.value.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(seed).encode("utf-8"))
    return h.hexdigest()[:32]


@dataclass(frozen=True)
class LLMRequest:
    role: LLMRole
    prompt: str
    seed: int = 0
    temperature: float = 0.0
    max_tokens: int = 1024
    #: Free-form structured payload the sim backend reads instead of parsing
    #: the natural-language prompt back apart. A real provider ignores it, so
    #: nothing on the interview plane may put ground truth here — the firewall
    #: test checks this field too.
    context: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    @property
    def hash(self) -> str:
        return prompt_hash(self.role, self.prompt, self.seed)


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMClient(Protocol):
    """The whole provider surface. Four lines is the point."""

    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token).

    Used by the offline backends so cost and budget accounting exercise the
    same code paths they will under a real provider, where the number comes
    from the API response instead.
    """
    return max(1, len(text) // 4)


class FakeLLM:
    """Deterministic canned-response client.

    Responses are looked up by prompt hash, then by role, then fall back to a
    registered default. An unmatched call raises rather than inventing output:
    a test that silently got a placeholder grade is a test that proves nothing.
    """

    name = "fake"
    model = "fake-1"

    #: A queued response is either a literal string or a function of the
    #: request. Callables exist so a fixture can echo real character offsets
    #: back out of the answer it was given — a canned grade with hard-coded
    #: spans would fail the grader's own span validator, which is the one
    #: check most worth exercising.
    Responder = str | Callable[["LLMRequest"], str]

    def __init__(
        self,
        by_hash: dict[str, str] | None = None,
        by_role: dict[LLMRole, Responder | list[Responder]] | None = None,
        strict: bool = True,
    ) -> None:
        self.by_hash = dict(by_hash or {})
        self.by_role: dict[LLMRole, list[Any]] = {}
        for role, val in (by_role or {}).items():
            self.by_role[role] = list(val) if isinstance(val, list) else [val]
        self.strict = strict
        self.calls: list[LLMRequest] = []
        self._role_cursor: dict[LLMRole, int] = {}

    def register(self, role: LLMRole, payload: Any) -> None:
        """Queue a response for ``role``. Dicts are JSON-encoded and callables
        are kept as-is, so tests can hand over a model dump directly."""
        if not isinstance(payload, str) and not callable(payload):
            payload = json.dumps(payload)
        self.by_role.setdefault(role, []).append(payload)

    def register_hash(self, request: LLMRequest, payload: Any) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        self.by_hash[request.hash] = text

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        text = self._lookup(request)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=estimate_tokens(request.prompt),
            completion_tokens=estimate_tokens(text),
            latency_ms=0.0,
        )

    def _lookup(self, request: LLMRequest) -> str:
        if request.hash in self.by_hash:
            return self.by_hash[request.hash]
        queue = self.by_role.get(request.role)
        if queue:
            # Cycle rather than exhaust: a fixed-policy interview asks n
            # questions and a test should not have to enumerate all n.
            idx = self._role_cursor.get(request.role, 0)
            self._role_cursor[request.role] = idx + 1
            entry = queue[idx % len(queue)]
            return entry(request) if callable(entry) else entry
        if self.strict:
            raise KeyError(
                f"FakeLLM has no response for role={request.role.value} hash={request.hash}"
            )
        return "{}"


class NullLLM:
    """Records calls, returns empty JSON. Used only by the firewall test, which
    cares about what was *sent*, not what came back."""

    name = "null"
    model = "null-0"

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(text="{}", model=self.model)


class LatencySimulator:
    """Wraps a client and charges it wall-clock time without actually
    sleeping, so budget-ceiling logic is testable in milliseconds."""

    def __init__(self, inner: LLMClient, ms_per_call: float = 250.0) -> None:
        self.inner = inner
        self.ms_per_call = ms_per_call
        self.name = f"{inner.name}+latency"
        self.model = inner.model
        self.virtual_ms = 0.0

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        resp = self.inner.complete(request)
        self.virtual_ms += self.ms_per_call
        resp.latency_ms = self.ms_per_call + (time.perf_counter() - started) * 1000.0
        return resp


def get_client(backend: str, *, seed: int = 0, **kwargs: Any) -> LLMClient:
    """Backend factory. Imports are lazy so a machine with no ``anthropic``
    package installed can still run the entire offline pipeline."""
    backend = backend.lower()
    if backend == "fake":
        return FakeLLM(strict=kwargs.pop("strict", False))
    if backend == "null":
        return NullLLM()
    if backend == "sim":
        from probe.sim.llm_sim import SimLLM

        return SimLLM(seed=seed, **kwargs)
    if backend == "anthropic":
        from probe.runtime.anthropic_client import AnthropicClient

        return AnthropicClient(**kwargs)
    raise ValueError(f"unknown backend {backend!r} (expected sim|fake|null|anthropic)")
