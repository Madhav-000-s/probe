"""The repair loop.

Models emit malformed JSON. Not often, but at 800 interviews × 12 turns × 3
calls per turn, "not often" is still hundreds of times, and an unhandled
``ValidationError` in the middle of a sweep costs a night. So every structured
call walks a four-stage ladder and the run continues regardless:

1. **parse** the first response;
2. **repair** — re-prompt with the validator's own error message appended,
   which is far more effective than a generic "return valid JSON" nudge;
3. **degrade** to a deterministic non-model path (for a grade: the neutral
   score with an empty-flagged rationale), so the turn still yields a usable
   record;
4. **give up** — mark the turn ``unrecoverable`` and keep going.

The stage that fired is recorded. Schema-violation and repair-success rates are
a reported robustness metric, not a hidden implementation detail, so this
ladder is instrumented rather than silent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from probe.runtime.llm import LLMRequest

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ParseOutcome(StrEnum):
    OK = "ok"
    REPAIRED = "repaired"
    DEGRADED = "degraded"
    UNRECOVERABLE = "unrecoverable"


@dataclass
class ParseResult:
    value: Any | None
    outcome: ParseOutcome
    attempts: int = 1
    errors: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome in (ParseOutcome.OK, ParseOutcome.REPAIRED)

    @property
    def usable(self) -> bool:
        return self.value is not None


def extract_json(text: str) -> str:
    """Pull the JSON object out of a response that may be wrapped in prose or
    a code fence. Deliberately forgiving — a correct grade wrapped in "Here you
    go:" is a formatting problem, not a grading problem."""
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    if text.startswith("{"):
        return text
    m = _OBJECT.search(text)
    return m.group(0) if m else text


def parse_model(model_cls: type[T], text: str) -> tuple[T | None, str | None]:
    """Parse ``text`` into ``model_cls``. Returns ``(value, error_message)``."""
    try:
        payload = json.loads(extract_json(text))
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"expected a JSON object, got {type(payload).__name__}"
    try:
        return model_cls.model_validate(payload), None
    except ValidationError as exc:
        return None, _compact_validation_error(exc)


def _compact_validation_error(exc: ValidationError) -> str:
    """One line per bad field. The repair prompt carries this verbatim; a
    model told exactly which field failed and why fixes it far more reliably
    than one told the output was invalid."""
    parts = []
    for err in exc.errors()[:6]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def build_repair_prompt(original: str, error: str, schema_hint: str) -> str:
    return (
        f"{original}\n\n"
        "--- REPAIR ---\n"
        "Your previous response could not be parsed. The validator reported:\n"
        f"  {error}\n\n"
        "Return ONLY a JSON object matching this schema, with no prose and no "
        "code fence:\n"
        f"{schema_hint}\n"
    )


def schema_hint(model_cls: type[BaseModel]) -> str:
    schema = model_cls.model_json_schema()
    return json.dumps(
        {"required": schema.get("required", []), "properties": schema.get("properties", {})},
        indent=None,
    )[:1200]


def structured_call(
    client: Any,
    request: LLMRequest,
    model_cls: type[T],
    *,
    degraded: Callable[[], T] | None = None,
    postcheck: Callable[[T], str | None] | None = None,
    max_repairs: int = 1,
) -> ParseResult:
    """Run one structured call through the full ladder.

    ``postcheck`` runs *after* schema validation and returns an error string
    when a semantically-invalid-but-schema-valid response should be rejected.
    The grader uses it to reject grades whose evidence spans do not actually
    resolve against the answer text: schema-valid, meaning-invalid, and
    exactly the class of failure that quietly poisons an audit trail.
    """
    errors: list[str] = []
    raws: list[str] = []
    attempts = 0
    prompt = request.prompt

    for attempt in range(max_repairs + 1):
        attempts += 1
        req = (
            request
            if attempt == 0
            else LLMRequest(
                role=request.role,
                prompt=prompt,
                seed=request.seed + attempt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                context={**request.context, "repair_attempt": attempt},
                run_id=request.run_id,
            )
        )
        response = _complete(client, req, repair_attempt=attempt)
        raws.append(response.text)
        value, error = parse_model(model_cls, response.text)
        if value is not None and postcheck is not None:
            error = postcheck(value)
            if error:
                value = None
        if value is not None:
            _mark(client, True)
            return ParseResult(
                value=value,
                outcome=ParseOutcome.OK if attempt == 0 else ParseOutcome.REPAIRED,
                attempts=attempts,
                errors=errors,
                raw=raws,
            )
        _mark(client, False)
        errors.append(error or "unknown parse failure")
        prompt = build_repair_prompt(request.prompt, errors[-1], schema_hint(model_cls))

    if degraded is not None:
        try:
            return ParseResult(
                value=degraded(),
                outcome=ParseOutcome.DEGRADED,
                attempts=attempts,
                errors=errors,
                raw=raws,
            )
        except Exception as exc:  # pragma: no cover - degraded paths are trivial
            errors.append(f"degraded path failed: {exc}")

    return ParseResult(
        value=None, outcome=ParseOutcome.UNRECOVERABLE, attempts=attempts, errors=errors, raw=raws
    )


def _complete(client: Any, request: LLMRequest, repair_attempt: int):
    try:
        return client.complete(request, repair_attempt=repair_attempt)
    except TypeError:
        # Plain LLMClient implementations do not take the tracing kwargs.
        return client.complete(request)


def _mark(client: Any, ok: bool) -> None:
    marker = getattr(client, "mark_parse_result", None)
    if callable(marker):
        marker(ok)


@dataclass
class RepairStats:
    """Aggregated ladder outcomes for one run. Feeds the schema-violation and
    repair-success robustness numbers."""

    total: int = 0
    ok: int = 0
    repaired: int = 0
    degraded: int = 0
    unrecoverable: int = 0

    def observe(self, result: ParseResult) -> None:
        self.total += 1
        setattr(self, result.outcome.value, getattr(self, result.outcome.value) + 1)

    @property
    def violation_rate(self) -> float:
        """Fraction of calls whose first response failed validation."""
        if self.total == 0:
            return 0.0
        return (self.repaired + self.degraded + self.unrecoverable) / self.total

    @property
    def repair_success_rate(self) -> float:
        """Of the calls that failed first time, the fraction the repair prompt
        rescued before the degraded path was needed."""
        failed = self.repaired + self.degraded + self.unrecoverable
        return self.repaired / failed if failed else 1.0
