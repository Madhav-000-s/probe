"""The trace store.

Every LLM call and every turn lands in DuckDB before the next turn begins.
Two properties follow, and both are tested:

* **Reconstructability** — a run can be re-rendered from the store and diffed
  byte-for-byte against the live transcript. If they differ, the trace is
  lying and every metric computed from it is suspect.
* **Resumability** — a killed run restarts from the last committed turn, keyed
  on ``(run_id, turn_idx)``. Writes are upserts, so a turn that was persisted
  just before the kill does not become a duplicate on resume.

DuckDB rather than SQLite because the dominant read path is analytics: the
metric modules scan every turn of every run and group by arm.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

from probe.models import (
    BeliefSnapshot,
    Grade,
    LLMCallRecord,
    LLMRole,
    RunRecord,
    StopReason,
    Transcript,
    Turn,
)
from probe.runtime.llm import LLMClient, LLMRequest, LLMResponse

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id             VARCHAR PRIMARY KEY,
    arm                VARCHAR NOT NULL,
    persona_id         VARCHAR NOT NULL,
    style_id           VARCHAR NOT NULL,
    bank_version       VARCHAR NOT NULL,
    population_version VARCHAR NOT NULL,
    code_commit        VARCHAR NOT NULL,
    seed               BIGINT  NOT NULL,
    stop_reason        VARCHAR,
    n_turns            INTEGER NOT NULL DEFAULT 0,
    total_tokens       BIGINT  NOT NULL DEFAULT 0,
    wallclock_seconds  DOUBLE  NOT NULL DEFAULT 0.0,
    usd_cost           DOUBLE  NOT NULL DEFAULT 0.0,
    followups_enabled  BOOLEAN NOT NULL DEFAULT TRUE,
    style_separation   BOOLEAN NOT NULL DEFAULT TRUE,
    grader_model       VARCHAR NOT NULL DEFAULT 'sim-grader',
    completed          BOOLEAN NOT NULL DEFAULT FALSE,
    partial            BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS turns (
    run_id            VARCHAR NOT NULL,
    turn_idx          INTEGER NOT NULL,
    question_id       VARCHAR NOT NULL,
    competency_id     VARCHAR NOT NULL,
    question_text     VARCHAR NOT NULL,
    answer            VARCHAR NOT NULL,
    grade_json        VARCHAR,
    belief_json       VARCHAR NOT NULL,
    eig_at_selection  DOUBLE,
    selection_reason  VARCHAR NOT NULL DEFAULT '',
    elapsed_seconds   DOUBLE  NOT NULL DEFAULT 0.0,
    tokens_used       INTEGER NOT NULL DEFAULT 0,
    unrecoverable     BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (run_id, turn_idx)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id           VARCHAR PRIMARY KEY,
    run_id            VARCHAR,
    role              VARCHAR NOT NULL,
    prompt            VARCHAR NOT NULL,
    prompt_hash       VARCHAR NOT NULL,
    model             VARCHAR NOT NULL,
    seed              BIGINT  NOT NULL,
    temperature       DOUBLE  NOT NULL DEFAULT 0.0,
    raw_output        VARCHAR NOT NULL,
    parsed_ok         BOOLEAN NOT NULL,
    repair_attempt    INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms        DOUBLE  NOT NULL DEFAULT 0.0
);
"""


class TraceStore:
    """Single-writer DuckDB handle guarded by a lock.

    The lock rather than a connection pool because the async runner fans out
    interviews inside one process; DuckDB is happiest with one writer and the
    contention is negligible next to model latency.
    """

    def __init__(self, path: str | Path = ":memory:", read_only: bool = False) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = duckdb.connect(self.path, read_only=read_only)
        if not read_only:
            self._con.execute(SCHEMA)

    # ------------------------------------------------------------- lifecycle

    def close(self) -> None:
        with self._lock:
            self._con.close()

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def cursor(self) -> Iterator[duckdb.DuckDBPyConnection]:
        with self._lock:
            yield self._con

    # ----------------------------------------------------------------- write

    def upsert_run(self, run: RunRecord) -> None:
        with self.cursor() as con:
            con.execute("DELETE FROM runs WHERE run_id = ?", [run.run_id])
            con.execute(
                """INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    run.run_id,
                    run.arm,
                    run.persona_id,
                    run.style_id,
                    run.bank_version,
                    run.population_version,
                    run.code_commit,
                    run.seed,
                    run.stop_reason.value if run.stop_reason else None,
                    run.n_turns,
                    run.total_tokens,
                    run.wallclock_seconds,
                    run.usd_cost,
                    run.followups_enabled,
                    run.style_separation,
                    run.grader_model,
                    run.completed,
                    run.partial,
                ],
            )

    def upsert_turn(self, turn: Turn) -> None:
        """Idempotent by ``(run_id, turn_idx)``. Replaying a turn after a crash
        overwrites rather than duplicates — the resumability test asserts
        exactly one row per turn."""
        with self.cursor() as con:
            con.execute(
                "DELETE FROM turns WHERE run_id = ? AND turn_idx = ?",
                [turn.run_id, turn.turn_idx],
            )
            con.execute(
                """INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    turn.run_id,
                    turn.turn_idx,
                    turn.question_id,
                    turn.competency_id,
                    turn.question_text,
                    turn.answer,
                    turn.grade.model_dump_json() if turn.grade else None,
                    turn.belief_after.model_dump_json(),
                    turn.eig_at_selection,
                    turn.selection_reason,
                    turn.elapsed_seconds,
                    turn.tokens_used,
                    turn.unrecoverable,
                ],
            )

    def record_call(self, record: LLMCallRecord) -> None:
        with self.cursor() as con:
            con.execute(
                """INSERT OR REPLACE INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    record.call_id,
                    record.run_id,
                    record.role.value,
                    record.prompt,
                    record.prompt_hash,
                    record.model,
                    record.seed,
                    record.temperature,
                    record.raw_output,
                    record.parsed_ok,
                    record.repair_attempt,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.latency_ms,
                ],
            )

    # ------------------------------------------------------------------ read

    def last_turn_idx(self, run_id: str) -> int:
        with self.cursor() as con:
            row = con.execute(
                "SELECT max(turn_idx) FROM turns WHERE run_id = ?", [run_id]
            ).fetchone()
        return -1 if row is None or row[0] is None else int(row[0])

    def run_exists(self, run_id: str) -> bool:
        with self.cursor() as con:
            row = con.execute("SELECT 1 FROM runs WHERE run_id = ?", [run_id]).fetchone()
        return row is not None

    def is_complete(self, run_id: str) -> bool:
        with self.cursor() as con:
            row = con.execute(
                "SELECT completed FROM runs WHERE run_id = ?", [run_id]
            ).fetchone()
        return bool(row and row[0])

    def load_run(self, run_id: str) -> RunRecord | None:
        with self.cursor() as con:
            row = con.execute(
                "SELECT * FROM runs WHERE run_id = ?", [run_id]
            ).fetchone()
            cols = [d[0] for d in con.description]
        if row is None:
            return None
        data = dict(zip(cols, row, strict=True))
        if data.get("stop_reason"):
            data["stop_reason"] = StopReason(data["stop_reason"])
        return RunRecord(**data)

    def load_turns(self, run_id: str) -> list[Turn]:
        with self.cursor() as con:
            rows = con.execute(
                "SELECT * FROM turns WHERE run_id = ? ORDER BY turn_idx", [run_id]
            ).fetchall()
            cols = [d[0] for d in con.description]
        out: list[Turn] = []
        for row in rows:
            d = dict(zip(cols, row, strict=True))
            grade_json = d.pop("grade_json")
            belief_json = d.pop("belief_json")
            out.append(
                Turn(
                    **d,
                    grade=Grade.model_validate_json(grade_json) if grade_json else None,
                    belief_after=BeliefSnapshot.model_validate_json(belief_json),
                )
            )
        return out

    def load_transcript(self, run_id: str) -> Transcript:
        run = self.load_run(run_id)
        if run is None:
            raise KeyError(run_id)
        return Transcript(
            run_id=run_id,
            candidate_id=run.persona_id,
            arm=run.arm,
            turns=self.load_turns(run_id),
        )

    def run_ids(self, arm: str | None = None, completed_only: bool = True) -> list[str]:
        sql = "SELECT run_id FROM runs WHERE 1=1"
        params: list[Any] = []
        if arm is not None:
            sql += " AND arm = ?"
            params.append(arm)
        if completed_only:
            sql += " AND completed"
        sql += " ORDER BY run_id"
        with self.cursor() as con:
            return [r[0] for r in con.execute(sql, params).fetchall()]

    def df(self, sql: str, params: list[Any] | None = None):
        """Escape hatch for the metric modules and notebooks."""
        with self.cursor() as con:
            return con.execute(sql, params or []).fetch_df()

    def all_prompts(self) -> list[tuple[str, str]]:
        """``(role, prompt)`` for every recorded call. The ground-truth
        firewall test scans this."""
        with self.cursor() as con:
            return [
                (r[0], r[1])
                for r in con.execute("SELECT role, prompt FROM llm_calls").fetchall()
            ]

    def counts(self) -> dict[str, int]:
        with self.cursor() as con:
            return {
                t: int(con.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
                for t in ("runs", "turns", "llm_calls")
            }


class TracedClient:
    """Decorates an :class:`~probe.runtime.llm.LLMClient` so that every call is
    persisted before its result is used.

    Ordering matters: the record is written *after* the response arrives but
    *before* the caller parses it, so a parse failure still leaves the raw
    output on disk. Debugging a repair loop without the raw output is
    guesswork.
    """

    def __init__(
        self,
        inner: LLMClient,
        store: TraceStore | None = None,
        run_id: str | None = None,
    ) -> None:
        self.inner = inner
        self.store = store
        self.run_id = run_id
        self.name = inner.name
        self.model = inner.model
        self.calls: list[LLMCallRecord] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._consumed_tokens = 0

    def bind_run(self, run_id: str) -> None:
        self.run_id = run_id

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def take_token_delta(self) -> int:
        """Tokens spent since the last call to this method.

        The interview loop charges the budget per turn, so it needs a delta
        rather than a running total; keeping the bookkeeping here means the
        loop never reaches into the client's internals.
        """
        delta = self.total_tokens - self._consumed_tokens
        self._consumed_tokens = self.total_tokens
        return delta

    def complete(
        self, request: LLMRequest, *, parsed_ok: bool | None = None, repair_attempt: int = 0
    ) -> LLMResponse:
        response = self.inner.complete(request)
        record = LLMCallRecord(
            call_id=uuid.uuid4().hex,
            run_id=request.run_id or self.run_id,
            role=request.role,
            prompt=request.prompt,
            prompt_hash=request.hash,
            model=response.model,
            seed=request.seed,
            temperature=request.temperature,
            raw_output=response.text,
            parsed_ok=True if parsed_ok is None else parsed_ok,
            repair_attempt=repair_attempt,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )
        self.calls.append(record)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        if self.store is not None:
            self.store.record_call(record)
        return response

    def mark_parse_result(self, ok: bool) -> None:
        """Amend the most recent record once the caller knows whether the
        output actually parsed."""
        if not self.calls:
            return
        self.calls[-1].parsed_ok = ok
        if self.store is not None:
            self.store.record_call(self.calls[-1])

    def role_counts(self) -> dict[LLMRole, int]:
        out: dict[LLMRole, int] = {}
        for c in self.calls:
            out[c.role] = out.get(c.role, 0) + 1
        return out


def new_run_id(arm: str, persona_id: str, style_id: str, seed: int, suffix: str = "") -> str:
    """Deterministic run ids. Deterministic because resume has to be able to
    recompute the id of the run it is resuming without consulting the store."""
    base = f"{arm}.{persona_id}.{style_id}.s{seed}"
    return f"{base}.{suffix}" if suffix else base


def json_or_empty(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
