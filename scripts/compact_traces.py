"""Compact the trace store for release.

DuckDB does not reclaim space in place, so a file rewritten by repeated sweeps
grows well past the size of the data in it — 103 MB here for about 37 MB of
content. Copying into a fresh database compacts it.

The release copy also truncates the stored prompt to its hash plus a short
prefix. Prompts are the bulk of the store (25 MB of 37 MB) and they are fully
regenerable: ``make experiment`` rebuilds them from the frozen constants, and
the hash is what identifies a call anyway. Everything ``make eval`` reads —
runs, turns, belief snapshots, token counts, parse outcomes — is preserved
exactly, which is what the reproducibility contract actually needs.

The full-fidelity store is what a live run produces; this is the artefact that
ships.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import duckdb

#: Characters of each prompt kept for eyeballing. The hash identifies the call.
PROMPT_PREFIX = 160


def compact(source: Path, destination: Path, keep_prompts: bool = False) -> tuple[int, int]:
    if destination.exists():
        destination.unlink()

    src = duckdb.connect(str(source), read_only=True)
    dst = duckdb.connect(str(destination))
    try:
        from probe.runtime.tracing import SCHEMA

        dst.execute(SCHEMA)
        # ATTACH does not accept bind parameters, so the path is inlined with
        # quotes escaped. Paths here come from argv, not from user data.
        escaped = str(source).replace("'", "''")
        dst.execute(f"ATTACH '{escaped}' AS src (READ_ONLY)")
        dst.execute("INSERT INTO runs SELECT * FROM src.runs")
        dst.execute("INSERT INTO turns SELECT * FROM src.turns")

        prompt_expr = (
            "prompt"
            if keep_prompts
            else f"substr(prompt, 1, {PROMPT_PREFIX}) || '… [truncated for release; "
            "regenerate with `make experiment`]'"
        )
        dst.execute(
            f"""
            INSERT INTO llm_calls
            SELECT call_id, run_id, role, {prompt_expr}, prompt_hash, model, seed,
                   temperature, raw_output, parsed_ok, repair_attempt,
                   prompt_tokens, completion_tokens, latency_ms
            FROM src.llm_calls
            """
        )
        dst.execute("DETACH src")
    finally:
        src.close()
        dst.close()

    return source.stat().st_size, destination.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compact the trace store for release.")
    parser.add_argument("--source", default="traces/probe.duckdb")
    parser.add_argument("--keep-prompts", action="store_true")
    parser.add_argument(
        "--in-place", action="store_true", help="Replace the source with the compacted copy."
    )
    args = parser.parse_args(argv)

    source = Path(args.source)
    destination = source.with_suffix(".compact.duckdb")
    before, after = compact(source, destination, keep_prompts=args.keep_prompts)

    print(f"{source}: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB ({after / before:.0%})")
    if args.in_place:
        shutil.move(str(destination), str(source))
        print(f"replaced {source}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
