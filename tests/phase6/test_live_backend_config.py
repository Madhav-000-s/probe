"""Credential and model wiring for the live backend.

The offline pipeline is the one the committed numbers come from, so this suite
is small: it checks that supplying a key and choosing a model works the way the
README says it does, and — the part worth a test — that neither can leak.
"""

from __future__ import annotations

import os

import pytest

from probe.config import ROOT, load_dotenv
from probe.experiment import SweepPlan
from probe.runtime.anthropic_client import PRICING, estimate_cost, resolve_model

# ------------------------------------------------------------------ .env

def test_dotenv_sets_only_what_is_missing(tmp_path, monkeypatch):
    """An exported variable beats the file. A stale `.env` silently overriding
    the key you just exported is exactly the debugging session nobody wants."""
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "\n"
        "ANTHROPIC_API_KEY='sk-from-file'\n"
        "export PROBE_MODEL=haiku\n"
        "MALFORMED\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-shell")
    monkeypatch.delenv("PROBE_MODEL", raising=False)

    loaded = load_dotenv(env)

    assert loaded == ["PROBE_MODEL"]
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-from-shell"
    assert os.environ["PROBE_MODEL"] == "haiku"


def test_dotenv_is_absent_and_ignored(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []


def test_every_entry_point_loads_the_env_file():
    """The CLI loaded `.env`; the eval entry points are separate `python -m`
    modules and did not, so `--backend anthropic` on a suite failed with "key
    not set" while the identical key worked through `probe`."""
    import ast

    missing = []
    for name in ("evals/run_eval.py", "evals/run_suites.py", "evals/build_gold.py"):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        main = next(
            (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None
        )
        assert main is not None, f"{name} has no main()"
        calls = {
            n.func.id
            for n in ast.walk(main)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if "load_dotenv" not in calls:
            missing.append(name)
    assert not missing, f"entry points that never load .env: {missing}"


def test_dotenv_is_not_committed_but_the_example_is():
    """A key in the history is a key that has to be rotated."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", ".env", ".env.example"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert ".env" not in tracked, "a real .env is tracked by git"
    assert ".env.example" in tracked


def test_example_env_carries_no_real_key():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "sk-ant-..." in text
    assert not any(
        len(token) > 30 and token.startswith("sk-ant-")
        for token in text.split()
    ), "the example file looks like it contains a real key"


# ----------------------------------------------------------------- model

@pytest.mark.parametrize(
    "given,expected",
    [
        ("haiku", "claude-haiku-4-5-20251001"),
        ("HAIKU", "claude-haiku-4-5-20251001"),
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
        ("", "claude-sonnet-5"),
    ],
)
def test_model_aliases_resolve(given, expected, monkeypatch):
    monkeypatch.delenv("PROBE_MODEL", raising=False)
    assert resolve_model(given) == expected


def test_env_model_is_the_fallback(monkeypatch):
    monkeypatch.setenv("PROBE_MODEL", "haiku")
    assert resolve_model() == "claude-haiku-4-5-20251001"
    assert resolve_model("opus") == "claude-opus-5", "an explicit flag must win"


def test_every_alias_is_priced():
    """An unpriced model would silently fall back to Sonnet's rates and print a
    cost estimate that is wrong by 3x."""
    for target in ("haiku", "sonnet", "opus"):
        assert resolve_model(target) in PRICING


def test_haiku_estimate_is_cheaper_than_sonnet():
    cheap = estimate_cost(24, model="haiku")
    dear = estimate_cost(24, model="sonnet")
    assert cheap.n_calls == dear.n_calls
    assert cheap.usd < dear.usd


# ------------------------------------------------------------ plumbing

def test_model_is_forwarded_only_to_the_live_backend():
    """The offline backends take no model argument; forwarding one would raise
    at the first interview of a sweep rather than at the flag."""
    assert SweepPlan(backend="anthropic", model="haiku").client_kwargs == {"model": "haiku"}
    assert SweepPlan(backend="sim", model="haiku").client_kwargs == {}
    assert SweepPlan(backend="anthropic").client_kwargs == {}


def test_live_backend_refuses_to_start_without_a_key(monkeypatch):
    from probe.runtime.llm import get_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_client("anthropic")
