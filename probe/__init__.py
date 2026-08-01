"""probe — an adaptive interviewing agent and the harness that measures it.

Two planes live in this package tree:

* the **interview plane** (`rubric`, `belief`, `policy`, `bank`, `grader`,
  `runtime`, `report`) which conducts an interview and never sees ground truth;
* the **measurement plane** (`sim`, plus the top-level `evals` package) which
  generates candidates with hidden ground truth and scores the interview plane
  against it.

The separation is the load-bearing invariant of the whole project: a leak of
``theta_star`` into the interview plane makes every recovery number meaningless.
It is enforced by a test (``tests/phase0/test_ground_truth_firewall.py``) that
greps every logged prompt, and that test never leaves the suite.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
