"""The measurement plane's offline half.

Every metric here is a pure function from traces (plus persona ground truth) to
a number with a bootstrap interval. Nothing in this package calls a model: if a
metric needed an LLM to compute, it would be a second thing to validate rather
than a measurement of the first.

Populated in Phase 4.
"""
