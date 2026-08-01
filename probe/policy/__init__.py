"""Question-selection policies — the research core.

Four arms, one interface. The comparison is only meaningful because every arm
sees the same bank, the same grader and the same stop rule, and differs solely
in :meth:`~probe.policy.base.Policy.next_question`.
"""

from probe.policy.base import Ask, Decision, Policy, Stop
from probe.policy.fixed import FixedPolicy

__all__ = ["Ask", "Decision", "Policy", "Stop", "FixedPolicy"]
