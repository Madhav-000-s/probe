# results-log

The running lab notebook. Every number that ends up in the README passes
through here first, along with the decisions that produced it — including the
ones that went the wrong way. Losses are logged as diligently as wins; a repo
where nothing ever failed is a repo where nothing was ever really tested.

Entries are dated and append-only. After the Phase 3 gate, any change to a
frozen constant requires an entry here explaining why, and every number
computed under the old value is re-run or retracted.

---

## 2026-08-03 — Phase 1: the fidelity gate failed twice before it passed

The plan calls the simulator fidelity gate the project's keystone: if a
persona's answers do not encode their hidden ability, every downstream number
is decoration. It failed on the first run and the failures were both real.

**Failure 1 — rho = 0.593, threshold 0.60.** Diagnosis found two independent
bugs, neither of them "the threshold is too strict".

*Bug A: terse styling deleted content.* `style.render` shortened terse answers
by dropping trailing sentences. Concepts are front-loaded, so this *usually*
removed padding — but not always, and a terse persona could lose two or three
concepts it had genuinely named. That is content loss caused by a style
transform, which is exactly the confound the content/style split exists to
prevent. Fixed by tagging padding as its own segment kind: terseness now
removes `FILLER` and nothing else, by construction rather than by a heuristic
that happened to usually work.

*Bug B: the gate's own monotonicity check was noise.* Competencies became
eligible at six answers, which meant tertiles of two. Raised to twelve, and
the default sample from 100 to 240. The gate also now holds style neutral: it
asks whether ability reaches an answer's *content*, and style-induced
recognition loss is Q3's subject, not this gate's. Measuring them together
would have conflated "the simulator does not encode ability" with "the
extractor cannot always see it".

**Failure 2 — rho = 0.503 at n = 400.** The larger sample made things look
worse, which is the useful direction for a surprise to run in: the first
number had been small-sample luck. The cause was the item bank, not the
simulator.

A diagnostic against the population's actual ability distribution
(theta ~ N(0, 1.01) after clipping, not the uniform spread I first assumed)
gives the single-response ceiling as a function of discrimination:

| `a` | rho(theta, blind rating) |
|---|---|
| 1.0 | 0.461 |
| 1.3 | 0.559 |
| 1.6 | 0.635 |
| 1.9 | 0.685 |
| 2.2 | 0.728 |

The authoring default was `a = 1.0`. At that discrimination the gate was
unreachable regardless of how good the simulator was — a restriction-of-range
effect I had not accounted for when picking the placeholder. Threshold spread
barely matters (rho moves ~0.01 across `b` spreads of +-1.0 to +-1.5), so the
discrimination is the whole story.

Set `DEFAULT_A = 1.9`. The justification is not "it passes": a five-point
rubric whose anchors name *which concepts each level requires* should
discriminate substantially better than a holistic one, and 1.0 describes an
item that barely separates anyone. Phase 3 replaces every one of these with a
maximum-likelihood fit from the calibration split, so this is a starting point,
not a result.

**Recorded so it stays honest:** the fidelity number is not evidence that the
simulator is realistic. It is evidence that ability propagates into answer
content and survives a blind read, which is the minimum required for anything
downstream to be measuring what it claims. What it cannot tell us is whether
the simulated construct resembles a real candidate — that limitation is stated
first in the README and does not go away.

### Two more bugs the end-to-end sanity check found

Neither showed up as an error. Both produced perfectly valid-looking reports,
which is the failure mode worth being frightened of.

*The starter bank did not cover the roles being hired for.* It was built from
the first twenty taxonomy entries — all backend — while one of the two job
descriptions hires for data/ML. Those personas compiled a rubric the bank had
no items for and the interview terminated at turn zero, reporting every
competency as "unprobed" with a wide interval and no error anywhere. The bank
is now built from the union of what the JDs actually require. There is a test
asserting every persona can spend its whole question budget, because "ran out
of script" and "reached confidence" are indistinguishable in a results table
and only one of them is a finding.

*Keyword matching had no word boundaries.* The compiler matched `jd_keywords`
as bare substrings, so the two-letter keywords — `ml`, `ci`, `cd` — matched
inside unrelated words and a data/ML job description came out requiring half
the backend taxonomy. Fixed with boundary-anchored patterns.

`FixedPolicy` also asked one question per competency, which exhausted the
script before a twelve-question budget. An arm that stops early because it ran
out of script has not reached confidence sooner; it has been handed a shorter
interview. Depth is now three, matching the bank, so the budget is the binding
constraint for every arm — which is the only way an arm comparison means
anything.

### Phase 1 exit numbers

| Quantity | Value |
|---|---|
| Fidelity gate, pooled Spearman rho(blind rating, theta*) | **0.728** (threshold 0.60) |
| Sample | 400 answers, style held neutral |
| Competencies with monotone tertile means | **100%** (threshold 70%) |
| p | 2.9e-67 |
| Grader exact agreement, same seed x 3 | 1.000 |
| Population | 10 honest personas, 6 calibration / 4 eval |
| Bank | v1-starter, 60 items over 20 competencies, uncalibrated |
| Tests green | 132 (phase 0 + phase 1) |

The grader agreement figure of 1.000 is exact by construction under a
deterministic backend and is recorded as such rather than as a finding. The
number the plan actually wants from it — agreement across *different* grader
seeds — is test-retest variance, and it belongs to the Phase 4 reliability
suite. A separate test asserts seed-to-seed variance is non-zero, because a
grader with none would make that suite vacuous.

---

## 2026-08-05 — Phase 2: the EIG arm wins on information and loses on the stop rule

The numerical core is verified: graded-response categories against
hand-computed fixtures to 1e-9, posterior coverage inside the 78–82% band over
500 synthetic runs, analytic EIG against a 20k-sample Monte-Carlo estimate on
50 random states to within 0.02 nats. Those are not the interesting part of
this phase.

### The first directional result

Ten honest personas, budget 12, everything except the arm held identical.

| arm | resolved (SD < tau) | mean posterior SD | recovery rho | n probed |
|---|---|---|---|---|
| `fixed` | 13.9% | 0.731 | 0.722 | 100 |
| `heuristic` | 12.1% | 0.693 | 0.729 | 100 |
| `eig` | **2.5%** | **0.688** | 0.717 | 92 |

The ordering on mean posterior SD is the one the project predicts:
`eig` < `heuristic` < `fixed`. The ordering on *resolved fraction* is exactly
backwards.

### Why, precisely

Both are true simultaneously, and the tension is real rather than a bug.
Entropy-greedy selection maximises nats; the confidence stop rule counts
threshold crossings. Those are different objectives, and against tau = 0.55
with a = 1.9 items the gap is arithmetic:

| starting state | prior SD | questions to cross tau = 0.55 |
|---|---|---|
| resume-evidenced competency | 0.60 | **1** |
| resume-silent competency | 1.15 | **3** |

A breadth-first script buys roughly three threshold crossings for every one a
widest-first policy buys — while extracting less total information. `eig` goes
where the entropy is, which is exactly where crossings are most expensive.

This is the "eig lost — what broke?" case, and it goes in the main text of the
report rather than an appendix. It is also the reason PLAN.md sets tau
empirically in Phase 3 rather than guessing it: at tau = 0.55 no arm reaches
confidence within budget at all, so questions-to-confidence is fully censored
and cannot discriminate between arms. The Phase 2 gate therefore asserts the
continuous metric (mean posterior SD at fixed budget — the accuracy-vs-budget
curve in scalar form) and pins the threshold-count inversion in its own test,
so that if a later change reverses it the suite says so instead of letting the
write-up go stale.

### Three bugs found on the way, in increasing order of embarrassment

**Credible intervals were biased by half a grid step.** ``cumsum(pmf)[i]`` is
the probability of falling at or below the **right edge** of bin `i`, but the
interval was interpolating against bin *centres*. Every interval in every
report was shifted, and it would have surfaced much later as a calibration
failure with no obvious cause. Caught by comparing an 80% interval against the
analytic +-1.2816.

**The EIG arm had no idea tau existed.** The interview stops when every
required competency is under tau, so a competency already under it contributes
nothing to termination — yet pure entropy-greedy selection will happily keep
mining it because it is still the widest thing on the board. The candidate set
now excludes resolved competencies. This is alignment with the stop rule, not
an optimisation.

**The rubric contained competencies the bank could not ask about.** The
compiler emitted 14 required competencies; the bank had items for 8. The other
6 sat at their prior interval for the whole interview, reported cleanly as
"unprobed", counting against every arm identically and diluting every
efficiency number by a constant. The compiler now takes the bank's coverage
and records what it had to drop. An interview plan you have no questions for
is not a plan.

Also: every item in the bank had identical GRM parameters, so no item was more
informative than another about a given candidate and adaptive selection had
nothing to select on beyond which competency to probe. Items within a
competency now span a difficulty range, which is ordinary test construction —
an author writes an approachable question, a middling one and a hard one — and
is what calibration would recover anyway.

### What this does not yet show

The `eig` margin on mean posterior SD is 0.043 nats-equivalent (0.688 vs
0.731), with n = 10 personas and no confidence interval. That is a direction,
not a result, and the phase gate asks for nothing more. Phase 3 calibrates the
bank, scales the population to 60 with a held-out split, and freezes tau; Phase
4 attaches bootstrap intervals. If the margin does not survive that, the
report says so.
