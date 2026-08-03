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
