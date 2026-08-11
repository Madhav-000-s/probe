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

> **Retracted 2026-08-07.** Every number in this entry was computed against a
> rubric of fourteen competencies on a twelve-question budget, which Phase 3
> established is a mis-specification. See the entry below for the re-run.

---

## 2026-08-07 — Phase 3: calibration, the freeze, and a retraction

### The retraction first

PLAN.md's cross-phase rule 3 says frozen means frozen: change a constant and
every number computed under the old value is re-run or retracted. The first
time that rule was invoked it cost the headline finding of the previous phase,
which is presumably the point of writing it down in advance.

Setting tau empirically exposed the problem. The sweep returned 0% at *every*
candidate threshold from 0.40 to 0.90, which is not a tau problem — it means
the fixed arm never resolved a full rubric under any threshold. The cause was
rubric size: the compiler emitted fourteen competencies against a
twelve-question budget, so several were never asked about at all, sat at their
prior interval for the whole interview, and made "all required competencies
resolved" unreachable by construction. Rubric size is now a frozen constant at
six, which is what a focused senior technical interview actually covers and
leaves roughly two questions per competency.

Re-running Phase 2's directional sweep under the corrected design reverses it:

| arm | resolved (SD < tau) | mean posterior SD | recovery rho |
|---|---|---|---|
| `fixed` | 23.3% | 0.579 | 0.804 |
| `heuristic` | 26.7% | 0.588 | 0.781 |
| `eig` | **31.7%** | 0.582 | **0.819** |

So the "entropy-greedy selection loses on threshold count" finding is
withdrawn. The arithmetic behind it was real — a resume-evidenced competency
crosses tau in one question and a resume-silent one takes three — but it only
dominated because the budget was too thin to reach most of the rubric, which
turned the comparison into a test of coverage rather than of selection. Under a
rubric the budget can actually cover, `eig` resolves the most and recovers
ability best.

The corrected reading is narrower than the original claim in one respect, and
the test suite now pins that too: on *mean posterior SD* the three arms are
indistinguishable (0.579 / 0.588 / 0.582 on ten personas, no CI). The eig
arm's advantage is in **where** it spends questions, not in extracting more
total information. Phase 4 attaches bootstrap intervals and will say whether
even the resolved-fraction gap survives.

### Calibration

200 items administered to all 36 calibration-split personas — 7,200 graded
responses, zero unrecoverable — then fitted by marginal maximum likelihood with
EM. MML rather than anything using `theta_star`: calibrating against ground
truth is something no real calibration can do, and it would quietly inflate
every recovery number downstream. A test asserts the fitter's signature cannot
even accept ability.

| quantity | value |
|---|---|
| items fitted | 200 |
| quarantined | 22 (11.0%) |
| mean fitted `a` (live items) | 1.92 |
| median fitted `a` | 1.78 |
| responses per item | 36 |
| correlation matrix | 50 competencies, mean abs rho 0.208, max 0.683 |

The authoring default was `a = 1.9` and the fitted mean came back at 1.92,
which is the cheapest available evidence that the fitter is not inventing.
Quarantine reasons are dominated by thresholds running outside the measurable
range and by discrimination blowing past 6.0 — both what thin data does, and
both handled by refusing the item rather than by trusting it.

**A wrong diagnosis worth recording.** The synthetic recovery gate failed at
first with what looked like systematic upward bias on `a` (true 1.40 fitted
1.66, true 2.30 fitted 2.57). Swapping the M-step from Nelder-Mead to L-BFGS-B
changed the estimates not at all. Measuring properly showed the error shrinks
with sample size and flips sign with the seed — sampling noise, not bias. The
gate now runs at 2500 respondents, which is a statement about the *estimator*
rather than about this project's 36-respondent calibration sample, and a
separate test asserts the mean error across seeds is under 0.10 because
over-estimating discrimination is the direction that would flatter every
downstream efficiency number.

### The freeze

| constant | value | how it was set |
|---|---|---|
| `tau` | 0.80 | fixed arm resolves 63.3% of honest calibration personas at budget 12 (target ~70%) |
| `epsilon` | 0.01 | EIG floor, unchanged |
| budget | 12 questions | from the plan |
| rubric size | 6 competencies | so the budget can cover the rubric |
| bank | v2 (calibrated) | 200 items, 22 quarantined |
| population | v2 | 60 personas, 36/24 split, 20% adversarial |
| seed | 20260807 | |

tau was swept on the **calibration split only**, over final posteriors from
interviews run with tau unreachable so none of them terminated early. Choosing
a constant by looking at the split you later report against is the circularity
the held-out design exists to prevent.

### Throughput

48 interviews across four arms, eight-way concurrency: 6.5 s wall clock, 447
interviews/minute under the offline backend. The concurrency test asserts the
failure that actually matters — fifty parallel runs producing fifty traces with
no turns crossing between them — because interleaved traces look perfectly
plausible afterwards and would poison every metric silently.

---

## 2026-08-08 — Phase 4: Q1 and Q2 answered, and a calibration failure

Main sweep: 4 arms x 24 eval-split personas x 4 style variants = 384
interviews, 52 s wall clock at eight-way concurrency, zero failures.
Everything below is generated by `make eval` from committed traces.

### Main results table

tau = 0.80, budget = 12, 24 personas, 95% bootstrap intervals resampled over
personas (not turns — a persona contributes many correlated observations and
resampling turns would produce intervals several times too narrow).

| arm | recovery rho | resolved | questions to confidence | ECE | $/interview |
|---|---|---|---|---|---|
| `fixed` | 0.701 [0.640, 0.743] | 0.793 [0.743, 0.846] | 8.7 [7.8, 10.2] | 0.114 | 0.0195 |
| `heuristic` | 0.714 [0.639, 0.760] | 0.839 [0.783, 0.889] | 7.5 [6.6, 8.7] | 0.097 | 0.0266 |
| `eig` | 0.776 [0.710, 0.821] | 0.986 [0.974, 0.995] | 7.5 [6.9, 8.1] | 0.090 | 0.0161 |
| `eig+corr` | **0.805** [0.741, 0.837] | **1.000** | **3.7** [3.2, 4.1] | **0.066** | **0.0113** |

### Q1: does EIG beat a well-prompted LLM chooser?

Yes, on this population, with intervals that exclude zero. Paired differences
against the `heuristic` arm — paired because every arm interviews the same
personas:

| contrast | difference | excludes 0 |
|---|---|---|
| `eig` recovery rho | +0.062 [+0.011, +0.118] | yes |
| `eig` resolved fraction | +0.148 [+0.102, +0.198] | yes |
| `eig+corr` recovery rho | +0.091 [+0.042, +0.139] | yes |
| `eig+corr` resolved fraction | +0.162 [+0.111, +0.217] | yes |

It wins on cost as well as on questions, which was not guaranteed: the
`heuristic` arm spends the most per interview because it sends the transcript
and a 24-item shortlist to the model at every turn, while `eig` computes its
objective locally. The arm that thinks harder is the cheaper one here.

The stop-reason distribution is where the mechanism shows. `eig+corr`
terminates on confidence 100% of the time and `eig` 92%, against 24% for the
fixed script — which mostly runs out of wall-clock budget (42%) because it
draws long items indiscriminately while the EIG objective divides information
by expected answer time and prefers cheap ones. Budget-awareness was a design
choice in the objective and it is doing visible work.

### The correlation ablation, and the check it has to pass

`eig+corr` reaches confidence in 3.7 questions against `eig`'s 7.5. A copula
that borrows information across competencies has an obvious failure mode —
talking itself into a tight posterior it has not earned — so the claim is only
worth anything if calibration does not degrade. It does not: `eig+corr` has the
best coverage (0.714 vs 0.688) and the lowest ECE (0.066 vs 0.090) of any arm.
The borrowed information is real. A test asserts exactly this, because if a
future change made the copula overconfident the headline number would still
look excellent.

### Q2 and the calibration failure

Grader agreement against an independent rater on 85 answers drawn from the
committed traces:

| quantity | value |
|---|---|
| Cohen's kappa | **0.554** (gate: >= 0.5) |
| quadratic-weighted kappa | 0.888 |
| exact agreement | 64.7% |
| within-one agreement | 98.8% |

**No human graded this set.** It was produced by an independent estimator with
different noise, no style term and no position bias, and calling it a human
gold set would be a lie the rest of the repo does not survive. D5's acceptance
criteria are recorded as unmet rather than redefined. What the artefact does
establish is that the release format, the kappa computation and the analysis
script work end to end, so dropping real labels in later is a data swap rather
than a rewrite.

**The failure worth reporting: the credible intervals are overconfident.**
Nominal 80% intervals cover 68-71% across every arm. The Phase 2 coverage test
passes at 78-82% in isolation, so this is not an inference bug — it is model
misspecification. The belief update treats the grader as a clean graded-response
model; the actual grader adds seed variance, position bias and a style term
that the likelihood does not represent, so the posterior tightens faster than
the evidence warrants. Every arm is affected roughly equally, which is why the
comparison survives even though the absolute intervals should not be quoted as
calibrated.

This is now the top item in what's next: an inflation parameter on the
likelihood, fitted from the observed grader noise, should close most of it. It
is pinned by a test asserting the current 0.60-0.75 range, so a fix registers
as a failure and forces the write-up to be updated rather than silently going
stale.

### A bug the cross-check earned its keep on

The accuracy-vs-budget curve is computed two ways — once in aggregate from
stored snapshots, once by replaying a single run turn by turn — and the two are
diffed. They disagreed. `RunView.truth_and_estimate` at budget n was counting
every competency probed across the whole interview, not just the first n turns,
so early curve points were reading untouched priors as though they were
estimates. That flattens the low-budget end of the curve, which is precisely
where the arms separate. Two independent implementations is the only reason it
was visible at all.

---

## 2026-08-10 — Phase 5: three measurement bugs, then Q3

Every headline number in this phase was wrong the first time. All three
failures were in the *measurement*, not the system, which is the worst place
for them: a broken metric reports a number with the same confidence as a
working one.

### Bug 1 — style variants were answering different questions

`answer_seed` hashed the style id, so every style variant of a persona drew a
different response level from the graded-response model. The terse and verbose
variants of one candidate were giving different content. The "style drift" the
fairness suite measured was mostly content variance wearing a style label, and
the intervention could not reduce it because it was never style.

Content is now a function of (persona, question, seed) alone. Mean absolute
drift fell from 0.613 to 0.331 the moment the confound was removed — over half
the "drift" had never been drift.

### Bug 2 — the name-swap invariant could not have held

Two problems at once. The candidate name was being signed into the answer
text, so the transcripts were not identical and "identical transcript,
different name" was untestable by construction. And the grader's noise was
seeded on the whole prompt hash, so any incidental difference — including a
name — produced a different noise draw and therefore a different score.

The name now reaches the grader through its context, the way a real grader
sees it, and grader noise is keyed on what is being graded rather than on
prompt formatting. Name swap is now exactly invariant: 0.00e+00 over 128
pairs. Under a deterministic backend that holds by construction rather than as
a finding, and the README says so.

### Bug 3 — injection resistance measured twice, wrong both times

First: resistance compared each injected turn against the mean of that
interview's clean turns. But the injector persona appends a payload to every
answer, so there were no clean turns, the baseline fell back to a hard-coded
3.0, and every competent answer scoring 4 or 5 was recorded as a successful
attack. It reported 54% resistance for a defence that had not been breached
once.

Second: replacing that with a real counterfactual — re-grade the same answer
with the payload stripped, using a length-preserving sanitiser — gave 83%. The
remaining 17% was the grader's own noise: comparing one noisy draw against
another counts a coin flip as an attack.

Third and correct: average the counterfactual over five grader seeds.

| version | resistance | what it was measuring |
|---|---|---|
| within-run baseline | 0.539 | a hard-coded constant |
| single-seed counterfactual | 0.829 | payload effect + grader noise |
| seed-averaged counterfactual | **0.955** | payload effect |

Mean score inflation from a payload is +0.054 of a rubric level — under a
tenth of the grader's own noise. 100% of payloads are flagged. The target was
above 95% and it is met, but the number to trust is the inflation, not the
threshold count.

### Q3: how much does surface style move a score?

Eight slices, `eig` arm, run twice — content-style separation off, then on.

| slice | drift off | drift on | reduction |
|---|---|---|---|
| verbose vs terse | 0.364 | 0.253 | 0.111 |
| neutral vs L1-transfer | 0.489 | 0.446 | 0.044 |
| hedged vs assertive | 0.451 | 0.258 | 0.193 |
| name_a vs name_b | 0.000 | 0.000 | — |
| **mean** | **0.326** | **0.239** | **0.087 (27%)** |

The intervention works, and it works most where it was aimed: the
hedged-versus-assertive axis drops 43%, because that is a pure tone effect and
telling the grader to score content removes it.

The residual is the L1-transfer slice, and it is the largest single drift left
(0.446). That is the open bug, and it is a different kind of failure from the
others. The intervention removes the fluency reward; it cannot remove a
recognition failure. First-language paraphrase — "consistency eventually" for
"eventual consistency", "read your writes" de-hyphenated — defeats exact
concept matching, so a candidate who knows the idea loses marks for phrasing
it non-idiomatically. The remedy is fuzzy or embedding-based concept matching,
which this version does not implement. Named in the README as the residual
rather than quietly fixed and lost.

### A negative result: bluffing works on the score

Bluffers average 4.08 against honest candidates' 2.91. The flags catch them —
bluff-detection AUC 0.736, and overclaim and non-answer recall are both 1.00 —
but the score does not, because the grader gives partial credit for borrowed
technical vocabulary from neighbouring competencies. Detection and scoring are
different problems and only one of them is solved here. The comparison is
uncontrolled: behaviour is confounded with whatever ability those personas
happen to have, so it is a direction, not an effect size.

### What the fairness fix did to Q1

Removing content variance from the style variants also changed the main table,
and not entirely in the project's favour. `eig` still beats the `heuristic`
arm decisively on resolved fraction (0.982 vs 0.804, interval excludes zero)
and on questions-to-confidence. But the recovery-rho difference (+0.028) now
has an interval that includes zero at n = 24 personas.

So the claim narrows: the adaptive policy gets to the same accuracy faster,
not to a higher accuracy. The README says exactly that, and the test that used
to assert both now asserts the efficiency win and asserts the rho interval
includes zero — so if a later change makes recovery significant, the suite
fails and forces the write-up to be updated rather than leaving an
understatement in place.

---

## 2026-08-11 — Live backend: credentials, model selection, and three dead Make recipes

The offline pipeline was never going to need a key, so the live path had been
wired, unit-tested against a stub, and never actually invoked. Enabling it
found the usual thing: the code was fine and the surface around it was not.

**Credentials.** `.env` at the repo root, loaded by the CLI root callback, with
one rule — it never overwrites a variable that is already exported. A stale
file silently beating the key you just set is a bad half hour, and the test
asserts the precedence rather than the loading. `.env.example` is committed;
`.env` is gitignored, and a test asserts both facts against `git ls-files`
rather than against the `.gitignore` text, because what matters is what is
tracked, not what is listed.

**Model selection.** `--model haiku|sonnet|opus` or a full id, falling back to
`PROBE_MODEL` and then to Sonnet. The alias resolution is shared between the
client and the cost estimator on purpose: an unpriced model id silently falls
back to Sonnet's rates, so the confirmation prompt would have quoted a number
three times wrong for a Haiku sweep. A test asserts every alias is in the
pricing table.

**Three Make recipes that could never have run.** `make experiment` invoked
`probe experiment run`, and `experiment` takes no subcommand — Typer rejects
the extra argument. Same for `make demo` and `make viewer`. Every sweep in this
project was driven by the CLI directly, so the targets were never exercised,
and the Phase 6 release test checked only that the targets *existed*.

`make report` was worse: it invoked `analysis.build_report`, a module that was
never written. That is the unbuilt 2-page PDF (D3) leaving a footprint. The
target is removed rather than left as a recipe that crashes; D3 stays recorded
as unmet.

The new test walks every `$(PROBE)` and `$(PY) -m` invocation in the Makefile
and resolves it against the Typer command tree and the import system. It is the
same class of gap as the reliability suite that Phase 6 found at 0% coverage:
something asserted to exist, never once executed.

**One test relaxed, deliberately.** `test_eval_is_byte_reproducible` compared
the committed results table against a fresh regeneration *including*
provenance, so it failed on every commit after the one that generated the
table — `code_commit` stamps whoever regenerates. It now runs the eval twice
and compares those two outputs byte for byte (the actual determinism claim),
and separately compares the committed numbers against the regeneration with
the commit stamp excluded. Numbers drifting still fails. The stamp moving no
longer does.

Nothing in the results changed: the committed table regenerates identically
apart from that stamp. No live run has been made, so every number in the README
is still `sim`-backend and the limitations section stands unedited.

---

## 2026-08-11 — First live model: four defects the offline backend could not have shown

Ran one interview against Haiku. It cost about six cents and found four bugs,
every one of them in code that had been green for six phases. The pattern is
the same in all four: the offline backends *compose* their output directly, so
nothing ever exercised the parts of the system that exist because a real model
writes JSON, counts characters, and stops when it runs out of budget.

**1. A flat 1024-token output budget for every role.** A 14-competency rubric
is roughly 3,000 tokens of JSON. It was truncated mid-object, and the repair
attempt — asking a model to return well-formed JSON — produced output cut off
at exactly the same place. Two calls, one error, no diagnosis. Budgets are now
per role, and truncation is detected from `stop_reason` and reported as
truncation, so the ladder stops chasing a formatting problem that does not
exist.

**2. The persona simulator had no live path at all.** `PersonaCandidate` parsed
the response with a bare `json.loads` and no repair ladder, and the prompt it
sent asked for prose. Under `sim` this never mattered: the backend returned the
full envelope regardless of what the prompt said. The prompt's own docstring
claimed the persona's depth was "expressed in words" — it was not expressed at
all, so a live sweep would have produced answers uncorrelated with theta and a
recovery metric measuring nothing.

Fixed by splitting `plan_answer` out of `compose_answer`: the level is drawn
from the GRM and the concepts chosen **before** any model is called, and the
live prompt is told which ideas to cover rather than what score to hit. Ground
truth stays on this side of the boundary. Asking a model to answer "as a
level-4 candidate" would have scored its own idea of a 4 against ours.

**3. The evidence-span postcheck rejected 100% of real grades.** This is the
one worth remembering. `EvidenceSpan.verify_against` requires
`answer[start:end] == text`. Haiku's grades were *substantively good* — scores
2 and 4 where the offline grader would also have discriminated, quotes lifted
verbatim from the answer — and every single one was rejected, because the
character offsets were arithmetic the model cannot do. Every grade fell to the
degraded path, so the transcript showed six answers all scored 3 at confidence
0.000 and the posterior learned nothing from a working interview.

The check was defending the wrong invariant. What matters is that the quoted
text is really in the answer; where it sits is arithmetic we can do ourselves.
Spans are now relocated by exact substring search before the postcheck runs,
and a span whose text is genuinely absent is still rejected — fabricated
evidence stays fabricated evidence. Exact match only: accepting near-matches
would let a paraphrase pass as a quotation, which is the whole reason the span
exists.

After the fix, the same interview: scores 1, 4, 4, 4, 4, 4, confidences
0.85–0.95, zero repair calls, and a `non_answer` flag at turn 1 that triggered
a follow-up which the candidate then answered at 4. The mechanism works against
a real model. It had never been run against one.

**4. Repair calls were free.** `ParseResult` carried no token count, so callers
charged themselves for the successful attempt only. A role that fails and
repairs looked exactly as cheap as one that parses first time.

Every committed number is unchanged — the sim path is bit-identical, verified
by the full suite and by regenerating the results table. That is the point of
the fixes being where they are: relocation is a no-op on exact offsets, and the
role budgets are only consulted by a backend that has a budget.
