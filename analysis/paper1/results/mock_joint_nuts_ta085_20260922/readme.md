# Mock joint NUTS, target_accept 0.85, warm-started — refused

The second full attempt. It ran 11.08 h and finished cleanly (EXIT=0), and it
does not clear the figure's publication gate.

| diagnostic | cold run | this run | gate |
|---|---:|---:|---|
| max split R-hat | 1.0077 | **1.0024** | < 1.01, passes |
| ess_min | 242 | **520.3** | >= 400, passes |
| divergences | 19 | **79** | 0, fails |

chi2/dof: truth 1.0738, MAP warm start 1.0480, fitted **1.3075**.

## What this run establishes

Doubling the draws fixed the effective sample size and R-hat. The warm start
did not fix the divergences and made them worse.

That combination is the useful part. The chains *started* inside the basin --
the single L-BFGS start reached chi2/dof 1.0480, better than truth's 1.0738 --
and ended at 1.3075 with four times the divergences of the cold run. A better
initialization producing more divergences excludes initialization as the
cause. The launch note set this up deliberately by holding target_accept at
its default so that persisting divergences would attribute to geometry; they
did not merely persist, they quadrupled.

## What it does not establish

Nothing about recovery. The run's own summary says so: the fit is 345.7 worse
in chi2 than truth, so it never reached the mode, and the per-parameter deltas
printed in the log measure where the sampler ended up rather than where the
posterior is. Do not quote them, and do not quote the largest-delta line
(neb_eline_sigma_kms +10.2178) as a recovery figure.

## Provenance

No `provenance` block: the process loaded the code before that field existed,
so which tree produced it is recorded nowhere in the file. It is this
worktree's src at the time of launch, 2026-09-22 01:15, but the artifact does
not say so and cannot be made to say so after the fact.

Kept for comparison against the next attempt, which raises target_accept to
0.95 — the one lever left, changed alone.
