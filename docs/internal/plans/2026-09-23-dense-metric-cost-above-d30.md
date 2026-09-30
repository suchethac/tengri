# What a dense metric actually costs above D=30, and what the advisory says it costs

**Status: finding, with one measurement still in flight.** Nothing is changed
here. The proposal at the end is a message correction and a question about the
auto policy, neither of which should land before the running fit reports its
final peak.

## The claim under test

`inference/_dimension_guard.py` warns on an explicit `mcmc_nuts` above
`NUTS_WARN_D = 30`:

> NUTS warmup memory is dominated by the mass matrix, which is O(D^2) when
> `dense_mass_matrix=True`, problems this size have been measured at 20+ GB and
> can OOM the process.

Two things in that sentence are not supported by the evidence the same module
cites, and the module's own docstring is the place that says so.

**"problems this size".** The docstring attributes the figure precisely:
"measured peaks of 3-6 GB on small photometry problems (D <= 7) reach 20+ GB by
D ~ 8 **with a dense_basis mean SFH**". The driver named there is the SFH, at
D ~ 8. No measurement of a D > 30 problem at 20+ GB is cited anywhere in the
module, the tests, or #319. The warning fires on a dimension band and then
reports a number measured in a different band, for a different reason.

**"dominated by the mass matrix".** At D = 36 a dense mass matrix is 36 x 36
float64: about 10 kB. It cannot dominate 20 GB, or 1 GB. The docstring's fuller
sentence is "dominated by the mass matrix **and the per-step trajectory
buffer**", and it is the second term that scales with the work actually done.
The emitted message drops it and leaves the kilobyte term carrying the claim.

## What the repository already measured

| where | D | SFH | metric | peak |
|---|---:|---|---|---|
| #319 | ~8 | `dense_basis` | dense | 22.78 GB |
| `test_bug_1454` (revised 2026-09-11, `ctl-dpl`) | 8 | non-`dense_basis` | dense | 1.1-1.9 GB |
| #1454 | 9 | — | dense (HMC) | 13.47 GB, SIGKILLed |

That 2026-09-11 revision is the point: re-measuring at D = 8 **without**
`dense_basis` gave 1.1-1.9 GB rather than 20+, and the auto-policy cliff moved
from D = 8 to D = 12 as a result. So the project has already established once
that the 20+ GB figure belongs to the SFH and not to D. The advisory above
D = 30 was not revisited at the same time.

## The measurement that was attempted, and why it measured nothing

The intent was: the paper's joint mock, D = 36, continuity SFH,
`dense_mass_matrix=True` forced on, 4 chains, 1000 warmup, 600 samples. It ran
for roughly eight hours and reached a peak RSS of 3538 MB.

**That number does not measure a dense metric, and no number from this run
does.** The run's own log says why, in a fourth warning that only appears once
the sampler starts:

> `mcmc_nuts: dense_mass_matrix=True at D=36 exceeds the D<=30 cap (the mass
> matrix alone is O(D^2), and warmup has been measured at 20+ GB well below
> this size, #319). Falling back to a DIAGONAL metric.`

So the lever was never pulled. Eight hours of "dense" run was a fourth
diagonal run, and the earlier provisional figure recorded here -- 1218 MB at
3 h -- was a diagonal run too. Both are struck. The caution written into this
section at the time ("do not quote this until the run ends") was right for the
wrong reason: the problem was not that the number was early, it was that the
configuration under test was never active.

## The larger finding: a named mode is silently downgraded

The advisory above D = 30 is one thing; the hard cap underneath it is another,
and it is the more serious of the two. A caller who asks for
`dense_mass_matrix=True` at D > 30 does not get it. They get a diagonal metric
and a warning, and the posterior that comes back carries no field saying which
metric produced it -- the run's own JSON records `dense_mass_matrix: true`,
which is what was *requested*, not what ran.

That is the shape this repository already guards against elsewhere: a name may
only map to a driver that runs the algorithm the name promises, and stand-ins
are the flat seam's founding bug. Refusing loudly, or recording the resolved
metric beside the requested one, would both be consistent with that rule. The
present behavior is neither.

It also invalidates a diagnostic path that looked sound: `posterior_conditioning`
measured a condition number of 464 on this posterior and concluded a dense
metric was the remedy. Acting on that conclusion is currently impossible at
D = 36 through the documented knob, and nothing says so until a warning scrolls
past in a log.

## Why this is not a pedantic complaint

The advisory recommends `dense_mass_matrix=False` first. For this posterior
that recommendation is expensive. `analysis/paper1/posterior_conditioning.py`
measured the correlation matrix a diagonal metric leaves behind on the ta085
draws: condition number **464.3**, so trajectory length scales as its square
root, about **21x** in leapfrog steps per draw. Three NUTS attempts on a
diagonal metric failed to reach the basin, which is what prompted forcing dense
in the first place.

So a user at D = 36 with a continuity SFH reads "20+ GB and can OOM", turns off
the metric that addresses their conditioning, and pays 21x in trajectory length
to avoid a cost that -- on the evidence so far -- this model class does not
incur.

## Proposal

1. **Correct the message, not the threshold.** The threshold, the
   advisory-rather-than-refusal design, and the GHMC/ChEES exclusions are all
   well reasoned and should stay. The message should say what the docstring
   says: that the peak depends on the model and not on D alone, that the 20+ GB
   figure was measured at D ~ 8 with a `dense_basis` SFH, and that the
   trajectory buffer rather than the metric is the term that grows.
   `tests/regression/bug/test_bug_319_nuts_oom_warning.py` selects the warning
   by the substring `"20+ GB"`, so any rewording must keep it or update that
   test deliberately.
2. **Ask whether the auto policy's D = 12 cliff should know about the SFH.**
   It already moved once for exactly this reason. This is a question for the
   owner, not a change to make here: it is shared inference behavior that many
   sessions build against.

Neither is worth doing on a 3-hour in-flight number. Revisit when the fit ends.
