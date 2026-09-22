# Landing the Paper I trees on main: nss, pin, and what each side must not lose

Three trees carry Paper I work and **no two of them contain each other**. This
records what is where, in what order to merge, and the two places where both
sides changed the same decision — because in each of those a plausible
resolution silently deletes a guard that exists for a measured reason.

Measured 2026-09-23 against the tips of `origin/main`,
`origin/paper1/pin-2026-09-14` and `origin/paper1/nss-profile-mass`. Re-measure
before acting: all three move.

## The shape

| pair | merge base | ahead / ahead | conflicting paths |
|---|---|---|---|
| pin ↔ nss | `a3ff0e362` | pin 139 / nss 19 | 3 |
| pin ↔ main | `57ee58185` | pin 134 / main 25 | 24 |

An earlier brief said pin is nss's ancestor and the merge target is therefore
nss into main. It is not, and that instruction loses 139 commits. The confusion
was between the **pinned commit** `338109404`, which *is* an ancestor of nss,
and the **branch** `paper1/pin-2026-09-14`, which is not. Say which is meant.

Absent from nss and present only on pin, as examples of what the shortcut would
have dropped: `fc4ec0ba6` (`igm_fold="auto"`), `20a82f495` (the wall-clock
adoption fix), `773786b63` (`config_forward_digest`, the tool that produced the
forward comparison the decision rests on), `1ab901af6` (the performance doc
whose slow-AGN remedy named a torus that does not exist), and
`analysis/paper1/fig04_gpu_datacenter.py`, which renders `fig:gpu` — a figure
the paper includes.

## Order

1. **nss into pin.** Three conflicting paths, below.
2. **The combined branch into main**, as one PR.

Nothing merges *into* nss until the 20x6 grid finishes. Row III is already on
nss; the grid is a controlled comparison only while every row runs on one tree,
and that cannot be repaired after the fact.

The log-normal onset fix exists as two commits with one subject line —
`f01975f46` on pin, `973157207` on nss. `git patch-id --stable` gives both
`5041303ed23b36ab405ca9277245d089ecd35e31`, so it is one patch cherry-picked
twice and git should reconcile it rather than conflict. A SHA is an identity,
not a change; when branches trade patches, `patch-id` is the question to ask.

`.git/rr-cache` held 326 entries when this was written. On a merge this size
rerere will replay resolutions nobody in this decision reviewed, and CHANGELOG
is this repository's known repeat offender. Disable it for the merge, or diff
every auto-resolved hunk.

## `analysis/paper1/configs.py` — both sides capped a time prior, differently

Pin `ef848e0c0` makes the cap unconditional:

```python
tau_upper = age_at_z(z)
"tau_gyr": Uniform(0.5, tau_upper),
```

nss `ac51f8fcb` makes it opt-in and adds `II_taucap` as a probe row to test
whether capping helps at all:

```python
tau_hi = age_at_z(z) if tau_cap else 13.0
"tau_gyr": Uniform(0.5, tau_hi),
```

Same line, incompatible intents, and **each naive resolution breaks something
real**:

- Take pin's side and `II_taucap` is *vacuous*. Both rows would be capped, so
  the probe compares capped against capped and can never detect anything. A
  probe that cannot observe its own subject is the failure this repository
  keeps finding; arriving through a merge makes it no less one.
- Take nss's side and `analysis/paper1/tests/test_time_prior_bounds.py` fails.
  That file is on pin **and already on main**, absent only from nss, and its
  parametrized `test_time_prior_bounds_respect_cosmic_age` requires exactly
  this bound: its table records `dpl tau (turnover) 467 - 121000 FLAT outside
  -> capped, not exempt` beside `delayed tau (e-folding) 64 - 79 well
  conditioned -> exempt`.

So the owner's pending "tau-prior ruling" and this conflict are the same
question, and **main has already answered half of it**: the test asserting the
cap is on main today. The consistent resolution is to keep the unconditional
cap and give `II_taucap` a subject it can actually observe — an explicitly
*uncapped* II, if the comparison is still wanted — or to drop the probe row and
say why. Deleting the test to keep the probe would be resolving a science
question by deleting the thing that measures it.

Also on this file, additive and uncontested: pin `9c9a4acde` (results-directory
audit) and nss's `II_taucap` SSP entry.

## `analysis/paper1/fit_one.py` — nss owns it, one pin commit must survive

nss is dominant here (+349/−54 across eight commits): per-attempt code
revision, seed and effective PRNG key, adapted step size and warmup/tree-depth
stats, load average and concurrent fits, the cross-process reinsertion lock,
and the derived-quantities rewrite that took a cell from 28 min to 9 s. Take
that wholesale.

Pin has two, and `f3d639bd8` is not optional: *"the mock joint fit discarded its
entire posterior."* Confirm it survives by running the mock fit's persistence
test, not by reading the diff.

## `src/tengri/inference/mass_profile.py` — keep both, and this one has teeth

Pin is dominant (+457/−115 across eight commits, each with an issue number):
float32 on gradient backends (#2355), chunked `lax.map` reinsertion
(#2356/#2358), the linearity probe and its refusal wording (#2359/#2365,
#2359/#2370), chunk-width reporting at INFO (#2367), scoring a measured
line-flux block (#2372, #2374), and the float32 Hessian refusal (#2378/#2389).

nss has three, and `5b0ca20b9` caps the reinsertion chunk at 64 draws. Pin's
chunking is **not** in the merge base, so these are competing implementations
of one mechanism rather than one sitting on the other. The cap is not a
preference:

> `temp_size_in_bytes` of the per-chunk program under-reports the realized peak
> by an order of magnitude on the paper-1 CANDELS models: measured on
> configuration V (D=5 profiled, 1200 draws, 384 quadrature nodes) the analysis
> derived 756 draws per chunk and the run allocated 1.5 GB buffers past an
> 18 GB cap, while the same fit at 64 draws per chunk peaked 1.0 GB above
> baseline. At 756 the reinsertion spike reached 23-29 GB inside a NUTS fit and
> the shared box's 40 GB watchdog killed four cells.

Resolving to pin's side alone re-exposes that spike and the grid loses cells
again, with nothing in the diff to say why. Keep pin's mechanism **and** nss's
ceiling on top of it, and keep nss's rationale comment, which is the only place
the measurement is written down.

## After the merge

- `pytest analysis/paper1/tests/ -q` — the time-prior bounds and the mock
  posterior persistence are the two that encode the decisions above.
- `pytest tests/contract/test_igm_exact_fold.py -q`. Note this file is itself
  an add/add conflict against main, where `76b01dd62` independently wrote a
  suite for the same feature; the union has to be taken by intent, and the
  merged file then mutation-validated against a broken exact fold. Running it
  cannot validate a resolution when the file *is* the resolution.
- The #2474 tests, since main carries the Lyman-continuum mask the pin does not.
- `tools/check_british_spelling.py`, `ruff check`, `ruff format --check`.
