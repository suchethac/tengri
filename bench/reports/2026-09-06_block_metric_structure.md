# Block-structured mass matrices are the wrong structure for this posterior

**Date:** 2026-09-06
**Issue:** #2166
**Script:** `bench/scripts/probe_block_metric_structure.py`
**Raw run:** `bench/results/2026-09-06_block_metric_structure.txt`
**Fixtures:** `05` (D = 8 photometry), `ctl-jwst` (D = 9 continuity SFH),
`stoch-field` (D = 74 stochastic field)

**Verdict: measured, and declined.** A block-diagonal mass matrix — dense blocks
on correlated groups, diagonal elsewhere — loses to a rank-`k` correction to a
diagonal on **both** axes at **every** storage budget tested, on all three
fixtures. The decisive number is the D = 74 field posterior: a rank-3 correction
stores **299** matrix entries and leaves the sampler a condition number of
**10.9**, while the best block layout the measurement itself endorses stores
**4106** entries — 75 % of full dense — and leaves **2.1e4**. That is 14x the
memory for 1900x worse conditioning.

**And the reason is structural, not a bad partition.** Measuring each candidate
group's own numbers rather than assuming them (below, "the per-group verdict")
produces a dilemma with no third side: the groups whose coordinates genuinely
correlate *among themselves* are either too small to save any memory, or so large
that blocking them **is** the dense matrix. The groups in between couple outward
more than inward, and a block-diagonal matrix cannot represent a correlation that
crosses its own partition, by construction.

Four candidate design principles for where blocks should go were tested rather
than adopted; one is refuted outright on this posterior (hierarchical
hyperparameters do **not** belong diagonal here — the field's scale parameters
send 84-93 % of their coupling into the latents they scale), and one had its
premise falsified cheaply enough that the sampling run it would have justified
was not spent.

Nothing was sampled to reach this. It is a statement about one matrix per
fixture, so it is deterministic, contention-immune, and cost three Hessians.

## What was measured

`preconditioning.py` builds the analytic metric `G = -grad^2 log p` at the
expansion point. For a candidate structure, apply that structure to the matrix an
adaptation would actually estimate, then ask how much of the whitening survives.

Window adaptation estimates the **inverse** mass matrix, which is a covariance
`Sigma`, not the metric. So the arm that matches a shippable implementation forms
`Sigma = G^-1`, applies the structure to `Sigma`, and whitens with
`M = structure(Sigma)^-1`. The report carries the metric-side convention
(`structure(G)` directly) in a second column, because the two diverge sharply for
low-rank and it would be easy to reach the opposite conclusion from the wrong one
(Finding 6).

Four arms:

| arm | what it stores | entries |
|---|---|---|
| full dense | everything (the ceiling, `cond` = 1 by construction) | `D^2` |
| diagonal | marginal scales only (the control) | `D` |
| **block** | dense blocks on a partition, diagonal elsewhere | `D + sum_i (d_i^2 - d_i)` |
| **low-rank** | diagonal plus a rank-`k` correction | `D + kD + k` |

Reported as `recovered = 1 - log10(cond_whitened) / log10(cond_raw)`: 1.0 is a
perfect metric, 0.0 no metric, negative means it made the geometry worse.

Both structured arms are given their **best possible estimate** — exact
within-block covariance, exact top-`k` eigenpairs — so this compares *ceilings*.
That is what makes the conclusion strong: no improvement to a block estimator can
lift it above its own ceiling, and its ceiling is below low-rank's everywhere here.

## The per-group verdict

Blocking is not all-or-nothing across groups. A group earns a dense block only if
its coordinates correlate with **each other**; a group that is uncorrelated
internally spends `d_i^2` entries representing structure that is not there, and a
group that couples outward more than inward has correlations no block can hold.
Two numbers decide it, both read off the correlation form of `G`:

* **internal** — off-diagonal Frobenius mass inside the group's own submatrix, as
  a fraction of that submatrix's norm. This is what a block would buy.
* **int/ext** — that internal mass over the group's coupling to everything
  outside it. Below 1, the group talks to the rest of the model more than to
  itself.

The script's thresholds are `internal >= 0.25` and `int/ext >= 1.0`.

| fixture | group | size | internal | int/ext | submatrix cond | block entries | earns a block? |
|---|---|---|---|---|---|---|---|
| `05` | `sfh` | 5 | 0.828 | 2.038 | 96.0 | 20 | **yes** |
| `05` | `dust` | 2 | 0.001 | 0.001 | 1.002 | 2 | no — internally uncorrelated |
| `05` | `met` | 1 | — | — | 1 | 0 | no — single parameter |
| `ctl-jwst` | `sfh` | 7 | 0.910 | 2.395 | 1060 | 42 | **yes** |
| `ctl-jwst` | `dust` | 1 | — | — | 1 | 0 | no — single parameter |
| `ctl-jwst` | `met` | 1 | — | — | 1 | 0 | no — single parameter |
| `stoch-field` | `psd` | 64 | 0.974 | 2.401 | 4.08e4 | 4032 | **yes** |
| `stoch-field` | `sfh` | 7 | 0.855 | 0.338 | 9908 | 42 | no — couples outward |
| `stoch-field` | `dust` | 2 | 0.601 | 0.159 | 7.05 | 2 | no — couples outward |
| `stoch-field` | `met` | 1 | — | — | 1 | 0 | no — single parameter |

Three of the four physical priors held and one did not. SFH shape parameters do
correlate strongly among themselves (0.83, 0.91, 0.86 internal — the double-power-law
and continuity shape parameters trade off against each other by construction).
Metallicity is a single parameter and can never earn a block. Total mass does
couple broadly, and it couples *out* of its own group: `sfh_tsnorm_log_total_mass`
x `dust_tau_diff` at |R| = 0.804 on `05`.

**Where the cross-subsystem coupling actually lives — and it is not total
mass.** The row-wise question was: does total mass's off-diagonal mass
concentrate in one group, spread across several, or dominate the matrix? On `05`
and `ctl-jwst` it **concentrates**: `sfh_tsnorm_log_total_mass` sends 0.88 of its
off-diagonal mass into its own SFH group (0.92 for `sfh_cont_log_total_mass` on
`ctl-jwst`). Total mass has a home, and the prefix rule already puts it there.

The parameters with no home are the *other* two, and each is a lone parameter
whose entire coupling points out of its own group:

| fixture | parameter | own group share | goes to |
|---|---|---|---|
| `05` | `dust_tau_diff` | 0.00 | `sfh` 0.99, `met` 0.16 |
| `05` | `met_logzsol` | 0.00 | `sfh` 0.90, `dust` 0.44 |
| `ctl-jwst` | `dust_tau_diff` | 0.00 | `sfh` 0.99, `met` 0.12 |
| `ctl-jwst` | `met_logzsol` | 0.00 | `sfh` 0.85, `dust` 0.52 |
| `stoch-field` | every named parameter | 0.15-0.46 | `psd` 0.84-0.94 |

So the attenuation-mass-age degeneracy is not a floating normalization looking
for a block. It is two single parameters, in two different subsystems, each
bound almost entirely to a third. The only layout that holds them is one that
puts them in the SFH block — which at D = 8 is 7 or 8 of 8 coordinates, i.e. the
dense matrix. On `stoch-field` it is worse: **all ten** named parameters send
84-94 % of their coupling into the 64-coordinate field block, so a partition that
respects the coupling is a single block over the whole model.

**The dust pair does not correlate with itself.** On `05`, `dust_tau_bc` and
`dust_tau_diff` carry an internal off-diagonal mass of **0.001** and an int/ext
ratio of **0.001**: the two optical depths are essentially independent of each
other and couple almost entirely to the SFH and mass. A prefix rule would have
blocked them, which is exactly the waste this refinement exists to catch. It is
also worth noting that a two-parameter block is 2 entries — catching it saves
nothing measurable, and that is the whole problem (Finding 3).

## The four-arm comparison

`cond` and `recovered` under the covariance convention (the shippable one).

### `05` — D = 8 photometry, the primary gate fixture. `cond(G)` raw 6.4e4

| arm | cond | recovered | entries |
|---|---|---|---|
| full dense (ceiling) | 1 | 1.000 | 64 |
| diagonal (control) | 133 | 0.558 | 8 |
| **low-rank k=3** | **3.67** | **0.882** | **35** |
| block, earned [`sfh`] | 50.7 | 0.645 | 28 |
| block, prefix (`dust` + `sfh`) | 50.7 | 0.645 | 30 |
| block, `sfh+dust` merged | 13.4 | 0.765 | 50 |
| block, `sfh+dust+met` merged | 1 | 1.000 | 64 |
| block, derived \|R\|>=0.9 | 73.0 | 0.612 | 12 |
| block, derived \|R\|>=0.7 | 13.5 | 0.765 | 28 |

### `ctl-jwst` — D = 9 continuity SFH. `cond(G)` raw 7.1e4

| arm | cond | recovered | entries |
|---|---|---|---|
| full dense (ceiling) | 1 | 1.000 | 81 |
| diagonal (control) | 1074 | 0.375 | 9 |
| **low-rank k=3** | **2.42** | **0.921** | **39** |
| block, earned [`sfh`] = prefix | 515 | 0.441 | 51 |
| block, `sfh+dust` merged | 74.1 | 0.614 | 65 |
| block, derived \|R\|>=0.9 | 75.6 | 0.613 | 51 |

### `stoch-field` — D = 74. `cond(G)` raw 5.4e4

| arm | cond | recovered | entries |
|---|---|---|---|
| full dense (ceiling) | 1 | 1.000 | 5476 |
| diagonal (control) | 2.40e4 | 0.074 | 74 |
| **low-rank k=3** | **10.9** | **0.780** | **299** |
| **low-rank k=10** | **1.45** | **0.966** | **824** |
| **low-rank k=20** | **1.11** | **0.990** | **1574** |
| block, earned [`psd`] | 2.09e4 | 0.087 | 4106 |
| block, prefix (`dust` + `psd` + `sfh`) | 1.23e4 | 0.136 | 4150 |
| block, `sfh+dust+met` merged | 1.07e4 | 0.149 | 4196 |
| block, derived \|R\|>=0.9 | 3978 | 0.239 | 682 |
| block, derived \|R\|>=0.7 | 615 | 0.411 | 2832 |
| block, derived \|R\|>=0.5 | 1.90 | 0.941 | 4234 |

## Findings

**1. Block layouts are dominated on both axes, everywhere.** At D = 8, low-rank
k=3 recovers 0.882 for 35 entries against the earned block layout's 0.645 for 28.
At D = 9, 0.921 for 39 against 0.441 for 51 — *more* memory for *less*
conditioning. At D = 74 the gap is three orders of magnitude in `cond` at 14x the
memory. There is no fixture and no storage budget in this set where a block
layout is the better buy.

**2. The per-group refinement is right, and it does not save the design.** It
works as intended — dropping the internally-uncorrelated `dust` block on `05`
cost exactly zero conditioning (50.73 both ways) and saved 2 entries, and on
`stoch-field` it correctly rejects the two groups whose correlations leave the
group. But the layout it endorses on `stoch-field` is *worse* than the naive
prefix layout (0.087 against 0.136) while still storing 4106 entries, because the
one group that earns a block is 64 of 74 coordinates. Measuring which groups earn
blocks improves the layout; it does not move the frontier.

**3. The dilemma has no third side.** A group earns a block only when its
coordinates correlate among themselves. On these fixtures such a group is either
2 parameters (`dust`, and only where it correlates at all — 2 entries, no memory
to save) or the field latent vector (64 of 74 — blocking it *is* the dense
matrix). Everything in between couples outward. The middle term the design is
reaching for requires groups that are simultaneously large enough to matter and
internally closed, and this posterior does not have them.

**4. The cross-group degeneracy is real and dominant.**
`dust_tau_diff x sfh_tsnorm_log_total_mass` at |R| = 0.804 on `05`;
`dust_tau_diff x sfh_cont_ratio_4` at |R| = 0.954 on `ctl-jwst`;
`psd_xi[50] x sfh_field_psd_tau_myr` at |R| = 0.995 on `stoch-field`. The
attenuation–mass–age stiffness runs *across* the domain prefixes, and merging the
groups to capture it repairs the conditioning by becoming dense: on `05`,
`sfh+dust+met` reaches `cond` 1 because at D = 8 that is all eight coordinates in
one block. Same on `ctl-jwst`. The merge that fixes the physics is the merge that
deletes the memory saving.

**5. Deriving the layout from the metric does not rescue it either.**
Thresholding the metric's own correlation form is a strictly better-informed
layout than any name-based rule, and produces the same trade. On `stoch-field`,
|R|>=0.9 buys 0.239 for 682 entries where low-rank k=10 buys 0.966 for 824. To
beat a rank-3 correction at all, block structure needs |R|>=0.5 at **4234
entries, 77 % of dense** — which retires the memory argument the design rests on.

**6. Why low-rank wins is a statement about the covariance, and getting the
convention wrong reverses the answer.** Under the metric-side convention low-rank
k=3 on `stoch-field` looks useless (`cond` 4.05e4, recovered 0.026); under the
covariance convention it is 10.9 and 0.780. Both are correct arithmetic on the
same matrix. The posterior's stiffness lives in a few *global degenerate*
directions, which are large eigenvalues of `Sigma` and small ones of `G`. An
adaptation estimates `Sigma`, so it sees the shape a low-rank correction
represents efficiently — and that no partition can represent at all, because a
global direction has support on every coordinate.

**7. The diagonal control is worse than useless at D = 74 on the metric side**
(`cond` 1.55e5 against a raw 5.41e4, recovered −0.097) and marginal on the
covariance side (0.074). That is independent confirmation that
`dense_mass_matrix=False` is a real cost on field posteriors, not a free
fallback — but the answer to it is the low-rank path, not blocks.

**8. An implementation hazard, recorded because it would have bitten.** The
raveled pytree keys are not the spec's free-parameter names: the field latents
flatten as `psd_xi[...]`, and `psd` is not one of NAMING_CONTRACT section 3.2's
domain prefixes. A prefix-based block resolver keyed on `spec.free_params` would
have silently produced a different partition from one keyed on the flat layout,
on the exact fixture where the partition matters most.

## Four layout hypotheses, tested rather than adopted

Each of these is a plausible design principle for where blocks should go. They
were treated as claims to check against this posterior, not rules to follow.

**H1 — "a block is one physical structure's parameters, not one parameter kind."
Distinction is real; outcome unchanged here.** The two layouts are identical by
construction on `05` and `ctl-jwst`, which carry one structure per domain, so
those fixtures cannot test it. `stoch-field` can: grouping by kind puts the mean
SFH shape and the stochastic field's hyperparameters in one `sfh` group of 7,
while grouping by structure splits them into `sfh_dpl` (5) and `sfh_field` (2).
The split is the better description — but neither piece earns a block
(`sfh_dpl` int/ext 0.266, `sfh_field` 0.144), so both layouts collapse to the
same `[psd]` block and score identically. H1 is a better *reason* for a grouping
than the prefix rule; on these fixtures it does not change what gets built.

There is one place the kind-vs-structure distinction does bite, and the earned
rule catches it independently: `dust_tau_bc` and `dust_tau_diff` are the same
*kind* of quantity (an optical depth) belonging to two different *structures* (a
birth-cloud screen and a diffuse screen). On `05` their mutual correlation is
0.001. A kind-based rule blocks them; the measurement says do not.

**H2 — "global or hierarchical hyperparameters stay diagonal." Refuted here, and
not marginally.** `stoch-field`'s field hyperparameters `sfh_field_psd_sigma` and
`sfh_field_psd_tau_myr` sit above the 64 latents, which is exactly the shape the
claim describes. They send **0.84 and 0.93** of their off-diagonal mass into the
latent block, and `psd_xi[50] x sfh_field_psd_tau_myr` is |R| = **0.995** — the
eighth-strongest correlation in a 74-dimensional matrix. Under the non-centered
parameterization tengri ships, a hyperparameter that sets the field's scale is
coupled to every latent it scales. Leaving these diagonal would discard the
strongest hyperparameter coupling in the model. The hierarchical population path
was not in scope for these fixtures, so this verdict covers field
hyperparameters only.

**H3 — "do not merge blocks; a merged block estimates spurious covariance."
Premise does not hold here, so the efficiency test was not run.** The claim is
only interesting when the cross-block mass is genuinely small. It is not:

| fixture | pair | cross mass | vs the larger group's own internal mass |
|---|---|---|---|
| `05` | `dust` x `sfh` | 1.538 | 3.302 (47 %) |
| `ctl-jwst` | `dust` x `sfh` | 2.390 | 5.825 (41 %) |
| `stoch-field` | `psd` x `sfh` | 12.666 | 4.361 (**291 %**) |

On `stoch-field` the cross-group mass is nearly three times the SFH group's
entire internal mass. A merged block there would be estimating real covariance,
not noise. That is why the sampling half of H3 — separate versus merged blocks
over six seeds — was **not** run: the regime that would make it informative does
not occur on any fixture here, and running it anyway would spend the compute
budget to test a premise the cheap check already falsified. It stays available if
a fixture with weak cross-group mass turns up.

Note also what the merge does at the ceiling: merging `sfh+dust` improves
conditioning on every fixture (50.7 to 13.4 on `05`). A ceiling comparison cannot
see estimation noise, so this is not evidence *against* H3's mechanism — it is
evidence that the covariance being estimated is real. Both statements are
consistent, and both point away from block structure, because the merge that
captures the mass is the merge that becomes dense.

**H4 — "derive the layout from the model instance rather than a fixed list."**
A design choice, not an empirical claim, and not tested as one. It is the right
choice and it is what the script does: groups come from the parameter set
actually being fit, so a subsystem reduced to one free parameter yields no block
without a special case. Recorded here so it is not mistaken for a measured result.

## What this does not say

* Nothing here is a measurement of the **shipped** low-rank sampler
  (`_hmc_low_rank_full_scan`). These are ceilings from exact eigenpairs; the
  shipped path estimates from draws and gradients and will do worse. The claim is
  that block structure's ceiling is below low-rank's ceiling, which is a
  statement about the two structures and holds however well either is estimated.
* All three fixtures are photometry at a single expansion point. The metric is
  position-dependent (`preconditioning.py`'s closing note: whitened stiffness
  runs 3.7e2 to 1.7e5 one posterior standard deviation out), so these are the
  geometry at the MAP, not over the region a chain explores.
* No spectroscopy fixture and no emission-line model was probed. A line-amplitude
  posterior has a genuinely block-separable nuisance structure and is the one
  place the conclusion might not transfer; it is also the place where #2146's
  analytic amplitude solve removes those coordinates from the sampled space
  entirely, which is a better answer than a block metric for the same problem.
* No sampler ran, so there is no seed sweep, no ESS and no wall clock here, by
  design. A conditioning ceiling is the cheapest thing that can falsify the
  design, and it did. The one sampling comparison that was on the table — separate
  versus merged blocks over six seeds — is gated on a structural precondition
  (small cross-block mass) that no fixture here satisfies, so it was not run.
* The `earned` thresholds (`internal >= 0.25`, `int/ext >= 1.0`) are round numbers
  chosen to separate the obvious cases, not calibrated ones. Nothing in the
  verdict turns on them: every group here is far from both boundaries (internal
  0.001 or 0.5-0.97; int/ext 0.001-0.34 or 2.0-2.4).

## Recommendation

**Do not build the block metric on this evidence.** The reusable artifact is the
per-group test itself: `internal` and `int/ext` on the correlation form of `G`
turn "which groups should be blocked" from a convention into a measurement, and
it applies unchanged to any future structured-metric proposal.

Carry the D = 74 rows to the low-rank work instead. `stoch-field` under a rank-10
correction has a ceiling of `cond` 1.45 at 15 % of dense's memory, which is the
result the block metric was hoping to be, and the sampler survey
(`bench/reports/2026-08-31_blackjax_sampler_survey.md`) never got a measured
low-rank row on that fixture. That is the follow-up worth funding.

Three adjacent findings, unrelated to whether blocks ship, are worth their own
issues: the diagonal fallback measurably *worsens* the D = 74 geometry
(Finding 7); `dust_tau_bc`/`dust_tau_diff` are mutually independent on the gate
fixture while both couple hard to the SFH (the per-group table), which is a
statement about the dust parameterization rather than about mass matrices; and
the non-centered field's hyperparameters are strongly coupled to their own
latents (H2), which bears on how that parameterization is sampled quite apart
from the metric.

## Reproduce

```bash
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/probe_block_metric_structure.py
```

Fixtures, mocks, seeds and every pinned `Fixed` value come from
`bench/scripts/benchmark_notebook_sampler.py`'s `NOTEBOOKS` registry unchanged —
no construction was re-pinned for this report, so the rows sit directly beside
every other campaign that used those fixtures.
