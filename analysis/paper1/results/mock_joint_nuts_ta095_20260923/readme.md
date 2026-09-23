# Mock joint NUTS, target_accept 0.95, warm-started — refused

The third full attempt, and the one that closes the tuning question. It ran
14.70 h (52934 s), finished cleanly, and does not clear the figure's gate.

| diagnostic | cold | ta085 | **this run** | gate |
|---|---:|---:|---:|---|
| max split R-hat | 1.0077 | 1.0024 | **1.0099** | < 1.01, passes |
| ess_min | 242 | 520.3 | **117.7** | >= 400, **fails** |
| divergences | 19 | 79 | **2** | 0, **fails** |
| chi2/dof fitted | — | 1.3075 | **1.5474** | — |
| worse than truth | — | 345.7 | **700.5** | — |

Truth is chi2/dof 1.0738 and the L-BFGS warm start reaches 1.0480, *better*
than truth. So the optimizer finds the basin and hands it to the sampler.

## What this run establishes

Raising target_accept is not the remedy, and the three runs together say why.

Divergences fell 79 → 2, which is what the lever was pulled for. Everything
else got worse: the effective sample size fell 520 → 118, R-hat rose, and the
chain ended **twice as far from the mode** as at 0.85 — 700.5 worse in chi2
against 345.7.

That combination is the point. A smaller step buys few divergences precisely
by never attempting the transitions that would diverge. This is not a sampler
whose pathologies were fixed; it is a sampler that stopped moving. Read
divergence count alone and 0.95 solved the problem. Read it beside ESS and
chi2 and it made it worse, at 3.6 h of extra wall clock.

Three attempts, roughly 26 h of sampling, none reached the basin. The lever
that remains untouched is the one `divergence_geometry` pointed at after
ta085: the metric. `dense_mass_matrix` is `False` on all three runs, and the
ta085 geometry verdict was *diffuse* — the offending directions lie **between**
coordinates, which a diagonal metric cannot represent at any step size. Both
launches also carried the library's own warning, unheeded: "NUTS with 36
dimensions may be slow. Consider method='vi' or method='mcmc_raytrace' for
D>20."

## What it does not establish

Nothing about recovery. The run's own summary says so: the fit never reached
the mode, so the per-parameter deltas in the log measure where the sampler
stopped, not where the posterior is. **Do not quote them**, and in particular
do not quote the largest-delta line (`neb_eline_sigma_kms` +9.7799) as a
recovery figure. The same warning is on the ta085 archive for the same reason.

Section 3's three TBDs cannot be filled from this run.

## The geometry verdict changed meaning, not value

`divergence_geometry` on ta085 read *diffuse* from 79 divergent draws. On this
run it returns **no verdict**: two divergent draws cannot locate a shape, since
the median offset carries an uncertainty of about 0.89 sd at n=2 against a
2.0 sd funnel threshold. Before `316ac3352` the module would have answered
"diffuse" here too, from a median of two numbers, and the two runs would have
appeared to agree. They do not agree; the second one is silent.

## Provenance

The `provenance` block records `2e5cdac5c +uncommitted changes`. The commit is
one made *during* the run and does not identify the tree that produced it; the
dirty file was an untracked `results/grid_summary.json` written by a separate
analysis, since removed. No `src/tengri/` file changed during the run, so the
forward model is the tree as it stood at launch, 2026-09-22 18:08.

`mock_joint_mcmc_nuts.npz` is **not** in git — `.gitignore` excludes `*.npz`,
as it does for the ta085 archive. The draws exist only on the machine that ran
them.

## The metric lever, measured rather than argued

`posterior_conditioning` (added 2026-09-23) puts a number on the claim above. A
diagonal mass matrix rescales each coordinate by its own standard deviation and
nothing else, so what it leaves behind is exactly the correlation matrix, and
its condition number is the conditioning HMC still faces afterwards.

On **ta085**, 2400 draws, 36 moving parameters, ess_min 520.3 (14.5 per
parameter): **condition number 464.3**. Trajectory length scales as its square
root, so roughly **21x** in leapfrog steps per draw.

The top pairs are structural rather than incidental — adjacent continuity-SFH
bins trading mass, and the two dust screens trading optical depth:

    -0.756  sfh_cont_ratio_3        x  sfh_cont_ratio_4
    -0.742  sfh_cont_log_total_mass x  sfh_cont_ratio_4
    +0.692  dust_alpha_dl14         x  dust_gamma_dl
    -0.666  sfh_cont_ratio_1        x  sfh_cont_ratio_2
    -0.636  dust_tau_bc             x  dust_tau_diff

Two pairs exceed |r| = 0.7 and **none** exceeds 0.9, while the matrix is
conditioned at 464. That gap is the finding: this is many moderate correlations
compounding across 36 dimensions, not one bad pair a reparameterization would
fix — which is `divergence_geometry`'s "diffuse" verdict restated in a form
that names a remedy.

**This run cannot supply the number.** At 117.7 effective samples over 36
parameters it reaches 3.3 per parameter, under the tool's floor of 5, and is
refused: a correlation matrix estimated from that is noise wearing a condition
number. The figure above is ta085's alone.
