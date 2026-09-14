# Reproduction parity: residuals grouped by mechanism

The six reproduction notebooks compare tengri against CIGALE, BAGPIPES, Prospector/FSPS,
ProSpect, AGNfitter-rX and Synthesizer. Read notebook by notebook, their >10 % residuals look
like eighteen separate disagreements. Read across the notebooks, they fall into a handful of
families that recur against four different reference codes at once — the signature of one
convention on one side, not of four codes each differing in their own way.

This page records what each family turned out to be, so the same residual is not attributed
twice.

## The verdicts

| Mechanism | Symptom span | Verdict |
|---|---|---|
| M1 AGN luminosity bookkeeping | prospector 0.45×, synthesizer 0.24×, prospect_r 0.28–0.35×, cigale 0.094×, agnfitter 1.22–1.39× | **Notebook setup.** `agn_torus_frac` is a covering factor with default 0.5; the notebooks left it unset while the reference builds emit into 4π. Pinning it to 1.0 moves cigale's torus 0.4989× → 0.9978× and tengri/FSPS 0.52× → 1.04×. The AGN-parity crossval tests already pin it. Not a code defect. |
| M2 Nebular line ratios | cigale [O III] 0.36–0.44× and [O II] 2.08–2.83×, synthesizer 3.98×, prospector 0.45×, bagpipes 0.22× at 2 Z☉ | **Genuine physics, one-sided.** tengri fits its ionizing spectrum to the population it is given — the seven `ionspec_*` shape parameters resolve from the SSP weights, and the values Cue receives change with the star formation history. pcigale's nebular grid is indexed on (log U, Z_gas) alone and carries its own assumed continuum. The asymmetry is in the reference, is not removable inside the comparison, and BAGPIPES — which also derives its spectrum — agrees with tengri to 1 %. |
| M3 IR template normalization | agnfitter `dh02_ce01` 0.342 dex, prospect_r Dale 0.74–0.88×, prospector DL07 0.89×, cigale dl2014 1.36× | **Partly metric artefact.** The `dh02_ce01` libraries are bit-identical between the two codes; 0.342 dex was peak-normalizing two arms whose maxima sit at different wavelengths. Common-wavelength normalization gives 0.242 dex, whose cause is not yet identified. A separate defect found on the way: `dust_log_lir` carries a hardcoded closure default of 10.0, so the library's *shape* does not track the luminosity it is asked for. |
| M4 Two-screen and bump mapping | cigale 0.579×, prospector 0.78×, prospect_r 0.66–0.896×, bagpipes A(2175)/A_V 2.09 vs 2.32 | Open. |
| M5 Parametric SFH closed forms | cigale periodic 100 % of peak, synthesizer LogNormal 37 %, prospect_r 1.75×, prospector 17.3 % | Open. Each reference parameterizes the same family differently — truncation center, lognormal median against mode, phase origin — so part of this is definitional rather than a disagreement. |

## Three metric artefacts, each found more than once

Worth naming, because each reads as total disagreement while both arms are correct:

1. **Regrid zero-fill.** Interpolating one arm onto the other's grid outside its support fills
   with zeros, and the ratio then reports the gap rather than the physics. `sweep_fig` and
   `overlay_ratio_fig` now refuse a case with no overlapping positive region instead of
   plotting one.
2. **Band-median blindness.** Wherever energy balance pins the integral, a median over a broad
   band cannot see a shape difference — the two arms agree by construction in that statistic.
3. **Peak normalization across mismatched support.** Dividing each arm by its own maximum is
   meaningful only when the maxima sit at the same wavelength. M3's 0.342 dex was this.

## Comparison-setup corrections

These showed as large residuals while comparing unlike quantities. Each is now like-for-like:

| Notebook | Was | Now |
|---|---|---|
| prospector §3 | panels labeled different mass conventions, formed against surviving | one convention, stated once |
| synthesizer §6b | modified blackbody against an *unmodified* blackbody, 170× | 1.000× at β_ir = 0 |
| synthesizer §4b | `narayanan_z` evaluated at z = 0 only | swept over z = 0–6 |
| cigale §6c | `schreiber2016` dust temperature swept past the template's range | swept inside it, range stated |
| bagpipes | two-component screen against a single screen | dropped, with the reason |
| agnfitter §6 | cold dust off a node | on a shared node |

An off-grid sweep is the same disease in a different place: `Z = 0.041` sits between pcigale's
0.020 and 0.050 nodes and between BAGPIPES' 0.0142 and 0.0355, so its 43.5× was extrapolation
on both sides.

## The SFH time axis

Three SFH comparisons drew nothing at all, and the cause was not the time range but the
*order*. `numpy.interp` requires an increasing sample axis and does not verify it; given a
descending one it returns the fill value at every point. `sfh_grid_lbt_yr` ascends in lookback
time, so `t_cosmic = age − lbt` descends, and every regridded point became the `left=0.0`
fill — 0 of 40 points positive in a minimal reproduction. Both helpers now sort each arm onto
an increasing axis, which covers all 41 sweep call sites rather than three cells.

The same axis also ran negative for samples older than the galaxy, to −8.8 Gyr on a 5 Gyr
delayed-τ, so the cells additionally keep the epoch after formation. That costs 0.01 % of the
formed mass on the delayed-τ and 0.3 % on the sfh2exp (∫SFR dt = 0.9999 and 0.9970 against a
unit target).

What the blank figures were hiding is agreement: SFR(t) shape, tengri against pcigale on
tengri's own grid, is a median 1.00002× for the delayed-τ and 1.00015× for the sfh2exp.
