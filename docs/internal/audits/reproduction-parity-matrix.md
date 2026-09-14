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
| M1 AGN luminosity bookkeeping | prospector 0.45×, prospect_r 0.28–0.35×, cigale 0.094×, agnfitter 1.22–1.39× | **Notebook setup, for the cases that are flat.** `agn_torus_frac` is a covering factor with default 0.5; the notebooks left it unset while the reference builds emit into 4π. Pinning it to 1.0 moves cigale's torus 0.4989× → 0.9978× and tengri/FSPS 0.52× → 1.04×. The AGN-parity crossval tests already pin it. Not a code defect. **But the M1 grouping was too broad** — see the two rows below, which were folded into it on the strength of a band median and are not normalization at all. |
| M1a SKIRTOR opening angle | cigale §9e | **Localized, not a sustained offset.** Band medians: every `oa=40°` case agrees to 0.973–0.988× across three optical depths and three inclinations, while `oa=20°` is 0.927× and `oa=60°` 1.078×. The medians are all within 8 %; what distinguishes the off-nominal opening angles is the *worst band* — 1.761× and 1.687×, with 11 of 13 bands outside 5 % against 6 of 13 at `oa=40°`. So the disagreement is a band-localized excursion that appears away from the fiducial opening angle, not a scale error. Optical depth and inclination are innocent. Established: the grid is the CIGALE v3 lineage at `data/skirtor_templates_v3.h5`, and its `spectra/norm` factor rises 199.7× across the opening-angle axis. Whether tengri and pcigale combine that factor identically is **not established** — two probe attempts did not produce the comparison, the second because its own pcigale call silently fell back to defaults. |
| M1c Fritz 2006 torus | cigale §9e **0.141–0.244×, 13/13 bands outside 5 %** on all three nodes; prospect_r §9 0.28–0.35× | **The largest AGN discrepancy in the set, and it is not localized.** Every band of every node is low by a factor of four to seven against pcigale and by about three against ProSpect. Two independent reference codes disagreeing in the same direction, on every band, points at tengri's Fritz normalization rather than at either reference. This was folded into the M1 span and is the one AGN row that most warrants a fix. |
| M1b Torus long-wavelength slope | synthesizer §9i, 0.24× median | **Not normalization — an emissivity difference in the Rayleigh-Jeans tail.** tengri and Synthesizer agree to 1.00× at and shortward of the peak at every temperature; longward of ~10⁵ Å tengri falls steeply while Synthesizer declines as a shallower power law, and the ratio runs off the bottom of the panel. A median over a broad band read that as a flat 0.24× and put it in M1. The quantity to reconcile is the modified-blackbody emissivity index, or whether one side emits a pure blackbody where the other applies a graybody. |
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
   The converse also bites: a median *reports* a flat factor where the arms actually agree
   exactly over part of the range and diverge over the rest. M1b was grouped as a 0.24×
   normalization offset on that basis, and is a slope difference in the Rayleigh-Jeans tail
   beside 1.00× agreement at the peak. Read the figure before grouping by the statistic.
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
