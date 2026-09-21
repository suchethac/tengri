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
| M1a SKIRTOR opening angle | cigale §9e | **Normalization is correct; the residual is spectral shape along one axis.** The panel compares AGN dust as torus + polar summed on both sides. The integrated AGN dust budget agrees to **0.55 %**: pcigale's ∫(torus + polar) dν is 7.0145e32 erg/s against tengri's 7.0533e32. Band medians hold flat against optical depth (0.899×, 0.898×, 0.900× at τ_9.7 = 3, 7, 11) and against inclination (0.895× at i = 0°, 0.923× at 70°), while the opening angle carries a **1.8× swing** — 0.824× at 20°, 0.898× at 40°, 1.485× at 60°. So τ and i are innocent and the opening angle is the single axis to reconcile. The grid is the CIGALE v3 lineage at `data/skirtor_templates_v3.h5`, whose `spectra/norm` rises 199.7× across that same axis, which makes the combination of that factor the first thing to check. **`agn_torus_frac` is not the knob here:** the build passes `agn_ir_frac: Fixed(0.3)`, putting SKIRTOR on the `cigale_joint` tie, and `sed_model.py:1161` refuses an explicit covering factor beside an active fracAGN because `AGNSEDComponent` derives it from the dust-absorbed stellar luminosity and a hand-set value would be silently discarded. |
| M1c Fritz 2006 §9e | cigale §9e, band medians 0.095× / 0.195× / 0.129×, 12/12 bands outside 5 % | **An AGN residual after all, and it sits in the polar screen — the host explanation is withdrawn.** The panel now compares AGN dust as torus + polar summed, and the number barely moved from the old full-SED 0.094×, which already rules the host out: **neither side re-emits host dust here.** The tengri build carries no `dust_emission` group (`sed_dust_ir` is exactly 0.0) and pcigale's `_cigale_fritz_sed` chain carries no `dale2014` module, so they are consistent in lacking it and there is no "non-torus remainder" to chase. The earlier "torus ≈ 1.0×, host ~6.75 %" reading was wrong on both halves. Component by component at the edge-on node, tengri's **torus** carries 0.817× of pcigale's while its **polar screen** carries 0.097× — a ~10× polar deficit beside a roughly correct torus. The within-node spread (0.195× median against 0.040× worst) says shape, not one scale factor. **Caveat:** the component ratios come from a reconstructed build whose absolute scale does not reproduce the notebook's, so treat the torus-to-polar *ratio* as the finding and the absolute normalization as unestablished; the notebook needs its own diagnostic print to pin it. |
| Opening-angle convention | cigale §9e second Fritz node | **Not an out-of-prior node — two conventions for one angle.** pcigale's `opening_angle` is the *full* torus opening angle, with three allowed values (60°, 100°, 140°) of which 100° is the module default (`fritz2006.py:174-179`), and the module converts internally to a half-angle: `self.opening_angle = (180 - opening_angle) / 2` (`fritz2006.py:243`). tengri's `fritz_oa` is that half-angle, and the notebook applies the same conversion, so the 100° node reaches tengri as **40.0** — inside the `Uniform(20.0, 80.0)` declared at `reproduction/cigale/01_cigale.py:3287`. Nothing extrapolates, and the 0.244× needs its cause found elsewhere. Recorded because the two conventions share a name and a degree sign: a grid node reading `oa=100` beside a prior of (20, 80) invites the wrong reading, and the conversion is a line of arithmetic in the middle of a cell. |
| M1b Torus long-wavelength slope | synthesizer §9i, 0.24× median | **Not normalization — an emissivity difference in the Rayleigh-Jeans tail.** tengri and Synthesizer agree to 1.00× at and shortward of the peak at every temperature; longward of ~10⁵ Å tengri falls steeply while Synthesizer declines as a shallower power law, and the ratio runs off the bottom of the panel. A median over a broad band read that as a flat 0.24× and put it in M1. The quantity to reconcile is the modified-blackbody emissivity index, or whether one side emits a pure blackbody where the other applies a graybody. |
| M2 Nebular line ratios | cigale [O III] 0.36–0.44× and [O II] 2.08–2.83×, synthesizer 3.98×, prospector 0.45×, bagpipes 0.22× at 2 Z☉ | **Genuine physics, one-sided.** tengri fits its ionizing spectrum to the population it is given — the seven `ionspec_*` shape parameters resolve from the SSP weights, and the values Cue receives change with the star formation history. pcigale's nebular grid is indexed on (log U, Z_gas) alone and carries its own assumed continuum. The asymmetry is in the reference, is not removable inside the comparison, and BAGPIPES — which also derives its spectrum — agrees with tengri to 1 %. |
| M3 IR template normalization | agnfitter `dh02_ce01` 0.342 dex, prospect_r Dale 0.74–0.88×, prospector DL07 0.89×, cigale dl2014 1.36× | **Partly metric artefact.** The `dh02_ce01` libraries are bit-identical between the two codes; 0.342 dex was peak-normalizing two arms whose maxima sit at different wavelengths. Common-wavelength normalization gives 0.242 dex, whose cause is not yet identified. A separate defect found on the way: `dust_log_lir` carries a hardcoded closure default of 10.0, so the library's *shape* does not track the luminosity it is asked for. |
| M4 Two-screen and bump mapping | cigale 0.579×, prospector 0.78×, prospect_r 0.66–0.896×, bagpipes A(2175)/A_V 2.082 vs 2.079 (closed) | **One arm closed; three open with the method now specified.** The **bagpipes** arm closed via #2397 (PR #2442): `salim_sbl18` now normalizes the bump by the modified R_V, and the re-rendered §7 reads BAGPIPES 2.082 vs tengri 2.079 at δ = +0.3, B = 3 (0.14%, against the 10% gap before the fix; both sides anchored at exactly 2175 and 5500 Å — the earlier nearest-node normalization put a ~0.4% grid-sampling term on top). The original diagnosis, kept for the record: `salim_sbl18` normalized its 2175 Å Drude bump by a fixed `R_V = 4.05` where Salim, Boquien & Lee (2018) Eq. 3 uses the modified `R_V,mod` of Eq. 4, `R_V,Cal/[(R_V,Cal+1)(4400/5500)^δ − R_V,Cal]`. That expression is the tilted curve's own R_V, derivable by imposing E(B−V) = 1, and the paper's footnote 7 names the resulting bias as a factor `R_V,Cal/R_V,mod` — 0.673 at δ = 0.3, which accounts for the 0.8986× gap once the `k(5500)` renormalization is applied. Note the fixed `rv` in `salim_sbl18` cancels against its own `k_5500` renormalization, so the quantity that differs is the bump-to-base ratio, `B/4.05` against `B/R_V(δ)`. Scope any fix to that law alone: `kriek_conroy` and `noll09` place the bump *inside* the tilt, correctly, per FSPS `dust_type=4`/Kriek & Conroy (2013) Eq. 3 and Noll et al. (2009). **The cigale, prospector and prospect_r arms remain unsupported.** A(λ)/A_V is recoverable from any code without a curve API by pushing a flat `L_lambda = 1` through its attenuation step and taking `−2.5 log10(out/in)` normalized at 5500 Å; the requirement that attempt missed is that **both arms must be the same law** — a tengri Salim curve measured against pcigale's `dustatt_2powerlaws` reports the difference between two laws, not an implementation difference. |
| M5 Parametric SFH closed forms | cigale periodic 100 % of peak, synthesizer LogNormal 37 %, prospect_r 1.75×, prospector 17.3 % | **Split four ways; two candidate causes eliminated.** **LogNormal** is a genuine form difference: tengri's is a lognormal in cosmic time since formation (`T = age − t_lookback`) with a 1/T Jacobian and `sigma = width·ln(10)`, mode at `exp(mu − sigma²)`, and the residual survives normalizing both arms to the same formed mass. **Periodic** is a burst-edge phase offset — the rectangular on/off edges land at different lookback times, so at a discontinuity one arm is on while the other is off and the difference reaches 100 % of the peak by construction; not quantified as a time offset. **Delayed-τ 17.3 %: the SSP age-binning explanation is insufficient.** Switching `age_kernel` between `'cic'` and `'dsps'` at τ = 0.3, age = 1 moves L_nu by only 0.7–1.2 % (1500–10000 Å), and 1.2–2.3 % at τ = 1, age = 5 — consistent with the ~1.2 % optical CSP bias already documented for that kernel, and short of 17.3 % by more than an order of magnitude. Cause unidentified. **Truncated 1.75 %: the soft-edge explanation is insufficient.** `window_weight` returns the exact cell average of the window indicator, so it differs from a hard cut in **one** grid cell; formed mass beyond the truncation boundary is 2.3 % unskewed, **0.0 %** at skew = 5, and 8.4 % on a wider peak. A one-cell edge cannot sustain a 1.75× tail ratio. Cause unidentified. Caveat: both refutations are one-sided (tengri only) — they eliminate the stated mechanism without measuring the reference arm. |

## Four metric artefacts, each found more than once

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
4. **A conserved quantity cannot test a distribution.** ∫(torus + polar) dν is pinned to the
   AGN dust budget on both sides, so its agreement validates the budget and nothing else.
   Setting `agn_torus_frac` to 1.0 on the Fritz build drives tengri's polar term to exactly
   zero and its torus to 3.53× pcigale's, and the summed integral then lands on 1.0092× — the
   same figure as the absorbed-energy ratio, and an apparent confirmation from two numbers that
   were never independent, because one determines the other. The component split is what
   distinguishes the two codes; the sum is the one statistic guaranteed not to. This is the
   general form of artefact 2, and it bites hardest when two arms are compared through a
   quantity that energy conservation fixes on both sides.

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
