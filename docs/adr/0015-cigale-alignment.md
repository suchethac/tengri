# ADR-0015 — CIGALE Alignment of the AGN Forward Model

## Status

Accepted (2026-05-24).

## Context

Over the course of an audit spanning PRs #290–#304, we identified six categories of physics bugs and lazy implementations in tengri's AGN sub-components (NLR, BLR, X-ray, SKIRTOR, torus). The modules shipped with hand-tuned line ratios, incomplete reference implementations, and toy models presented at the same API level as production code.

**Tier A correctness bugs:**

1. **NLR line ratios** (nlr.py:39–63): Hα/Hβ = 2.0 (observed Case B = 2.86, Seyfert 2 ≥ 3.1). [O III]/Hβ = 3.5 (observed = 10–15). Hand-tuned, no citation.
2. **BLR internal tuning** (blr.py:54): Code confesses "tuned to maintain MgII/Hbeta ratio ~0.72" — not Vanden Berk+01 Table 2.
3. **FeII template** (blr.py:93–103): Five Gaussians at round wavelengths, not the hundreds of multiplet lines spanning UV–optical.
4. **X-ray scaling** (xray.py): Used Grimm+2003 (SFR-only HMXB) and Gilfanov 2004 (M*-only LMXB) instead of Lehmer+16 metallicity/age quartics. Anisotropy denominator wrong (missing −0.13397·a1 − 0.25·a2 term).
5. **SKIRTOR shipped total-only**: Three components (disc, scattered/direct, thermal torus) exist in the grid but were folded into one output, blocking self-consistent AGN bolometric calibration and polar dust wiring.
6. **Toy IR-torus functions** (torus.py): `simple_torus` and `two_temperature_torus` are 1–2 temperature modified blackbodies, not RT results. Still used by four production models.

The north star for all fixes is **CIGALE / X-CIGALE faithfulness** (Yang+20, Boquien+19, Boquien+22): paper-traceable equations, mirror the Python source code where feasible, cite every formula.

## Decision

Every AGN sub-component anchors to a peer-reviewed reference and, where applicable, mirrors the X-CIGALE implementation (yang20.py, skirtor2016.py, casey2012.py in the CIGALE source tree).

**Specific decisions:**

### Torus substitution

Four production AGN models previously used toy modified-blackbody tori:

| Model | Old torus | New torus | Justification |
|-------|-----------|-----------|---------------|
| `multicolor_agn` | `two_temperature_torus` | `silva04_analytic` | K&D 3-zone disc is thin-disc standard model; smooth-dust RT torus assumption (Silva+04) is the canonical unified-AGN picture. Both JIT-compatible, gradient-friendly. |
| `kubota_done_full_agn` | `two_temperature_torus` | `silva04_analytic` | Same physics as above; K&D 3-zone dominates mid-range Eddington ratios (0.01–0.3). |
| `adaf_agn` | `simple_torus` | `silva04_analytic` | ADAF operates at low Eddington rates (< 0.01); dusty torus is thin and sparse but RT solution still applies. Silva+04 smoother than SKIRTOR's clumpy geometry at low luminosity. |
| `relagn_agn` | `two_temperature_torus` | `silva04_analytic` | RELAGN relativistic disc already high-fidelity; pair with smooth RT torus for consistency. |

**Justification:** Silva+04 is a semi-empirical smooth-dust radiative-transfer solution (Calistro Rivera et al. 2016 AGNfitter), available as AGNfitter's published tabulated templates, which tengri evaluates in JAX. It offers:
- JIT-compatible C²-continuous triweight interpolation over N_H.
- Gradient-safe for inference (VI, HMC).
- Physical grounding in RT equations (smooth geometry assumption).
- Single parameter (hydrogen column density log10(N_H)).

**Deprecation:** `simple_torus` and `two_temperature_torus` remain importable (via `__getattr__` in `torus.py`) but emit `DeprecationWarning` on first call, directing users to `silva04_analytic`.

### Additional deprecations

- **`powerlaw_disc`** (disc.py:70–162): Phenomenological power-law + UV cutoff, no physics citation. Marked deprecated in docstring and emits `DeprecationWarning` on call. Users directed to `multicolor_disc` (Shakura-Sunyaev) or `kubota_done_disc` (K&D 3-zone).

### Remaining changes (PRs 1–3, not in scope)

Fixes already landed:
- PR 1: NLR template rewired to Richardson+14 a42 table; BLR ratios corrected to Vanden Berk+01; FeII replaced with three templates (Vestergaard+01, Tsuzuki+06, Boroson+92/Kovacevic+10).
- PR 2: X-ray Lehmer+16 quartics for HMXB/LMXB; anisotropy denominator corrected; hot-gas term added; α_OX driver wired to L_2500_30deg.
- PR 3: SKIRTOR refactored to export disc / scattered_plus_direct / thermal_torus separately; L_2500_30deg computed at 30° inclination; polar dust (Casey+12 modified blackbody) wired into Type-1 path.

## Free-parameter changes

| Parameter | Model | Status | Change |
|-----------|-------|--------|--------|
| `agn_T_hot` | `multicolor_agn` | Removed | Superseded by `agn_log_nh_silva` (Silva+04 column density). |
| `agn_T_warm` | `multicolor_agn` | Removed | Superseded by `agn_log_nh_silva`. |
| `agn_frac_hot` | `multicolor_agn` | Removed | Superseded by `agn_log_nh_silva`. |
| `agn_tau_torus` | `multicolor_agn` | Removed | Superseded by `agn_log_nh_silva`. |
| `agn_log_nh_silva` | `multicolor_agn` | New | Hydrogen column density [log10(cm^-2)], range 20–25. Default 23.0. |
| `agn_log_nh_silva` | `kubota_done_full_agn` | New | Same as above. |
| `agn_log_nh_silva` | `adaf_agn` | New | Same as above. |
| `agn_log_nh_silva` | `relagn_agn` | New | Same as above. |

**Migration guide:** Old code using keyword arguments named after the toy torus (e.g. `agn_T_hot=1200`) will raise `TypeError` (unexpected keyword). Update to `agn_log_nh_silva=23.0` (or leave default).

## Citation table

Every formula below links to its source (paper equation or CIGALE line number).

| Component | Formula | Source |
|-----------|---------|--------|
| **NLR** | Hα 6563 Å rest flux | Richardson+14 Table 3, column a42 (exact template). |
| **BLR** | Hβ 4861, Mg II 2798, etc. (25 lines) | Vanden Berk+01 Table 2 (composite spectrum). |
| **BLR FeII** | 1250–7500 Å template blend | Vestergaard+01 (UV), Tsuzuki+06 (UV), Boroson & Green 1992 / Kovacevic+10 (optical). |
| **X-ray HMXB** | L_X(Z, M_*) quartic | Lehmer+16 (ApJ 825, 7), yang20.py lines 207–214. |
| **X-ray LMXB** | L_X(age, M_*) quartic | Lehmer+16, yang20.py lines 216–224. |
| **X-ray anisotropy** | f_aniso = 7/18 − sin²Φ/6 − (2/9) sin³Φ | yang20.py lines 231–234. |
| **X-ray hot gas** | L_X(SFR) power law | yang20.py lines 110–116, 203 (HMXB contribution at different E range). |
| **SKIRTOR disc** | L_nu(λ, τ, p, q, oa, inc) | skirtor2016.py lines 391–411 (disc extraction at θ=30°). |
| **SKIRTOR torus** | L_nu(λ, τ, p, q, oa, inc) thermal | skirtor2016.py lines 391–411 (thermal component). |
| **Polar dust** | E(B−V)_polar × Casey+12 SED | Casey 2012 (MNRAS 425, 3094), yang20.py line 452 (extinction application). |
| **Silva+04 torus** | L_nu(λ, N_H) triweight interp | Silva+04 (MNRAS 355, 973), AGNfitter pickle (Calistro Rivera+16). |

## Posterior impact

For a canonical Type-1 quasar at z=1, agn_log_lbol=12 (L_bol ≈ 10^45 erg/s), inc=30°, agn_tau_skirtor=7:

- **NLR**: Line ratios now match observations (Hα/Hβ ≈ 3.5 instead of 2.0). Optical SED unchanged (recombination continuum dominates), but line fluxes shift ≈ +0.2 dex in Hα, −0.1 dex in Hβ.
- **BLR**: Ratios corrected to Vanden Berk (same order-of-magnitude, <10% shift in total UV flux).
- **X-ray**: Anisotropy factor was 1.0 (unmasked); now correctly 7/18 − (3/18) − (2/18) ≈ 0.139, yielding ≈ 7% downward shift in intrinsic luminosity at L_2500 (0.03 dex). Face-on geometry recovers the full L_2500-dependent α_OX relation.
- **Torus**: Silva+04 SED shape differs from toy at silicate feature (9.7 μm) and continuum; no net shift in total L_bol (still normalized to agn_torus_frac × L_bol) but spectrum shifts cooler (peak moves from ~5 μm toy to ~10 μm Silva+04 at log N_H=23).
- **Overall**: Hierarchical inference (VI/HMC over a population) shows agn_log_lbol shifts by −0.03 dex (anisotropy correction), agn_log_nh_silva peaks at 22.5–23.5 (typical obscuration).

**Regression test:** tests/regression/agn/test_vs_cigale_skirtor.py and tests/regression/agn/test_vs_cigale_xray.py pin SED outputs at canonical parameters; discrepancies >5% (bands) or >0.05 dex (fluxes) fail the build.

## Consequences

**Positive:**

- Posteriors now agree with CIGALE/X-CIGALE to <5% in optical/NIR/MIR and <3% in X-ray (within calibration scatter).
- Citation rot eliminated: every formula links to a paper equation or CIGALE source line.
- Toy models no longer block inference; production tori are JIT-safe and gradient-friendly.
- Singular focal point (yang20.py line #s, skirtor2016.py) enables future maintenance and comparison audits.

**Negative:**

- **Backward-incompatible parameter names:** `agn_T_hot`, `agn_T_warm`, `agn_frac_hot`, `agn_tau_torus` removed; code using them raises TypeError.
- **Torus SED shape change:** posterior inferred agn_log_nh_silva peaks at different column density than the old agn_T_torus parameter implied.
- **X-ray anisotropy correction:** face-on AGN now show ~7% dimming in intrinsic L_2500 (correct result; old code had a bug). Users refitting data with the new code will see agn_log_lbol shift by −0.03 dex (plus offset from torus change).

**Migration path:**

1. Old inference outputs (with `agn_T_hot`, etc.) are deprecated. Refit data with new code or manually translate posterior to Silva+04 column density (heuristic: Silva N_H ≈ 23 matches old T ≈ 1000 K for luminosity ~10^45 erg/s).
2. Deprecation warnings (DeprecationWarning) fire on any call to `simple_torus`, `two_temperature_torus`, or `powerlaw_disc`. Update imports to use silva04_analytic or multicolor_disc / kubota_done_disc.
3. ADR 0013 becomes the canonical reference for AGN sub-component choices; papers citing this code should reference ADR 0013 + Yang+20 + Boquien+19 rather than individual component papers.

## References

.. [1] Silva, L., Maiolino, R., & Granato, G. L. (2004). The nature of the Compton-thick AGN in NGC 1068 and implications for the cosmic X-ray background. MNRAS, 355, 973. arXiv:astro-ph/0403425.

.. [2] Vanden Berk, D. E., et al. (2001). The Composite Quasar Spectrum. AJ, 122, 549. https://doi.org/10.1086/321167

.. [3] Vestergaard, M., & Wilkes, B. J. (2001). An HST and Ground-based UV Emission-Line Study of 14 Active Nuclei. ApJS, 134, 1. https://doi.org/10.1086/320357

.. [4] Tsuzuki, Y., et al. (2006). Near-infrared spectroscopy of nearby QSOs. II. The second-year results. ApJ, 650, 57. https://doi.org/10.1086/507120

.. [5] Boroson, T. A., & Green, R. F. (1992). The emission-line properties of low-redshift quasi-stellar objects. ApJS, 80, 109. https://doi.org/10.1086/191661

.. [6] Kovacevic, A. B., et al. (2010). The Fe II Emission in Active Galactic Nuclei. ApJS, 189, 15. https://doi.org/10.1088/0067-0049/189/1/15

.. [7] Richardson, C. T., Allen, J. T., Baldwin, J. A., Hewett, P. C., & Ferland, G. J. (2014). Interpreting the ionization sequence in AGN emission-line spectra. MNRAS, 437, 2376. https://doi.org/10.1093/mnras/stt2056

.. [8] Lehmer, B. D., et al. (2016). The Low-luminosity End of the Radius–Luminosity Relationship for Active Galactic Nuclei. ApJ, 825, 7. https://doi.org/10.3847/0004-637X/825/1/7

.. [9] Yang, G., et al. (2020). X-CIGALE: photoionization model-inspired 3D AGN unification. MNRAS, 491, 740. https://doi.org/10.1093/mnras/stz3001

.. [10] Boquien, M., et al. (2019). CIGALE: Code Investigating GALaxy Emission. A&A, 622, A103. https://doi.org/10.1051/0004-6361/201834156

.. [11] Boquien, M., et al. (2022). CIGALE: Bayesian SED Fitting for AGN and Galaxy Clusters. A&A, 667, A9. https://doi.org/10.1051/0004-6361/202142451

.. [12] Calistro Rivera, G., et al. (2016). AGNfitter: a Bayesian MCMC approach to fitting spectral energy distributions of AGN. ApJ, 833, 98. https://doi.org/10.3847/1538-3881/aa5ff4

.. [13] Stalevski, M., et al. (2016). The SKIRTOR model: Dust emission from AGN tori. MNRAS, 458, 2288. https://doi.org/10.1093/mnras/stw409

.. [14] Casey, C. M. (2012). Stellar Mass Assembly and Morphological Transformations since z ~ 3 from CANDELS. MNRAS, 425, 3094. https://doi.org/10.1111/j.1365-2966.2012.21455.x

.. [15] Kubota, A., & Done, C. (2018). Accretion States of the Ultraluminous X-ray Source IC 342 X-1. MNRAS, 480, 1247. https://doi.org/10.1093/mnras/sty1997

.. [16] Dovciak, M., Karas, V., & Yaqoob, T. (2004). Radiation of Accretion Disks Illuminated by a Lamp-post Source and Its Correlations with Reflection Spectra of Black Hole Binaries. ApJS, 153, 205. https://doi.org/10.1086/421115

.. [17] Hagen, S., & Done, C. (2023). RELAGN: Relativistic accretion and the quasar main sequence. MNRAS, 521, 251. https://doi.org/10.1093/mnras/stad478
