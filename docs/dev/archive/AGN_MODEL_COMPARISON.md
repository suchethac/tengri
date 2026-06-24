# AGN Model Comparison Across SED Fitting Codes

**Last updated:** 2026-03-18

## Summary Comparison Table

| Feature | **tengri** | **CIGALE** | **Prospector/FSPS** | **ProSpect** | **Bagpipes** | **Synthesizer** | **AGNfitter-rx** | **qsogen** |
|---|---|---|---|---|---|---|---|---|
| **Disc model** | Power-law; SS73 multicolor; Kubota & Done (2018) 3-zone | Schartmann (2005) or Feltre (2012) parametric; slope adjustable via delta_AGN | Broken power-law (from Nenkova 2008 templates) | Fritz (2006) isotropic central source (power-law, 0.001-20 um) | None | Broken power-law; Kubota & Done (2018) qsosed; Hagen & Done (2023) relqso | Richards (2006) empirical; Slone & Netzer (2012) alpha-disc; Kubota & Done (2018) AGNSED; Temple (2021) empirical composite | Empirical broken power-law continuum with BB hot dust |
| **Disc type** | Analytic (JAX) | Template library | Template library | Template library | -- | Analytic + CLOUDY grid | Template libraries (4 options) | Empirical parametric |
| **Torus model** | Simple BB; Two-temperature (SKIRTOR-inspired) | Fritz (2006) smooth RT; SKIRTOR (2016) clumpy two-phase | Nenkova (2008) CLUMPY | Fritz (2006) smooth RT | None | Analytic (within UnifiedAGN) | Silva (2004) smooth; Nenkova (2008) clumpy; SKIRTOR (2016) two-phase; CAT3D-Wind (Honig & Kishimoto 2017) | BB hot dust component |
| **Torus type** | Analytic (modified BB + silicate opacity) | Radiative transfer templates | Radiative transfer templates | Radiative transfer templates | -- | Analytic | RT template libraries (4 options) | Empirical (single BB) |
| **Torus geometry** | Smooth (simple) or two-temperature (standard/kubota_done) | Smooth (Fritz) or clumpy two-phase (SKIRTOR) | Clumpy | Smooth | -- | Combined with disc | Smooth or clumpy or two-phase or wind (choice of 4) | N/A (empirical) |
| **Torus params** | 2-5 (T, tau, f_hot, T_warm, covering factor) | 7 (Fritz); 6+2 (SKIRTOR + polar dust) | 1 (agn_tau) | 7 (Fritz) | -- | Geometry via inclination + covering factor | 2-4 per model choice | 1 (tbb, bbnorm) |
| **Emission lines** | Yes: NLR (11 forbidden + Balmer lines) + BLR (9 permitted lines), analytic Gaussian profiles | No (AGN emission lines not included) | No (AGN lines not included; nebular lines from SF only) | No | -- | Yes: NLR + BLR via CLOUDY photoionization grids | Partial (THB21 disc model includes blended lines) | Yes: 4 empirical emission-line templates with Baldwin effect |
| **NLR/BLR decomposition** | Yes (unified_nlr_blr model with geometric masking) | No | No | No | -- | Yes (separate NLR/BLR covering fractions) | No explicit decomposition | Partial (scal_nlr parameter scales narrow lines) |
| **Geometric masking** | Yes (smooth sigmoid for differentiability) | Yes (viewing angle determines Type 1/2) | No | Yes (psy parameter) | -- | Yes (inclination-dependent) | No explicit masking | No (Type 1 quasars only) |
| **Polar dust** | No | Yes (E(B-V), T=100K, SMC extinction law) | No | No | -- | No | No | No |
| **X-ray extension** | Yes (separate xray_agn_corona module) | Yes (X-CIGALE: alpha_ox, Gamma) | No | No | -- | No | Yes (alpha_ox, Gamma, L_2keV) | No |
| **Radio extension** | Yes (separate radio_agn module) | Yes (radio module) | No | No | -- | No | Yes (power-law + double power-law) | No |
| **BH physics** | Yes (M_BH, L/L_Edd, spin, ISCO) | No | No | No | -- | Yes (M_BH, mdot, spin in relqso) | Yes (M_BH, mdot in KD18/SN12) | No |
| **Differentiable** | Yes (pure JAX, JIT-compilable) | No (grid-based chi-sq / Bayesian) | No (MCMC sampling over templates) | No (R, grid-based) | -- | Partial (Python/C, not autodiff) | No (MCMC) | No (forward model only) |
| **Total AGN free params** | 3 (simple), 6 (standard), 9 (kubota_done), 15 (unified_nlr_blr) | 7 (Fritz) or 8-9 (SKIRTOR + polar dust + fracAGN + delta_AGN) | 2 (fagn, agn_tau) | 7+ (Fritz params + AGN fraction) | 0 | 4-6+ (M_BH, mdot, inclination, metallicity, covering fractions) | 5-19 (depends on component selection) | ~10-15 (continuum + lines + dust + galaxy) |
| **Inference method** | HMC/NUTS, geoVI, ray-tracing (gradient-based) | Chi-sq minimization + Bayesian (grid) | Nested sampling (dynesty) or MCMC (emcee) | Optimization (optim/MCMC in R) | Nested sampling (MultiNest/nautilus) | Forward model (not a fitter) | MCMC (emcee) | Forward model (not a fitter) |

---

## Detailed Code-by-Code Analysis

### 1. tengri (this code)

**Architecture:** Four AGN complexity tiers, all pure JAX and fully differentiable.

**Disc models:**
- `simple`: Power-law F_nu ~ nu^alpha with exponential UV cutoff. 2 params: alpha, T_max.
- `standard` / `kubota_done`: Shakura-Sunyaev multicolor disc. Radial temperature profile T(r) ~ r^{-3/4} integrated over 50 logarithmic annuli. Params: log M_BH, log L/L_Edd, spin (sets ISCO via Bardeen-Press-Teukolsky formula), cos(inclination).

**Torus models:**
- `simple_torus`: Single-temperature modified blackbody with silicate opacity at 9.7 um. 2 params: T_torus, tau_torus.
- `two_temperature_torus`: Hot (sublimation, ~1200K) + warm (outer, ~300K) mixture with silicate opacity. SKIRTOR-inspired. 4 params: T_hot, T_warm, f_hot, tau_torus.

**Emission lines (unified_nlr_blr only):**
- NLR: 11 lines ([OII], [NeIII], H-beta, [OIII] doublet, [OI], [NII] doublet, H-alpha, [SII] doublet) with Gaussian profiles, FWHM ~500 km/s. Isotropic (not masked by torus).
- BLR: 9 lines (Ly-alpha, NV, SiIV+OIV], CIV, CIII], MgII, H-gamma, H-beta, H-alpha) with broad Gaussians, FWHM ~5000 km/s. Masked by torus via smooth sigmoid.
- Line strengths calibrated to Vanden Berk et al. (2001) composite and typical Seyfert 2 spectra.

**Key strengths:**
- Fully differentiable end-to-end: enables gradient-based inference (HMC/NUTS, geoVI).
- Modular: can mix disc + torus + NLR + BLR independently.
- BH physics: ISCO from spin, Eddington ratio, gravitational radius.
- Smooth geometric masking preserves gradient flow (sigmoid, not hard cutoff).

**Key weaknesses:**
- Analytic line profiles, not from photoionization grids (less physically accurate line ratios than CLOUDY).
- No polar dust component.
- Two-temperature torus is a simplification of full RT; cannot reproduce detailed silicate feature profiles.
- No dust-reprocessed emission from NLR/BLR.

---

### 2. CIGALE (+ X-CIGALE)

**Disc model:** Parametric power-law disc from Schartmann (2005) or updated Feltre et al. (2012). A free parameter delta_AGN adjusts the slope in the 0.125-10 um range. Template-based, not analytic.

**Torus models (two options):**

*Fritz et al. (2006) -- smooth torus:*
- Radiative transfer through smooth dust distribution.
- 7 parameters: r_ratio (R_out/R_in, default 60), tau (optical depth at 9.7 um), beta (central source slope), gamma (dust density gradient), opening_angle (full, default 100 deg), psy (viewing angle), fracAGN.
- Includes scattered + thermal dust emission with energy conservation to 1-10%.

*SKIRTOR (Stalevski et al. 2016) -- clumpy two-phase torus:*
- 3D radiative transfer, two-phase medium (high-density clumps + smooth inter-clump dust).
- 6 torus template parameters: tau_9.7 (edge-on optical depth), p (radial density gradient), q (polar density gradient), opening angle (half-opening of dusty cone), Y (R_out/R_in ratio), inclination.
- Additional CIGALE parameters: delta_AGN (disc slope), fracAGN (AGN fraction of total IR), E(B-V) polar dust (SMC extinction), polar dust temperature (~100K), polar dust emissivity index.
- Total: ~8-9 free parameters.

**Emission lines:** Not included in AGN module. Galaxy nebular emission is separate.

**Key strengths:**
- Full radiative transfer torus templates: physically accurate IR SEDs including silicate features.
- SKIRTOR is state-of-the-art clumpy torus with polar dust.
- X-CIGALE extends to X-ray (alpha_ox correlation) and radio.
- Large user community, extensive validation.

**Key weaknesses:**
- No AGN emission lines (UV/optical BLR/NLR not modeled).
- Template grid: discrete parameter values, no interpolation for gradients.
- No BH physics (mass, spin, accretion rate not parameters).
- Cannot do gradient-based inference.

---

### 3. Prospector / FSPS

**Disc model:** Broken power-law incident spectrum from Nenkova et al. (2008) CLUMPY templates. Not independently parameterized -- the disc shape is baked into the CLUMPY template.

**Torus model:** Nenkova et al. (2008) CLUMPY torus. Radiative transfer through clumpy medium. In FSPS, the full CLUMPY parameter space (which has ~6 params) is collapsed to a single shape parameter: agn_tau (optical depth at 5500A of individual clumps, range 5-150).

**Parameters:**
- `fagn`: AGN luminosity as fraction of stellar bolometric luminosity. Can exceed 1.
- `agn_tau`: Torus optical depth (shape parameter). Default 10. Valid range 5-150.
- Total: **2 free parameters**.

**Emission lines:** Not included for AGN. Nebular emission from star formation is separate.

**Key strengths:**
- Minimal parameter overhead: easy to include AGN without blowing up parameter space.
- Physically motivated CLUMPY templates.
- Well-integrated with Prospector's Bayesian framework (dynesty).
- Widely used and validated (Leja et al. 2018).

**Key weaknesses:**
- Only 2 parameters: very limited AGN complexity. Cannot fit detailed AGN SEDs.
- No BH physics, no emission lines, no geometric masking.
- Template extrapolation outside tau=[5,150] is unreliable.
- No disc/torus decomposition: everything is one combined template.

---

### 4. ProSpect

**Disc + Torus model:** Uses Fritz et al. (2006) smooth torus radiative transfer templates, same underlying model as CIGALE's Fritz module.

**Parameters:** Same 7 Fritz parameters as CIGALE: r_ratio, tau, beta, gamma, opening_angle, psy (viewing angle), fracAGN. The help functions `?Fritz` and `?AGN_UnOb` provide details.

**Emission lines:** Not included for AGN.

**Key strengths:**
- Full Fritz (2006) RT templates: proper radiative transfer.
- Integrated into ProSpect's R-based fitting framework (Robotham & Bellstedt 2020).
- 7 parameters give good flexibility for torus fitting.

**Key weaknesses:**
- No clumpy torus option (smooth only).
- No AGN emission lines.
- No BH physics.
- R-based: not easily combined with Python/JAX frameworks.
- No polar dust.

---

### 5. Bagpipes

**AGN model:** **None.** Bagpipes (Carnall et al. 2018) does not include an AGN component as of v1.3.5 (Feb 2026). Objects with known AGN contamination are typically excluded from fitting.

**Components available:** Stellar populations (BC03), dust emission (Draine & Li 2007), nebular emission, dust attenuation.

**Key weakness:** Cannot fit AGN-host decomposition at all.

---

### 6. Synthesizer (UnifiedAGN)

**Disc models (3 options):**
- Broken power-law (similar to Feltre et al. 2016).
- `qsosed`: Kubota & Done (2018) simplified AGNSED. Free params: M_BH, accretion rate (mdot). Bolometric luminosity computed from mass and mdot with radiative efficiency eta=0.1.
- `relqso`: Hagen & Done (2023) extension adding BH spin as a third parameter. Full GR ray tracing.

**Torus model:** Integrated within the UnifiedAGN framework (analytic, geometry-dependent).

**Emission lines:**
- NLR and BLR modeled via CLOUDY photoionization grids.
- Grids cover physical conditions expected in both regions.
- Default covering fractions: 0.1 for both NLR and BLR.
- Default ionization parameter: 0.1, hydrogen density: 10^5 cm^-3.

**Parameters:** M_BH, accretion rate, inclination, metallicity, NLR/BLR covering fractions. Total: ~4-6+.

**Key strengths:**
- Self-consistent disc + NLR + BLR + torus model.
- CLOUDY grids: physically accurate emission-line ratios (unlike analytic profiles).
- Multiple disc models including relativistic effects.
- Open-source Python.

**Key weaknesses:**
- Forward model / grid generator, NOT a fitter. Must be coupled to external inference.
- UnifiedAGN paper (Wilkins et al.) still in prep; full details not yet published.
- Not differentiable.

---

### 7. AGNfitter / AGNfitter-rx

**Disc models (4 options):**
- R06 (Richards et al. 2006): Semi-empirical quasar SED + NIR BB tail + SMC dust reddening. 2-3 params.
- SN12 (Slone & Netzer 2012): Theoretical alpha-disc, 108 templates. Params: log M_BH (7.4-9.8), log mdot (-4 to 2.49), E(B-V).
- KD18 (Kubota & Done 2018): AGNSED with outer disc + warm Comptonization + hot corona. Params: log M_BH (6-10), log mdot (-1.5 to 0), photon index Gamma.
- THB21 (Temple et al. 2021): Empirical composite including broad + narrow emission lines + blended line emission. 2-3 params.

**Torus models (4 options):**
- S04 (Silva et al. 2004): Semi-empirical smooth dust. 2 params: normalization, log N_H (21-25).
- NK08 (Nenkova et al. 2008): Clumpy Gaussian polar / power-law radial clouds. 2-4 params.
- SKIRTOR (Stalevski et al. 2016): Two-phase clumpy torus. 2-4 params: inclination (0-90), opening angle (10-80), tau (3-300).
- CAT3D-Wind (Honig & Kishimoto 2017): Clumpy disc + outflowing wind. 2-4 params: radial power-law index, wind fraction.

**X-ray:** alpha_ox--L_2500A correlation, photon index Gamma.

**Radio:** Single or double power-law + transition frequency.

**Emission lines:** Included via THB21 disc model (empirical broad + narrow + blended lines).

**Total parameter space:** Up to 19 free parameters depending on component selection.

**Key strengths:**
- Most flexible AGN code: 4 disc x 4 torus combinations = 16 model configurations.
- Covers X-ray to radio.
- Physically motivated BH parameters in SN12 and KD18.
- Modern torus options including wind model.

**Key weaknesses:**
- Combinatorial complexity can lead to overfitting.
- MCMC inference is slow for high-dimensional parameter space.
- No differentiable/gradient-based inference.
- Emission lines only available via one specific disc model (THB21).

---

### 8. qsogen

**Disc model:** Empirical broken power-law continuum with two slopes (plslp1, plslp2) joined at a break wavelength (plbrk1). Additional steep slope below Ly-alpha. Hot dust modeled as single blackbody (temperature tbb, normalization bbnorm). Not a physical disc model.

**Torus model:** Hot dust blackbody component only. Not a physical torus model.

**Emission lines:** Yes, sophisticated treatment:
- 4 empirical emission-line templates (median, peaky/high-EW, windy/blueshifted, narrow-line).
- emline_type parameter interpolates between templates (-2 to +3).
- Baldwin effect: beslope parameter scales EW with luminosity.
- Separate scaling for H-alpha, Ly-alpha, and narrow lines.

**Host galaxy:** S0 template from SWIRE, fraction controlled by fragal parameter.

**Additional components:** IGM absorption, dust reddening (empirical quasar extinction curve, ebv parameter).

**Total parameters:** ~10-15 (continuum slopes, BB temperature, emission-line type/scaling, dust, galaxy fraction).

**Key strengths:**
- Empirically calibrated to real quasar populations.
- Sophisticated emission-line treatment with Baldwin effect.
- Excellent for generating realistic quasar colors/magnitudes.
- Fast forward model.

**Key weaknesses:**
- Purely empirical: no physical disc/torus model. Cannot constrain BH mass or accretion rate.
- Type 1 quasars only (no obscured AGN).
- Forward model only, not a fitter.
- No radiative transfer.

---

## Feature Matrix (Quick Reference)

| Capability | tengri | CIGALE | Prospector | ProSpect | Bagpipes | Synthesizer | AGNfitter-rx | qsogen |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Accretion disc | Y (3 models) | Y (parametric) | Y (template) | Y (template) | N | Y (3 models) | Y (4 models) | Y (empirical) |
| Physical disc (SS73/KD18) | Y | N | N | N | N | Y | Y (KD18) | N |
| Smooth torus | Y | Y (Fritz) | N | Y (Fritz) | N | Y | Y (S04) | N |
| Clumpy torus | N | Y (SKIRTOR) | Y (CLUMPY) | N | N | N | Y (NK08, SKIRTOR, CAT3D) | N |
| Torus RT templates | N | Y | Y | Y | N | N | Y | N |
| NLR emission | Y (analytic) | N | N | N | N | Y (CLOUDY) | Partial | Y (empirical) |
| BLR emission | Y (analytic) | N | N | N | N | Y (CLOUDY) | Partial | Y (empirical) |
| Geometric masking | Y (sigmoid) | Y (viewing angle) | N | Y (psy) | N | Y (inclination) | N | N |
| Polar dust | N | Y (SKIRTOR) | N | N | N | N | N | N |
| BH mass param | Y | N | N | N | N | Y | Y (SN12, KD18) | N |
| BH spin param | Y | N | N | N | N | Y (relqso) | N | N |
| X-ray module | Y | Y (X-CIGALE) | N | N | N | N | Y | N |
| Radio module | Y | Y | N | N | N | N | Y | N |
| Differentiable | Y (JAX) | N | N | N | N | N | N | N |
| Built-in fitter | Y | Y | Y | Y | Y | N | Y | N |
| Baldwin effect | N | N | N | N | N | N | N | Y |

---

## Key Takeaways

1. **tengri is unique in being fully differentiable (JAX).** No other SED fitting code offers gradient-based inference for AGN parameters. This enables HMC/NUTS and variational inference, which scale far better than MCMC for the 9-15 parameter AGN models.

2. **CIGALE and AGNfitter-rx have the most comprehensive torus models**, with multiple RT template libraries. CIGALE's SKIRTOR with polar dust is the current standard for IR AGN SED fitting.

3. **Prospector's AGN is deliberately minimal** (2 params), designed as a nuisance component for galaxy SED fitting rather than detailed AGN characterization.

4. **Synthesizer's UnifiedAGN is the closest analog to tengri's unified_nlr_blr**, with CLOUDY-based NLR/BLR. However, Synthesizer is a forward model only (no built-in fitter).

5. **Emission lines are a major differentiator.** Only tengri, Synthesizer, and qsogen include AGN emission lines. CIGALE, Prospector, ProSpect, and Bagpipes all lack AGN emission lines entirely. This matters for rest-frame optical/UV fitting where BLR lines contribute significantly to broadband photometry.

6. **The main weakness of tengri's AGN relative to template-based codes** is the analytic (rather than RT) torus model. The two-temperature modified blackbody cannot reproduce the detailed silicate feature profiles that SKIRTOR/CLUMPY templates can. However, for broadband photometry fitting (as opposed to MIR spectroscopy), this is a minor limitation.

7. **BH physics (mass, spin, accretion rate) as model parameters** are available only in tengri, Synthesizer, and AGNfitter-rx (KD18/SN12 models). This enables physical constraints on the central engine rather than just phenomenological AGN fractions.

---

## References

- Carnall et al. 2018, MNRAS, 480, 4379 (Bagpipes)
- Calistro Rivera et al. 2016, ApJ, 833, 98 (AGNfitter)
- Conroy & Gunn 2010, ApJ, 712, 833 (FSPS)
- Fritz et al. 2006, MNRAS, 366, 767 (smooth torus RT)
- Hagen & Done 2023 (relqso, GR ray-traced disc)
- Honig & Kishimoto 2017, ApJL, 838, L20 (CAT3D-Wind)
- Johnson et al. 2021, ApJS, 254, 22 (Prospector)
- Kubota & Done 2018, MNRAS, 480, 1247 (AGNSED disc model)
- Leja et al. 2018, ApJ, 854, 62 (Prospector AGN)
- Nenkova et al. 2008, ApJ, 685, 147 (CLUMPY torus)
- Robotham et al. 2020, MNRAS, 495, 905 (ProSpect)
- Stalevski et al. 2012, MNRAS, 420, 2756; 2016, MNRAS, 458, 2288 (SKIRTOR)
- Temple et al. 2021, MNRAS, 508, 737 (qsogen)
- Vijarnwannaluk et al. 2022 (NLR covering fractions)
- Wilkins et al. in prep (Synthesizer UnifiedAGN)
- Yang et al. 2020, MNRAS, 491, 740 (X-CIGALE)
- Zhuang et al. 2024, arXiv:2405.12111 (AGNfitter-rx)

---

## Sources Consulted

- [CIGALE Fritz2006 source](https://github.com/JohannesBuchner/cigale/blob/master/pcigale/sed_modules/fritz2006.py)
- [CIGALE SKIRTOR database builder](https://gitlab.lam.fr/cigale/cigale/-/tree/master/database_builder/skirtor2016)
- [X-CIGALE paper (Yang et al. 2020)](https://arxiv.org/abs/2001.08263)
- [Prospector documentation](https://prospect.readthedocs.io/en/latest/advanced.html)
- [Python-FSPS StellarPop API](https://dfm.io/python-fsps/current/stellarpop_api/)
- [ProSpect GitHub](https://github.com/asgr/ProSpect)
- [ProSpect paper (Robotham et al. 2020)](https://arxiv.org/abs/2002.06980)
- [Bagpipes documentation](https://bagpipes.readthedocs.io/)
- [Bagpipes GitHub](https://github.com/ACCarnall/bagpipes)
- [Synthesizer paper (2025)](https://arxiv.org/html/2508.03888)
- [AGNfitter-rx paper](https://arxiv.org/html/2405.12111v1)
- [AGNfitter GitHub](https://github.com/GabrielaCR/AGNfitter)
- [qsogen GitHub](https://github.com/MJTemple/qsogen)
- [qsogen config.py](https://github.com/MJTemple/qsogen/blob/main/config.py)
- [SKIRTOR SED library](https://sites.google.com/site/skirtorus/sed-library)
- [SKIRTOR model description](https://sites.google.com/site/skirtorus/model)
- [CIGALE low-luminosity AGN module](https://www.aanda.org/articles/aa/full_html/2024/12/aa50510-24/aa50510-24.html)

---

## AGNfitter master + rX — audit and port targets

**Audit date:** 2026-04-22. Scope: `GabrielaCR/AGNfitter` master (Calistro
Rivera et al. 2016) and branch `AGNfitter-rX_v0.1` (Zhuang / Martínez-Ramírez
et al. 2024, arXiv:2405.12111). Both branches were read from a discardable
`/tmp` scratch clone; pickle binaries were inspected via `pickletools.dis`
only (no `pickle.load` on untrusted data).

### Finding 1 — Template pickles are identical across branches

The entire `models/{TORUS,BBB}/*.pickle` set exists unchanged on both
master and rX. The branch split lives entirely in `functions/MODEL_AGNfitter.py`
(which pickles are exposed) and `PRIORS_AGNfitter.py` (which informative
priors are enabled). The torus / disc SED libraries ported to tengri are
therefore indistinguishable between the two branches.

### Finding 2 — Grid shapes and parameter axes (from `MODEL_AGNfitter.py`)

| Pickle | Container | Axes | Notes |
|---|---|---|---|
| `TORUS/S04.pickle` | dict of lists | `Nh-values` | 1-parameter Silva+04; `SED` and `wavelength` are lists of per-Nh arrays |
| `TORUS/NK0_mean_{1,2,3}p.pickle` | pandas DataFrame | `incl`, +`oa`, +`tv` | Nenkova+08 CLUMPY averaged |
| `TORUS/SKIRTOR_mean_{1,2,3}p.pickle` | pandas DataFrame | `incl`, +`oa`, +`tv` | Stalevski+16 |
| `TORUS/CAT3D_mean_3p.pickle` | pandas DataFrame | `incl`, `a`, `fwd` | Hönig & Kishimoto 17 |
| `BBB/R06.pickle` | dict | (none; EBV free only) | Richards+06 composite, 1 SED |
| `BBB/SN12.pickle` | dict of arrays | `logBHmass-values`, `logEddra-values` | Slone & Netzer 12 |
| `BBB/KD18.pickle` | pandas DataFrame | `logBHmass`, `logEddra` | Kubota & Done 18 AGNSED |
| `BBB/KD18_warmInd.pickle` | pandas DataFrame | +`warmIndex` | 3-param KD18 variant |
| `BBB/THB21.pickle` | pandas object | (none; EBV free only) | Temple+21 composite |

All template SEDs are stored as `F_nu` vs `log10(nu)` in the rest frame.
Frequency grid convention: `16.685 ≈ log10(nu)` marks the 200 eV boundary
above which X-ray power-law replaces the BBB continuum; `4.83598e17 Hz` is
2 keV; X-ray cutoff at `7.254e19 Hz` (~300 keV) is hard-coded.

### Finding 3 — Component sum and normalization

AGNfitter's predicted total, per `PARAMETERSPACE_AGNfitter.ymodel`:

```
L_nu_total(nu, z) = 10^GA · GALAXY_reddened(nu, E(B-V)_gal)
                  + 10^SB · STARBURST(nu)
                  + 10^BB · [BBB_reddened(nu, E(B-V)_bbb) + XRAYS(...)]
                  + 10^TO · TORUS(nu)
                  + 10^RAD · AGN_RAD(nu)    (optional)
```

`GA, SB, BB, TO, RAD` are log-space multiplicative normalization free
parameters with hardcoded uniform priors on **[-10, +10]** (clamped to
`-9.8` when `turn_on_AGN=False`). The per-component template
re-normalization factors in `renorm_template` (`TO /= 1e-40`, `BB /= 1e60`,
`GA /= 1e18`, `SB /= 1e20`, `AGN_RAD *= 1e-30`) are **numerical conditioning
for emcee**, not physics — they are absorbed by the norm free params.
Takeaway: for tengri the Silva+04 / CAT3D grid values should be divided
by these conditioners at build time so the downstream `agn_torus_frac` /
`agn_log_lbol` normalization semantics match what AGNfitter fitters report.

### Finding 4 — Grid lookup is nearest-neighbor, not interpolated

In `DICTIONARIES_AGNfitter.get_model.pick_nD`, every `par_type == 'grid'`
parameter picks the closest grid value via `np.abs(grid - mcmc_value).argmin()`.
This introduces discontinuities that are invisible to emcee but break HMC /
NUTS. Tengri must interpolate (triweight / linear) in its ports — this is
the reason tengri's SKIRTOR grid is precomputed continuously in
`skirtor_precompute.py`.

### Finding 5 — X-ray α_ox–L_2500: verified parity, two tiny nuances

`MODEL_AGNfitter.XRAYS` implements Just et al. 2007:
`mean_alpha = -0.137 · log10(L_2500) + 2.638`, with a free `alphaScat`
adding scatter, and `Gamma` as a grid parameter for the X-ray power-law.
Directly compared term-by-term against tengri's
`components/xray/xray.py::alpha_ox_from_l2500` and
`xray_agn_corona_from_disc` on 2026-04-22:

| Aspect | AGNfitter | tengri | Diff |
|---|---|---|---|
| α_ox–L_2500 slope / intercept | `-0.137 · log10(L_2500) + 2.638` | identical coefficients (line 186) | **none** |
| α_ox → L_2keV constant | `0.3838` | `0.384` | ~0.05 %, both in use in the literature |
| Photon index default Γ | 1.8 | 1.8 | **none** |
| High-E cutoff | `exp(-ν / 7.254e19 Hz)` ≈ 300 keV | `E_cut = 300 keV` | **none** |
| Scatter parameter | `alphaScat` additive | `delta_alpha_ox` additive | equivalent |
| Low-E extent | starts at 10^16.685 Hz (200 eV) | `λ < 124 Å` (down to 100 eV) | tengri extends lower — superset |
| Extras | — | α_IRX-based corona (Lopez+24), XRB SFR/M*-scaled | tengri-only |

**Status: verified parity.** The 0.384 vs 0.3838 constant is a literature
rounding difference (< 0.05 % on the derived L_2keV); tengri's additional
low-E extent and IRX / XRB modules are strict supersets. No port needed.

### Finding 6 — Informative priors as opt-in penalty terms

`PRIORS_AGNfitter.PRIORS` adds additional log-prior penalties when
`modelsettings['PRIOR_*'] == True`:
- `PRIOR_energy_balance` — galaxy attenuated luminosity must match
  starburst emitted IR luminosity (Flexible vs Restrictive tolerance).
- `PRIOR_AGNfraction` — AGN fraction in 1–30 μm constrained above some
  empirical floor.
- `PRIOR_midIR_UV` — mid-IR torus vs UV disc luminosity tie.
- `PRIOR_radio` / `prior_UV_xrays` — FIR-radio correlation and
  X-UV correlation.

All of these are straightforward Gaussian or clipped-Gaussian prior
additions. Tengri's `parameters/priors.py` could adopt any subset as
optional flags; current tengri design already handles the same couplings
through the unified parameter space, but exposing them as named
informative priors is low-cost and would aid users coming from AGNfitter.

### Finding 7 — Reddening laws

- AGN disc: `BBBred_Prevot` (Prevot SMC, `R_V = 2.72`,
  `k(λ) = 1.39 (λ_μm/1e-4)^-1.2 - 0.38`).
- Galaxy: `GALAXYred_Calzetti` (`R_V = 4.05`) and `GALAXYred_CharlotFall`
  (`R_V = 5.9`).

Tengri's `components/dust/` has Calzetti and Charlot&Fall already. **SMC
Prevot for the AGN disc is a small gap** — worth adding a Prevot branch to
`components/dust/` in a follow-up, not a full port.

### Port targets (concrete, for Phases 2–4)

| Phase | Component | Gap? | Port strategy |
|---|---|---|---|
| 2 | **Silva+04 smooth torus** (1 param: log N_H) | Yes — unique to AGNfitter | dict-of-lists pickle is the simplest to extract; grid-build script loads it once (with user-approved `pickle.load` in a quarantined venv) and writes `data/silva04_torus_grid.h5`. JAX component reuses `skirtor_precompute.py` pattern (linear/triweight interpolation over log N_H). |
| 3 | **CAT3D-Wind** (3 params: incl, `a`, `fwd`) | Yes — unique vs tengri | Same pattern; pickle is a pandas DataFrame so build script needs pandas. |
| 4 | **THB21** (no new component) | Partial — tengri has qsogen/Temple+21 | Audit-only: compare tengri's `components/agn/qsogen.py` continuum + emission-line EWs vs AGNfitter's THB21 pickle at matched redshift. No new file expected unless numerics diverge > 5 %. |
| (optional) | SMC Prevot reddening for disc | Small | Add a Prevot branch in `components/dust/` alongside Calzetti / CF00. |
| (optional) | Informative priors (energy balance, AGN fraction, mid-IR–UV tie) | Soft | Add as optional penalty hooks in `parameters/priors.py`. |

### Not porting (parity or by design)

- X-ray α_ox–L_2500, Γ, alphaScat — **already implemented** in tengri's
  `components/xray/xray.py`.
- Radio single/double-power-law + blazar cutoff — tengri's `radio.py` is
  a continuous, differentiable superset.
- KD18 / SN12 / SKIRTOR / NK08 discs and tori — tengri has differentiable
  analytic or precomputed equivalents.
- emcee / ultranest inference — tengri's HMC / NUTS / geoVI stack is
  gradient-based and does not admit nearest-neighbor grid lookup.
- BC03 galaxy templates, DH02/CE01 starburst — tengri uses DSPS + DL07.
- `corner.py`, `triangle.py`, `PLOTandWRITE_AGNfitter.py` — tengri has
  its own plotting stack.

### Finding 8 — Parity verification of other AGN components (term-by-term)

Checked on 2026-04-22 by reading AGNfitter `MODEL_AGNfitter.py` against the
named tengri modules. Not taken on faith from earlier iterations of this doc.

**Radio (`AGN_RAD` vs `components/radio/radio.py::radio_agn_dpl`):**
AGNfitter line 523:
``agnrad_Fnu = (nu/nu_t)**alpha1 · (1 − exp(−(nu_t/nu)**(alpha1−alpha2))) · exp(−nu/nu_cut)``.
Tengri's `_dpl_shape` is term-by-term identical (same break, same turnover,
same exponential aging). AGNfitter hand-tunes grids of (`alpha1`, `alpha2`,
`nu_t`, `E_cutoff`); tengri exposes these continuously and
differentiably. **Status: verified parity; tengri is a continuous
superset.** The single-PL (`nRADdata==1`) and blazar-cutoff (`SPL`)
branches reduce to specializations of `radio_agn` / `radio_agn_dpl` with
defaults pinned. No port needed.

**qsogen (Temple+21) vs `BBB/THB21.pickle`:**
AGNfitter's THB21 path is just one pre-tabulated composite SED from the
Temple+21 paper:
``bbb_nu, bbb_Fnu = THB21dict['nu'].values.item(), THB21dict['SED'].values.item()``,
with `EBVbbb` as the only continuous parameter (plus optional α_ox scatter
and X-ray Γ). Tengri's `components/agn/qsogen.py` reimplements the full
Temple+21 parametric model — 7 continuous parameters (``plslp1``,
``plslp2``, ``plbrk``, ``tbb``, ``bbnorm``, ``emline``, ``ebv``) plus the
Baldwin effect (`slope = 0.183`) and 4 emission-line templates
(median / peaky / windy / narrow). **Status: tengri is a strict superset
of AGNfitter's THB21.** No port needed; the Phase-4 audit envisaged in the
original plan is closed in tengri's favor.

**KD18 disc (`BBB/KD18.pickle` vs `components/agn/disc.py::kubota_done_disc`):**
AGNfitter stores a pre-tabulated AGNSED run on a (`logBHmass`, `logEddra`)
grid and does nearest-neighbor lookup (see Finding 4). Tengri integrates
the 3-zone Kubota & Done (2018) disc analytically from SS73 + GR ISCO +
warm Comptonization on every forward call. Both compute the same physical
model; numerical agreement depends on shared assumptions about the hard-
corona Γ, `kt_warm`, `kt_hot`, and `r_warm_ratio`. **Status: same model,
different numerics.** A real cross-validation would fit a tengri SEDModel
to a synthetic SED generated from AGNfitter's KD18.pickle and check the
recovered (`logBHmass`, `logEddra`) — not done here; recorded as an open
cross-val task for `tests/crossval/`.

**SKIRTOR (`TORUS/SKIRTOR_mean_{1,2,3}P.pickle` vs
`components/agn/skirtor.py` + `data/skirtor_templates*.h5`):**
File names on AGNfitter are literally `SKIRTOR_mean_*P.pickle` — these are
*per-axis-averaged* SEDs, not the full Stalevski+16 library. Tengri's
`skirtor.py` consumes the full SKIRTOR grid (separately downloaded from
the Stalevski site via `scripts/download_skirtor_templates.py`), preserving
the full 5D (`tau`, `p`, `q`, `oa`, `cos_inc`) resolution with triweight
interpolation. **Status: tengri uses a richer grid; not numerically
equivalent at intermediate parameter values.** AGNfitter's 1/2/3-parameter
flattened libraries are strict projections of the full grid. A future
regression test could confirm that tengri's full-grid values, when
averaged over the same axes AGNfitter averaged out, recover the
`_mean_NP.pickle` values. Filed as a cross-val task, not a port.

**Gaps confirmed as real (all are Phase-2+ port targets):**

- **Silva+04 smooth torus** — no tengri analog. **Ported in Phase 2**
  (`components/agn/silva04.py` + `data/silva04_torus_grid.h5` via
  `scripts/build_silva04_grid.py`). Verified via
  `tests/unit/components/agn/test_silva04.py` (8/8 pass).
- **CAT3D-Wind torus** — no tengri analog. Phase 3 target.
- **Nenkova+08 CLUMPY torus** (`NK0_mean_*P.pickle`) — no tengri analog
  by design (non-differentiable template stack). Do not port unless the
  user explicitly revises the design philosophy.
- **SN12 α-disc** — tengri's SS73 `standard` disc covers the same physics
  analytically; do not port the tabulated library.

### Citation verification TODO

Before any of Silva+04, CAT3D-Wind or Temple+21 text is written into
tengri source docstrings, the corresponding `.tex` / `.bib` entries in
*(private paper draft)* must be cross-checked. Specifically:

- Silva, Maiolino & Granato 2004 — verify journal, volume, DOI, arXiv ID.
- Hönig & Kishimoto 2017 — verify CAT3D-Wind paper reference (ApJL 838,
  L20) and model parameter definitions (`a`, `fwd` meaning).
- Just et al. 2007 — confirm exact coefficients of α_ox–L_2500 (currently
  `-0.137 ... + 2.638` in AGNfitter; tengri's `xray.py` should agree).
- Temple et al. 2021 — confirm emission-line template structure for the
  Phase 4 audit.
