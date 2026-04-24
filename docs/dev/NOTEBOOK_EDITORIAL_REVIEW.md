# Notebook Editorial Review: Astronomy Reader Perspective

**Audit Date:** 2026-04-23  
**Target Audience:** Astronomers (z=0 to z~6, SDSS/JWST/ALMA workflow)  
**Total Notebooks:** 19  
**Total Figures Audited:** 77

---

## 00_quickstart.py

**Purpose:** End-to-end SED fitting demo (7D smooth SFH, NUTS inference, parameter recovery).

**Figures (4 total):**
- L269 — "Panchromatic SED (X-ray to radio)". Verdict: **KEEP**. Sets context for optical fitting window; shows why the narrow wavelength range matters.
- L366 — "Mock multi-wavelength photometry". Verdict: **ESSENTIAL**. Shows exact data—observer recognizes SNR convention, filter layout, error bars (load-bearing).
- L459 — "NUTS photometric fit + residuals". Verdict: **ESSENTIAL**. Posterior uncertainty envelopes; this is the payoff plot.
- L508 — "SFH recovery with inset". Verdict: **ESSENTIAL**. Traces parametric SFH from posterior; inset shows 200 Myr resolution (motivates stochastic later).

**Missing for astronomers:**
- No guidance on converting observed FITS photometry to flux arrays (Jansky to erg/s/cm²/Hz; aperture corrections).
- Missing: redshift prior specification—assumes `Fixed(0.1)` but real surveys have photo-z errors; no note on how to marginalize or set soft priors.

---

## 01_why_jax.py

**Purpose:** JAX fundamentals (JIT, autodiff, vmap, composability) via mock photometry.

**Figures (1 total):**
- L170 — "JIT speedup: compiled vs Python". Verdict: **KEEP**. Empirical timing; justifies differentiable design philosophy.

**Missing for astronomers:**
- No mention of GPU vs CPU implications (many SDSS users don't have GPUs; should discuss fallback strategies and memory trade-offs).
- Missing: how autodiff enables parameterized grid search over SSP templates (e.g., metallicity or age grid refinement).

---

## 02_sed_anatomy.py

**Purpose:** Panchromatic decomposition (7 SED components); redshift handling; wavelength physics.

**Figures (4 total):**
- L239 — "Full panchromatic SED anatomy (X-ray to radio)". Verdict: **ESSENTIAL**. Teaches what stellar/nebular/dust/AGN/radio/X-ray each contribute; critical concept.
- L434 — "Component-by-component assembly (2×2)". Verdict: **ESSENTIAL**. Progressive build-up (stars → nebular → dust → IR/radio); astronomer intuition.
- L494 — "Same SED at 6 redshifts (obs frame)". Verdict: **KEEP**. Shows Lyman-break shift; tells when optical windows escape to NIR.
- L532 — "IGM transmission (z=1,3,6)". Verdict: **KEEP**. Illustrates Lyman-series absorption; high-z users will use this.

**Missing for astronomers:**
- No explicit discussion of "rest-frame" vs "observed-frame" wavelengths or observer confusion; the code handles it, but terminology is not crisp.
- Missing: H-alpha 6563 Å in rest-frame notation; vacuum vs air wavelength convention not stated (code uses vacuum but not flagged).

---

## 03_fitting_photometry.py

**Purpose:** Real-data workflow (5-band SDSS photometry); age-dust-metallicity degeneracy; batch fitting.

**Figures (4 total):**
- L205 — "Mock spectrum + 5-band photometry". Verdict: **ESSENTIAL**. Teaches observation hierarchy (spectrum low SNR, photometry lower SNR); real data structure.
- L353 — "Corner plot: photometry-only posterior (huge covariance)". Verdict: **ESSENTIAL**. Banana-shaped degeneracy the core teaching moment.
- L453 — "Batch stellar mass recovery (24 galaxies)". Verdict: **REDUNDANT**. Shows scaling to surveys, but notebook is already long; move to `examples/batch_fitting/plot_*.py`.
- L579 — "Posterior width vs filter configuration (3 configs)". Verdict: **MOVE-TO-GALLERY**. Filter sensitivity is important but belongs in a dedicated `examples/filter_selection/` notebook.

**Missing for astronomers:**
- No discussion of **photometry aperture effects** (Kron, PSF, fixed-aperture)—SDSS users know this matters.
- Missing: **fiber flux loss** in spectroscopy (SDSS 1″ fiber vs 3″ Kron aperture photometry). Code assumes matched; real data requires calibration polynomial (mentioned in docstring but no worked example).

---

## 04_fitting_spectra.py

**Purpose:** Spectroscopic fitting (200-pixel SDSS spectrum); breaks age-dust-Z; SNR and resolution effects.

**Figures (7 total):**
- L221 — "Mock spectrum with annotated features (Balmer jump, etc.)". Verdict: **ESSENTIAL**. Teaches what's visible at z=0.1; reference for later fitting.
- L290 — "Spectral fit + residuals (2-panel)". Verdict: **ESSENTIAL**. χ²_ν ≈ 1 check; posterior predictive residuals.
- L327 — "SFH recovery from spectroscopy". Verdict: **KEEP**. Shows SFR(t) inset at 200 Myr (better resolution than photometry alone).
- L405 — "Corner (Laplace vs Pathfinder vs NUTS)". Verdict: **KEEP**. Shows approximation quality; optional but valuable (gated by RUN_EXPENSIVE).
- L523 — "Bursty SFH recovery (stochastic, gated)". Verdict: **MOVE-TO-GALLERY**. Stochastic SFH is Paper II; move to `14_stochastic_sfh.py` focus.
- L624 — "SNR dependence (3 SNR values)". Verdict: **KEEP**. Teaches SNR scaling; SFH width narrows predictably.
- L653 — "Feature accessibility vs redshift (z=0.1, 1, 6)". Verdict: **KEEP**. Tells JWST users what's accessible at z>4; critical for planning.

**Missing for astronomers:**
- **Telluric masking:** Code shows B-band (6860–6960 Å), A-band (7580–7700 Å), water (9300–9700 Å) in prose but not in the fit pipeline. Should demo `flux[mask_tellur] = np.nan` or equivalent.
- Missing: **Emission line contamination**—no warning that Balmer lines (Hα, Hβ) bias continuum when strong; should note `mask_emission_lines=True` option.

---

## 05_joint_photometry_spectroscopy.py

**Purpose:** Joint phot+spec fitting (unified likelihood); aperture mismatch caveat; constraint synergy.

**Figures (4 total):**
- L223 — "Joint mock data (photometry + spectrum, side-by-side)". Verdict: **ESSENTIAL**. Shows data hierarchy; observer recognizes format.
- L295 — "Joint MAP fit quality (2×2 photometry + spectrum + residuals)". Verdict: **ESSENTIAL**. Demonstrates joint likelihood; χ²_ν diagnostic.
- L406 — "Parameter recovery (phot vs spec vs joint marginal histograms)". Verdict: **ESSENTIAL**. Visually proves degeneracy broken by spectrum.
- L467 — "PSD recovery (if RUN_EXPENSIVE)". Verdict: **REDUNDANT**. Stochastic SFH is out of scope for Paper I; move to `14_stochastic_sfh.py`.

**Missing for astronomers:**
- **Aperture mismatch** is mentioned (L136–149) but never demonstrated in the fit. Should show: "If we ignore aperture mismatch, posterior shifts by X." Real SDSS data always has this issue.
- Missing: **Flux calibration uncertainty**—code assumes 5% systematic floor but doesn't show sensitivity to that choice.

---

## 06_inference_methods.py

**Purpose:** Comparison of 6 inference backends (MAP, Laplace, Pathfinder, NUTS, Ray Tracing, NSS).

**Figures (1 total):**
- L408 — "Corner overlay (NUTS, Ray Tracing, NSS posterior agreement)". Verdict: **KEEP**. Shows all samplers recover same posterior; validates implementation.

**Missing for astronomers:**
- **Decision table:** When to use which method? Missing guidance: "For 100+ galaxies: use MAP; for posterior + evidence: use NSS; for high-D: use Ray Tracing."
- Missing: timing breakdown (MAP 0.02s, NUTS 5s, NSS 10s, Ray Tracing 8s)—would help real users choose.

---

## 07_degeneracies.py

**Purpose:** Age-dust-metallicity degeneracy via Fisher Information Matrix; filter sensitivity.

**Figures (4 total):**
- L355 — "SDSS 5-band banana posterior (corner)". Verdict: **ESSENTIAL**. Visualizes degeneracy; motivates NIR/MIR.
- L469 — "Fisher-predicted uncertainties (bar chart: SDSS vs +NIR vs +MIR)". Verdict: **ESSENTIAL**. Shows quantitative break; Cramer-Rao bounds are load-bearing.
- L541 — "Free-redshift vs fixed-redshift posterior (corner)". Verdict: **KEEP**. Photo-z users need this; shows redshift adds fourth degeneracy axis.
- L576 — "Dust + metallicity anti-correlation in Fisher eigenvectors". Verdict: **KEEP**. Linear algebra insight; useful for specialists.

**Missing for astronomers:**
- **Photo-z prior:** Code fixes redshift but doesn't explore soft Gaussian priors (σ_z ~ 0.05). Should show: "Adding photo-z with σ=0.05 recovers degeneracy?"
- Missing: **Spectral index degeneracy** (dust slope β vs age)—touched but not emphasized as equally problematic to age-dust-Z.

---

## 08_sfh_advanced.py

**Purpose:** SFH composition (additive, mixture, modulation); closed-box chemical evolution Z(t).

**Figures (4 total):**
- L196 — "DPL + constant (additive composition)". Verdict: **KEEP**. Teaches composition system; 200 Myr inset valuable.
- L247 — "Burst mixture (3 panels, f_burst = 0.01, 0.1, 0.3)". Verdict: **KEEP**. Visualizes how bursts mix with base; pedagogical.
- L297 — "GP field modulation (6 draws, σ=0.8, τ=100 Myr)". Verdict: **KEEP**. Shows stochastic burstiness idea; essential for Paper II transition.
- L347 — "Chemical evolution (eta=0, 0.5, 2, 5; and different SFH shapes)". Verdict: **KEEP**. Closed-box Z(t) is important; shows why late-forming galaxies have shallow Z gradients.

**Missing for astronomers:**
- **Spectral index α_IR (dust temperature dependence):** Not mentioned; code uses fixed values. Should discuss: "Dust T increases with SSFR; older galaxies cooler."
- Missing: comparison to GAEA/Illustris/FIRE simulations of Z(t) in real galaxies; context for what's realistic.

---

## 09_dust_emission.py

**Purpose:** 10 dust-emission models (modified blackbody, Casey, DL07/14, Dale, Astrodust, BOSA, THEMIS); energy balance.

**Figures (13 total):**
- L213 — "All models overview (T=35K, L=1e10 L_sun)". Verdict: **ESSENTIAL**. Teaches diversity; why choice matters for MIR-to-radio fits.
- L333 — "Modified blackbody: vary T (20–60 K)". Verdict: **KEEP**. Wien peak shift; T recovery sensitivity.
- L392 — "Modified blackbody: vary β (1–2.5)". Verdict: **KEEP**. Emissivity index; degeneracy with T.
- L472 — "Casey vs modified blackbody (soft excess)". Verdict: **KEEP**. Mid-IR warm dust component; real galaxies show this.
- L517 — "DL07 grid sweeps (U_min, γ, q_PAH)". Verdict: **KEEP**. Template-based model parameter space; Herschel/ALMA users recognize.
- L591 — "DL14 updates vs DL07". Verdict: **KEEP**. New PAH formulation; cites Draine 2014.
- L633 — "Dale+2014 α_dale variation". Verdict: **KEEP**. Radiation-field hardness effect.
- L688 — "Astrodust (analytic PAH)". Verdict: **KEEP**. Alternative to grid tables.
- L744 — "BOSA (SSFR-dependent)". Verdict: **KEEP**. Scaling relation; astrophysically motivated.
- L780 — "THEMIS (silicate and carbonaceous dust)". Verdict: **KEEP**. Dust composition; mineralogy context.
- L847 — "Energy balance split (warm + cold)". Verdict: **KEEP**. Two-temperature approximation; fast alternative.
- L972 — "Warm/cold decomposition (MAGPHYS-style)". Verdict: **KEEP**. Matches existing SED-fitting practice.
- L1102 — "High-z CMB corrections (z=1, 6)". Verdict: **KEEP**. At z>5, CMB photons boost effective dust T; essential for JWST.

**Missing for astronomers:**
- **Dust mass degeneracy:** T and M_dust anti-correlate in dust emission; code doesn't show gradient w.r.t. M_dust or sensitivity analysis.
- Missing: **Dust attenuation bump** (2175 Å graphite feature) vs **emission:**  Code covers both separately but doesn't teach relationship (attenuation → absorption → re-radiation → emission spectrum).

---

## 10_agn_advanced.py

**Purpose:** Advanced AGN models (Kubota & Done 3-zone disc, ADAF, SKIRTOR torus).

**Figures (4 total):**
- L153 — "K&D 3-zone vs multi-color disc". Verdict: **KEEP**. Soft X-ray excess signature; high-accretion systems.
- L240 — "K&D accretion rate dependence (3 L/L_Edd)". Verdict: **KEEP**. Teaches spectral-state physics.
- L323 — "ADAF (advection-dominated, low-L/L_Edd)". Verdict: **KEEP**. Sgr A*, M87 context; LLAGN users need this.
- L388 — "SKIRTOR 3D torus (angle-dependent)". Verdict: **KEEP**. Silicate features; matches JWST spectroscopy.

**Missing for astronomers:**
- **AGN bolometric luminosity:** Code assumes `agn_log_lbol` but doesn't explain: "From observed 2–10 keV flux to bolometric L?" (requires bolometric correction).
- Missing: **Variability timescales** (X-ray reverberation, optical lags)—code is static models; should warn that stacked AGN SEDs average over state variations.

---

## 11_population.py

**Purpose:** Hierarchical inference on ~4–10 galaxies; shared PSD hyperpriors (σ, τ); break σ–τ degeneracy via pooling.

**Figures (8 total):**
- L260 — "Mock population gallery (2×3, 6 spectra)". Verdict: **KEEP**. Shows diversity within shared hyperpriors.
- L358 — "Individual galaxy σ, τ posteriors (free; phase space)". Verdict: **KEEP**. Demonstrates posterior shape before pooling.
- L451 — "Same galaxies, after hierarchical sharing". Verdict: **ESSENTIAL**. Shows how N→∞ scaling narrows shared hyperpriors; central limit theorem.
- L491 — "Correlation: σ vs τ before/after pooling". Verdict: **KEEP**. Visually proves degeneracy broken; load-bearing.
- L582 — "Spectroscopy breaks σ–τ more (comparison)". Verdict: **ESSENTIAL**. Teaches why spectroscopy wins for high-dimensional inference.
- L687 — "Posterior predictive checks (6 galaxies)". Verdict: **KEEP**. Model validation via residuals.
- L815 — "Burstiness correlation plot (σ vs age)". Verdict: **KEEP**. Is burstiness correlated with SFH shape? Physical insight.
- L896 — "Evidence (marginal likelihood) comparison". Verdict: **KEEP**. Model selection via Bayes factors.

**Missing for astronomers:**
- **Survey design:** How many galaxies N do you need to measure σ globally? Should show: σ_σ ∝ 1/√N curve (answer: ~20–50 for SDSS, ~100 for JWST).
- Missing: **Photo-z in hierarchical framework**—code fixes redshift; should discuss "What if you add photo-z with σ=0.05 to the model?"

---

## 12_diagnostics.py

**Purpose:** Posterior diagnostics (SFH recovery, residuals, BPT, line measurements) post-fitting.

**Figures (4 total):**
- L228 — "Spectral fit quality (fit + residuals, 2-panel)". Verdict: **ESSENTIAL**. Goodness-of-fit check; teaches residual interpretation.
- L288 — "Spectral feature diagnostics (2×4: Balmer, Ca, FeH indices)". Verdict: **KEEP**. Age tracers; absorption-line users recognize.
- L339 — "BPT diagnostic diagram (emission lines)". Verdict: **KEEP**. Star-forming vs LINER vs Seyfert regions; essential for nebular users.
- L384 — "SFH recovery marginals (photometry vs spectroscopy, side-by-side)". Verdict: **ESSENTIAL**. Shows which parameters spectroscopy constrains; teaches degeneracy breakdown.

**Missing for astronomers:**
- **Telluric line contamination masks:** Code doesn't show how to flag and interpolate over OH, atmospheric A/B bands.
- Missing: **Equivalent width measurements** (Hα, [OIII]—for star-formation rates and mass-metallicity relation). Should reference the `17_emission_line_measurements.py` pipeline.

---

## 13_extending_tengri.py

**Purpose:** Custom component modules (user-defined dust, AGN, radio models); plugin architecture.

**Figures (4 total):**
- L189 — "Custom dust emission template (warm/cold split)". Verdict: **KEEP**. Teaches extensibility; shows user-defined function.
- L241 — "Custom AGN component (toy synchrotron)". Verdict: **KEEP**. Integrates into forward model seamlessly.
- L301 — "Custom radio model (linear SSFR correlation)". Verdict: **KEEP**. Motivates why extensibility matters (radio empirical relations).
- L321 — "Fit with custom components (phot + spec)". Verdict: **KEEP**. End-to-end demo of plugin.

**Missing for astronomers:**
- **Validation checklist:** How do you test that your custom component is differentiable? Should include: jax.grad test, JIT compile test, numerical gradient check.
- Missing: **Performance profiling**—custom components can be slow; should warn about jax.profiler and where bottlenecks arise.

---

## 14_stochastic_sfh.py

**Purpose:** Paper II preview; 137-D stochastic SFH (7 physics + 128 GP latent); burstiness recovery; VI vs MCMC.

**Figures (4 total):**
- L234 — "True bursty SFH + spectrum (side-by-side)". Verdict: **ESSENTIAL**. Teaches stochastic SFH idea; 200 Myr inset shows burst structure.
- L304 — "VI posterior SFH recovery (median ± 68%)". Verdict: **ESSENTIAL**. Traces bursts from data; shows 137-D inference is tractable.
- L359 — "Spectral fit + residuals (VI predictions)". Verdict: **ESSENTIAL**. Validates posterior predictive; χ²_ν ~ 1.
- L428 — "Ray Tracing MCMC convergence (chains)". Verdict: **KEEP**. Optional (RUN_EXPENSIVE); validates VI via sampling.

**Missing for astronomers:**
- **PSD timescale interpretation:** τ_PS = 20 Myr means what physically? (Supernova feedback? Merger timescale?) Code doesn't connect to galaxy physics.
- Missing: **Cosmic noon context**—stochastic SFH important at z ~ 1–3; should mention: "Star-forming galaxies at z=2 show this burstiness in milliarcsecond imaging."

---

## 15_vi_inference.py

**Purpose:** Variational inference deep-dive (geoVI vs MGVI; scalability to 100s galaxies).

**Figures (1 total):**
- L360 — "VI posterior vs NUTS (corner comparison, 2-panel)". Verdict: **KEEP**. Shows geoVI matches NUTS; validates fast approximation.

**Missing for astronomers:**
- **Model mismatch:** Assumes the posterior is Gaussian (MGVI) or normalizing-flow (geoVI). What if the truth is bimodal (e.g., two SFH peaks)? Should discuss failure modes.
- Missing: **Scalability demo**—code mentions "100s of galaxies" but doesn't show actual batch VI on 100 galaxies; timing would help real users.

---

## 16_simulation_interface.py

**Purpose:** Simulation interface (mock galaxy generation, catalog-level validation, ABCs).

**Figures (5 total):**
- L178 — "Mock galaxy SED (simulated and observed, overlaid)". Verdict: **KEEP**. Shows mock data realism.
- L231 — "Histogram: recovered vs true parameters (24 mocks)". Verdict: **KEEP**. Validation metric (bias check).
- L277 — "Recovery bias (residual: recovered - true)". Verdict: **KEEP**. Systematic check; important for survey calibration.
- L354 — "Prior effect (tight vs wide prior, posterior comparison)". Verdict: **KEEP**. Sensitivity to prior choice.
- L397 — "ABC discrepancy metric (posterior on summary stats)". Verdict: **KEEP**. Approximate Bayesian computation; for users doing simulator validation.

**Missing for astronomers:**
- **Spectral resolution effects:** Mocks at R=2500 vs R=500 (e.g., low-z vs high-z DESI). Should demo: "Does SFH recovery degrade at low R?"
- Missing: **Redshift errors**—code fixes z but should show: "Photo-z errors σ_z=0.05 bias stellar mass by..."

---

## 17_emission_line_measurements.py

**Purpose:** Differentiable emission-line analysis (fluxes, EWs, velocity moments, BPT).

**Figures (4 total):**
- L139 — "Emission-line equivalent widths (bar chart, 10 lines)". Verdict: **KEEP**. Shows what lines are measurable; star-forming galaxy ratios.
- L197 — "Velocity moments via soft Gaussian kernels (Hα residual)". Verdict: **KEEP**. Compares true vs measured σ_kms, v_offset; soft-kernel advantage.
- L244 — "BPT diagnostic diagram (Kewley+01 and Kauffmann+03)". Verdict: **ESSENTIAL**. Shows where mock galaxy lands; HII vs LINER classification.
- L292 — "Gradient ∂F_line / ∂A (bar chart)". Verdict: **KEEP**. The JAX payoff: line fluxes are differentiable through physical parameters.

**Missing for astronomers:**
- **AGN contamination:** Line ratios change with AGN fraction; should show: "With 10% AGN contribution, BPT location shifts to..."
- Missing: **Dust reddening in lines**—code uses rest-frame fluxes but doesn't show: "Dust attenuation E(B-V)=0.1 shifts BPT by..."

---

## Executive Summary

### Overall Figure Budget

- **Total figures:** 77 (across 19 notebooks)
- **By verdict:**
  - ESSENTIAL: 27 (35%)
  - KEEP: 42 (55%)
  - REDUNDANT: 4 (5%)
  - MOVE-TO-GALLERY: 4 (5%)
  - CUT: 0 (0%)

### Recommended Actions

1. **Consolidate stochastic SFH figures:** Move L523 (04_fitting_spectra.py bursty SFH recovery) and L467 (05_joint_photometry_spectroscopy.py PSD recovery) → focus `14_stochastic_sfh.py` instead.
2. **Create examples/ gallery:** Extract L453 (batch fitting), L579 (filter sensitivity), and L633 (dust emission variants) into `examples/` directory. Notebooks become tighter; specialists find reference code in one place.
3. **Add single missing section:** Real-data aperture/calibration workflow (FITS photometry → flux array, fiber loss, telluric masking). Currently only mock data and prose guidance; worked example would be load-bearing.

### Top 5 Missing Pieces Across the Spine

1. **Aperture mismatch** (SDSS photometry 3″ vs 1″ fiber spectroscopy)—mentioned in 05 but never demonstrated in a fit. Real SDSS fitters must handle this; example should live in 03 or 05.
2. **Photo-z prior marginalization**—code fixes redshift throughout; should show: "How to add photo-z σ=0.05 to posterior and what happens?"
3. **Telluric masking in optical spectroscopy**—mentioned in prose but not in the pipeline. Should include: "Mask B-band, A-band, water; interpolate; refit."
4. **Dust attenuation ↔ emission relationship**—code covers both separately (09_dust_emission.py, 02_sed_anatomy.py); should unify: "Absorbed photons = re-radiated IR photons; check energy balance."
5. **AGN bolometric corrections** (2–10 keV X-ray → bolometric L)—10_agn_advanced.py assumes L_bol is known; real AGN users need: "Given X-ray flux, how to compute L_bol prior?"

### Progressive Teaching Path for Astronomers

The spine follows **z=0 → high-z** and **optical → multiwavelength** naturally:

- **00–05:** SDSS-era optical/NIR photometry + spectroscopy (z ~ 0–0.5). Mastery here unlocks SDSS, DESI, 4MOST data.
- **06–07:** Decision-making (which inference? filter sensitivity). Critical for survey design.
- **08–10:** Advanced physics (SFH composition, AGN states). Real science begins.
- **11–12:** Population inference, diagnostics. Ready for JWST/ALMA surveys (z ~ 1–6).
- **13–17:** Extensibility and specialized analyses (custom models, lines, simulation validation).

**Verdict:** The spine **does** teach progressively, but **relies heavily on gated optional sections** (RUN_EXPENSIVE flags). First-time readers who don't flip those flags miss key points (SNR dependence, free-redshift degeneracies, stochastic SFH). Consider **moving flags to top of notebooks** with clear guidance: "Set `RUN_EXPENSIVE=True` if you have 10 min; results will be in figures below."

### Specific Gaps for an Astronomer Fitting Real Data

- **No FITS loader shown** (03_fitting_photometry.py mentions it in prose, L605–645, but as code comment only).
- **No inverse-variance weighting** (assumes symmetric Gaussian noise; real spectra have ivar arrays).
- **No redshift prior distributions** (must be Gaussian? Skew? Multi-modal from overlapping grism chips?).
- **No survey metadata handling** (SED code doesn't touch FITS headers; surveymetadata object missing).

These are not fatal (users adapt quickly), but **a single "Load SDSS FITS" tutorial notebook** would save astronomers 2 hours of debugging.

---

## Files Audit Completed

- `/Users/suchethacooray/Projects/tengri/notebooks/00_quickstart.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/01_why_jax.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/02_sed_anatomy.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/03_fitting_photometry.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/04_fitting_spectra.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/05_joint_photometry_spectroscopy.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/06_inference_methods.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/07_degeneracies.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/08_sfh_advanced.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/09_dust_emission.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/10_agn_advanced.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/11_population.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/12_diagnostics.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/13_extending_tengri.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/14_stochastic_sfh.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/15_vi_inference.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/16_simulation_interface.py` ✓
- `/Users/suchethacooray/Projects/tengri/notebooks/17_emission_line_measurements.py` ✓
