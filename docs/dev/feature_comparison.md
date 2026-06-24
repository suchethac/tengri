# Feature Comparison: tengri vs External SED Fitting Codes

**Last updated:** 2026-04-07  
**Scope:** All physics features (stellar populations, SFH, dust, nebular, AGN, IGM, inference, ancillary) that external codes implement and tengri does not — or vice versa.  
**AGN-specific detail:** See [`AGN_MODEL_COMPARISON.md`](AGN_MODEL_COMPARISON.md).

---

## Quick-reference feature matrix

`Y` = supported, `N` = not supported, `~` = partial/limited.

| Feature | **tengri** | **bagpipes** | **FSPS/Prospector** | **pCIGALE** | **synthesizer** |
|---|:---:|:---:|:---:|:---:|:---:|
| **Stellar populations** | | | | | |
| DSPS (MIST tracks) | Y | N | N | N | N |
| BC03 SSP grid | N | Y | ~ | Y | Y |
| BPASS binary stars | N | N | ~ | Y | Y |
| Maraston+2011 | N | N | ~ | Y | N |
| E-MILES / XSL libraries | N | N | ~ | N | N |
| AGB circumstellar dust | N | N | Y | N | N |
| Wolf-Rayet spectral features | N | N | Y | N | N |
| Stellar remnants (WD, NS, BH) | N | N | Y | N | N |
| Hot bottom burning on AGB | N | N | Y | N | N |
| TP-AGB control parameter | N | N | Y | N | N |
| IMF switching (Salpeter/Chabrier/Kroupa) | N | N | Y | Y | Y |
| IMF as free parameter | N | N | N | N | N |
| **Star formation histories** | | | | | |
| Double power law (DPL) | Y | Y | N | Y | Y |
| Exponential (τ-model) | ~ | Y | Y | Y | Y |
| Delayed exponential (τ·t·e^{-t/τ}) | N | Y | Y | Y | ~ |
| Log-normal SFH | N | Y | Y | Y | N |
| Two-burst (sfh2exp) | N | N | N | Y | N |
| Periodic/bursty SFH | N | N | N | Y | N |
| Bursty continuity (Tacchella+2022) | N | N | Y | N | N |
| Non-parametric continuity | Y | N | Y | N | N |
| Non-parametric Dirichlet | Y | N | Y | N | N |
| IFT correlated field SFH | Y | N | N | N | N |
| Particle-based SFH from simulations | N | N | N | N | Y |
| Tabulated SFH | Y | Y | Y | Y | Y |
| **Dust attenuation** | | | | | |
| Charlot & Fall (two-component) | Y | Y | N | ~ | N |
| Calzetti (2000) | Y | Y | Y | Y | Y |
| SMC, LMC extinction curves | Y | Y | Y | Y | Y |
| Cardelli/CCM Milky Way | Y | N | Y | Y | Y |
| Witt & Gordon (2000) geometries | Y | N | N | N | N |
| Narayanan+2024 (star-forming, free slope) | Y | N | N | N | N |
| Lo Faro+2017 (3-component birth cloud) | N | N | N | N | N |
| Kriek & Conroy (UV bump free parameter) | N | N | Y | N | N |
| Reddy+2015 (Lyman-break galaxy curve) | N | N | N | N | N |
| **Dust emission** | | | | | |
| Draine & Li 2007 (tabulated) | Y | Y | N | Y | N |
| Dale+2014 (tabulated templates) | Y | N | N | Y | N |
| Casey (2012) MBB + mid-IR power law | Y | N | N | N | N |
| MAGPHYS two-component cold+warm MBB | N | N | N | N | N |
| THEMIS grain model (Jones+2017) | N | N | N | N | N |
| Schreiber+2018 (free MBB β, mid-IR) | N | N | N | N | N |
| Full dust RT via SKIRT | N | N | N | N | Y |
| **Nebular emission** | | | | | |
| CLOUDY photoionization grids | Y | Y | Y | Y | Y |
| CUE deep neural network (Liner+2024) | Y | N | N | N | N |
| Analytic parametric nebular continuum | Y | N | N | N | N |
| Emission line marginalization (analytic) | Y | N | N | N | N |
| MAPPINGS-III shock grids | ~ | N | N | N | Y |
| Full BPT (including LINER regime) | N | N | N | N | Y |
| Ly-α radiative transfer | N | N | N | N | N |
| 3MdBs photoionization database | N | N | N | N | N |
| **IGM** | | | | | |
| Inoue+2014 (LAF + LLS) | Y | Y | N | N | N |
| Madau+1995 | N | N | Y | N | N |
| Meiksin+2006 | N | N | N | Y | N |
| Patchy IGM scatter (free σ_IGM) | N | N | N | N | N |
| DLA-based attenuation | N | N | N | N | N |
| **AGN (see AGN_MODEL_COMPARISON.md for detail)** | | | | | |
| Accretion disc (analytic) | Y | N | ~ | N | Y |
| Torus (analytic) | Y | N | N | N | Y |
| Torus (radiative transfer templates) | N | N | Y | Y | N |
| Fritz+2006 smooth torus RT | N | N | N | Y | N |
| SKIRTOR clumpy two-phase RT | N | N | N | Y | N |
| CLUMPY (Nenkova+2008) RT | N | N | Y | N | N |
| Polar dust | N | N | N | Y | N |
| AGN NLR + BLR emission | Y | N | N | N | Y |
| AGN X-ray module | Y | N | N | Y | N |
| AGN radio module | Y | N | N | Y | N |
| BH mass + spin as parameters | Y | N | N | N | Y |
| Baldwin effect on line EW | N | N | N | N | N |
| Quasar SED templates (qsogen-style) | N | N | N | Y | N |
| **Radio / submm / X-ray (non-AGN)** | | | | | |
| Synchrotron power law (star-forming) | Y | N | N | Y | N |
| Free-free (thermal bremsstrahlung) | N | N | N | Y | N |
| CO molecular line emission | N | N | N | N | N |
| Thermal SZ effect | N | N | N | N | N |
| XRB scaling (non-AGN X-ray) | Y | N | N | N | N |
| **Spectroscopy** | | | | | |
| Photometric filter convolution | Y | Y | Y | Y | Y |
| Spectral resolution convolution | Y | Y | Y | Y | Y |
| Chebyshev calibration polynomial | Y | N | Y | N | N |
| Emission line fitting in spectra | Y | Y | N | N | N |
| Emission line marginalization | Y | N | N | N | N |
| Lick spectral indices (Mg_b, D_n4000) | N | N | Y | N | N |
| Full-spectrum stellar kinematics (σ_*) | N | N | N | N | N |
| Telluric correction modeling | N | N | N | N | N |
| **Inference** | | | | | |
| MAP / gradient descent | Y | N | ~ | N | N |
| Variational inference (geoVI/MGVI) | Y | N | N | N | N |
| HMC / NUTS | Y | N | ~ | N | N |
| Ray Tracing (Behroozi 2025) | Y | N | N | N | N |
| Laplace approximation | Y | N | N | N | N |
| Pathfinder (L-BFGS path) | Y | N | N | N | N |
| Elliptical slice sampling | Y | N | N | N | N |
| Nested slice sampling (evidence log Z) | Y | N | N | N | N |
| MultiNest nested sampling | N | Y | N | N | N |
| nautilus nested sampling | N | Y | N | N | N |
| dynesty nested sampling | N | N | Y | N | N |
| emcee affine-invariant MCMC | N | N | Y | Y | N |
| Sequential Monte Carlo (SMC) | N | N | N | N | N |
| Simulation-based inference (SBI/NPE) | N | N | N | N | N |
| Hierarchical Bayesian (shared PSD) | Y | N | N | N | N |
| Hierarchical population inference | ~ | N | Y | Y | N |
| **Photometric redshift** | | | | | |
| Redshift as free parameter | Y | Y | Y | Y | N |
| p(z) marginalization output | N | N | ~ | Y | N |
| EAZY-style template fitting | N | N | N | N | N |
| Galaxy type marginalization | N | N | N | N | N |
| **Observation modeling** | | | | | |
| Joint photometry + spectroscopy | Y | N | Y | N | N |
| Upper limit photometry | N | Y | ~ | Y | N |
| Multi-component galaxy (merger) | N | N | N | Y | Y |
| Spatially resolved SED fitting | N | N | N | N | Y |
| Mock image generation | N | N | N | N | Y |
| Strong lensing magnification | N | N | N | N | N |
| Differentiable end-to-end (JAX) | Y | N | N | N | N |

---

## Features tengri has that others lack

These are tengri's distinctive capabilities not found in the external codes surveyed:

1. **IFT correlated field SFH** — models bursty SFHs as Gaussian Process draws with a parameterized PSD. No other SED fitting code models the SFH as a stochastic field with an inferred PSD.

2. **Fully differentiable forward model (pure JAX)** — enables gradient-based inference (HMC, variational inference) for all parameters simultaneously, including PSD hyperparameters and 130+ dimensional latent fields.

3. **Gradient-based variational inference (geoVI/MGVI)** — via NIFTy integration. Scales to ~137 dimensions without the O(D²) cost of NUTS. No other SED fitting code offers this.

4. **Ray Tracing sampler** (Behroozi 2025) — stochastic-gradient-resilient exact MCMC. Verified bit-for-bit identical to the reference implementation.

5. **CUE nebular emission** (deep neural network, Liner et al. 2024) — covers wider parameter space than CLOUDY grids alone.

6. **Analytic calibration polynomial marginalization** — Chebyshev polynomial flux calibration uncertainty marginalized analytically (Johnson+2021/Prospector approach, but implemented in JAX).

7. **XRB non-AGN X-ray** — scaling relation from Fragos+2013, no other open SED code includes this.

8. **Hierarchical PSD inference** — shared SFH hyperparameters across a galaxy population. Others (Prospector, pCIGALE) support simpler population inference but not the PSD-governed burstiness prior.

9. **Witt & Gordon (2000) dust geometry functions** — `wg00_shell`, `wg00_cloudy`, `wg00_dusty` transmission curves from full RT.

10. **Narayanan+2024 attenuation** — empirical free-slope law calibrated to FIRE simulations.

---

## Priority gaps for Paper I/II scope

These features appear in multiple external codes and are relevant to the mock recovery and real-data papers:

### High priority (affects science conclusions)

| Gap | Codes that have it | Why it matters |
|---|---|---|
| Delayed-exponential SFH | bagpipes, FSPS, pCIGALE | Standard comparison SFH in literature; needed for apples-to-apples cross-validation |
| Log-normal SFH | bagpipes, FSPS, pCIGALE | Produces more realistic quenching histories; used in Prospector |
| IMF switching | FSPS, pCIGALE, synthesizer | Cannot constrain IMF-dependent M/L ratios |
| SKIRTOR/CLUMPY torus RT | pCIGALE, FSPS, AGNfitter | Detailed silicate feature profile fitting |
| dynesty / MultiNest | Prospector, bagpipes | Evidence comparison and prior-volume-corrected posteriors |

### Medium priority (known gaps, deferred)

| Gap | Codes that have it | Notes |
|---|---|---|
| BPASS binary stellar populations | synthesizer, pCIGALE | Major systematic for very young and old populations |
| Kriek & Conroy attenuation (UV bump) | FSPS/Prospector | Important for populations with SMC-like dust |
| MAPPINGS-III shock grids | synthesizer | Relevant for post-starburst, AGN outflows |
| Patchy IGM scatter | — | Needed for z > 5 Lyman-break fitting |
| Lick spectral indices | FSPS | Useful for old stellar populations; not a SED-fitting feature per se |
| Full-spectrum stellar kinematics | ppxf | Out of scope for photometric SED fitting |

### Low priority / architectural (long-term)

| Gap | Notes |
|---|---|
| Particle-based SFH (simulation interface) | synthesizer speciality; requires different input API |
| Full dust RT via SKIRT | Requires SKIRT binary; major integration effort |
| CO line emission / SZ | Far-submm instruments; needed for ALMA/Herschel-only work |
| Simulation-based inference (SBI) | Compatible with JAX; possible future addition |
| Spatially resolved SED fitting | Requires pixel/spaxel loop; large-scale change |

---

## Notes on delayed-exponential SFH

The standard "delayed-τ" model (`SFR(t) ∝ t · exp(−t/τ)`) is not named in tengri. The `tsnorm` model is related but peaks at `tau_peak_gyr` with a skewed shape. For cross-validation against bagpipes/FSPS, a true delayed-τ can be approximated with `tsnorm` by tuning `tau_peak_gyr ≈ τ`, but the shapes differ enough to cause bias at high τ.

A minimal delayed-τ SFH can be added to `models/sfh/parametric.py` without touching the IFT or inference layers.

---

## Notes on BPASS binary stellar populations

BPASS is available via synthesizer and python-fsps (requires `fsps.StellarPopulation(imf_type=2, add_stellar_remnants=True, ...)`). BPASS systematically affects:
- Post-AGB / hot subdwarf UV upturn in old populations
- Ionizing photon budget at young ages (~2× higher than single stars)
- Stellar mass estimates at z > 2 (can shift M* by ~0.1–0.2 dex)

Integration into tengri would require a separate BPASS SSP grid loader (parallel to `dsps_wrapper.py`), since DSPS does not support binary populations.

---

## References

- Bruzual & Charlot 2003, MNRAS, 344, 1000 (BC03)
- Carnall et al. 2018, MNRAS, 480, 4379 (bagpipes)
- Conroy & Gunn 2010, ApJ, 712, 833 (FSPS)
- Eldridge et al. 2017, PASA, 34, e058 (BPASS)
- Johnson et al. 2021, ApJS, 254, 22 (Prospector)
- Jones et al. 2017, A&A, 602, A46 (THEMIS)
- Kriek & Conroy 2013, ApJL, 775, L16 (UV bump attenuation)
- Liner et al. 2024 (CUE DNN nebular model)
- Lo Faro et al. 2017, ApJ, 838, 10 (3-component dust)
- Maraston & Strömbäck 2011, MNRAS, 418, 2785
- Narayanan et al. 2024 (dust attenuation from FIRE)
- Noll et al. 2009, A&A, 507, 1793 (pCIGALE)
- Tacchella et al. 2022, ApJ, 926, 134 (bursty continuity SFH)
- Wilkins et al. 2024/2025, arXiv (synthesizer)
- Witt & Gordon 2000, ApJ, 528, 799 (WG00 geometries)
- Yang et al. 2020, MNRAS, 491, 740 (X-CIGALE)
