# Paper Analysis Plan: diffsed Paper I

## Paper: "Information Field Theory for Galaxy SED Fitting: Reconstructing Bursty Star Formation Histories"

**Target journal:** ApJ (AASTeX 6.31)
**Draft location:** `~/writing-workspace/projects/differentiable_psd_sed_fitting/`

---

## Figure Plan

### Methods Figures (Section 2)

**Figure 1: Framework overview schematic** (`fig_overview`)
- Flowchart: ξ → [IFT standardization] → SFH field → [DSPS SPS] → [Dust] → [Photometry/Spectra] → [geoVI inference]
- Show the modular pipeline with the key equation $\mathcal{H}(\xi|d)$
- *Status: placeholder in draft*

**Figure 2: PSD → GP → SFH** (`fig_psd_sfh`)
- 3-panel: P(ω), GP realizations, full SFH (mean × exp(GP))
- 4 regimes: smooth, moderate, bursty, highly bursty
- *Source: NB01 figures 01_psd_overview.png, 01_sfh_ensemble_linear.png*

**Figure 3: Forward model pipeline** (`fig_forward_model`)
- SFH → weights → CSP SED → dust → observed photometry
- Show SED with filter transmission curves overlaid
- *Source: NB02 figures*

### Recovery Test Figures (Section 3)

**Figure 4: Test matrix** (`fig_test_matrix`)
- Summary table/schematic of the mock program
- Axes: redshift (0.1, 2, 6) × data type (phot, spec) × PSD regime
- *Status: needs creating*

**Figure 5: Individual SFH recovery — photometry** (`fig_sfh_recovery_phot`)
- 2×2 or 3×2 grid: PSD regime × (truth, recovered)
- SFH in linear time with 16-84% posterior fill
- Compare PSD prior vs continuity prior
- *Needs: run geoVI on mock galaxies at z=0.1 with SDSS ugriz*

**Figure 6: Individual SFH recovery — spectroscopy** (`fig_sfh_recovery_spec`)
- Same layout as Fig 5 but with spectral fitting
- Show improved recent-time SFH recovery
- *Needs: implement spectral fitting mock + inference*

**Figure 7: PSD parameter recovery (individual)** (`fig_psd_individual`)
- Corner plot: σ_PSD vs τ_PSD posteriors for single galaxy
- Show constrainability depends on data type and S/N
- *Needs: run individual fits with PSD params free*

**Figure 8: 2D PSD recovery map** (`fig_psd_recovery_map`)
- 4×4 grid in (σ_PSD, τ_PSD) space
- Each cell: bias and scatter in recovered PSD params
- Color-coded: green=recoverable, red=unconstrained
- *Needs: 16×100 mock + fit runs*

**Figure 9: S/N dependence** (`fig_snr_dependence`)
- SFH recovery quality vs photometric S/N (5, 10, 20, 50, 100)
- Compare PSD prior vs continuity prior improvement
- *Source: NB04 SNR comparison, but needs geoVI version*

**Figure 10: Posterior predictive checks** (`fig_ppc`)
- Photometry + spectrum with posterior spread overlaid on data
- Show that the model can reproduce the data
- *Source: NB05 05_posterior_predictive.png*

### Population-Level Figures (Section 4)

**Figure 11: Hierarchical PSD recovery** (`fig_hierarchical_psd`)
- Posterior on shared (σ_PSD, τ_PSD) from N=500 galaxies
- Show convergence as N increases (50, 100, 200, 500)
- *Needs: hierarchical inference implementation*

**Figure 12: Population distinction** (`fig_population_distinction`)
- Two populations with different PSD settings
- Show posteriors are well-separated
- *Needs: hierarchical inference*

### Computational Performance (Section 3.4)

**Figure 13: Speed comparison** (`fig_speed`)
- Wall time: MAP vs Ray Tracing vs NUTS vs geoVI on smooth (5D) and stochastic (137D)
- Compare with Prospector/dynesty literature benchmarks
- Include approximate photometry speedup
- *Source: NB05 timing table + Zacharegkas+2025 comparison*

**Figure 14: Scaling with batch size** (`fig_scaling`)
- GPU-parallel mock generation and fitting
- *Source: NB03 mock_batch benchmarks*

---

## Analysis Scripts Needed

### Phase 1: Individual Galaxy Recovery (code exists, need analysis scripts)

1. **`analysis/01_sfh_recovery_phot.py`** — Generate 100 mocks per PSD regime at z=0.1 with SDSS photometry, fit each with geoVI, compute SFH residuals
2. **`analysis/02_sfh_recovery_spec.py`** — Same but with spectral fitting (R=100, 1000-8000Å rest)
3. **`analysis/03_psd_individual_recovery.py`** — Fit individual galaxies with σ_PSD and τ_PSD as free params
4. **`analysis/04_snr_dependence.py`** — Vary S/N from 5-100, quantify SFH recovery improvement
5. **`analysis/05_psd_recovery_map.py`** — 4×4 grid in PSD space, 100 mocks each

### Phase 2: Population-Level Recovery (needs hierarchical inference)

6. **`analysis/06_hierarchical_psd.py`** — Shared PSD recovery from N galaxies
7. **`analysis/07_population_distinction.py`** — Two-population test

### Phase 3: Computational Benchmarks

8. **`analysis/08_speed_benchmarks.py`** — Wall time comparison table (include Ray Tracing sampler alongside MAP, NUTS, and geoVI)
9. **`analysis/09_photometry_approximation.py`** — Accuracy of Zacharegkas+2025 approximation

---

## Paper Section Status

| Section | Status | Key missing items |
|---------|--------|-------------------|
| 1. Introduction | ~80% drafted | Update with final results |
| 2. Methods | ~70% drafted | Figure 1 schematic, minor updates |
| 3. Recovery tests | Structure only | All analysis + figures |
| 4. Results | Structure only | Population-level analysis |
| 5. Discussion | Structure only | Write after results |
| 6. Conclusion | Structure only | Write last |
| Appendix | Placeholder | Mathematical details |

---

## Immediate Next Steps

1. **Quickstart update**: Show ugriz photometry fitting AND spectral fitting in NB00, compare posteriors
2. **Analysis script infrastructure**: Create `analysis/` directory with common mock generation + fitting utilities
3. **Figure 5 (SFH recovery)**: The most important paper figure — run 100 mocks × 4 PSD regimes × geoVI
4. **Speed benchmarks**: Formalize the timing comparison table
5. **NB04 spectral fitting**: Add spectral fitting demonstration

---

## Key Numbers to Report

- 21.6× gradient speedup from approximate photometry (already measured)
- NUTS: 1/500 divergences at target_accept=0.9 (smooth model)
- geoVI: ~45s for smooth (5D), ~70s for stochastic (137D)
- Mock generation: wall time for 100-1000 galaxies via mock_batch
- SFH recovery: RMSE in log(SFR) at <100 Myr, <1 Gyr, full history
