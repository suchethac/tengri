# Paper Analysis Plan: diffsed Paper I

## Paper: "Information Field Theory for Galaxy SED Fitting: Reconstructing Bursty Star Formation Histories"

**Target journal:** ApJ (AASTeX 6.31)
**Draft location:** `~/writing-workspace/projects/differentiable_psd_sed_fitting/`

---

## Revised Figure Plan (8 figures)

### Methods Figures (Section 2)

**Figure 1: Framework overview schematic** (`fig_overview`)
- Flowchart: ξ → [IFT standardization] → SFH field → [DSPS SPS] → [Dust] → [Photometry/Spectra] → inference
- Show the modular pipeline with the key equation $\mathcal{H}(\xi|d)$
- *Status: placeholder in draft — manual diagram*

**Figure 2: PSD → GP → SFH** (`fig_psd_sfh`)
- 3-panel: P(ω), GP realizations, full SFH (mean × exp(GP))
- 4 regimes: smooth, moderate, bursty, highly bursty
- *Source: NB01 figures — EXISTS*

**Figure 3: Forward model pipeline** (`fig_forward_model`)
- SFH → weights → CSP SED → dust → observed photometry
- Show SED with filter transmission curves overlaid
- *Source: NB02 figures — EXISTS*

### Recovery Test Figures (Section 3)

**Figure 4: Individual SFH recovery** (`fig_sfh_recovery`) — **MOST IMPORTANT**
- 4×2 grid: PSD regime × (photometry, spectroscopy)
- SFH in linear time with truth (black) + posterior median + 68% CI
- Shows how spectroscopy improves recent-time SFH recovery
- *Script: `analysis/fig04_sfh_recovery.py` — IMPLEMENTED*

**Figure 5: PSD parameter recovery** (`fig_psd_recovery`)
- 2×2 grid: moderate/bursty × phot/spec
- 2D KDE contours of (σ_PSD, τ_PSD) posterior with truth marked
- Shows when PSD params are constrainable
- *Script: `analysis/fig05_psd_recovery.py` — IMPLEMENTED*

**Figure 6: Posterior predictive checks** (`fig_ppc`)
- 2-panel: photometric PPC (left) + spectroscopic PPC (right)
- Green (phot posterior) + red (spec posterior) overlaid on data
- *Source: NB00/NB05 — EXISTS*

### Computational Performance (Section 3.4)

**Figure 7: Speed comparison** (`fig_speed`)
- Grouped bar chart: MAP vs Ray Tracing vs NUTS vs geoVI
- Smooth (5D) and stochastic (137D) configurations
- *Script: `analysis/fig07_speed_benchmarks.py` — IMPLEMENTED*

**Figure 8: End-to-end gradient sensitivity** (`fig_gradient`)
- Jacobian heatmap: ∂flux/∂θ for all params × all bands
- Demonstrates full differentiability and parameter–band coupling
- *Script: `analysis/fig08_gradient_sensitivity.py` — IMPLEMENTED*

### Deferred to Paper II

- Hierarchical PSD recovery (needs hierarchical inference implementation)
- Population distinction test
- 2D PSD recovery map (16×100 mock runs)
- S/N dependence study
- Batch scaling benchmarks

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
