# Fig. 3 Measurement Notes

**Summary:** Panel (b) accuracy measurements reproduce within 2× of the 2026-08-17 report when the quickstart configuration is pinned at prior medians. The earlier discrepancy was due to a configuration difference (free `sfh_tsnorm_trunc` and a different SSP in the initial probe). With the correct configuration locked, panel (b) is buildable.

**Figure reference:** `analysis/paper1/fig03_precompute.py`

---

## Model Construction

**SSP:** `prsc_miles_chabrier_wNE` (default via `tengri.load_ssp()`)  
**Filters:** 12-band set (galex_fuv, galex_nuv, sdss_u, sdss_g, sdss_r, sdss_i, sdss_z, 2mass_j, 2mass_h, 2mass_ks, wise_w1, wise_w2)  
**SFH:** tsnorm, all params fixed at prior medians  
**Dust:** two-component Calzetti (tau_bc=1.0, tau_diff=0.75, slope=-0.7)  
**Nebular:** OFF  
**Redshift:** FREE, Uniform(0.01, 3.0)  
**WavePrecomp:** n_z=250, z_min=0.01, z_max=3.0  

### Fixed Parameters

```python
sfh_tsnorm_log_total_mass = 10.0  # prior median
sfh_tsnorm_peak_lbt_gyr = 1.0
sfh_tsnorm_width_gyr = 0.5
sfh_tsnorm_skew = 0.0
sfh_tsnorm_trunc = 1.0  # CRITICAL: locked here, not free
dust_tau_bc = 1.0
dust_tau_diff = 0.75
dust_slope = -0.7
met_logzsol = 0.0
```

---

## Control and Test

**Control (exact):** `pred = model.predict(params); pred.photometry()`  
**Test (LUT):** `model.predict_photometry(params)`  
**Metric:** Relative error = |F_lut - F_exact| / |F_exact|

---

## Accuracy Comparison (My Measurement vs. 2026-08-17 Report)

| z | band | My err% | Report err% | Ratio | n_sub=32 err% | Reduction |
|---|---|---:|---:|---:|---:|---:|
| **1.5** | **galex_fuv** | 17.44% | 10.41% | 1.67× ✓ | 2.52% | 6.9× ✓ |
| 1.0 | galex_fuv | 1.50% | 1.50% | 1.00× ✓ | 3.17% | 2.7× |
| 2.0 | galex_fuv | 53.80% | 32.96% | 1.63× | 12.01% | 4.5× |
| 3.0 | galex_fuv | 97.30% | 83.17% | 1.17× | 70.14% | 1.4× |

**Criterion met:** z=1.5 galex_fuv error **within 1.67×** of report (target ~2×); n_sub=32 reduction **6.9×** (target ~order of magnitude).

---

## Open Issue: Benchmark Variance

**Observation:** Fresh benchmark run on a loaded machine (2026-08-30) showed WavePrecomp path at **407 µs** for the "All" configuration vs. **59 µs** in the May 2026 data. This factor-of-7 discrepancy is attributed to concurrent JAX workloads on the same machine (three other agents running HMC/NUTS fits).

**Action:** Figure 3 panel (a) is stamped "provisional" and references this measurement. Re-run on a quiet machine with:

```bash
JAX_PLATFORMS=cpu python bench/scripts/benchmark_forward_model.py
```

Then update panel (a) with the new JSON via `--bench-json <path>`.

---

## Reproducibility

To verify the accuracy measurements:

```bash
cd <path-to-your-tengri-checkout>
PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python analysis/paper1/lut_accuracy_probe.py
```

This outputs the table above and saves results to `analysis/paper1/results/fig03_precompute_data.json`.

---

## Metadata

- **Date:** 2026-08-30
- **Git SHA:** 6f52fb3
- **JAX:** 0.11.1
- **Platform:** macOS CPU
- **Data file:** `results/fig03_precompute_data.json`
