# Missing features vs. competitive SED-fitting codes

**Last updated:** 2026-05-03
**Source:** Spun off from the 2026-05-02 loose-ends audit, Section 4. Originally compared against Prospector, CIGALE, BAGPIPES, MAGPHYS.

This is a **long-lived backlog** of features that competitive codes ship and `tengri` does not. Items are *not* bugs — they are deliberate scope decisions to revisit. Priorities: **physics components** and **observation layer** first, per project direction. Re-verify the gap (read current code) before opening a PR — some entries may have been quietly addressed since the audit.

## Verification check

Before working on any item, run a quick sanity check:

```bash
# Search for hints the feature already exists
rg -i "<keyword>" src/tengri/ docs/

# Check if a related TODO/known-bug entry already covers it
rg -i "<keyword>" docs/known_bugs.md HANDOFF.md docs/dev/
```

If you find evidence the feature shipped, **update this file** (move the item to a "Resolved" section at the bottom) rather than starting work.

---

## Tier 1 — Physics-component gaps

### 1. Variable IMF / BPASS option
- **Status:** entirely absent. Chabrier hardcoded in most SSP sources.
- **What's needed:** user-facing IMF switch + a path to BPASS-derived SSP grids for binary-star ionizing spectra.
- **Where:** `src/tengri/components/sps/` (new IMF module), `src/tengri/components/nebular/` (CLOUDY grid choice).

### 2. Stellar vs. gas metallicity decoupling
- **Status:** has `evolving_metallicity` ramp + `chem_evol` (gas-regulator); missing **independent** stellar-vs-gas Z and α/Fe abundance freedom.
- **Where:** `src/tengri/components/sps/`, `src/tengri/components/nebular/`, parameter spec.
- **Note:** check `chem_evol.py` first — the audit assumed full decoupling is missing; verify before scoping.

### 3. AGN+host decomposition products
- **Status:** AGN physics complete, but no extracted "AGN fraction at λ", host-only SED, or bolometric-decomposition outputs in `Posterior` results.
- **Where:** `src/tengri/inference/posterior.py` (new derived methods), `src/tengri/forward/sed_model.py` (split prediction path).
- **Pairs with:** the BPT diagnostics in §6 below.

### 4. Energy-balance diagnostic
- **Status:** module exists in `dust/` but no user-facing "absorbed ≈ emitted" check / diagnostic output.
- **Where:** `src/tengri/analysis/diagnostics/` (new utility), surfaced via `Posterior`.

### 5. Additional SFH parameterizations
- **Status:** has delayed-exp, double-power-law, truncated skew-normal, GP/IFT nonparametric.
- **Missing:** constant SFR, rising-only, piecewise/linear (Prospector-`continuity`), composite (quiescent + post-quench).
- **Where:** `src/tengri/components/sfh/`.

---

## Tier 2 — Observation-layer gaps

### 6. Photometric zero-point systematic floor
- **Status:** noise model supports calibration floors via Student-t + Chebyshev calibration polynomial, but **per-band / per-survey** ZP systematics are not first-class.
- **Where:** `src/tengri/observation/noise.py`, `src/tengri/observation/photometry.py`.

### 7. Telluric / sky-residual model
- **Status:** spectroscopy supports Chebyshev calibration polynomial; no explicit telluric absorption template or sky-residual nuisance.
- **Where:** `src/tengri/observation/spectroscopy.py`.

### 8. Velocity-dispersion fitting
- **Status:** has variable-R LSF; no explicit σ_v free parameter or kinematic broadening kernel.
- **Where:** `src/tengri/observation/spectroscopy.py`, `src/tengri/forward/_kernels/`.

### 9. Correlated noise / jitter terms
- **Status:** Gaussian / Student-t i.i.d. only. No GP-correlated noise in time/wavelength, no per-pixel jitter.
- **Where:** `src/tengri/observation/noise.py`.

### 10. Aperture corrections
- **Status:** entirely absent — pipeline assumes pre-corrected photometry.
- **Where:** `src/tengri/observation/photometry.py` (preprocessing hook), or upstream catalog tooling.

### 11. Surface-brightness dimming / extended-source corrections
- **Status:** utilities exist for SB conversions; no extended-source dimming model in the forward path.
- **Where:** `src/tengri/utils/`, `src/tengri/forward/sed_model.py`.

### 12. Photo-z prior objects
- **Status:** wide uniform priors only. No GMM-from-external-photo-z import path or hierarchical photo-z PDF.
- **Where:** `src/tengri/parameters/priors.py` (new prior class), `src/tengri/observation/`.

---

## Tier 3 — Diagnostic / science-product gaps

### 13. BPT-style classification utilities
- **Status:** `Posterior.bpt_nii()` exists for coordinates; no Kewley+2001 / Kauffmann+2003 demarcation classification utility.
- **Where:** `src/tengri/analysis/diagnostics/lines.py` or `inference/posterior.py`.

### 14. Balmer decrement dust correction
- **Status:** `Posterior.balmer_decrement()` exposes the ratio; no standard "decrement-derived A_V → de-redden Hα" utility.
- **Where:** `src/tengri/analysis/diagnostics/`.

### 15. Doublet-ratio constraints in likelihood
- **Status:** `[OIII] 4959/5007`, `[NII] 6548/6584`, `[SII] 6717/6731` ratios not enforced in the marginalized-line likelihood.
- **Where:** `src/tengri/observation/emission_lines.py`, `src/tengri/inference/`.

---

## Resolved (move items here, do not delete)

- *(none yet)*
