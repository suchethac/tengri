# ADR-0017 — Photometric Filter-Convolution Convention

## Status

Accepted (2026-05-30). Related: ADR-0016 (separate filter-integration from flux
dimming) introduced `lnu_filter_integral`, the bare-L_ν step this ADR re-weights.

## Context

Broadband photometry turns an SED into a band flux via a weighted bandpass
integral,

```
            ∫ F_ν(λ) T_b(λ) w(λ) dλ
⟨F_ν⟩_b  =  ───────────────────────,
              ∫ T_b(λ) w(λ) dλ
```

and the **bandpass weight `w(λ)`** is a convention choice. The SED-fitting
ecosystem uses two physically-distinct conventions (verified against each
code's source):

| Convention | `w(λ)` | Detector model | Used by |
|---|---|---|---|
| **photon (Bessell)** | `1/λ` | photon-counting | DSPS, FSPS, sedpy, prospector |
| **energy** | `1/λ²` | energy / flat-in-frequency | CIGALE, bagpipes |

They agree exactly for a flat-`F_ν` source (the AB reference) and diverge by
**5–40 mmag**, band- and SED-slope-dependent, for real SEDs (first measured in
issue #436).

Through 2026-05, tengri's kernel weighted `F_ν` by **`λ`** (`compute_flux_density`,
`grid_interp.preintegrate_grid`, and the paper appendix `eq:bandpass_flux`) and
its docstring falsely claimed "matches DSPS." That weight matches *neither*
convention: it is an f_λ→f_ν units transplant — the energy formula `∫ T L_λ dλ`
(correct in f_λ) rewritten onto an `L_ν` SED as `∫ F_ν T λ dλ`, but
`L_λ = L_ν c/λ²`, not `L_ν λ` (error = two factors of λ). This biased colours,
hence photo-z / stellar mass / dust inference. Issue #436 (closed via #446) had
papered over a related symptom by routing `predict_magnitudes` and
`predict_photometry` through the *same* (wrong) kernel; issues #443/#515 tracked
the deeper convention question but framed it as "three valid conventions"
(photon-Tokunaga / photon-Bessell / energy) — an over-generalisation: tengri's
`λ` weight is a bug, not a third convention.

The decision is also entangled with **ADR-0015** (CIGALE-faithfulness for the
AGN / panchromatic model): CIGALE uses the energy convention, so reproducing it
requires that convention to be available.

## Decision

1. **Default to the photon-counting (Bessell) convention, `w(λ) = 1/λ`.** It is
   the physically correct mean for photon-counting detectors (every optical/NIR
   CCD) and how the AB system is realised by surveys
   (Oke 1983; Hogg+2002 Eq. 5; Fukugita+1996 Eq. 7), and it matches tengri's own
   SSP engine, DSPS (Hearin+2023, with `T` = photon transmission probability).
   This eliminates the prior λ-weighting bug and the internal mag↔phot
   inconsistency of #436 *correctly* (the shared kernel is now the right one).

2. **Expose the energy convention, `w(λ) = 1/λ²`, as a selectable option** for
   CIGALE / bagpipes parity (ADR-0015). A single `FilterConvention` enum
   (`bessell` | `energy`) selects `w(λ)` at one chokepoint shared by the exact
   kernel and the build-time preintegration.

3. **The convention is a build-time choice.** It is baked into the preintegrated
   `WavePrecomp` lookup table, so the exact path (`approx=None`) and the
   precomputed path must use the same convention; the effective/pivot wavelength
   and the Taylor dust moment are recomputed consistently with `w(λ)` (so the
   moment vanishes for a flat template).

4. **Do not keep the `λ` weight as a blessed option** — it is simply removed.
   Supersede the "three conventions" framing of #443/#515: there are two
   *legitimate* conventions (a bug + one alternative definition), not three.

5. **DSPS is the reference of record.** `bessell` is pinned bit-faithful to
   `dsps.photometry` by `tests/crossval/test_filter_convention_parity.py`
   (≤1e-6 on the band-mean L_ν; ≤1e-4 mag on `calc_obs_mag` colours); `energy`
   is pinned to an independent analytic reference.

## Consequences

- Default photometry values shift (the λ→1/λ fix); kernel snapshot fixtures
  regenerate. Colours move toward FSPS/DSPS.
- New public surface: `tengri.FilterConvention`, `tengri.list_filter_conventions()`;
  `compute_flux_density(..., convention=)`, `preintegrate_grid(..., convention=)`.
- Touched paths: `observation/photometry.py`, `utils/grid_interp.py`,
  `utils/optimizations.py`, `analysis/diagnostics/{green_functions,spectral}.py`;
  shared definition in the leaf module `utils/filter_convention.py`.
- The paper appendix (`999-appendix.tex` `eq:bandpass_flux`, `eq:lambda_eff`,
  `eq:ssp_moment`) and `docs/units.md` are corrected to state `w(λ)` and the
  default convention.
- Related: closes #443 (reframed), advances #515; #438 (WavePrecomp `n_z`
  convergence) is a distinct interpolation issue in the same file.

## References

- Oke 1983, ApJ 266, 713 (AB system); Hogg, Baldry, Blanton & Eisenstein 2002,
  arXiv:astro-ph/0210394, Eq. 5; Fukugita et al. 1996, AJ 111, 1748, Eq. 7;
  Bessell & Murphy 2012, PASP 124, 140; Hearin et al. 2023 (DSPS), arXiv:2112.06830;
  Conroy, Gunn & White 2009 (FSPS); Boquien et al. 2019, A&A 622, A103 (CIGALE);
  Brown et al. 2016, AJ 152, 102.
- See also `docs/units.md` (Photometric filter-convolution convention),
  ADR-0015 (CIGALE alignment), ADR-0004 (kernel strategy).
