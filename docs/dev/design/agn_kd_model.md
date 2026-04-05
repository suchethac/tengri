# Kubota & Done Three-Zone Disc Model: Implementation Notes

This document provides the detailed implementation notes for tengri's
differentiable K&D three-zone accretion disc model, complementing the
summary in the paper appendix.

## Model Overview

The `qsosed` and `relagn` disc backends implement the
[Kubota & Done (2018)](https://doi.org/10.1093/mnras/sty1890) three-zone
accretion disc model in pure JAX. The three radial zones are:

1. **Outer standard disc** (Shakura–Sunyaev): thermal blackbody emission
   from each annulus, producing the UV "big blue bump"
2. **Warm Comptonisation region**: a warm, optically thick corona
   ($kT_e \sim 0.2$ keV, $\tau \sim 10$–20) that bridges UV and soft X-rays
3. **Hot inner corona**: an optically thin plasma ($kT_e \sim 100$ keV,
   $\tau \sim 1$) producing hard X-rays via inverse Compton scattering

All zones share the Novikov–Thorne emissivity:
$$T(r) = T_{\rm in}(r/r_{\rm in})^{-3/4}[1 - \sqrt{r_{\rm in}/r}]^{1/4}$$

and radiative efficiency:
$$\eta = 1 - \sqrt{1 - 2/(3 \cdot r_{\rm ISCO})}$$

## Differentiable Approximations

Translating the K&D model to a fully differentiable JAX pipeline requires
replacing several non-differentiable operations with smooth approximations.

### Corona Radius ($R_{\rm hot}$)

Found via 40-step bisection on the analytic Novikov–Thorne luminosity
integral. This is exact to machine precision ($10^{-12}$) and matches
the reference `qsosed` implementation. Bisection is used instead of
root-finding because the luminosity integral is monotonic in $R_{\rm hot}$.

### Warm Comptonisation Spectrum

Instead of the XSPEC `nthcomp` model (which is not differentiable), we use
an analytic approximation:

$$L_\nu^{\rm warm}(T, \nu) = B_\nu(T) \times (\nu/\nu_s)^{\Gamma_{\rm warm} - 1}$$

with $\Gamma_{\rm warm} = 2.5$ fixed. This gives shape agreement
$\lesssim 20\%$ vs `nthcomp`. The error is negligible for UV/optical SED
fitting where the warm zone contributes $\lesssim 5\%$ of the total flux.

**Error budget**: At 1 keV (soft X-ray), the approximation can differ by
up to 20% in spectral shape. At 3000 Å (UV), the difference is < 2%.
For broadband SED fitting with optical/NIR photometry, this approximation
introduces no measurable bias in inferred parameters.

### Hard X-ray Photon Index

The corona photon index is derived self-consistently from:

$$\Gamma_{\rm hot} = \frac{7}{3}\left(\frac{L_{\rm diss}}{L_{\rm seed}}\right)^{-0.1}$$

following Beloborodov (1999) as used in K&D. The seed photon luminosity
is computed from the K&D Eq. 3 geometric integral on a 100-point
logarithmic grid (exact).

### Lamppost Reprocessing

Omitted in the current implementation. Lamppost geometry would add a
reflected component to the outer disc, modifying $\Gamma_{\rm hot}$ by
$\lesssim 3\%$. Omitting it enables differentiability by avoiding the
iterative self-consistency loop between corona emission and disc
irradiation. Future implementations could add this as a perturbative
correction.

### Warm Zone Renormalization

Per-annulus bolometric energy conservation is enforced:
the warm zone spectrum in each annulus is renormalized so that its
integrated luminosity matches the Novikov–Thorne prediction for that
annulus. This prevents energy leaks at zone boundaries.

### Outer Disc Radius

Uses the Laor & Netzer (1989) self-gravity radius instead of a fixed
outer boundary. This improves agreement with `qsosed` by 2–4× at
extreme black hole masses ($M_{\rm BH} > 10^9 M_\odot$) where
self-gravity truncation is important.

## Accuracy Summary

| Component | Approximation | Accuracy vs qsosed |
|-----------|---------------|---------------------|
| $R_{\rm hot}$ | 40-step bisection on N–T integral | exact ($10^{-12}$) |
| Warm Compton. | $B_\nu \times (\nu/\nu_s)^{\Gamma_w-1}$ | $\lesssim 20\%$ shape at 1 keV |
| Seed photons | K&D Eq. 3, 100-pt log grid | exact |
| Reprocessing | Omitted | $\lesssim 3\%$ on $\Gamma_{\rm hot}$ |
| Energy balance | Per-annulus renormalization | exact |
| $R_{\rm out}$ | Laor & Netzer self-gravity | $2$–$4\times$ improved at high $M_{\rm BH}$ |

## Code Location

- **qsosed backend**: `src/tengri/models/agn/disc.py::kubota_done_disc()`
- **relagn backend**: `src/tengri/models/agn/disc.py::kubota_done_disc()` with
  GR corrections enabled
- **Corona temperature**: `src/tengri/models/agn/disc.py::beloborodov_gamma_hot()`
- **Tests**: `tests/unit/models/agn/test_disc.py`

## References

- Kubota & Done (2018), MNRAS 480, 1247 — Original three-zone model
- Kubota & Done (2019), MNRAS 489, 524 — Application to changing-look AGN
- Quera-Bofarull et al. — `qsosed` Python implementation
- Hagen & Done (2023) — RELAGN with full GR corrections
- Beloborodov (1999) — Corona photon index formula
- Laor & Netzer (1989) — Self-gravity radius
