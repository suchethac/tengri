# Alpha-Enhancement in tengri — Text for Paper

**Pass this to the paper-writing agent for inclusion in the methods paper.**

---

## Section: Alpha-Element Enhancement

### Context (for the introduction/motivation)

Real galaxies do not have solar-scaled abundance patterns. Stars formed at early cosmic times, before Type Ia supernovae enriched the ISM with iron-peak elements, exhibit elevated [α/Fe] ratios of +0.3 to +0.5 dex (Thomas, Maraston & Bender 2003; Conroy, Graves & van Dokkum 2014). This α-enhancement is ubiquitous in massive ellipticals (whose stars formed rapidly), the Milky Way thick disk, and high-redshift quiescent galaxies at z > 2 (Beverage et al. 2024). Ignoring α-enhancement biases stellar population parameters: at fixed total metallicity, an α-enhanced population has weaker Fe lines and stronger Mg/Ca/Ti features, which can be misinterpreted as an age or metallicity shift if only solar-scaled templates are available.

Most SED fitting codes handle α-enhancement through the "effective metallicity" approximation: [Z/H]_eff ≈ [Fe/H] + 0.75 × [α/Fe] (Thomas et al. 2003; Vazdekis et al. 2015). This maps the α-enhanced spectrum onto a solar-scaled template at a shifted metallicity. While adequate for broadband photometry, this approximation breaks down for spectroscopic fitting because the spectral signatures of α-enhancement (e.g., Mg b at 5177 Å, Ca triplet at 8498–8662 Å, and UV iron line blanketing at 1500–1900 Å) are qualitatively different from a simple metallicity shift.

### Methods text (for the model description)

tengri supports α-enhanced stellar population synthesis through a four-dimensional SSP grid with axes (total metallicity [M/H], alpha enhancement [α/Fe], stellar age, wavelength). This replaces the effective metallicity approximation with proper bilinear interpolation across pre-computed α-enhanced templates from sMILES (Knowles et al. 2023), BPASS v2.3 (Byrne et al. 2022), or the α-MC library (Park et al. 2024). All three libraries provide SSPs at [α/Fe] = {−0.2, 0.0, +0.2, +0.4, +0.6} dex, with self-consistent treatment of α-enhancement at both the isochrone and stellar atmosphere levels.

The grid interpolation is bilinear in (log Z, [α/Fe]) space:

    SSP(Z, [α/Fe]) = (1−f_Z)(1−f_α) SSP_{Z_lo, α_lo}
                    + f_Z (1−f_α) SSP_{Z_hi, α_lo}
                    + (1−f_Z) f_α  SSP_{Z_lo, α_hi}
                    + f_Z f_α      SSP_{Z_hi, α_hi}

where f_Z and f_α are the linear interpolation fractions clamped to [0, 1]. This is implemented as a JIT-compiled JAX function, preserving full differentiability with respect to both [M/H] and [α/Fe].

A critical convention: when using 4D α-enhanced grids, the metallicity parameter [M/H] represents *total* metallicity (including the contribution of α-elements), not iron abundance [Fe/H]. This follows the sMILES/α-MC convention where each [α/Fe] slice has the same total metal content but different Fe/α partition. The relationship is (Salaris, Chieffi & Straniero 1993; Knowles et al. 2023 Eq. 2):

    [M/H] = [Fe/H] + 0.66154 × [α/Fe] + 0.20465 × [α/Fe]²

#### Time-evolving [α/Fe]

In the simplest mode, a single [α/Fe] is applied uniformly to all stellar populations in the galaxy. However, tengri also supports a time-evolving [α/Fe](t) that captures the physical expectation from chemical evolution: old stellar populations formed from α-enriched gas (before Type Ia enrichment), while young populations formed from gas with approximately solar abundance ratios. We parameterize this as a linear ramp in lookback time:

    [α/Fe](t_lookback) = [α/Fe]_young + ([α/Fe]_old − [α/Fe]_young) × (t_lookback / t_universe)

This adds one free parameter ([α/Fe]_old, with [α/Fe]_young typically fixed at 0.0) and is analogous to the metallicity evolution parameterization already used for [M/H](t). The per-age-bin [α/Fe] values are computed deterministically and used in a per-age bilinear interpolation of the 4D SSP grid, where each age bin can have both a different metallicity and a different [α/Fe].

#### Backward compatibility

When 4D α-enhanced SSP grids are not available, tengri falls back to the effective metallicity approximation with [Z/H]_eff = [Fe/H] + 0.75 × [α/Fe]. This approximation is explicitly *not* the default when 4D grids are loaded, ensuring that the proper spectral effects of α-enhancement are captured whenever the data support it.

### Key numbers for the paper

- **Grid dimensions:** (n_met, n_alpha, n_age, n_wave) — typically (10, 5, 53, ~4000) for sMILES, (13, 5, ~50, ~20000) for BPASS
- **Free parameters added:** 0 (global [α/Fe]) to 1 (time-evolving [α/Fe]_old)
- **Interpolation cost:** negligible — bilinear is 4 multiplications per wavelength pixel
- **Memory:** ~46 MB (float32) for sMILES 4D grid, ~2.6 GB for full BPASS

### References to cite

- Knowles, A. T., et al. 2023, MNRAS, 523, 3450 (sMILES)
- Byrne, C. M., & Stanway, E. R. 2022, MNRAS, 512, 5329 (BPASS v2.3 α-enhanced)
- Park, M., et al. 2024, arXiv:2410.21375 (α-MC)
- Thomas, D., Maraston, C., & Bender, R. 2003, MNRAS, 339, 897 (effective metallicity)
- Vazdekis, A., et al. 2015, MNRAS, 449, 1177 (MILES α-enhanced)
- Salaris, M., Chieffi, A., & Straniero, O. 1993, ApJ, 414, 580 ([M/H]–[Fe/H]–[α/Fe] relation)
- Conroy, C., Graves, G. J., & van Dokkum, P. G. 2014, ApJ, 780, 33 (α-enhancement in ETGs)
- Beverage, A. G., et al. 2024, ApJ, 966, 1 (Heavy Metal Survey, z~2 [α/Fe])

---

## Appendix: Metallicity Convention Details (for methods section or appendix)

Three metallicity conventions exist in the literature. tengri uses [Fe/H] as
the canonical SSP grid axis:

- **[Fe/H]** — iron abundance relative to solar: log₁₀(N_Fe/N_H) − log₁₀(N_Fe/N_H)_☉
- **[M/H]** (or [Z/H]) — total metal mass fraction relative to solar: log₁₀(Z/X) − log₁₀(Z/X)_☉
- **Z** — absolute metal mass fraction

At solar abundance ratios ([α/Fe] = 0.0), all three conventions reduce to the
same quantity: [Fe/H] = [M/H] = log₁₀(Z/Z_☉).

When [α/Fe] ≠ 0, they diverge because α-elements (O, Mg, Si, Ca, Ti) contribute
~70% of the metal mass. The Salaris relation (Salaris, Chieffi & Straniero 1993)
connects them:

    [M/H] = [Fe/H] + 0.66154 × [α/Fe] + 0.20465 × [α/Fe]²

This is a semi-empirical fit to detailed stellar interior models. At [α/Fe] = +0.4
(typical for massive ellipticals and MW thick disk), [M/H] exceeds [Fe/H] by
~0.30 dex. The quadratic term is small (0.033 dex at [α/Fe] = 0.4) but
non-negligible at [α/Fe] = 0.6.

We adopt [Fe/H] as the canonical grid axis because: (i) it is directly measured
by spectroscopic surveys (SDSS, GALAH, APOGEE); (ii) it is the native variable
of MESA/MIST stellar evolution models used by α-MC (Park et al. 2024); and
(iii) interpolating at fixed [Fe/H] cleanly isolates the spectral effect of
varying [α/Fe], since the iron-line blanketing is held constant.

The commonly used approximation [Z/H]_eff ≈ [Fe/H] + 0.75 × [α/Fe]
(Thomas, Maraston & Bender 2003; Vazdekis et al. 2015) is a linearization
of the Salaris relation with a different coefficient (0.75 vs 0.66) reflecting
different solar mixture assumptions. This approximation is used only when
alpha-enhanced SSP grids are not available; when 4D grids are loaded, tengri
uses proper bilinear interpolation across the [α/Fe] dimension.
