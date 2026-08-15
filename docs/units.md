# Units and conventions

Tengri carries no `astropy.units`-style runtime tagging. Every array is
a plain JAX array in a *fixed, documented unit*; conversions live as
pure functions in `tengri.units`.

This page lists the unit each layer expects, the conventions for
metallicity / SFR / time, and the conversion helpers you'll reach for
most often.

## The conventions, in one table

| Quantity | Unit | Where it shows up |
|---|---|---|
| Wavelength | Å, vacuum | `wave_obs`, `wave_rest`, filter curves, line catalogs |
| Spectral flux density `F_ν` | erg s⁻¹ cm⁻² Hz⁻¹ | observed photometry / spectroscopy |
| Spectral luminosity `L_ν` | erg s⁻¹ Hz⁻¹ | every `SEDComponent.apply` output |
| Bolometric luminosity | erg s⁻¹ (preferred) or `L_sun` | AGN `agn_log_lbol = log10(L_bol/L_sun)` |
| Time / age | yr (internal), Myr / Gyr (user-facing) | SFH age grids, PSD timescales |
| SFR | M☉ yr⁻¹ | SFH outputs, `predict_sfh` |
| Stellar mass | M☉ | mass normalizations |
| Metallicity (SSP grid) | `log10(Z)` absolute | DSPS grid axis |
| Metallicity (user API) | `log10(Z/Z_sun)` | `met_logzsol`, with `LOG10_ZSUN = -1.848` offset |
| Magnitudes | AB system | photometry helpers default to AB |
| Distance modulus | mag | `distance_modulus_from_dl` |
| Redshift | dimensionless | `redshift` parameter |

Vacuum wavelengths throughout — including emission lines (`H_alpha = 6564.61 Å`).
**Never use air wavelengths.** If you need them, convert at the boundary:

```python
from tengri.units import vacuum_to_air
wave_air = vacuum_to_air(wave_vac)  # Å -> Å
```

## Where unit boundaries live

Unit translations happen at three places. Everywhere else the unit is
fixed:

1. **`Parameters` → forward model.** User-facing names (`psd_tau_myr`,
   `agn_log_lbol`, `met_logzsol`) are translated to internal units
   (years, erg/s, `log10(Z)` absolute) by the `internal_param_map` on
   each registered component. The user only sees the friendly unit;
   the model only sees the internal one.
2. **Forward model → observation.** `SEDModel.predict_*` returns flux
   in the unit declared by the `Observation` — F_ν cgs by default.
3. **Observation → user analysis.** Use the conversions below to take
   F_ν cgs to Jy, AB mag, maggies, etc., for plotting or comparison
   with catalog data.

## Conversion helpers

All in `tengri.units` and JAX-native (JIT/grad/vmap-safe):

```python
from tengri import units

# Spectral flux density — every one has its inverse
units.fnu_to_jy(fnu)              # erg/s/cm²/Hz -> Jansky
units.jy_to_fnu(jy)
units.fnu_to_ujy(fnu)             # ... -> microjansky
units.ujy_to_fnu(ujy)
units.fnu_to_njy(fnu)             # ... -> nanojansky
units.njy_to_fnu(njy)
units.fnu_to_maggies(fnu)         # ... -> SDSS maggies
units.maggies_to_fnu(maggies)
units.fnu_to_flambda(fnu, wave)   # F_ν -> F_λ (erg/s/cm²/Å)
units.flambda_to_fnu(flambda, wave)

# AB magnitudes
units.fnu_to_ab_mag(fnu)
units.ab_mag_to_fnu(mag_ab)
units.lnu_to_absolute_ab_mag(lnu)
units.absolute_ab_mag_to_lnu(mag_abs)

# L_ν ↔ F_ν at a given luminosity distance
units.lnu_to_fnu(lnu, d_lum_mpc, redshift)
units.fnu_to_lnu(fnu, d_lum_mpc, redshift)

# L_ν ↔ L_λ
units.lnu_to_llambda(lnu, wave)
units.llambda_to_lnu(llambda, wave)

# Absolute / apparent / surface brightness
units.absolute_to_apparent(M_abs, dm)
units.apparent_to_absolute(m_app, dm)
units.distance_modulus_from_dl_mpc(d_lum_mpc)
units.distance_modulus_from_dl(d_lum_cm)
units.cosmological_dimming(dm, redshift)   # dimming-corrected distance modulus
units.mag_to_surface_brightness(mag, area_arcsec2)
units.surface_brightness_to_mag(mu, area_arcsec2)

# Wavelength media
units.air_to_vacuum(wave_air)
units.vacuum_to_air(wave_vac)

# Energy
units.lsun_to_erg_per_s(lsun)
units.erg_per_s_to_lsun(ergs)

# Optical depth ↔ attenuation
units.tau_to_attenuation(tau)     # = exp(-tau)
units.attenuation_to_tau(att)
```

### Vega magnitudes

`ab_to_vega` and `vega_to_ab` take a **float offset**, not a band name —
`mag_Vega = mag_AB − offset`. The offsets ship as a dict:

```python
from tengri import units

units.ab_to_vega(20.0, units.AB_VEGA_OFFSETS["V"])   # -> 19.98
units.vega_to_ab(19.98, units.AB_VEGA_OFFSETS["V"])  # -> 20.0
```

Its keys are Johnson/Bessel and SDSS **short** band names, not
filter-registry ids:

```python
list(units.AB_VEGA_OFFSETS)
# ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'u', 'g', 'r', 'i', 'z']
```

So `AB_VEGA_OFFSETS["r"]`, never `AB_VEGA_OFFSETS["sdss_r"]` — there is no
filter-registry lookup for the offset, and a registry id raises `KeyError`.
Values are from Blanton & Roweis 2007, AJ, 133, 734 (Tables 3, 5).

The full list (about 35 helpers) is `tengri.units.__all__` —
`from tengri import units; help(units)` shows it.

## Why no `astropy.units`

Tengri is JAX-native and pre-1.0 research code. Wrapping every array in
an `astropy.units.Quantity` would (a) break JIT, since `Quantity` is
not a JAX type, (b) double the memory of every internal array, and
(c) introduce subtle conversion edge cases at every `vmap` boundary.

The convention here — *fixed unit per layer, conversion helpers at the
boundary* — has been stable since v0.1 and matches what working
SED-fitting codes (BAGPIPES, Prospector, FSPS) actually do under the
hood. If you need to interoperate with `astropy.units` at the user
layer, convert your inputs to plain arrays in tengri's units before
the fit and tag the outputs after.

## Common gotchas

- **`agn_log_lbol`** is *always* `log10(L_bol / L_sun)` at the API
  level. The AGN component converts to erg/s internally; user code
  should never multiply by 3.828e33 itself.
- **PSD timescale** is **Myr** as `psd_tau_myr` at the API level,
  **years** as `psd_tau_yr` internally. The internal-param map handles
  the `1e6` factor.
- **Metallicity offset.** `met_logzsol` is `log10(Z/Z☉)` (user) but
  the SSP grid is `log10(Z)` absolute. The translation adds
  `LOG10_ZSUN = -1.848`, defined in `tengri.utils.physics_constants`.
  Do not reproduce this constant by hand.
- **Emission lines** are vacuum throughout. `H_alpha = 6564.61 Å`,
  not 6562.8 Å (which is air).
- **All SED components return `erg/s/Hz`** (standardized
  2026-04-08). If you're implementing a component from another code,
  normalize to this unit before returning.
- **Per-band photometry units.** `predict_photometry` returns one F_ν
  value per filter band, in cgs — the bandpass-averaged F_ν. The exact
  weighting is the *filter-convolution convention*; see the dedicated
  section below.

## Photometric filter-convolution convention

This section is the **single ground truth** for how tengri turns an SED
into a broadband flux. Read it before touching any photometry code.

### The formula

The bandpass-averaged flux density through filter `b` is

```
            ∫ F_ν(λ) T_b(λ) w(λ) dλ
⟨F_ν⟩_b  =  ───────────────────────
              ∫ T_b(λ) w(λ) dλ
```

where `T_b(λ)` is the filter transmission and `w(λ)` is the **bandpass
weight** set by the convention. For observed-frame photometry the SED is
first redshifted (`λ_obs = (1+z) λ_rest`) and the result is scaled by
`(1+z) / (4π d_L²)` (the `lnu_to_fnu` factor) to go from rest-frame `L_ν`
to observed `F_ν`. The AB magnitude is `m_AB = −2.5 log₁₀(⟨F_ν⟩ / 3631 Jy)`;
equivalently the AB zero point enters as `AB₀ = 1.13492×10⁻¹³ L_⊙/Hz`
(3631 Jy at 10 pc), `m = −2.5 log₁₀(∫ F_ν T w dλ / (AB₀ ∫ T w dλ)) …`.

### The two conventions

| Convention | `w(λ)` | Detector model | Matches |
|---|---|---|---|
| **`bessell`** (default) | `1/λ` | photon-counting | DSPS, FSPS, sedpy, Prospector |
| **`energy`** | `1/λ²` | energy / flat-in-frequency | CIGALE, BAGPIPES |

- **`bessell`** is the photon-counting AB convention — the physically
  correct mean for photon-counting detectors (every optical/NIR CCD) and
  how the AB system is realized by surveys. `∫ F_ν T dλ/λ ÷ ∫ T dλ/λ`.
  This is the **default** and matches tengri's own SSP engine (DSPS).
- **`energy`** is the flat-in-frequency mean, `∫ F_ν T dν ÷ ∫ T dν =
  ∫ F_ν T dλ/λ² ÷ ∫ T dλ/λ²`. Use it to reproduce CIGALE/BAGPIPES.

The two agree exactly for a flat-`F_ν` source (the AB reference) and
diverge by **5–40 mmag**, band- and SED-slope-dependent, for real SEDs.
Pick the convention the observed catalog's fluxes were synthesized
with: optical/NIR broadband → `bessell`; CIGALE-reduced products →
`energy`.

> **History / correctness note.** Through 2026-05, tengri weighted `F_ν`
> by `λ` (not `1/λ`) — an f_λ→f_ν units transplant that matched neither
> convention and biased colors. It is fixed; `bessell` is now bit-identical
> to DSPS (pinned by `tests/crossval/test_filter_convention_parity.py`).

### Choosing and introspecting

```python
import tengri
tengri.list_filter_conventions()
# name     short_doc
# ───────  ─────────────────────────────────────────────────────────────────────────
# bessell  Photon-counting, weight 1/lambda (default; DSPS/FSPS/sedpy).
# energy   Energy-counting, weight 1/lambda^2 / flat-in-frequency (CIGALE/BAGPIPES).
# [2 results — filter_convention]
```

Like every other `list_*` verb it returns a `_RegistryTable`, not a dict
(unified across the `list_*` verbs), so index it by position or go through its helpers —
`.names()` for the bare names, `.to_dict("name")` if you want the mapping
this page used to show.

The convention is a **build-time** choice: it is baked into the
preintegrated `WavePrecomp` lookup table, so the exact path
(`approx=None`) and the precomputed path must use the same convention
(both default to `bessell`). The kernels
(`tengri.observation.photometry.compute_flux_density`,
`utils.grid_interp.preintegrate_grid`) take a `convention=` argument; the
effective/pivot wavelength and the Taylor dust moment are recomputed
consistently with `w(λ)`.

### References

- Oke, J. B. 1974, ApJS 27, 21 (AB system).
- Hogg, Baldry, Blanton & Eisenstein 2002, "The K correction",
  arXiv:astro-ph/0210394, Eq. 5 (photon-counting AB).
- Fukugita et al. 1996, AJ 111, 1748, Eq. 7 (FSPS's cited form).
- Bessell & Murphy 2012, PASP 124, 140 (photonic passbands, pivot λ).
- Hearin et al. 2023, "DSPS", arXiv:2112.06830 (`w = 1/λ`, `T_Q` = photon
  transmission probability).
- Boquien et al. 2019, A&A 622, A103 (CIGALE energy convention).
- Brown et al. 2016, AJ 152, 102 (interpreting broadband flux; photon vs
  energy).

## See also

- [`docs/dev/NAMING_CONTRACT.md`](https://github.com/suchethac/tengri/blob/main/docs/dev/NAMING_CONTRACT.md) —
  parameter naming rules including unit suffixes (`_myr`, `_gyr`,
  `_kms`) for user-facing parameters.
- [`tengri.utils.physics_constants`](https://github.com/suchethac/tengri/blob/main/src/tengri/utils/physics_constants.py) —
  the canonical source for `LOG10_ZSUN`, `L_SUN`, etc.
- [`tengri.cosmology`](https://github.com/suchethac/tengri/blob/main/src/tengri/cosmology) —
  Planck18 distance / age helpers for redshift conversions.
