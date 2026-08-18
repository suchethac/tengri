# SPDX-License-Identifier: BSD-3-Clause
"""Fundamental physical constants in CGS units.

All constants are defined with explicit SI→CGS derivations so the origin of
every numerical value can be audited against the primary reference.

Sources
-------

- **CODATA 2018** (NIST): https://physics.nist.gov/cuu/Constants/
  h, k_B, c, G, σ_T, m_p, m_e are exact or CODATA-2018 recommended.
  Note: h, k_B, and c are *exact* in SI since the 2019 SI redefinition.
- **IAU 2015 nominal solar values** (IAU 2015 B3 resolution):
  L_sun = 3.828e26 W (exact nominal value).
- **IAU 2012** (IAU 2012 B2 resolution):
  1 pc = 648 000 / π AU (exact nominal value by definition).
  1 AU = 1.495 978 707e11 m (exact).

Unit system
-----------
All exported values are in CGS (cm, g, s, erg). The derivation comment on
each line shows::

    SI value  ×  [SI → CGS conversion]  →  CGS value

Naming conventions
------------------

- ``H_PLANCK``: Planck's constant h
- ``K_BOLTZ``: Boltzmann constant k_B
- ``C_CGS``: speed of light c in cm/s
- ``C_AA``: speed of light c in Å/s (used for ν = c/λ with λ in Å)
- ``C_KM_S``: speed of light c in km/s (for Doppler calculations)
- ``SIGMA_SB``: Stefan-Boltzmann constant σ
- ``G_GRAV``: gravitational constant G
- ``SIGMA_T``: Thomson cross section σ_T
- ``M_PROTON``: proton mass m_p
- ``M_ELECTRON``: electron mass m_e
- ``M_SUN``: solar mass M_⊙
- ``L_SUN``: solar luminosity L_⊙ (IAU 2015 nominal)
- ``PC_CM``: parsec in cm
- ``AA_TO_CM``: Ångström → cm conversion factor
- ``KEV_TO_ERG``: keV → erg energy conversion
- ``KEV_TO_HZ``: keV → Hz frequency conversion (E = hν)
- ``K_BOLTZ_KEV``— k_B in keV/K (for temperature → seed photon energy)

Special constants
-----------------
``L_SUN_CUE = 3.839e33`` is the solar luminosity used by the CUE neural-net
training convention.  It must *only* be used in ``nebular/cue.py`` and must
never replace ``L_SUN`` elsewhere.  See Note [1] below.

Notes
-----
[1] CUE (Tacchella+2022) was trained with L_sun = 3.839e33 erg/s (the older
    Bahcall & Soneira 1980 / Allen's Astrophysical Quantities value), not the
    IAU 2015 value 3.828e33.  All CUE outputs are implicitly in that convention,
    so the un-normalization step in cue.py must use 3.839e33 to reproduce the
    correct line fluxes.  Changing it to the IAU value would introduce a 0.3%
    systematic offset in all CUE-predicted line luminosities.
"""

from __future__ import annotations

import math

# ── Speed of light ────────────────────────────────────────────────

C_CGS: float = 2.99792458e10
"""Speed of light c [cm s⁻¹].

Derivation: c = 2.99792458e8 m/s (exact, SI 2019)
            × 100 cm/m
            = 2.99792458e10 cm/s
"""

C_AA: float = 2.99792458e18
"""Speed of light c [Å s⁻¹].

Derivation: c = 2.99792458e8 m/s (exact)
            × 1e10 Å/m
            = 2.99792458e18 Å/s

Used for ν = c/λ when λ is in Ångström.
"""

C_KM_S: float = 2.99792458e5
"""Speed of light c [km s⁻¹].

Derivation: c = 2.99792458e8 m/s (exact)
            × 1e-3 km/m
            = 2.99792458e5 km/s

Used for Doppler-shift and line-width calculations.
"""

# ── Radio continuum frequencies ───────────────────────────────────

WAVE_1P4GHZ_AA: float = 21.41374e8
"""1.400 GHz rest-frame wavelength [Å].

Derivation: λ = c / ν
            = 2.99792458e8 m/s / 1.400e9 Hz
            = 0.214137... m
            = 21.4137... × 1e8 Å
            ≈ 21.41374e8 Å

This is the standard continuum frequency for FIR-radio correlations (Bell 2003,
Delvecchio+2021) and NVSS radio surveys. NOT the 21 cm HI line (1.4204 GHz /
21.106 cm). See issue #1057 for the distinction.

Used to extract monochromatic radio luminosity L_1.4GHz from rest-frame
SED grids: ``L_1p4ghz = jnp.interp(WAVE_1P4GHZ_AA, wave, L_radio)``.
"""

# ── Planck's constant ─────────────────────────────────────────────

H_PLANCK: float = 6.62607015e-27
"""Planck constant h [erg s].

Derivation: h = 6.62607015e-34 J s (exact, CODATA 2018 / SI 2019)
            × 1e7 erg/J
            = 6.62607015e-27 erg s
"""

# ── Boltzmann constant ────────────────────────────────────────────

K_BOLTZ: float = 1.380649e-16
"""Boltzmann constant k_B [erg K⁻¹].

Derivation: k_B = 1.380649e-23 J K⁻¹ (exact, CODATA 2018 / SI 2019)
            × 1e7 erg/J
            = 1.380649e-16 erg K⁻¹
"""

K_BOLTZ_KEV: float = 8.617333262e-8
"""Boltzmann constant k_B [keV K⁻¹].

Derivation: k_B = 1.380649e-23 J K⁻¹ (exact)
            / 1.602176634e-16 J/keV  (exact)
            = 8.617333262e-8 keV K⁻¹

Used to convert ring/corona temperature → seed photon energy for
nthcomp (Comptonization) models.
"""

# ── Stefan-Boltzmann constant ─────────────────────────────────────

SIGMA_SB: float = 5.670374419e-5
"""Stefan-Boltzmann constant σ [erg cm⁻² s⁻¹ K⁻⁴].

Derivation: σ = 5.670374419e-8 W m⁻² K⁻⁴  (derived from exact h, k_B, c)
            × 1e7 erg/J       (W → erg/s)
            / 1e4 cm²/m²      (m⁻² → cm⁻²)
            = 5.670374419e-5 erg cm⁻² s⁻¹ K⁻⁴

Reference: Eq. σ = 2π⁵k_B⁴/(15h³c²).  Numerically exact from CODATA 2018.
"""

# ── Gravitational constant ────────────────────────────────────────

G_GRAV: float = 6.674e-8
"""Gravitational constant G [cm³ g⁻¹ s⁻²].

Derivation: G = 6.674e-11 m³ kg⁻¹ s⁻²  (CODATA 2018)
            × 1e6 cm³/m³
            / 1e3 g/kg
            = 6.674e-8 cm³ g⁻¹ s⁻²

Note: CODATA 2018 gives G = 6.67430(15)e-11 m³ kg⁻¹ s⁻².  The value 6.674e-8
used here is rounded to 4 sig. fig., sufficient for accretion disc calculations
where M_BH and spin dominate the uncertainty.
"""

# ── Thomson cross section ─────────────────────────────────────────

SIGMA_T: float = 6.6524e-25
"""Thomson cross section σ_T [cm²].

Derivation: σ_T = 6.6524587158e-29 m²  (CODATA 2018)
            × 1e4 cm²/m²
            = 6.6524587158e-25 cm²  ≈ 6.6524e-25 cm²

Used in Eddington luminosity: L_Edd = 4πGMm_p c / σ_T.
"""

# ── Particle masses ───────────────────────────────────────────────

M_PROTON: float = 1.6726e-24
"""Proton mass m_p [g].

Derivation: m_p = 1.67262192369e-27 kg  (CODATA 2018)
            × 1e3 g/kg
            = 1.67262192369e-24 g  ≈ 1.6726e-24 g

Used in Eddington luminosity and accretion efficiency calculations.
"""

M_ELECTRON: float = 9.1094e-28
"""Electron mass m_e [g].

Derivation: m_e = 9.1093837015e-31 kg  (CODATA 2018)
            × 1e3 g/kg
            = 9.1093837015e-28 g  ≈ 9.1094e-28 g
"""

E_CHARGE_ESU: float = 4.80326e-10
"""Elementary charge e [statcoulomb (esu)].

Derivation: e = 1.602176634e-19 C  (exact, SI 2019)
            × 2.99792458e9 statC/C  (SI → Gaussian CGS)
            = 4.80326e-10 statC

Used in Ly-alpha cross-section for DLA absorption models:
σ_α = (√π e² f) / (m_e c Δν_D) × H(a, x).
"""

# ── Solar constants ───────────────────────────────────────────────

L_SUN: float = 3.828e33
"""Solar luminosity L_⊙ [erg s⁻¹].

Derivation: L_⊙ = 3.828e26 W  (IAU 2015 B3 nominal solar luminosity, exact)
            × 1e7 erg/J   (W → erg/s)
            = 3.828e33 erg s⁻¹

Reference: IAU 2015 Resolution B3, https://www.iau.org/static/resolutions/IAU2015_English.pdf
"""

M_SUN: float = 1.989e33
"""Solar mass M_⊙ [g].

Derivation: M_⊙ = 1.98892e30 kg  (IAU standard gravitational parameter
                  GM_⊙ = 1.32712440018e20 m³ s⁻² divided by G)
            × 1e3 g/kg
            = 1.98892e33 g  ≈ 1.989e33 g

Note: The IAU 2015 nominal solar mass parameter is
      (GM)_⊙,N = 1.3271244e20 m³ s⁻².  At G = 6.674e-11, M_⊙ = 1.9885e30 kg.
"""

L_SUN_CUE: float = 3.839e33
"""CUE-convention solar luminosity [erg s⁻¹].

L_⊙ = 3.839e33 erg/s  (older Allen / Bahcall-Soneira convention used by the
CUE neural-net training set).

**Only use this constant in nebular/cue.py.**  See module-level Note [1].
Do NOT use as a general-purpose solar luminosity.
"""

Z_SUN: float = 0.0142
"""Solar metallicity Z_⊙ [dimensionless mass fraction].

Reference: Asplund et al. (2009), ARA&A 47, 481, "The Chemical Composition of
the Sun", https://doi.org/10.1146/annurev.astro.46.060407.145222

This is the photospheric present-day value, and matches the MIST/DSPS
convention.  Other SSP libraries adopt different solar scales — BC03/Padova
= 0.0190, PARSEC = 0.0152, BASTI = 0.0200.  Per-library values live in
``LOG10_ZSUN_BY_LIBRARY`` (:mod:`tengri.parameters.translate`); use those,
not this constant, when reproducing another code bit-exactly.
"""

LOG10_ZSUN: float = math.log10(Z_SUN)
"""log₁₀ of the solar metallicity, log₁₀(Z_⊙) [dex] = -1.8477116556169435.

Derivation: log₁₀(Z_⊙) = log₁₀(0.0142) = -1.8477116556169435  (exact in
            IEEE-754 double; round-trips back to Z_SUN exactly).

Computed from :data:`Z_SUN` rather than hardcoded so the two cannot drift
apart.  The SSP grids are tabulated in **absolute** log₁₀(Z), while the
user-facing ``met_logzsol`` and ``neb_logZ_gas`` are log₁₀(Z/Z_⊙); this
constant is the offset between them (``param_map`` adds it).

Reference: Asplund et al. (2009), ARA&A 47, 481,
https://doi.org/10.1146/annurev.astro.46.060407.145222
"""

# ── Distances ─────────────────────────────────────────────────────

PC_CM: float = 3.0856775814913674e18
"""Parsec in cm.

Derivation:
    1 AU     = 1.495 978 707e11 m  (exact, IAU 2012 B2)
    1 pc     = 648 000 / π  AU     (exact, by definition)
             = 3.085 677 581 491...e16 m
    × 100 cm/m
             = 3.085 677 581 491...e18 cm

Used for flux ↔ luminosity conversions:
    f_ν = L_ν / (4π d²)  with d in cm.
"""

# ── Unit conversion factors ───────────────────────────────────────

AA_TO_CM: float = 1.0e-8
"""Ångström to cm conversion.

Derivation: 1 Å = 1e-10 m × 100 cm/m = 1e-8 cm  (exact by definition)

Used throughout the code as ``wavelength_cm = wavelength_aa * AA_TO_CM``.
"""

KEV_TO_ERG: float = 1.6022e-9
"""keV to erg energy conversion.

Derivation: 1 eV  = 1.602176634e-19 J  (exact, SI 2019)
            1 keV = 1.602176634e-16 J
            × 1e7 erg/J
            = 1.602176634e-9 erg  ≈ 1.6022e-9 erg

Used for X-ray photon energy grids:
    E_erg = E_keV * KEV_TO_ERG
"""

KEV_TO_HZ: float = 2.41799e17
"""keV to Hz frequency conversion  (E = hν → ν = E/h).

Derivation: ν [Hz] = E [erg] / h [erg s]
            = (1.6022e-9 erg/keV) / (6.62607015e-27 erg s)
            = 2.41799e17 Hz/keV

Used for X-ray SED grids:
    nu_hz = energy_keV * KEV_TO_HZ
"""

V_BAND_ANGSTROM: float = 5500.0
"""V-band central wavelength [Å].

Canonical wavelength for optical normalization, used in dust attenuation
laws and SED spectral shape calculations. All attenuation curves k(λ)
are normalized so that k(5500 Å) = 1.

Used in dust attenuation normalization and optical flux calculations.
"""

# ── AB magnitude zeropoint ────────────────────────────────────────

JY_CGS: float = 1e-23
"""Jansky in CGS [erg s⁻¹ cm⁻² Hz⁻¹].

Derivation: 1 Jy = 10⁻²⁶ W m⁻² Hz⁻¹  (IAU definition)
            × 1e7 erg/J / 1e4 cm²/m²
            = 10⁻²³ erg s⁻¹ cm⁻² Hz⁻¹

Used for flux density conversions: f_ν [CGS] = f_ν [Jy] × JY_CGS.
"""

MAGGIES_ZP_CGS: float = 3.631e-20
"""AB magnitude zeropoint flux [erg s⁻¹ cm⁻² Hz⁻¹].

Derivation: AB system is defined so that m_AB = 0 corresponds to
            f_ν = 3631 Jy = 3631 × 10⁻²³ erg s⁻¹ cm⁻² Hz⁻¹
            = 3.631e-20 erg s⁻¹ cm⁻² Hz⁻¹

1 maggie ≡ flux of a zero-magnitude AB source = 3631 Jy.
Reference: Oke & Gunn 1983, ApJ, 266, 713.
"""

MPC_CM: float = 3.0856775814913674e24
"""Megaparsec in cm.

Derivation: 1 Mpc = 10⁶ pc = 10⁶ × 3.0856775814913674e18 cm
            = 3.0856775814913674e24 cm

Used for cosmological distance conversions (DSPS returns Mpc).
"""

TEN_PC_CM: float = 3.0856775814913674e19
"""10 parsec in cm (absolute magnitude standard distance).

Derivation: 10 pc = 10 × 3.0856775814913674e18 cm
            = 3.0856775814913674e19 cm

Used for absolute magnitude: M = m_AB(f_ν at 10 pc).
"""

AB_ZEROPOINT: float = -48.6
"""AB magnitude zeropoint [mag].

Definition: m_AB = -2.5 log10(f_ν [erg s⁻¹ cm⁻² Hz⁻¹]) - 48.6

Reference: Oke & Gunn 1983, ApJ, 266, 713.
The zeropoint corresponds to f_ν = 3.631e-20 erg s⁻¹ cm⁻² Hz⁻¹ = 3631 Jy
at all wavelengths.
"""
