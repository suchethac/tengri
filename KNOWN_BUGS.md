# Known Bugs

All physics bugs in this file have been fixed. See `docs/known_bugs.md` for
the full audit history.

---

## ADAF synchrotron self-absorption spectral index — FIXED

**File:** `src/tengri/components/agn/disc.py`, `adaf_disc`

**Was:** `sync_thick = nu_ratio_sa**2.5` — incorrect exponent for non-thermal
power-law electrons, not the thermal ADAF case.

**Fix (applied):** Changed to `nu_ratio_sa**2.0` per Mahadevan (1997, ApJ
477, 585) Eq. 19/23: Rayleigh-Jeans self-absorbed regime of thermal electrons
gives `ν²`.

---

## Bell 2003 synchrotron suppression formula — FIXED

**File:** `src/tengri/components/radio/radio.py`, `_synchrotron_suppression`

**Was:** `L / (1 + (L₀/L)²)` — quadratic suppression, not matching Bell.

**Fix (applied):** Replaced with Bell (2003, ApJ 586, 794) Eq. 3 piecewise
non-thermal fraction: `n = 0.9` for `L > L*`; `n = 0.9 * (L/L*)^{0.3}` for
`L ≤ L*`, where `L*` is the 1.4 GHz luminosity corresponding to M_V = −21.

---

## ADAF electron temperature approximation — NOTE (not a bug)

**File:** `src/tengri/components/agn/disc.py`, `adaf_disc`

`T_e = 5e9 * sqrt(delta / mdot)` is a rough scaling approximation.
Mahadevan (1997) Section 5.1 solves T_e self-consistently from ion-electron
energy balance. The approximation is acceptable for broadband SED fitting.
