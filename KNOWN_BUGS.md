# Known Bugs

## ADAF synchrotron self-absorption spectral index

**File:** `src/tengri/models/agn/disc.py`, function `adaf_disc`, line ~1062

**Bug:** The self-absorbed synchrotron spectrum uses `nu^{5/2}` (the standard result for a power-law electron energy distribution). Mahadevan (1997, ApJ 477, 585) derives the self-absorbed spectrum for the thermal ADAF electrons using the Rayleigh-Jeans approximation (Eq. 19, 23), which gives `nu^2`. The `nu^{5/2}` exponent is appropriate for non-thermal power-law electrons (e.g., Rybicki & Lightman Ch. 6) but not for the mildly relativistic thermal electrons in an ADAF.

**Impact:** Affects only the ADAF backend at frequencies below the synchrotron self-absorption frequency (typically radio). The difference between nu^2 and nu^{5/2} is modest for broadband SED fitting since the self-absorbed regime is narrow, but the spectral shape in the radio is incorrect.

**Fix:** Change `sync_thick = nu_ratio_sa**2.5` to `sync_thick = nu_ratio_sa**2.0` and verify against Mahadevan 1997 Fig. 1.

---

## Bell 2003 synchrotron suppression formula

**File:** `src/tengri/models/radio.py`, function `_synchrotron_suppression`

**Bug:** The suppression formula `L_corr = L / (1 + (L_0/L)^2)` with `L_0 = 3e28 erg/s/Hz` does not match Bell (2003, ApJ 586, 794). Bell's actual model (Eq. 3) parameterizes the non-thermal fraction `n` as a function of galaxy luminosity:

```
n = 0.9                  for L > L*
n = 0.9 * (L/L*)^{0.3}  for L <= L*
```

where L* corresponds to M_V = -21. The total radio luminosity then follows from the non-thermal fraction: `L_radio = L_thermal + n * L_nonthermal`.

The `1/(1+(L_0/L)^2)` functional form may originate from a different reference or be a convenient smooth approximation. It captures the same qualitative behavior (suppression at low L) but has a different shape.

**Impact:** Affects radio luminosity predictions for low-SFR galaxies. The Bell (2003) parameterization gives a gentler suppression (`L^{0.3}` scaling) compared to the `L^2` suppression in the current code.

**Fix:** Replace with Bell's Eq. 3 non-thermal fraction parameterization, or document the actual source of the current formula.

---

## ADAF electron temperature approximation

**File:** `src/tengri/models/agn/disc.py`, function `adaf_disc`, line ~1044

**Note (not strictly a bug):** The electron temperature `T_e = 5e9 * sqrt(delta / mdot)` is a rough scaling approximation. Mahadevan (1997) solves T_e self-consistently from the coupled ion-electron energy balance (Section 5.1). The approximate scaling captures the qualitative behavior but may differ from the self-consistent solution by factors of a few at extreme accretion rates. This is acceptable for a broadband SED model but should be documented.
