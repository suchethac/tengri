# Mass Remaining Tables: Data Attribution

## BC03 (Padova 1994 + STELIB + Chabrier IMF)

**Source:** Bruzual & Charlot (2003) BC03 models, Updated version (2016)

**Reference:** Bruzual, G., & Charlot, S. (2003). Stellar population synthesis at arbitrary metallicity. MNRAS, 344(4), 1000-1028.

**Distribution terms:** The BC03 model outputs are made available by the authors (G. Bruzual & S. Charlot) for scientific use under the terms of proper attribution. Redistributed data must cite the original paper (Bruzual & Charlot 2003, MNRAS 344, 1000) and acknowledge the authors.

**Data description:** The mass_remaining table extracts the surviving stellar mass fraction (living stars + remnants per unit formed mass) from BC03's pre-computed *.4color files. This quantity represents the fraction of a single-age population that persists as living stars plus stellar remnants (white dwarfs, neutron stars, black holes) at a given age, computed using BC03's internal remnant prescription (based on the Padova 1994 isochrones and standard stellar evolution models).

**Isochrones:** Padova 1994 (Bertelli et al. 1994)
**Spectral library:** STELIB (Le Borgne et al. 2003)
**IMF:** Chabrier lognormal + power law
**Metallicity grid:** 6 points (m22, m32, m42, m52, m62, m72) corresponding to log10(Z) = -4.0, -3.398, -2.398, -2.097, -1.699, -1.301
**Age grid:** 220 points (log10 age yr = 5.0 to 10.301) interpolated to Tengri's 221-point grid

**Attribution in use:** When stellar_mass_surviving or related quantities are used in publications, cite:
- Bruzual, G., & Charlot, S. (2003). MNRAS, 344, 1000-1028.
- Note: "Surviving stellar mass fractions from BC03 Padova 1994 + STELIB + Chabrier IMF models"

**Files:**
- `mass_remaining_bc03pdva94_chabrier.h5` — HDF5 table with log10_age_yr, log10_z_abs, mass_remaining datasets
- `../build_mass_remaining_bc03.py` — Generator script (reproducible from BC03 source)
