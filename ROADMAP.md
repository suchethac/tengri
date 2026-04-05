# Tengri Roadmap

Planned physics modules not yet implemented. These are tracked here rather
than in the Sphinx docs to avoid documenting unimplemented features.

## Planned Physics Modules

### Chemical Evolution Z(t)
Time-evolving metallicity Z(t) coupled to the star formation history.
Motivation: breaks the age-metallicity degeneracy in old stellar populations.
See `docs/dev/roadmap/chemical_evolution.md` for detailed specifications.

### Shock Emission (MAPPINGS)
Radiative shock emission from MAPPINGS V grids.
Motivation: critical for AGN-host and starburst-driven outflow diagnostics.
See `docs/dev/roadmap/shock_emission.md` for detailed specifications.

### ADAF Disc
Advection-dominated accretion flow model for low-luminosity AGN (Mahadevan 1997).
Motivation: needed for quiescent black holes below the AGN threshold.
See `docs/dev/roadmap/adaf_disc.md` for detailed specifications.

### MAGPHYS-style Dust
Energy-balance dust model from da Cunha+2008.
Motivation: alternative to DL07 for high-z submillimetre-selected sources.
See `docs/dev/roadmap/magphys_dust.md` for detailed specifications.

### THEMIS Dust
Jones+2017 dust grain size distribution with amorphous carbon.
Motivation: physically motivated alternative to silicate/graphite mixtures.
See `docs/dev/roadmap/themis_dust.md` for detailed specifications.

### Patchy IGM Reionization
Neutral hydrogen bubble attenuation at z > 6 (two free parameters: x_HI, R_bubble).
Motivation: required for Lyman-alpha transmission statistics at cosmic dawn.
See `docs/dev/roadmap/patchy_igm.md` for detailed specifications.

### PAH Emission Features
Mid-infrared PAH complex (6.2, 7.7, 8.6, 11.3 µm).
Motivation: strong star formation diagnostic for JWST/MIRI observations.
See `docs/dev/roadmap/pah_features.md` for detailed specifications.

### Astrodust + PAH Dust Model
Alternative to DL07 using parameterized PAH fraction and silicate/graphite mix.
Motivation: finer control over composition for emission line fitting.
See `docs/dev/roadmap/astrodust.md` for detailed specifications.

### BOSA Dust Emission Templates
Black-body adjusted spectral energy distribution dust templates (BOSA).
Motivation: fast empirical dust emission without full radiative transfer.
See `docs/dev/roadmap/bosa_templates.md` for detailed specifications.

### TEA Dust Attenuation Model
Infrared excess attenuation model for Milky Way environments.
Motivation: improved UV-to-IR energy balance consistency.
See `docs/dev/roadmap/tea_attenuation.md` for detailed specifications.
