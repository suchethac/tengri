# FSPS MIST Chabrier mass-remaining table provenance

The per-(age, metallicity) surviving stellar mass fraction for FSPS MIST
isochrones and Chabrier IMF. `load_ssp_data` attaches this table to an SSP grid
that carries no `ssp_mass_remaining` when the grid is a python-fsps product
(`fsps_*` / `ssp_*`) on MIST isochrones with a Chabrier IMF and its age and
metallicity nodes equal the table's; every other grid gets the
metallicity-independent DSPS fit.

**Source**: `data/fsps_mass_remaining_chabrier.h5` (tracked in the repository)

**Attributes**:
- `imf`: Chabrier (2003)
- `isochrone`: mist
- `spectral_library`: miles

**Format**: Plain text, 14 rows total. Row 1: `log_age_yr` (107 log10-age nodes in years); Row 2: `z_absolute` (12 absolute metallicity nodes); Rows 3–14: `mass_remaining[i_z, :]` (surviving fraction at each age, for each metallicity). All values formatted as `%.17g` for exact round-trip.
