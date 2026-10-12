# MIST top-metallicity nodes (opt-in `top_z="mist"`)

`mist_top_node_miles_chabrier.h5` and `mist_top_node_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5`
hold one simple stellar population each: FSPS with the MIST isochrones at its highest metallicity
node, [Z/H] = +0.5, on the MILES spectral library and the Chabrier IMF, at the 93 ages and 5994
wavelengths of the PARSEC grids `fsps_prsc_miles_chabrier` and
`ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0`.

**Why.** The Z = 0.06 node of the PARSEC grids has no stars above 12.01 Msun for log age <= 7.15
(the PARSEC massive-star tracks stop at Z = 0.04; Chen et al. 2015, MNRAS 452, 1068,
arXiv:1506.01681). `load_ssp_data(..., top_z="mist")` inserts this node and uses it as the clip
target for that part of the grid. See `components/stellar/sps/top_z_validity.py`.

**Metallicity.** Z = 0.0142 x 10^0.5 = 0.04490 (FSPS `zlegend` entry 0.04490434). MIST defines
[Z/H] against the protosolar Z_sun = 0.0142 (Asplund et al. 2009) and takes [Z/H] = [Fe/H] for its
solar-scaled models; Choi et al. 2016, ApJ 823, 102 (arXiv:1604.08592, Sect. 1 and 3.1, Table 1:
X = 0.7154, Y = 0.2703, Z = 0.0142). The node is labeled with this Z, never 0.06. The top MIST
node is [Z/H] = +0.5 (Choi et al. 2016, abstract; Dotter 2016, ApJS 222, 8, arXiv:1601.05144).

**Surviving mass.** `mass_remaining` is `StellarPopulation.stellar_mass` of the same build
(`add_stellar_remnants=1`, Renzini & Ciotti 1993), so spectra and mass share isochrones, IMF and Z.

**Build.** `scripts/build_mist_top_z_node.py`, run with the shared environment's python-fsps 0.4.7
(compiled with MIST and MILES, `libraries` = mist, miles, DL07; the isochrone set is a compile-time
choice and the default build already is MIST, so no recompile was needed). Nebular emission for the
wNE node is FSPS `add_neb_emission=True, gas_logu=-3.0, gas_logz=0.0`. The same call under a PARSEC
build reproduces the shipped PARSEC grids' Z = 0.04 node: max relative difference 1.2e-7 (bare) and
1.0e-7 (wNE). Fluxes are in FSPS's native Lsun/Hz per Msun formed, like the grid files.

**Not shipped.** Other spectral libraries (C3K, BaSeL) have no node: `top_z="mist"` refuses them by
name.
