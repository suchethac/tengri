# AGN Extinction Curve Provenance

## qsogen_quasar_ext.dat

**Original paper**: Temple, M. J., Hewett, P. C., & Banerji, M. 2021, MNRAS,
508, 737. DOI: [10.1093/mnras/stab2586](https://doi.org/10.1093/mnras/stab2586).
arXiv: [2109.04472](https://arxiv.org/abs/2109.04472).

**What it is**: the empirically-derived *quasar* dust extinction curve used by
the qsogen SED model (<https://github.com/MJTemple/qsogen>, file
`pl_ext_comp_03.sph`). Temple+2021 §2.4 derive it from SDSS DR7 quasars at
2.0 < z < 3.0: i- and K-band photometry set the rest-frame optical
(3000–6500 Å) reddening E(B−V), and composite spectra of samples binned by
reddening define the ultraviolet (1100–2500 Å) shape. It is *not* the SMC
Prevot law; the paper notes it is "somewhat similar to those presented by
Czerny et al. (2004) and Gallerani et al. (2010)".

**Convention**: the file stores **E(λ−V)/E(B−V)** as a function of wavelength
(so the value at V = 5500 Å is 0). qsogen applies it as

    A_λ = E(B−V) · [ E(λ−V)/E(B−V) + R ],   flux → flux · 10^(−A_λ/2.5)

with R settable (`qsosed.redden_spectrum(R=3.1)`, default 3.1). This is a
different convention from AGNfitter's `BBBred_Prevot`, which stores an analytic
SMC fit directly as A_λ/E(B−V).

**Source**: reproduced from qsogen's own `reddening_curve` output (the
`pl_ext_comp_03.sph` content, columns `[wavelength/Å, E(λ−V)/E(B−V)]`),
log-resampled to 1200 points spanning 500–60000 Å. Maximum deviation from the
source sampling is 0.0024 mag/mag (at 1950 Å; median 0.0), i.e. < 0.03 % of the
local A_λ/E(B−V). The curve is monotonic through 2175 Å (no MW-like bump).

**Output file**: `qsogen_quasar_ext.dat`, plain text two-column, sha256
`1fba0911a2a8d12de02531a153ffcadacd44a6ab8cbd46b590bdd4078efa30ed`. The
numerical data is the Temple+2021 quasar extinction curve itself, not qsogen
code. Tengri ships it under BSD-3-Clause; the underlying scientific curve is
attributable to Temple+2021.
