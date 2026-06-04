"""Build the GRAHSP HDF5 data bundle from upstream raw template files.

Outputs ``data/grahsp/grahsp_templates.h5`` containing:

- ``feii_bruhweiler2008/wave_nm`` (n,) — Bruhweiler+Verner 2008 FeII forest
  template wavelengths in nm (de-redshifted from the upstream catalog
  default of z=0.004, since the raw CIGALE database stores the de-redshifted
  template; see paper §2.1.2).
- ``feii_bruhweiler2008/lumin`` (n,) — relative intensity, normalised so
  that the integrated H-beta-equivalent luminosity scaling matches upstream.
- ``netzer1990_lines/wave_nm`` (n_lines,) — central wavelengths in nm.
- ``netzer1990_lines/broad`` (n_lines,) — broad-line strengths relative to
  H-beta (broad).
- ``netzer1990_lines/narrow_sy2`` (n_lines,) — Sy2 narrow-line strengths
  relative to H-beta (narrow).
- ``netzer1990_lines/narrow_liner`` (n_lines,) — LINER narrow-line strengths.
- ``netzer1990_lines/name`` (n_lines,) — line names (UTF-8).
- ``torus/wave_nm`` (n_torus,) — fixed wavelength grid used by the upstream
  ``activategtorus`` module (see source for provenance).

Run::

    .venv/bin/python tools/build_grahsp_hdf5.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from tools.generate_grahsp_fixtures import (
    _load_full_torus_wave,
    parse_mor_netzer_lines,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "grahsp"
OUT = REPO_ROOT / "data" / "grahsp" / "grahsp_templates.h5"

FEII_RAW = RAW / "feii_bruhweiler2008_d11_m20_20p5.txt"
LINES_RAW = RAW / "mor_netzer_2012_emission_lines.txt"
TORUS_MN12_RAW = RAW / "mor_netzer_2012_torus_extended.txt"
MULLANEY_RAW = RAW / "mor_netzer_2012_mullaney.txt"
FEII_VC04_RAW = RAW / "feii_veroncetty2004.txt"
DISC_RAW = RAW / "netzer_disc_models.txt"

# H-beta normalisation constant: upstream uses L(line)/L(5100Å) ratios where
# L(5100) = lambda*L_lambda. See database_builder/activate/agn/mor_netzer_2012/readme.
HBETA_BROAD_RATIO = 0.02  # L(Hb_broad) / L(5100)
HBETA_NARROW_RATIO = 0.002  # L(Hb_narrow) / L(5100)

# Arbitrary speed-of-light constant. The disc/torus templates store *shapes*
# only — each is renormalised to 1 at a reference wavelength — so the units of
# c cancel. Mirrors ``c = 3.0e18`` in upstream ``database_builder/__init__.py``.
_C_ARB = 3.0e18

# Netzer disc course-grid model ordering, mirroring upstream
# ``[(Mv, av, Mdotv) for av in a for Mdotv in Mdot for Mv in M]`` (16 models).
_DISC_M = ["6.0", "7.0", "8.0", "9.0"]
_DISC_A = ["0.998", "0"]
_DISC_MDOT = ["0.3", "0.03"]
_DISC_HEADER_LINES = 23  # rows to skip before the freq/wave/L_nu table


def _load_mn12_torus():
    """Load the Mor & Netzer 2012 template torus (avg/lo/hi + silicate).

    Mirrors the ``MorNetzer2012Torus`` import block of upstream
    ``database_builder/__init__.py``.

    The ``mor_netzer_mean_and_uncertainty_extended`` columns are wavelength
    [micron] then arbitrary :math:`\\nu L_\\nu` for the mean (avg), 25th-
    percentile (lo) and 75th-percentile (hi) intrinsic AGN SEDs. Each is
    converted to :math:`L_\\lambda \\propto \\nu L_\\nu / \\lambda`, renormalised
    to 1 at 12 µm, and flat-extrapolated below 2 µm (the model is valid above
    ~1 µm). The silicate feature is the Mullaney+2011 difference spectrum,
    isolated over 7–19 µm and normalised by the 12 µm continuum.

    Returns
    -------
    wave_nm, avg, lo, hi : ndarray
        Torus continuum grid [nm] and the three normalised :math:`L_\\lambda`
        templates.
    si_wave_nm, si_lumin : ndarray
        Silicate-feature grid [nm] and normalised :math:`L_\\lambda`.
    """
    data = np.genfromtxt(TORUS_MN12_RAW)
    wave_nm = data[:, 0] * 1000.0  # micron -> nm
    freq = _C_ARB / wave_nm
    norm_mask = wave_nm == 12000.0
    out = []
    for col in (1, 2, 3):  # avg, lo, hi
        nu_lnu = data[:, col]
        l_lambda = nu_lnu / freq * _C_ARB / wave_nm**2
        norm = l_lambda[norm_mask][0]
        l_lambda = l_lambda / norm
        # Model is valid above ~1 µm; pick above 2 µm and flat-extrapolate.
        valid = wave_nm > 2000.0
        l_lambda[~valid] = l_lambda[valid][0]
        out.append(l_lambda)
    avg, lo, hi = out

    # Mullaney+2011 silicate difference spectrum.
    mdata = np.genfromtxt(MULLANEY_RAW)
    swave_nm = mdata[:, 0] * 1000.0
    sfreq = _C_ARB / swave_nm
    i18 = np.where(swave_nm > 18000.0)[0][0]
    spec_a = mdata[:, 2]
    spec_b = mdata[:, 3] * (mdata[i18, 2] / mdata[i18, 3])  # equalise at 18 µm
    cont = (spec_a + spec_b) / 2.0 * sfreq
    cont_llam = cont / sfreq * _C_ARB / swave_nm**2
    cont_norm = cont_llam[swave_nm == 12000.0][0]
    diff = spec_a - spec_b
    diff[np.logical_and(diff < 0, swave_nm < 8000.0)] = 0.0
    diff[np.logical_and(diff < 0, swave_nm > 18000.0)] = 0.0
    diff = diff * sfreq
    si_llam = diff / sfreq * _C_ARB / swave_nm**2
    si_mask = np.logical_and(swave_nm > 7000.0, swave_nm < 19000.0)
    return wave_nm, avg, lo, hi, swave_nm[si_mask], si_llam[si_mask] / cont_norm


def _load_veroncetty_feii():
    """Load and normalise the Veron-Cetty+2004 FeII template (no de-redshift).

    Mirrors upstream FeII import for ``'Veron-Cetty04'`` (z=0): convert
    :math:`L_\\nu \\to L_\\lambda` and normalise to the peak near rest 4575 Å.
    """
    arr = np.genfromtxt(FEII_VC04_RAW)
    wave_angstrom = arr[:, 0]
    l_nu = arr[:, 1]
    l_lambda = l_nu * _C_ARB / wave_angstrom**2
    norm = l_lambda[np.argmin(np.abs(wave_angstrom - 4575.0))]
    return wave_angstrom / 10.0, l_lambda / norm


def _load_netzer_disc():
    """Load the Netzer accretion-disc course grid (16 (M, a, Mdot) models).

    Mirrors the ``NetzerDisk`` (course grid) import block of upstream
    ``database_builder/__init__.py``. The raw ``table_of_disk_models`` has 23
    header rows, then columns ``[freq (Hz), wave (Å), L_nu_1 ... L_nu_16]``.
    Rows are reversed (upstream ``[::-1]``) so wavelength is ascending. Each
    model's :math:`L_\\nu` becomes :math:`L_\\lambda = L_\\nu \\nu^2 / c` and is
    renormalised to 1 at 510 nm (5100 Å).

    Returns
    -------
    wave_nm : ndarray, shape (n_wave,)
    lumin : ndarray, shape (16, n_wave)
        Normalised :math:`L_\\lambda` per model.
    m, a, mdot : ndarray of str, shape (16,)
        Grid labels (log10 M_BH/Msun, spin, Eddington ratio).
    """
    with open(DISC_RAW) as fh:
        body = "".join(fh.readlines()[_DISC_HEADER_LINES:][::-1])
    import io

    data = np.genfromtxt(io.BytesIO(body.encode()))
    wave_nm = data[:, 1] * 0.1  # Å -> nm
    freq = data[:, 0]
    options = [(mv, av, mdv) for av in _DISC_A for mdv in _DISC_MDOT for mv in _DISC_M]
    lumin = np.empty((len(options), wave_nm.size), dtype=np.float64)
    m_lab = np.empty(len(options), dtype="S8")
    a_lab = np.empty(len(options), dtype="S8")
    mdot_lab = np.empty(len(options), dtype="S8")
    for i, (mv, av, mdv) in enumerate(options):
        l_nu = data[:, 2 + i]
        l_lambda = l_nu * freq**2 / _C_ARB
        norm = np.interp(510.0, wave_nm, l_lambda)
        lumin[i] = l_lambda / norm
        m_lab[i], a_lab[i], mdot_lab[i] = mv, av, mdv
    return wave_nm, lumin, m_lab, a_lab, mdot_lab


def _load_feii_template():
    """Load and normalise the Bruhweiler+Verner 2008 FeII template.

    Mirrors upstream ``database_builder/__init__.py`` (FeII import block):

    1. Raw file columns: observed-frame wavelength [Å], :math:`L_\\nu`
       (arbitrary units). The catalog z = 4593.4/4575 - 1 ≈ 0.00404.
    2. Convert :math:`L_\\nu \\to L_\\lambda = L_\\nu c / \\lambda_{\\rm obs}^2`.
    3. De-redshift wavelengths to rest frame.
    4. Normalise so :math:`L_\\lambda(4575\\,\\mathrm{\\AA, rest}) = 1`.

    Returns wave_nm (rest-frame, nm) and the normalised :math:`L_\\lambda`.
    """
    from scipy import constants as cst

    arr = np.loadtxt(FEII_RAW)
    wave_obs_angstrom = arr[:, 0]
    L_nu = arr[:, 1]
    # Upstream catalog redshift (paper §2.1.2): the template's FeII 4593.4
    # peak appears at observed 4575.
    z = 4593.4 / 4575.0 - 1.0
    wave_rest_angstrom = wave_obs_angstrom / (1.0 + z)
    # Convert L_nu -> L_lambda using observed-frame wave (upstream convention).
    L_lambda = L_nu * cst.c / wave_obs_angstrom**2
    # Normalise at rest-frame 4575 Å.
    norm_idx = np.argmin(np.abs(wave_rest_angstrom - 4575.0))
    norm = L_lambda[norm_idx]
    L_lambda = L_lambda / norm
    return wave_rest_angstrom / 10.0, L_lambda


def main():
    feii_wave, feii_lumin = _load_feii_template()
    lines = parse_mor_netzer_lines(LINES_RAW)
    line_names = np.array([r[0] for r in lines], dtype="S")
    line_wave_nm = np.array([r[1] for r in lines], dtype=np.float64)
    line_broad = np.array([r[2] for r in lines], dtype=np.float64)
    line_narrow_sy2 = np.array([r[3] for r in lines], dtype=np.float64)
    line_narrow_liner = np.array([r[4] for r in lines], dtype=np.float64)
    torus_wave_nm = _load_full_torus_wave()
    mn12_wave, mn12_avg, mn12_lo, mn12_hi, si_wave, si_lumin = _load_mn12_torus()
    vc04_wave, vc04_lumin = _load_veroncetty_feii()
    disc_wave, disc_lumin, disc_m, disc_a, disc_mdot = _load_netzer_disc()

    with h5py.File(OUT, "w") as f:
        f.attrs["source"] = "JohannesBuchner/GRAHSP @ database_builder/activate/agn/"
        f.attrs["paper"] = "Buchner et al. 2024, arXiv:2405.19297"
        f.attrs["license"] = "CeCILL-v2 (upstream)"

        feii = f.create_group("feii_bruhweiler2008")
        feii.attrs["density"] = "n_H = 1e11 cm^-3"
        feii.attrs["microturbulence"] = "xi = 20 km/s"
        feii.attrs["ionizing_flux"] = "phi_H = 10^20.5 cm^-2 s^-1"
        feii.attrs["redshift_dereddened_from"] = 0.004
        feii.attrs["citation"] = "Bruhweiler & Verner 2008, ApJ, 675, 83"
        feii.create_dataset("wave_nm", data=feii_wave)
        feii.create_dataset("lumin", data=feii_lumin)

        lns = f.create_group("netzer1990_lines")
        lns.attrs["citation"] = "Netzer 1990; Mor & Netzer 2012; H-gamma from Rakshit+ 2020"
        lns.attrs["normalisation_broad"] = HBETA_BROAD_RATIO
        lns.attrs["normalisation_narrow"] = HBETA_NARROW_RATIO
        lns.attrs["units"] = "wave_nm in nm; broad/narrow in L(line)/L(Hbeta) ratios"
        lns.create_dataset("name", data=line_names)
        lns.create_dataset("wave_nm", data=line_wave_nm)
        lns.create_dataset("broad", data=line_broad)
        lns.create_dataset("narrow_sy2", data=line_narrow_sy2)
        lns.create_dataset("narrow_liner", data=line_narrow_liner)

        torus = f.create_group("torus")
        torus.attrs["source"] = "activategtorus.py self.wave (nm)"
        torus.create_dataset("wave_nm", data=torus_wave_nm)

        mn12 = f.create_group("torus_mn12")
        mn12.attrs["citation"] = "Mor & Netzer 2012, MNRAS, 420, 526; Mullaney+ 2011 silicate"
        mn12.attrs["source"] = "activatetorus.py (mor-avg/lo/hi + mullaney-silicate)"
        mn12.attrs["units"] = "L_lambda templates, normalised to 1 at 12 um (continuum)"
        mn12.create_dataset("wave_nm", data=mn12_wave)
        mn12.create_dataset("avg", data=mn12_avg)
        mn12.create_dataset("lo", data=mn12_lo)
        mn12.create_dataset("hi", data=mn12_hi)
        mn12.create_dataset("si_wave_nm", data=si_wave)
        mn12.create_dataset("si_lumin", data=si_lumin)

        vc04 = f.create_group("feii_veroncetty2004")
        vc04.attrs["citation"] = "Veron-Cetty, Joly & Veron 2004, A&A, 417, 515"
        vc04.attrs["units"] = "L_lambda, normalised to the peak near rest 4575 A; no de-redshift"
        vc04.create_dataset("wave_nm", data=vc04_wave)
        vc04.create_dataset("lumin", data=vc04_lumin)

        disc = f.create_group("netzer_disc")
        disc.attrs["citation"] = "Netzer & Trakhtenbrot 2014; Netzer 2013 (AGN book)"
        disc.attrs["source"] = "activatedisk.py course grid (table_of_disk_models)"
        disc.attrs["units"] = "L_lambda per (M, a, Mdot) model, normalised to 1 at 510 nm; inc=0"
        disc.attrs["model_order"] = "a in (0.998, 0); Mdot in (0.3, 0.03); M in (6,7,8,9)"
        disc.create_dataset("wave_nm", data=disc_wave)
        disc.create_dataset("lumin", data=disc_lumin)
        disc.create_dataset("m", data=disc_m)
        disc.create_dataset("a", data=disc_a)
        disc.create_dataset("mdot", data=disc_mdot)

    print(f"wrote {OUT}")
    print(f"  feii: {feii_wave.size} samples, {feii_wave.min():.1f}-{feii_wave.max():.1f} nm")
    print(f"  lines: {len(lines)} lines, {line_wave_nm.min():.0f}-{line_wave_nm.max():.0f} nm")
    print(
        f"  torus grid: {torus_wave_nm.size} points, "
        f"{torus_wave_nm.min():.1f}-{torus_wave_nm.max():.0f} nm"
    )
    print(f"  mn12 torus: {mn12_wave.size} pts; silicate {si_wave.size} pts")
    print(f"  vc04 feii: {vc04_wave.size} samples, {vc04_wave.min():.1f}-{vc04_wave.max():.1f} nm")
    print(f"  netzer disc: {disc_lumin.shape[0]} models x {disc_wave.size} wave pts")


if __name__ == "__main__":
    main()
