"""Generate reference fixtures for the GRAHSP module from upstream pure-NumPy code.

This script does not require installing the upstream ``grahsp`` package. It
imports the pure mathematical formulas from
``arxiv_library/code/grahsp/pcigale/creation_modules/`` and evaluates them on
fixed wavelength grids and parameter sets.

The resulting ``.npz`` files in ``tests/fixtures/grahsp/`` are the oracles
that the tengri ``components/agn/grahsp/`` JAX implementations must match.

Run from the repo root::

    .venv/bin/python tools/generate_grahsp_fixtures.py

Re-running is idempotent.

Wavelength convention
---------------------
Upstream GRAHSP (CIGALE-derived) uses **nm** internally. Tengri uses **Å**.
The fixtures are stored in **nm** to match upstream byte-for-byte; the JAX
adapters perform the unit conversion. All luminosities are CIGALE's W/nm
(``L_lambda``); the unit factors carry through trivially.

References
----------
.. [1] Buchner, J., Starck, H., Salvato, M., et al. 2024,
       "Genuine Retrieval of the AGN Host Stellar Population (GRAHSP)",
       arXiv:2405.19297.
.. [2] Ryde, F. 1999, ApJ, 511, 692. Smooth bending power-law parameterisation.
.. [3] Prevot, M. L., Lequeux, J., Maurice, E., et al. 1984, A&A, 132, 389.
       SMC dust attenuation curve.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "grahsp"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. SBPL big blue bump (activatepl.py, lines 49-68 — the pure-NumPy variant)
# ---------------------------------------------------------------------------
def sbpl_upstream(x, norm, lam1, lam2, x0, xbrk, Lambda):
    """Verbatim pure-NumPy implementation from upstream ``activatepl.sbpl``."""
    with np.errstate(over="ignore"):
        q = np.log(x / xbrk) / Lambda
        qpiv = np.log(x0 / xbrk) / Lambda
        return (
            norm
            * (x / x0) ** ((lam1 + lam2 + 2) / 2.0)
            * ((np.exp(q) + np.exp(-q)) / (np.exp(qpiv) + np.exp(-qpiv)))
            ** ((lam2 - lam1) / 2.0 * Lambda)
            * (x0 / x)
        )


def make_sbpl_fixture():
    # Wave grid in nm (CIGALE convention): 10 nm to 1e5 nm log-spaced.
    wave_nm = np.logspace(1, 5, 401)
    # 5 parameter sets covering the paper's Brown 2019 fits range.
    cases = [
        # (uvslope, plslope, plbendloc, plbendwidth, lum5100A, cutoff_nm)
        # Default GRAHSP-ish
        dict(
            uvslope=0.0,
            plslope=-1.7,
            plbendloc=100.0,
            plbendwidth=1.0,
            lum5100A=1.0e36,
            cutoff=10000.0,
        ),
        # MRK231-ish (steep)
        dict(
            uvslope=0.0,
            plslope=-1.7,
            plbendloc=160.0,
            plbendwidth=0.2,
            lum5100A=1.0e38,
            cutoff=10000.0,
        ),
        # Flat slope
        dict(
            uvslope=0.5,
            plslope=-1.0,
            plbendloc=80.0,
            plbendwidth=2.0,
            lum5100A=2.5e37,
            cutoff=-1.0,
        ),
        # Sharp bend
        dict(
            uvslope=0.0,
            plslope=-2.5,
            plbendloc=120.0,
            plbendwidth=0.5,
            lum5100A=1.0e35,
            cutoff=10000.0,
        ),
        # Wide bend
        dict(
            uvslope=0.0,
            plslope=-2.0,
            plbendloc=100.0,
            plbendwidth=5.0,
            lum5100A=1.0e36,
            cutoff=10000.0,
        ),
    ]
    spectra = np.zeros((len(cases), wave_nm.size))
    for i, p in enumerate(cases):
        l_agn = p["lum5100A"] / 510.0  # cigale convention: lum5100A = lambda*L_lambda
        bbb = sbpl_upstream(
            wave_nm, l_agn, p["uvslope"], p["plslope"], 510.0, p["plbendloc"], p["plbendwidth"]
        )
        if p["cutoff"] > 0:
            bbb = bbb * -np.expm1(-(p["cutoff"] / wave_nm))
        spectra[i] = bbb
    np.savez(
        FIXTURE_DIR / "sbpl_bbb.npz",
        wave_nm=wave_nm,
        spectra=spectra,
        params=np.array(
            [
                (
                    p["uvslope"],
                    p["plslope"],
                    p["plbendloc"],
                    p["plbendwidth"],
                    p["lum5100A"],
                    p["cutoff"],
                )
                for p in cases
            ],
            dtype=[
                ("uvslope", "f8"),
                ("plslope", "f8"),
                ("plbendloc", "f8"),
                ("plbendwidth", "f8"),
                ("lum5100A", "f8"),
                ("cutoff", "f8"),
            ],
        ),
    )
    return wave_nm.size, len(cases)


# ---------------------------------------------------------------------------
# 2. Bi-attenuation (biattenuation.py — pure NumPy attenuation curve)
# ---------------------------------------------------------------------------
def biatten_curve_upstream(wave_nm, opt_index, nir_index, norm, lam_break):
    """Verbatim from biattenuation.BiAttenuationLaw.get_attenuation."""
    return norm * (wave_nm / lam_break) ** np.where(wave_nm < lam_break, opt_index, nir_index)


def make_biatten_fixture():
    wave_nm = np.logspace(1, 5, 401)
    # Default GRAHSP/Prevot
    curve_default = biatten_curve_upstream(wave_nm, -1.2, -3.0, 1.2, 1100.0)
    # Fawcett+22 alternative
    curve_fawcett = biatten_curve_upstream(wave_nm, -1.0, -2.6, 1.0, 1100.0)
    # Compute attenuation factors for several E(B-V), E(B-V)-AGN pairs
    cases = [
        (0.1, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.5, 0.3),
        (1.0, 1.0),
    ]
    factor_gal = np.zeros((len(cases), wave_nm.size))
    factor_agn = np.zeros((len(cases), wave_nm.size))
    for i, (ebv, ebv_agn) in enumerate(cases):
        factor_gal[i] = 10 ** (ebv * curve_default / -2.5)
        factor_agn[i] = 10 ** ((ebv + ebv_agn) * curve_default / -2.5)
    np.savez(
        FIXTURE_DIR / "biattenuation.npz",
        wave_nm=wave_nm,
        curve_default=curve_default,
        curve_fawcett=curve_fawcett,
        cases=np.array(cases, dtype=[("ebv", "f8"), ("ebv_agn", "f8")]),
        factor_gal=factor_gal,
        factor_agn=factor_agn,
    )
    return wave_nm.size


# ---------------------------------------------------------------------------
# 3. AGN torus (activategtorus.py)
# ---------------------------------------------------------------------------
# Hard-coded torus wave grid from upstream (line 91 of activategtorus.py).
TORUS_WAVE_UM = np.array(
    [
        0.360,
        0.450,
        0.580,
        0.750,
        1.000,
        1.009,
        1.019,
        1.028,
        1.038,
        1.047,
        1.057,
        1.067,
        1.076,
        1.086,
        1.096,
        1.107,
        1.117,
        1.127,
        1.138,
        1.148,
    ]
)  # truncated for brevity in test fixture; full grid loaded from upstream below


def _load_full_torus_wave():
    _UPSTREAM = REPO_ROOT / "arxiv_library" / "code" / "grahsp"
    src = _UPSTREAM / "pcigale" / "creation_modules" / "activategtorus.py"
    text = src.read_text()
    start = text.find("self.wave = 1000 * np.array([")
    end = text.find("])", start)
    arr_text = text[start + len("self.wave = 1000 * np.array(") : end + 1]
    arr = 1000.0 * np.array(ast.literal_eval(arr_text))
    return arr  # nm, matching upstream `1000 * np.array([...])`


def torus_upstream(wave_torus_nm, wave_si_nm, params):
    """Reproduce activategtorus.ActivateGTorus.process for a single param set."""
    p = params
    log_wave = np.log10(wave_torus_nm / 1000.0)  # log10(wave/um)
    norm_index = int(np.argmin(np.abs(10**log_wave - 12.0)))

    fcov = p["fcov"]
    logCOOLlam = np.log10(p["COOLlam"])
    COOLwidth = p["COOLwidth"]
    HOTfcov = p["HOTfcov"]
    logHOTlam = np.log10(p["HOTlam"])
    HOTwidth = p["HOTwidth"]
    Si = p["Si"]
    SiEmAmpl = 0.4
    SiAbsAmpl = SiEmAmpl * p.get("SiRatio", 0.29)
    SiEmlam = p.get("SiEmlam", 9841.0)
    SiAbslam = p.get("SiAbslam", 14224.0)
    SiEmWidth = p.get("SiEmWidth", 1025.3)
    SiAbsWidth = p.get("SiAbsWidth", 1163.5)
    lum5100A = p["lum5100A"]

    l_torus = 2.5 * lum5100A * fcov  # at 12 um (lambda * L_lambda)
    cool_spectrum = np.exp(-0.5 * ((log_wave - logCOOLlam) / COOLwidth) ** 2)  # W = sigma (upstream drops the 1/2)
    hot_spectrum = (
        HOTfcov
        * 10 ** (logCOOLlam - logHOTlam)
        * np.exp(-0.5 * ((log_wave - logHOTlam) / HOTwidth) ** 2)
    )
    total_spectrum = cool_spectrum + hot_spectrum
    torus_spectrum = l_torus / 12000.0 * total_spectrum / total_spectrum[norm_index]
    si_spectrum = (
        l_torus
        / 12000.0
        * Si
        * (
            SiEmAmpl * np.exp(-0.5 * ((wave_si_nm - SiEmlam) / SiEmWidth) ** 2)
            - SiAbsAmpl * np.exp(-0.5 * ((wave_si_nm - SiAbslam) / SiAbsWidth) ** 2)
        )
    )
    return torus_spectrum, si_spectrum


def make_torus_fixture():
    wave_torus_nm = _load_full_torus_wave()
    wave_si_nm = np.logspace(np.log10(3000), np.log10(50000), 201)  # 3 um to 50 um
    cases = [
        # Faint AGN, no Si
        dict(
            fcov=0.1,
            Si=0.0,
            COOLlam=17.0,
            COOLwidth=0.45,
            HOTlam=2.0,
            HOTwidth=0.5,
            HOTfcov=0.0,
            lum5100A=1.0e36,
        ),
        # Si emission
        dict(
            fcov=0.4,
            Si=1.0,
            COOLlam=20.0,
            COOLwidth=0.5,
            HOTlam=3.0,
            HOTwidth=0.5,
            HOTfcov=1.0,
            lum5100A=1.0e37,
        ),
        # Si absorption
        dict(
            fcov=0.6,
            Si=-2.0,
            COOLlam=25.0,
            COOLwidth=0.4,
            HOTlam=2.5,
            HOTwidth=0.4,
            HOTfcov=2.0,
            lum5100A=1.0e38,
        ),
        # Wide cool peak
        dict(
            fcov=0.3,
            Si=0.5,
            COOLlam=30.0,
            COOLwidth=0.65,
            HOTlam=4.0,
            HOTwidth=0.6,
            HOTfcov=0.5,
            lum5100A=1.0e36,
        ),
    ]
    torus_spectra = np.zeros((len(cases), wave_torus_nm.size))
    si_spectra = np.zeros((len(cases), wave_si_nm.size))
    for i, p in enumerate(cases):
        t, s = torus_upstream(wave_torus_nm, wave_si_nm, p)
        torus_spectra[i] = t
        si_spectra[i] = s
    np.savez(
        FIXTURE_DIR / "torus.npz",
        wave_torus_nm=wave_torus_nm,
        wave_si_nm=wave_si_nm,
        torus_spectra=torus_spectra,
        si_spectra=si_spectra,
        params=np.array(
            [
                (
                    p["fcov"],
                    p["Si"],
                    p["COOLlam"],
                    p["COOLwidth"],
                    p["HOTlam"],
                    p["HOTwidth"],
                    p["HOTfcov"],
                    p["lum5100A"],
                )
                for p in cases
            ],
            dtype=[
                ("fcov", "f8"),
                ("Si", "f8"),
                ("COOLlam", "f8"),
                ("COOLwidth", "f8"),
                ("HOTlam", "f8"),
                ("HOTwidth", "f8"),
                ("HOTfcov", "f8"),
                ("lum5100A", "f8"),
            ],
        ),
    )
    return wave_torus_nm.size


# ---------------------------------------------------------------------------
# 4. Emission lines (activatelines.py — pure-Python part of `add_lines`)
# ---------------------------------------------------------------------------
from io import StringIO

import scipy.constants as cst

C_KMS = cst.c / 1000.0
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def parse_mor_netzer_lines(path: Path):
    """Parse the upstream emission line table.

    Returns list of (name, wave_nm, broad, narrow_sy2, narrow_liner) tuples.
    Wavelengths are converted from Å to nm (matching upstream
    ``database_builder/__init__.py`` line 905: ``data['wave'] * 0.1``).
    """
    rows = []
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split()
        try:
            broad = float(parts[-3])
            narrow_sy2 = float(parts[-2])
            narrow_liner = float(parts[-1])
            wave_angstrom = float(parts[-4])
        except ValueError:
            continue
        name = " ".join(parts[:-4])
        rows.append((name, wave_angstrom * 0.1, broad, narrow_sy2, narrow_liner))
    return rows


def lines_upstream(wave_nm_grid, lines_rows, lum5100A, A_lines, line_width_kms, agn_type=1):
    """activatelines.add_lines on a flat user-provided wave grid, with a unit-area Gaussian.

    Upstream normalizes with sqrt(pi sigma^2), making every line sqrt(2) too strong;
    tengri uses sqrt(2 pi sigma^2), so each line integrates to its tabulated flux.
    """
    l_agn = lum5100A / 510.0  # W/nm
    l_broad = 0.02 * l_agn * A_lines  # H-beta broad scale [erg/s] / [W/nm]
    l_narrow = 0.002 * l_agn * A_lines

    bl_lumin = np.zeros_like(wave_nm_grid)
    nl_lumin = np.zeros_like(wave_nm_grid)
    for _name, lam0, broad, sy2, liner in lines_rows:
        width_nm = lam0 * (line_width_kms * 1000.0) / cst.c
        sigma = width_nm * FWHM_TO_SIGMA
        norm_factor = 510.0 / np.sqrt(2.0 * np.pi * sigma**2)  # unit-area Gaussian (upstream uses pi, not 2 pi)
        shape = np.exp(-0.5 * (wave_nm_grid - lam0) ** 2 / sigma**2)
        if agn_type == 1:
            bl_lumin += l_broad * broad * shape * norm_factor
            nl_lumin += l_narrow * sy2 * shape * norm_factor
        elif agn_type == 2:
            nl_lumin += l_narrow * sy2 * shape * norm_factor
        elif agn_type == 3:
            nl_lumin += l_narrow * liner * shape * norm_factor
    return bl_lumin, nl_lumin


def feii_upstream(wave_nm_grid, feii_template_path, lum5100A, A_lines, A_FeII):
    """Upstream FeII pipeline: L_nu -> L_lambda -> de-redshift -> normalize at 4575 Å rest."""
    from scipy import constants as cst

    arr = np.loadtxt(StringIO(feii_template_path.read_text()))
    wave_obs_angstrom = arr[:, 0]
    L_nu = arr[:, 1]
    z = 4593.4 / 4575.0 - 1.0
    wave_rest_angstrom = wave_obs_angstrom / (1.0 + z)
    L_lambda = L_nu * cst.c / wave_obs_angstrom**2
    norm_idx = np.argmin(np.abs(wave_rest_angstrom - 4575.0))
    L_lambda = L_lambda / L_lambda[norm_idx]
    wave_template_nm = wave_rest_angstrom / 10.0
    interp_flux = np.interp(wave_nm_grid, wave_template_nm, L_lambda, left=0.0, right=0.0)
    l_broadlines = 0.02 * (lum5100A / 510.0) * A_lines
    return wave_template_nm, L_lambda, interp_flux * A_FeII * l_broadlines


def make_lines_fixture():
    line_table = REPO_ROOT / "data" / "grahsp" / "mor_netzer_2012_emission_lines.txt"
    feii_template = REPO_ROOT / "data" / "grahsp" / "feii_bruhweiler2008_d11_m20_20p5.txt"
    rows = parse_mor_netzer_lines(line_table)
    wave_nm_grid = np.linspace(100.0, 25000.0, 10001)  # 100 nm to 2.5 um, 2.5nm steps

    cases = [
        dict(lum5100A=1.0e36, A_lines=1.0, line_width_kms=5000.0, agn_type=1, A_FeII=5.0),
        dict(lum5100A=1.0e37, A_lines=2.0, line_width_kms=10000.0, agn_type=1, A_FeII=2.0),
        dict(lum5100A=1.0e36, A_lines=1.0, line_width_kms=300.0, agn_type=2, A_FeII=0.0),
    ]
    bl = np.zeros((len(cases), wave_nm_grid.size))
    nl = np.zeros((len(cases), wave_nm_grid.size))
    feii = np.zeros((len(cases), wave_nm_grid.size))
    for i, p in enumerate(cases):
        b, n = lines_upstream(
            wave_nm_grid, rows, p["lum5100A"], p["A_lines"], p["line_width_kms"], p["agn_type"]
        )
        bl[i] = b
        nl[i] = n
        _, _, feii_lumin = feii_upstream(
            wave_nm_grid, feii_template, p["lum5100A"], p["A_lines"], p["A_FeII"]
        )
        feii[i] = feii_lumin
    np.savez(
        FIXTURE_DIR / "lines.npz",
        wave_nm=wave_nm_grid,
        broad_lines=bl,
        narrow_lines=nl,
        feii=feii,
        n_lines=len(rows),
        params=np.array(
            [
                (p["lum5100A"], p["A_lines"], p["line_width_kms"], p["agn_type"], p["A_FeII"])
                for p in cases
            ],
            dtype=[
                ("lum5100A", "f8"),
                ("A_lines", "f8"),
                ("line_width_kms", "f8"),
                ("agn_type", "i4"),
                ("A_FeII", "f8"),
            ],
        ),
    )
    return len(rows)


def balmer_upstream(wave_nm, lum5100A, ABC, linewidth_kms):
    """activatelines.ActivateLines BC shape for a single param set, physically corrected.

    Grandi (1982) continuum with the Balmer edge smoothed by a Gaussian
    (upstream ``activatelines.py`` lines 137-175). Upstream uses the line FWHM as the
    Gaussian standard deviation and drops the smeared tail above the edge; here the
    standard deviation is ``FWHM / (2 sqrt(2 ln 2))`` and the tail is kept, so the
    smoothing conserves the emitted energy.
    """
    from scipy.special import erfc

    BE_wave = 364.6
    BC_tau = 1.0
    BC_T = 15000.0
    h_c_per_k_B = 1.439e7  # nm * K

    black_body = wave_nm ** (-5) / np.expm1(h_c_per_k_B / (BC_T * wave_nm))
    black_body0 = BE_wave ** (-5) / np.expm1(h_c_per_k_B / (BC_T * BE_wave))

    x = wave_nm / BE_wave
    truncation = -np.expm1(-BC_tau * x**3)
    truncation0 = -np.expm1(-BC_tau)

    alpha = 1.8
    beta = -0.8
    sigma = (linewidth_kms * 1000.0) / cst.c / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    z = (x - 1.0) * 2.0 ** (-0.5) / sigma

    term_b = 0.5 * erfc(z)
    term_a12 = 0.5 * x * erfc(z)
    term_a3 = -sigma * (2.0 * np.pi) ** (-0.5) * np.exp(-(z**2))

    convolved = (beta * term_b + alpha * (term_a12 + term_a3)) * (1.0 - np.exp(-1.0))

    # Use convolved above 250 nm, raw truncation below
    truncation_convolved = np.where(wave_nm > 250.0, convolved, truncation)

    BC_shape = (black_body / black_body0) * (truncation_convolved / truncation0)

    l_agn = lum5100A / 510.0
    l_bc = l_agn * ABC
    return l_bc * BC_shape, wave_nm


def make_balmer_fixture():
    """Generate Balmer continuum reference fixture from upstream formula."""
    # Wavelength grid covering the Balmer edge region
    wave_nm = np.linspace(200.0, 400.0, 400)  # 200 nm to 400 nm

    cases = [
        # Weak BC, moderate linewidth
        dict(lum5100A=1.0e36, ABC=0.1, linewidth_kms=5000.0),
        # Strong BC, wide lines
        dict(lum5100A=1.0e37, ABC=0.5, linewidth_kms=10000.0),
        # Very strong BC, narrow lines
        dict(lum5100A=1.0e36, ABC=1.0, linewidth_kms=1000.0),
        # Moderate case
        dict(lum5100A=5.0e36, ABC=0.3, linewidth_kms=3000.0),
    ]

    # Collect results for all cases
    bc_spectra = []
    params_list = []

    for p in cases:
        bc_shape, _wave_edge = balmer_upstream(
            wave_nm, p["lum5100A"], p["ABC"], p["linewidth_kms"]
        )
        bc_spectra.append(bc_shape)
        params_list.append((p["lum5100A"], p["ABC"], p["linewidth_kms"]))

    bc_spectra_padded = np.asarray(bc_spectra)

    np.savez(
        FIXTURE_DIR / "balmer.npz",
        wave_nm=wave_nm,
        balmer_spectra=bc_spectra_padded,
        params=np.array(
            params_list,
            dtype=[
                ("lum5100A", "f8"),
                ("ABC", "f8"),
                ("linewidth_kms", "f8"),
            ],
        ),
    )
    return len(wave_nm), len(cases)


# ---------------------------------------------------------------------------
# 6. Netzer accretion disc (activatedisk.py template grid), lambda*L_lambda(5100 A) = l5100
# ---------------------------------------------------------------------------


def make_netzer_disc_fixture():
    """Netzer disc spectra on the template's native grid, in tengri's convention.

    Upstream ``activatedisk`` returns ``l5100 * T(lambda)`` with ``T(510 nm) = 1``,
    i.e. it scales ``L_lambda(510 nm)``, which is 510 times brighter than its
    stated ``lambda*L_lambda(5100 A)``. tengri's convention is
    ``lambda*L_lambda(5100 A) = l5100`` (as for the power-law disc, torus and
    lines), so the oracle is ``l5100 / 510 * T(lambda) / T(510 nm)`` with
    ``T(510 nm)`` from log-log interpolation of the native template (the
    stored normalization used a linear interpolation).
    """
    import h5py

    with h5py.File(REPO_ROOT / "data" / "grahsp" / "grahsp_templates.h5", "r") as h:
        wave_nm = np.asarray(h["netzer_disc/wave_nm"][:], dtype=np.float64)
        lumin = np.asarray(h["netzer_disc/lumin"][:], dtype=np.float64)
        m_lab = [x.decode() if isinstance(x, bytes) else str(x) for x in h["netzer_disc/m"][:]]
        a_lab = [x.decode() if isinstance(x, bytes) else str(x) for x in h["netzer_disc/a"][:]]
        mdot_lab = [x.decode() if isinstance(x, bytes) else str(x) for x in h["netzer_disc/mdot"][:]]
    cases = [
        (l5100, idx)
        for l5100 in (1.0e43, 1.0e44)
        for idx in (0, 2, 10, 15)
    ]
    spectra = np.zeros((len(cases), wave_nm.size))
    params = np.empty((len(cases), 5, 2), dtype=object)
    j = int(np.searchsorted(wave_nm, 510.0))  # bracketing native nodes j-1, j
    frac = np.log(510.0 / wave_nm[j - 1]) / np.log(wave_nm[j] / wave_nm[j - 1])
    for i, (l5100, idx) in enumerate(cases):
        lo, hi = lumin[idx, j - 1], lumin[idx, j]
        t_510 = np.exp((1.0 - frac) * np.log(lo) + frac * np.log(hi))
        spectra[i] = (l5100 / 510.0) * lumin[idx] / t_510
        params[i] = [
            ["l5100", l5100],
            ["model_idx", idx],
            ["M", m_lab[idx]],
            ["a", a_lab[idx]],
            ["Mdot", mdot_lab[idx]],
        ]
    np.savez(
        FIXTURE_DIR / "netzer_disc.npz",
        wave_disc_nm=wave_nm,
        disc_spectra=spectra,
        params=params,
    )
    return len(cases), wave_nm.size


# ---------------------------------------------------------------------------
# 7. Mor & Netzer 2012 template torus (activatetorus.py), lambda*L_lambda(12 um) = 2.5 fcov l5100
# ---------------------------------------------------------------------------


def make_torus_mn12_fixture():
    """MN12 template torus + Si feature on the native template grids.

    Upstream ``activatetorus`` uses ``l_torus = 2.5 * l5100 * fcov / 12.0 * 0.510``
    per nm, 510 times brighter than the Netzer relation
    ``lambda*L_lambda(12 um) = 2.5 * fcov * l5100`` that its log-Gaussian sibling
    (``activategtorus``) satisfies. tengri uses ``l_torus = 2.5 * l5100 * fcov / 12000``
    per nm, so the templates (which are 1 at 12 um) give the Netzer relation.
    """
    import h5py

    with h5py.File(REPO_ROOT / "data" / "grahsp" / "grahsp_templates.h5", "r") as h:
        g = h["torus_mn12"]
        wave_nm = np.asarray(g["wave_nm"][:], dtype=np.float64)
        avg = np.asarray(g["avg"][:], dtype=np.float64)
        lo = np.asarray(g["lo"][:], dtype=np.float64)
        hi = np.asarray(g["hi"][:], dtype=np.float64)
        si_wave_nm = np.asarray(g["si_wave_nm"][:], dtype=np.float64)
        si_lumin = np.asarray(g["si_lumin"][:], dtype=np.float64)
    cases = [
        dict(l5100=1.0e36, fcov=0.4, tor_temp=0.0, tor_cutoff_um=1.2, si=0.5),
        dict(l5100=1.0e36, fcov=0.4, tor_temp=0.5, tor_cutoff_um=1.2, si=0.5),
        dict(l5100=1.0e36, fcov=0.4, tor_temp=-0.5, tor_cutoff_um=1.2, si=0.5),
        dict(l5100=2.0e36, fcov=0.5, tor_temp=0.2, tor_cutoff_um=1.7, si=-0.3),
    ]
    torus = np.zeros((len(cases), wave_nm.size))
    si_spec = np.zeros((len(cases), si_wave_nm.size))
    for i, p in enumerate(cases):
        l_torus = 2.5 * p["l5100"] * p["fcov"] / 12000.0
        t = p["tor_temp"]
        dev = (hi - avg) * t if t > 0 else (lo - avg) * (-t)
        cutoff = 1.0 - np.exp(-((wave_nm / 1000.0 / p["tor_cutoff_um"]) ** 2))
        torus[i] = l_torus * (avg + dev) * cutoff
        si_spec[i] = l_torus * si_lumin * p["si"]
    np.savez(
        FIXTURE_DIR / "torus_mn12.npz",
        wave_mn12_nm=wave_nm,
        wave_si_nm=si_wave_nm,
        mn12_avg=avg,
        mn12_lo=lo,
        mn12_hi=hi,
        mn12_si_lumin=si_lumin,
        params=np.array(
            [(p["l5100"], p["fcov"], p["tor_temp"], p["tor_cutoff_um"], p["si"]) for p in cases],
            dtype=[
                ("l5100", "f8"),
                ("fcov", "f8"),
                ("tor_temp", "f8"),
                ("tor_cutoff_um", "f8"),
                ("si", "f8"),
            ],
        ),
        torus_spectra_native=torus,
        si_spectra_native=si_spec,
    )
    return len(cases), wave_nm.size


def main():
    n_wave_sbpl, n_cases = make_sbpl_fixture()
    print(f"  sbpl_bbb.npz: {n_cases} cases x {n_wave_sbpl} wavelengths")
    n_wave_atten = make_biatten_fixture()
    print(f"  biattenuation.npz: 5 cases x {n_wave_atten} wavelengths")
    n_wave_torus = make_torus_fixture()
    print(f"  torus.npz: 4 cases x {n_wave_torus} wavelengths")
    n_lines = make_lines_fixture()
    print(f"  lines.npz: 3 cases x {n_lines} lines")
    n_wave_balmer, n_cases_balmer = make_balmer_fixture()
    print(f"  balmer.npz: {n_cases_balmer} cases x {n_wave_balmer} wavelengths")
    n_cases_disc, n_wave_disc = make_netzer_disc_fixture()
    print(f"  netzer_disc.npz: {n_cases_disc} cases x {n_wave_disc} wavelengths")
    n_cases_mn12, n_wave_mn12 = make_torus_mn12_fixture()
    print(f"  torus_mn12.npz: {n_cases_mn12} cases x {n_wave_mn12} wavelengths")


if __name__ == "__main__":
    main()
