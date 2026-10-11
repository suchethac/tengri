# SPDX-License-Identifier: BSD-3-Clause
"""Reader for upstream GRAHSP's committed spectra, and tengri's public composable build.

Upstream GRAHSP (Buchner et al. 2024, A&A 692, A161; arXiv:2405.19297) is a pcigale module
set, not a pip package. ``data/grahsp_upstream_reference.h5`` holds its own spectra for 38
parameter sets (a fiducial plus one-at-a-time sweeps), built by
``scripts/build_grahsp_reference.py``. This driver reads that file and nothing else: upstream
GRAHSP is never imported at render time. tengri is evaluated only through the public
``SEDModel.build`` grammar and ``model.predict``.

Units. Wavelengths are nm in upstream arrays, Angstrom in tengri's model grid. Upstream arrays
are per-nm luminosity densities :math:`L_\\lambda` [erg/s/nm] at ``lum5100A = 1``.

Explicit corrections, applied to upstream values on the page (never inside ``src/tengri``):

* ``LINE_FLUX_SCALE``: upstream's emission lines carry :math:`\\sqrt{2}` times the unit-area
  normalization, so upstream lines are divided by :math:`\\sqrt{2}`.
* ``WIDTH_SCALE``: upstream's torus log-Gaussian is :math:`\\exp(-x^2/W^2)`; paper Eq. 2 has
  :math:`\\exp(-x^2/(2W^2))` with :math:`W` in dex, so tengri receives
  :math:`W/\\sqrt{2}`.
* ``UPSTREAM_510_FACTOR``: upstream's Netzer disc and MN12 torus are normalized to
  :math:`L_\\lambda(510\\,\\mathrm{nm})=1` per nm, so
  :math:`\\lambda L_\\lambda(510\\,\\mathrm{nm})=510`; they are divided by 510.

Interpolation. Upstream is interpolated linearly in wavelength onto tengri's model grid. That
is upstream's own convention for its tabulated templates.

References
----------
.. [1] Buchner, J., Starck, H., Salvato, M., et al. 2024, A&A 692, A161, "Genuine Retrieval of
   the AGN Host Stellar Population (GRAHSP)", arXiv:2405.19297.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = REPO_ROOT / "data" / "grahsp_upstream_reference.h5"
TEMPLATE_PATH = REPO_ROOT / "data" / "grahsp" / "grahsp_templates.h5"

LINE_FLUX_SCALE = math.sqrt(2.0)
WIDTH_SCALE = 1.0 / math.sqrt(2.0)
UPSTREAM_510_FACTOR = 510.0
XRAY_FLOOR_NM = 12.4  # tengri zeroes the disc below this wavelength (corona overlap)
TORUS_WINDOW_NM = (400.0, 99000.0)
C_NM_PER_S = 2.99792458e17
C_ANGSTROM_PER_S = 2.99792458e18
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
C_KMS = 299792.458
REL_FLOOR = 1e-6

# Upstream component arrays, keyed by the short names the page uses.
COMPONENTS = {
    "disc": "agn.activate_Disk",
    "bl": "agn.activate_EmLines_BL",
    "nl": "agn.activate_EmLines_NL",
    "feii": "agn.activate_FeLines",
    "torus": "agn.activate_Torus",
    "si": "agn.activate_Torus_Si",
    "balmer": "agn.activate_BC",
    # netzer_mn12 variant of the builder: upstream activatedisk and activatetorus outputs.
    "netzer": "netzer_mn12.agn.activate_Disk",
    "mn12_torus": "netzer_mn12.agn.activate_Torus",
    "mn12_si": "netzer_mn12.agn.activate_TorusSi",
}

_FEII_TEMPLATE_NAMES = {"BruhweilerVerner08": "bruhweiler2008", "Veron-Cetty04": "veroncetty2004"}


def require_available() -> None:
    """Raise with the regeneration command if the committed reference is missing."""
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(
            f"{REFERENCE_PATH} is missing. It is committed to the repository; restore it from "
            "there, or regenerate it with scripts/build_grahsp_reference.py "
            "--upstream <GRAHSP checkout> --python <venv python>."
        )


def list_param_sets() -> tuple[str, ...]:
    """Parameter-set names in the reference file, in file order."""
    require_available()
    with h5py.File(REFERENCE_PATH, "r") as f:
        return tuple(k for k in f if k != "wavelength_nm")


@dataclass(frozen=True)
class UpstreamSet:
    """One upstream parameter set: its raw parameters and per-component spectra."""

    name: str
    params: dict  # upstream parameter name -> value
    wave_nm: np.ndarray  # reference grid [nm]
    spectra: dict  # short component name -> upstream L_lambda [erg/s/nm, unit normalization]


@functools.cache
def load_set(name: str) -> UpstreamSet:
    """Read one parameter set from the committed reference file."""
    require_available()
    with h5py.File(REFERENCE_PATH, "r") as f:
        grp = f[name]
        params = {
            k.removeprefix("param_"): (v if isinstance(v, str) else v.item())
            for k, v in grp.attrs.items()
        }
        raw = {k: np.asarray(grp[k][:]) for k in grp}
        wave = np.asarray(f["wavelength_nm"][:])
    spectra = {short: raw.get(key, np.zeros_like(wave)) for short, key in COMPONENTS.items()}
    return UpstreamSet(name, params, wave, spectra)


def composable_agn(
    up: dict,
    *,
    disc: str = "grahsp_sbpl",
    torus: str = "grahsp",
    a_feii: float | None = None,
    a_bc: float | None = None,
    width_correction: bool = True,
) -> dict:
    """Public ``SEDModel.build`` ``agn={...}`` spec for one upstream parameter set.

    The disc normalization is ``agn_grahsp_log_l5100 = 0`` (``l5100 = 1``), the unit upstream
    uses. ``a_lines``/``linewidth`` are top-level because they are shared across blocks.
    ``width_correction=False`` passes the upstream torus width unchanged.
    """
    from tengri import DEFAULT, Fixed

    def f(x):
        return Fixed(float(x))

    w_scale = WIDTH_SCALE if width_correction else 1.0
    disc_spec = {"type": disc, "agn_grahsp_log_l5100": f(0.0)}
    if disc == "grahsp_sbpl":
        disc_spec.update(
            {
                "agn_grahsp_uvslope": f(up["uvslope"]),
                "agn_grahsp_plslope": f(up["plslope"]),
                "agn_grahsp_plbendloc_nm": f(up["plbendloc"]),
                "agn_grahsp_plbendwidth": f(up["plbendwidth"]),
                "agn_grahsp_cutoff_nm": f(up["cutoff"]),
            }
        )
    else:  # Netzer template at the builder's fixed disc: log M = 8, a = 0, Mdot = 0.3
        disc_spec.update(
            {
                "agn_grahsp_netzer_log_mbh": f(8.0),
                "agn_grahsp_netzer_spin": f(0.0),
                "agn_grahsp_netzer_log_mdot": f(math.log10(0.3)),
            }
        )
    if torus == "grahsp":
        torus_spec = {
            "type": torus,
            "agn_grahsp_fcov": f(up["fcov"]),
            "agn_grahsp_si": f(up["si"]),
            "agn_grahsp_cool_lam_um": f(up["coollam"]),
            "agn_grahsp_cool_width": f(float(up["coolwidth"]) * w_scale),
            "agn_grahsp_hot_lam_um": f(up["hotlam"]),
            "agn_grahsp_hot_width": f(float(up["hotwidth"]) * w_scale),
            "agn_grahsp_hot_fcov": f(up["hotfcov"]),
        }
    else:  # MN12 template torus
        torus_spec = {
            "type": torus,
            "agn_grahsp_fcov": f(up["fcov"]),
            "agn_grahsp_si": f(up["si"]),
            "agn_grahsp_tor_temp": f(0.0),
            "agn_grahsp_tor_cutoff_um": f(1.2),
        }
    feii_spec = {"type": "grahsp"}
    if a_feii is not None:
        feii_spec["agn_grahsp_a_feii"] = f(a_feii)
    blr_spec = {"type": "grahsp"}
    if a_bc is not None:
        blr_spec["agn_grahsp_a_bc"] = f(a_bc)
    return {
        "type": "composable",
        "disc": disc_spec,
        "torus": torus_spec,
        "nlr": {"type": "grahsp"},
        "blr": blr_spec,
        "feii": feii_spec,
        "atten": {"type": "none"},
        "agn_log_lbol": Fixed(44.0),
        "agn_grahsp_a_lines": f(up["alines"]),
        "agn_grahsp_linewidth_kms": f(up["linewidth"]),
        "all_params": Fixed(DEFAULT),
        "norm": "independent",
    }


def agn_predict(ssp, up: dict, **kwargs):
    """Build the AGN-only public model for one parameter set and return its prediction."""
    from tengri import DEFAULT, Fixed, SEDModel

    sfh = {
        "type": "declining_exp",
        "sfh_declining_exp_tau_gyr": Fixed(1.0),
        "sfh_declining_exp_age_gyr": Fixed(4.9),
        "sfh_declining_exp_log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    }
    no_dust = {
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    }
    model = SEDModel.build(
        ssp_data=ssp,
        sfh=sfh,
        dust_attenuation=no_dust,
        agn=composable_agn(up, **kwargs),
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    return model.predict({})


def lambda_l_lambda_per_nm(pred, key: str) -> np.ndarray:
    """A component of ``pred.sed.components`` (L_nu [erg/s/Hz]) as L_lambda [erg/s/nm]."""
    wave_nm = np.asarray(pred.wave_rest) * 0.1
    l_nu = np.asarray(pred.sed.components[key])
    return l_nu * C_NM_PER_S / wave_nm**2


def upstream_on(
    wave_nm_model: np.ndarray, wave_nm_ref: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Upstream's spectrum linearly interpolated (in wavelength) onto a model grid."""
    return np.interp(wave_nm_model, wave_nm_ref, values, left=0.0, right=0.0)


def dex_error(t: np.ndarray, u: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    """Number of compared points and max |log10(t/u)| over ``mask`` where |u| is above floor."""
    t, u = np.asarray(t, float), np.asarray(u, float)
    peak = float(np.max(np.abs(u)))
    keep = mask & (np.abs(u) > REL_FLOOR * peak) & (np.abs(t) > 0)
    n = int(keep.sum())
    if n == 0:
        return 0, float("nan")
    with np.errstate(divide="ignore"):
        return n, float(np.max(np.abs(np.log10(np.abs(t[keep]) / np.abs(u[keep])))))


def netzer_template() -> tuple[np.ndarray, np.ndarray]:
    """tengri's Netzer template at the builder's disc parameters, on its native grid [nm]."""
    from tengri.components.agn.grahsp.disc import select_disc_model
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    t = load_grahsp_templates()
    idx = select_disc_model(t.disc_m, t.disc_a, t.disc_mdot, m="8.0", a="0", mdot="0.3")
    return np.asarray(t.disc_wave_nm), np.asarray(t.disc_lumin[idx])


def loglog(wave_nm, tw, tl) -> np.ndarray:
    """Power-law (log-log) interpolation of a template, the convention of tengri's resampler."""
    lx, ly = np.log(tw), np.log(tl)
    lq = np.log(np.asarray(wave_nm, dtype=float))
    i = np.clip(np.searchsorted(lx, lq) - 1, 0, lx.size - 2)
    frac = (lq - lx[i]) / (lx[i + 1] - lx[i])
    return np.exp(ly[i] + frac * (ly[i + 1] - ly[i]))


def line_centres_nm() -> np.ndarray:
    """Emission-line centres [nm] from tengri's repackaged line table (upstream's line list)."""
    with h5py.File(TEMPLATE_PATH, "r") as f:
        return np.asarray(f["netzer1990_lines"]["wave_nm"][:])


def line_windows(line_wave_nm: np.ndarray, linewidth_kms: float, nsig: float = 6.0):
    """Merged ``(lo, hi)`` windows [nm] of ``nsig`` sigma around each line centre."""
    sigma = line_wave_nm * (linewidth_kms / C_KMS) * FWHM_TO_SIGMA
    edges = sorted(zip(line_wave_nm - nsig * sigma, line_wave_nm + nsig * sigma))
    merged: list[list[float]] = []
    for lo, hi in edges:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def line_flux_centroid_width(wave_nm, y, windows):
    """Per-window flux, centroid [nm], and rms width / centroid, on the given grid."""
    flux, cen, wid = [], [], []
    for lo, hi in windows:
        m = (wave_nm >= lo) & (wave_nm <= hi)
        if m.sum() < 5:
            flux.append(0.0)
            cen.append(np.nan)
            wid.append(np.nan)
            continue
        i0 = np.trapezoid(y[m], wave_nm[m])
        mean = np.trapezoid(y[m] * wave_nm[m], wave_nm[m]) / i0
        var = np.trapezoid(y[m] * (wave_nm[m] - mean) ** 2, wave_nm[m]) / i0
        flux.append(i0)
        cen.append(mean)
        wid.append(math.sqrt(var) / mean)
    return np.array(flux), np.array(cen), np.array(wid)
