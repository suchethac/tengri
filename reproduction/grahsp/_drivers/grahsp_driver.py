# SPDX-License-Identifier: BSD-3-Clause
"""Reader for upstream GRAHSP's committed spectra, and tengri's evaluation at matched inputs.

Upstream GRAHSP (Buchner et al. 2024, A&A 692, A161; arXiv:2405.19297) is a pcigale module
set, not a pip package. ``data/grahsp_upstream_reference.h5`` holds its own spectra for 38
parameter sets (a fiducial plus one-at-a-time sweeps), built by
``scripts/build_grahsp_reference.py``. This driver reads that file and nothing else: upstream
GRAHSP is never imported at render time.

Units. Wavelengths are nm. Upstream arrays are per-nm luminosity densities
(:math:`L_\\lambda` [erg/s/nm] at the unit normalization ``lum5100A = 1``), so
:math:`\\lambda L_\\lambda(510\\,\\mathrm{nm}) = 1` on both sides and no conversion applies.

Explicit corrections. Each upstream error the page discusses is applied here as a named
conversion, never inside ``src/tengri``:

* ``LINE_FLUX_SCALE`` (:math:`\\sqrt{2}`): upstream integrates each emission line to
  :math:`\\sqrt{2}` times the unit area the paper's normalization gives; tengri's lines are
  multiplied by this factor before comparison.
* ``WIDTH_SCALE`` (:math:`1/\\sqrt{2}`): upstream's torus log-Gaussian is
  :math:`\\exp(-x^2/W^2)`; Eq. 2 of the paper has :math:`\\exp(-x^2/(2W^2))`, so the width fed
  to tengri is :math:`W_\\mathrm{tengri} = W_\\mathrm{upstream}/\\sqrt{2}`.

References
----------
.. [1] Buchner, J., Starck, H., Salvato, M., et al. 2024, A&A 692, A161, "Genuine Retrieval of
   the AGN Host Stellar Population (GRAHSP)", arXiv:2405.19297, doi:10.1051/0004-6361/202449372.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from tengri.components.agn.grahsp.model import GRAHSPSED, GRAHSPParams, evaluate_grahsp_agn
from tengri.components.agn.grahsp.templates import GRAHSPTemplates, load_grahsp_templates

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = REPO_ROOT / "data" / "grahsp_upstream_reference.h5"
TEMPLATE_PATH = REPO_ROOT / "data" / "grahsp" / "grahsp_templates.h5"

LINE_FLUX_SCALE = math.sqrt(2.0)  # upstream lines carry sqrt(2) x the unit-area normalization
WIDTH_SCALE = 1.0 / math.sqrt(2.0)  # upstream torus W -> tengri W (paper Eq. 2 exponent)
L5100_UNIT = 1.0  # upstream lum5100A = 1: lambda L_lambda(510 nm) = 1 on both sides
XRAY_FLOOR_NM = 12.4  # tengri zeroes the disc below this wavelength (corona overlap)
TORUS_WINDOW_NM = (400.0, 99000.0)  # upstream torus is hard zero outside 0.36-100 um
C_NM_PER_S = 2.99792458e17
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
C_KMS = 299792.458
REL_FLOOR = 1e-6  # compare only where |upstream| exceeds this fraction of its own peak

# upstream attribute -> (GRAHSPParams field, conversion)
_FEII_NAMES = {"BruhweilerVerner08": "bruhweiler2008", "Veron-Cetty04": "veroncetty2004"}
PARAM_MAP = {
    "plslope": ("plslope", float),
    "uvslope": ("uvslope", float),
    "plbendloc": ("plbendloc_nm", float),
    "plbendwidth": ("plbendwidth", float),
    "cutoff": ("cutoff_nm", float),
    "afeii": ("a_feii", float),
    "alines": ("a_lines", float),
    "linewidth": ("linewidth_kms", float),
    "type": ("agn_type", int),
    "fcov": ("fcov", float),
    "si": ("si", float),
    "coollam": ("cool_lam_um", float),
    "coolwidth": ("cool_width", lambda x: float(x) * WIDTH_SCALE),
    "hotlam": ("hot_lam_um", float),
    "hotwidth": ("hot_width", lambda x: float(x) * WIDTH_SCALE),
    "hotfcov": ("hot_fcov", float),
    "ebv": ("ebv", float),
    "ebv_agn": ("ebv_agn", float),
    "abc": ("a_bc", float),
    "feii": ("feii_template", lambda x: _FEII_NAMES[str(x)]),
}

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
UPSTREAM_510_FACTOR = 510.0  # upstream lambda L_lambda(510 nm) = 510 at l5100 = 1 (see 3.5)


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
    attenuation: (
        dict  # short component name -> upstream signed difference (attenuated - intrinsic)
    )


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
    attenuation = {
        short: raw.get("attenuation." + key, np.zeros_like(wave))
        for short, key in COMPONENTS.items()
    }
    return UpstreamSet(name, params, wave, spectra, attenuation)


def tengri_params(up: dict, *, width_correction: bool = True) -> GRAHSPParams:
    """Translate upstream parameters into :class:`GRAHSPParams`.

    ``width_correction=False`` passes the upstream torus width through unchanged, which is how
    the page shows the effect of the missing factor of one half.
    """
    kwargs = {}
    for key, (field, conv) in PARAM_MAP.items():
        if key not in up:
            continue
        if not width_correction and key in ("coolwidth", "hotwidth"):
            kwargs[field] = float(up[key])
        else:
            kwargs[field] = conv(up[key])
    return GRAHSPParams(l5100=L5100_UNIT, **kwargs)


@functools.cache
def templates() -> GRAHSPTemplates:
    """Cached tengri template bundle (repackaged upstream tables)."""
    return load_grahsp_templates()


def evaluate(wave_nm, params: GRAHSPParams) -> GRAHSPSED:
    """tengri's GRAHSP components on ``wave_nm`` [nm], via the public component function."""
    return evaluate_grahsp_agn(jnp.asarray(wave_nm), params, templates())


def torus_on_upstream_nodes(params: GRAHSPParams) -> tuple[np.ndarray, np.ndarray]:
    """tengri's torus on upstream's template nodes, the grid upstream interpolates from.

    Returns ``(node_wave_nm, torus_L_lambda)``. Upstream's torus is tabulated at its template
    nodes and linearly interpolated, so the comparison interpolates tengri the same way.
    """
    with h5py.File(TEMPLATE_PATH, "r") as f:
        nodes = np.asarray(f["torus"]["wave_nm"][:])
    sed = evaluate(nodes, params)
    return nodes, np.asarray(sed.torus)


def line_nodes_and_values(params: GRAHSPParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upstream's line sampling and tengri's broad and narrow lines on those nodes.

    Upstream evaluates the summed lines on nine nodes per line (centre plus or minus three FWHM)
    and interpolates linearly, using the float32 line table. tengri is evaluated at the same
    nodes with the same float32 centres. Returns ``(nodes_nm, broad, narrow)``, before the
    :data:`LINE_FLUX_SCALE` correction.
    """
    t = templates()
    centres32 = (np.asarray(t.line_wave_nm) * 10.0).astype(np.float32) * np.float32(0.1)
    chunks = []
    for lam in centres32:
        width = lam * (params.linewidth_kms * 1000.0) / np.float32(2.99792458e8)
        chunks.append(np.linspace(lam - 3.0 * width, lam + 3.0 * width, 9))
    nodes = np.unique(np.concatenate(chunks).astype(np.float64))
    t32 = dataclasses.replace(t, line_wave_nm=centres32.astype(np.float64))
    sed = evaluate_grahsp_agn(jnp.asarray(nodes), params, t32)
    return nodes, np.asarray(sed.broad_lines), np.asarray(sed.narrow_lines)


NETZER_DISC_PARAMS = {"disc_model": "netzer", "disc_m": "8.0", "disc_a": "0", "disc_mdot": "0.3"}
MN12_PARAMS = {"torus_model": "mn12", "tor_temp": 0.0, "tor_cutoff_um": 1.2}


def netzer_template() -> tuple[np.ndarray, np.ndarray]:
    """tengri's Netzer template at the builder's disc parameters, on its native grid [nm]."""
    from tengri.components.agn.grahsp.disc import select_disc_model

    t = templates()
    idx = select_disc_model(t.disc_m, t.disc_a, t.disc_mdot, m="8.0", a="0", mdot="0.3")
    return np.asarray(t.disc_wave_nm), np.asarray(t.disc_lumin[idx])


def loglog(wave_nm, tw, tl) -> np.ndarray:
    """Power-law (log-log) interpolation of a template, the convention of tengri's resampler."""
    lx, ly = np.log(tw), np.log(tl)
    lq = np.log(np.asarray(wave_nm, dtype=float))
    i = np.clip(np.searchsorted(lx, lq) - 1, 0, lx.size - 2)
    frac = (lq - lx[i]) / (lx[i + 1] - lx[i])
    return np.exp(ly[i] + frac * (ly[i + 1] - ly[i]))


def dex_error(t: np.ndarray, u: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    """Number of compared points and max |log10(t/u)| over ``mask`` where |u| is above floor."""
    t, u = np.asarray(t, float), np.asarray(u, float)
    peak = float(np.max(np.abs(u)))
    keep = mask & (np.abs(u) > REL_FLOOR * peak)
    n = int(keep.sum())
    if n == 0:
        return 0, float("nan")
    with np.errstate(divide="ignore"):
        return n, float(np.max(np.abs(np.log10(np.abs(t[keep]) / np.abs(u[keep])))))


def integrate_nu_to_lnu(wave_nm: np.ndarray, lam_l_lam: np.ndarray) -> np.ndarray:
    """Convert L_lambda [erg/s/nm] on a wavelength grid to L_nu [erg/s/Hz]."""
    return lam_l_lam * wave_nm**2 / C_NM_PER_S


def lam_l_lam_at(wave_nm, values, lam_nm: float) -> float:
    """lambda L_lambda at ``lam_nm`` from a grid, for unsorted grids."""
    o = np.argsort(np.asarray(wave_nm))
    return float(lam_nm * np.interp(lam_nm, np.asarray(wave_nm)[o], np.asarray(values)[o]))


def upstream_window_mask(wave_nm: np.ndarray, component: str) -> np.ndarray:
    """Support of each upstream component: the torus window, and the disc above the X-ray floor."""
    if component == "torus":
        lo, hi = TORUS_WINDOW_NM
        return (wave_nm >= lo) & (wave_nm <= hi)
    if component == "disc":
        return wave_nm >= XRAY_FLOOR_NM
    return np.ones_like(wave_nm, dtype=bool)
