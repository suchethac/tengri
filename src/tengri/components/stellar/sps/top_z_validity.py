# SPDX-License-Identifier: BSD-3-Clause
"""Validity limit of an SSP grid's top metallicity node, and how the loader honors it.

A grid's highest-metallicity node can lack the most massive stars at young ages
because the isochrone set stops short of that metallicity. FSPS's PARSEC set is
the case on record: ``ISOCHRONES/PARSEC/isoc_z0.0600.dat`` caps the initial mass
at 12.01 Msun for every log age up to 7.15, because the PARSEC massive-star
tracks reach only Z = 0.04 [1]_. The Z = 0.06 node of every FSPS-PARSEC grid is
therefore ~800-2700x low in ionizing photons (Q_H(Z = 0.06) / Q_H(Z = 0.04) =
3.7e-4, 4.1e-4 and 1.3e-3 at log age 5.5, 6.0 and 6.5) and ~16x low at 1500 A.

This module holds that limit as registry data (keyed by the isochrone code
the surviving-mass registry already assigns each grid) and implements the two
policies :func:`tengri.load_ssp_data` offers through ``top_z=``:

``"clip"`` (default)
    The incomplete node is never used where it is incomplete. The effective
    metallicity is held at the highest complete node, with a taper (below) so the
    stellar population is continuous in age.
``"mist"``
    As ``"clip"``, but the highest complete node is a MIST [Z/H] = +0.5 SSP, inserted
    as an extra metallicity node at FSPS's ``zlegend`` value
    Z = Z_sun 10^[Z/H] = 0.0142 x 10^0.5 = 0.0449 [2]_ (Z_sun = 0.0142).
``"raw"``
    The file's node as shipped. For cross-code parity only.

With ``top_z="mist"`` the young part of the metallicity axis between Z = 0.040 (PARSEC)
and Z = 0.0449 (MIST) switches isochrone physics, not just metallicity: the model's
population at a given young age passes from the PARSEC to the MIST isochrones across that
interval. Measured on
``fsps_prsc_miles_chabrier`` (MIST node / PARSEC Z = 0.04 node):

=========  =====  ==========  ====================
log age    Q_H    L_nu(1500)  V-band mean L_nu
=========  =====  ==========  ====================
6.0        0.909  0.908       0.775
6.5        1.72   0.773       0.749
7.0        0.434  0.783       0.911
=========  =====  ==========  ====================

(V-band: mean L_nu over 5000-5900 A.) A gradient in ``met_logzsol`` over about 0.45-0.50
at young ages therefore reflects that switch and not the metallicity dependence of one
isochrone set. The default ``"clip"`` has no such switch.

References
----------
.. [1] Chen, Y., Bressan, A., Girardi, L., et al. 2015, MNRAS, 452, 1068,
       "PARSEC evolutionary tracks of massive stars up to 350 Msun at
       metallicities 0.0001 <= Z <= 0.04", arXiv:1506.01681,
       doi:10.1093/mnras/stv1281.
.. [2] Choi, J., Dotter, A., Conroy, C., et al. 2016, ApJ, 823, 102,
       "Mesa Isochrones and Stellar Tracks (MIST). I. Solar-scaled Models",
       arXiv:1604.08592, doi:10.3847/0004-637X/823/2/102. Z_sun = 0.0142
       (protosolar, Asplund et al. 2009) is the reference of [Z/H]; [Z/H] = +0.5
       is Z = 0.0142 x 10^0.5 = 0.0449.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from typing import NamedTuple

import h5py
import numpy as np

#: ``top_z=`` values accepted by the public loaders.
TOP_Z_CLIP = "clip"
TOP_Z_MIST = "mist"
TOP_Z_RAW = "raw"
TOP_Z_MODES = (TOP_Z_CLIP, TOP_Z_MIST, TOP_Z_RAW)

#: Metallicity nodes match the registry to this [dex].
LOGZ_ATOL = 1e-6
#: A grid age within this of a node-file age is that age [dex].
AGE_ATOL = 1e-5
#: Weights this close to 0 or 1 are those values (grid-age float rounding is ~1e-15).
_SNAP = 1e-9

_PACKAGE = "tengri.data.ssp_top_z"


class TopZValidity(NamedTuple):
    """What a loaded grid knows about its top node (hashable; never a JIT leaf).

    Parameters
    ----------
    node_log_z_abs : float
        The incomplete node [log10 absolute Z].
    fallback_log_z_abs : float
        The metallicity the node is clipped to at young ages [log10 absolute Z]: the
        highest complete node (``"clip"``) or the inserted MIST node (``"mist"``).
    lg_age_incomplete_max_yr : float
        Last age at which the node is incomplete [log10 yr].
    taper_dex : float
        Width of the age taper above the limit [dex].
    policy : str
        ``"clip"``, ``"mist"`` or ``"raw"``.
    source : str
        Provenance of the limit.
    """

    node_log_z_abs: float
    fallback_log_z_abs: float
    lg_age_incomplete_max_yr: float
    taper_dex: float
    policy: str
    source: str


class YoungTopMetallicityWarning(UserWarning):
    """A metallicity range reaches a top node that is incomplete at young ages."""


#: Half-width of the metallicity scatter kernel, in units of its sigma: DSPS's triweight
#: kernel (``dsps.utils._tw_cuml_kern``) is exactly 0 below -3 sigma and 1 above +3 sigma.
SCATTER_SPAN_SIGMA = 3.0


def reach_warning_text(
    validity: TopZValidity,
    log_z_abs_max: float,
    name: str,
    scatter_dex_max: float = 0.0,
) -> str | None:
    """Warning text for a metallicity that can exceed the highest complete node.

    Parameters
    ----------
    validity : TopZValidity
        The loaded grid's record.
    log_z_abs_max : float
        Highest absolute metallicity the parameter itself can take [log10 Z].
    name : str
        Parameter (or history) name for the message.
    scatter_dex_max : float, optional
        Highest lognormal metallicity scatter sigma the model can take [dex]. The scatter
        kernel spans ``SCATTER_SPAN_SIGMA`` sigma above the mean, so the metallicity
        the SSP lookup reaches is ``log_z_abs_max + SCATTER_SPAN_SIGMA * scatter_dex_max``.

    Returns
    -------
    str or None
        ``None`` when that reach stays at or below the highest complete node.
    """
    reach = log_z_abs_max + SCATTER_SPAN_SIGMA * scatter_dex_max
    if reach <= validity.fallback_log_z_abs + 1e-9:
        return None
    z_top, z_fb = 10.0**validity.node_log_z_abs, 10.0**validity.fallback_log_z_abs
    lg_hi = validity.lg_age_incomplete_max_yr
    lg_full = lg_hi + validity.taper_dex
    if scatter_dex_max > 0.0:
        reach_txt = (
            f"{name} reaches Z = {10.0**reach:.4f} (its upper bound, Z = "
            f"{10.0**log_z_abs_max:.4f}, plus {SCATTER_SPAN_SIGMA:g} x the metallicity-scatter "
            f"upper bound of {scatter_dex_max:g} dex that the scatter kernel spans)"
        )
    else:
        reach_txt = f"{name} reaches Z = {10.0**reach:.4f}"
    head = (
        f"{reach_txt}, above the highest complete node of this SSP grid (Z = {z_fb:.4f}). "
        f"The Z = {z_top:.4f} node has no stars above ~12 Msun for log10(age/yr) <= "
        f"{lg_hi:.2f}: {validity.source}. "
    )
    if validity.policy == TOP_Z_RAW:
        return head + (
            "top_z='raw' keeps that node as shipped, so young populations at this metallicity "
            "are missing their ionizing flux (Q_H ~800-2700x low, 1500 A ~16x low at log age "
            "5.5-6.5). Use the default top_z='clip'."
        )
    held = (
        f"Z = {z_fb:.4f} (a MIST [Z/H] = +0.5 node)"
        if validity.policy == TOP_Z_MIST
        else f"Z = {z_fb:.4f}"
    )
    tail = (
        f"The metallicity is therefore held at {held} for log10(age/yr) <= {lg_hi:.2f}, and the "
        f"top node's weight rises linearly with log age to 1 at {lg_full:.2f}; older "
        f"populations use the top node unchanged. "
    )
    if validity.policy == TOP_Z_CLIP:
        tail += (
            "To use a MIST [Z/H] = +0.5 node (FSPS zlegend Z = 0.0449) at young ages instead, "
            "load the grid with top_z='mist': tengri.load_ssp(<name>, top_z='mist') or "
            "load_ssp_data(<path>, top_z='mist'). That node has MIST isochrones, so between "
            "Z = 0.0400 and 0.0449 the young population switches isochrone physics, not only "
            "metallicity."
        )
    else:
        tail += (
            "Between Z = 0.0400 (PARSEC) and 0.0449 (MIST) the young population switches "
            "isochrone physics, not only metallicity; a gradient in met_logzsol there reflects "
            "that switch."
        )
    return head + tail


@dataclass(frozen=True)
class TopZLimit:
    """Registry row: where an isochrone set's top metallicity node is incomplete.

    Parameters
    ----------
    node_log_z_abs : float
        The incomplete node [log10 absolute Z].
    fallback_log_z_abs : float
        The highest complete node [log10 absolute Z].
    lg_age_incomplete_max_yr : float
        Last grid age at which the node is incomplete [log10 yr]; the node is
        complete above it.
    taper_dex : float
        Age span above the limit over which the node's weight rises from 0 to 1
        [dex]; the node is used in full from ``lg_age_incomplete_max_yr + taper_dex``.
        Must be positive (the weight divides by it).
    source : str
        Where the limit was measured.

    Raises
    ------
    ValueError
        ``taper_dex`` is not a positive finite number.
    """

    node_log_z_abs: float
    fallback_log_z_abs: float
    lg_age_incomplete_max_yr: float
    taper_dex: float
    source: str

    def __post_init__(self):
        if not (np.isfinite(self.taper_dex) and self.taper_dex > 0.0):
            raise ValueError(f"taper_dex must be a positive finite number, got {self.taper_dex!r}")


#: Isochrone code (``MassRemainingEntry.isoc``) -> limit.
TOP_Z_REGISTRY: dict[str, TopZLimit] = {
    "prsc": TopZLimit(
        node_log_z_abs=float(np.log10(0.06)),
        fallback_log_z_abs=float(np.log10(0.04)),
        lg_age_incomplete_max_yr=7.15,
        # One grid step (0.05 dex): FSPS's isoc_z0.0600.dat is incomplete at log age 7.15
        # (capped main-sequence tip, K-band ratio to Z = 0.04 of 0.833) and complete at
        # 7.20 (lifetime-limited, core He-burning tip 11.33 Msun, ratios to Z = 0.04
        # matching 7.25 and 7.30), so the node's weight is 0 at 7.15 and 1 at 7.20.
        taper_dex=0.05,
        source=(
            "FSPS ISOCHRONES/PARSEC/isoc_z0.0600.dat: Mini capped at 12.01 Msun for every "
            "log age 5.50-7.15 (343 rows at 5.50; isoc_z0.0200.dat reaches 300 Msun); the "
            "PARSEC massive-star tracks stop at Z = 0.04 (Chen et al. 2015, MNRAS 452, 1068)"
        ),
    ),
}


def top_node_weight(lg_age_yr, lg_age_incomplete_max_yr: float, taper_dex: float) -> np.ndarray:
    """Weight of the incomplete top node as a function of age.

    .. math::

        s(t) = \\mathrm{clip}\\left(\\frac{\\log_{10} t - L}{W},\\, 0,\\, 1\\right)

    where :math:`t` is the population age [yr], :math:`L` the last age at which the
    node is incomplete [log10 yr] and :math:`W` the taper width [dex].

    Parameters
    ----------
    lg_age_yr : array_like
        Population age [log10 yr]; ``-inf`` (t = 0) gives 0.
    lg_age_incomplete_max_yr : float
        :math:`L` [log10 yr].
    taper_dex : float
        :math:`W` [dex].

    Returns
    -------
    ndarray
        :math:`s` in [0, 1], same shape as ``lg_age_yr`` [dimensionless].

    Notes
    -----
    The node's metallicity weight in the interpolation is multiplied by :math:`s`
    and the highest complete node receives the rest, so the top node is never used
    at or below :math:`L` (exact zero), is used in full from :math:`L + W` (exact
    one, bit-for-bit the file's values), and the weights are continuous (piecewise
    linear) in between: there is no jump at :math:`L`. Evaluated once on the host at
    load; the SSP arrays it produces are ordinary constants, so the forward model
    stays JIT/grad/vmap-safe with no data-dependent branch.
    """
    age = np.asarray(lg_age_yr, dtype=np.float64)
    return np.clip((age - lg_age_incomplete_max_yr) / taper_dex, 0.0, 1.0)


def limit_for_isochrones(isoc: str | None) -> TopZLimit | None:
    """Registry limit of an isochrone code, or ``None`` when none is on record."""
    return None if isoc is None else TOP_Z_REGISTRY.get(isoc)


def mist_node_filename(stem: str) -> str:
    """Package-data file holding the MIST node for a grid stem.

    Parameters
    ----------
    stem : str
        Grid file stem, e.g. ``fsps_prsc_miles_chabrier`` or
        ``ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0``.

    Returns
    -------
    str
        ``mist_top_node_<library>_<imf>[_wNE<suffix>].h5``.
    """
    base, sep, neb = stem.partition("_wNE")
    tokens = base.split("_")
    library, imf = tokens[2], tokens[-1]
    return f"mist_top_node_{library}_{imf}{sep}{neb}.h5"


class MistNode(NamedTuple):
    """The MIST top node read from package data.

    Attributes
    ----------
    log_z_abs : float
        Metallicity of the node [log10 absolute Z]: FSPS's ``zlegend`` value
        Z = Z_sun 10^[Z/H] = 0.0449 (Z_sun = 0.0142; Choi et al. 2016).
    lg_age_yr : ndarray, shape (n_age,)
        Ages, equal to the PARSEC grid's [log10 yr].
    wave : ndarray, shape (n_wave,)
        Rest wavelength [Angstrom].
    flux : ndarray, shape (n_age, n_wave)
        SSP spectra in FSPS's native unit [Lsun/Hz per Msun formed].
    mass_remaining : ndarray, shape (n_age,)
        Surviving mass fraction of the same FSPS build, remnants included [dimensionless].
    filename : str
        Package-data file the node was read from.
    """

    log_z_abs: float
    lg_age_yr: np.ndarray
    wave: np.ndarray
    flux: np.ndarray
    mass_remaining: np.ndarray
    filename: str


def load_mist_node(stem: str) -> MistNode:
    """Read the MIST top-node file for a grid stem.

    Parameters
    ----------
    stem : str
        Grid file stem, e.g. ``fsps_prsc_miles_chabrier``; the node file is chosen
        by its spectral library, IMF and nebular setting (:func:`mist_node_filename`).

    Returns
    -------
    MistNode
        The node's metallicity, age and wavelength axes, spectra and surviving mass.

    Raises
    ------
    FileNotFoundError
        No node was built for this spectral library / nebular setting; the
        message lists the nodes that ship.
    """
    name = mist_node_filename(stem)
    resource = files(_PACKAGE) / name
    if not resource.is_file():
        shipped = sorted(p.name for p in files(_PACKAGE).iterdir() if p.name.endswith(".h5"))
        raise FileNotFoundError(
            f"top_z='mist' has no node for '{stem}' (looked for {name}). "
            f"Shipped nodes: {shipped}. Build another with scripts/build_mist_top_z_node.py."
        )
    with as_file(resource) as path, h5py.File(path, "r") as f:
        return MistNode(
            float(f["log10_z_abs"][()]),
            np.array(f["log10_age_yr"][:], dtype=np.float64),
            np.array(f["ssp_wave"][:], dtype=np.float64),
            np.array(f["ssp_flux"][:], dtype=np.float64),
            np.array(f["mass_remaining"][:], dtype=np.float64),
            name,
        )


def _snap(s: np.ndarray) -> np.ndarray:
    """Put nodes within rounding of a ramp end exactly on it.

    Grid ages come from ``log10(age/Gyr) + 9`` and carry ~1e-15 of float error, so a
    node sitting on the limit or on the end of the taper would otherwise get 4e-15 or
    0.9999999999999964 and not reproduce the file's values bit for bit.
    """
    return np.where(s < _SNAP, 0.0, np.where(s > 1.0 - _SNAP, 1.0, s))


def _node_index(lgmet: np.ndarray, target: float) -> int | None:
    i = int(np.argmin(np.abs(lgmet - target)))
    return i if abs(float(lgmet[i]) - target) <= LOGZ_ATOL else None


def apply_top_z_policy(
    *,
    stem: str,
    isoc: str | None,
    policy: str,
    wave: np.ndarray,
    lg_age_yr: np.ndarray,
    lgmet: np.ndarray,
    flux: np.ndarray,
    mass: np.ndarray,
    flux_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TopZValidity | None]:
    """Apply the ``top_z=`` policy to a freshly loaded grid.

    Parameters
    ----------
    stem : str
        Grid file stem.
    isoc : str or None
        Isochrone code from the surviving-mass registry (``None``: unregistered).
    policy : {"clip", "mist", "raw"}
        See the module docstring.
    wave : ndarray, shape (n_wave,)
        Rest wavelength [Angstrom].
    lg_age_yr : ndarray, shape (n_age,)
        Ages [log10 yr].
    lgmet : ndarray, shape (n_met,)
        Absolute metallicity nodes [log10 Z].
    flux : ndarray, shape (n_met, n_age, n_wave)
        SSP spectra [Lsun/Hz/Msun in the file's unit].
    mass : ndarray, shape (n_met, n_age)
        Surviving mass fraction [dimensionless].
    flux_scale : float, optional
        Factor the loader applied to the file's flux unit; the MIST node gets the same.

    Returns
    -------
    flux, mass, lgmet, validity
        The (possibly new) arrays and the validity record, ``None`` when the grid has no
        limit on record or does not contain the incomplete node.

    Raises
    ------
    ValueError
        Unknown policy; ``"mist"`` on a grid whose node file disagrees with it in
        ages or wavelengths.
    FileNotFoundError
        ``"mist"`` with no node file for this grid.

    Notes
    -----
    The result is exact: for ages above the taper the arrays are the input's,
    value for value. See :func:`top_node_weight` for the weights.
    """
    if policy not in TOP_Z_MODES:
        raise ValueError(f"top_z must be one of {TOP_Z_MODES}, got {policy!r}")
    limit = limit_for_isochrones(isoc)
    if limit is None:
        if policy == TOP_Z_MIST:
            raise ValueError(
                f"top_z='mist' needs a grid with a recorded top-metallicity limit; '{stem}' has "
                f"none (registered: {sorted(TOP_Z_REGISTRY)} isochrones)."
            )
        return flux, mass, lgmet, None
    i_top = _node_index(lgmet, limit.node_log_z_abs)
    i_fb = _node_index(lgmet, limit.fallback_log_z_abs)
    if i_top is None or i_fb is None or i_top != lgmet.shape[0] - 1:
        return flux, mass, lgmet, None  # the incomplete node is not in this grid

    def record(fallback: float, pol: str) -> TopZValidity:
        return TopZValidity(
            limit.node_log_z_abs,
            fallback,
            limit.lg_age_incomplete_max_yr,
            limit.taper_dex,
            pol,
            limit.source,
        )

    if policy == TOP_Z_RAW:
        return flux, mass, lgmet, record(limit.fallback_log_z_abs, TOP_Z_RAW)

    s = _snap(top_node_weight(lg_age_yr, limit.lg_age_incomplete_max_yr, limit.taper_dex))
    fallback_z = limit.fallback_log_z_abs
    if policy == TOP_Z_MIST:
        flux, mass, lgmet, i_fb = _insert_mist_node(
            stem, wave, lg_age_yr, lgmet, flux, mass, i_fb, i_top, s, flux_scale
        )
        fallback_z = float(lgmet[i_fb])
        i_top = lgmet.shape[0] - 1

    full = (s >= 1.0)[:, None]
    new_top = np.where(
        full, flux[i_top], (1.0 - s)[:, None] * flux[i_fb] + s[:, None] * flux[i_top]
    )
    new_top_mass = np.where(s >= 1.0, mass[i_top], (1.0 - s) * mass[i_fb] + s * mass[i_top])
    flux = np.concatenate([flux[:i_top], new_top[None]], axis=0)
    mass = np.concatenate([mass[:i_top], new_top_mass[None]], axis=0)
    return flux, mass, lgmet, record(fallback_z, policy)


def _insert_mist_node(stem, wave, lg_age_yr, lgmet, flux, mass, i_fb, i_top, s, flux_scale):
    """Insert the MIST node between the fallback and the top node.

    The node is MIST at ages up to the limit and blends to the PARSEC log-Z
    interpolation at its own Z with the same weights ``s`` as the top node, so that
    above the taper it equals what the model would have produced there anyway.
    """
    node = load_mist_node(stem)
    z_lo, z_hi = float(lgmet[i_fb]), float(lgmet[i_top])
    if not z_lo < node.log_z_abs < z_hi:
        raise ValueError(
            f"{node.filename}: MIST node log10 Z = {node.log_z_abs} is not between the "
            f"grid's fallback ({z_lo}) and top ({z_hi}) nodes."
        )
    if node.wave.shape != wave.shape or not np.allclose(node.wave, wave, rtol=0, atol=1e-6):
        raise ValueError(f"{node.filename}: wavelength grid differs from the SSP grid's.")
    if node.lg_age_yr.shape != lg_age_yr.shape or not np.allclose(
        node.lg_age_yr, lg_age_yr, rtol=0, atol=AGE_ATOL
    ):
        raise ValueError(f"{node.filename}: age grid differs from the SSP grid's.")

    f = (node.log_z_abs - z_lo) / (z_hi - z_lo)
    interp_flux = (1.0 - f) * flux[i_fb] + f * flux[i_top]
    interp_mass = (1.0 - f) * mass[i_fb] + f * mass[i_top]
    mist_flux = node.flux * flux_scale
    full = (s >= 1.0)[:, None]
    new_flux = np.where(
        full, interp_flux, (1.0 - s)[:, None] * mist_flux + s[:, None] * interp_flux
    )
    new_mass = np.where(s >= 1.0, interp_mass, (1.0 - s) * node.mass_remaining + s * interp_mass)
    at = i_top
    flux = np.concatenate([flux[:at], new_flux[None], flux[at:]], axis=0)
    mass = np.concatenate([mass[:at], new_mass[None], mass[at:]], axis=0)
    lgmet = np.concatenate([lgmet[:at], [node.log_z_abs], lgmet[at:]])
    return flux, mass, lgmet, at
