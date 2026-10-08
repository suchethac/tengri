# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapters for analytic accretion disc models.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
three disc models:

1. **powerlaw_disc**: Simple power-law optical/UV continuum with exponential
   UV cutoff. One free axis: ``agn_alpha`` (power-law index).

2. **disc_ss** (Shakura-Sunyaev): Multi-color thin disc with temperature
   gradient. Two free axes: ``agn_log_mbh``, ``agn_log_lbol`` (the post-#846
   shape driver, the Eddington ratio, hence the disc temperature profile, is
   derived from ``agn_log_lbol`` and ``agn_log_mbh``).

3. **cigale_disc** (piecewise power-law from CIGALE) (not one of CIGALE's
   `disk_type` discs; see #2670): Empirical disc model with fixed wavelength
   breakpoints and power-law segments. No free axes (shape is fixed; only
   ``agn_log_lbol`` scales at runtime).

Each model is preintegrated through filter curves at model-initialization time.
Auto-collapses axes whose corresponding parameters are ``Fixed`` in the user's
``Parameters``: e.g., a user who pins ``agn_alpha`` gets a scalar template.

References
----------
.. [1] A. Kubota and C. Done, "A physical model of the broad-band continuum
   of AGN and its implications for the UV/X relation and optical variability,"
   MNRAS, 480, 1247 (2018). arXiv:1804.00171.
   https://doi.org/10.1093/mnras/sty1890
.. [2] J. M. Bardeen, W. H. Press, and S. A. Teukolsky, "Rotating black holes:
   Locally nonrotating frames, energy extraction, and scalar synchrotron radiation,"
   ApJ, 178, 347 (1972). https://doi.org/10.1086/151796
.. [3] A. Laor and H. Netzer, "Massive thin accretion discs – I. Calculated spectra,"
   MNRAS, 238, 897 (1989). https://doi.org/10.1093/mnras/238.3.897
.. [4] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy Emission,"
   A&A, 622, A103 (2019). https://doi.org/10.1051/0004-6361/201834156
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._params import DEFAULT_AGN_LUM_RATIO, PARAMS as _AGN_PARAMS
from tengri.components.agn.disc import (
    multicolor_disc as _multicolor_disc,
    powerlaw_disc as _powerlaw_disc,
)
from tengri.components.agn.disc_cigale import schartmann2005_disk_spectrum, skirtor_disk_spectrum
from tengri.forward.precompute import reach_axes
from tengri.forward.precompute.templates import (
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import PreintegratedGrid, interp_nd_pchip
from tengri.utils.physics_constants import C_AA, L_SUN

# ── Axis definitions per model ──────────────────────────────────


# powerlaw_disc: parametrized by agn_alpha (power-law index)
AXIS_PARAMS_POWERLAW = ("agn_alpha",)

# disc_ss (Shakura-Sunyaev): parametrized by BH mass and bolometric luminosity.
# Since #846 the disc shape is self-consistent with agn_log_lbol (the Eddington
# ratio, and hence T_in / r_out, is DERIVED from L_bol and M_bh), so L_bol is a
# genuine shape axis. The former agn_log_mdot axis fed the now-ignored
# agn_log_ledd and was silently degenerate (#902).
AXIS_PARAMS_SS = ("agn_log_mbh", "agn_log_lbol")

# cigale_disc: the slope modulator delta is the one axis; disk_type is a build-time choice.
AXIS_PARAMS_CIGALE = ("agn_cigale_disk_delta",)

# Protocol-required dict form for multi-model module (see test_precompute_protocol.py).
AXIS_PARAMS: dict[str, tuple[str, ...]] = {
    "powerlaw_disc": AXIS_PARAMS_POWERLAW,
    "ss_disc": AXIS_PARAMS_SS,
    "cigale_disc": AXIS_PARAMS_CIGALE,
}


def _build_grid_powerlaw(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    alpha_grid: np.ndarray,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_T_max: float = 1e5,
) -> PreintegratedGrid:
    """Preintegrate powerlaw_disc over a 1D grid of alpha values.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter wavelength arrays [Angstrom].
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    redshift : float
        Source redshift.
    alpha_grid : ndarray, shape (n_alpha,)
        Power-law spectral index grid.
    agn_lum_ratio : float
        Disc fraction. Default 1.0.
    agn_T_max : float
        UV cutoff temperature [K]. Default 1e5.

    Returns
    -------
    PreintegratedGrid
        Preintegrated photometry with shape (n_alpha, n_filters).
    """
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64)

    # Standard rest-frame wavelength grid for integration
    # (covers ~10 Angstrom to ~1 mm with fine sampling)
    wave_rest = np.logspace(1, 5, 1000, dtype=np.float64)

    # Precompute L_nu for each alpha value
    phot_grid = []
    for alpha in alpha_grid:
        # Per L_sun (agn_log_lbol = 0); the lookup scales it by 10**agn_log_lbol.
        l_nu = np.asarray(
            _powerlaw_disc(
                jnp.asarray(wave_rest),
                agn_log_lbol=0.0,
                agn_lum_ratio=agn_lum_ratio,
                agn_alpha=float(alpha),
                agn_T_max=agn_T_max,
            )
        )
        phot_grid.append(l_nu)

    templates = np.array(phot_grid, dtype=np.float64)  # (n_alpha, n_wave)

    # Preintegrate through filters using template helper
    return precompute_template_photometry(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(alpha_grid,),
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=False,  # per L_sun by construction (agn_log_lbol = 0 above)
        units="lnu",
    )


def _build_grid_ss(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    mbh_grid: np.ndarray,
    lbol_grid: np.ndarray,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
) -> PreintegratedGrid:
    """Preintegrate disc_ss (Shakura-Sunyaev) over 2D grid of (M_bh, L_bol).

    Both axes drive the disc *shape*: since #846 the Eddington ratio (hence the
    temperature profile) is derived from ``agn_log_lbol`` and ``agn_log_mbh``.
    Templates are energy-normalized to unit bolometric luminosity so the grid
    captures pure shape variation; the absolute normalization is reintroduced by
    the runtime ``agn_log_lbol`` scaling in :func:`build_lookup`. Varying L_bol
    here (rather than the now-ignored ``agn_log_ledd``) fixes the silently
    degenerate second axis of #902.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter wavelength arrays [Angstrom].
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    redshift : float
        Source redshift.
    mbh_grid : ndarray, shape (n_mbh,)
        Black hole mass grid [log10(M_sun)].
    lbol_grid : ndarray, shape (n_lbol,)
        Bolometric luminosity grid [log10(L_sun)]. Drives the disc temperature
        profile via the derived Eddington ratio.
    agn_lum_ratio : float
        Disc fraction. Default 1.0.

    Returns
    -------
    PreintegratedGrid
        Preintegrated photometry with shape (n_mbh, n_lbol, n_filters).
    """
    mbh_grid = np.asarray(mbh_grid, dtype=np.float64)
    lbol_grid = np.asarray(lbol_grid, dtype=np.float64)
    n_mbh = len(mbh_grid)
    n_lbol = len(lbol_grid)

    wave_rest = np.logspace(1, 5, 1000, dtype=np.float64)

    phot_grid = []
    for mbh in mbh_grid:
        for lbol in lbol_grid:
            # agn_log_lbol drives both normalization and (post-#846) the shape.
            l_nu = np.asarray(
                _multicolor_disc(
                    jnp.asarray(wave_rest),
                    agn_log_lbol=float(lbol),
                    agn_lum_ratio=agn_lum_ratio,
                    agn_log_mbh=float(mbh),
                    agn_a_spin=0.0,  # non-spinning for simplicity
                    agn_cos_inc=0.5,  # 60 degree inclination
                    n_radii=50,
                )
            )
            phot_grid.append(l_nu)

    templates = np.array(phot_grid, dtype=np.float64).reshape(n_mbh, n_lbol, len(wave_rest))

    return precompute_template_photometry(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(mbh_grid, lbol_grid),
        redshift=redshift,
        dl_cm=1.0,
        # Shape-only grid: unit-bolometric templates; runtime agn_log_lbol
        # reintroduces the absolute scale (no double-count).
        energy_normalize=True,
        units="lnu",
    )


# Rest-frame grid of the cigale_disc template [Angstrom]. The CIGALE discs span [8, 1e6] nm, so
# the grid covers 10 Angstrom to 1e7 Angstrom and the energy normalization is taken on it.
_CIGALE_REST_WAVE_AA = np.logspace(1, 7, 2000, dtype=np.float64)

# CIGALE disc of each skirtor2016 disk_type: 0 = SKIRTOR, 1 = Schartmann 2005.
_CIGALE_DISKS = (skirtor_disk_spectrum, schartmann2005_disk_spectrum)


def _cigale_template(disk_type: int, delta: float) -> np.ndarray:
    """Per-L_sun L_nu of one CIGALE disc on the cigale rest grid [s].

    The disc density is per nm and unit-area. Per Angstrom it is the per-nm value over 10, and
    L_nu = L_lambda * lambda^2 / c with lambda in Angstrom and c in Angstrom/s.

    Parameters
    ----------
    disk_type : int
        0 for SKIRTOR, 1 for Schartmann 2005.
    delta : float
        Slope modulator of the disc, dimensionless.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu per unit bolometric luminosity on ``_CIGALE_REST_WAVE_AA`` [s].
    """
    wave_aa = _CIGALE_REST_WAVE_AA
    density_nm = np.asarray(_CIGALE_DISKS[disk_type](jnp.asarray(wave_aa / 10.0), delta=delta))
    return (density_nm / 10.0) * wave_aa**2 / C_AA


def _build_grid_cigale(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    disk_type: int,
    delta_grid: np.ndarray,
) -> PreintegratedGrid:
    """Preintegrate the CIGALE disc of ``disk_type`` over a grid of the slope modulator delta.

    Each template is energy-normalized to unit bolometric luminosity on the rest grid, so the
    runtime ``agn_log_lbol`` sets the absolute scale, as in :func:`_build_grid_ss`.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter wavelength arrays [Angstrom].
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    redshift : float
        Source redshift.
    disk_type : int
        0 for SKIRTOR, 1 for Schartmann 2005.
    delta_grid : ndarray, shape (n_delta,)
        Slope modulator nodes, dimensionless.

    Returns
    -------
    PreintegratedGrid
        Preintegrated photometry with shape (n_delta, n_filters).
    """
    delta_grid = np.asarray(delta_grid, dtype=np.float64)
    templates = np.array([_cigale_template(disk_type, float(d)) for d in delta_grid])
    return precompute_template_photometry(
        templates=templates,
        wave_rest=_CIGALE_REST_WAVE_AA,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(delta_grid,),
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=True,
        units="lnu",
    )


# Node counts per axis at the declared prior range. The alpha axis keeps the spacing
# (2 / 14) of the first literal grid; the ss_disc log axes take 0.25 dex spacing, the
# density at which the node-exact PCHIP holds 2e-3 off-node (9-13 nodes give 3e-2).
# Accuracy is stated with :func:`build_lookup`.
_DEFAULT_NODES: dict[str, int] = {
    "agn_alpha": 15,
    "agn_log_mbh": 17,
    "agn_log_lbol": 25,
    "agn_cigale_disk_delta": 9,
}

# Luminosity each template is scaled by at runtime, in erg/s per unit of 10**agn_log_lbol.
# ``powerlaw_disc`` templates are per L_sun; ``ss_disc`` templates are energy-normalized to
# unit bolometric power in erg/s, so the runtime scale is the bolometric power in erg/s.
_LOOKUP_UNIT_ERG = {"powerlaw_disc": 1.0, "ss_disc": L_SUN, "cigale_disc": L_SUN}

# The ln of a band flux is taken in float64 at build, so its floor is the float64 smallest normal.
_FLOAT64_TINY = np.finfo(np.float64).tiny


def _axis(param_name: str, supplied: Any, parameters: Any) -> np.ndarray:
    """Node axis of ``param_name``: the supplied one checked against the reach, or the default.

    The default spans the declared prior extended to the model's reach, at
    :data:`_DEFAULT_NODES` nodes over the declared range (see ``reach_axes.default_axis``).
    """
    declared = reach_axes.declared_bounds(_AGN_PARAMS, param_name)
    support = reach_axes.active_support(param_name, parameters, declared)
    if supplied is None:
        return reach_axes.default_axis(
            param_name, _DEFAULT_NODES[param_name], support, declared=declared
        )
    axis = np.asarray(supplied, dtype=np.float64)
    reach_axes.check_user_axis(param_name, axis, support)
    return axis


# ── Protocol-shaped entry points ──────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any,
    *,
    model: str = "powerlaw_disc",
    alpha_grid: np.ndarray | None = None,
    mbh_grid: np.ndarray | None = None,
    lbol_grid: np.ndarray | None = None,
    delta_grid: np.ndarray | None = None,
    disk_type: int = 0,
) -> dict:
    """Build preintegrated disc grid, auto-collapsing Fixed-parameter axes.

    Multi-model entry point. Dispatches to the appropriate builder based
    on ``model`` parameter.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0–1).
    redshift : float
        Source redshift. [dimensionless]
    parameters : Parameters | None
        Parameters spec, used to detect Fixed-axis parameters.
    model : str, keyword-only
        One of "powerlaw_disc", "ss_disc", "cigale_disc".
        Default: "powerlaw_disc".
    alpha_grid : ndarray, optional
        Grid for agn_alpha (powerlaw_disc only). If None, the declared prior [-2, 0] with
        15 nodes, extended to the parameter's reach (``Fixed`` value or finite prior bounds).
    mbh_grid : ndarray, optional
        Grid for agn_log_mbh (ss_disc only). If None, the declared prior [6, 10] with 17 nodes,
        extended to the reach.
    lbol_grid : ndarray, optional
        Grid for agn_log_lbol (ss_disc only) [log10(L_sun)]. If None, the declared prior
        [8, 14] with 25 nodes, extended to the reach. A supplied grid is checked
        against the reach and refused if it does not span it.
    delta_grid : ndarray, optional
        Grid for agn_cigale_disk_delta (cigale_disc only), dimensionless. If None, the declared
        prior [-0.5, 0.5] with 9 nodes, extended to the reach. A Fixed delta collapses the axis.
    disk_type : int, optional
        cigale_disc only: 0 for the SKIRTOR disc (default), 1 for Schartmann 2005. A build-time
        choice, not a lookup axis.

    Returns
    -------
    dict
        Keys: "grid_phot" (photometry array), "axes" (free axes), "_preint"
        (PreintegratedGrid), optionally "_collapsed_axes" (if any axes fixed).

    References
    ----------
    .. [1] A. Kubota and C. Done, "A physical model of the broad-band continuum
       of AGN and its implications for the UV/X relation and optical variability,"
       MNRAS, 480, 1247 (2018). arXiv:1804.00171.
    .. [2] M. Boquien et al., "CIGALE: Code Investigating GALaxy Emission,"
       A&A, 622, A103 (2019).

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
    """
    if model == "powerlaw_disc":
        alpha_grid = _axis("agn_alpha", alpha_grid, parameters)
        preint = _build_grid_powerlaw(filter_waves, filter_trans, redshift, alpha_grid)
        result = {
            "grid_phot": preint.phot,
            "axes": (jnp.asarray(alpha_grid),),
            "_preint": preint,
        }
        axis_params = AXIS_PARAMS_POWERLAW

    elif model == "ss_disc":
        mbh_grid = _axis("agn_log_mbh", mbh_grid, parameters)
        lbol_grid = _axis("agn_log_lbol", lbol_grid, parameters)
        preint = _build_grid_ss(filter_waves, filter_trans, redshift, mbh_grid, lbol_grid)
        result = {
            "grid_phot": preint.phot,
            "axes": (jnp.asarray(mbh_grid), jnp.asarray(lbol_grid)),
            "_preint": preint,
        }
        axis_params = AXIS_PARAMS_SS

    elif model == "cigale_disc":
        if disk_type not in (0, 1):
            raise ValueError(
                "cigale_disc disk_type must be 0 (SKIRTOR) or 1 (Schartmann 2005), "
                f"got {disk_type!r}"
            )
        delta_grid = _axis("agn_cigale_disk_delta", delta_grid, parameters)
        preint = _build_grid_cigale(filter_waves, filter_trans, redshift, disk_type, delta_grid)
        result = {
            "grid_phot": preint.phot,
            "axes": (jnp.asarray(delta_grid),),
            "_preint": preint,
        }
        axis_params = AXIS_PARAMS_CIGALE

    else:
        raise ValueError(f"Unknown disc model: {model}")

    # Auto-collapse any Fixed axes
    preint: PreintegratedGrid = result["_preint"]
    collapsed, remaining_axes, fixed = collapse_fixed_axes(
        preint, axis_params, parameters, origin=f"disc_precompute[{model}]"
    )
    if not fixed:
        return result

    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
    }


def build_lookup(
    preint: dict, *, model: str = "powerlaw_disc", free_param_names: tuple[str, ...] | None = None
):
    """Build the runtime disc photometry lookup from a preintegrated dict.

    The band photometry is interpolated with node-exact PCHIP on ``ln`` of the band flux
    (:func:`~tengri.utils.grid_interp.interp_nd_pchip`), so the lookup returns the tabulated
    value at every node and is C¹ in the query. The former triweight smoother did not
    reproduce its own nodes (the powerlaw table was 1.08 to 1.67 times the tabulated value at
    the nodes it spans).

    Parameters
    ----------
    preint : dict
        Preintegrated data dict with keys "grid_phot", "axes", optionally
        "_collapsed_axes".
    model : str, keyword-only
        One of "powerlaw_disc", "ss_disc". Sets the luminosity scale of the templates.
    free_param_names : tuple of str, optional
        Names of remaining free axes in the collapsed case.
        Not used in the default (no-collapse) case.

    Returns
    -------
    callable
        JIT-compiled photometry lookup function with signature::

            fn(agn_log_lbol, *free_axis_values) -> ndarray, shape (n_filters,)

        Returns disc L_ν [erg/s/Hz].

    Raises
    ------
    ValueError
        If ``model`` is not a disc model with a lookup (``powerlaw_disc``, ``ss_disc``,
        ``cigale_disc``).

    Notes
    -----
    **JIT-compatible**: yes, the returned function uses ``jnp`` and PCHIP primitives.

    **Gradient-safe**: yes; the gradient is C¹ in every axis and finite inside the nodes.
    The lookup holds the edge value beyond the nodes, so the axes come from
    :func:`reach_axes.default_axis` over the parameter's reach (see :func:`precompute`).

    **Accuracy**: at the default nodes, against the exact closure at 40 seeded random
    off-node points of the reach (``agn_alpha`` in [-3, 0.5], ``agn_log_lbol`` in [8, 14]),
    band-averaged over 1400-1600, 4000-5000, 8000-9000 and 60000-90000 A: ``powerlaw_disc``
    at most 1.6e-2 relative (1.5e-2 in the far-IR band). ``ss_disc`` at off-node
    ``agn_log_mbh`` in [6, 10]: 6-9e-2 relative; the floor is set by the energy normalization
    of the template on its own rest-frame grid, not by the node density (9 and 17 mbh nodes
    give the same error).

    The leading argument is ``agn_log_lbol = log10(L_bol / L_sun)``. See
    :mod:`~tengri.forward.precompute.protocol` for the unified AGN adapter
    convention.
    """
    if model not in _LOOKUP_UNIT_ERG:
        raise ValueError(f"build_lookup has no table for model {model!r}")
    unit_erg = _LOOKUP_UNIT_ERG[model]
    if preint.get("_collapsed_axes"):
        grid_phot = np.asarray(preint["grid_phot"], dtype=np.float64)
        axes = tuple(jnp.asarray(ax) for ax in preint["axes"])
    else:
        grid_phot = np.asarray(preint["_preint"].phot, dtype=np.float64)
        axes = tuple(jnp.asarray(ax) for ax in preint["_preint"].axes)
    if not axes:
        # The scalar form keeps a leading axis of length 1, shape (1, n_filters).
        flat = jnp.asarray(grid_phot).reshape(1, -1)

        @jax.jit
        def disc_phot_scalar(agn_log_lbol):
            """Compute disc photometry from log10 bolometric luminosity (no shape axes)."""
            return (10.0**agn_log_lbol) * unit_erg * flat

        return disc_phot_scalar

    log_grid = jnp.log(jnp.maximum(jnp.asarray(grid_phot), _FLOAT64_TINY))

    @jax.jit
    def disc_phot(agn_log_lbol, *free_axis_values):
        """Compute disc photometry from log10 bolometric luminosity.

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        normed = jnp.exp(interp_nd_pchip(log_grid, axes, tuple(free_axis_values)))
        return (10.0**agn_log_lbol) * unit_erg * normed

    return disc_phot
