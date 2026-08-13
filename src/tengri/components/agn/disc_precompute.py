# SPDX-License-Identifier: BSD-3-Clause
"""Precompute adapters for analytic accretion disc models.

Implements :class:`~tengri.forward.precompute.protocol.PrecomputeModule` for
three disc models:

1. **powerlaw_disc**: Simple power-law optical/UV continuum with exponential
   UV cutoff. One free axis: ``agn_alpha_pl`` (power-law index).

2. **disc_ss** (Shakura-Sunyaev): Multi-color thin disc with temperature
   gradient. Two free axes: ``agn_log_mbh``, ``agn_log_lbol`` (the post-#846
   shape driver — the Eddington ratio, hence the disc temperature profile, is
   derived from ``agn_log_lbol`` and ``agn_log_mbh``).

3. **cigale_disc** (piecewise power-law from CIGALE): Empirical disc model
   with fixed wavelength breakpoints and power-law segments. No free axes
   (shape is fixed; only ``agn_log_lbol`` scales at runtime).

Each model is preintegrated through filter curves at model-initialization time.
Auto-collapses axes whose corresponding parameters are ``Fixed`` in the user's
``Parameters`` — e.g., a user who pins ``agn_alpha_pl`` gets a scalar template.

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
   (CITATION-AUDIT NOTE: not found in ~/writing-workspace; verify against source)
.. [4] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy Emission,"
   A&A, 622, A103 (2019). https://doi.org/10.1051/0004-6361/201834156
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components._collapsed_lookup import interp_collapsed
from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL, DEFAULT_AGN_LUM_RATIO
from tengri.components.agn.disc import (
    multicolor_disc as _multicolor_disc,
    powerlaw_disc as _powerlaw_disc,
)
from tengri.components.agn.disc_cigale import piecewise_powerlaw_disk as _piecewise_pl
from tengri.forward.precompute.templates import (
    build_template_photometry_lookup,
    collapse_fixed_axes,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import PreintegratedGrid
from tengri.utils.interpolation import edges_for_grid

# ── Axis definitions per model ──────────────────────────────────


# powerlaw_disc: parametrized by agn_alpha (power-law index)
AXIS_PARAMS_POWERLAW = ("agn_alpha_pl",)

# disc_ss (Shakura-Sunyaev): parametrized by BH mass and bolometric luminosity.
# Since #846 the disc shape is self-consistent with agn_log_lbol (the Eddington
# ratio, and hence T_in / r_out, is DERIVED from L_bol and M_bh), so L_bol is a
# genuine shape axis. The former agn_log_mdot axis fed the now-ignored
# agn_log_ledd and was silently degenerate (#902).
AXIS_PARAMS_SS = ("agn_log_mbh", "agn_log_lbol")

# cigale_disc: shape is fixed; no grid axes (purely template-scaled by L_bol)
AXIS_PARAMS_CIGALE = ()

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
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
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
    agn_log_lbol : float
        Reference bolometric luminosity [log10(L/L_sun)].
        Defaults to the declared ``agn_log_lbol`` default.
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
    len(alpha_grid)

    # Standard rest-frame wavelength grid for integration
    # (covers ~10 Angstrom to ~1 mm with fine sampling)
    wave_rest = np.logspace(1, 5, 1000, dtype=np.float64)

    # Precompute L_nu for each alpha value
    phot_grid = []
    for alpha in alpha_grid:
        # Call the JAX disc function with reference L_bol and collect L_nu
        l_nu = np.asarray(
            _powerlaw_disc(
                jnp.asarray(wave_rest),
                agn_log_lbol=agn_log_lbol,
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
        energy_normalize=False,  # already normalized per L_sun from disc function
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
    here — rather than the now-ignored ``agn_log_ledd`` — fixes the silently
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


def _build_grid_cigale(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
) -> PreintegratedGrid:
    """Preintegrate cigale piecewise-powerlaw disc (scalar template, no axes).

    The CIGALE disc model uses empirical wavelength breakpoints and power-law
    indices. It is a fixed shape, scaled only by luminosity at runtime.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter wavelength arrays [Angstrom].
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    redshift : float
        Source redshift.

    Returns
    -------
    PreintegratedGrid
        Preintegrated photometry with shape (1, n_filters) for scalar access.
    """
    # CIGALE disc default parameters: limits and power-law indices
    # From disc_cigale.py skirtor_disk_spectrum:
    # delta ≈ 0 → limits=[100, 400, 1500, 5000, 20000], coefs=[-0.5, -0.3, 1.5, 1.0]
    limits = np.array([100.0, 400.0, 1500.0, 5000.0, 20000.0], dtype=np.float64)
    coefs = np.array([-0.5, -0.3, 1.5, 1.0], dtype=np.float64)

    wave_rest = np.logspace(1, 5, 1000, dtype=np.float64)

    # Call piecewise_powerlaw_disk once to get unit-normalized spectrum
    spec = np.asarray(
        _piecewise_pl(
            jnp.asarray(wave_rest),
            limits=jnp.asarray(limits),
            coefs=jnp.asarray(coefs),
        )
    )

    # Wrap in shape (1, n_wave) for compatibility with precompute_template_photometry
    templates = np.array([spec], dtype=np.float64)

    # No grid axes for CIGALE: preintegrate as a single template
    return precompute_template_photometry(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(),  # No axes: scalar template
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=True,  # Template is dimensionless; normalize to unit L_bol
        units="lnu",
    )


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
        Grid for agn_alpha_pl (powerlaw_disc only). If None, uses a default
        range [-2, 0] with 15 points.
    mbh_grid : ndarray, optional
        Grid for agn_log_mbh (ss_disc only). If None, uses [6, 7, 8, 9, 10].
    lbol_grid : ndarray, optional
        Grid for agn_log_lbol (ss_disc only) [log10(L_sun)]. If None, uses
        [9, 10, 11, 12, 13] — faint-Seyfert through bright-quasar bolometric
        luminosities (sub-Eddington across the default M_bh grid; the disc peak
        sweeps from the optical into the EUV over this range).

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
    **JIT-compatible**: no — this is a build-time function using NumPy.
    """
    if model == "powerlaw_disc":
        if alpha_grid is None:
            alpha_grid = np.linspace(-2.0, 0.0, 15, dtype=np.float64)
        result = {
            "grid_phot": _build_grid_powerlaw(
                filter_waves, filter_trans, redshift, alpha_grid
            ).phot,
            "axes": (jnp.asarray(alpha_grid),),
            "_preint": _build_grid_powerlaw(filter_waves, filter_trans, redshift, alpha_grid),
        }
        axis_params = AXIS_PARAMS_POWERLAW

    elif model == "ss_disc":
        if mbh_grid is None:
            mbh_grid = np.array([6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
        if lbol_grid is None:
            lbol_grid = np.array([9.0, 10.0, 11.0, 12.0, 13.0], dtype=np.float64)
        result = {
            "grid_phot": _build_grid_ss(
                filter_waves, filter_trans, redshift, mbh_grid, lbol_grid
            ).phot,
            "axes": (jnp.asarray(mbh_grid), jnp.asarray(lbol_grid)),
            "_preint": _build_grid_ss(filter_waves, filter_trans, redshift, mbh_grid, lbol_grid),
        }
        axis_params = AXIS_PARAMS_SS

    elif model == "cigale_disc":
        result = {
            "grid_phot": _build_grid_cigale(filter_waves, filter_trans, redshift).phot,
            "axes": (),
            "_preint": _build_grid_cigale(filter_waves, filter_trans, redshift),
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

    Delegates to the template helper for triweight interpolation.

    Parameters
    ----------
    preint : dict
        Preintegrated data dict with keys "grid_phot", "axes", optionally
        "_collapsed_axes".
    model : str, keyword-only
        Disc model name (for documentation; not used in lookup logic).
    free_param_names : tuple of str, optional
        Names of remaining free axes in the collapsed case.
        Not used in the default (no-collapse) case.

    Returns
    -------
    callable
        JIT-compiled photometry lookup function with signature::

            fn(agn_log_lbol, *free_axis_values) -> ndarray, shape (n_filters,)

        Returns disc L_ν [erg/s/Hz]. Caller applies flux scaling.

    References
    ----------
    .. [1] A. Kubota and C. Done, "A physical model of the broad-band continuum
       of AGN and its implications for the UV/X relation and optical variability,"
       MNRAS, 480, 1247 (2018).

    Notes
    -----
    **JIT-compatible**: yes — the returned function uses ``jnp`` and triweight
    interpolation.

    **Gradient-safe**: yes — triweight kernel is fully differentiable.
    """
    if not preint.get("_collapsed_axes"):
        # No axes collapsed: use template helper directly
        return build_template_photometry_lookup(preint["_preint"])

    # Collapsed case: return a wrapped lookup that takes remaining free params
    grid_phot = preint["grid_phot"]
    axes = preint["axes"]
    if axes:
        edges = tuple(edges_for_grid(ax) for ax in axes)
    else:
        edges = ()

    @jax.jit
    def disc_phot_collapsed(agn_log_lbol, *free_axis_values):
        """Compute disc photometry with some axes collapsed (fixed).

        Returns filter-integrated L_nu [erg/s/Hz] at runtime.
        """
        l_bol_lsun = 10.0**agn_log_lbol
        normed = interp_collapsed(
            grid_phot, axes, free_axis_values, kernel="triweight", edges=edges
        )
        return l_bol_lsun * normed

    return disc_phot_collapsed
