# SPDX-License-Identifier: BSD-3-Clause
"""Filter-integrated precompute path for the GRAHSP AGN model.

Mirrors :mod:`tengri.components.agn.qsogen_precompute`. Builds a
preintegrated photometry grid over the most frequently varied GRAHSP
parameters; runtime evaluation reduces to a triweight interpolation at
JIT cost ~10 us per likelihood call (vs ~ms for the wavelength-grid path).

Axis selection
--------------
GRAHSP has 18 free parameters. Realistic SED fits constrain a small subset
strongly; the rest are sampled but rarely vary by orders of magnitude.
We precompute over a 2-D grid::

    axes = (agn_grahsp_plslope, agn_grahsp_ebv)

These two together span the largest UV-to-IR shape variations. Other
parameters stay at the GRAHSP defaults during the precompute build; the
runtime ``GRAHSPSEDComponent.apply`` path remains available for full-
parameter exploration. Auto-collapse of ``Fixed`` axes is supported
(matching :mod:`qsogen_precompute`).

References
----------
Buchner, J. et al. 2024, arXiv:2405.19297, §2.2 (computational optimizations).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components._collapsed_lookup import interp_collapsed
from tengri.components.agn.grahsp.model import GRAHSPParams, evaluate_grahsp_agn
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tengri.forward.precompute.templates import (
    build_template_photometry_lookup,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    slice_fixed_axes,
)
from tengri.utils.interpolation import edges_for_grid

__all__ = ["AXIS_PARAMS", "build_lookup", "precompute"]

AXIS_PARAMS: tuple[str, ...] = ("agn_grahsp_plslope", "agn_grahsp_ebv")
"""Parameters varied during precomputation. Other GRAHSP params held at defaults."""


def _build_grid_grahsp(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    plslope_grid: np.ndarray,
    ebv_grid: np.ndarray,
    *,
    l5100: float = 1.0e44,
    uvslope: float = 0.0,
    plbendloc_nm: float = 100.0,
    plbendwidth: float = 1.0,
    a_lines: float = 1.0,
    a_feii: float = 5.0,
    linewidth_kms: float = 5000.0,
    fcov: float = 0.4,
    si: float = 0.0,
    cool_lam_um: float = 17.0,
    cool_width: float = 0.45,
    hot_lam_um: float = 2.0,
    hot_width: float = 0.5,
    hot_fcov: float = 1.0,
    ebv_agn: float = 0.05,
    agn_type: int = 1,
    a_bc: float = 0.0,
    tor_temp: float = 0.0,
    tor_cutoff_um: float = 1.2,
    torus_model: str = "gaussian",
    feii_template: str = "bruhweiler2008",
    disc_model: str | None = None,
    disc_m: str = "8.0",
    disc_a: str = "0",
    disc_mdot: str = "0.3",
) -> PreintegratedGrid:
    """Pre-integrate GRAHSP over a 2-D grid of (plslope, E(B-V)).

    Structural selectors (``torus_model``, ``feii_template``, ``disc_model``,
    ``disc_*``) and the non-axis continuous parameters (``a_bc``, ``tor_temp``,
    ``tor_cutoff_um``, ...) are held fixed at the supplied values for the whole
    grid — exactly as the other non-axis GRAHSP parameters are. Pass them so a
    ``WavePrecomp`` build of a non-default variant (e.g. ``torus_model="mn12"``)
    pre-integrates the *correct* SED rather than silently defaulting.
    """
    plslope_grid = np.asarray(plslope_grid, dtype=np.float64)
    ebv_grid = np.asarray(ebv_grid, dtype=np.float64)

    # Standard rest-frame wavelength grid: 100 Å (1 nm = 10 Å) to 1e6 Å (100 um).
    wave_rest_angstrom = np.logspace(2.0, 6.0, 1500, dtype=np.float64)
    templates_grahsp = load_grahsp_templates()

    spectra = []
    for plslope in plslope_grid:
        for ebv in ebv_grid:
            params = GRAHSPParams(
                l5100=float(l5100),
                uvslope=float(uvslope),
                plslope=float(plslope),
                plbendloc_nm=float(plbendloc_nm),
                plbendwidth=float(plbendwidth),
                a_lines=float(a_lines),
                a_feii=float(a_feii),
                linewidth_kms=float(linewidth_kms),
                agn_type=int(agn_type),
                fcov=float(fcov),
                si=float(si),
                cool_lam_um=float(cool_lam_um),
                cool_width=float(cool_width),
                hot_lam_um=float(hot_lam_um),
                hot_width=float(hot_width),
                hot_fcov=float(hot_fcov),
                ebv=float(ebv),
                ebv_agn=float(ebv_agn),
                a_bc=float(a_bc),
                tor_temp=float(tor_temp),
                tor_cutoff_um=float(tor_cutoff_um),
                torus_model=torus_model,
                feii_template=feii_template,
                disc_model=disc_model,
                disc_m=disc_m,
                disc_a=disc_a,
                disc_mdot=disc_mdot,
            )
            sed = evaluate_grahsp_agn(
                jnp.asarray(wave_rest_angstrom * 0.1),  # nm
                params,
                templates_grahsp,
            )
            # AGN-side L_lambda [erg/s/nm] = bbb_attenuated + torus_attenuated
            L_lambda_nm = sed.bbb_attenuated + sed.torus_attenuated
            # Convert to L_lambda in erg/s/Å for downstream filter helper.
            L_lambda_angstrom = np.asarray(L_lambda_nm) * 0.1
            spectra.append(L_lambda_angstrom)

    templates = np.array(spectra, dtype=np.float64).reshape(
        plslope_grid.size, ebv_grid.size, wave_rest_angstrom.size
    )

    return precompute_template_photometry(
        templates=templates,
        wave_rest=wave_rest_angstrom,
        filter_waves=[np.asarray(fw, dtype=np.float64) for fw in filter_waves],
        filter_trans=[np.asarray(ft, dtype=np.float64) for ft in filter_trans],
        axes=(plslope_grid, ebv_grid),
        redshift=redshift,
        dl_cm=1.0,
        energy_normalize=False,  # GRAHSP normalizes to l5100; do not double-normalize
        units="llam",
    )


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters: Any | None = None,
    *,
    plslope_grid: np.ndarray | None = None,
    ebv_grid: np.ndarray | None = None,
    torus_model: str = "gaussian",
    feii_template: str = "bruhweiler2008",
    disc_model: str | None = None,
    disc_m: str = "8.0",
    disc_a: str = "0",
    disc_mdot: str = "0.3",
) -> dict:
    """Build the GRAHSP preintegrated grid, auto-collapsing Fixed-parameter axes.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Per-filter wavelength curves [Å, observed frame].
    filter_trans : list[ndarray]
        Per-filter transmission curves.
    redshift : float
        Source redshift.
    parameters : Parameters | None
        If supplied, axes whose corresponding parameter is ``Fixed`` (or
        absent from the spec) are collapsed at the fixed value / GRAHSP
        default.
    plslope_grid : ndarray, optional
        Defaults to ``[-2.5, -1.7, -1.0]`` covering Brown 2019 atlas range.
    ebv_grid : ndarray, optional
        Defaults to ``[0.0, 0.05, 0.1, 0.3, 1.0]`` covering Sy1 to ULIRG
        attenuations.
    torus_model, feii_template, disc_model, disc_m, disc_a, disc_mdot
        Structural variant selectors, forwarded to :func:`_build_grid_grahsp`
        so a non-default ``WavePrecomp`` build pre-integrates the matching SED.
        These must mirror the :class:`GRAHSPSEDComponentConfig` chosen for the
        fit.

    Returns
    -------
    dict
        Keys: ``grid_phot``, ``axes``, ``_preint``,
        optionally ``_collapsed_axes``.
    """
    if plslope_grid is None:
        plslope_grid = np.array([-2.5, -1.7, -1.0], dtype=np.float64)
    if ebv_grid is None:
        ebv_grid = np.array([0.0, 0.05, 0.1, 0.3, 1.0], dtype=np.float64)

    preint = _build_grid_grahsp(
        filter_waves,
        filter_trans,
        redshift,
        plslope_grid,
        ebv_grid,
        torus_model=torus_model,
        feii_template=feii_template,
        disc_model=disc_model,
        disc_m=disc_m,
        disc_a=disc_a,
        disc_mdot=disc_mdot,
    )
    result = {
        "grid_phot": preint.phot,
        "axes": (jnp.asarray(plslope_grid), jnp.asarray(ebv_grid)),
        "_preint": preint,
    }

    if parameters is None:
        return result

    _AXIS_DEFAULTS = {
        "agn_grahsp_plslope": -1.7,
        "agn_grahsp_ebv": 0.05,
    }
    fixed_values = parameters.get_fixed_values()
    free_param_names = set(parameters.free_params)
    fixed: dict[int, float] = {}
    for i, pname in enumerate(AXIS_PARAMS):
        if pname in fixed_values:
            fixed[i] = float(fixed_values[pname])
        elif pname not in free_param_names:
            fixed[i] = float(_AXIS_DEFAULTS[pname])

    if not fixed:
        return result

    collapsed = slice_fixed_axes(preint, fixed)
    remaining_axes = tuple(ax for i, ax in enumerate(result["axes"]) if i not in fixed)
    return {
        "grid_phot": collapsed.phot,
        "axes": remaining_axes,
        "_preint": collapsed,
        "_collapsed_axes": fixed,
    }


def build_lookup(preint: dict, *, free_param_names: tuple[str, ...] | None = None):
    """Build the runtime GRAHSP photometry lookup from a preintegrated dict.

    Returns a JIT-compiled callable that takes the remaining free axis
    values and returns L_nu [erg/s/Hz] per filter.
    """
    del free_param_names  # not currently consumed
    if not preint.get("_collapsed_axes"):
        return build_template_photometry_lookup(preint["_preint"])

    grid_phot = preint["grid_phot"]
    axes = preint["axes"]
    if axes:
        edges = tuple(edges_for_grid(ax) for ax in axes)
    else:
        edges = ()

    @jax.jit
    def grahsp_phot_collapsed(*free_axis_values):
        """Filter-integrated L_nu [erg/s/Hz] with collapsed axes."""
        return interp_collapsed(grid_phot, axes, free_axis_values, kernel="triweight", edges=edges)

    return grahsp_phot_collapsed
