# SPDX-License-Identifier: BSD-3-Clause
r"""SKIRTOR-averaged (mean_1p) clumpy torus library (Stalevski et al. 2016).

One-parameter torus library interpolated differentiably in inclination over a
precomputed HDF5 grid derived from AGNfitter-rX's ``SKIRTOR_mean_1p.pickle``. The
sightline is the model's one inclination, ``agn_cos_inc``.

This is the coarsest of the three SKIRTOR-AGNfitter reductions tengri ships:
:mod:`tengri.components.agn.skirtor_agnfitter` (``SKIRTOR_mean_3p``, three
axes) is the AGNfitter-faithful default; :mod:`tengri.components.agn.skirtor_agnfitter_2p`
(``SKIRTOR_mean_2p``) adds the opening-angle axis; this module averages over
both opening angle and optical depth, retaining only inclination. All three
share the same inclination axis convention and extent (0-90 deg).

Axis semantics
--------------
``incl_axis``
    Inclination angle :math:`i` measured from the polar axis [deg]. Range: 0-90 deg.
    0 = face-on, 90 = edge-on. Keyed by ``agn_cos_inc`` (:math:`\cos i`) through
    ``_skirtor_agnfitter_sightline.skirtor_incl_from_cos_inc``,
    :math:`i = \arccos(\cos i)`.

Runtime normalization
---------------------
SKIRTOR computes every model for one central source luminosity (Stalevski et al. 2016,
Sect. 2.1), so its records share one scale across inclination. The stored inclination
power ratios equal those of the SKIRTOR v3 records (multiplied back by their ``norm``)
averaged over tau, p, q, opening angle and radius ratio to 0.4 % at every node, so the
reduction is on that scale (face-on/edge-on 2.12).
The template is therefore emitted as two factors
(``_torus_inclination.torus_lnu_anisotropic``): the shape of the records normalized to
unit power and interpolated as before, and the records' own powers over their model's
mean, a scalar table interpolated with the same node-exact PCHIP. The angle-integrated
torus power is ``L_bol * agn_torus_frac``, each sightline keeps the library's flux
relative to the others, and the shape between nodes is the per-sightline one exactly.

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
   torus around AGN, the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski, C. Ricci, Y. Ueda, P. Lira, J. Fritz, and M. Baes,
   "The dust covering factor in active galactic nuclei," MNRAS, 458,
   2288 (2016). arXiv:1602.06954. bibcode:2016MNRAS.458.2288S.
   https://doi.org/10.1093/mnras/stw444
.. [3] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from tengri.components.agn._params import DEFAULT_AGN_COS_INC, DEFAULT_AGN_LOG_LBOL
from tengri.components.agn._skirtor_agnfitter_sightline import (
    accepts_retired_incl_skirtor,
    skirtor_incl_from_cos_inc,
)
from tengri.components.agn._template_grid import TorusTemplateGrid
from tengri.components.agn._torus_inclination import (
    TorusInclination,
    anisotropy_values,
    torus_lnu_anisotropic,
)

#: The library's inclination axis: the records share one radiative-transfer scale
#: across inclination, so the template carries its anisotropy as a scalar factor
#: (``_torus_inclination.torus_lnu_anisotropic``).
TORUS_INCLINATION = TorusInclination(axis=0, variable="incl_deg")

__all__ = [
    "TORUS_INCLINATION",
    "create_skirtor_agnfitter_1p_from_grid",
    "skirtor_agnfitter_1p_sed",
]


def _load_skirtor_agnfitter_1p_arrays(grid_path: str) -> dict:
    """Load raw numpy arrays from the ``SKIRTOR_mean_1p`` grid HDF5.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean1p_torus_grid.h5``.

    Returns
    -------
    dict
        Keys ``incl_axis``, ``wavelength``, and ``template``.

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O at grid-load time.
    """
    import h5py

    with h5py.File(grid_path, "r") as f:
        g = f["skirtor_mean1p"]
        return {
            "incl_axis": np.asarray(g["incl_axis"][:], dtype=np.float64),
            "wavelength": np.asarray(g["wavelength"][:], dtype=np.float64),
            "template": np.asarray(g["template"][:], dtype=np.float64),
        }


@functools.cache
def load_skirtor_agnfitter_1p_grid(grid_path: str) -> TorusTemplateGrid:
    """Load a ``SKIRTOR_mean_1p`` grid HDF5 into a :class:`TorusTemplateGrid` pytree.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean1p_torus_grid.h5``.

    Returns
    -------
    TorusTemplateGrid
        Template arrays with a single ``incl`` axis (degrees), as numpy
        arrays.

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.
    KeyError
        If the grid file is missing the expected ``/skirtor_mean1p``
        datasets.

    Notes
    -----
    **JIT-compatible**: no, performs HDF5 I/O. Call outside the trace and
    pass the result in as an argument.
    """
    raw = _load_skirtor_agnfitter_1p_arrays(grid_path)
    axes = (np.asarray(raw["incl_axis"]),)
    return TorusTemplateGrid(
        template=np.asarray(raw["template"]),
        axes=axes,
        wave_grid=np.asarray(raw["wavelength"]),
        anisotropy=anisotropy_values(raw["template"], raw["wavelength"], axes, TORUS_INCLINATION),
    )


@accepts_retired_incl_skirtor
def skirtor_agnfitter_1p_sed_from_grid(
    grid: TorusTemplateGrid,
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    r"""``SKIRTOR_mean_1p`` torus SED at a single inclination.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 10.0.
    agn_cos_inc : float, optional
        :math:`\cos i`, the model's one inclination, :math:`i` from the polar axis
        (1 = face-on) [dimensionless]. Defaults to the declared ``agn_cos_inc`` default,
        ``cos(30 deg)``. The library is keyed at :math:`i = \arccos(\cos i)` [deg].
        The retired ``agn_incl_skirtor`` [deg] is translated with a
        ``DeprecationWarning``.
    agn_torus_frac : float, optional
        Fraction of L_bol reprocessed by the torus. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    .. math::

        L_\nu(\lambda;\,\mu) = L_{\rm bol}\,f_{\rm torus}\,
                               \frac{T(\lambda;\,\mu)}{\int T(\nu;\,\mu)\,\mathrm{d}\nu}\,
                               \frac{a(\mu)}{\bar a}, \qquad
        \bar a = \int_0^1 a(\mu)\,\mathrm{d}\mu,

    where :math:`T` is the tabulated template at the parameter point (node-exact PCHIP),
    :math:`\mu = \cos i`, the frequency integral is that of the interpolant on the
    template's own wavelength nodes, and :math:`a` is each record's own power over its
    model's mean, interpolated with the same PCHIP. The angle-integrated power is
    :math:`f_{\rm torus} L_{\rm bol}`; along :math:`\mu` the torus carries
    :math:`f_{\rm torus} L_{\rm bol}\,a(\mu)/\bar a`.

    **JIT-compatible**: yes.

    **Approximation**: the grid is drawn from the ``SKIRTOR_mean_1p``
    library (Stalevski et al. 2016 [1]_, [2]_) as packaged by
    AGNfitter-rX [3]_, which averages out opening angle, clumpiness
    (p, q), and optical depth of the full SKIRTOR parameter space.
    """
    # Anisotropy restored here (#2851). Provenance: AGNfitter-rX SKIRTOR_mean_1p.pickle
    # (tag AGNfitter-rX_v0.1, models/TORUS), field 'SED' per inclination, repackaged in
    # skirtor_mean1p_torus_grid.h5 'template'; its power ratios between inclinations match
    # the SKIRTOR v3 records (the CIGALE-lineage dust_emission x spectra/norm; one central
    # source luminosity per model, Stalevski et al. 2016 Sect. 2.1) averaged over tau, p,
    # q, opening angle and R to 0.4 % at every node. Sightline: 'incl_axis' is the
    # inclination i from the polar axis [deg] (0 face-on), ascending, axis 0.
    return torus_lnu_anisotropic(
        grid,
        wavelength,
        (skirtor_incl_from_cos_inc(agn_cos_inc),),
        TORUS_INCLINATION,
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
    )


def create_skirtor_agnfitter_1p_from_grid(grid_path: str) -> Callable:
    """Load a ``SKIRTOR_mean_1p`` grid and bind it to the SED evaluator.

    Parameters
    ----------
    grid_path : str
        Path to ``skirtor_mean1p_torus_grid.h5``.

    Returns
    -------
    callable
    """
    return functools.partial(
        skirtor_agnfitter_1p_sed_from_grid, load_skirtor_agnfitter_1p_grid(grid_path)
    )


_GRID_SEARCH_PATHS: tuple[str, ...] = (
    "data/skirtor_mean1p_torus_grid.h5",
    "skirtor_mean1p_torus_grid.h5",
)

_NOT_FOUND_MSG = (
    "SKIRTOR_mean_1p torus grid not found. Build it with: "
    "python scripts/build_agnfitter_torus_reductions.py --which skirtor_mean1p"
)


def _find_skirtor_agnfitter_1p_grid() -> str:
    from tengri._data_setup import require_data

    return require_data(_GRID_SEARCH_PATHS[-1], _NOT_FOUND_MSG)


def load_skirtor_agnfitter_1p_default_grid() -> TorusTemplateGrid:
    """Load the packaged ``SKIRTOR_mean_1p`` grid pytree (discovery + cache).

    This is the ``template_loader`` the torus block registers.

    Returns
    -------
    TorusTemplateGrid

    Raises
    ------
    FileNotFoundError
        If no ``SKIRTOR_mean_1p`` grid HDF5 is present on disk.
    """
    return load_skirtor_agnfitter_1p_grid(_find_skirtor_agnfitter_1p_grid())


@functools.cache
def _load_skirtor_agnfitter_1p_default() -> Callable:
    return create_skirtor_agnfitter_1p_from_grid(_find_skirtor_agnfitter_1p_grid())


def skirtor_agnfitter_1p_sed(
    *args, _template: TorusTemplateGrid | None = None, **kwargs
) -> jnp.ndarray:
    r"""``SKIRTOR_mean_1p`` torus (auto-loaded from the packaged HDF5 grid).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 10.0.
    agn_cos_inc : float, optional
        :math:`\cos i`, the model's one inclination (1 = face-on) [dimensionless].
        Default ``cos(30 deg)``.
    agn_torus_frac : float, optional
        Torus reprocessing fraction. Default 0.5.
    **kwargs
        Accepted and ignored for unified-dispatch compatibility.

    Returns
    -------
    ndarray, shape (n_wave,)
        Torus SED [erg/s/Hz].

    Notes
    -----
    This function auto-discovers the grid file from the package data
    directory or the current working directory. To use a non-standard grid
    location, call :func:`create_skirtor_agnfitter_1p_from_grid` directly.
    """
    if _template is not None:
        return skirtor_agnfitter_1p_sed_from_grid(_template, *args, **kwargs)
    fn = _load_skirtor_agnfitter_1p_default()
    return fn(*args, **kwargs)
