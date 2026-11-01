# SPDX-License-Identifier: BSD-3-Clause
r"""Library anisotropy of the composable torus blocks whose sightline power is not their mean.

A torus library whose records share one radiative-transfer scale per model keeps its flux
ratio between sightlines (``_torus_inclination``): the block emits, along :math:`\mu = \cos i`,

.. math::

    L_{\rm torus}(\mu) = f L_{\rm bol}\,\frac{a(\mu)}{\bar a}, \qquad
    \bar a = \int_0^1 a(\mu)\,d\mu,

so its angle-integrated power is :math:`f L_{\rm bol}` and its sightline power differs from it
by :math:`a/\bar a`. Each function here returns :math:`\log_{10}(a/\bar a)` for one block, at the
same library point and sightline the block evaluates (the block's own keyword defaults, read off
its signature, so the two cannot drift). The composable runner's energy ledger subtracts it from
the sightline torus power to publish the angle-integrated one (NAMING_CONTRACT section 4c,
"Template tori" and "Composable runner").

A torus block that registers nothing here carries the same power along every sightline: the
libraries normalized per sightline (``skirtor``, ``skirtor_agnfitter``, ``skirtor_agnfitter_2p``,
``cat3d_wind``, ``cat3d_wind_lowfwd``) and those without an inclination axis (``silva04``,
``nenkova``, ``simple``, ``two_temperature``).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping

import jax.numpy as jnp
import numpy as np

from tengri.components.agn import (
    fritz as _fritz,
    nenkova_agnfitter as _nk1,
    nenkova_agnfitter_2p as _nk2,
    nenkova_agnfitter_3p as _nk3,
    skirtor_agnfitter_1p as _sk1,
)
from tengri.components.agn._params import DEFAULT_AGN_COS_INC
from tengri.components.agn._skirtor_agnfitter_sightline import skirtor_incl_from_cos_inc
from tengri.components.agn._template_grid import TorusTemplateGrid
from tengri.components.agn._torus_inclination import library_anisotropy, log10_anisotropy
from tengri.components.agn.blocks._protocol import register_torus_anisotropy
from tengri.components.agn.blocks.torus import (
    fritz_torus_block,
    nenkova_agnfitter_2p_torus_block,
    nenkova_agnfitter_3p_torus_block,
)

__all__ = [
    "fritz_log10_anisotropy",
    "library_cos_nodes",
    "nenkova_agnfitter_2p_log10_anisotropy",
    "nenkova_agnfitter_3p_log10_anisotropy",
    "nenkova_agnfitter_log10_anisotropy",
    "skirtor_agnfitter_1p_log10_anisotropy",
]


def _read(block: Callable, params: Mapping, name: str):
    """``params[name]``, else the block's own keyword default for it."""
    if name in params:
        return params[name]
    return inspect.signature(block).parameters[name].default


def _cos_inc(params: Mapping):
    """The model's one inclination ``agn_cos_inc``, at its declared default when absent."""
    return params.get("agn_cos_inc", DEFAULT_AGN_COS_INC)


def _template_grid(templates, loader: Callable) -> TorusTemplateGrid:
    """The threaded library, or the block's cached default when the call is unthreaded."""
    return templates if isinstance(templates, TorusTemplateGrid) else loader()


@register_torus_anisotropy("fritz")
def fritz_log10_anisotropy(templates, dtype, **params):
    r"""``log10(a/a_bar)`` of the ``fritz`` torus block at its library point and sightline.

    Parameters
    ----------
    templates : FritzGrid or None
        The threaded Fritz library; ``None`` loads the default grid.
    dtype : dtype
        Evaluation dtype (unused: the anisotropy table is O(1)).
    **params
        The block's keyword arguments (``agn_fritz_*``, ``agn_cos_inc``).

    Returns
    -------
    ndarray, shape ()
        :math:`\log_{10}(a(\mu)/\bar a)` [dex] (``fritz.fritz_anisotropy``, the records'
        ``norm`` over their model's mean, at the elevation ``psi = 90 deg - i``).

    Notes
    -----
    **JIT/grad/vmap-safe**: yes.
    """
    del dtype
    grid = (
        templates if isinstance(templates, _fritz.FritzGrid) else _fritz.load_fritz_default_grid()
    )
    point = (
        _read(fritz_torus_block, params, "agn_fritz_r_ratio"),
        _read(fritz_torus_block, params, "agn_fritz_tau"),
        _read(fritz_torus_block, params, "agn_fritz_beta"),
        _read(fritz_torus_block, params, "agn_fritz_gamma"),
        _read(fritz_torus_block, params, "agn_fritz_oa"),
        _fritz.fritz_psy_from_cos_inc(_cos_inc(params)),
    )
    return log10_anisotropy(_fritz.fritz_anisotropy(grid), point, _fritz.TORUS_INCLINATION)


def _library_log10_anisotropy(grid: TorusTemplateGrid, inc, coords: tuple, dtype):
    """``log10(a/a_bar)`` of a ``TorusTemplateGrid`` library at ``coords``."""
    return log10_anisotropy(library_anisotropy(grid, inc, dtype), coords, inc)


@register_torus_anisotropy("nenkova_agnfitter")
def nenkova_agnfitter_log10_anisotropy(templates, dtype, **params):
    r"""``log10(a/a_bar)`` of the ``nenkova_agnfitter`` (``NK0_mean_1p``) torus block.

    Parameters
    ----------
    templates : TorusTemplateGrid or None
        The threaded library; ``None`` loads the default grid.
    dtype : dtype
        Evaluation dtype of the anisotropy table.
    **params
        The block's keyword arguments (``agn_cos_inc``).

    Returns
    -------
    ndarray, shape ()
        :math:`\log_{10}(a(\mu)/\bar a)` [dex] along ``agn_cos_inc``.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes.
    """
    grid = _template_grid(templates, _nk1.load_nenkova_agnfitter_default_grid)
    return _library_log10_anisotropy(grid, _nk1.TORUS_INCLINATION, (_cos_inc(params),), dtype)


@register_torus_anisotropy("nenkova_agnfitter_2p")
def nenkova_agnfitter_2p_log10_anisotropy(templates, dtype, **params):
    r"""``log10(a/a_bar)`` of the ``nenkova_agnfitter_2p`` (``NK0_mean_2p``) torus block.

    Parameters
    ----------
    templates : TorusTemplateGrid or None
        The threaded library; ``None`` loads the default grid.
    dtype : dtype
        Evaluation dtype of the anisotropy table.
    **params
        The block's keyword arguments (``agn_cos_inc``, ``agn_oa_nenkova``).

    Returns
    -------
    ndarray, shape ()
        :math:`\log_{10}(a(\mu)/\bar a)` [dex] along ``agn_cos_inc``.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes.
    """
    grid = _template_grid(templates, _nk2.load_nenkova_agnfitter_2p_default_grid)
    coords = (
        _cos_inc(params),
        _read(nenkova_agnfitter_2p_torus_block, params, "agn_oa_nenkova"),
    )
    return _library_log10_anisotropy(grid, _nk2.TORUS_INCLINATION, coords, dtype)


@register_torus_anisotropy("nenkova_agnfitter_3p")
def nenkova_agnfitter_3p_log10_anisotropy(templates, dtype, **params):
    r"""``log10(a/a_bar)`` of the ``nenkova_agnfitter_3p`` (``NK0_mean_3p``) torus block.

    Parameters
    ----------
    templates : TorusTemplateGrid or None
        The threaded library; ``None`` loads the default grid.
    dtype : dtype
        Evaluation dtype of the anisotropy table.
    **params
        The block's keyword arguments (``agn_cos_inc``, ``agn_oa_nenkova``,
        ``agn_tv_nenkova``).

    Returns
    -------
    ndarray, shape ()
        :math:`\log_{10}(a(\mu)/\bar a)` [dex] along ``agn_cos_inc``.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes.
    """
    grid = _template_grid(templates, _nk3.load_nenkova_agnfitter_3p_default_grid)
    coords = (
        _cos_inc(params),
        _read(nenkova_agnfitter_3p_torus_block, params, "agn_oa_nenkova"),
        _read(nenkova_agnfitter_3p_torus_block, params, "agn_tv_nenkova"),
    )
    return _library_log10_anisotropy(grid, _nk3.TORUS_INCLINATION, coords, dtype)


@register_torus_anisotropy("skirtor_agnfitter_1p")
def skirtor_agnfitter_1p_log10_anisotropy(templates, dtype, **params):
    r"""``log10(a/a_bar)`` of the ``skirtor_agnfitter_1p`` (``SKIRTOR_mean_1p``) torus block.

    Parameters
    ----------
    templates : TorusTemplateGrid or None
        The threaded library; ``None`` loads the default grid.
    dtype : dtype
        Evaluation dtype of the anisotropy table.
    **params
        The block's keyword arguments (``agn_cos_inc``).

    Returns
    -------
    ndarray, shape ()
        :math:`\log_{10}(a(i)/\bar a)` [dex] at the library inclination
        ``i = arccos(agn_cos_inc)``.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes.
    """
    grid = _template_grid(templates, _sk1.load_skirtor_agnfitter_1p_default_grid)
    coords = (skirtor_incl_from_cos_inc(jnp.asarray(_cos_inc(params))),)
    return _library_log10_anisotropy(grid, _sk1.TORUS_INCLINATION, coords, dtype)


def _skirtor_cos_nodes() -> np.ndarray:
    """``cos i`` nodes of the SKIRTOR disc column (the library sightline's axis 5)."""
    from tengri.components.agn.skirtor import load_skirtor_disc_atten_grid

    return np.asarray(load_skirtor_disc_atten_grid().axes[5], dtype=np.float64)


def _fritz_cos_nodes() -> np.ndarray:
    """``cos i = sin(psi)`` at the Fritz library's elevation nodes (axis 5)."""
    psy = np.asarray(_fritz.load_fritz_default_grid().axes[5], dtype=np.float64)
    return np.sin(np.radians(psy))


def _template_cos_nodes(loader, inc) -> np.ndarray:
    """``cos i`` at a ``TorusTemplateGrid`` library's inclination nodes."""
    x = np.asarray(loader().axes[inc.axis], dtype=np.float64)
    if inc.variable == "cos_inc":
        return x
    if inc.variable == "incl_deg":
        return np.cos(np.radians(x))
    return np.sin(np.radians(x))


#: Torus block -> its library's inclination nodes in ``cos i`` (host), where the interpolated
#: sightline factors (the disc screen of SKIRTOR and Fritz, the anisotropy of the others) break.
_COS_NODE_LOADERS = {
    "skirtor": _skirtor_cos_nodes,
    "fritz": _fritz_cos_nodes,
    "nenkova_agnfitter": lambda: _template_cos_nodes(
        _nk1.load_nenkova_agnfitter_default_grid, _nk1.TORUS_INCLINATION
    ),
    "nenkova_agnfitter_2p": lambda: _template_cos_nodes(
        _nk2.load_nenkova_agnfitter_2p_default_grid, _nk2.TORUS_INCLINATION
    ),
    "nenkova_agnfitter_3p": lambda: _template_cos_nodes(
        _nk3.load_nenkova_agnfitter_3p_default_grid, _nk3.TORUS_INCLINATION
    ),
    "skirtor_agnfitter_1p": lambda: _template_cos_nodes(
        _sk1.load_skirtor_agnfitter_1p_default_grid, _sk1.TORUS_INCLINATION
    ),
}


def library_cos_nodes(torus_block: str) -> np.ndarray:
    """A torus library's inclination nodes in ``cos i``, or an empty array.

    Parameters
    ----------
    torus_block : str
        Torus selector.

    Returns
    -------
    ndarray, shape (n_inc,)
        Host values in [0, 1]; empty for a torus without an inclination axis that shapes the
        emitted pieces' angular profile, or whose library file is absent.
    """
    loader = _COS_NODE_LOADERS.get(torus_block)
    if loader is None:
        return np.zeros(0)
    try:
        return loader()
    except FileNotFoundError:
        return np.zeros(0)
