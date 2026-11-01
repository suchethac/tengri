# SPDX-License-Identifier: BSD-3-Clause
"""Where each composable AGN block's spectrum is discontinuous (``register_spectral_edges``).

A fixed-grid frequency integral converges at first order across a discontinuity, with an
error set by where the step falls between two nodes, and at second order once a node sits on
each side of it. The energy ledger of the composable runner (``_ledger``) puts a node pair at
every edge registered here for the selected blocks, so each published power is the integral of
its piece of the SED to the ledger's quadrature (NAMING_CONTRACT section 4c, "Composable
runner").

An edge is a band cut written in the block (the multicolor disc's EUV tail ends at 30 A, the
power-law disc's band at 10 A and 1e8 A, the CIGALE discs at 8 nm and 1e6 nm, the GRAHSP and
QSOgen discs at the 124 A X-ray floor), or the first or last node of the template the block
interpolates (zero outside it), read off the packaged library. A slope break inside a band
(a kink) needs no node: the trapezoid is second order across it. The library tori's first
wavelength node is also the step in the sightline transmission of the disc behind them.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np

from tengri.components.agn.blocks._protocol import (
    AGN_BLOCK_TEMPLATE_LOADERS,
    register_spectral_edges,
)

__all__ = ["block_edges", "template_wave_ends"]


def template_wave_ends(library) -> tuple[float, ...]:
    """First and last wavelength node [Angstrom] of a packaged block library.

    Parameters
    ----------
    library : object
        A loaded library carrying ``wave_grid`` (a dict key, an attribute, or the attribute of
        its ``torus`` / ``disc_dust`` members for the SKIRTOR bundle).

    Returns
    -------
    tuple of float
        The ends of every wavelength grid found.

    Raises
    ------
    TypeError
        If the library carries no ``wave_grid``.
    """
    grids = []
    if isinstance(library, dict) and "wave_grid" in library:
        grids.append(library["wave_grid"])
    elif hasattr(library, "wave_grid"):
        grids.append(library.wave_grid)
    for member in ("torus", "disc_dust"):
        sub = getattr(library, member, None)
        if sub is not None and hasattr(sub, "wave_grid"):
            grids.append(sub.wave_grid)
    if not grids:
        raise TypeError(f"{type(library).__name__} carries no wave_grid")
    ends: list[float] = []
    for grid in grids:
        arr = np.asarray(grid, dtype=np.float64)
        ends += [float(arr.min()), float(arr.max())]
    return tuple(ends)


def _template_edges(category: str, name: str) -> Callable[[], tuple[float, ...]]:
    """Edges of a block read off its registered template loader."""

    def edges() -> tuple[float, ...]:
        return template_wave_ends(AGN_BLOCK_TEMPLATE_LOADERS[(category, name)]())

    return edges


def block_edges(selectors: dict[str, str]) -> tuple[float, ...]:
    """Every registered edge [Angstrom] of a recipe's selected blocks, sorted and unique.

    Parameters
    ----------
    selectors : dict
        ``{"agn_disc_block": ..., ..., "agn_attenuation_block": ...}``.

    Returns
    -------
    tuple of float
        Host values; a block whose library file is absent contributes nothing (it cannot be
        evaluated either).
    """
    from tengri.components.agn.blocks._protocol import SPECTRAL_EDGE_BLOCKS

    found: set[float] = set()
    for kwarg, name in selectors.items():
        category = kwarg[len("agn_") : -len("_block")]
        fn = SPECTRAL_EDGE_BLOCKS.get((category, name))
        if fn is None:
            continue
        try:
            # Concrete even inside a trace: an edge function may build its constants with jnp.
            with jax.ensure_compile_time_eval():
                found.update(float(e) for e in fn())
        except FileNotFoundError:
            continue
    return tuple(sorted(found))


# Template-backed blocks: the library's own wavelength range.
for _category, _name in (
    ("disc", "kd18_agnfitter"),
    ("disc", "kd18_agnfitter_warmindex"),
    ("disc", "relagn"),
    ("disc", "slone_netzer"),
    ("disc", "schartmann2005_skirtor_atten"),
    ("torus", "cat3d_wind"),
    ("torus", "cat3d_wind_lowfwd"),
    ("torus", "nenkova_agnfitter"),
    ("torus", "nenkova_agnfitter_2p"),
    ("torus", "nenkova_agnfitter_3p"),
    ("torus", "silva04"),
    ("torus", "skirtor_agnfitter"),
    ("torus", "skirtor_agnfitter_1p"),
    ("torus", "skirtor_agnfitter_2p"),
):
    if (_category, _name) == ("disc", "schartmann2005_skirtor_atten"):
        continue
    register_spectral_edges(_category, _name)(_template_edges(_category, _name))


def _screen_xray_edges() -> tuple[float, ...]:
    """The Morrison & McCammon (1983) absorption edges below the torus libraries [Angstrom].

    The SKIRTOR and Fritz disc screens absorb the X-rays below the libraries' first node
    photoelectrically (``torus_screen.torus_screen_transmission``), with the piecewise cross
    section of ``xray.wabs_cross_section``, which steps at each of its segment edges.
    """
    from tengri.components.agn.blocks.torus_screen import XRAY_LIBRARY_EDGE_AA
    from tengri.components.xray.xray import _MM83_E_EDGES
    from tengri.utils.physics_constants import HC_KEV_ANGSTROM

    lam = HC_KEV_ANGSTROM / np.asarray(_MM83_E_EDGES, dtype=np.float64)
    return tuple(float(x) for x in lam if x < XRAY_LIBRARY_EDGE_AA)


def _screened_torus_edges(name: str) -> Callable[[], tuple[float, ...]]:
    """A screened library torus's edges: its library ends and its X-ray absorption edges."""
    library = _template_edges("torus", name)

    def edges() -> tuple[float, ...]:
        return library() + _screen_xray_edges()

    return edges


for _name in ("skirtor", "fritz"):
    register_spectral_edges("torus", _name)(_screened_torus_edges(_name))


@register_spectral_edges("disc", "multicolor")
def _multicolor_edges() -> tuple[float, ...]:
    """The EUV tail's band, 30 A to the Lyman limit (``disc._EUV_TAIL_LAMBDA_*``)."""
    from tengri.components.agn import disc

    return (disc._EUV_TAIL_LAMBDA_CUT_AA, disc._EUV_TAIL_LAMBDA_BREAK_AA)


@register_spectral_edges("disc", "powerlaw")
def _powerlaw_edges() -> tuple[float, ...]:
    """The power-law disc's band, 10 A to 1e8 A (``disc._PL_BAND_*``)."""
    from tengri.components.agn import disc

    return (disc._PL_BAND_LO_AA, disc._PL_BAND_HI_AA)


def _cigale_limits_aa(limits) -> tuple[float, ...]:
    """First and last breakpoint of a CIGALE disc [nm], in Angstrom."""
    arr = np.asarray(limits, dtype=np.float64)
    return (10.0 * float(arr[0]), 10.0 * float(arr[-1]))


@register_spectral_edges("disc", "skirtor")
def _skirtor_disc_edges() -> tuple[float, ...]:
    """CIGALE ``skirtor2016`` disc band (``disc_cigale.skirtor_disk_limits``)."""
    from tengri.components.agn.disc_cigale import skirtor_disk_limits

    return _cigale_limits_aa(skirtor_disk_limits())


@register_spectral_edges("disc", "schartmann2005")
def _schartmann_edges() -> tuple[float, ...]:
    """Schartmann (2005) disc band (``disc_cigale.schartmann2005_disk_limits``)."""
    from tengri.components.agn.disc_cigale import schartmann2005_disk_limits

    return _cigale_limits_aa(schartmann2005_disk_limits())


@register_spectral_edges("disc", "schartmann2005_skirtor_atten")
def _schartmann_atten_edges() -> tuple[float, ...]:
    """The Schartmann band and the SKIRTOR attenuation library's range."""
    return _schartmann_edges() + _template_edges("disc", "schartmann2005_skirtor_atten")()


@register_spectral_edges("disc", "adaf_lopez2024")
def _adaf_lopez_edges() -> tuple[float, ...]:
    """CIGALE ADAF + thin-disc band; its ends do not move with the blend weight."""
    from tengri.components.agn.disc_cigale import adaf_disk_limits

    return _cigale_limits_aa(adaf_disk_limits())


def _xray_floor_aa() -> float:
    """The 124 A floor below which the GRAHSP and QSOgen discs emit nothing."""
    from tengri.components.agn.grahsp.bbb import XRAY_FLOOR_NM

    return 10.0 * XRAY_FLOOR_NM


@register_spectral_edges("disc", "grahsp_sbpl")
def _grahsp_sbpl_edges() -> tuple[float, ...]:
    """The GRAHSP disc's X-ray floor (``grahsp.bbb.XRAY_FLOOR_NM``)."""
    return (_xray_floor_aa(),)


@register_spectral_edges("disc", "grahsp_netzer")
def _grahsp_netzer_edges() -> tuple[float, ...]:
    """The X-ray floor and the Netzer template's range (``netzer_disc/wave_nm``)."""
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    wave = np.asarray(load_grahsp_templates().disc_wave_nm, dtype=np.float64) * 10.0
    return (_xray_floor_aa(), float(wave.min()), float(wave.max()))


@register_spectral_edges("disc", "qsogen")
def _qsogen_disc_edges() -> tuple[float, ...]:
    """The QSOgen continuum's X-ray floor (``qsogen._XRAY_FLOOR_LAMBDA_AA``)."""
    import importlib

    # The package re-exports a ``qsogen`` function over the module's name.
    qsogen = importlib.import_module("tengri.components.agn.qsogen")
    return (qsogen._XRAY_FLOOR_LAMBDA_AA,)


@register_spectral_edges("disc", "richards2006")
def _richards_edges() -> tuple[float, ...]:
    """The Richards et al. (2006) template's range."""
    from tengri.components.agn.richards2006_disc import RICHARDS2006_WAVE_AA

    wave = np.asarray(RICHARDS2006_WAVE_AA, dtype=np.float64)
    return (float(wave.min()), float(wave.max()))


@register_spectral_edges("torus", "nenkova")
def _nenkova_toy_edges() -> tuple[float, ...]:
    """The vendored Nenkova et al. (2008) CLUMPY grid's range."""
    from tengri.components.agn import torus

    wave = _ends(torus._load_nenkova_arrays(torus._find_nenkova_grid())["wavelength"])
    return wave


@register_spectral_edges("torus", "grahsp_mn12")
def _grahsp_mn12_edges() -> tuple[float, ...]:
    """The Mor & Netzer (2012) torus template's range (``torus_mn12_wave_nm``)."""
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    return _ends(np.asarray(load_grahsp_templates().torus_mn12_wave_nm) * 10.0)


@register_spectral_edges("attenuation", "qsogen")
def _qsogen_ext_edges() -> tuple[float, ...]:
    """The Temple et al. (2021) extinction curve's domain (zero extinction outside it)."""
    from tengri.components.dust import qsogen_ext

    return (qsogen_ext._WAVE_MIN, qsogen_ext._WAVE_MAX)


def _ends(wave) -> tuple[float, ...]:
    """First and last value of a wavelength array [Angstrom]."""
    arr = np.asarray(wave, dtype=np.float64)
    return (float(arr.min()), float(arr.max()))
