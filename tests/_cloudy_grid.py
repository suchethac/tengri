# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic CLOUDY grid and ionizing SSP for tests that must run without data.

The real ``cloudy_grid_<tag>.h5`` files are built locally by
``scripts/convert_fsps_cloudy_grid.py`` from raw CLOUDY output. They are not
tracked in git and no download route serves them, so a CI runner never has
one. A test that gates on the real grid therefore skips on every CI run.

The contracts that consume these grids are format-level: a band table must
reproduce the per-call path of the same backend, whatever the numbers in the
grid are. :func:`write_synthetic_cloudy_grid` writes a grid in exactly the HDF5
layout :func:`tengri.components.nebular.cloudy_grid.load_cloudy_grid` reads.
It varies smoothly along every axis, so interpolation is exercised along all
three of metallicity, age and ionization parameter, rather than being bit-exact
on a constant slab.

:func:`synthetic_ionizing_ssp` returns an :class:`SSPData` whose far-UV flux is
large enough that the young-bin Q_H passes the bare-stellar check in
``CloudyGridBackend``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.parameters.translate import LOG10_ZSUN

__all__ = ["synthetic_ionizing_ssp", "write_synthetic_cloudy_grid"]

#: Log10(Z/Zsun) nodes. They cover the metallicity and gas-metallicity priors
#: the contract tests draw from, (-1.3, 0.2), with a node of margin on each side.
_LOG_MET_AXIS = np.array([-1.5, -1.0, -0.5, 0.0, 0.5])

#: log10(age / yr) nodes, from 0.1 Myr to 100 Myr (the backend's nebular age cap).
_LOG_AGE_AXIS = np.array([5.0, 6.0, 6.5, 7.0, 7.5, 8.0])

#: log10(U) nodes, covering the ionization-parameter prior (-5, 0).
_LOG_U_AXIS = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0])

#: Rest-frame wavelengths [Angstrom] of six nebular lines, Lyman-alpha to Halpha.
_LINE_WAVE_AA = np.array([1216.0, 3727.0, 4861.0, 4959.0, 5007.0, 6563.0])

#: Extra log10 luminosity on the Ly-alpha line (index 0). Without it the line is
#: about 1e-13 of the FUV band and Ly-alpha escape cannot move any band. With it,
#: Ly-alpha dominates the FUV band, as it does in CLOUDY nebular spectra.
_LYA_BOOST_DEX = 12.0

#: Nebular continuum wavelengths [Angstrom], spanning the Lyman limit to 10 um.
_CONT_WAVE_AA = np.logspace(np.log10(912.0), 5.0, 60)

#: SSP grid for the synthetic ionizing population. Ages and metallicities are
#: chosen so the SSP age and metallicity axes sit inside the grid's own axes.
_SSP_LG_AGE_GYR = np.linspace(-3.0, 1.14, 25)
_SSP_LGMET = np.array([-4.0, -2.65, -1.3])

#: Scale on the far-UV power law. Unscaled, the young-bin log10 Q_H is near 63;
#: 1e-16 brings it to about 47, inside the bare-stellar range of 10^47 to 10^50.
_UV_FLUX_SCALE: float = 1.0e-16


def _log_line_luminosity(log_u: np.ndarray) -> np.ndarray:
    """log10 L_line / Q_H on the (met, age, U, line) grid, smooth in every axis."""
    met = _LOG_MET_AXIS[:, None, None, None]
    age = _LOG_AGE_AXIS[None, :, None, None]
    logu = log_u[None, None, :, None]
    line = np.arange(len(_LINE_WAVE_AA))[None, None, None, :]
    return (
        -12.0
        + 0.3 * line
        + 0.4 * met
        + 0.2 * (age - 7.0)
        - 0.5 * (logu + 2.0)
        + 0.1 * met * (logu + 2.0)
        + _LYA_BOOST_DEX * (line == 0)
    )


def _log_continuum_luminosity(log_u: np.ndarray) -> np.ndarray:
    """log10 L_nu / Q_H on the (met, age, U, wave) grid, smooth in every axis."""
    met = _LOG_MET_AXIS[:, None, None, None]
    age = _LOG_AGE_AXIS[None, :, None, None]
    logu = log_u[None, None, :, None]
    log_wave = np.log10(_CONT_WAVE_AA)[None, None, None, :]
    return -14.0 - 0.5 * (log_wave - 3.0) + 0.2 * met + 0.1 * (age - 7.0) - 0.3 * (logu + 2.0)


def write_synthetic_cloudy_grid(
    path: str | Path, log_u_axis: Sequence[float] | None = None
) -> Path:
    """Write a deterministic CLOUDY-format grid to ``path`` and return it.

    The file has the datasets ``load_cloudy_grid`` reads: ``lines/`` and
    ``continuum/``, each with a ``wavelength``, a ``luminosity`` and an ``axes/``
    group holding ``log_met`` (log10 Z/Zsun), ``log_age_yr`` and ``log_U``.
    Luminosities are stored linear, as the converter writes them.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination ``.h5`` file. It is overwritten if it exists.
    log_u_axis : sequence of float, optional
        The log10(U) nodes [dex], strictly increasing. Defaults to the module
        axis, -5 to 0. Pass the real grids' hull, (-4, -3, -2, -1), to test
        behavior at that hull.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    out = Path(path)
    log_u = _LOG_U_AXIS if log_u_axis is None else np.asarray(log_u_axis, dtype=float)
    lines_lum = 10.0 ** _log_line_luminosity(log_u)
    cont_lum = 10.0 ** _log_continuum_luminosity(log_u)
    with h5py.File(out, "w") as f:
        for group, wave, lum in (
            ("lines", _LINE_WAVE_AA, lines_lum),
            ("continuum", _CONT_WAVE_AA, cont_lum),
        ):
            grp = f.create_group(group)
            grp.create_dataset("wavelength", data=wave)
            grp.create_dataset("luminosity", data=lum)
            axes = grp.create_group("axes")
            axes.create_dataset("log_met", data=_LOG_MET_AXIS)
            axes.create_dataset("log_age_yr", data=_LOG_AGE_AXIS)
            axes.create_dataset("log_U", data=log_u)
        f.attrs["source"] = "synthetic (tests/_cloudy_grid.py)"
    return out


def synthetic_ionizing_ssp() -> SSPData:
    """A bare-stellar SSP with a far-UV flux that gives a physical Q_H.

    Returns
    -------
    SSPData
        Three metallicities by 25 ages by 1600 wavelengths from 100 A to 10 mm.
        The flux is a steep power law, bright in the far UV, so the young bins
        carry ionizing photons and the wNE check passes.
    """
    wave = jnp.logspace(2.0, 7.0, 1600)
    base = (5000.0 / wave) ** 2
    age_factor = 10.0 ** (-0.3 * (_SSP_LG_AGE_GYR - _SSP_LG_AGE_GYR.mean()))
    flux = (
        _UV_FLUX_SCALE
        * base[None, None, :]
        * jnp.asarray(age_factor)[None, :, None]
        * (1.0 + 0.1 * _SSP_LGMET)[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=jnp.asarray(_SSP_LG_AGE_GYR),
        ssp_lgmet=jnp.asarray(_SSP_LGMET + LOG10_ZSUN),
    )
