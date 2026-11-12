# SPDX-License-Identifier: BSD-3-Clause
"""Refuse a splat SSP flux cube at build time (#3350).

An SSP flux whose elements are all equal (for example ``jnp.ones``) reaches
the stellar contraction as a splat constant. On jaxlib 0.10.2 through 0.12.0.1
(the newest measured) XLA CPU fuses the batched contraction into a YNNPACK kernel
that returns wrong values under ``jit(vmap(...))``. ``predict_photometry`` came
back as exact zeros; eager and single-draw ``jit`` were correct. A near-constant
or non-constant SSP is unaffected. The upstream fix is pending, so the model
refuses the splat cube with a message rather than returning a wrong number.

The check is a numpy comparison at construction. A traced cube (a model built
inside a ``jax.jit`` trace) cannot be inspected and is passed through unchanged.
"""

from __future__ import annotations

import jax
import numpy as np

from tengri.config.exceptions import ConfigError

SPLAT_SSP_MESSAGE = (
    "SSP flux is a splat: every element of ssp_flux is the same value. On jaxlib "
    "0.10.2 through 0.12.0.1 the XLA CPU YNNPACK fusion returns wrong values "
    "(exact zeros) for jit(vmap(...)) over a splat model constant, so "
    "predict_photometry would be silently wrong under vmap. Upstream bug, not a "
    "tengri physics error. Fix: use an SSP grid whose flux varies, e.g. a real "
    "stellar population library, or a smooth non-constant test grid."
)


def check_ssp_flux_not_splat(ssp_flux) -> None:
    """Raise :class:`ConfigError` when every element of ``ssp_flux`` is equal.

    Parameters
    ----------
    ssp_flux : array_like, shape (n_met, n_age, n_wave) or (n_met, n_alpha, n_age, n_wave)
        SSP flux cube. A traced value is not checked.

    Raises
    ------
    ConfigError
        If the cube is non-empty and all its elements are equal.

    Notes
    -----
    Not JIT-compatible by design: it runs once, on concrete values, at build time.
    """
    if isinstance(ssp_flux, jax.core.Tracer):
        return
    flux = np.asarray(ssp_flux)
    if flux.size == 0:
        return
    if np.all(flux == flux.flat[0]):
        raise ConfigError(SPLAT_SSP_MESSAGE)
