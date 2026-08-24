# SPDX-License-Identifier: BSD-3-Clause
r"""Photometric filter-convolution convention and bandpass weights.

A leaf module (no internal ``tengri`` imports) so both the exact kernel
(:mod:`tengri.observation.photometry`) and the build-time preintegration
(:mod:`tengri.utils.grid_interp`) can share one definition of the bandpass
weight :math:`w(\lambda)` without a circular import.

See ``docs/units.md`` (Photometric filter-convolution convention) and
``docs/adr/0017-photometric-filter-convention.md`` for the authoritative
reference.
"""

from __future__ import annotations

from enum import StrEnum

import jax.numpy as jnp
import numpy as np

__all__ = [
    "FilterConvention",
    "filter_weight",
    "filter_weight_np",
    "list_filter_conventions",
]


class FilterConvention(StrEnum):
    r"""Broadband filter-convolution convention (the bandpass weight).

    The band-averaged flux density is

    .. math::

        \langle F_\nu\rangle = \frac{\int F_\nu(\lambda)\,T(\lambda)\,w(\lambda)\,d\lambda}
                                    {\int T(\lambda)\,w(\lambda)\,d\lambda},

    and the member selects the weight :math:`w(\lambda)`:

    - ``BESSELL``: photon-counting, :math:`w = 1/\lambda`. The physically
      correct mean for photon-counting detectors (all optical/NIR CCDs) and
      how the AB system is realized by surveys. Matches DSPS, FSPS, sedpy,
      and prospector. **This is the default.**
    - ``ENERGY``: energy-counting, :math:`w = 1/\lambda^2` (i.e. flat in
      frequency, :math:`\int F_\nu T\,d\nu / \int T\,d\nu`). Matches CIGALE
      and bagpipes; use it when reproducing those codes.

    The two agree exactly for a flat-:math:`F_\nu` source (the AB reference)
    and diverge by 5–40 mmag, band- and SED-slope-dependent, otherwise.

    References
    ----------
    .. [1] Hogg, Baldry, Blanton & Eisenstein 2002, "The K correction",
       arXiv:astro-ph/0210394, Eq. 5 (photon-counting AB definition).
    .. [2] Fukugita et al. 1996, AJ 111, 1748, Eq. 7 (FSPS's cited form).
    .. [3] Hearin et al. 2023, "DSPS", arXiv:2112.06830 (rest-frame mag,
       :math:`w = 1/\lambda`, :math:`T_Q` = photon transmission probability).
    .. [4] Boquien et al. 2019, A&A 622, A103 (CIGALE energy convention).

    See Also
    --------
    list_filter_conventions: Names + descriptions of the supported conventions.

    """

    BESSELL = "bessell"
    ENERGY = "energy"


def list_filter_conventions():
    """Return the supported photometric filter conventions and their meaning.

    Returns
    -------
    _RegistryTable
        One row per convention: ``{"name": ..., "kind": "filter_convention",
        "short_doc": ...}``. Renders as a table in a notebook.

        This used to return ``dict[str, str]``, one of only two ``list_*``
        that did (#1285). Use ``.to_dict()`` for the old mapping.

    Examples
    --------
    >>> from tengri.utils.filter_convention import list_filter_conventions
    >>> list_filter_conventions().to_dict()["bessell"]
    'Photon-counting, weight 1/lambda (default; DSPS/FSPS/sedpy).'

    """
    from tengri.registry import _RegistryTable

    return _RegistryTable(
        [
            {
                "name": FilterConvention.BESSELL.value,
                "kind": "filter_convention",
                "short_doc": "Photon-counting, weight 1/lambda (default; DSPS/FSPS/sedpy).",
            },
            {
                "name": FilterConvention.ENERGY.value,
                "kind": "filter_convention",
                "short_doc": (
                    "Energy-counting, weight 1/lambda^2 / flat-in-frequency (CIGALE/bagpipes)."
                ),
            },
        ]
    )


def filter_weight(filter_wave: jnp.ndarray, convention: FilterConvention) -> jnp.ndarray:
    r"""Bandpass weight :math:`w(\lambda)` for the chosen convention (JAX).

    Returns ``1/lambda`` for :class:`FilterConvention.BESSELL` (default) and
    ``1/lambda**2`` for :class:`FilterConvention.ENERGY`. Zero-padded filter
    entries (``filter_wave == 0``) map to weight ``0`` so they contribute
    nothing to the integral and never produce ``inf``/``nan``.

    Parameters
    ----------
    filter_wave: ndarray, shape (n_filt,)
        Filter wavelength grid [Ångström]; may contain zero padding.
    convention: FilterConvention
        Selects the weight (a static Python value, not a traced array).

    Returns
    -------
    ndarray, shape (n_filt,)
        The weight at each wavelength.

    """
    positive = filter_wave > 0
    safe = jnp.where(positive, filter_wave, 1.0)
    inv = jnp.where(positive, 1.0 / safe, 0.0)
    if convention == FilterConvention.ENERGY:
        return inv * inv
    return inv


def filter_weight_np(fw: np.ndarray, convention: FilterConvention) -> np.ndarray:
    """Bandpass weight ``w(lambda)`` (numpy build-time twin of :func:`filter_weight`).

    ``1/lambda`` for ``BESSELL`` (default), ``1/lambda**2`` for ``ENERGY``;
    zero-padded entries (``fw == 0``) map to weight 0.
    """
    pos = fw > 0
    inv = np.where(pos, 1.0 / np.where(pos, fw, 1.0), 0.0)
    return inv * inv if convention == FilterConvention.ENERGY else inv
