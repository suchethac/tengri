# SPDX-License-Identifier: BSD-3-Clause
r"""One band-projection contract: the filter set, its convention and its sub-band scheme.

Every band integral in tengri, build-time table or runtime projection, integrates a rest-frame
basis through a filter set with one bandpass weight :math:`w(\lambda)` (photon-counting
``1/lambda`` or energy ``1/lambda^2``) and, for a multiplicative screen, one sub-band
partition. :class:`BandSpec` carries those three choices together, with no default, so the
convention cannot diverge between a table builder and the runtime projection that reads it.

:func:`project_bands` is the single entry point. A build-time table (``grid_dims`` given) runs
the union-grid preintegration engine (:func:`~tengri.utils.grid_interp.preintegrate_transposed`).
A runtime SED runs the exact union-grid integral
(:func:`~tengri.observation.photometry.lnu_filter_integral_batch`). Both integrate the
transmission-weighted quotient

.. math::

    \Phi_b = \frac{\int L_\nu(\lambda)\, T_b(\lambda_{\rm obs})\, w(\lambda_{\rm obs})\,
    d\lambda}{\int T_b\, w\, d\lambda}.

**JIT-compatible**: the runtime engine is; the build engine is NumPy and runs at build time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from tengri.utils.filter_convention import FilterConvention

__all__ = ["BandSpec", "project_bands"]


@dataclass(frozen=True, eq=False)
class BandSpec:
    """The band-integral choices every table builder and runtime projection shares.

    Parameters
    ----------
    filters : tuple of (sequence of array_like, sequence of array_like)
        ``(filter_waves, filter_trans)``: per-band observed-frame wavelengths [Angstrom] and
        dimensionless transmission, one array per band, ascending and unpadded.
    convention : FilterConvention
        Bandpass weight. No default: a caller states the observation's convention.
    n_subbands : int or None
        Equal-filter-mass sub-bands per band for a multiplicative screen; ``None`` or ``0``
        disables the sub-band tables.
    lyc_gate : bool
        Whether a live nebular Lyman-continuum mask is in force (forces the physical edge into
        the sub-band partition and computes the Lyman-continuum split).

    Notes
    -----
    Equality and hashing follow :meth:`key`, so a spec is a valid static JIT argument and a
    cache-key component.
    """

    filters: tuple[Sequence[Any], Sequence[Any]]
    convention: FilterConvention
    n_subbands: int | None
    lyc_gate: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "convention", FilterConvention(self.convention))
        waves, trans = self.filters
        if len(waves) != len(trans):
            raise ValueError(
                f"BandSpec.filters: {len(waves)} wavelength arrays but "
                f"{len(trans)} transmission arrays"
            )

    @property
    def n_filters(self) -> int:
        """Number of bands."""
        return len(self.filters[0])

    @property
    def subbands(self) -> int:
        """Sub-band count with ``None`` read as zero."""
        return int(self.n_subbands or 0)

    def key(self) -> str:
        """Content hash of the filter curves, convention, sub-band scheme and Lyman gate.

        Returns
        -------
        str
            Stable hex digest; feeds LUT cache keys and the compile signature.
        """
        h = hashlib.sha256()
        for arrays in self.filters:
            for a in arrays:
                h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
                h.update(b"|")
            h.update(b"#")
        h.update(f"{self.convention.value}:{self.subbands}:{int(self.lyc_gate)}".encode())
        return h.hexdigest()[:24]

    def __hash__(self) -> int:
        return hash(self.key())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BandSpec) and self.key() == other.key()

    def padded(self) -> tuple[np.ndarray, np.ndarray]:
        """The filter curves zero-padded to one length, as the runtime projections read them.

        Returns
        -------
        fw_pad, ft_pad : ndarray, shape (n_filters, max_len)
            Observed-frame wavelengths [Angstrom] and transmission, host arrays.
        """
        cached = self.__dict__.get("_padded")
        if cached is None:
            from tengri.observation.photometry import pad_filters_host

            fw, ft, _ = pad_filters_host(list(self.filters[0]), list(self.filters[1]))
            cached = (fw, ft)
            object.__setattr__(self, "_padded", cached)
        return cached

    def replace(self, **changes: Any) -> BandSpec:
        """Return a copy with the named fields changed (the spec is frozen).

        Parameters
        ----------
        **changes
            Field values to override.

        Returns
        -------
        BandSpec
            The new spec.
        """
        fields = {
            "filters": self.filters,
            "convention": self.convention,
            "n_subbands": self.n_subbands,
            "lyc_gate": self.lyc_gate,
        }
        return BandSpec(**{**fields, **changes})


def project_bands(
    basis: Any,
    wave_rest: Any,
    spec: BandSpec,
    z: Any,
    *,
    grid_dims: tuple[int, ...] | None = None,
    axes: tuple[Any, ...] = (),
    taylor: bool = False,
    dl_cm: float = 1.0,
) -> Any:
    r"""Project a rest-frame basis onto the bands of ``spec``.

    Parameters
    ----------
    basis : ndarray or jax.Array
        With ``grid_dims`` given, a NumPy array of shape ``(n_wave, n_grid_points)`` (the
        transposed template layout, see :func:`~tengri.utils.grid_interp.transpose_templates`)
        is a build-time table. Without it, an array of shape ``(n_wave,)`` is one runtime SED
        [erg/s/Hz].
    wave_rest : array_like, shape (n_wave,)
        Rest-frame wavelength of ``basis`` [Angstrom].
    spec : BandSpec
        Filters, convention, sub-band scheme.
    z : float or jax.Array
        Redshift.
    grid_dims : tuple of int, optional
        Leading shape of the build-time result; selects the build-time engine.
    axes : tuple of ndarray, optional
        Grid axes for runtime interpolation of the build-time result.
    taylor : bool
        Build-time only: also precompute the first spectral moment.
    dl_cm : float
        Build-time luminosity distance [cm].

    Returns
    -------
    PreintegratedGrid or jax.Array
        The preintegrated table for a NumPy basis; per-band filter-weighted ``L_nu`` of shape
        ``(n_filters,)`` [erg/s/Hz] for a JAX basis.
    """
    if grid_dims is not None:
        from tengri.utils.grid_interp import preintegrate_transposed

        return preintegrate_transposed(
            basis,
            tuple(grid_dims),
            np.asarray(wave_rest),
            list(spec.filters[0]),
            list(spec.filters[1]),
            float(z),
            dl_cm,
            axes=axes,
            taylor=taylor,
            n_subbands=spec.subbands,
            convention=spec.convention,
            lyc_gate=spec.lyc_gate,
        )
    from tengri.observation.photometry import lnu_filter_integral_batch

    fw_pad, ft_pad = spec.padded()
    return lnu_filter_integral_batch(
        basis, wave_rest, fw_pad, ft_pad, z, convention=spec.convention
    )


def _flatten_spec(spec: BandSpec):
    return (), spec


def _unflatten_spec(spec: BandSpec, children) -> BandSpec:
    return spec


# A spec carries no traced values: it rides through a jit boundary as static structure.
jax.tree_util.register_pytree_node(BandSpec, _flatten_spec, _unflatten_spec)
