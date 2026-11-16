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
import jax.numpy as jnp
import numpy as np

from tengri.utils.filter_convention import FilterConvention
from tengri.utils.scale import representable_denominator

__all__ = ["BandSpec", "delta_line_matrix", "line_band_weights", "project_bands"]


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

    def pivot_wavelengths(self) -> np.ndarray:
        r"""Observed-frame effective wavelength of each band, :math:`\int\lambda T w/\int T w`.

        Evaluated on the filter's own nodes under the convention weight; it does not depend
        on redshift, so it is the point a transmission read once per band (the IGM) uses.

        Returns
        -------
        ndarray, shape (n_filters,)
            Pivot wavelengths [Angstrom].
        """
        from tengri.utils.filter_convention import filter_weight_np
        from tengri.utils.grid_interp import _np_trapezoid

        pivots = []
        for fw, ft in zip(*self.filters, strict=True):
            fw_np, ft_np = np.asarray(fw), np.asarray(ft)
            tw = ft_np * filter_weight_np(fw_np, self.convention)
            pivots.append(_np_trapezoid(tw * fw_np, fw_np) / max(_np_trapezoid(tw, fw_np), 1e-30))
        return np.array(pivots)

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
    node_transmission: Any = None,
    select: Any = None,
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
    node_transmission : array_like, shape (n_wave,), optional
        A factor evaluated at each node of ``wave_rest`` and multiplied into the basis
        before it is integrated (the IGM transmission on the rest grid, for example):
        the integral then reads :math:`\int L_\nu T\,T_b\,w\,d\lambda`, with ``T``
        interpolated linearly between nodes along with the SED.
    select : array_like of int, optional
        Runtime engine only: the bands to integrate, as indices into ``spec.filters``
        (may be traced). Every other band is not evaluated.

    Returns
    -------
    PreintegratedGrid or jax.Array
        The preintegrated table for a NumPy basis; per-band filter-weighted ``L_nu`` of shape
        ``(n_filters,)`` [erg/s/Hz] for a JAX basis.
    """
    if grid_dims is not None:
        from tengri.utils.grid_interp import preintegrate_transposed

        if node_transmission is not None:
            basis = basis * np.asarray(node_transmission, dtype=np.float64)[:, None]
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
    if select is not None:
        fw_pad, ft_pad = jnp.asarray(fw_pad)[select], jnp.asarray(ft_pad)[select]
    if node_transmission is not None:
        basis = basis * node_transmission
    return lnu_filter_integral_batch(
        basis, wave_rest, fw_pad, ft_pad, z, convention=spec.convention
    )


def line_band_weights(
    line_waves_rest: Any,
    spec: BandSpec,
    z: Any,
    *,
    per_hz: bool,
    sigma_kms: Any = None,
    delta_v_kms: Any = None,
) -> Any:
    r"""Band response of emission lines: the one line projection.

    A delta line at rest wavelength :math:`\lambda_0` seen through band :math:`b` of
    convention weight :math:`w` has the observed wavelength
    :math:`\lambda_{\rm obs}=(1+z)\lambda_0` and

    .. math::

        r_b = \frac{T_b(\lambda_{\rm obs})\,w(\lambda_{\rm obs})}{\int T_b\,w\,d\lambda}.

    With ``per_hz`` the response of a unit line luminosity [erg/s] to the band-averaged
    :math:`L_\nu` [1/Hz] is returned, :math:`r_b\,(1+z)\lambda_0^2/c`; the Jacobian
    :math:`(1+z)\lambda_0^2/c` converts the delta in frequency to the delta in wavelength.
    For the photon-counting weight this is :math:`\lambda_0 T_b/(c\int T_b/\lambda\,d\lambda)`;
    the energy weight carries :math:`1/\lambda_{\rm obs}` less
    (:func:`~tengri.utils.filter_convention.photon_relative_weight`). The normalization
    :math:`\int T_b w\,d\lambda` runs over the filter's own nodes, as in the exact integral.

    A line of velocity dispersion :math:`\sigma` is a Gaussian in :math:`\ln\lambda` of width
    :math:`\sigma/c`, centered at the line shifted by the offset
    (:math:`\lambda_0(1+\Delta v/c)`). Its response is the Gaussian average of the same
    integrand, :math:`\int \mathcal{N}(s)\,F(\lambda_c e^{s})\,ds` with :math:`F` the delta
    response above. :math:`T_b` is linear between filter nodes and :math:`F` a power law in
    :math:`\lambda`, so each segment is a sum of two moments :math:`\int e^{ks}\mathcal{N}\,ds`,
    composed from :func:`~tengri.utils.gaussian_segments.ndtr_diff` (``k = 0``) and
    :func:`~tengri.utils.gaussian_segments.gaussian_exp_moments` (``k = 1, 2``; ``k = -1, -2``
    by reflecting the segment, since the kernel is symmetric).

    Parameters
    ----------
    line_waves_rest : array_like, shape (n_lines,)
        Rest wavelengths of the lines as the SED renders them [Angstrom].
    spec : BandSpec
        Filters and convention.
    z : float or jax.Array
        Redshift; may be traced.
    per_hz : bool
        Include the Jacobian :math:`(1+z)\lambda_0^2/c` (response in [1/Hz]); without it the
        weight is per Angstrom of observed wavelength.
    sigma_kms : array_like, shape (n_lines,) or scalar, optional
        Intrinsic velocity dispersion of each line [km/s]; may be traced. ``None`` is a delta
        line. A width of exactly 0 returns the delta result bit for bit.
    delta_v_kms : array_like, shape (n_lines,) or scalar, optional
        Velocity offset of each line from the systemic redshift [km/s]; may be traced.

    Returns
    -------
    jax.Array, shape (n_lines, n_filters)
        :math:`r_b` [1/Angstrom], or [1/Hz] with ``per_hz``.

    Notes
    -----
    **JIT-compatible**: yes (``spec`` is static). **Gradient-safe**: the width enters only
    through :math:`\sigma/c` divided into the segment limits, with a safe stand-in where the
    width is 0, so a zero width has a finite gradient (the delta branch, constant in the
    width); for a positive width the gradient with respect to it is that of the Gaussian
    average. Float32 loses a few parts in :math:`10^{6}` to the two moments of a segment
    nearly canceling.
    """

    from tengri.utils.filter_convention import filter_weight, photon_relative_weight
    from tengri.utils.physics_constants import C_AA

    line_waves = jnp.asarray(line_waves_rest)
    if delta_v_kms is not None:
        from tengri.utils.physics_constants import C_KM_S

        line_waves = line_waves * (1.0 + jnp.asarray(delta_v_kms) / C_KM_S)
    obs = (1.0 + jnp.asarray(z)) * line_waves
    cols = []
    for fw, ft in zip(*spec.filters, strict=True):
        fw = jnp.asarray(fw)
        ft = jnp.asarray(ft)
        trans = jnp.interp(obs, fw, ft, left=0.0, right=0.0)
        norm = jnp.trapezoid(ft * filter_weight(fw, spec.convention), fw)
        if per_hz:
            rel = photon_relative_weight(obs, spec.convention)
            delta = line_waves * trans * rel / (C_AA * norm)
        else:
            delta = (
                trans
                * filter_weight(obs, spec.convention)
                / jnp.maximum(norm, representable_denominator(1e-30))
            )
        if sigma_kms is None:
            cols.append(delta)
            continue
        gauss = _gaussian_line_response(
            obs, jnp.asarray(z), sigma_kms, fw, ft, norm, spec.convention, per_hz
        )
        cols.append(jnp.where(jnp.asarray(sigma_kms) > 0.0, gauss, delta))
    return jnp.stack(cols, axis=-1)


def _gaussian_line_response(obs, z, sigma_kms, fw, ft, norm, convention, per_hz):
    r"""Response of Gaussian lines (width in :math:`\ln\lambda`) through one band.

    The Gaussian average of the delta response over the piecewise-linear transmission,
    one segment at a time (see :func:`line_band_weights`). ``sigma_kms`` of 0 is replaced
    by a stand-in width so no division by zero reaches the graph; the caller selects the
    delta result there.
    """
    from tengri.utils.filter_convention import FilterConvention
    from tengri.utils.gaussian_segments import gaussian_exp_moments, ndtr_diff
    from tengri.utils.physics_constants import C_AA, C_KM_S

    sigma = jnp.asarray(sigma_kms)
    width = jnp.where(sigma > 0.0, sigma / C_KM_S, 1.0)  # ln-lambda width, never 0
    energy = convention == FilterConvention.ENERGY
    # F(lambda) = lambda**q T(lambda): the delta response is T w J up to the constant below,
    # with J = lambda**2 (per_hz) or 1, and w = lambda**-1 or lambda**-2.
    q = (0 if energy else 1) if per_hz else (-2 if energy else -1)

    lam_lo, lam_hi = fw[:-1], fw[1:]
    slope = (ft[1:] - ft[:-1]) / (lam_hi - lam_lo)
    alpha = ft[:-1] - slope * lam_lo
    mu = jnp.log(obs)[..., None]
    s_lo = jnp.log(lam_lo) - mu
    s_hi = jnp.log(lam_hi) - mu
    w = width[..., None]

    def moment(k):
        if k == 0:
            return ndtr_diff(s_hi / w, s_lo / w)
        if k > 0:
            return gaussian_exp_moments(s_lo, s_hi, w)[k - 1]
        return gaussian_exp_moments(-s_hi, -s_lo, w)[-k - 1]

    lam_c = obs[..., None]
    total = jnp.sum(
        alpha * lam_c**q * moment(q) + slope * lam_c ** (q + 1) * moment(q + 1), axis=-1
    )
    if per_hz:
        return total / ((1.0 + z) * C_AA * norm)
    return total / jnp.maximum(norm, representable_denominator(1e-30))


def delta_line_matrix(
    line_waves_rest: Any,
    filter_waves: Sequence[Any],
    filter_trans: Sequence[Any],
    redshift: float,
    convention: FilterConvention,
    sigma_kms: Any = None,
    delta_v_kms: Any = None,
) -> np.ndarray:
    r"""Response of each unit-luminosity line to each band, as a float64 matrix.

    The build-time form of :func:`line_band_weights` (``per_hz=True``) for the line-emitter
    adapters: ``matrix.T @ line_luminosity [erg/s]`` is the band-averaged :math:`L_\nu`
    [erg/s/Hz] of the lines.

    Parameters
    ----------
    line_waves_rest : array_like, shape (n_lines,)
        Rest wavelengths [Angstrom].
    filter_waves, filter_trans : sequence of array_like
        Observed-frame filter curves [Angstrom], [dimensionless].
    redshift : float
        Source redshift [dimensionless].
    convention : FilterConvention
        Bandpass weight.
    sigma_kms, delta_v_kms : array_like, optional
        Per-line velocity dispersion and offset [km/s] (see :func:`line_band_weights`);
        ``None`` is a delta line at the systemic redshift.

    Returns
    -------
    ndarray, shape (n_lines, n_filters)
        Band response [1/Hz].
    """
    spec = BandSpec(
        filters=([np.asarray(fw) for fw in filter_waves], [np.asarray(ft) for ft in filter_trans]),
        convention=convention,
        n_subbands=None,
        lyc_gate=False,
    )
    weights = line_band_weights(
        np.asarray(line_waves_rest, dtype=np.float64),
        spec,
        redshift,
        per_hz=True,
        sigma_kms=sigma_kms,
        delta_v_kms=delta_v_kms,
    )
    return np.asarray(weights, dtype=np.float64)


def _flatten_spec(spec: BandSpec):
    return (), spec


def _unflatten_spec(spec: BandSpec, children) -> BandSpec:
    return spec


# A spec carries no traced values: it rides through a jit boundary as static structure.
jax.tree_util.register_pytree_node(BandSpec, _flatten_spec, _unflatten_spec)
