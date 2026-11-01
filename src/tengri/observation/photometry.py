# SPDX-License-Identifier: BSD-3-Clause
"""Photometric filter convolution.

Computes observed flux densities by convolving the rest-frame SED
(redshifted) through filter transmission curves.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

# FilterConvention + the bandpass weight live in a leaf module so the exact
# kernel here and the build-time preintegration (utils.grid_interp) share one
# definition without a circular import. Re-exported here for back-compat.
from tengri._cache_keys import KeyPolicy, content, derive_key
from tengri.components.lyc import LYMAN_LIMIT_AA, edge_bracket_values, edge_interp
from tengri.parameters.resolve import require_redshift
from tengri.units import fnu_to_ab_mag, lnu_to_fnu
from tengri.utils.filter_convention import (
    FilterConvention,
    filter_weight as _filter_weight,
    filter_weight_np,
    list_filter_conventions,
)
from tengri.utils.scale import representable_denominator

__all__ = [
    "FilterConvention",
    "FilterCurve",
    "ab_mag_from_flux",
    "compute_flux_density",
    "compute_flux_density_batch",
    "compute_photometry",
    "list_filter_conventions",
    "lnu_filter_integral",
    "lnu_filter_integral_batch",
    "pad_filters",
    "project_photometry",
]


@dataclasses.dataclass(frozen=True)
class FilterCurve:
    """Photometric filter transmission curve.

    Represents a single broad-band photometric filter via its wavelength-dependent
    transmission profile. Used to convolve SEDs and compute observed flux densities.

    Attributes
    ----------
    wave : array, shape (n_wave,)
        Wavelength grid [Ångstrom]. Should be at least 10 points spanning
        the transmission curve from near zero to near zero.
    trans : array, shape (n_wave,)
        Transmission at each wavelength (dimensionless, 0.0–1.0).
        Typically peaked at 1.0 and falls to 0 at the filter edges.
    name : str, optional
        Filter identifier (e.g., ``"sdss_r"``, ``"jwst_f200w"``, ``"hsc_i"``).
        Used for diagnostic output and filter registry lookups. Default empty string.

    Notes
    -----
    **Standard sources**:

    - Optical/NIR: Spanish Virtual Observatory (SVO) Filter Profile Service
    - JWST: STScI filter definitions (via astropy.io.fits)
    - Custom: User-provided arrays

    **Filter conventions**:

    - Transmission is not flux-normalized (raw instrumental response)
    - Wavelength grid should be uniform or fine enough to resolve structure
    - Outside [wave[0], wave[-1]], transmission is assumed zero

    See Also
    --------
    load_filter_set : Load filter set from SVO database.
    compute_flux_density : Convolve SED through this filter.
    pad_filters : Stack variable-length filter arrays.

    """

    wave: jnp.ndarray = dataclasses.field(hash=False)
    trans: jnp.ndarray = dataclasses.field(hash=False)
    name: str = ""

    def cache_key(self) -> tuple:
        """Return a hashable cache key for this filter curve.

        Returns
        -------
        tuple
            Cache key derived from wave, trans, and name.

        Notes
        -----
        All attributes are included by content.
        """
        return derive_key(self, _FILTER_CURVE_CACHE_KEY_POLICY)


_FILTER_CURVE_CACHE_KEY_POLICY: KeyPolicy = {
    "wave": content("wavelength array is part of the filter definition"),
    "trans": content("transmission curve is part of the filter definition"),
    "name": content("filter name identifies the filter"),
}


def _filter_integral_union(
    L_nu: jnp.ndarray,
    wave_obs: jnp.ndarray,
    filter_wave: jnp.ndarray,
    filter_trans: jnp.ndarray,
    convention: FilterConvention,
    lyman_edge_obs_aa: jnp.ndarray | float | None = None,
) -> jnp.ndarray:
    r"""Filter-weighted mean of ``L_nu`` on the union quadrature grid (#960).

    Integrates on the sorted union of the SED nodes and the filter nodes,
    interpolating the *transmission* (smooth by construction) rather than
    the SED. Point-sampling the SED at the filter table's nodes, the
    pre-#960 quadrature, biases any band whose table is coarser than the
    spectral structure it covers: SVO/sedpy instrument tables are 25–70 Å
    spaced while MILES spectra carry ~1 Å absorption-line structure, which
    produced band-dependent errors up to 3 % (SDSS g). FSPS
    (``getmags.f90``) and sedpy integrate on the SED grid for the same
    reason; the union grid additionally stays exact when a narrow filter
    falls on a coarse region of the SED grid.

    Both node arrays must be ascending; padded filter rows must be made
    ascending first (:func:`_ascending_padded_filter`).

    The denominator floor is sized for the **derivative**, not the value
    (#1860). ``pad_filters_to_bucket`` pads the filter-count axis with all-zero
    rows, so a padded row arrives here with ``num == den == 0``. Forward that is
    safe at any floor, but the quotient's VJP carries ``-num/den**2``, and the
    former literal ``1e-30`` squares to exactly ``0.0`` in float32, the reverse
    pass then divided by zero and returned a **NaN redshift gradient**. Only
    redshift saw it: ``den`` integrates over ``grid``, which scales with
    ``(1+z)``, so z is the one parameter that reaches the denominator at all.
    ``representable_floor`` does not catch this, ``1e-30`` is above float32's
    ``tiny`` and passes through untouched. float64 is bit-identical.

    Parameters
    ----------
    lyman_edge_obs_aa : float, optional
        Observed-frame Lyman limit, ``LYMAN_LIMIT_AA * (1+z)`` (#2447, one
        Lyman edge). When given, ``L_nu`` is resampled onto ``grid`` with
        :func:`tengri.components.lyc.edge_interp` (a step in the bracket
        cell straddling the edge, ordinary linear interpolation elsewhere),
        and the edge itself is additionally inserted into the quadrature
        grid as a zero-width node pair
        (:func:`tengri.components.lyc.edge_bracket_values`) so the ordinary
        ``jnp.trapezoid`` numerator integrates the step exactly even though
        the transmission x bandpass weight varies continuously across the
        bracket cell (that continuous variation is why a single
        ``edge_trapezoid`` call on the already-multiplied integrand would
        only be approximate here, unlike the pure-SED case). Exact
        regardless of how many extra filter nodes subdivide the straddling
        SSP panel, because ``L_nu`` already carries the physical masking
        (and any dust/IGM multiplicative factor) PER NODE -- there is no
        separate correction term to compute, unlike the retired
        ``_lyc_photometry_correction`` this replaces. ``None`` (default)
        reproduces ordinary (non-edge-aware) quadrature, correct whenever no
        live Lyman-continuum mask ran (e.g. ``neb={'type': 'none'}``) and
        ``L_nu`` is smooth across the edge.
    """
    # Quadrature confined to the filter's own support (#2817). The union grid is
    # unchanged; both filter edges are nodes of it, so every trapezoid interval is
    # wholly inside or wholly outside ``[filter_wave.min(), filter_wave.max()]``.
    # Intervals outside are masked out, so the transmission's zero fill past the
    # edge never ramps across a straddle segment (the ramp is what jumped as an SED
    # node crossed the edge). Nothing is clamped, so the gradient is clean.
    grid = jnp.sort(jnp.concatenate([wave_obs, filter_wave]))
    if lyman_edge_obs_aa is None:
        L_on_grid = jnp.interp(grid, wave_obs, L_nu, left=0.0, right=0.0)
        trans_on_grid = jnp.interp(grid, filter_wave, filter_trans, left=0.0, right=0.0)
        weight = trans_on_grid * _filter_weight(grid, convention)
        x_q, L_q, w_q = grid, L_on_grid, weight
    else:
        # One Lyman edge (#2447): the edge is inserted as a zero-width node pair
        # carrying the one-sided values, so an ordinary trapezoid integrates the
        # step exactly (see edge_bracket_values).
        y_a, y_b = edge_bracket_values(wave_obs, L_nu, edge_aa=lyman_edge_obs_aa)
        edge_pair = jnp.full((2,), lyman_edge_obs_aa, dtype=grid.dtype)
        all_x = jnp.concatenate([grid, edge_pair])
        L_base = edge_interp(grid, wave_obs, L_nu, edge_aa=lyman_edge_obs_aa)
        all_y = jnp.concatenate([L_base, jnp.stack([y_a, y_b])])
        order = jnp.argsort(all_x, stable=True)
        x_q = all_x[order]
        L_q = all_y[order]
        trans_on_grid = jnp.interp(x_q, filter_wave, filter_trans, left=0.0, right=0.0)
        w_q = trans_on_grid * _filter_weight(x_q, convention)
    lo, hi = jnp.min(filter_wave), jnp.max(filter_wave)
    num = _support_trapezoid(L_q * w_q, x_q, lo, hi)
    den = _support_trapezoid(w_q, x_q, lo, hi)

    return num / jnp.maximum(den, representable_denominator(1e-30))


FILTER_COVERAGE_TOLERANCE: float = 1e-3
"""Largest tolerated out-of-axis fraction of a band's weighted transmission (#2620)."""


def filter_coverage_fraction(
    filter_waves,
    filter_trans,
    wave_obs,
    convention: FilterConvention = FilterConvention.BESSELL,
) -> np.ndarray:
    r"""Fraction of each band's weighted transmission outside the model's observed axis.

    For band :math:`k` with transmission :math:`T_k(\lambda)` and bandpass
    weight :math:`w(\lambda)` of the given convention, the returned fraction is

    .. math::

        f_k = \frac{\int_{\lambda \notin [\lambda_{\min}, \lambda_{\max}]}
              w(\lambda)\, T_k(\lambda)\, d\lambda}
              {\int w(\lambda)\, T_k(\lambda)\, d\lambda}

    where :math:`[\lambda_{\min}, \lambda_{\max}]` is the span of the model's
    observed-frame wavelength axis. The model SED is zero-filled outside that
    span, so :math:`f_k` is the share of the band's denominator that carries no
    model flux. The integral is taken over the filter's own support, with the
    axis edges inserted as quadrature nodes so each segment is wholly inside or
    wholly outside the axis.

    Parameters
    ----------
    filter_waves : sequence of array_like, each shape (n_wave_k,)
        Observed-frame filter wavelengths per band [Angstrom]. Zero entries are
        padding and are ignored.
    filter_trans : sequence of array_like, each shape (n_wave_k,)
        Dimensionless transmission per band, matching ``filter_waves``.
    wave_obs : array_like, shape (n_wave,)
        Model observed-frame wavelength axis at the redshift of interest [Angstrom].
    convention : FilterConvention, optional
        Bandpass weighting convention (default Bessell, ``w = 1/lambda``).

    Returns
    -------
    ndarray, shape (n_filters,)
        Out-of-axis fraction per band, in ``[0, 1]``. A band with no weight
        returns 0.

    Notes
    -----
    Pure NumPy, evaluated at build or fit setup, never inside a JIT. It does not
    alter the photometry integral: a band over :data:`FILTER_COVERAGE_TOLERANCE`
    is reported, not corrected.
    """
    axis = np.asarray(wave_obs, dtype=float)
    lo, hi = float(np.min(axis)), float(np.max(axis))
    fractions = np.zeros(len(filter_waves), dtype=float)
    for k, (fw_raw, ft_raw) in enumerate(zip(filter_waves, filter_trans, strict=True)):
        fw = np.asarray(fw_raw, dtype=float)
        ft = np.asarray(ft_raw, dtype=float)
        keep = fw > 0.0
        order = np.argsort(fw[keep])
        fw, ft = fw[keep][order], ft[keep][order]
        if fw.size < 2:
            continue
        inner = np.unique(np.concatenate([fw, [lo, hi]]))
        nodes = inner[(inner >= fw[0]) & (inner <= fw[-1])]
        weighted = np.interp(nodes, fw, ft) * filter_weight_np(nodes, convention)
        seg = 0.5 * (weighted[1:] + weighted[:-1]) * np.diff(nodes)
        mid = 0.5 * (nodes[1:] + nodes[:-1])
        beyond = (mid < lo) | (mid > hi)
        total = float(seg.sum())
        if total > 0.0:
            fractions[k] = float(seg[beyond].sum()) / total
    return fractions


def _ascending_padded_filter(
    fw_padded: jnp.ndarray, ft_padded: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Make a zero-padded filter row ascending with zero-width padding (#960, #2817).

    ``jnp.interp`` requires ascending nodes; zero-padding (:func:`pad_filters`)
    breaks that. Each pad node (``wave == 0``) is clamped to the last tabulated
    wavelength and given that node's transmission. The union-grid trapezoid
    then sees only zero-width intervals at the table edge, which add exactly
    nothing, so the padded row integrates the same as the unpadded table. (Pads
    at ``max + 1 + i`` with transmission 0 added a 1 A ramp past the edge.)

    A fully padded-out row (no tabulated node) becomes an ascending dummy grid
    with zero transmission, so its flux is 0.

    Returns
    -------
    fw_ascending : ndarray, shape (max_len,)
        Ascending observed-frame filter wavelengths [Angstrom].
    ft_ascending : ndarray, shape (max_len,)
        Transmission matching ``fw_ascending`` (dimensionless).
    """
    pos = jnp.arange(fw_padded.shape[0], dtype=fw_padded.dtype)
    # Last tabulated node: waves ascend and pads are 0. A max over the matching
    # nodes, not an argmax index, keeps this on plain reductions under vmap.
    edge_wave = jnp.max(fw_padded)
    edge_trans = jnp.max(jnp.where(fw_padded == edge_wave, ft_padded, 0.0))
    has_edge = edge_wave > 0.0
    valid = fw_padded > 0.0
    fw_asc = jnp.where(valid, fw_padded, jnp.where(has_edge, edge_wave, 1.0 + pos))
    ft_asc = jnp.where(valid, ft_padded, jnp.where(has_edge, edge_trans, 0.0))
    return fw_asc, ft_asc


def _support_trapezoid(y, x, lo, hi):
    """Trapezoid of ``y`` over ``x``, counting only intervals inside ``[lo, hi]``.

    An interval is inside when its midpoint is; the caller guarantees both edges
    are nodes of ``x``, so an interval is never split by the support edge.
    """
    mid = 0.5 * (x[1:] + x[:-1])
    inside = (mid >= lo) & (mid <= hi)
    panels = 0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])
    return jnp.sum(jnp.where(inside, panels, 0.0))


@functools.partial(jax.jit, static_argnames=("convention", "has_lyc_edge"))
def lnu_filter_integral(
    L_nu_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    filter_wave: jnp.ndarray,
    filter_trans: jnp.ndarray,
    redshift: float,
    convention: FilterConvention = FilterConvention.BESSELL,
    has_lyc_edge: bool = False,
) -> jnp.ndarray:
    r"""Filter-weighted rest-frame L_ν on the observed-frame filter grid.

    Returns the filter-weighted rest-frame specific luminosity, no
    cosmological dimming. The flux conversion is a separate step
    (compose with :func:`lnu_to_fnu` or :func:`compute_flux_density`).

    .. math::

        L_\nu^{\rm filter}
        = \frac{\int L_\nu(\lambda_{\rm rest}=\lambda_{\rm obs}/(1+z))
                T(\lambda_{\rm obs}) \, w(\lambda_{\rm obs}) \, d\lambda_{\rm obs}}
               {\int T(\lambda_{\rm obs}) \, w(\lambda_{\rm obs}) \, d\lambda_{\rm obs}}

    where the bandpass weight is :math:`w=1/\lambda` for the photon-counting
    ``BESSELL`` convention (default; matches DSPS/FSPS) and :math:`w=1/\lambda^2`
    for ``ENERGY`` (CIGALE). See :class:`FilterConvention`.

    Parameters
    ----------
    L_nu_rest : array, shape (n_wave,)
        Rest-frame specific luminosity [erg/s/Hz].
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength grid [Ångstrom].
    filter_wave : array, shape (n_filt,)
        Filter wavelength grid [Ångstrom], in observed frame.
    filter_trans : array, shape (n_filt,)
        Filter transmission (dimensionless, 0–1).
    redshift : float
        Source redshift z.
    convention : FilterConvention, optional
        Bandpass weight (``BESSELL`` 1/lambda default, ``ENERGY`` 1/lambda^2).

    Returns
    -------
    L_nu_filter : float
        Filter-weighted rest-frame L_ν [erg/s/Hz].

    Notes
    -----
    **JIT/grad-safe.** Pure ``jnp`` primitives; ``convention`` is static.

    The :math:`1/\lambda` weight is the photon-counting AB convention of FSPS
    (``getmags.f90``; Fukugita+1996 Eq. 7) and DSPS (Hearin+2023; Hogg+2002
    Eq. 5). Introduced in #398.e (per ADR-0016) to give components publishing
    ``_phot_lnu_precomp`` tensors a named function for "the L_ν step".

    **Quadrature (#960)**: the integral is evaluated on the sorted union of
    the SED nodes and the filter nodes, with the transmission interpolated,
    never the SED alone at the filter table's nodes. Instrument filter tables
    (25–70 Å spacing) under-sample MILES-resolution spectra; the pre-#960
    point-sampling quadrature biased SDSS-like bands by up to 3 %.

    has_lyc_edge : bool, optional
        Whether ``L_nu_rest`` carries a live per-node Lyman-continuum mask
        (``state.derived["lyc_transmission"]`` published, #2447/one Lyman
        edge), i.e. some photoionized nebular backend ran. When True, the
        observed-frame edge ``LYMAN_LIMIT_AA * (1+z)`` is threaded into
        :func:`_filter_integral_union`'s step-model quadrature; ``L_nu_rest``
        itself needs no decomposition (no unmasked SED, no separate
        transmission array) because the masking (and any dust/IGM factor)
        is already baked in per node. Static (affects which quadrature
        branch is traced). Default False (ordinary quadrature, exact when
        no live mask means ``L_nu_rest`` is smooth across the edge).

    See Also
    --------
    compute_flux_density : The full L→F conversion (composes this with
        :func:`lnu_to_fnu`).
    FilterConvention : The supported bandpass weights.
    """
    wave_obs = wave_rest * (1.0 + redshift)
    lyman_edge_obs_aa = LYMAN_LIMIT_AA * (1.0 + redshift) if has_lyc_edge else None
    return _filter_integral_union(
        L_nu_rest, wave_obs, filter_wave, filter_trans, convention, lyman_edge_obs_aa
    )


def lnu_filter_integral_batch(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    fw_padded: jnp.ndarray,
    ft_padded: jnp.ndarray,
    redshift,
    convention: FilterConvention = FilterConvention.BESSELL,
) -> jnp.ndarray:
    r"""Exact rest-frame filter-weighted L_ν of one SED through many filters.

    Vectorized, zero-padding-safe form of :func:`lnu_filter_integral` over a
    stack of filters ``(n_filters, max_len)``. This is the *exact* per-band
    projection, the identical union-grid quadrature the exact photometry
    path uses (:func:`compute_flux_density_batch`; see #960), minus the
    cosmological ``lnu_to_fnu`` step (the caller, ``predict_via_precomp``,
    applies cosmology after summing the L_ν families).

    Used by additive, unattenuated emitters under WavePrecomp, dust IR
    re-emission, radio, X-ray, AGN, so a band carrying both the stellar
    continuum and one of these emitters matches the exact path bit-for-bit
    (only the stellar × dust-attenuation term keeps the effective-wavelength
    LUT, which is where the speedup lives). Sampling such a component at a
    single filter pivot is *not* exact when the emitter has structure across
    the bandpass (PAH features, steep IR rise).

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame specific luminosity on ``wave_rest`` [erg/s/Hz].
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength grid [Ångström], ascending.
    fw_padded : array, shape (n_filters, max_len)
        Zero-padded observed-frame filter wavelengths [Ångström].
    ft_padded : array, shape (n_filters, max_len)
        Zero-padded filter transmission (dimensionless).
    redshift : float
        Source redshift.
    convention : FilterConvention, optional
        Bandpass weight (``BESSELL`` 1/λ default, matching the SSP Φ-tensor LUT).

    Returns
    -------
    L_nu_filter : array, shape (n_filters,)
        Filter-weighted rest-frame L_ν per band [erg/s/Hz].

    Notes
    -----
    **JIT/grad-safe.** Pure ``jnp`` primitives; ``convention`` static. Zero-pad
    entries contribute ~0 because ``trans=0`` there and real filters taper to 0
    at their edges (same assumption as :func:`_compute_flux_density_padded`).

    Each padded row is first rewritten to ascend
    (:func:`_ascending_padded_filter`): the union-grid quadrature
    interpolates on the filter nodes, and ``jnp.interp`` on a raw zero-pad
    tail (…, 4130, 0, 0) is unsorted-input garbage that silently zeroed
    every shorter-table band for heterogeneous filter sets, same-length
    sets never pad, which is why homogeneous fixtures missed it.
    """

    def _one(fw, ft):
        fw_safe, ft_safe = _ascending_padded_filter(fw, ft)
        return lnu_filter_integral(sed_rest, wave_rest, fw_safe, ft_safe, redshift, convention)

    return jax.vmap(_one)(fw_padded, ft_padded)


@functools.partial(jax.jit, static_argnames=("convention",))
def compute_flux_density(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    filter_wave: jnp.ndarray,
    filter_trans: jnp.ndarray,
    redshift: float,
    dl_cm: float,
    convention: FilterConvention = FilterConvention.BESSELL,
) -> float:
    r"""Compute observed flux density through a single photometric filter.

    Evaluates the rest-frame SED at observed wavelengths (redshifted), convolves
    with filter transmission, and integrates to produce a single observed flux
    density. Uses the standard flux-weighted approach: flux is the filter-weighted
    integral of redshifted SED, normalized by the filter response integral,
    and scaled by the luminosity distance and (1+z) redshift factor.

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame spectral luminosity density [erg/s/Hz] or [L☉/Hz] at
        rest-frame wavelengths.
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength grid [Ångstrom].
    filter_wave : array, shape (n_filt,)
        Filter wavelength grid [Ångstrom], already in observed frame
        (redshifted by the model).
    filter_trans : array, shape (n_filt,)
        Filter transmission at each wavelength (dimensionless, 0–1).
    redshift : float
        Source redshift z. Used to redshift rest-frame wavelengths and
        scale flux by (1+z) factor.
    dl_cm : float
        Luminosity distance [cm]. Typically from :func:`luminosity_distance`.
    convention : FilterConvention, optional
        Bandpass weight. ``BESSELL`` (default) is photon-counting
        (:math:`w=1/\\lambda`, matching DSPS/FSPS/sedpy); ``ENERGY`` is the
        flat-in-frequency mean (:math:`w=1/\\lambda^2`, matching CIGALE).

    Returns
    -------
    flux_density : float
        Observed flux density [erg/s/cm²/Hz] in the AB system.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives;
    ``convention`` is static. Safe to call inside :func:`jax.jit`.

    **Gradient-safe**: yes, differentiable w.r.t. all inputs except
    filter curves (considered fixed).

    **Filter convolution formula** (photon-counting, ``BESSELL`` default):

    .. math::

        f_\\nu^{\\rm obs} = \\frac{1+z}{4\\pi d_L^2} \\;
        \\frac{\\int L_\\nu(\\lambda_\\mathrm{rest}) T(\\lambda_\\mathrm{obs})
               \\, d\\lambda_\\mathrm{obs} / \\lambda_\\mathrm{obs}}
             {\\int T(\\lambda_\\mathrm{obs})
              \\, d\\lambda_\\mathrm{obs} / \\lambda_\\mathrm{obs}}

    where :math:`L_\\nu` is the rest-frame SED [erg/s/Hz],
    :math:`T(\\lambda_\\mathrm{obs})` is the filter transmission,
    :math:`z` is redshift, and :math:`d_L` is luminosity distance. The
    :math:`1/\\lambda` weight is the photon-counting AB convention of FSPS
    (``getmags.f90``; Fukugita+1996 Eq. 7) and DSPS (Hearin+2023; Hogg+2002
    Eq. 5). ``ENERGY`` replaces :math:`1/\\lambda` with :math:`1/\\lambda^2`
    (CIGALE; Boquien+2019). See :class:`FilterConvention`.

    **Quadrature (#960)**: the integral runs on the sorted union of the SED
    nodes and the filter nodes; the (smooth) transmission is interpolated onto
    that grid, never the (structured) SED onto the filter table alone. This
    stays accurate both when the filter table is coarse (SVO instrument
    tables vs MILES line structure) and when a narrow filter falls on a
    coarse region of the SED grid.

    **Edge handling**: :math:`L_\\nu = 0` outside the SED wavelength domain
    (set via ``left=0.0, right=0.0``).

    See Also
    --------
    FilterCurve : Photometric filter transmission curve.
    FilterConvention : The supported bandpass weights.
    pad_filters : Stack variable-length filter arrays.

    """
    # Composition of the two canonical operations (ADR-0016, 2026-05):
    #   1. ``lnu_filter_integral``, filter-weighted rest-frame L_ν
    #   2. ``lnu_to_fnu``, apply (1+z) / (4π d_L²) cosmological dimming
    L_nu_filter = lnu_filter_integral(
        sed_rest, wave_rest, filter_wave, filter_trans, redshift, convention=convention
    )
    return lnu_to_fnu(L_nu_filter, dl_cm, redshift)


def pad_filters(filter_waves: list, filter_trans: list):
    """Pad variable-length filter arrays to a common length and stack.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grids per filter (different lengths allowed, units
        [Angstrom]).
    filter_trans : list[ndarray]
        Transmission per filter (same lengths as ``filter_waves``, dimensionless).

    Returns
    -------
    fw_padded : ndarray, shape (n_filters, max_len)
        Zero-padded filter wavelengths [Angstrom].
    ft_padded : ndarray, shape (n_filters, max_len)
        Zero-padded filter transmissions (dimensionless).
    n_valid : ndarray, shape (n_filters,), dtype int
        Number of valid (non-padded) points per filter.

    Notes
    -----
    Not JIT-compatible (uses Python list operations and loops).

    """
    # Assemble in numpy and convert once: the per-filter scatter used to compile
    # one XLA kernel per distinct filter length. The arrays keep the dtypes the
    # old jnp path produced (canonical float and int for the active x64 mode).
    float_dtype = jax.dtypes.canonicalize_dtype(np.float64)
    int_dtype = jax.dtypes.canonicalize_dtype(np.int64)
    max_len = max(len(fw) for fw in filter_waves)
    fw_padded = np.zeros((len(filter_waves), max_len), dtype=float_dtype)
    ft_padded = np.zeros((len(filter_trans), max_len), dtype=float_dtype)
    for i, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        n = len(fw)
        fw_padded[i, :n] = np.asarray(fw)
        ft_padded[i, :n] = np.asarray(ft)
    n_valid = np.asarray([len(fw) for fw in filter_waves], dtype=int_dtype)
    return jnp.asarray(fw_padded), jnp.asarray(ft_padded), jnp.asarray(n_valid)


FILTER_COUNT_BUCKETS: tuple[int, ...] = (4, 6, 8, 10, 12, 16, 20)


def _next_filter_bucket(n: int, buckets: tuple[int, ...] = FILTER_COUNT_BUCKETS) -> int:
    """Smallest bucket ≥ n, or ``n`` itself if larger than all buckets."""
    for b in buckets:
        if b >= n:
            return b
    return n  # force-compile fallback


def pad_filters_to_bucket(filter_waves: list, filter_trans: list):
    """Pad both filter-length AND filter-count axes for compile reuse.

    Like :func:`pad_filters` but additionally pads the leading axis (number
    of filters) up to the next entry of :data:`FILTER_COUNT_BUCKETS`. The
    extra rows are all-zero, which contribute zero to ``compute_flux_density``
    (the integrand is ``trans × wave × SED`` and the denominator divides
    out, :func:`_filter_integral_union` floors the denominator).

    That floor makes the all-zero rows harmless in the **forward** pass only.
    This docstring previously cited the literal ``1e-30`` as the reason they are
    safe; in float32 it was the reason they were not (#1860). An all-zero row
    gives ``num == den == 0``, and the quotient's VJP carries ``-num/den**2``,
    which needs ``1/floor**2`` representable, a strictly stronger condition than
    the forward pass imposes. The floor is now sized with
    :func:`~tengri.utils.scale.representable_denominator`. Padding the count axis
    is therefore not free: it is safe *because* the floor is derivative-sized,
    and a future change to either must keep that pairing.

    This collapses what would otherwise be N separate XLA compiles (one per
    distinct (n_filters, max_len) shape encountered across a project) down
    to one compile per bucket. Two observations with 5 and 6 filters at the
    same max wavelength length share a compile by padding both to 6.

    For ``n_filters > max(FILTER_COUNT_BUCKETS)``, no padding is applied;
    each unique large count gets its own compile (the "force compilation"
    escape hatch).

    Returns
    -------
    fw_padded : ndarray, shape (n_padded, max_len)
    ft_padded : ndarray, shape (n_padded, max_len)
    n_valid : ndarray, shape (n_padded,), dtype int
        Number of valid samples per filter; 0 for padded-out filters.
    n_filters_real : int
        Original number of filters (use to slice the projected result).
    """
    fw_padded, ft_padded, n_valid = pad_filters(filter_waves, filter_trans)
    n_real = fw_padded.shape[0]
    n_padded = _next_filter_bucket(n_real)
    if n_padded == n_real:
        return fw_padded, ft_padded, n_valid, n_real
    pad_rows = n_padded - n_real
    max_len = fw_padded.shape[1]
    fw_padded = jnp.concatenate([fw_padded, jnp.zeros((pad_rows, max_len))], axis=0)
    ft_padded = jnp.concatenate([ft_padded, jnp.zeros((pad_rows, max_len))], axis=0)
    n_valid = jnp.concatenate([n_valid, jnp.zeros((pad_rows,), dtype=n_valid.dtype)], axis=0)
    return fw_padded, ft_padded, n_valid, n_real


def _compute_flux_density_padded(
    sed_rest,
    wave_rest,
    filter_wave_padded,
    filter_trans_padded,
    redshift,
    dl_cm,
    convention: FilterConvention = FilterConvention.BESSELL,
    has_lyc_edge: bool = False,
):
    """Compute flux density for a single padded filter.

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame SED [erg/s/Hz].
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    filter_wave_padded : array, shape (max_len,)
        Zero-padded filter wavelengths [Angstrom].
    filter_trans_padded : array, shape (max_len,)
        Zero-padded filter transmission (dimensionless).
    redshift : float
        Source redshift.
    dl_cm : float
        Luminosity distance [cm].
    convention : FilterConvention, optional
        Bandpass weight (``BESSELL`` 1/lambda default, ``ENERGY`` 1/lambda^2).

    Returns
    -------
    flux_density : float
        Observed flux density [erg/s/cm²/Hz].

    Notes
    -----
    Zero-pad entries are rewritten to an ascending tail above the filter
    (:func:`_ascending_padded_filter`) so the union-grid interpolation
    (#960) stays valid; their transmission is zero, so they contribute
    nothing. Private helper for compute_flux_density_batch.

    has_lyc_edge : bool, optional
        Whether ``sed_rest`` carries a live per-node Lyman-continuum mask
        (#2447/one Lyman edge); see :func:`lnu_filter_integral`. Static.

    """
    wave_obs = wave_rest * (1.0 + redshift)
    fw_safe, ft_safe = _ascending_padded_filter(filter_wave_padded, filter_trans_padded)
    lyman_edge_obs_aa = LYMAN_LIMIT_AA * (1.0 + redshift) if has_lyc_edge else None
    mean_lnu = _filter_integral_union(
        sed_rest, wave_obs, fw_safe, ft_safe, convention, lyman_edge_obs_aa
    )
    # Apply the (1+z)/(4π d_L²) dimming to the filter-integrated L_ν directly.
    # Extracting it as a standalone ``flux_scale = lnu_to_fnu(1.0, ...)`` is
    # ~1e-58 and underflows float32 to zero on its own (peak 1.0 absorbs none
    # of the -58 decades); applied to ``mean_lnu`` (~1e30), apply_log10_scale
    # folds the offset into that peak and the product stays in range. Identical
    # in float64 (#1206).
    return lnu_to_fnu(mean_lnu, dl_cm, redshift)


@functools.partial(jax.jit, static_argnames=("convention", "has_lyc_edge"))
def compute_flux_density_batch(
    sed_rest,
    wave_rest,
    fw_padded,
    ft_padded,
    redshift,
    dl_cm,
    convention: FilterConvention = FilterConvention.BESSELL,
    has_lyc_edge: bool = False,
):
    """Compute flux densities through all filters at once via vmap.

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame SED [erg/s/Hz].
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    fw_padded : array, shape (n_filters, max_len)
        Zero-padded filter wavelengths [Angstrom] (from ``pad_filters``).
    ft_padded : array, shape (n_filters, max_len)
        Zero-padded filter transmissions (dimensionless, from ``pad_filters``).
    redshift : float
        Source redshift.
    dl_cm : float
        Luminosity distance [cm].
    convention : FilterConvention, optional
        Bandpass weight (``BESSELL`` 1/lambda default, ``ENERGY`` 1/lambda^2).

    Returns
    -------
    ndarray, shape (n_filters,)
        Observed flux density per filter [erg/s/cm²/Hz].

    has_lyc_edge : bool, optional
        Whether ``sed_rest`` carries a live per-node Lyman-continuum mask
        (#2447/one Lyman edge); see :func:`lnu_filter_integral`. Static;
        shared (unmapped) across every filter in the vmap.

    Notes
    -----
    JIT-compatible: yes, vmapped over filters; ``convention`` and
    ``has_lyc_edge`` are static.
    Gradient-safe: yes.

    """
    return jax.vmap(
        functools.partial(
            _compute_flux_density_padded, convention=convention, has_lyc_edge=has_lyc_edge
        ),
        in_axes=(None, None, 0, 0, None, None),
    )(sed_rest, wave_rest, fw_padded, ft_padded, redshift, dl_cm)


def project_photometry(state, params, photometry, *, dl_cm=None) -> jnp.ndarray:
    r"""Project a panchromatic model SED onto photometric filter bands.

    Consolidates the photometric projection pipeline: extract rest-frame SED
    and redshift from ``state`` and ``params``, apply IGM attenuation when
    present, compute observed-frame flux densities through all filters via
    :func:`compute_flux_density_batch`. This is the single canonical exact
    projection path, the fast precompute LUT is available via
    :meth:`Observation.predict_via_precomp`.

    **This is the canonical exact (compositional) projection path for photometry.**
    Integrates ``state.sed_intrinsic`` through each filter without approximation.
    Clients that need post-build integration of arbitrary filters through the
    identical photometry path should call this kernel directly.

    Parameters
    ----------
    state : ForwardState
        Orchestrator output. Reads ``state.sed_intrinsic`` (rest-frame
        L_nu [erg/s/Hz]), ``state.wave`` (rest-frame Angstrom), and
        optionally ``state.derived["igm_transmission"]`` (dimensionless,
        same shape as ``state.wave``).
    params : Mapping[str, jnp.ndarray]
        Parameter dict. Reads ``params["redshift"]`` for cosmology and
        redshift. If redshift is absent, defaults to 0.0.
    photometry : Photometry
        Photometry configuration. Reads ``n_filters`` (count of real filters),
        ``_fw_padded`` and ``_ft_padded`` (zero-padded filter wavelengths and
        transmissions), and ``convention`` (FilterConvention for bandpass weight).
    dl_cm : float or jnp.ndarray, optional
        Luminosity distance [cm]. If ``None``, derived from
        ``params["redshift"]`` via :func:`tengri.cosmology.luminosity_distance`.

    Returns
    -------
    ndarray, shape (n_filters,)
        Observed flux density per filter [erg/s/cm²/Hz].

    Notes
    -----
    **JIT-compatible**: yes, delegates to vmapped
    :func:`compute_flux_density_batch`. ``redshift`` and ``dl_cm``
    must be array scalars or scalars; ``photometry`` properties are
    pytree leaves and statically known. IGM multiplication is traceable.

    **Gradient-safe**: yes, all operations are JAX operations.

    **Composition pattern**: This kernel applies IGM attenuation internally
    when ``state.derived["igm_transmission"]`` is present. The transmission
    curve captures the sharp Lyman break across broad photometric bands at
    high redshift (#932).

    See Also
    --------
    compute_flux_density_batch : Low-level batched filter convolution.
    project_spectrum : Spectroscopy projection twin (IGM composed by callers).
    """
    from tengri.cosmology import luminosity_distance

    z = jnp.asarray(require_redshift(params, "observation.photometry.project_photometry"))
    if dl_cm is None:
        dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
    else:
        dl_cm = jnp.asarray(dl_cm)

    sed_rest = state.sed_intrinsic
    wave_rest = state.wave

    # IGM attenuation is an observed-frame transmission the IGM component
    # publishes on the rest grid (``T`` evaluated at ``wave_obs =
    # wave*(1+z)``). Multiplying here, before redshifting in
    # ``compute_flux_density_batch``, applies the full transmission curve
    # (not a single per-band effective-wavelength factor) and captures the
    # sharp Lyman break across broad bands at high redshift (#932).
    # ``T`` shares the rest grid with ``sed_rest``; the key is absent
    # (structural no-op) when IGM is disabled, so low-z / IGM-off models are
    # bit-unchanged.
    igm_trans = state.derived.get("igm_transmission", None) if state.derived is not None else None
    if igm_trans is not None:
        sed_rest = sed_rest * igm_trans
    # Milky Way foreground screen (#3140), published on the same rest grid by
    # ``SEDModel.predict_state``. Applied once here, so the photometric block of
    # ``Observation.predict`` and every caller of this kernel see one screen.
    foreground_trans = (
        state.derived.get("foreground_transmission", None) if state.derived is not None else None
    )
    if foreground_trans is not None:
        sed_rest = sed_rest * foreground_trans

    # Exact Lyman-continuum edge on THIS projection's own (finer) union grid
    # (#2447, one Lyman edge): ``sed_rest`` already carries the physical
    # masking (and any dust/IGM multiplicative factor) PER NODE, whatever
    # component produced it -- there is nothing to reconstruct, only the
    # edge POSITION is needed. ``lyc_transmission`` is published whenever a
    # photoionized nebular backend ran (component.py); absent (e.g.
    # ``neb={'type': 'none'}``) is a structural no-op: ``sed_rest`` is then
    # smooth across the edge and ordinary quadrature is exact.
    has_lyc_edge = state.derived is not None and state.derived.get("lyc_transmission") is not None

    n_real = photometry.n_filters
    fluxes = compute_flux_density_batch(
        sed_rest,
        wave_rest,
        photometry._fw_padded,
        photometry._ft_padded,
        z,
        dl_cm,
        convention=photometry.convention,
        has_lyc_edge=has_lyc_edge,
    )
    return fluxes[:n_real]


def compute_photometry(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    filters: list,
    redshift: float,
    dl_cm: float,
    convention: FilterConvention = FilterConvention.BESSELL,
) -> jnp.ndarray:
    """Compute photometry through multiple filters.

    Convenience wrapper that calls :func:`compute_flux_density` for each filter
    in sequence and returns stacked flux densities.

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame SED [erg/s/Hz].
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    filters : list of FilterCurve
        Filter transmission curves to convolve.
    redshift : float
        Source redshift [dimensionless].
    dl_cm : float
        Luminosity distance [cm].

    Returns
    -------
    ndarray, shape (n_filters,)
        Observed flux density per filter [erg/s/cm²/Hz].

    Notes
    -----
    **JIT-compatible**: no, uses Python list comprehension and loops.
    For JIT-compiled photometry over many filters, use
    :func:`compute_flux_density_batch` with padded filter arrays.

    **Gradient-safe**: yes, each call to :func:`compute_flux_density`
    is differentiable w.r.t. ``sed_rest``.

    See Also
    --------
    compute_flux_density : Single filter convolution (JIT-compatible).
    compute_flux_density_batch : Vectorized convolution via vmap.
    """
    fluxes = []
    for filt in filters:
        f = compute_flux_density(
            sed_rest, wave_rest, filt.wave, filt.trans, redshift, dl_cm, convention=convention
        )
        fluxes.append(f)
    return jnp.array(fluxes)


@jax.jit
def ab_mag_from_flux(flux_cgs: jnp.ndarray) -> jnp.ndarray:
    """Convert flux density to AB magnitude.

    Parameters
    ----------
    flux_cgs : array, shape (n_band,)
        Flux density [erg/s/cm²/Hz].

    Returns
    -------
    ndarray, shape (n_band,)
        AB magnitude (dimensionless).

    Notes
    -----
    JIT-compatible: yes. Gradient-safe: yes.
    Delegates to :func:`tengri.utils.magnitudes.fnu_to_ab_mag`.

    """
    return fnu_to_ab_mag(flux_cgs)
