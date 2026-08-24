# SPDX-License-Identifier: BSD-3-Clause
"""Spectral index definitions and observed data for direct fitting.

Supports two index types:

- **EW** (equivalent width): measures absorption or emission strength
  relative to a pseudo-continuum defined by sideband windows.
- **break** (flux ratio): ratio of mean fluxes in two continuum windows
  (e.g., Dn4000).

Index values are measured on rest-frame spectra. The forward model generates
a spectrum covering the required wavelength range, measures indices on it,
and compares against observed values via a chi2 likelihood term.

Usage::

    from tengri.observation.spectral_indices import SpectralIndexData

    indices = SpectralIndexData.from_names(
        names=["Dn4000", "HdA"],
        values=[1.8, -1.2],
        errors=[0.05, 0.3],
    )
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp

# ── Index definition ──────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SpectralIndexDef:
    """Definition of a single spectral index.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. ``"Dn4000"``).
    index_type : str
        ``"EW"`` (equivalent width), ``"break"`` (flux ratio), or
        ``"slope"`` (power-law spectral slope β over a window, e.g. the
        UV continuum slope, Calzetti+1994).
    continuum : tuple of tuple
        Continuum/sideband windows as ``((lo1, hi1), (lo2, hi2), ...)``.
        Rest-frame wavelengths in Angstrom.
        For EW indices: blue and red pseudo-continuum sidebands.
        For break indices: exactly two windows (numerator, denominator).
        For slope indices: unused (pass ``()``); the fit range is ``feature``.
    feature : tuple of float or None
        Feature window ``(lo, hi)`` in Angstrom. Required for EW indices
        (the absorption feature) and slope indices (the fit range), None
        for break indices.
    units : str
        ``"AA"`` for Angstrom (default) or ``"mag"`` for magnitude.

    Examples
    --------
    >>> from tengri import SpectralIndexDef
    >>> dn4000 = SpectralIndexDef(
    ...     name="Dn4000",
    ...     index_type="break",
    ...     continuum=((3850.0, 3950.0), (4000.0, 4100.0)),
    ... )
    >>> dn4000.wave_min, dn4000.wave_max
    (3850.0, 4100.0)
    >>> hda = SpectralIndexDef(
    ...     name="HdA",
    ...     index_type="EW",
    ...     continuum=((4041.6, 4079.75), (4128.5, 4161.0)),
    ...     feature=(4083.5, 4122.25),
    ... )
    >>> hda.index_type
    'EW'
    """

    name: str
    index_type: str
    continuum: tuple[tuple[float, float], ...]
    feature: tuple[float, float] | None = None
    units: str = "AA"

    def __post_init__(self):
        if self.index_type not in ("EW", "break", "slope"):
            raise ValueError(
                f"index_type must be 'EW', 'break', or 'slope', got {self.index_type!r}"
            )
        if self.index_type == "EW" and self.feature is None:
            raise ValueError("EW indices require a feature window.")
        if self.index_type == "break" and len(self.continuum) != 2:
            raise ValueError("Break indices require exactly 2 continuum windows.")
        if self.index_type == "slope" and self.feature is None:
            raise ValueError("Slope indices require a feature window (the fit range).")

    @property
    def wave_min(self) -> float:
        """Minimum wavelength needed to measure this index.

        Returns
        -------
        float
            Minimum wavelength [Angstrom] across all continuum and feature
            windows defined in this index.

        Notes
        -----
        Computed as the minimum of all window edges. Useful for determining
        the minimum wavelength coverage needed in the forward model spectrum.

        """
        vals = [w for pair in self.continuum for w in pair]
        if self.feature is not None:
            vals.extend(self.feature)
        return min(vals)

    @property
    def wave_max(self) -> float:
        """Maximum wavelength needed to measure this index.

        Returns
        -------
        float
            Maximum wavelength [Angstrom] across all continuum and feature
            windows defined in this index.

        Notes
        -----
        Computed as the maximum of all window edges. Useful for determining
        the maximum wavelength coverage needed in the forward model spectrum.

        """
        vals = [w for pair in self.continuum for w in pair]
        if self.feature is not None:
            vals.extend(self.feature)
        return max(vals)


# ── Standard index catalog ────────────────────────────────────────

#: Catalog of the 13 single-passband spectral indices tengri ships, keyed by
#: index name. Three kinds are represented: ``break`` ratios (``Dn4000``,
#: ``D4000``), Lick-style equivalent widths (``HdA``, ``HdF``, ``HgA``,
#: ``HgF``, ``Hbeta``, ``Mgb``, ``Fe4383``, ``Fe5270``, ``Fe5335``,
#: ``Ca4227``), and one continuum ``slope`` (``uv_slope_beta``). Values are
#: :class:`SpectralIndexDef` records carrying the passband definitions in
#: rest-frame vacuum Angstrom. Pass a key to
#: :func:`tengri.measure.spectral_index` or :func:`tengri.measure_index_jax`.
STANDARD_INDICES: dict[str, SpectralIndexDef] = {
    "Dn4000": SpectralIndexDef(
        name="Dn4000",
        index_type="break",
        continuum=((3850.0, 3950.0), (4000.0, 4100.0)),
    ),
    "D4000": SpectralIndexDef(
        name="D4000",
        index_type="break",
        continuum=((3750.0, 3950.0), (4050.0, 4250.0)),
    ),
    "HdA": SpectralIndexDef(
        name="HdA",
        index_type="EW",
        continuum=((4041.60, 4079.75), (4128.50, 4161.00)),
        feature=(4083.50, 4122.25),
    ),
    "HdF": SpectralIndexDef(
        name="HdF",
        index_type="EW",
        continuum=((4057.25, 4088.50), (4114.75, 4137.25)),
        feature=(4091.00, 4112.25),
    ),
    "HgA": SpectralIndexDef(
        name="HgA",
        index_type="EW",
        continuum=((4283.50, 4319.75), (4367.25, 4419.75)),
        feature=(4319.75, 4363.50),
    ),
    "HgF": SpectralIndexDef(
        name="HgF",
        index_type="EW",
        continuum=((4283.50, 4319.75), (4354.75, 4384.75)),
        feature=(4331.25, 4352.25),
    ),
    "Mgb": SpectralIndexDef(
        name="Mgb",
        index_type="EW",
        continuum=((5142.63, 5161.38), (5191.38, 5206.38)),
        feature=(5160.13, 5192.63),
    ),
    "Fe5270": SpectralIndexDef(
        name="Fe5270",
        index_type="EW",
        continuum=((5233.15, 5248.15), (5285.65, 5318.15)),
        feature=(5245.65, 5285.65),
    ),
    "Fe5335": SpectralIndexDef(
        name="Fe5335",
        index_type="EW",
        continuum=((5304.63, 5315.88), (5353.38, 5363.38)),
        feature=(5312.13, 5352.13),
    ),
    "Hbeta": SpectralIndexDef(
        name="Hbeta",
        index_type="EW",
        continuum=((4827.88, 4847.88), (4876.63, 4891.63)),
        feature=(4847.88, 4876.63),
    ),
    "Fe4383": SpectralIndexDef(
        name="Fe4383",
        index_type="EW",
        continuum=((4359.13, 4370.38), (4442.88, 4455.38)),
        feature=(4369.13, 4420.38),
    ),
    "Ca4227": SpectralIndexDef(
        name="Ca4227",
        index_type="EW",
        continuum=((4211.00, 4219.75), (4241.00, 4251.00)),
        feature=(4222.25, 4234.75),
    ),
    # UV continuum slope β (Calzetti+1994), f_λ ∝ λ^β over 1250–2600 Å.
    "uv_slope_beta": SpectralIndexDef(
        name="uv_slope_beta",
        index_type="slope",
        continuum=(),
        feature=(1250.0, 2600.0),
    ),
}


# ── Composite indices ─────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class CompositeIndexDef:
    """A spectral index that is a function of atomic indices.

    Composite indices break the age-metallicity degeneracy by combining
    multiple Lick measurements (Worthey & Ottaviani 1997; Thomas, Maraston
    & Bender 2003). Standard examples include ``[MgFe]'`` (sensitive to
    [Fe/H] but not [alpha/Fe]) and ``<Fe>`` (the mean of Fe5270 and Fe5335).

    Parameters
    ----------
    name : str
        Human-readable name, e.g. ``"[MgFe]'"`` or ``"<Fe>"``.
    components : tuple of SpectralIndexDef
        The atomic indices this composite is built from.
    combiner : Callable
        Function that takes one argument per atomic index (in the same
        order as ``components``) and returns the composite value. Must be
        JAX-compatible (operate on ``jnp`` arrays) so the composite stays
        differentiable through ``measure_index_jax``.
    units : str
        Units of the composite value, for display only. Default ``"AA"``.

    Examples
    --------
    The Thomas+2003 [MgFe]' index::

        from tengri import STANDARD_INDICES, CompositeIndexDef

        mgfe_prime = CompositeIndexDef(
            name="[MgFe]'",
            components=(
                STANDARD_INDICES["Mgb"],
                STANDARD_INDICES["Fe5270"],
                STANDARD_INDICES["Fe5335"],
            ),
            combiner=lambda mgb, fe1, fe2: jnp.sqrt(
                jnp.maximum(mgb * (0.72 * fe1 + 0.28 * fe2), 0.0)
            ),
        )
    """

    name: str
    components: tuple[SpectralIndexDef, ...]
    combiner: Callable[..., jnp.ndarray]
    units: str = "AA"

    @property
    def wave_min(self) -> float:
        return min(c.wave_min for c in self.components)

    @property
    def wave_max(self) -> float:
        return max(c.wave_max for c in self.components)


#: Catalog of the four composite indices tengri ships, keyed by index name.
#: Each combines two or three entries of :data:`STANDARD_INDICES` into the
#: standard abundance- and age-tracer combinations: ``[MgFe]'`` and ``<Fe>``
#: for metallicity, ``HdA+HgA`` and ``HdF+HgF`` for the Balmer age
#: diagnostics. Values are :class:`CompositeIndexDef` records.
STANDARD_COMPOSITE_INDICES: dict[str, CompositeIndexDef] = {
    # Thomas, Maraston & Bender 2003, MNRAS 339, 897, [MgFe]' is the
    # canonical [alpha/Fe]-insensitive [Fe/H] tracer.
    "[MgFe]'": CompositeIndexDef(
        name="[MgFe]'",
        components=(
            STANDARD_INDICES["Mgb"],
            STANDARD_INDICES["Fe5270"],
            STANDARD_INDICES["Fe5335"],
        ),
        combiner=lambda mgb, fe1, fe2: jnp.sqrt(jnp.maximum(mgb * (0.72 * fe1 + 0.28 * fe2), 0.0)),
    ),
    # Mean iron (Faber+1985 / Worthey 1994).
    "<Fe>": CompositeIndexDef(
        name="<Fe>",
        components=(STANDARD_INDICES["Fe5270"], STANDARD_INDICES["Fe5335"]),
        combiner=lambda fe1, fe2: 0.5 * (fe1 + fe2),
    ),
    # Higher-order Balmer sums used as age indicators that are insensitive
    # to abundance ratios (Worthey & Ottaviani 1997).
    "HdA+HgA": CompositeIndexDef(
        name="HdA+HgA",
        components=(STANDARD_INDICES["HdA"], STANDARD_INDICES["HgA"]),
        combiner=lambda hda, hga: hda + hga,
    ),
    "HdF+HgF": CompositeIndexDef(
        name="HdF+HgF",
        components=(STANDARD_INDICES["HdF"], STANDARD_INDICES["HgF"]),
        combiner=lambda hdf, hgf: hdf + hgf,
    ),
}


# ── JAX-compatible index measurement ──────────────────────────────


def measure_index_jax(
    wave_rest: jnp.ndarray,
    flux: jnp.ndarray,
    index_def: SpectralIndexDef | CompositeIndexDef,
) -> jnp.ndarray:
    """Measure a spectral index on a rest-frame spectrum.

    Parameters
    ----------
    wave_rest : ndarray, shape (n_pix,)
        Rest-frame wavelengths [Angstrom]. Must cover all windows defined
        in ``index_def``.
    flux : ndarray, shape (n_pix,)
        Flux density (any consistent units, only ratios matter).
    index_def : SpectralIndexDef or CompositeIndexDef
        Atomic index (EW or break) or a composite that combines several
        atomic measurements via a user-provided function.

    Returns
    -------
    ndarray, shape ()
        Measured index value (scalar). Units per ``index_def.units``:
        [Angstrom] for EW; [dimensionless] for break and slope (beta) indices.

    Notes
    -----
    **JIT-compatible**: yes, uses soft sigmoid edges for differentiability
    rather than hard window boundaries.

    **Gradient-safe**: yes, fully differentiable w.r.t. flux.

    """
    if isinstance(index_def, CompositeIndexDef):
        atomic_values = tuple(measure_index_jax(wave_rest, flux, c) for c in index_def.components)
        return index_def.combiner(*atomic_values)
    if index_def.index_type == "break":
        return _measure_break(wave_rest, flux, index_def)
    elif index_def.index_type == "slope":
        return _measure_slope(wave_rest, flux, index_def)
    else:
        return _measure_ew(wave_rest, flux, index_def)


def _window_mean_flux(
    wave: jnp.ndarray, flux: jnp.ndarray, lo: float, hi: float, edge_width: float = 1.0
) -> jnp.ndarray:
    """Mean flux in a wavelength window, using soft sigmoid edges for differentiability."""
    w_lo = jax.nn.sigmoid((wave - lo) / edge_width)
    w_hi = jax.nn.sigmoid((hi - wave) / edge_width)
    weights = w_lo * w_hi
    n = jnp.maximum(jnp.sum(weights), 1e-10)
    return jnp.sum(flux * weights) / n


# ── Single-sourced index arithmetic ───────────────────────────────
#
# A break or EW index is a small piece of arithmetic on *window mean fluxes*.
# There are two ways to get those means, integrate the reconstructed SED
# (the exact path) or read the precomputed window LUT (the FeaturePrecomp fast
# path), but the arithmetic that turns means into an index value is identical.
# These two helpers ARE that arithmetic, written once. Both the exact
# ``_measure_*`` functions and the LUT ``measure_indices_from_windows`` call
# them, so there is no hand-synced "mirror": one measurement, two window-mean
# sources. Add a new index kind here and both paths get it.


def _break_from_means(f_blue: jnp.ndarray, f_red: jnp.ndarray) -> jnp.ndarray:
    """Break ratio (red continuum mean flux / blue continuum mean flux).

    The single break primitive, shared by :func:`_measure_break` (means from the
    reconstructed SED) and :func:`measure_indices_from_windows` (means from the
    window LUT).
    """
    return f_red / jnp.maximum(f_blue, 1e-30)


def _ew_from_means(
    cont_means: jnp.ndarray, feat_flux: jnp.ndarray, feat_width: float, units: str
) -> jnp.ndarray:
    """Equivalent width from continuum + feature mean fluxes.

    Averages the continuum-window means, forms the continuum-to-feature ratio
    scaled by the feature width, and (for ``units == "mag"``) converts to a
    magnitude index. The single EW primitive, shared by :func:`_measure_ew`
    and :func:`measure_indices_from_windows` so the EW arithmetic (and its
    ``mag`` variant) is single-sourced.
    """
    cont_flux = jnp.mean(jnp.asarray(cont_means))
    ew = feat_width * (cont_flux - feat_flux) / jnp.maximum(cont_flux, 1e-30)
    if units == "mag":
        return -2.5 * jnp.log10(jnp.maximum(1.0 - ew / feat_width, 1e-30))
    return ew


def _measure_break(wave: jnp.ndarray, flux: jnp.ndarray, idx: SpectralIndexDef) -> jnp.ndarray:
    """Compute spectral break ratio (red window mean flux / blue window mean flux)."""
    blue_lo, blue_hi = idx.continuum[0]
    red_lo, red_hi = idx.continuum[1]
    f_blue = _window_mean_flux(wave, flux, blue_lo, blue_hi)
    f_red = _window_mean_flux(wave, flux, red_lo, red_hi)
    return _break_from_means(f_blue, f_red)


def _measure_ew(wave: jnp.ndarray, flux: jnp.ndarray, idx: SpectralIndexDef) -> jnp.ndarray:
    """Compute equivalent width from continuum-to-feature flux ratio and feature window width."""
    cont_fluxes = [_window_mean_flux(wave, flux, lo, hi) for lo, hi in idx.continuum]
    feat_lo, feat_hi = idx.feature
    feat_flux = _window_mean_flux(wave, flux, feat_lo, feat_hi)
    feat_width = feat_hi - feat_lo
    return _ew_from_means(cont_fluxes, feat_flux, feat_width, idx.units)


def _measure_slope(wave: jnp.ndarray, flux: jnp.ndarray, idx: SpectralIndexDef) -> jnp.ndarray:
    """Power-law spectral slope β over the feature window (e.g. UV slope).

    Fits ``f_λ ∝ λ^β``. With the SED in f_ν units, β = d ln(f_ν)/d ln(λ) − 2
    (Calzetti+1994 convention, matches
    :func:`tengri.utils.sed_quantities.compute_uv_slope_beta`). Uses a soft
    sigmoid window (differentiable) for the weights, then analytic weighted
    least squares in log-log space.
    """
    lo, hi = idx.feature
    edge_width = 1.0
    w = jax.nn.sigmoid((wave - lo) / edge_width) * jax.nn.sigmoid((hi - wave) / edge_width)
    log_wave = jnp.log(jnp.maximum(wave, 1.0))
    log_fnu = jnp.log(jnp.maximum(flux, 1e-50))

    sw = jnp.sum(w)
    sx = jnp.sum(w * log_wave)
    sy = jnp.sum(w * log_fnu)
    sxx = jnp.sum(w * log_wave**2)
    sxy = jnp.sum(w * log_wave * log_fnu)
    denom = sxx - sx**2 / jnp.maximum(sw, 1e-30)
    slope_fnu = (sxy - sx * sy / jnp.maximum(sw, 1e-30)) / jnp.maximum(denom, 1e-30)
    return slope_fnu - 2.0


# ── FeaturePrecomp: window-integral LUT for break / EW indices ─────
#
# The WavePrecomp-analog for spectral indices. A window mean flux is a linear
# functional of the SED, and the SED is a weight-sum of SSP spectra, so a break
# or EW index measured on ``SED = Σ_ij w_ij · SSP_ij`` can be evaluated from
# per-window SSP integrals precomputed once at build time, a cheap SFH-weighted
# sum instead of a full-resolution ``measure_index_jax`` on the reconstructed
# SED. Parity is EXACT (up to floating point) when the SED carries no dust,
# because the window mean commutes with the SFH weight sum:
#
#     <SED>_win = Σ_λ (Σ_ij w_ij SSP_ij(λ)) W(λ) / Σ_λ W(λ)
#               = Σ_ij w_ij · [Σ_λ SSP_ij(λ) W(λ)] / Σ_λ W(λ)
#               = Σ_ij w_ij · ssp_window_integral_ij / window_norm .
#
# Slope indices are NOT expressible this way (they need the SED shape within the
# window, not one integral) and are excluded, callers fall back to the exact
# ``measure_index_jax`` path for them.


@dataclasses.dataclass(frozen=True)
class IndexWindowPrecomputation:
    """Precomputed SSP window integrals for break / EW spectral indices.

    Built once at model construction (``approx=FeaturePrecomp()``) from the SSP
    grid and the configured index windows. Consumed per evaluation by
    :func:`measure_indices_from_windows` after the stellar component SFH-weights
    ``window_integrals`` into per-window mean fluxes.

    Attributes
    ----------
    window_integrals : ndarray, shape (n_met, n_age, n_window)
        :math:`\\sum_\\lambda \\mathrm{SSP}_{ij}(\\lambda)\\,W_w(\\lambda)`, the
        soft-window integral of each SSP spectrum, in the SSP flux units
        [erg/s/Hz/Msun · Å] summed on the SSP wave grid.
    window_norms : ndarray, shape (n_window,)
        :math:`\\sum_\\lambda W_w(\\lambda)`, window normalization, so
        ``mean = integral / norm`` matches :func:`_window_mean_flux`.
    window_centers : ndarray, shape (n_window,)
        Window mid-wavelength ``0.5*(lo+hi)`` [Å], for per-window dust.
    index_slots : tuple
        Per index, ``(kind, payload, meta)`` describing which window slots the
        index consumes and how to combine them, see
        :func:`measure_indices_from_windows`. ``kind`` is ``"break"``,
        ``"EW"``, or ``"slope"`` (the last carries ``payload=None`` and is a
        sentinel that the caller must measure exactly).
    names : tuple of str
        Index names in order, for diagnostics / alignment with observed data.
    """

    window_integrals: jnp.ndarray
    window_norms: jnp.ndarray
    window_centers: jnp.ndarray
    index_slots: tuple
    names: tuple

    @property
    def has_slope(self) -> bool:
        """Whether any configured index is a slope (needs the exact path)."""
        return any(kind == "slope" for kind, _, _ in self.index_slots)


def _round_window(lo: float, hi: float) -> tuple[float, float]:
    return (round(float(lo), 4), round(float(hi), 4))


def soft_window_ssp_integral(ssp_wave, ssp_flux, lo, hi, edge_width: float = 1.0):
    """Soft top-hat window integral of every SSP spectrum over ``[lo, hi]``.

    The shared window-integral primitive for both the spectral-index LUT
    (:func:`precompute_index_windows`) and the emission-line-flux LUT
    (:func:`tengri.observation.line_measurement.precompute_line_windows`), so the
    two precomputes integrate the SSP grid identically. Uses the same sigmoid
    edges as :func:`_window_mean_flux` (``mean = integral / norm``).

    Parameters
    ----------
    ssp_wave : ndarray, shape (n_wave,)
        SSP wavelength grid [Å].
    ssp_flux : ndarray, shape (n_met, n_age, n_wave)
        SSP spectra [erg/s/Hz/Msun].
    lo, hi : float
        Window bounds [Å].
    edge_width : float, default 1.0
        Sigmoid edge width [Å].

    Returns
    -------
    integral : ndarray, shape (n_met, n_age)
        :math:`\\sum_\\lambda \\mathrm{SSP}(\\lambda)\\,W(\\lambda)`.
    norm : ndarray, shape ()
        :math:`\\sum_\\lambda W(\\lambda)`.
    """
    w = jax.nn.sigmoid((ssp_wave - lo) / edge_width) * jax.nn.sigmoid((hi - ssp_wave) / edge_width)
    integral = jnp.tensordot(ssp_flux, w, axes=([2], [0]))  # (n_met, n_age)
    return integral, jnp.maximum(jnp.sum(w), 1e-10)


def precompute_index_windows(
    ssp_wave: jnp.ndarray,
    ssp_flux: jnp.ndarray,
    index_defs,
    edge_width: float = 1.0,
) -> IndexWindowPrecomputation:
    """Precompute SSP window integrals for break / EW indices.

    Parameters
    ----------
    ssp_wave : ndarray, shape (n_wave,)
        Rest-frame SSP wavelength grid [Å].
    ssp_flux : ndarray, shape (n_met, n_age, n_wave)
        SSP spectra [erg/s/Hz/Msun].
    index_defs : sequence of SpectralIndexDef
        The indices to precompute. Slope indices are recorded as sentinels
        (no window integrals) so the caller falls back to the exact path.
    edge_width : float, default 1.0
        Sigmoid edge width [Å], MUST match :func:`_window_mean_flux` so the
        LUT and exact paths agree.

    Returns
    -------
    IndexWindowPrecomputation
        Window integrals, norms, centers, and per-index window-slot recipe.

    Notes
    -----
    **JIT-compatible**: yes (built once at construction; pure ``jnp``). Windows
    shared across indices (e.g. two indices sharing a continuum band) are
    deduplicated so each unique window is integrated once.
    """
    ssp_wave = jnp.asarray(ssp_wave)
    ssp_flux = jnp.asarray(ssp_flux)

    unique: dict[tuple[float, float], int] = {}
    integrals: list[jnp.ndarray] = []
    norms: list[jnp.ndarray] = []
    centers: list[float] = []

    def _slot(lo, hi) -> int:
        key = _round_window(lo, hi)
        if key in unique:
            return unique[key]
        integral, norm = soft_window_ssp_integral(ssp_wave, ssp_flux, lo, hi, edge_width)
        integrals.append(integral)  # (n_met, n_age)
        norms.append(norm)
        centers.append(0.5 * (float(lo) + float(hi)))
        unique[key] = len(integrals) - 1
        return unique[key]

    slots = []
    names = []
    for idx in index_defs:
        names.append(idx.name)
        if idx.index_type == "break":
            b = _slot(*idx.continuum[0])
            r = _slot(*idx.continuum[1])
            slots.append(("break", (b, r), None))
        elif idx.index_type == "EW":
            cont = tuple(_slot(lo, hi) for lo, hi in idx.continuum)
            feat = _slot(*idx.feature)
            feat_width = idx.feature[1] - idx.feature[0]
            slots.append(("EW", (cont, feat), (feat_width, idx.units)))
        else:  # slope, not expressible from a single window integral
            slots.append(("slope", None, None))

    if integrals:
        window_integrals = jnp.stack(integrals, axis=-1)  # (n_met, n_age, n_window)
        window_norms = jnp.stack(norms)
    else:
        # All indices are slope (no break/EW windows): empty LUT, exact fallback.
        n_met, n_age = ssp_flux.shape[0], ssp_flux.shape[1]
        window_integrals = jnp.zeros((n_met, n_age, 0))
        window_norms = jnp.zeros((0,))
    return IndexWindowPrecomputation(
        window_integrals=window_integrals,
        window_norms=window_norms,
        window_centers=jnp.asarray(centers),
        index_slots=tuple(slots),
        names=tuple(names),
    )


def measure_indices_from_windows(
    window_means: jnp.ndarray, precomp: IndexWindowPrecomputation
) -> jnp.ndarray:
    """Evaluate break / EW indices from precomputed per-window mean fluxes.

    Parameters
    ----------
    window_means : ndarray, shape (n_window,)
        SFH-weighted (and optionally dust-attenuated) mean flux in each unique
        window: ``Σ_ij w_ij window_integrals_ijw / window_norm_w``.
    precomp : IndexWindowPrecomputation
        The build-time window recipe.

    Returns
    -------
    ndarray, shape (n_index,)
        Index values in ``precomp.names`` order. Slope slots return ``nan``;
        the caller must fill them from the exact path.

    Notes
    -----
    **JIT-compatible**: yes. Calls the same :func:`_break_from_means` /
    :func:`_ew_from_means` primitives as the exact path, only the window-mean
    source differs (precomputed LUT here vs integrated SED there), so there is no
    duplicated index arithmetic to keep in sync.
    """
    out = []
    for kind, payload, meta in precomp.index_slots:
        if kind == "break":
            b, r = payload
            out.append(_break_from_means(window_means[b], window_means[r]))
        elif kind == "EW":
            cont_slots, feat = payload
            feat_width, units = meta
            cont_means = [window_means[c] for c in cont_slots]
            out.append(_ew_from_means(cont_means, window_means[feat], feat_width, units))
        else:  # slope
            out.append(jnp.asarray(jnp.nan))
    return jnp.stack(out)


def measure_indices_from_window_lut(
    joint_weights: jnp.ndarray,
    scale: jnp.ndarray,
    transmission_at_centers: jnp.ndarray,
    precomp: IndexWindowPrecomputation,
) -> jnp.ndarray:
    """Measure break/EW features from the per-(met,age) window LUT with dust.

    The FeaturePrecomp fast path for baked-in (wNE) templates, where emission
    lines and indices are spectral features on the SSP and must be measured from
    the spectrum (no direct line output). Instead of reconstructing the full-grid
    SED (~1.0 ms) and measuring on it, contract the precomputed SSP window
    integrals with the published SFH+metallicity weights and apply the
    age-dependent two-component screen at each window center (~18 µs for this
    contraction; ~58x the full-grid measurement):

    .. math::

        \\langle F\\rangle_w = \\mathrm{scale}\\cdot \\sum_a
            T(a, \\lambda_c^w)\\,
            \\frac{\\sum_m w_{ma}\\,\\Phi_{maw}}{\\mathcal{N}_w}

    where :math:`\\Phi_{maw}` is ``precomp.window_integrals`` and :math:`T(a,
    \\lambda)` is the two-component transmission per SSP age.

    **Nebular emission through the birth cloud.** The two-component screen gives
    the youngest SSP age bins (age < ``t_birth``) the FULL birth-cloud + diffuse
    attenuation and older bins the diffuse screen only. For a baked-in SSP the
    nebular emission lives in those youngest bins, so applying :math:`T` per age
    reddens the emission by birth-cloud + diffuse automatically, matching the
    exact forward's ``lnu_age * transmission`` (validated < 4e-4 on Hα-EW /
    Dn4000 / Balmer). (For an *additive* nebular backend the emitted SED must be
    reddened at y=1 explicitly, that is the additive path, not this one.)

    Parameters
    ----------
    joint_weights : ndarray, shape (n_met, n_age)
        Published SFH × metallicity CSP weights (sum to 1).
    scale : float
        ``stellar_mass_scale`` = total_mass · L_sun [erg/s per (Msun weight)];
        cancels for break/EW ratios but keeps the window means physical.
    transmission_at_centers : ndarray, shape (n_age, n_window)
        Two-component transmission evaluated at each window center per SSP age
        (``two_component_dust(window_centers, ssp_ages, tau_bc, tau_diff, ...)``).
    precomp : IndexWindowPrecomputation
        Per-(met, age) window integrals from :func:`precompute_index_windows`.

    Returns
    -------
    ndarray, shape (n_index,)
        Index / emission-EW values, matching a full-SED measurement to < 4e-4
        (the residual is the intra-window transmission variation across the
        narrow feature windows).

    Notes
    -----
    **JIT-compatible**: yes, one ``einsum`` + a weighted age sum + the ratio
    measurement. This is the per-evaluation hot path replacing the full-grid SED
    reconstruction + measurement. Measured (CPU, PRSC wNE grid, 4 indices):
    ~18 µs for this contraction alone and ~60 µs end-to-end including the
    SED-free weight extract (:meth:`StellarSEDComponent.compute_joint_weights`)
    and the transmission evaluation, versus ~1.0 ms for the full-grid path, a
    ~17x per-evaluation win end-to-end (~58x for the measurement step in
    isolation).
    """
    # marginalize metallicity, keep age: (n_age, n_window)
    wint_age = jnp.einsum("ma,maw->aw", joint_weights, precomp.window_integrals)
    window_means = (
        scale * jnp.sum(transmission_at_centers * wint_age, axis=0) / precomp.window_norms
    )
    return measure_indices_from_windows(window_means, precomp)


# ── Observed data container ───────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SpectralIndexData:
    """Observed spectral index values for fitting.

    Parameters
    ----------
    index_defs : tuple of SpectralIndexDef
        Index definitions.
    values : jnp.ndarray
        Observed index values, shape ``(n_indices,)``.
    errors : jnp.ndarray
        1-sigma uncertainties, shape ``(n_indices,)``.

    Returns
    -------
    SpectralIndexData
        Spectral index data container with validation.

    Attributes
    ----------
    index_defs : tuple[SpectralIndexDef, ...]
        Index definitions.
    values : ndarray, shape (n_indices,)
        Observed index values [dimensionless].
    errors : ndarray, shape (n_indices,)
        1-sigma measurement uncertainties [dimensionless].

    Notes
    -----
    **Immutable container**: All fields are read-only by convention. Construct
    once with validated data, do not modify.

    **Indexing and access**: Use ``names`` property to get human-readable
    line identifiers, ``n_indices`` for count, and ``index_defs`` for the
    full definition metadata.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import SpectralIndexData
    >>> sid = SpectralIndexData.from_names(
    ...     names=["Dn4000", "HdA"],
    ...     values=[1.35, 5.2],
    ...     errors=[0.05, 0.3],
    ... )
    >>> sid.n_indices
    2
    >>> sid.names
    ('Dn4000', 'HdA')
    """

    index_defs: tuple[SpectralIndexDef, ...]
    values: jnp.ndarray = dataclasses.field(hash=False)
    errors: jnp.ndarray = dataclasses.field(hash=False)

    def __post_init__(self) -> None:
        n = len(self.index_defs)
        if n == 0:
            raise ValueError("SpectralIndexData requires at least one index.")

        values = jnp.asarray(self.values)
        errors = jnp.asarray(self.errors)

        if values.shape != (n,):
            raise ValueError(f"values shape {values.shape} does not match expected ({n},)")
        if errors.shape != (n,):
            raise ValueError(f"errors shape {errors.shape} does not match expected ({n},)")

    @property
    def n_indices(self) -> int:
        """Number of spectral indices.

        Returns
        -------
        int
            Number of indices in this dataset.

        Notes
        -----
        Computed from the length of the ``index_defs`` tuple. Constant
        for the lifetime of the object (immutable).

        """
        return len(self.index_defs)

    @property
    def names(self) -> tuple[str, ...]:
        """Tuple of spectral index names.

        Returns
        -------
        tuple[str, ...]
            Index names in the same order as ``index_defs``, e.g.
            ``("Dn4000", "HdA")``.

        Notes
        -----
        Names match the keys in STANDARD_INDICES.

        Examples
        --------
        >>> from tengri import SpectralIndexData
        >>> sid = SpectralIndexData.from_names(["Dn4000", "HdA"], [1.3, 5.1], [0.05, 0.3])
        >>> sid.names
        ('Dn4000', 'HdA')
        """
        return tuple(d.name for d in self.index_defs)

    @property
    def wave_range(self) -> tuple[float, float]:
        """Rest-frame wavelength range needed to measure all indices.

        Returns
        -------
        tuple[float, float]
            Tuple ``(wave_min, wave_max)`` [Angstrom] covering all continuum
            and feature windows across all indices.

        Notes
        -----
        Useful for determining minimum wavelength coverage required in the
        forward model spectrum to compute all indices.

        """
        lo = min(d.wave_min for d in self.index_defs)
        hi = max(d.wave_max for d in self.index_defs)
        return (lo, hi)

    @classmethod
    def from_names(
        cls,
        names: list[str],
        values: list[float],
        errors: list[float],
    ) -> SpectralIndexData:
        """Construct from standard index names.

        Parameters
        ----------
        names : list[str]
            Index names from ``STANDARD_INDICES`` (e.g. ``["Dn4000", "HdA"]``).
        values : list[float]
            Observed index values. Units depend on index type: [Angstrom] for
            EW indices, [dimensionless] for break indices.
        errors : list[float]
            1-sigma uncertainties (same units as ``values``).

        Returns
        -------
        SpectralIndexData
            Spectral index data object with index definitions looked up from
            ``STANDARD_INDICES``.

        Raises
        ------
        ValueError
            If any name is not in ``STANDARD_INDICES``.

        Notes
        -----
        The wavelength coverage required to measure all indices can be obtained
        via :func:`wave_range` property.

        """
        defs = []
        for name in names:
            if name not in STANDARD_INDICES:
                available = sorted(STANDARD_INDICES.keys())
                raise ValueError(f"Unknown index name {name!r}. Available: {available}")
            defs.append(STANDARD_INDICES[name])

        return cls(
            index_defs=tuple(defs),
            values=jnp.array(values),
            errors=jnp.array(errors),
        )

    def chi2(self, model_values: jnp.ndarray) -> jnp.ndarray:
        """Chi-squared statistic.

        Parameters
        ----------
        model_values : ndarray, shape (n_indices,)
            Model-predicted index values (same units as ``values``).

        Returns
        -------
        ndarray, shape ()
            Sum of ``((obs - model) / error)^2`` [dimensionless].

        Notes
        -----
        **JIT-compatible**: yes, uses only jnp primitives.

        **Gradient-safe**: yes, differentiable w.r.t. ``model_values``.

        """
        residual = (self.values - model_values) / self.errors
        return jnp.sum(residual**2)

    def log_likelihood(self, model_values: jnp.ndarray) -> jnp.ndarray:
        """Gaussian log-likelihood.

        Parameters
        ----------
        model_values : ndarray, shape (n_indices,)
            Model-predicted index values (same units as ``values``).

        Returns
        -------
        ndarray, shape ()
            Total log-likelihood summed over all indices [dimensionless].

        Notes
        -----
        **JIT-compatible**: yes, uses only jnp primitives.

        **Gradient-safe**: yes, differentiable w.r.t. ``model_values``.

        Assumes Gaussian uncertainties on the observed indices.

        """
        residual = (self.values - model_values) / self.errors
        return jnp.sum(-0.5 * residual**2 - jnp.log(self.errors) - 0.5 * jnp.log(2.0 * jnp.pi))

    def summary(self) -> str:
        """Return a human-readable summary of the spectral indices.

        Returns
        -------
        str
            Summary string (e.g., ``"2 indices (Dn4000, HdA)"``).

        Notes
        -----
        Intended for logging and diagnostics, not for programmatic parsing.

        """
        return f"{self.n_indices} indices ({', '.join(self.names)})"
