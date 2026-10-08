# SPDX-License-Identifier: BSD-3-Clause
r"""Spectral index definitions and observed data for direct fitting.

Supports two index types:

- **EW** (equivalent width): measures absorption or emission strength
  relative to a pseudo-continuum defined by sideband windows. The default
  follows the Lick definition (Trager et al. 1998): the pseudo-continuum is
  the straight line through the mean flux of the two sidebands, placed at the
  sideband mid-wavelengths, and the flux is :math:`F_\lambda`. The additive
  offsets of Trager et al. (1998) Sect. 5 that place indices on the IDS system
  are not applied.
- **break** (flux ratio): ratio of mean fluxes in two continuum windows
  (e.g., Dn4000).

Index values are measured on rest-frame spectra. The forward model generates
a spectrum covering the required wavelength range, measures indices on it,
and compares against observed values via a chi2 likelihood term.

The ``flux`` argument of every measuring function is a flux density per unit
frequency (:math:`L_\nu` or :math:`F_\nu`; any consistent units). EW and
magnitude indices convert it to :math:`F_\lambda \propto F_\nu/\lambda^2`
internally; break indices are ratios of :math:`F_\nu` window means (Balogh et
al. 1999). A spectrum in :math:`F_\lambda` is passed as ``flux_lambda *
wave_rest**2``.

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
import numpy as np

from tengri._cache_keys import KeyPolicy, content, derive_key, shape
from tengri.utils.air_vacuum import air_to_vac
from tengri.utils.break_windows import BREAK_VACUUM_WINDOWS
from tengri.utils.scale import representable_denominator, representable_floor

#: Reference wavelength [Å] of the F_λ conversion ``F_λ ∝ F_ν (λ_ref/λ)²``. The
#: proportionality constant cancels in every EW and magnitude index; the
#: dimensionless ratio keeps the converted flux at the scale of the input
#: (``F_ν/λ²`` is smaller by 1/λ² ~ 4e-8, which puts an SSP-scale flux under the
#: float32 floor of the continuum denominator).
_LAMBDA_REF_AA = 5000.0

#: Distance from the feature window [Å] beyond which the 1 Å sigmoid window weight is
#: below 1e-17 (40 edge widths).
_EDGE_REACH_AA = 40.0

#: Allowed values of :attr:`SpectralIndexDef.pseudo_continuum`.
_PSEUDO_CONTINUUM_MODES = ("linear", "mean")

# ── Index definition ──────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SpectralIndexDef:
    r"""Definition of a single spectral index.

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
    pseudo_continuum : {"linear", "mean"}
        How an EW index builds its pseudo-continuum (ignored by break and slope
        indices). ``"linear"`` (default) is the Lick definition [1]_: the
        straight line through the mean :math:`F_\lambda` of the two sidebands
        placed at the sideband mid-wavelengths, with
        :math:`\mathrm{EW} = \int (1 - F_\lambda/F_{C\lambda})\,d\lambda`
        over the feature window (magnitude indices: :math:`-2.5\log_{10}` of
        the feature-window mean of :math:`F_\lambda/F_{C\lambda}`), measured
        on :math:`F_\lambda` obtained from the ``flux`` argument
        (:math:`F_\lambda \propto F_\nu/\lambda^2`). ``"mean"`` is
        BAGPIPES' arithmetic (``bagpipes.input.spectral_indices.single_index``)
        and not the Lick definition: a constant continuum equal to the mean of
        the sideband means, ``W (C - <F>)/C``, on the ``flux`` array exactly as
        given (no frame conversion). It exists to compare with BAGPIPES.
        ``"linear"`` needs exactly two continuum windows.

    References
    ----------
    .. [1] Trager, S. C., Worthey, G., Faber, S. M., Burstein, D., Gonzalez,
           J. J., 1998, ApJS, 116, 1 (Sect. 2.2, Eqs. 1-3).

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
    pseudo_continuum: str = "linear"

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
        if self.pseudo_continuum not in _PSEUDO_CONTINUUM_MODES:
            raise ValueError(
                f"pseudo_continuum must be one of {_PSEUDO_CONTINUUM_MODES}, "
                f"got {self.pseudo_continuum!r}"
            )
        if self.index_type == "EW" and self.pseudo_continuum == "linear":
            if len(self.continuum) != 2:
                raise ValueError(
                    "EW indices with pseudo_continuum='linear' require exactly 2 "
                    f"continuum windows (blue, red), got {len(self.continuum)}."
                )
            (b_lo, b_hi), (r_lo, r_hi) = self.continuum
            if not 0.5 * (r_lo + r_hi) > 0.5 * (b_lo + b_hi):
                raise ValueError(
                    "The red sideband of a linear pseudo-continuum must lie at a longer "
                    "mid-wavelength than the blue one."
                )

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

#: Published Lick passbands in **air** Angstrom, exactly as tabulated by their
#: papers: ``name -> (continuum windows, feature window)``. Hbeta, Mgb, Fe5270,
#: Fe5335, Fe4383 and Ca4227 are Trager et al. (1998, ApJS 116, 1, Table 1;
#: Worthey et al. 1994 definitions); HdA, HdF, HgA and HgF are Worthey &
#: Ottaviani (1997, ApJS 111, 377, Table 1). Both were defined on spectra
#: calibrated against arc lamps in air, and FSPS likewise treats them as air
#: (``sps_setup.f90`` converts ``allindices.dat`` with ``airtovac``). They are
#: never used directly: :func:`_lick_index_def` converts every edge to vacuum
#: once, here at import, so the windows apply to tengri's vacuum spectra.
_LICK_AIR_WINDOWS: dict[str, tuple[tuple[tuple[float, float], ...], tuple[float, float]]] = {
    "HdA": (((4041.60, 4079.75), (4128.50, 4161.00)), (4083.50, 4122.25)),
    "HdF": (((4057.25, 4088.50), (4114.75, 4137.25)), (4091.00, 4112.25)),
    "HgA": (((4283.50, 4319.75), (4367.25, 4419.75)), (4319.75, 4363.50)),
    "HgF": (((4283.50, 4319.75), (4354.75, 4384.75)), (4331.25, 4352.25)),
    "Mgb": (((5142.63, 5161.38), (5191.38, 5206.38)), (5160.13, 5192.63)),
    "Fe5270": (((5233.15, 5248.15), (5285.65, 5318.15)), (5245.65, 5285.65)),
    "Fe5335": (((5304.63, 5315.88), (5353.38, 5363.38)), (5312.13, 5352.13)),
    "Hbeta": (((4827.88, 4847.88), (4876.63, 4891.63)), (4847.88, 4876.63)),
    "Fe4383": (((4359.13, 4370.38), (4442.88, 4455.38)), (4369.13, 4420.38)),
    "Ca4227": (((4211.00, 4219.75), (4241.00, 4251.00)), (4222.25, 4234.75)),
}


def _window_to_vacuum(window: tuple[float, float]) -> tuple[float, float]:
    """Convert one air-wavelength window ``(lo, hi)`` to vacuum [Angstrom]."""
    lo, hi = (float(w) for w in air_to_vac(np.asarray(window, dtype=np.float64)))
    return (lo, hi)


def _lick_index_def(name: str) -> SpectralIndexDef:
    """The vacuum-frame :class:`SpectralIndexDef` of a published air Lick index.

    Parameters
    ----------
    name : str
        Key of :data:`_LICK_AIR_WINDOWS`.

    Returns
    -------
    SpectralIndexDef
        Equivalent-width index whose continuum and feature windows are the
        published air edges converted with
        :func:`tengri.utils.air_vacuum.air_to_vac`.
    """
    continuum_air, feature_air = _LICK_AIR_WINDOWS[name]
    return SpectralIndexDef(
        name=name,
        index_type="EW",
        continuum=tuple(_window_to_vacuum(w) for w in continuum_air),
        feature=_window_to_vacuum(feature_air),
    )


#: Catalog of the 13 single-passband spectral indices tengri ships, keyed by
#: index name. Three kinds are represented: ``break`` ratios (``Dn4000``,
#: ``D4000``), Lick-style equivalent widths (``HdA``, ``HdF``, ``HgA``,
#: ``HgF``, ``Hbeta``, ``Mgb``, ``Fe4383``, ``Fe5270``, ``Fe5335``,
#: ``Ca4227``), and one continuum ``slope`` (``uv_slope_beta``). The Lick
#: equivalent widths use the Lick pseudo-continuum (see
#: :attr:`SpectralIndexDef.pseudo_continuum`). Values are
#: :class:`SpectralIndexDef` records carrying the passband definitions in
#: rest-frame **vacuum** Angstrom: the Lick windows (:data:`_LICK_AIR_WINDOWS`)
#: and the 4000 A break windows (:mod:`tengri.utils.break_windows`) are published
#: in air and converted once at import; the UV slope range
#: (Calzetti+1994, 1250-2600 A, measured on space-based IUE spectra) is vacuum. Pass a key
#: to :func:`tengri.measure.spectral_index` or :func:`tengri.measure_index_jax`.
STANDARD_INDICES: dict[str, SpectralIndexDef] = {
    "Dn4000": SpectralIndexDef(
        name="Dn4000",
        index_type="break",
        continuum=BREAK_VACUUM_WINDOWS["Dn4000"],
    ),
    "D4000": SpectralIndexDef(
        name="D4000",
        index_type="break",
        continuum=BREAK_VACUUM_WINDOWS["D4000"],
    ),
    **{name: _lick_index_def(name) for name in _LICK_AIR_WINDOWS},
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
    r"""Measure a spectral index on a rest-frame spectrum.

    Parameters
    ----------
    wave_rest : ndarray, shape (n_pix,)
        Rest-frame wavelengths [Angstrom]. Must cover all windows defined
        in ``index_def``.
    flux : ndarray, shape (n_pix,)
        Flux density **per unit frequency**, :math:`L_\nu` or :math:`F_\nu`
        (any consistent units; only ratios matter, so no distance is needed).
        EW and magnitude indices convert it to :math:`F_\lambda \propto
        F_\nu/\lambda^2` internally (Lick definition); break indices use it as
        given (ratio of :math:`F_\nu` window means); a slope index fits it as
        :math:`F_\nu`. A spectrum in :math:`F_\lambda` is passed as
        ``flux_lambda * wave_rest**2``. An index defined with
        ``pseudo_continuum="mean"`` measures ``flux`` as given, in any frame.
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

    An EW index integrates :math:`1 - F_\lambda/F_{C\lambda}` over the soft
    feature window (weights :math:`w`, normalized so :math:`\int w\,d\lambda`
    stands for the window width :math:`W`): ``W (1 - <F/F_C>_w)``.

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


def _soft_window(wave: jnp.ndarray, lo: float, hi: float, edge_width: float = 1.0) -> jnp.ndarray:
    """Soft top-hat weights with sigmoid edges (differentiable window boundaries)."""
    return jax.nn.sigmoid((wave - lo) / edge_width) * jax.nn.sigmoid((hi - wave) / edge_width)


def _to_flam(wave: jnp.ndarray, flux_nu: jnp.ndarray) -> jnp.ndarray:
    """Convert a per-frequency flux density to F_λ up to a constant factor.

    ``F_λ ∝ F_ν/λ²``; the returned ``F_ν (λ_ref/λ)²`` differs from F_λ by the
    constant ``λ_ref²/c`` only, which cancels in an EW or magnitude index.
    """
    return flux_nu * (_LAMBDA_REF_AA / wave) ** 2


def _window_mean_flux(
    wave: jnp.ndarray, flux: jnp.ndarray, lo: float, hi: float, edge_width: float = 1.0
) -> jnp.ndarray:
    """Wavelength-averaged flux in a window: ∫(flux·dλ) / ∫dλ.

    Uses soft sigmoid edges for differentiability. The weight function is the
    edge sigmoid product, which is integrated with dλ to give the correct
    wavelength mean on any grid (uniform or clustered).

    Parameters
    ----------
    wave : ndarray
        Wavelength grid [Å]
    flux : ndarray
        Flux density array (any consistent units)
    lo, hi : float
        Window edges [Å]
    edge_width : float
        Sigmoid edge width [Å]. Default 1.0.

    Returns
    -------
    float
        Mean flux: ∫(flux·w·dλ) / ∫(w·dλ), where w is the sigmoid edge product.
    """
    weights = _soft_window(wave, lo, hi, edge_width)

    # Trapezoid mean: ∫(flux·w·dλ) / ∫(w·dλ), the same on any grid
    num = jnp.trapezoid(flux * weights, wave)
    den = jnp.trapezoid(weights, wave)

    # Avoid division by zero; use a safe denominator
    ok = den > 1e-20
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


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
    return f_red / jnp.maximum(f_blue, representable_denominator(1e-30))


def _linear_continuum(x, x_blue, x_red, f_blue, f_red):
    """Straight line through ``(x_blue, f_blue)`` and ``(x_red, f_red)``, evaluated at ``x``."""
    return f_blue + (f_red - f_blue) * (x - x_blue) / (x_red - x_blue)


def _index_from_ratio(ratio: jnp.ndarray, feat_width: float, units: str) -> jnp.ndarray:
    """EW ``W (1 - ratio)`` or magnitude ``-2.5 log10(ratio)`` from the mean ``F/F_C``."""
    if units == "mag":
        return -2.5 * jnp.log10(jnp.maximum(ratio, representable_floor(1e-30)))
    return feat_width * (1.0 - ratio)


def _lick_geometry(idx: SpectralIndexDef) -> tuple[float, float, float, float]:
    """Sideband mid-wavelengths and feature bounds ``(x_blue, x_red, feat_lo, feat_hi)`` [Å]."""
    (b_lo, b_hi), (r_lo, r_hi) = idx.continuum
    return (0.5 * (b_lo + b_hi), 0.5 * (r_lo + r_hi), idx.feature[0], idx.feature[1])


def _lick_continuum(x, geometry, f_blue, f_red):
    """Lick pseudo-continuum ``F_C(x)``: the line through the two sideband means.

    The single definition of the line (Trager et al. 1998, Eqs. 1-3), called by
    the exact path (:func:`_measure_ew`, ``x`` the pixel wavelengths) and by the
    window-LUT path (:func:`measure_indices_from_point_terms`, ``x`` the window
    grid points), so the two cannot drift. ``f_blue``, ``f_red`` are the
    :math:`F_\\lambda` sideband means, placed at the sideband mid-wavelengths.

    The line is evaluated within ``_EDGE_REACH_AA`` of the feature only: beyond
    that the window weight is < 1e-17, and an extrapolated zero crossing far away
    would floor ``F_C`` on pixels the index cannot see, where the VJP's
    ``flux / floor**2`` overflows float32.
    """
    x_blue, x_red, feat_lo, feat_hi = geometry
    x_eval = jnp.clip(x, feat_lo - _EDGE_REACH_AA, feat_hi + _EDGE_REACH_AA)
    return jnp.maximum(
        _linear_continuum(x_eval, x_blue, x_red, f_blue, f_red),
        representable_denominator(1e-30),
    )


def _ew_from_means(
    cont_means: jnp.ndarray, feat_flux: jnp.ndarray, feat_width: float, units: str
) -> jnp.ndarray:
    """Equivalent width from continuum + feature mean fluxes (constant continuum).

    Averages the continuum-window means, forms the continuum-to-feature ratio
    scaled by the feature width, and (for ``units == "mag"``) converts to a
    magnitude index. The EW primitive of ``pseudo_continuum="mean"`` (BAGPIPES'
    arithmetic), shared by :func:`_measure_ew` and
    :func:`measure_indices_from_point_terms`. The Lick (linear) definition goes
    through :func:`_lick_continuum`.
    """
    cont_flux = jnp.mean(jnp.asarray(cont_means))
    ew = (
        feat_width
        * (cont_flux - feat_flux)
        / jnp.maximum(cont_flux, representable_denominator(1e-30))
    )
    if units == "mag":
        return -2.5 * jnp.log10(jnp.maximum(1.0 - ew / feat_width, representable_floor(1e-30)))
    return ew


def _measure_break(wave: jnp.ndarray, flux: jnp.ndarray, idx: SpectralIndexDef) -> jnp.ndarray:
    """Compute spectral break ratio (red window mean flux / blue window mean flux)."""
    blue_lo, blue_hi = idx.continuum[0]
    red_lo, red_hi = idx.continuum[1]
    f_blue = _window_mean_flux(wave, flux, blue_lo, blue_hi)
    f_red = _window_mean_flux(wave, flux, red_lo, red_hi)
    return _break_from_means(f_blue, f_red)


def _measure_ew(wave: jnp.ndarray, flux: jnp.ndarray, idx: SpectralIndexDef) -> jnp.ndarray:
    """Equivalent (or magnitude) index on the reconstructed spectrum.

    ``pseudo_continuum="linear"``: Lick definition on F_λ; the feature-window
    mean of ``F_λ/F_C`` is integrated on the grid with the soft window weights.
    ``"mean"``: BAGPIPES' constant-continuum arithmetic on ``flux`` as given.
    """
    feat_lo, feat_hi = idx.feature
    feat_width = feat_hi - feat_lo
    if idx.pseudo_continuum == "mean":
        cont_fluxes = [_window_mean_flux(wave, flux, lo, hi) for lo, hi in idx.continuum]
        feat_flux = _window_mean_flux(wave, flux, feat_lo, feat_hi)
        return _ew_from_means(cont_fluxes, feat_flux, feat_width, idx.units)
    flam = _to_flam(wave, flux)
    (b_lo, b_hi), (r_lo, r_hi) = idx.continuum
    f_c = _lick_continuum(
        wave,
        _lick_geometry(idx),
        _window_mean_flux(wave, flam, b_lo, b_hi),
        _window_mean_flux(wave, flam, r_lo, r_hi),
    )
    weights = _soft_window(wave, feat_lo, feat_hi)
    den = jnp.trapezoid(weights, wave)
    ok = den > 1e-20
    ratio = jnp.where(ok, jnp.trapezoid(weights * flam / f_c, wave) / jnp.where(ok, den, 1.0), 0.0)
    return _index_from_ratio(ratio, feat_width, idx.units)


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
    denom = sxx - sx**2 / jnp.maximum(sw, representable_denominator(1e-30))
    slope_fnu = (sxy - sx * sy / jnp.maximum(sw, representable_denominator(1e-30))) / jnp.maximum(
        denom, representable_denominator(1e-30)
    )
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
#     <SED>_win = ∫ (Σ_ij w_ij SSP_ij(λ)) W(λ) dλ / ∫ W(λ) dλ
#               = Σ_ij w_ij · [∫ SSP_ij(λ) W(λ) dλ] / ∫ W(λ) dλ
#               = Σ_ij w_ij · ssp_window_integral_ij / window_norm .
#
# Slope indices are NOT expressible this way (they need the SED shape within the
# window, not one integral) and are excluded, callers fall back to the exact
# ``measure_index_jax`` path for them.


#: Sigmoid edge widths either side of a window beyond which its soft weight is
#: below 1e-13 of unity, smaller than a float64 trapezoid sum resolves.
WINDOW_SUPPORT_EDGES: float = 30.0


@dataclasses.dataclass(frozen=True)
class WindowPoints:
    """SSP window integrals resolved by grid point, for an in-window dust screen.

    The exact path multiplies the SED by the dust transmission :math:`T(\\lambda)`
    *before* it takes the window mean, so the mean is
    :math:`\\int F T W\\,d\\lambda`, not :math:`T(\\lambda_c)\\int F W\\,d\\lambda`.
    The two differ by the variation of :math:`T` across the window, which is a
    fraction of a percent of a window mean but a large fraction of a faint line
    that is a small difference of two large means beside a strong neighbor
    (13 % for [N II] 6584 next to Halpha, #2677). Holding the SSP integrand per
    grid point lets the LUT apply :math:`T` where the exact path does.

    Attributes
    ----------
    waves : ndarray, shape (n_point,)
        SSP wavelength of each point [Å] (the dust screen is evaluated here).
    integrands : ndarray, shape (n_met, n_age, n_point)
        :math:`\\mathrm{SSP}\\,W\\,\\Delta\\lambda_{\\rm trapz}` per point, so a
        window's integral is the sum over its points.
    window : ndarray of int, shape (n_point,)
        Window slot each point belongs to.
    """

    waves: jnp.ndarray
    integrands: jnp.ndarray
    window: jnp.ndarray


@dataclasses.dataclass(frozen=True)
class IndexWindowPrecomputation:
    """Precomputed SSP window integrals for break / EW spectral indices.

    Built once at model construction (``approx=FeaturePrecomp()``) from the SSP
    grid and the configured index windows. Consumed per evaluation by
    :func:`measure_indices_from_windows` (break and ``"mean"`` EW) or
    :func:`measure_indices_from_point_terms` (every index) after the stellar
    component SFH-weights the SSP integrands.

    Attributes
    ----------
    window_integrals : ndarray, shape (n_met, n_age, n_window)
        :math:`\\int \\mathrm{SSP}_{ij}(\\lambda)\\,W_w(\\lambda)\\,d\\lambda`, the
        soft-window trapezoid integral of each SSP spectrum
        [erg/s/Hz/Msun · Å] on the SSP wave grid.
    window_norms : ndarray, shape (n_window,)
        :math:`\\int W_w(\\lambda)\\,d\\lambda`, window width, so
        ``mean = integral / norm`` matches :func:`_window_mean_flux`.
    window_centers : ndarray, shape (n_window,)
        Window mid-wavelength ``0.5*(lo+hi)`` [Å], for per-window dust.
    index_slots : tuple
        Per index, ``(kind, payload, meta)`` describing which window slots the
        index consumes and how to combine them, see
        :func:`measure_indices_from_windows`. ``kind`` is ``"break"``,
        ``"EW"``, or ``"slope"`` (the last carries ``payload=None`` and is a
        sentinel that the caller must measure exactly). The EW ``meta`` is
        ``(feature_width, units, geometry)`` with ``geometry`` the Lick sideband
        and feature bounds of :func:`_lick_geometry` for a linear
        pseudo-continuum and ``None`` for ``pseudo_continuum="mean"``.
    names : tuple of str
        Index names in order, for diagnostics / alignment with observed data.
    points : WindowPoints
        The same window integrals resolved by SSP grid point, so the dust screen
        is applied inside each window as the exact path applies it (#2677).
    """

    window_integrals: jnp.ndarray
    window_norms: jnp.ndarray
    window_centers: jnp.ndarray
    index_slots: tuple
    names: tuple
    points: WindowPoints

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
        :math:`\\int \\mathrm{SSP}(\\lambda)\\,W(\\lambda)\\,d\\lambda` (trapezoid).
    norm : ndarray, shape ()
        :math:`\\int W(\\lambda)\\,d\\lambda` (trapezoid).
    """
    w = jax.nn.sigmoid((ssp_wave - lo) / edge_width) * jax.nn.sigmoid((hi - ssp_wave) / edge_width)
    # Δλ-weighted, as in _window_mean_flux: mean = integral / norm on any grid.
    integral = jnp.trapezoid(ssp_flux * w, ssp_wave, axis=-1)  # (n_met, n_age)
    return integral, jnp.maximum(jnp.trapezoid(w, ssp_wave), 1e-10)


def soft_window_ssp_points(
    ssp_wave, ssp_flux, lo, hi, edge_width: float = 1.0, support: float = WINDOW_SUPPORT_EDGES
):
    """Per-grid-point SSP integrand of a soft window, trapezoid-weighted.

    Summing the returned integrand over points reproduces
    :func:`soft_window_ssp_integral` (to the ``support`` truncation, below float64
    resolution at the default). Keeping the points separate lets the caller
    multiply the dust transmission in at each wavelength, as the exact path does.

    Returns
    -------
    waves : ndarray, shape (n_point,)
    integrand : ndarray, shape (n_met, n_age, n_point)
    """
    wave_np = np.asarray(ssp_wave, dtype=float)
    keep = np.nonzero(
        (wave_np >= lo - support * edge_width) & (wave_np <= hi + support * edge_width)
    )[0]
    dl = np.diff(wave_np)
    trapz_w = np.zeros_like(wave_np)
    trapz_w[:-1] += 0.5 * dl
    trapz_w[1:] += 0.5 * dl
    wave_k = jnp.asarray(ssp_wave)[keep]
    w = jax.nn.sigmoid((wave_k - lo) / edge_width) * jax.nn.sigmoid((hi - wave_k) / edge_width)
    return wave_k, jnp.asarray(ssp_flux)[..., keep] * (w * jnp.asarray(trapz_w)[keep])


def stack_window_points(waves: list, integrands: list, n_met: int, n_age: int, dtype):
    """Concatenate per-window point sets into one :class:`WindowPoints`."""
    if not waves:
        return WindowPoints(
            jnp.zeros((0,), dtype), jnp.zeros((n_met, n_age, 0), dtype), jnp.zeros((0,), int)
        )
    window = np.concatenate([np.full(len(w), k, dtype=int) for k, w in enumerate(waves)])
    return WindowPoints(
        waves=jnp.concatenate(waves),
        integrands=jnp.concatenate(integrands, axis=-1),
        window=jnp.asarray(window),
    )


def window_point_terms(joint_weights, transmission_at_points, points: WindowPoints, scale=1.0):
    """SFH-weighted, dust-screened window integrand at every window grid point.

    :math:`\\sum_a T(a,\\lambda_p) \\sum_m w_{ma}\\,\\mathrm{SSP}_{ma}(\\lambda_p)
    W(\\lambda_p)\\Delta\\lambda_p`: the per-point terms whose segment sum is the window
    integral (:func:`window_means_with_dust`). Each term already carries the
    trapezoid and soft-edge weight, so a measurement that needs the spectrum
    point by point (the Lick continuum, :func:`measure_indices_from_point_terms`)
    takes them as they are.

    Parameters
    ----------
    joint_weights : ndarray, shape (n_met, n_age)
    transmission_at_points : ndarray, shape (n_age, n_point)
    points : WindowPoints
    scale : float or ndarray, shape (), default 1.0
        See :func:`window_means_with_dust`.

    Returns
    -------
    ndarray, shape (n_point,)
    """
    wint = jnp.einsum("ma,map->ap", joint_weights * scale, points.integrands)
    return jnp.sum(transmission_at_points * wint, axis=0)


def window_means_with_dust(
    joint_weights, transmission_at_points, points: WindowPoints, norms, scale=1.0
):
    """SFH-weighted window means with the dust screen applied inside each window.

    :math:`\\langle F\\rangle_w = \\sum_p T(a,\\lambda_p) \\sum_{m,a} w_{ma}\\,
    \\mathrm{SSP}_{ma}(\\lambda_p) W_w(\\lambda_p)\\Delta\\lambda_p / \\mathcal N_w`,
    the same sum the exact path takes over the dust-attenuated SED.

    Parameters
    ----------
    joint_weights : ndarray, shape (n_met, n_age)
    transmission_at_points : ndarray, shape (n_age, n_point)
        Two-component transmission at ``points.waves`` per SSP age.
    points : WindowPoints
    norms : ndarray, shape (n_window,)
    scale : float or ndarray, shape (), default 1.0
        Multiplies the SFH weights before they meet the integrands. A caller that
        restores a large constant afterwards (``_LSUN_POW2`` on the line path)
        passes its small factor here rather than multiplying the returned means:
        ``(scale * mean) * 2**112`` is two adjacent scalar multiplies, which XLA
        reassociates in the backward pass into ``ct * (scale * 2**112)``, and that
        product (~1e44) overflows float32 although every true value is in range
        (#2677). Entering through the weights puts the contraction between them.

    Returns
    -------
    ndarray, shape (n_window,)
        Window means, per unit weight times ``scale`` [erg/s/Hz].
    """
    per_point = window_point_terms(joint_weights, transmission_at_points, points, scale)
    integral = jax.ops.segment_sum(per_point, points.window, num_segments=norms.shape[0])
    return integral / norms


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
    pt_waves: list[jnp.ndarray] = []
    pt_integrands: list[jnp.ndarray] = []

    def _slot(lo, hi) -> int:
        key = _round_window(lo, hi)
        if key in unique:
            return unique[key]
        integral, norm = soft_window_ssp_integral(ssp_wave, ssp_flux, lo, hi, edge_width)
        pw, pi = soft_window_ssp_points(ssp_wave, ssp_flux, lo, hi, edge_width)
        pt_waves.append(pw)
        pt_integrands.append(pi)
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
            geometry = _lick_geometry(idx) if idx.pseudo_continuum == "linear" else None
            slots.append(("EW", (cont, feat), (feat_width, idx.units, geometry)))
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
        points=stack_window_points(
            pt_waves, pt_integrands, ssp_flux.shape[0], ssp_flux.shape[1], ssp_flux.dtype
        ),
    )


def measure_indices_from_windows(
    window_means: jnp.ndarray, precomp: IndexWindowPrecomputation
) -> jnp.ndarray:
    """Evaluate break / constant-continuum EW indices from per-window mean fluxes.

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

    Raises
    ------
    ValueError
        If an EW index has ``pseudo_continuum="linear"``: its feature term
        :math:`\\int F_\\lambda/F_C\\,d\\lambda` is not a function of window
        means, so use :func:`measure_indices_from_point_terms`.

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
            feat_width, units, geometry = meta
            if geometry is not None:
                raise ValueError(
                    "A pseudo_continuum='linear' EW cannot be formed from window means; "
                    "use measure_indices_from_point_terms."
                )
            cont_means = [window_means[c] for c in cont_slots]
            out.append(_ew_from_means(cont_means, window_means[feat], feat_width, units))
        else:  # slope
            out.append(jnp.asarray(jnp.nan))
    return jnp.stack(out)


def _lick_ew_from_points(flam_terms, wave, window, flam_means, norms, cont_slots, feat, meta):
    """Lick EW from F_λ point terms: the point-wise twin of :func:`_measure_ew`.

    ``flam_terms`` are the F_λ integrand terms (trapezoid and soft-edge weight
    included) at the window grid points ``wave``; the continuum is
    :func:`_lick_continuum` evaluated at each feature point.
    """
    feat_width, units, geometry = meta
    f_c = _lick_continuum(wave, geometry, flam_means[cont_slots[0]], flam_means[cont_slots[1]])
    num = jnp.sum(jnp.where(window == feat, flam_terms / f_c, 0.0))
    den = norms[feat]
    ok = den > 1e-20
    ratio = jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)
    return _index_from_ratio(ratio, feat_width, units)


def measure_indices_from_point_terms(
    point_terms: jnp.ndarray, precomp: IndexWindowPrecomputation
) -> jnp.ndarray:
    """Evaluate every break / EW index from the per-grid-point window terms.

    The point-wise measurement: a Lick equivalent width (Trager et al. 1998,
    Eqs. 1-3) is :math:`\\int (1 - F_\\lambda/F_C(\\lambda))\\,d\\lambda` with
    :math:`F_C` the line through the :math:`F_\\lambda` sideband means, a
    nonlinear function of the spectrum. The LUT evaluates it as the exact path
    does, at the window grid points, with the same :func:`_lick_continuum`; the
    feature sum is :math:`\\sum_p F_{\\lambda,p}/F_C(\\lambda_p)` over the feature
    window's points. Breaks and ``pseudo_continuum="mean"`` EWs use the
    :math:`F_\\nu` window means of :func:`measure_indices_from_windows`.

    Parameters
    ----------
    point_terms : ndarray, shape (n_point,)
        Output of :func:`window_point_terms` (times any overall scale): the
        attenuated, SFH-weighted :math:`L_\\nu` integrand at each of
        ``precomp.points.waves``, trapezoid and soft-edge weight included.
    precomp : IndexWindowPrecomputation
        The build-time window recipe.

    Returns
    -------
    ndarray, shape (n_index,)
        Index values in ``precomp.names`` order; slope slots return ``nan``.

    Notes
    -----
    **JIT-compatible**: yes; no Python loop runs over points. The conversion to
    :math:`F_\\lambda` is :func:`_to_flam`, the one the exact path uses.
    """
    pts = precomp.points
    norms = precomp.window_norms
    n_window = norms.shape[0]
    nu_means = jax.ops.segment_sum(point_terms, pts.window, num_segments=n_window) / norms
    flam_terms = _to_flam(pts.waves, point_terms)
    flam_means = jax.ops.segment_sum(flam_terms, pts.window, num_segments=n_window) / norms
    out = []
    for kind, payload, meta in precomp.index_slots:
        if kind == "break":
            b, r = payload
            out.append(_break_from_means(nu_means[b], nu_means[r]))
        elif kind == "EW":
            cont_slots, feat = payload
            feat_width, units, geometry = meta
            if geometry is None:
                cont_means = [nu_means[c] for c in cont_slots]
                out.append(_ew_from_means(cont_means, nu_means[feat], feat_width, units))
            else:
                out.append(
                    _lick_ew_from_points(
                        flam_terms,
                        pts.waves,
                        pts.window,
                        flam_means,
                        norms,
                        cont_slots,
                        feat,
                        meta,
                    )
                )
        else:  # slope
            out.append(jnp.asarray(jnp.nan))
    return jnp.stack(out)


def measure_indices_from_window_lut(
    joint_weights: jnp.ndarray,
    scale: jnp.ndarray,
    transmission_at_points: jnp.ndarray,
    precomp: IndexWindowPrecomputation,
) -> jnp.ndarray:
    """Measure break/EW features from the per-(met,age) window LUT with dust.

    The FeaturePrecomp fast path for baked-in (wNE) templates, where emission
    lines and indices are spectral features on the SSP and must be measured from
    the spectrum (no direct line output). Instead of reconstructing the full-grid
    SED (~1.0 ms) and measuring on it, contract the precomputed SSP window
    integrals with the published SFH+metallicity weights and apply the
    age-dependent two-component screen at each window grid point (~18 µs for this
    contraction; ~58x the full-grid measurement):

    .. math::

        \\langle F\\rangle_w = \\frac{\\mathrm{scale}}{\\mathcal{N}_w}
            \\sum_{a,p \\in w} T(a, \\lambda_p)\\,\\sum_m w_{ma}\\,\\phi_{map}

    where :math:`\\phi_{map}` is ``precomp.points.integrands`` (the SSP window
    integrand at grid point :math:`p`) and :math:`T(a, \\lambda)` is the
    two-component transmission per SSP age.

    **Nebular emission through the birth cloud.** The two-component screen gives
    the youngest SSP age bins (age < ``t_birth``) the FULL birth-cloud + diffuse
    attenuation and older bins the diffuse screen only. For a baked-in SSP the
    nebular emission lives in those youngest bins, so applying :math:`T` per age
    reddens the emission by birth-cloud + diffuse automatically, matching the
    exact forward's ``lnu_age * transmission`` (validated < 4e-4 on Hα-EW /
    Dn4000 / Balmer). (For an *additive* nebular backend the emitted SED must be
    reddened at y=1 explicitly; that is the additive path, not this one.)

    Parameters
    ----------
    joint_weights : ndarray, shape (n_met, n_age)
        Published SFH × metallicity CSP weights (sum to 1).
    scale : float
        ``stellar_mass_scale`` = total_mass · L_sun [erg/s per (Msun weight)];
        cancels for break/EW ratios but keeps the window means physical.
    transmission_at_points : ndarray, shape (n_age, n_point)
        Two-component transmission per SSP age at every window grid point
        ``precomp.points.waves``, so the screen acts inside each window as in the
        exact path (#2677).
    precomp : IndexWindowPrecomputation
        Per-(met, age) window integrals from :func:`precompute_index_windows`.

    Returns
    -------
    ndarray, shape (n_index,)
        Index / emission-EW values, equal to a full-SED measurement to float
        rounding: the dust screen is applied at every window grid point, as the
        exact path applies it (#2677).

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
    point_terms = scale * window_point_terms(joint_weights, transmission_at_points, precomp.points)
    return measure_indices_from_point_terms(point_terms, precomp)


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

    def cache_key(self) -> tuple:
        """Return a hashable cache key for this spectral index data.

        Returns
        -------
        tuple
            Cache key derived from index definitions and array shapes.

        Notes
        -----
        Index definitions are keyed by content as they define the likelihood function.
        Value and error arrays are keyed by shape only (array values don't affect
        the program, only the per-galaxy data does).
        """
        return derive_key(self, _SPECTRAL_INDEX_DATA_CACHE_KEY_POLICY)

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


_SPECTRAL_INDEX_DATA_CACHE_KEY_POLICY: KeyPolicy = {
    "index_defs": content("index definitions determine which windows are measured"),
    "values": shape("per-galaxy data, not part of the structural program"),
    "errors": shape("per-galaxy data, not part of the structural program"),
}
