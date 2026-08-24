# SPDX-License-Identifier: BSD-3-Clause
"""SEDModel: high-level forward model wrapping the tengri SED pipeline.

SEDModel provides a clean API for:

- Forward predictions (SED, photometry, spectrum, SFH, derived quantities)
- Mock galaxy generation (single and batch)
- Convenience fitting (delegates to Fitter)

SEDModel translates between the user-facing parameter names and the
internal names used by the low-level functions, handling unit conversions
automatically. SFH computation is dispatched through the registry-driven
composed function, eliminating separate stochastic/parametric code paths.

Usage::

    from tengri import SEDModel, Parameters, Uniform, load_ssp_data, load_filter_set

    ssp = load_ssp_data("data/ssp.h5")
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    spec = Parameters(
        sfh_tsnorm_log_total_mass=Uniform(8, 12),
        sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
        sfh_tsnorm_width_gyr=Uniform(0.5, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        sfh_field_psd_sigma=Uniform(0.01, 1.0),
        sfh_field_psd_tau_myr=Uniform(10, 500),
        redshift=0.1,
    )
    model = SEDModel(spec, ssp, filters=filters)
    params = spec.sample(jax.random.PRNGKey(0))
    photometry = model.predict_photometry(params)

Navigation
----------
This file stays one module by design; use the ``# ── <section> ──``
marker lines to jump. In order:

- ``WavePrecomp`` / ``SpectrumPrecomp``, build-time approximation configs
- ``SEDModel`` class:

  - ``SubModel Protocol surface``
  - ``Construction``: ``__init__``, the ``_init_*`` chain, ``build()``
  - ``Deprecated filter/noise attributes → Observation delegation``
  - ``Core physics (SFH → SED pipeline)``
  - ``Predictions (public API)``: the ``predict_*`` surface
  - ``Component orchestrator path``: ``predict_state`` and the JIT
    kernel behind every prediction
  - ``Batch operations``
  - ``Private prediction dispatch``
  - ``Utilities``

"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import pathlib
import types
import warnings
from collections.abc import Mapping
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from tengri._deprecated import UNSET, resolve_renamed_flag
from tengri.components.stellar.sfh.registry import compute_field_gp, resolve_sfh
from tengri.components.stellar.sps.dsps_wrapper import csp_age_dt
from tengri.config.exceptions import (
    DeadGradientParameterWarning,
    DegenerateParameterPairWarning,
    ParameterMapError,
    warn_measured,
)
from tengri.cosmology import age_at_z, luminosity_distance
from tengri.forward.approx_policy import BAND_PROJECTION_KEYS, ApproxPolicy
from tengri.forward.sed_model_types import (
    MockData,
    PriorPredictive,
    SEDModelState,
)
from tengri.inference._backend_registry import DEFAULT_METHOD
from tengri.observation.photometry import ab_mag_from_flux
from tengri.parameters.translate import (
    _CUE_GAS_IDENTITY_PARAMS,
    _CUE_IONSPEC_IDENTITY_PARAMS,
    LOG10_ZSUN,
    _build_param_map,
    check_missing_free_params,
    check_unknown_params,
    get_internal_params,
)
from tengri.utils.filter_convention import FilterConvention
from tengri.utils.grid import (
    grid_spacing,
    interpolate_to_linear_time,
    log_age_to_age_yr,
    make_log_age_grid,
)
from tengri.utils.scale import apply_log10_scale, log10_four_pi_dl2

#: Second probe luminosity [erg/s] for the additive-emitter homogeneity check in
#: :meth:`SEDModel._dust_emission_band_response`. A physically plausible L_IR
#: (~1e44 erg/s ≈ 2.6e10 Lsun, a LIRG), paired with the L=1 unit probe. An emitter
#: whose template *shape* depends on luminosity (BOSA: L_TIR–sSFR) fails to scale
#: between the two and is refused a constant band response.
_L_IR_PROBE = 1.0e44

#: Properties whose value is read off the rest-frame SED, which the fast-nebular
#: grid path used to delete the nebular contribution from (#950 zeroes
#: ``nebular_sed``; that deleted Cue forward IS the speedup).
#:
#: This is a **census, not a refusal list**. ``predict_properties`` refused these
#: on a fast-nebular model until #1673 made ``predict_state`` materialize the
#: nebular component, after which they are served from a complete forward state
#: and match the exact path. The set is retained as the list of properties that
#: depend on the nebular continuum, which is exactly the set worth regression-
#: testing for equality --
#: ``test_sed_derived_properties_are_exact_on_the_fast_path`` iterates it.
#:
#: The census is established by **measurement**, not by reading which helpers
#: touch ``sed``: a broadband integral can be insensitive to the missing nebular
#: flux at printed precision, and guarding the read list would over-refuse. Each
#: name below was measured to move under ``approx=(WavePrecomp(),
#: FeaturePrecomp())`` against exact on an FSPS/MILES Cue model at z=0.05 --
#: 13 of 43 properties, worst first:
#:
#:   l_tir 30.07%   irx_fuv 29.26%   irx 27.22%   l_dust_absorbed 18.67%
#:   xi_ion 6.20%   fuv_flux 5.84%   rest_uv_color 2.78%   l_bol 2.19%
#:   nuv_flux 2.05%   uv_slope_beta 2.01%   dn4000 1.00%   balmer_break 0.64%
#:   m_uv 0.08%
#:
#: The energy-balance quantities are the worst hit, which is the tell: deleting
#: the nebular continuum removes reprocessed luminosity the dust budget is
#: balanced against, so l_tir/irx/l_dust_absorbed move by ~20-30% while the
#: shape indices move by ~1-3%.
_FAST_NEBULAR_UNSAFE_PROPERTIES = frozenset(
    {
        "balmer_break",
        "dn4000",
        "fuv_flux",
        "irx",
        "irx_fuv",
        "l_bol",
        "l_dust_absorbed",
        "l_tir",
        "m_uv",
        "nuv_flux",
        "rest_uv_color",
        "uv_slope_beta",
        "xi_ion",
    }
)


def _nebular_continuum_consumers(chain):
    """Components that read ``sed_nebular``, excluding the nebular component itself.

    **The single expression that decides whether the fast nebular grid may serve
    photometry.** Serving photometry from the per-Q_H grid requires zeroing
    ``sed_nebular``, so it is available only when nothing downstream reads the
    continuum. A non-empty result sets ``must_materialize_sed`` and disarms the
    shortcut.

    Extracted so that the code which *acts* on it
    (:meth:`SEDModel.enable_fast_nebular`) and the code which *advises about it*
    (:func:`~tengri.inference.fitter.fast_nebular_can_engage`, and the warnings that
    quote a speedup) cannot disagree. They did: after #1281 made materialization the
    default, ``DustSEDComponent`` disarmed the shortcut on every dusty model while
    three warnings and ``CLAUDE.md`` still advertised a ~21x line speedup that was
    measured at **1.00x, bit-identical compiled FLOPs** (#1748).

    Parameters
    ----------
    chain: sequence
        The assembled component chain.

    Returns
    -------
    list
        The consuming components. Empty means the grid may serve photometry.

    Notes
    -----
    The census sees the ADR-0009 component contract and only that. A reader that
    takes ``sed_nebular`` off ``state.derived`` without declaring an input is
    invisible to it, which is why ``predict_state`` materializes by default
    instead of relying on this list being complete (#1673).
    """
    from tengri.components.nebular.component import NebularSEDComponent
    from tengri.forward.orchestrator import components_consuming

    return [
        c
        for c in components_consuming(chain, "sed_nebular")
        if not isinstance(c, NebularSEDComponent)
    ]


def _chain_consumes(chain, key: str) -> bool:
    """Does any component in ``chain`` declare ``key`` as a (possibly optional) input?

    Reads the declared cross-component contract (ADR-0009), which components expose
    either as ``inputs``/``optional_inputs`` methods returning ``DerivedKey`` tuples
    (bare-Protocol components) or as ``inputs`` dicts (``SEDModelComponent``). Both
    shapes are accepted so a caller never has to know which kind it is holding.

    Parameters
    ----------
    chain: sequence
        The component chain.
    key: str
        Derived-state key, e.g. ``"L_ir"``.

    Returns
    -------
    bool
        True if some component reads ``key``.
    """
    for comp in chain:
        for attr in ("inputs", "optional_inputs"):
            declared = getattr(comp, attr, None)
            if declared is None:
                continue
            try:
                items = declared() if callable(declared) else declared
            except Exception:
                continue
            if not items:
                continue
            names = items if isinstance(items, Mapping) else [getattr(i, "name", i) for i in items]
            if key in names:
                return True
    return False


#: Relative tolerance for the rank-1 check in
#: :meth:`SEDModel._additive_term_band_response`. Two probe draws must reproduce
#: each term's spectral shape to this precision for the term to earn a constant
#: band response. Loose enough not to trip on fp re-association across a ~6000-node
#: grid, ~7 orders tighter than any shape error worth exploiting.
_RANK1_RTOL = 1e-9

# Re-export supporting types for backwards compatibility
__all__ = [
    "MockData",
    "PriorPredictive",
    "SEDModel",
    "SEDModelState",
    "SpectrumPrecomp",
    "WavePrecomp",
]


@dataclasses.dataclass(frozen=True)
class WavePrecomp:
    """Configuration for the ``wave_precomp`` approximation method.

    Pass this to :class:`SEDModel` via ``approx=`` to override the default
    redshift grid used when the model has a free ``redshift`` parameter. The
    SSP × filter integral is precomputed on the wavelength grid; for free
    redshift the result is interpolated through a ``(n_z,)`` table built on
    the same LUT.

    Parameters
    ----------
    n_z: int, default 250
        Number of grid points in the ztable. Higher → finer redshift
        interpolation, slower precompute. Default 250 holds the ztable's *own*
        contribution below 1 % across all bands over z ∈ [0, 1.5] with ~37s
        build overhead (#1134).

        That is a bound on the redshift interpolation alone, **not** on the
        LUT's total error, and the two are not close. Measured on a 12-band
        tsnorm + two-component-dust model against the exact projector, raising
        ``n_z`` 250 → 1000 moves the GALEX FUV error at z = 1.5 by nothing at
        all (10.410 % → 10.402 %), while ``n_subbands=32`` cuts the same number
        ~9× (→ 1.150 %). The dominant term is :attr:`band_integration`, because
        the Lyman break is a step *inside* the bandpass and quadrature converges
        as 1/K² only on smooth integrands. Reach for ``n_z`` to fix a wobble
        along the redshift axis; reach for the band-integration knobs to fix a
        band whose SED has an edge in it.
    z_min: float or None, default None
        Lower bound of the ztable grid. ``None`` → pull from the redshift
        prior with 1 % padding. Ignored when redshift is ``Fixed`` unless
        ``catalog_z_range`` is set.
    z_max: float or None, default None
        Upper bound of the ztable grid. ``None`` → pull from the redshift
        prior with 1 % padding. Ignored when redshift is ``Fixed`` unless
        ``catalog_z_range`` is set.
    catalog_z_range: tuple of float or None, default None
        Catalog-fit reuse knob (Approach A, 2026-05). When set to
        ``(z_min, z_max)``, the ztable mechanism is forced on even when
        ``redshift`` is ``Fixed`` in the spec. The Fixed value is then
        treated as a runtime input to the JIT-compiled forward pass, so
        a single :class:`SEDModel` instance handles a catalog of
        per-galaxy ``Fixed(redshift)`` values **with one compile**
        instead of one compile per row. Compile time amortizes across
        the catalog; runtime cost per fit is the ztable interpolation
        (~µs).

    Examples
    --------
    >>> SEDModel(..., approx=WavePrecomp())  # default ztable sampling
    >>> SEDModel(..., approx=WavePrecomp(n_z=200))  # finer ztable
    >>> SEDModel(..., approx=WavePrecomp(z_min=0.01, z_max=3.0, n_z=200))
    >>>
    >>> # Catalog fit: 10⁴ galaxies at per-galaxy known z.
    >>> sed = SEDModel.build(
    ...     ...,
    ...     redshift=FIXED,  # per-galaxy value supplied by the redshift column
    ...     approx=WavePrecomp(catalog_z_range=(0.05, 1.5), n_z=200),
    ... )
    >>> forward = ForwardModel.build(sed=sed, observation=obs)
    >>> posteriors = Catalog(forward, table, flux_unit="cgs_fnu", redshift_col="z").fit(
    ...     method="map", key=key
    ... )

    Notes
    -----
    This example used to read ``model.fit(row.data, params={"redshift":
    row.z})`` in a Python loop, a documented invocation that raised
    ``TypeError`` until #1384 plumbed ``params=`` through ``SEDModel.fit``
    (it now forwards to the same per-fit override ``ForwardModel.fit``
    takes, and on a ``catalog_z_range`` model the redshift rides
    ``data_args`` as a runtime input, #1316). ``Catalog`` remains the
    taught surface for a table of galaxies with a redshift column: one
    ingest, one validation, one compiled program.

    **Accuracy has an SNR ceiling, not just a percentage** (#1671). The
    LUT's forward photometry bias (measured 0.13-0.26 % on a 4-band
    reference model) is constant in SNR, so no forward check can see it,     but it enters the
    posterior gradient multiplied by SNR: ~5 % relative
    gradient error at SNR 30, ~50 % at SNR 300 on the same model. It is a
    bias, not noise: it moves the posterior mode, and better data makes it
    worse. Fits price this automatically, at run time one exact-vs-LUT
    forward estimates ``max(bias x SNR)`` and a
    :class:`~tengri.config.exceptions.PrecompBiasWarning` fires with the
    number when it is material. For final inference at high SNR, rerun with
    ``approx=None`` (exact path) or compare the two posteriors. The
    spectroscopy analog (:class:`SpectrumPrecomp`) was measured as a
    ~1-sigma posterior shift on a 50-pixel, 5 %-noise fixture (#1688).
    """

    n_z: int = 250
    z_min: float | None = None
    z_max: float | None = None
    catalog_z_range: tuple[float, float] | None = None

    band_integration: str | None = None
    """How the multiplicative dust screen is integrated through each bandpass.

    One of:

    ``"quadrature"`` (default)
        The screen is *evaluated* at :attr:`n_subbands` quadrature nodes per
        band and summed against the sub-band SSP x filter tensors (#1122).
        Converges as 1/K^2. This is the accurate scheme and the one to use
        for science.
    ``"taylor"``
        First-order spectral-moment expansion about the filter effective
        wavelength, ``A(lam_eff)*Phi + A'(lam_eff)*Psi`` (Zacharegkas+2025,
        #617). Retained for reproducing pre-#1122 published results and for
        comparison work. Biases the rest-UV badly, see
        :attr:`taylor_correction`.
    ``"effective_wavelength"``
        Zeroth order, ``A(lam_eff)*Phi``. The cheapest and least accurate.

    ``None`` (the default) resolves to ``"quadrature"``, or to whichever
    scheme the legacy :attr:`n_subbands` / :attr:`taylor_correction` pair
    describes when either was passed explicitly.

    Naming the scheme here is the supported way to pick one. The legacy pair
    selected it only *implicitly*, through an interaction that was easy to get
    wrong in one direction specifically: ``taylor_correction=True`` alone left
    the quadrature on and the flag inert, so a caller who asked for Taylor by
    name silently received quadrature unless they also knew to set
    ``n_subbands=0``.

    Examples
    --------
    >>> WavePrecomp()                                  # quadrature, K=5
    >>> WavePrecomp(band_integration="taylor")         # actually Taylor
    >>> WavePrecomp(band_integration="quadrature", n_subbands=8)
    """

    n_subbands: int | None = None
    """Sub-bands per filter for the multiplicative dust quadrature (#1122).

    The dust screen is *evaluated* at ``n_subbands`` quadrature nodes per band
    rather than extrapolated from a single effective wavelength. Each node is the
    template's own flux-weighted centroid in its sub-band, and both the nodes and
    the sub-band SSP × filter integrals are build-time constants, so the exact
    factorization of the fast path is preserved.

    Converges as 1/K². Worst case over z ≤ 1, τ ≤ 2 (GALEX FUV): K=1 → 8.7 %,
    K=3 → 1.4 %, **K=5 → 0.6 %** (default), K=8 → 0.3 %. Runtime is *cheaper* than
    the Taylor form it replaces (0.93× its gradient at K=5), because that form
    carries a second tensor and a ``pow`` with a traced exponent.

    Defaults to 5 when :attr:`band_integration` is ``"quadrature"``, and to 0
    otherwise. Setting it to ``0`` explicitly is the legacy spelling of
    ``band_integration="effective_wavelength"`` (or ``"taylor"``, if
    ``taylor_correction=True`` accompanies it); prefer naming the scheme.

    Only the *multiplicative* stellar screen needs this. Additive emitters (dust IR,
    radio, X-ray, AGN) factorize exactly through the rank-1/rank-K band response of
    #1107/#1117 and are unaffected. Nebular is not affected either, for the opposite
    reason: it never had a sub-band tensor to size, and since #1738 it does not need
    one, its screen is integrated through the band exactly, from the reddened
    continuum a dusty model materializes anyway. (This said "stellar + nebular" for
    as long as the nebular bucket was in fact screened at a single wavelength per
    band, which is the error #1738 removed.)"""

    taylor_correction: bool | None = None
    """First-order spectral-moment (Ψ) correction to the effective-wavelength dust
    attenuation (Zacharegkas+2025, #617). **Superseded by** ``n_subbands`` and off by
    default since #1122.

    It applies ``A(λ_eff)·Φ + A'(λ_eff)·Ψ``, i.e. it *extrapolates* the attenuation
    linearly away from one point per band. That is fine in the optical/IR and diverges
    in the rest-UV where the curve steepens: the residual is not the ~0.3 % once
    claimed here but **+45 % (z=0.05) to +215 % (z=1)** in GALEX FUV.

    .. deprecated::
        Use ``band_integration="taylor"``. This flag selected the scheme only in
        combination with ``n_subbands=0``; on its own it was silently inert,
        which meant asking for Taylor returned quadrature."""

    fast_dust_emission: bool = False
    """Approximate the dust IR re-emission *band projection* when its exact, fast
    form is unavailable. For a ``modified_blackbody`` with fixed shape (``dust_T``,
    ``dust_beta_ir``, ``dust_epsilon_mbb``) and fixed ``redshift``, the template is
    linear in ``L_ir`` and its per-band integral ``R`` is precomputed once
    (CIGALE ``dl2014`` style): the exact ``L_ir × R`` projection is then used
    automatically, fastest *and* exact, no flag needed. This flag only bites when
    that constant ``R`` cannot be formed (free emission shape / redshift, or
    structured templates): ``True`` then samples the self-normalizing template at
    the filter effective wavelength instead of integrating it through each
    bandpass, much cheaper than the exact per-band integral (#622), at a
    band-shape approximation (smooth on a modified blackbody; up to a few percent
    on a band crossing a steep IR rise or a PAH complex). The energy balance is
    unaffected either way."""

    _VALID_BAND_INTEGRATION: ClassVar[tuple[str, ...]] = (
        "quadrature",
        "taylor",
        "effective_wavelength",
    )

    def __post_init__(self):
        """Resolve the band-integration scheme once, in one place.

        Three source-level defaults used to disagree (this class said K=5 /
        taylor off; ``SEDModel._DEFAULT_APPROX`` said K=0 / taylor on;
        ``sps/precompute.py`` said taylor on), so which scheme ran depended on
        the constructor path taken. Resolving here means every consumer reads
        the same answer, and :attr:`n_subbands` / :attr:`taylor_correction`
        are left holding the concrete values that scheme implies.
        """
        scheme = self.band_integration
        legacy_passed = self.n_subbands is not None or self.taylor_correction is not None

        if scheme is not None:
            if scheme not in self._VALID_BAND_INTEGRATION:
                raise ValueError(
                    f"band_integration={scheme!r} is not a legal value. "
                    f"Choose one of {', '.join(map(repr, self._VALID_BAND_INTEGRATION))}. "
                    "'quadrature' (the default) evaluates the dust screen at "
                    "n_subbands nodes per band and is the accurate choice; "
                    "'taylor' and 'effective_wavelength' are kept for "
                    "reproducing pre-#1122 results and for comparison work."
                )
            # ``n_subbands`` is NOT redundant with an explicit scheme: it sets
            # K for the quadrature, so ``band_integration="quadrature",
            # n_subbands=8`` is correct usage and must stay silent. Warn only
            # for the genuinely meaningless combinations.
            if self.taylor_correction is not None:
                warnings.warn(
                    f"taylor_correction is ignored when band_integration is given "
                    f"explicitly (={scheme!r}). Drop it; the scheme name is the "
                    "selector.",
                    UserWarning,
                    stacklevel=3,
                )
            if self.n_subbands is not None and scheme != "quadrature":
                warnings.warn(
                    f"n_subbands={self.n_subbands} has no effect under "
                    f"band_integration={scheme!r}, it sets the node count for the "
                    "quadrature only.",
                    UserWarning,
                    stacklevel=3,
                )
        else:
            # Derive from the legacy pair, preserving its documented meaning.
            if legacy_passed:
                n_sub = (
                    ApproxPolicy().n_subbands if self.n_subbands is None else int(self.n_subbands)
                )
                taylor = bool(self.taylor_correction)
                if n_sub > 0 and taylor:
                    # The contradiction. Previously resolved silently toward
                    # quadrature, which made the flag read as a choice while
                    # doing nothing.
                    warnings.warn(
                        "taylor_correction=True has no effect while the "
                        f"quadrature is on (n_subbands={n_sub}); the quadrature "
                        "supersedes it. Pass band_integration='taylor' to select "
                        "the Taylor scheme, or n_subbands=0 alongside it for the "
                        "legacy spelling.",
                        UserWarning,
                        stacklevel=3,
                    )
                    scheme = "quadrature"
                elif n_sub > 0:
                    scheme = "quadrature"
                elif taylor:
                    scheme = "taylor"
                else:
                    scheme = "effective_wavelength"
                warnings.warn(
                    "n_subbands / taylor_correction select the band-integration "
                    f"scheme implicitly (resolved here to {scheme!r}). Prefer "
                    f"band_integration={scheme!r}, which says so directly; pass "
                    "n_subbands only to set K for the quadrature.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            else:
                scheme = "quadrature"

        # Write back the concrete values this scheme implies, so every
        # downstream consumer of n_subbands / taylor_correction agrees with
        # band_integration by construction rather than by convention.
        if scheme == "quadrature":
            if self.band_integration is not None and self.n_subbands == 0:
                # Contradictory: a quadrature needs at least one node. Silently
                # substituting K=5 here would be the same class of defect this
                # selector exists to remove, honoring a request nobody made.
                raise ValueError(
                    "band_integration='quadrature' with n_subbands=0 is "
                    "contradictory: the quadrature evaluates the dust screen at "
                    "n_subbands nodes per band, so it needs at least one. Pass "
                    "n_subbands>=1, or band_integration='effective_wavelength' "
                    "if you wanted the single-point form."
                )
            n_sub = 5 if self.n_subbands is None else int(self.n_subbands)
            if n_sub < 1:
                raise ValueError(f"n_subbands must be >= 1 for the quadrature, got {n_sub}.")
            taylor = False
        elif scheme == "taylor":
            n_sub, taylor = 0, True
        else:
            n_sub, taylor = 0, False

        object.__setattr__(self, "band_integration", scheme)
        object.__setattr__(self, "n_subbands", n_sub)
        object.__setattr__(self, "taylor_correction", taylor)


@dataclasses.dataclass(frozen=True)
class SpectrumPrecomp:
    """Configuration for spectrum-grid LUT precomputation.

    Pass this to :class:`SEDModel` via ``approx=`` to enable spectroscopic
    LUT precomputation. The SSP × dust × IGM stack is precomputed at
    spectrum pixel centers (effective wavelengths in the galaxy rest frame)
    and cached per redshift. This is analogous to the photometric
    ``WavePrecomp`` LUT path but for spectroscopy.

    The pixel grid is inherited from ``Observation.spectroscopy.wave_obs``.
    For a free redshift, a redshift table is built so the rest-frame pixel
    grid ``wave_obs / (1 + z)`` can be interpolated at runtime; ``n_z`` /
    ``z_min`` / ``z_max`` tune that table (mirroring :class:`WavePrecomp`).
    When redshift is Fixed, these are ignored.

    Parameters
    ----------
    n_z: int, default 100
        Number of grid points in the free-z redshift table.
    z_min, z_max: float or None
        Bounds of the free-z table. When None, taken from the redshift
        prior with 1% padding. Ignored for fixed redshift.

    Notes
    -----
    Emission lines are not representable by the per-pixel effective-wavelength
    LUT (they are delta-like). When a line-publishing nebular backend (Cue,
    CloudyGrid, CB19, MAPPINGS, Cue-NLR) is present, its discrete line
    luminosities are rasterized onto the pixel grid separately at projection
    time (design-doc option A); the smooth continuum still uses the LUT.

    Examples
    --------
    >>> from tengri import SEDModel, SpectrumPrecomp, WavePrecomp
    >>> SEDModel(..., approx=SpectrumPrecomp())  # spectrum LUT path
    >>> SEDModel(..., approx=SpectrumPrecomp(n_z=200))  # finer free-z table
    >>> # Joint fit, accelerate both channels with independent LUT configs:
    >>> SEDModel(..., approx=(WavePrecomp(n_z=200), SpectrumPrecomp()))
    """

    n_z: int = 100
    z_min: float | None = None
    z_max: float | None = None
    taylor_correction: bool = True
    """Kept for API symmetry with :class:`WavePrecomp`. The spectrum LUT evaluates
    dust attenuation at each pixel wavelength exactly (a pixel is a point, not a
    bandpass), so there is no effective-wavelength residual to correct and this
    flag does not change the spectroscopy result."""


@dataclasses.dataclass(frozen=True)
class FeaturePrecomp:
    r"""Configuration for the nebular precompute (the *feature* LUT path).

    Pass this to :class:`SEDModel` via ``approx=`` to serve the nebular
    calculation from a build-time lookup instead of the per-evaluation forward.
    It composes with :class:`WavePrecomp` (photometry) and
    :class:`SpectrumPrecomp` (spectroscopy).

    .. warning::

       **The name misleads: this is not a line-channel-only optimization.** For
       the Cue backend the grid replaces the *emulator call itself*, so a fit
       with **no line channel at all** benefits, often the most, because a
       photometry-only Cue fit otherwise re-runs Cue on every likelihood
       evaluation. Measured on a 10-parameter Cue model with free ``neb_logU``
       / ``neb_logZ_gas``, against an A/A control whose noise floor was 1.23x:
       a photometry-only fit's compiled MAP step goes 0.645 s to 0.093 s
       (**7x**) on adding this. With a line channel present the same model
       already sits near 0.16 s and neither opt-in resolves at all.
       :class:`WavePrecomp` alone does not resolve either (1.07x, under the
       floor). Measure before assuming either way, and quote a ratio only
       against its own noise floor; see ``docs/dev/api_migration_v0.x.md`` for
       the full grid.

       That a photometry-only fit was *slower* than the same fit with an extra
       data channel was a defect, not a property of the method, #1596, fixed:
       the ``"auto"`` fit policy now attempts this LUT for any photometry-only
       fit whose backend can tabulate, and #1683 extended that to a model built
       with ``approx=WavePrecomp()``, which both fit resolvers had returned
       untouched.

       Passing it explicitly still matters for **prediction**. No fit policy
       reaches ``model.predict_photometry`` / :meth:`SEDModel.predict`, which
       run whatever the build-time ``approx=`` says, so a build-time opt-in is
       what a forward-model benchmark or a mock-generation loop is choosing.
       The converse is the trap: ``Fitter(approx="auto")`` (the default)
       re-resolves the build-time knob, so *fit* arms that differ only in
       ``SEDModel.build(approx=...)`` can be one configuration wearing three
       labels.

    The line wavelengths default to those of ``Observation.line_fluxes``, the
    model already knows which lines it is being fitted against, so the common
    case needs no arguments at all.

    Which machinery is built depends on the nebular backend, because the two
    backends put their lines in physically different places:

    * **Cue** (and any backend publishing discrete line luminosities). Lines are
      linear in the ionizing photon rate, :math:`L_{\rm line} = Q_H \times
      \ell(\theta)`, with :math:`\ell` independent of the SFH *shape*. A grid of
      :math:`\ell` over the free ionization axes is built once (one Cue forward
      per node), and each evaluation reduces to :math:`Q_H \times
      \mathrm{interp}(\text{grid})`. Nothing downstream needs the full-wavelength
      SED, so XLA prunes the stellar einsum entirely.
    * **Baked-in / wNE.** The lines are *inside* the SSP templates, so they must
      be measured off the spectrum. A per-line window LUT of SSP integrals is
      built instead, and the measurement contracts it with SED-free SFH weights
      and the dust screen at the window centers.

    Parameters
    ----------
    n_grid: int or dict, default 16
        Grid points per free ionization axis (Cue backend only; ignored for
        baked-in, whose window LUT has no ionization axes). Denser is tighter.

        A scalar resolves every free axis alike. A dict ``{axis_name: n}``
        resolves them independently, the griddable axes are ``met_logzsol``,
        ``neb_logU`` and ``neb_logZ_gas``, omitted axes take 16, and any other
        key raises rather than being silently ignored. Build cost is the
        *product* over free axes, so per-axis resolution is what keeps a model
        with several free axes affordable: spend points on the axis whose lines
        actually move, not on the one you barely vary.
    lines: array_like or None, optional
        Rest-frame vacuum line wavelengths [Angstrom] to tabulate. ``None``
        (default) takes them from ``Observation.line_fluxes``.
    ranges: dict, optional
        Override ``{param: (lo, hi)}`` grid bounds (Cue only). Defaults to each
        free parameter's prior support.

    Notes
    -----
    **Explicit opt-in, by design.** This is a lossy approximation, so it never
    activates on its own, a model acquires it only because the user asked for
    it. An observation that merely *contains* lines does not switch it on.

    **Accuracy (Cue).** Reconstruction is exact at grid nodes and node-exact
    between them. Balmer lines are recombination lines (:math:`\propto Q_H`) and
    converge quickly. Collisionally excited lines ([OIII], [NII], [SII]) depend
    on the *shape* of the ionizing spectrum, which varies along the metallicity
    axis; validate with a dense sweep strictly inside the grid range, never with
    random draws (they under-sample the structure and report bounds that are
    optimistic by orders of magnitude).

    **Accuracy (baked-in).** The window LUT reconstructs the stellar + dust-screen
    spectrum only, so any component adding rest-frame flux is excluded, with one
    measured exception. A dust-IR component contributes a *smooth* continuum that
    is common to the line window and its sidebands, and so cancels in the
    continuum subtraction: the bias on the measured line flux is :math:`< 10^{-7}`
    even at :math:`\tau_{\rm bc}=4,\ \tau_{\rm diff}=3`, against a 3% contamination
    of the continuum level itself. Dust IR is therefore allowed for lines. It is
    *not* allowed for spectral indices, where a break is a flux **ratio** and a
    smooth additive offset does not cancel.

    **JIT-compatible**: the resulting line prediction is JIT- and gradient-safe;
    the one-time build is eager.

    Examples
    --------
    >>> from tengri import FeaturePrecomp, SEDModel, WavePrecomp
    >>> # lines from the observation, photometry on the LUT path too:
    >>> SEDModel.build(..., approx=(WavePrecomp(), FeaturePrecomp()))
    >>> SEDModel.build(..., approx=FeaturePrecomp(n_grid=24))  # denser Cue grid
    >>> # per-axis: dense where the lines move, coarse where they do not
    >>> SEDModel.build(..., approx=FeaturePrecomp(n_grid={"met_logzsol": 24, "neb_logU": 8}))
    """

    n_grid: int | dict[str, int] = 16
    lines: tuple[float, ...] | None = None
    ranges: dict | None = None

    def __post_init__(self):
        # Validate where the user typed it. The builder validates again at its own
        # entry (it is reachable directly), but by then the traceback points at
        # grid construction rather than at the config (#1311).
        from tengri.components.nebular.nebular_grid_precompute import validate_n_grid

        validate_n_grid(self.n_grid)


@dataclasses.dataclass(frozen=True)
class ApproxState:
    """The effective approximation state of a built model.

    A read-only summary of which build-time look-up tables resolved and
    activated. Read it off any model, :class:`SEDModel` or
    :class:`~tengri.forward.forward_model.ForwardModel`, via the ``approx``
    property, which answers the same question on both:

    >>> model.approx.wave_precomp  # doctest: +SKIP
    True
    >>> print(model.approx)  # doctest: +SKIP
    ApproxState(wave_precomp=True, n_subbands=5)

    This reports what the model *resolved*, not what was requested: a
    :class:`SpectrumPrecomp` that fell back to the exact path because the
    spectral resolution was too high reports ``spectrum_precomp=False``.

    Attributes
    ----------
    wave_precomp: bool
        The SSP x filter photometry LUT is active.
    spectrum_precomp: bool
        The spectrum LUT is active.
    feature_precomp: bool
        The emission-line LUT is active.
    ztable: bool
        Photometry is interpolated through a redshift table (free redshift, or
        a ``catalog_z_range`` reuse window).
    n_subbands: int
        Sub-band samples per filter used by the photometry LUT; ``0`` when the
        LUT is off or unsampled.

    Notes
    -----
    Truthiness reports whether *any* LUT is active, so ``if model.approx:``
    distinguishes an approximated model from an exact one::

        if not model.approx:
            ...  # exact wave-grid model

    Not JIT-relevant: this is Python-side introspection, never traced.
    """

    wave_precomp: bool = False
    spectrum_precomp: bool = False
    feature_precomp: bool = False
    ztable: bool = False
    n_subbands: int = 0

    def __bool__(self) -> bool:
        """Whether any build-time LUT is active."""
        return self.wave_precomp or self.spectrum_precomp or self.feature_precomp

    def __repr__(self) -> str:
        """Show only the active flags, the exact model prints as ``exact``."""
        on = [
            f"{name}={getattr(self, name)!r}"
            for name in ("wave_precomp", "spectrum_precomp", "feature_precomp", "ztable")
            if getattr(self, name)
        ]
        if self.n_subbands:
            on.append(f"n_subbands={self.n_subbands}")
        return f"ApproxState({', '.join(on)})" if on else "ApproxState(exact)"


def _warn_grid_warm_failed(label: str, exc: Exception) -> None:
    """Warn that a grid cache could not be warmed, naming the failure it invites."""
    warnings.warn(
        f"tengri: could not pre-load the {label} grid ({exc}). It will be loaded "
        f"lazily instead, which raises UnexpectedTracerError if the first load "
        f"happens inside a JIT trace.",
        RuntimeWarning,
        stacklevel=3,
    )


#: Free parameters that reach the SED only through a kernel whose derivative
#: rule does not differentiate them, so every gradient backend sees exactly
#: zero. Maps parameter name -> the reason, for the warning below.
#:
#: Empty since #1822, and kept rather than deleted: the check costs nothing and
#: the class of bug recurs (``met_alpha_fe`` raises the same warning from its own
#: site, and #1206/#1764 are the earlier instances).
#:
#: ``agn_kt_warm`` was the sole entry. It reached the SED only via
#: ``_nthcomp_lnu_interp``, whose ``custom_jvp`` supplied a ``gamma`` tangent and
#: discarded the ``kTe`` one, so the rule returned exactly ``0.0`` against a
#: central difference of ~7e41. #1822 added the kTe tangent, and the same
#: measurement now agrees with a converged central difference to 0.6%.
#:
#: The audit that produced #1822 also found the *other* half was worse than
#: documented: reverse-mode ``d/d(agn_gamma_warm)`` (the tangent believed to be
#: working) returned **NaN**, not a number, because the kernel forced a float32
#: output and the cotangent from a realistic ring luminosity (~1e66) overflows
#: float32. That one never appeared here, since a NaN gradient is not a dead one.
_DEAD_GRADIENT_PARAMS: dict[str, str] = {}


def _warn_dead_gradient_params(spec) -> None:
    """Warn when a freed parameter has an identically-zero gradient.

    Reads the **final** free-parameter list rather than any one group's, because
    a group-scoped version of this check would miss exactly the case that
    matters, see #1482, where a guard scoped to its own group never fired.

    Not an error: pinning the parameter is a legitimate configuration and the
    forward model is correct either way. The failure is silent, not wrong, the
    sampler leaves the parameter at its initial value and the posterior returns
    the prior, which reads as a fitted result. Making it loud is the whole fix.
    """
    freed = [name for name in _DEAD_GRADIENT_PARAMS if name in set(spec.free_params)]
    if not freed:
        return

    from tengri.config.exceptions import DeadGradientParameterWarning

    for name in freed:
        warnings.warn(
            f"{name!r} is a free parameter but its gradient is identically zero: "
            f"{_DEAD_GRADIENT_PARAMS[name]}. Every gradient-based backend (MAP, "
            "NUTS, VI) will leave it at its initial value, and the posterior will "
            "report the prior back as though it had been fitted. Pin it with "
            f"Fixed(...), or sample it with a gradient-free method. See #1206.",
            DeadGradientParameterWarning,
            stacklevel=2,
        )


def _warn_agn_dust_double_count(spec) -> None:
    """Warn when composable AGN and Dale2014 ``dust_frac_agn`` both inject AGN IR.

       The composable AGN's ``agn_ir_frac`` (CIGALE-joint tie) and Dale2014's
       embedded quasar template ``dust_frac_agn`` are two distinct AGN surfaces,
       both keyed off the same stellar ``L_absorbed`` (component_factory.py:346,
       ADR-0018 §5, issue #721). With both > 0 the AGN mid/far-IR is double-counted,
    the SKIRTOR/torus block already models AGN IR, so Dale2014's fracAGN should
       be 0 (matching CIGALE's skirtor2016-vs-dale2014-fracAGN choice).

       Value-aware (the structural ``build_components`` guard cannot be): a FREE
       param counts as positive-active; a Fixed param counts only if its value is
       > 0, so ``dust_frac_agn`` pinned to 0 (e.g. a torus-only AGN recipe) does
       not warn. Emits a filterable :class:`AGNDustDoubleCountWarning`.
    """
    if getattr(spec, "dust_emission", None) != "dale2014":
        return
    free = set(spec.free_params)
    fixed = spec.get_fixed_values()

    def _positive_active(name: str) -> bool:
        if name in free:
            return True
        return float(fixed.get(name, 0.0)) > 0.0

    if not (_positive_active("dust_frac_agn") and _positive_active("agn_ir_frac")):
        return

    from tengri.config.exceptions import AGNDustDoubleCountWarning

    warnings.warn(
        "Both AGN surfaces are active: the composable AGN (agn_ir_frac > 0) and "
        "Dale2014 dust emission (dust_frac_agn > 0). Both inject AGN-heated IR "
        "from the same stellar L_absorbed, so AGN mid/far-IR is DOUBLE-COUNTED. "
        "Use one surface: set dust_frac_agn=0 and let the composable AGN torus "
        "(e.g. SKIRTOR) own the AGN IR, recommended when a torus block is "
        "configured, or drop the composable AGN and use Dale2014's embedded "
        "quasar template alone. See ADR-0018 §5 / issue #721. Filter "
        "AGNDustDoubleCountWarning if the overlap is deliberate.",
        AGNDustDoubleCountWarning,
        stacklevel=3,
    )


def _validate_fracagn_requires_dust(spec) -> None:
    """Raise if AGN has fracAGN enabled without a dust component (#944).

    The composable AGN's fracAGN parameter (agn_ir_frac) ties the torus
    luminosity to the dust-absorbed stellar luminosity via the CIGALE
    skirtor2016 energy-balance convention. Without a dust component, the
    absorbed stellar luminosity is ~zero, so the torus (and under
    agn_norm='cigale_joint' the whole AGN) collapses silently.

    This is a build-time safety gate: fracAGN is only safe when paired
    with a dust component (dust_model != 'off').

    Raises
    ------
    ConfigError
        If agn_ir_frac is nonzero (FREE or Fixed>0) and dust_model == 'off'.

    See Also
    --------
    #944 : Silent torus luminosity drop when fracAGN used without dust.
    """
    from tengri.config.exceptions import ConfigError

    # Check if agn_ir_frac (the fracAGN parameter) is active
    free = set(spec.free_params)
    fixed = spec.get_fixed_values()

    def _is_positive_active(name: str) -> bool:
        """Check if a param is FREE or Fixed with value > 0."""
        if name in free:
            return True
        return float(fixed.get(name, 0.0)) > 0.0

    # agn_ir_frac is the lowered name for fracAGN
    if not _is_positive_active("agn_ir_frac"):
        return  # fracAGN is not active, no validation needed

    # Check dust configuration: dust_model='off' means no dust
    dust_model = getattr(spec, "dust_model", "off")

    # A model with dust_model='off' (dust_attenuation={'type': 'none'} or no dust) is unsafe
    if dust_model == "off":
        raise ConfigError(
            "fracAGN (agn_ir_frac) ties the AGN torus luminosity to the "
            "dust-absorbed stellar luminosity (CIGALE skirtor2016 convention). "
            "With dust_attenuation={'type':'none'} or no dust component, the absorbed "
            "stellar luminosity is ~zero and the torus would be silently zeroed. "
            "Fix: either (1) add a dust component "
            "(e.g. dust_attenuation={'type':'two_component'}), "
            "or (2) drop fracAGN and use agn_torus_frac for independent torus scaling. "
            "See issue #944."
        )


def _validate_dale2014_requires_no_sf_radio(spec) -> None:
    """Raise if dale2014 dust emission is combined with SF radio (#1970).

    The Dale+2014 dust emission template (component name 'dale2014') embeds a
    star-forming radio synchrotron continuum rising to 2.2459e9 Å (1.335 GHz).
    The stripped variant 'dale2014_cigale' removes the radio tail beyond
    7.727e7 Å per CIGALE convention.

    When dale2014 is paired with an active SF radio block (radio enabled and
    radio_sfr_mode != 'none'), the synchrotron is double-counted in rest_sed
    between ~1.34 and ~10 GHz (3–22 cm), and the composed SED steps down ~2x at
    the 1.335 GHz template edge (measured slope −4.93 vs. +0.77 expected).

    This is a build-time safety gate: dale2014 is only safe when combined with
    AGN-only radio (radio_sfr_mode='none') or when radio is disabled entirely.
    The remedy: switch to dale2014_cigale, which composes correctly with SF radio.

    Deliberately NOT guarded: the radio component's free-free term (emitted only
    when a nebular component publishes ``log_nion``; no grammar knob controls it)
    overlaps the template's embedded thermal radio at the <~10% level near
    1.4 GHz. Refusing it would block dale2014 + AGN radio + nebular with no
    grammar-reachable remedy, so that overlap is documented on both Dale
    components instead of guarded here.

    Raises
    ------
    ConfigError
        If dust.emission == 'dale2014' AND radio is active with SF synchrotron
        enabled (radio=True and radio_sfr_mode != 'none').

    See Also
    --------
    #1970 : Dale2014 embedded SF radio double-counted when combined with radio block.
    """
    from tengri.config.exceptions import ConfigError

    # Check if dust emission is dale2014 (the radio-bearing variant)
    if getattr(spec, "dust_emission", None) != "dale2014":
        return  # Not dale2014, no guard needed

    # Check if radio is enabled
    if not getattr(spec, "radio", False):
        return  # Radio disabled, no conflict

    # Check if SF synchrotron is active (radio_sfr_mode != 'none')
    # The default for radio_sfr_mode is 'bell2003', so if it exists and is not
    # 'none', we have an active SF radio component
    radio_sfr_mode = getattr(spec, "radio_sfr_mode", "bell2003")
    if radio_sfr_mode == "none":
        return  # SF synchrotron is disabled (AGN-only), no conflict

    # Both conditions met: dale2014 + active SF radio = double-count
    raise ConfigError(
        "The Dale+2014 dust emission template (dust.emission='dale2014') "
        "embeds its own star-forming radio synchrotron continuum to 1.335 GHz. "
        "Combining it with an active SF radio block (radio.sf.type != 'none') "
        "causes double-counting of the radio continuum (~2x in rest_sed "
        "between ~1.34 and ~10 GHz). "
        "Fix: use dust.emission='dale2014_cigale' instead, which has the radio "
        "tail stripped per CIGALE convention and composes correctly with the "
        "radio component. Alternatively, disable SF synchrotron with "
        "radio={'sf': {'type': 'none'}} if you only want AGN radio. "
        "See issue #1970."
    )


def _state_has_content(state) -> bool:
    """Report whether a component state carries anything beyond its name.

    Parameters
    ----------
    state: SEDComponentState or None
        Any component state, of any subclass.

    Returns
    -------
    bool
        True when at least one field other than ``name`` is not ``None``.

    Notes
    -----
    Read off the dataclass fields rather than a hand-written list of attribute
    names. The list this replaced named five (``data``, ``ssp_phot_lut``,
    ``ssp_spec_lut``, ``filter_waves``, ``k_lambda``) and so judged an
    :class:`~tengri.components.igm.component.IGMSEDComponentState` carrying only
    ``band_table`` to be empty -- a state class the list predated. Every such
    list goes stale the moment a component adds a field, and the failure is
    silent: the populated state is discarded and the model still returns
    plausible numbers (#1738).
    """
    if state is None:
        return False
    try:
        fields = dataclasses.fields(state)
    except TypeError:
        # Not a dataclass; fall back to "it exists, so it counts".
        return True
    return any(getattr(state, f.name, None) is not None for f in fields if f.name != "name")


def _fold_igm_into_subbands(igm_comp, stellar_state):
    r"""Fold the IGM transmission into the stellar sub-band quadrature weights.

    The photometry LUT's IGM band factor :math:`\langle T \rangle_f` averages the
    transmission *alone*, unweighted by the spectrum, so it forms
    :math:`\langle S \rangle \langle T \rangle` where the flux needs
    :math:`\langle S T \rangle`. Where :math:`T` varies strongly *inside* a
    bandpass, GALEX FUV at :math:`z \approx 0.8`, where it runs from ~1 to ~0,     that covariance
    term reaches −9.5 %.

    The sub-band quadrature already carries the machinery to fix it: evaluate
    :math:`T` at the same nodes the dust screen uses and multiply it into the
    weights,

    .. math::

        \Phi^{\rm IGM}[m, a, f, k] = \Phi[m, a, f, k]\;
        T_{\rm IGM}\!\left(\lambda^*[m, a, f, k]\,(1 + z),\; z\right)

    Both factors are build-time constants, so the product is too and the runtime
    einsum is unchanged in shape and cost, the correction is free.

    The fold happens on the **metallicity axis**, before the SSP grid is
    contracted, and that is load-bearing. The node published at runtime is a
    met-weighted average whose weights move with the free ``met_logzsol``, so
    :math:`T` at "the node" is a function of :math:`(z, Z)`, not of :math:`z`
    alone: across the SSP grid the node shifts by up to 68 % of a sub-band width
    and :math:`T` there by up to 1.3 % in GALEX FUV. Folding first is exact.

    Parameters
    ----------
    igm_comp: IGMSEDComponent
        Supplies :meth:`~IGMSEDComponent.subband_node_transmission` and the gate.
    stellar_state: StellarSEDComponentState
        Carrying the fixed-z photometry LUT or the free-z z-table.

    Returns
    -------
    StellarSEDComponentState
        With the IGM-folded sub-band tensor attached, or ``stellar_state``
        unchanged when there is nothing to fold.

    Notes
    -----
    **JIT-compatible**: build-time only.

    Returns the state untouched when the sub-band quadrature is off
    (``WavePrecomp(n_subbands=0)``) or when the transmission is not a function of
    :math:`(\lambda, z)` alone, patchy reionization and DLAs read free
    parameters. Those keep the live per-call evaluation, so the gate fails safe.
    """
    from dataclasses import replace as _replace

    if stellar_state is None:
        return stellar_state

    lut = getattr(stellar_state, "ssp_phot_lut", None)
    if lut is not None and lut.ssp_subband_phot is not None:
        trans = igm_comp.subband_node_transmission(lut.ssp_subband_waves_rest, [lut.redshift])
        if trans is None:
            return stellar_state
        return _replace(
            stellar_state,
            ssp_phot_lut=lut._replace(ssp_subband_phot_igm=lut.ssp_subband_phot * trans),
        )

    ztable = getattr(stellar_state, "ssp_phot_ztable", None)
    if ztable is not None and ztable.ssp_subband_phot_table is not None:
        trans = igm_comp.subband_node_transmission(ztable.subband_waves_rest_table, ztable.z_grid)
        if trans is None:
            return stellar_state
        return _replace(
            stellar_state,
            ssp_phot_ztable=ztable._replace(
                ssp_subband_phot_igm_table=ztable.ssp_subband_phot_table * trans
            ),
        )

    return stellar_state


#: Accepted ``csp_integration`` values. All are equivalent (#1500): the stellar
#: component integrates the CSP with cloud-in-cell age weights for every
#: configuration, so none of these reaches the SED. Kept so existing calls keep
#: working; non-default values warn and are slated for removal in v1.0.
_VALID_CSP_INTEGRATION = ("trapz", "log_trapz", "log_interp", "dsps_native", "dsps_met_table")
_DEFAULT_CSP_INTEGRATION = "trapz"


@functools.cache
def _init_keywords(cls: type) -> frozenset[str]:
    """The keywords ``cls.__init__`` accepts, everything else is grammar input.

    Derived rather than hand-listed, on the #1720 principle: a second copy of a
    census agrees with the first by convention and nothing else. ``build`` uses
    this to decide what may be forwarded to the constructor, so a hand-maintained
    copy drifting from the real signature is exactly the failure it prevents.
    """
    params = inspect.signature(cls.__init__).parameters
    return frozenset(params) - {"self", "spec", "ssp_data"}


class SEDModel:
    """Differentiable SED forward model with modular physics and clean API.

    The forward model maps physical parameters (stellar mass, SFH, metallicity,
    dust, AGN, etc.) to observables: photometry, spectrum, and derived SED
    quantities. Internally, it decomposes the SED pipeline into independent
    physics modules (stellar populations, star formation history, dust,
    nebular, AGN, IGM) that are composed into prediction kernels at
    initialization time, enabling fast inference and flexibility in model
    configuration.

    The SFH is computed via a registry-driven composed function that handles
    additive smooth models, burst mixture, and correlated-field (GP) modulation
    in a single call. Three prediction modes (compositional, hybrid, exact) trade
    accuracy for speed, with automatic fallback.

    Parameters
    ----------
    spec: Parameters
        Parameter specification from ``tengri.Parameters``. Defines
        free/fixed parameters and their priors.
    ssp_data: SSPData
        Pre-loaded SSP templates (from ``load_ssp_data()``). Contains
        absolute SSP grid in ``log10(Z)`` absolute, age array, and
        optional mass-remaining tables for stellar mass surviving
        constraints.
    filters: list or tuple, optional
        Filter transmission curves for photometric prediction. Accepts either:

        - 3-tuple from :func:`load_filter_set`: ``(filter_waves, filter_trans, filter_curves)``
        - List of :class:`FilterCurve` namedtuples

        If provided, enables photometry prediction and automatic precomputation
        at initialization. Either ``filters`` or ``observation`` may be passed,
        not both.
    observation: Observation, optional
        Unified observation config (photometry + spectroscopy + emission lines).
        Mutually exclusive with ``filters``.
    precompute: bool, optional
        **Legacy / largely superseded.** Builds the pre-``approx=``-era
        ``PrecomputedData`` container (fixed-z SSP photometry/spectroscopy grid
        defaults). This predates, and is NOT, the fast LUT path: the
        Zacharegkas+2025 fast-photometry / spectroscopy speedup is selected at
        build time via ``approx=WavePrecomp()`` / ``approx=SpectrumPrecomp()``
        (see ``approx`` below), which builds its own LUTs through the component
        chain and does not consume ``PrecomputedData``. ``precompute`` now only
        supplies a couple of grid fallbacks for the exact path and is scheduled
        to be folded into ``approx`` (tracked in the precompute-naming cleanup
        issue). Default True; leave it unless you know you need the legacy
        container. Set False to skip building it.
    forward_dtype: str or jnp.dtype, optional
        Dtype for forward model computation. Default ``"float64"``.

        .. deprecated:: 2026-07

           **Retired (#1433).** Passing anything but ``"float64"`` emits a
           ``DeprecationWarning``, a warning, not an exception; the call
           proceeds, and does nothing else. ``"float32"`` cast
           nothing and changed nothing, measured bit-for-bit identical
           photometry to ``"float64"`` on both the exact and the ``WavePrecomp``
           path. It no longer enters :meth:`compile_signature` either, so it no
           longer costs the second compile of an identical kernel that it used to.

           Retired rather than repaired because there is no second float32 mode
           worth maintaining: pure float32 is what the range protections in
           ``components/`` gate on, it is what #1206 delivers, and the knob sat
           dead for two months without a single report. Wiring it instead would
           mean reviving a distinct mixed-precision path with its own gate
           semantics and re-earning a speed claim that was never re-measured.

           The casts it names were real until ``1e57d973d`` (2026-05-20) deleted
           ``forward/_kernels/``; the kwarg, this docstring, the state field and
           the signature entry survived that refactor, and the six casts did not.
           This description previously promised "halves memory and gives ~1.5x
           speedup with <0.1% accuracy loss", none of which has held since.

           For float32 today use **pure** float32: enter a
           ``jax.enable_x64(False)`` context. That is a different mechanism (it
           changes JAX's default dtype rather than casting captured arrays), it is
           the mode the float32 range protections in ``components/`` gate on, and
           it is the one #1206 is making work end to end.

           The argument is still accepted so that existing callers keep working,
           nothing they compute was ever different, and it will be removed once
           the warning has been in a release.

        Independently of this knob, the multiplicative flux/distance seams
        (photometry, spectrum, line flux projections) apply cosmological factors as
        range-safe log offsets (see :mod:`tengri.utils.scale`), so float32 arrays do
        not materialize out-of-range intermediates there.
    approx: dict or bool, optional
        Control which approximations enter the component chain. Default True enables
        all approximations (fastest). False disables all (forces exact path
        everywhere). A dict enables selective control:

        - ``"ztable"``: SSP × filter lookup table indexed on redshift grid (True, default)
        - ``"wave_precomp"``: SSP × filter lookup table on fixed wavelength grid (False, default)

        Approximation dependencies (resolved at build time):

        - ``wave_precomp=True`` with free redshift auto-enables ``ztable=True``.
        - ``ztable=True`` requires ``wave_precomp=True``.
        - Unknown flag names raise ``ValueError`` with list of legal flags.

    compile: str, optional
        JIT-wrapping strategy for the forward pass. Default ``"per_component"``
        wraps each :class:`SEDComponent.apply` independently for faster cold-starts
        in notebooks; ``"fused"`` compiles the entire ``observation.predict ∘
        run_components`` chain at once for hot inference loops; ``"auto"`` is a
        stub that currently resolves to ``"per_component"``.

        **Legal values:** ``"per_component"`` (default), ``"fused"``, ``"auto"``.
        Invalid values raise ``ValueError``.

    csp_integration: str, optional
        **Deprecated and inert** (#1500); accepted so existing calls keep
        working, and slated for removal in v1.0. Any non-default value raises a
        :class:`DeprecationWarning`. Accepted: ``"trapz"`` (default),
        ``"log_trapz"``, ``"log_interp"``, ``"dsps_native"``,
        ``"dsps_met_table"``.

        .. warning::

           **No value changes any output.** The stellar component integrates the
           CSP with the age-weight kernel named by ``sfh={'age_kernel': ...}``
           (``components/stellar/component.py``), which this argument does not
           feed, so it never reached the SED under any configuration.
           ``predict_photometry`` is bit-identical across all five values;
           measured, not assumed.

           It formerly changed ``_predict_sfh_quantities`` alone, which meant the
           *reported stellar mass came from a different integration than the
           spectrum it was fitted to*: 0.32% off for ``"log_interp"`` and
           ``"dsps_native"``, and **NaN** for ``"dsps_met_table"``, a NaN that
           :class:`~tengri.inference.posterior.Posterior` then vmapped over every
           sample. Derived quantities are now computed from the SED's own age
           weights, so every value agrees to machine precision.

           To change how the CSP is integrated, use the knob that does reach it:
           ``sfh={'age_kernel': 'cic' | 'dsps'}`` (#964). ``tengri.list_age_kernels()``
           is the live menu. Unlike this argument, that one measurably moves the
           SED, 0.19% across SDSS *ugriz* for a double-power-law history.

    Attributes
    ----------
    observation: Observation or None
        Attached observation object containing photometry and/or spectroscopy
        configuration. Set by constructor if filters or observation= passed.
    spec: Parameters
        Parameter specification defining all free/fixed parameters and their priors.
    ssp_data: SSPData
        Pre-loaded stellar population synthesis templates (from ``load_ssp_data()``).
    config: ModelConfig
        Frozen model configuration (immutable after init).

    Notes
    -----
    **JIT-compatible**: yes, all prediction methods (except
    :meth:`predict` for lazy evaluation) are fully JAX differentiable
    and can be called inside :func:`jax.jit` and :func:`jax.vmap`.

    **Gradient-safe**: yes, all physical parameters are differentiable
    for inference via HMC, VI, and score-based methods.

    **Approximation scheme**: All prediction methods route through the
    single JIT-safe orchestrator :meth:`predict_observables_jit` (the
    SSP grid is threaded as a JIT runtime input). Historical mode-cascade
    strategies (compositional / hybrid / exact) collapsed into this one
    path in 2026-05; the orchestrator itself remains XLA-fused
    and bit-exact for the configured ``approx=`` policy.

    **Physical units** (internal):

    - Time: years (yr). User-facing API converts to Myr/Gyr.
    - Wavelength: Angstrom (Å).
    - Luminosity (SED components): erg/s/Hz (L_ν).
    - Luminosity (photometry): erg/s/cm²/Hz (f_ν).
    - Metallicity (SSP grid): log₁₀(Z) absolute. User API uses log₁₀(Z/Z☉).
    - AGN bolometric luminosity: log₁₀(L_bol/L☉) at API level.

    **IGM absorption gotcha**: :meth:`predict_obs_sed` applies IGM transmission
    at observed-frame wavelengths (input to ``igm_transmission()`` is redshifted).
    This is automatic when ``igm=True`` in spec.

    References
    ----------
    .. [1] A. Zacharegkas et al., "Fast Photometry with Precomputed
       Stellar Population Grids," ApJ, (2025).
    .. [2] S. Cooray et al., "Forward Model for Differentiable SED Fitting
       with Correlated SFH," (2026).

    Examples
    --------
    Standard photometric fit with DPL SFH::

        from tengri import SEDModel, Parameters, Uniform, load_ssp_data, Photometry

        ssp = load_ssp_data("data/ssp_miles.h5")
        phot = Photometry.from_names(["sdss_r", "sdss_i", "sdss_z"])
        spec = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
        )
        model = SEDModel(spec, ssp, observation=phot)
    """

    # ── SubModel Protocol surface ──────────────────────────────────────
    # See docs/dev/archive/forward-model-architecture.md §4. SEDModel directly
    # satisfies tengri.protocols.SubModel; ForwardModel's per-population
    # orchestration consumes the `run` and `declared_parameters` methods.

    name: str = "sed"

    # Default approximation settings (immutable, used as template only);
    # the settings themselves are owned by the components.
    # "wave_precomp" = SSP × filter LUT on fixed wavelength grid (stellar component)
    # "ztable" = SSP × filter LUT indexed on redshift grid, requires wave_precomp
    # ztable is auto-enabled when wave_precomp=True and redshift is free.
    # "igm" = pre-compute IGM transmission at filter effective wavelengths for
    # the hybrid kernel at fixed z. Default True matches the historic behavior
    # before the ``approx=`` flag was introduced (``_build_precomputed_data``
    # always computed ``igm_eff`` when ``_uses_igm`` and ``_z_fixed`` were set).
    # Structural switches only. The band-projection knobs are NOT written out
    # here, they are read off a default-constructed :class:`WavePrecomp`, which
    # owns them.
    #
    # They used to be a second, hand-maintained copy, and the copy disagreed:
    # this dict said ``taylor_correction=True, n_subbands=0`` while WavePrecomp
    # said ``False, 5``. That divergence is not cosmetic, it is a silent
    # accuracy change, and it shipped once. Before the ``or WavePrecomp()``
    # fallback below existed, ``approx=SpectrumPrecomp()`` on a joint
    # observation reached the projector with a live photometry LUT and picked
    # these values up, so the photometry silently ran the pre-#1122
    # effective-wavelength path.
    #
    # Deriving them means the two cannot drift apart again. Pinned by
    # ``tests/regression/bug/test_band_integration_is_explicit.py``.
    _DEFAULT_APPROX: ClassVar[ApproxPolicy] = ApproxPolicy()

    #: Maximum spectral resolution R = λ/Δλ for which the SpectrumPrecomp
    #: per-pixel effective-wavelength LUT is trusted. Above this the model
    #: falls back to the exact wave-grid path with a warning.
    _SPECTRUM_PRECOMP_R_MAX: ClassVar[float] = 3000.0

    @staticmethod
    def _max_spectral_resolution(observation) -> float | None:
        """Return the maximum spectral resolution R across the spectrum, or None.

        Reads ``observation.spectroscopy.resolution`` (scalar or per-pixel
        array). Returns ``None`` when no spectroscopy / resolution is set.
        """
        if observation is None or not getattr(observation, "can_do_spectroscopy", False):
            return None
        spectroscopy = getattr(observation, "spectroscopy", None)
        resolution = getattr(spectroscopy, "resolution", None) if spectroscopy else None
        if resolution is None:
            return None
        return float(jnp.max(jnp.asarray(resolution)))

    def _resolve_wave_precomp(self, cfg) -> None:
        """Activate the photometry SSP × filter LUT for one ``WavePrecomp`` config.

        The WavePrecomp LUT bakes the filter-convolution convention at build time
        (ADR-0017). Only the photon-counting Bessell weight is threaded through the
        component preintegration so far; refuse the energy convention rather than
        silently producing Bessell fluxes. Shared by the single-object and the
        composite ``(WavePrecomp, SpectrumPrecomp)`` paths (#610).
        """
        _phot = getattr(self.observation, "photometry", None)
        _conv = getattr(_phot, "convention", FilterConvention.BESSELL)
        if _conv != FilterConvention.BESSELL:
            raise NotImplementedError(
                f"approx=WavePrecomp(...) currently supports only the photon-counting "
                f"'bessell' convention, got convention={_conv!r}. Use the exact path "
                f"(approx=None) for the energy/CIGALE convention."
            )
        self._approx = self._approx.replace(wave_precomp=True)

    def _resolve_spectrum_precomp(self, cfg, observation) -> None:
        """Activate the per-pixel spectrum LUT for one ``SpectrumPrecomp`` config.

        Valid for low-to-medium R, where the continuum is smooth across the pixel
        kernel. At high R the kernel is narrower than the spectral features the
        approximation assumes are flat, so fall back to the exact wave-grid path
        (with a clear warning) rather than silently returning a biased spectrum.
        Line-publishing nebular backends (Cue, CloudyGrid, …) are supported: their
        discrete ``line_waves``/``line_lums`` are grid-independent and survive the
        LUT path. Sets :attr:`_approx_config_spec` to the config (or ``None`` on the
        R-fallback). Shared by the single-object and composite paths (#610).
        """
        r_max = self._max_spectral_resolution(observation)
        if r_max is not None and r_max > self._SPECTRUM_PRECOMP_R_MAX:
            import warnings

            warnings.warn(
                f"approx=SpectrumPrecomp() requested but the spectrum "
                f"resolution R≈{r_max:g} exceeds the SpectrumPrecomp limit "
                f"of {self._SPECTRUM_PRECOMP_R_MAX}. The per-pixel "
                f"effective-wavelength LUT is inaccurate near spectral "
                f"features at this resolution, so the model falls back to "
                f"the exact wave-grid path (no speed-up). Use approx=None "
                f"to silence this, or down-bin the spectrum.",
                stacklevel=3,
            )
            self._approx = self._approx.replace(spectrum_precomp=False)
            self._approx_config_spec = None
        else:
            self._approx = self._approx.replace(spectrum_precomp=True)
            self._approx_config_spec = cfg
            # #1166: the SpectrumPrecomp LUT point-interpolates the SSP onto the
            # pixel grid at build time, so it does NOT honor a flux-conserving
            # resample. Warn rather than silently ignore the request, the exact
            # path (approx=None) carries the conserving low-resolution fix.
            spectro = getattr(observation, "spectroscopy", None)
            if spectro is not None and getattr(spectro, "resample", "point") != "point":
                import numpy as _np

                if spectro.resolve_conserving(_np.asarray(self.ssp_data.ssp_wave)):
                    import warnings

                    warnings.warn(
                        f"resample={spectro.resample!r} requests a flux-conserving "
                        f"resample, but approx=SpectrumPrecomp() point-interpolates the "
                        f"SSP onto the pixel grid and does not apply it. Use approx=None "
                        f"for the flux-conserving low-resolution spectrum (#1166).",
                        stacklevel=3,
                    )

    # ── Construction ──────────────────────────────────────────────────

    def __init__(
        self,
        spec,
        ssp_data,
        filters=None,
        observation=None,
        precompute=True,
        forward_dtype="float64",
        approx=None,
        csp_integration="trapz",
        wave_chunk_size=None,
        agn_config=None,
        strategy=None,
        compile=None,
    ):
        # ``strategy`` is accepted for backwards-compat signature but ignored,         # the
        # kernel-selection strategy machinery was removed in 2026-05
        # (kernel adapters deleted). ``predict_observables_jit`` is the only
        # forward path now.
        del strategy
        self._agn_config = agn_config
        # ── Observation ────────────────────────────────────────────
        observation, spec = self._init_observation(spec, filters, observation)
        self.observation = observation
        self.spec = spec
        self.ssp_data = ssp_data
        self._forward_dtype = jnp.dtype(forward_dtype)
        if self._forward_dtype != jnp.dtype("float64"):
            # Retired, not merely undocumented (#1433). The knob has cast nothing
            # since ``1e57d973d`` deleted ``forward/_kernels/`` (2026-05-20), so
            # accepting it silently hands the caller float64 arithmetic under a
            # float32 name. Warn rather than raise: it is inert, so no result
            # changes either way, and a hard error would break callers for whom
            # nothing was ever different.
            warnings.warn(
                f"forward_dtype={str(self._forward_dtype)!r} is ignored and has been "
                "since 2026-05-20 (#1433): it casts nothing, returns bit-identical "
                "results to float64, and only costs an extra compile because it "
                "still enters the model's cache key. For float32, run inside a "
                "`with jax.enable_x64(False):` context, that is the mechanism the "
                "float32 range protections in components/ are written against.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._wave_chunk_size = wave_chunk_size

        # ── Observables NamedTuple (synthesized per model) ───────
        from tengri.observation.observables import build_observables_class

        self._Observables = (
            build_observables_class(self.observation) if self.observation is not None else None
        )

        # ── Compile mode + Approximation settings ─────────────────
        # Validate compile= kwarg
        if compile is None:
            compile = "per_component"
        if compile not in ("per_component", "fused", "auto"):
            legal = "per_component, fused, auto"
            raise ValueError(f"compile={compile!r} is illegal. Legal values: {legal}.")
        self._compile_mode = compile

        # Resolve and validate approximation kwarg.
        # Contract (2026-05-20):
        #   * ``approx=None`` (default): exact wave-grid integration.
        #   * ``approx=WavePrecomp(...)``: opt into the precomputed
        #     SSP × filter LUT path. ``WavePrecomp()`` gives the default
        #     ztable sampling; ``WavePrecomp(n_z=200, z_min=0.0, z_max=3.0)``
        #     for custom grids.
        # Dict / bool / string forms (the pre-3d surface) are rejected.
        # ``_approx_config`` is the primary z-grid/Taylor/catalog config (the
        # WavePrecomp when present, else the SpectrumPrecomp); ``_approx_config_spec``
        # holds the SpectrumPrecomp when a composite ``(WavePrecomp, SpectrumPrecomp)``
        # tuple is passed so a joint fit can accelerate BOTH channels (#610).
        self._approx_config: WavePrecomp | SpectrumPrecomp | None = None
        self._approx_config_spec: SpectrumPrecomp | None = None
        self._approx_config_wave: WavePrecomp | None = None
        self._approx_config_feature: FeaturePrecomp | None = None
        if approx is None:
            self._approx = self._DEFAULT_APPROX
            self._approx = self._approx.replace(wave_precomp=False)
            self._approx = self._approx.replace(ztable=False)
        else:
            # Accept a single config or a composite ``(WavePrecomp, SpectrumPrecomp,
            # FeaturePrecomp)`` sequence (order-independent, at most one of each).
            configs = list(approx) if isinstance(approx, (tuple, list)) else [approx]
            wave_cfgs = [c for c in configs if isinstance(c, WavePrecomp)]
            spec_cfgs = [c for c in configs if isinstance(c, SpectrumPrecomp)]
            feat_cfgs = [c for c in configs if isinstance(c, FeaturePrecomp)]
            known = (WavePrecomp, SpectrumPrecomp, FeaturePrecomp)
            unknown = [c for c in configs if not isinstance(c, known)]
            if (
                unknown
                or len(wave_cfgs) > 1
                or len(spec_cfgs) > 1
                or len(feat_cfgs) > 1
                or not configs
            ):
                raise TypeError(
                    f"approx={approx!r} is not a legal value. Legal forms: "
                    "None (default, exact wave-grid), "
                    "WavePrecomp() for the SSP × filter LUT path, "
                    "SpectrumPrecomp() for the spectrum LUT path, "
                    "FeaturePrecomp() for the emission-line LUT path, or a composite "
                    "tuple such as (WavePrecomp(...), FeaturePrecomp()) for a joint "
                    "photometry + line fit (at most one of each). The pre-3d dict / "
                    "bool / string forms (e.g. approx={'wave_precomp': True}, "
                    "approx=True, approx='wave_precomp') were removed."
                )
            self._approx = self._DEFAULT_APPROX
            if wave_cfgs:
                self._resolve_wave_precomp(wave_cfgs[0])
            if spec_cfgs:
                self._resolve_spectrum_precomp(spec_cfgs[0], observation)
            self._approx_config_wave = wave_cfgs[0] if wave_cfgs else None
            self._approx_config_feature = feat_cfgs[0] if feat_cfgs else None
            # Primary config: WavePrecomp drives the shared z-table / catalog
            # knobs when present (back-compat); else the SpectrumPrecomp (but only
            # when it actually activated, an R-fallback leaves it None).
            if wave_cfgs:
                self._approx_config = wave_cfgs[0]
            else:
                self._approx_config = self._approx_config_spec

        # Thread the photometry LUT's band-projection knobs into the approx dict,
        # so the stellar precompute knows what to build and predict_via_precomp
        # what to apply.
        #
        # These are *photometry* knobs: they correct the effective-wavelength
        # approximation of a bandpass. A spectrum pixel is a point, not a
        # bandpass, so SpectrumPrecomp has no meaningful value for any of them.
        # Source them from a WavePrecomp, never from whichever object the caller
        # happened to pass. On a joint observation ANY opt-in promotes photometry
        # onto the LUT, so ``approx=SpectrumPrecomp()`` reached this code with a
        # live photometry LUT and no ``n_subbands`` field; the old
        # ``getattr(cfg, "n_subbands", 0)`` then fell back to 0, not WavePrecomp's
        # default of 5, but the sentinel that *disables* the quadrature, and
        # picked up its ``taylor_correction=True``. The photometry silently ran the
        # pre-#1122 effective-wavelength path: several percent out in the rest-UV.
        if self._approx_config is not None:
            screen = self._approx_config_wave or WavePrecomp()
            # ``screen`` has already resolved these into mutual agreement in
            # ``WavePrecomp.__post_init__``; copy them as a SET, keyed off the
            # same tuple the defaults are built from. Copying field-by-field is
            # how a knob gets forgotten here and silently keeps a default that
            # contradicts the others.
            self._approx = self._approx.replace(
                **{key: getattr(screen, key) for key in BAND_PROJECTION_KEYS}
            )

        # Part A (joint precompute): on a joint photometry+spectroscopy
        # observation, any precompute opt-in builds BOTH LUT families. The
        # approx object's *type* historically selected the family
        # (WavePrecomp→photometry, SpectrumPrecomp→spectroscopy); for a joint
        # model we promote to both so the forward pass projects photometry via
        # ``predict_via_precomp`` AND spectroscopy via
        # ``predict_spectrum_via_precomp`` (the components publish both LUT
        # families in one pass). z-grid knobs (n_z/z_min/z_max) come from
        # whichever object was passed and apply to both families.
        if (
            observation is not None
            and getattr(observation, "is_joint", False)
            and (self._approx.get("wave_precomp") or self._approx.get("spectrum_precomp"))
        ):
            self._approx = self._approx.replace(wave_precomp=True)
            self._approx = self._approx.replace(spectrum_precomp=True)

        # Free-redshift ztable auto-extension. ``ztable`` is an internal
        # extension of ``wave_precomp`` (free-z interpolation on the same LUT),
        # not a user flag, it switches on transparently when the method is
        # ``wave_precomp`` and redshift is free.
        #
        # Catalog-fit override (Approach A, 2026-05): when the astronomer
        # passes ``WavePrecomp(catalog_z_range=(z_min, z_max))``, force the
        # ztable mechanism even when redshift is Fixed in the spec. The
        # forward pass then reads ``params["redshift"]`` at runtime, so a
        # single SEDModel handles many per-galaxy ``Fixed(z)`` values
        # without recompiling. See ``docs/dev/cross-compile-fixed-z-design.md``.
        self._catalog_z_range: tuple[float, float] | None = None
        if self._approx["wave_precomp"]:
            redshift_dist = spec.get_distribution("redshift")
            # ``catalog_z_range`` is a WavePrecomp-only knob, so read it from the
            # WavePrecomp slot rather than from the primary config, which under a
            # joint model may be a SpectrumPrecomp. A ``getattr(cfg, ..., None)``
            # here would fail *open*, silently returning the default for a knob
            # the caller may well have set, which is exactly how the sub-band
            # quadrature above went missing.
            cz = self._approx_config_wave.catalog_z_range if self._approx_config_wave else None
            if cz is not None:
                if redshift_dist.is_fixed:
                    self._approx = self._approx.replace(ztable=True)
                    self._catalog_z_range = (float(cz[0]), float(cz[1]))
                # Free-redshift case: catalog_z_range is harmless (ztable already on)
                # but record it so the compile_signature still distinguishes ranges.
                else:
                    self._approx = self._approx.replace(ztable=True)
                    self._catalog_z_range = (float(cz[0]), float(cz[1]))
            elif not redshift_dist.is_fixed:
                self._approx = self._approx.replace(ztable=True)

        # ── Stellar populations ───────────────────────────────────
        self._init_ssp(spec, ssp_data, csp_integration)

        # ── Collect parameter map deltas from each _init_* method ──
        param_map_deltas = []

        # ── Star formation history ────────────────────────────────
        param_map_deltas.append(self._init_sfh(spec))

        # ── Metallicity ───────────────────────────────────────────
        param_map_deltas.append(self._init_metallicity(spec))
        self._validate_metallicity_bounds(spec, ssp_data)
        self._validate_alpha_fe_identifiability(spec, ssp_data)

        # ── Dust (attenuation + emission) ─────────────────────────
        param_map_deltas.append(self._init_dust(spec))

        # ── IGM + DLA ─────────────────────────────────────────────
        self._init_igm(spec)

        # ── Nebular emission ──────────────────────────────────────
        param_map_deltas.append(self._init_nebular(spec, ssp_data))

        # ── AGN ───────────────────────────────────────────────────
        param_map_deltas.append(self._init_agn(spec))

        # ── Multiwavelength (radio, X-ray, shock) ─────────────────
        param_map_deltas.append(self._init_multiwavelength(spec, ssp_data))

        # ── Instrument (velocity dispersion, LSF) ─────────────────
        self._init_instrument(spec, observation)

        # ── Observation calibration coefficients ──────────────────
        # ``cal_c1..cN`` are dynamic, their count is the spectroscopy
        # ``calibration_order``, so, unlike the static noise params, they cannot
        # be declared in a ``components/*/_params.py`` that ``_build_param_map``
        # auto-derives. They are consumed as-is by the calibration polynomial, so
        # register plain identity mappings here (#1031: previously the auto-merged
        # ``cal_c*`` were free in the spec with no map entry → ParameterMapError).
        param_map_deltas.append(self._calibration_param_map(observation))

        # ── Cosmology (luminosity distance) ───────────────────────
        self._init_cosmology(spec)

        # ── Validate and freeze parameter map ─────────────────────
        self._validate_and_freeze_param_map(param_map_deltas)

        # ── Warm HDF5 grid caches BEFORE any JIT compilation ──────
        # (tracer-leak prevention; formerly the side-effect at the top of the
        # retired ``_build_precomputed_data``, #620). ``precompute`` is now an
        # accepted-but-ignored legacy kwarg, the fast path is opt-in via
        # ``approx=WavePrecomp()`` / ``approx=SpectrumPrecomp()``.
        del precompute
        self._warm_grid_caches()

        # ── Frozen runtime bundle for kernel layer (built BEFORE kernels) ──
        self._state = SEDModelState(
            spec=self.spec,
            ssp_data=self.ssp_data,
            filter_waves=self.filter_waves,
            filter_trans=self.filter_trans,
            rest_wavelength=self._rest_wavelength,
            log_age_grid=self.log_age_grid,
            age_yr=self.age_yr,
            d_log_age=self.d_log_age,
            n_grid=self._n_grid,
            ssp_log_ages_yr=self.ssp_log_ages_yr,
            ssp_ages_yr=self.ssp_ages_yr,
            csp_matrix=self._csp_matrix,
            csp_age_dt=self._csp_age_dt,
            csp_integration=self._csp_integration,
            forward_dtype=self._forward_dtype,
            met_interp=self._met_interp,
            met_mode=self._met_mode,
            z_interp=self._z_interp,
            lgmet_scatter=self._lgmet_scatter,
            sfh_fn=self._sfh_fn,
            sfh_internal_names=self._sfh_internal_names,
            uses_stochastic_sfh=self._uses_stochastic_sfh,
            gp_kernel=self._gp_kernel,
            dust_model=self._dust_model,
            dust_law_bc=self._dust_law_bc,
            dust_law_diff=self._dust_law_diff,
            dust_law_bc_fn=self._dust_law_bc_fn,
            dust_law_diff_fn=self._dust_law_diff_fn,
            dust_emission_model=self._dust_emission_model,
            nebular_backend=self._nebular_backend,
            agn_model=self._agn_model,
            agn_config=getattr(self, "_agn_config", None),
            agn_luminosity_mode=self._agn_luminosity_mode,
            uses_igm=self._uses_igm,
            uses_radio=self._uses_radio,
            uses_xray=self._uses_xray,
            radio_include_freefree=getattr(self, "_radio_include_freefree", None),
            radio_sfr_mode=getattr(self, "_radio_sfr_mode", None),
            radio_agn_model=getattr(self, "_radio_agn_model", None),
            z_fixed=self._z_fixed,
            dl_cm_fixed=self._dl_cm_fixed,
            param_map=self._param_map,
            igm_fn=self._igm_fn,
        )

        # Eagerly build + cache the component chain when SpectrumPrecomp is
        # active. The fixed-z spectrum LUT (precompute_spectroscopy) runs
        # numpy interpolation, so it MUST be constructed here from concrete
        # config values, not lazily on the first predict_state, which may
        # be inside a user's jax.jit trace (redshift would be a tracer and
        # the numpy LUT build would raise TracerArrayConversionError). The
        # photometry LUT path avoids this via predict_observables_jit's
        # separately-cached compiled function; the spectrum path runs eager
        # predict_state, so it pre-warms the chain cache here instead.
        if self._approx.get("spectrum_precomp"):
            self._cached_component_chain = self._build_component_chain()

        # Pre-build the dust energy-balance LUT at construction (eager, no JIT
        # trace active) so it is memoized as concrete arrays. Building it lazily
        # inside ``_template_data_for_jit`` would run the jnp integrals *during*
        # a user ``jax.jit(predict_photometry)`` trace, leaking tracers and
        # baking the LUT in as a constant (XLA constant-folds → ~100× slower).
        if self._approx.get("wave_precomp"):
            # The two precomputes are independent and fail independently. A single
            # try around both meant a band-response failure disabled the *energy
            # balance* LUT too, and reported itself under the energy-balance
            # warning, blaming the wrong subsystem.
            try:
                chain = self._build_component_chain()
                self._cached_component_chain = chain
                self._energy_balance_lut(chain)
            except Exception as e:
                # The exact full-wave energy-balance path is the correct fallback,
                # but it forfeits the speedup the astronomer opted into, so say so.
                warnings.warn(
                    f"WavePrecomp energy-balance LUT precompute failed ({e!r}); "
                    "falling back to the exact energy-balance path (correct, "
                    "but without the precomputed-LUT speedup).",
                    UserWarning,
                    stacklevel=2,
                )
                self._energy_balance_lut_cache = None

            try:
                self._dust_emission_band_response(self._cached_component_chain)
            except Exception as e:
                warnings.warn(
                    f"WavePrecomp dust-emission band-response precompute failed "
                    f"({e!r}); falling back to the exact per-call filter integral "
                    "(correct, but without the precomputed-response speedup).",
                    UserWarning,
                    stacklevel=2,
                )
                self._dust_band_response_cache = None

            for _emitter in ("xray", "radio"):
                try:
                    self._additive_term_band_response(self._cached_component_chain, _emitter)
                except Exception as e:
                    warnings.warn(
                        f"WavePrecomp {_emitter} term band-response precompute failed "
                        f"({e!r}); falling back to the exact per-call filter integral "
                        "(correct, but without the precomputed-response speedup).",
                        UserWarning,
                        stacklevel=2,
                    )
                    setattr(self, f"_{_emitter}_term_response_cache", None)

        # Build-time accuracy guard (#617): the photometry LUT bakes the
        # SSP×filter integral at zero dust and re-applies attenuation as a
        # first-order Taylor projection about each filter's effective
        # wavelength. That linear-in-λ model breaks down where the attenuation
        # curve is steep across the bandpass, the rest-UV, so blue bands at
        # moderate/high z are biased silently (the far-UV by >10×). Warn loudly
        # so no fit is biased without the astronomer knowing. Cheap: no SED
        # evaluation, so it is safe on every build.
        self._warn_if_wave_precomp_dust_blue_bias()

        # The emission-line precompute, if the astronomer asked for one. Last,
        # because it needs the nebular backend (``_init_nebular``) and the
        # component chain both to exist.
        self._fast_line_measurement = False
        if self._approx_config_feature is not None:
            self._resolve_feature_precomp(self._approx_config_feature, observation)

    def _resolve_feature_precomp(self, cfg, observation) -> None:
        """Build the emission-line precompute selected by ``approx=FeaturePrecomp()``.

        Dispatches on where the backend keeps its lines. Cue publishes a discrete
        catalog that is linear in Q_H, so a per-Q_H grid over the free ionization
        axes replaces the forward. The baked-in backend has no catalog, its lines
        are inside the SSP templates, so the lines must be *measured* off the
        spectrum, and it gets the window LUT instead (plus the flag that lets the
        likelihood reach it).

        Parameters
        ----------
        cfg: FeaturePrecomp
            The requested configuration.
        observation: Observation
            Source of the line wavelengths when ``cfg.lines`` is None.

        Raises
        ------
        ValueError
            If no lines can be resolved, or the backend supports neither path.
        """
        from tengri.observation.line_measurement import default_line_defs

        backend = self._nebular_backend
        # Cue-like: L_line = Q_H x l(theta), l independent of the SFH shape.
        cue_like = backend is not None and hasattr(backend, "predict_nebular_line_luminosities")

        lines = cfg.lines
        if lines is None:
            line_fluxes = getattr(observation, "line_fluxes", None) if observation else None
            if line_fluxes is not None:
                lines = line_fluxes.wavelengths
            elif cue_like:
                lines = []
            else:
                raise ValueError(
                    "approx=FeaturePrecomp() has no emission lines to tabulate: the "
                    "Observation carries no line_fluxes and FeaturePrecomp(lines=...) "
                    "was not given. Either fit lines, Observation(..., "
                    "line_fluxes=LineFluxData(...)), or name them explicitly."
                )
        lines = jnp.asarray(lines)

        if cue_like:
            self.enable_fast_nebular(lines, n_grid=cfg.n_grid, ranges=cfg.ranges)
            return

        if self._has_line_catalog():
            raise ValueError(
                f"approx=FeaturePrecomp() does not support the "
                f"{type(backend).__name__} nebular backend: it publishes a discrete "
                "line catalog but is not linear in Q_H, so neither the per-Q_H grid "
                "nor the SSP window LUT reconstructs it. Use approx=None for the "
                "lines, or switch to the Cue or baked-in backend."
            )

        # Baked-in / wNE: the lines are inside the SSP templates, so they are
        # measured off the spectrum through the window LUT. Build it eagerly (the
        # SSP grid is concrete at construction) and tell the likelihood to use it.
        self._feature_precomp_lines = lines
        self._line_window_precomp(tuple(default_line_defs(np.asarray(lines))))
        self._fast_line_measurement = True

    def _max_plausible_tau(self, name: str) -> float:
        """Representative upper optical depth for a dust τ parameter.

        Returns the fixed value (fixed param), the prior upper bound (bounded
        free param), a representative ``1.0`` (unbounded free param), or ``0.0``
        when the parameter is absent. Used by
        :meth:`_warn_if_wave_precomp_dust_blue_bias` to decide whether dust is
        in play at all.
        """
        try:
            d = self.spec.get_distribution(name)
        except Exception:
            return 0.0
        if d is None:
            return 0.0
        if getattr(d, "is_fixed", False):
            try:
                return float(d.value)
            except Exception:
                return 0.0
        hi = getattr(d, "hi", None)
        return float(hi) if hi is not None else 1.0

    def _representative_redshift(self) -> float:
        """A single representative redshift (fixed value or prior midpoint)."""
        try:
            d = self.spec.get_distribution("redshift")
        except Exception:
            return float(getattr(self, "_z_fixed", 0.0) or 0.0)
        if d is None:
            return float(getattr(self, "_z_fixed", 0.0) or 0.0)
        if getattr(d, "is_fixed", False):
            try:
                return float(d.value)
            except Exception:
                return 0.0
        lo, hi = getattr(d, "lo", None), getattr(d, "hi", None)
        if lo is not None and hi is not None:
            return 0.5 * (float(lo) + float(hi))
        return float(getattr(self, "_z_fixed", 0.0) or 0.0)

    def _warn_if_wave_precomp_dust_blue_bias(self) -> None:
        """Warn when the WavePrecomp first-order dust projection (#617) is unreliable.

        Fires only when the sub-band quadrature is **off**
        (``WavePrecomp(n_subbands=0)``). There the LUT re-applies dust as a
        first-order Taylor expansion of the attenuation about each filter's
        effective wavelength, a linear model that is accurate where the
        attenuation curve is smooth across the bandpass (optical/IR) but biases
        bands sampling the rest-UV, where the curve is steep and *extrapolated*:
        silently, and by an order of magnitude for far-UV bands at moderate/high
        redshift.

        With the default ``n_subbands=5`` the screen is **evaluated** at K nodes
        per band rather than extrapolated from one (#1122), and the IGM rides the
        same nodes (#1135), so there is no such bias to warn about, see
        :meth:`_fold_igm_into_subbands`.

        Configuration-level heuristic, fires only when a photometry LUT with the
        quadrature disabled, a non-trivial dust screen, and at least one rest-UV
        band coincide. Does **not** evaluate the SED, so it is cheap on every
        build. It flags the *risk*; quantify the actual per-band bias by comparing
        against ``approx=None``.
        """
        if not self._approx.get("wave_precomp"):
            return
        # The K-point sub-band quadrature EVALUATES the screen at each node
        # instead of extrapolating from λ_eff, and the IGM is folded into the
        # same nodes at build time. Measured against the exact path across
        # GALEX→WISE at z ≤ 1.5 with τ_diff=0.7 / τ_bc=1.0, the worst rest-UV
        # residual is ≤ 0.5 %, the Taylor-era warning does not apply.
        if int(self._approx.get("n_subbands", 0) or 0) > 0:
            return
        if self.observation is None or self.filter_waves is None:
            return
        # Dust screen with potentially non-trivial optical depth? (Zero dust →
        # the LUT is exact, so there is nothing to warn about.)
        if (
            max(self._max_plausible_tau("dust_tau_diff"), self._max_plausible_tau("dust_tau_bc"))
            < 0.1
        ):
            return
        z_rep = self._representative_redshift()
        names = list(getattr(self.observation.photometry, "names", []) or [])
        blue: list[tuple[str, float]] = []
        for i, (w, t) in enumerate(zip(self.filter_waves, self.filter_trans)):
            w = jnp.asarray(w)
            t = jnp.asarray(t)
            denom = float(jnp.sum(t * w))
            if denom <= 0.0:
                continue
            # Bessell photon-counting pivot ∫ λ²T dλ / ∫ λT dλ (observed frame).
            lam_eff_rest = float(jnp.sum(t * w * w) / denom) / (1.0 + z_rep)
            if lam_eff_rest < 2000.0:  # rest-UV: Calzetti steepens / is extrapolated
                blue.append((names[i] if i < len(names) else f"band[{i}]", lam_eff_rest))
        if not blue:
            return
        bands = ", ".join(f"{n} (rest~{lam:.0f} Å)" for n, lam in blue)
        warn_measured(
            "This model resolves to n_subbands=0, which applies dust as a first-order "
            f"Taylor projection across each filter (#617); at z~{z_rep:.2f} these "
            f"rest-UV band(s) are biased versus the exact path: {bands}. The bias "
            "grows steeply toward the far-UV (>10x for the bluest bands at "
            "moderate/high z) and with optical depth. The default WavePrecomp() "
            "EVALUATES the screen at n_subbands=5 quadrature nodes per band instead "
            "and does not have this bias, pass approx=WavePrecomp(), or approx=None "
            "for the exact path. On a joint photometry+spectroscopy observation the "
            "photometry channel routes through this same projection, so the warning "
            "applies there too. See docs/known_limitations.md.",
            UserWarning,
            stacklevel=2,
            representative_redshift=z_rep,
            n_biased_bands=len(blue),
        )

    def __repr__(self) -> str:
        """One-line summary of how this model is wired."""
        sfh = getattr(self.spec, "mean_sfh_type", "?")
        if isinstance(sfh, list | tuple):
            sfh_str = "+".join(str(s) for s in sfh)
        else:
            sfh_str = str(sfh)
        dust_str = getattr(self, "_dust_model", "?")
        agn_str = getattr(self, "_agn_model", None) or "off"
        if self._nebular_backend is None:
            neb_str = "off"
        else:
            neb_str = type(self._nebular_backend).__name__.replace("Backend", "").lower()
        n_filt = "?"
        if self.observation is not None and self.observation.photometry is not None:
            try:
                n_filt = len(self.observation.photometry.bands)
            except Exception:
                n_filt = "?"
        n_free = self.spec.n_free
        return (
            f"SEDModel(sfh={sfh_str!r}, dust={dust_str!r}, "
            f"agn={agn_str!r}, nebular={neb_str!r}, "
            f"n_filters={n_filt}, n_free={n_free})"
        )

    def __setattr__(self, name: str, value) -> None:
        """Warn on direct assignment to deprecated filter attributes.

        The attributes ``filter_waves`` and ``filter_trans`` are now read-only
        properties that delegate to ``self.observation.photometry``. Direct
        assignment triggers a deprecation warning.
        """
        if name in ("filter_waves", "filter_trans"):
            warnings.warn(
                f"Direct assignment to SEDModel.{name} is deprecated. "
                f"Access filters through self.observation.photometry instead. "
                f"The attribute will become read-only in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Don't actually set the attribute, it's a property now
            return
        object.__setattr__(self, name, value)

    # ── Deprecated filter/noise attributes → Observation delegation ──────
    # Step E: Make Observation the sole owner of filters and noise config.
    # These properties delegate to self.observation; direct assignment issues
    # a deprecation warning (not yet removed for backwards compatibility).

    @property
    def filter_waves(self):
        """Read-only view of photometric filter wavelengths.

        Delegates to ``self.observation.photometry.filter_waves`` if available.
        Returns None if no photometry is configured.

        Notes
        -----
        **Deprecated**: Access filters through ``self.observation.photometry``
        directly. Direct assignment is discouraged.
        """
        if self.observation is not None and self.observation.can_do_photometry:
            return list(self.observation.photometry.filter_waves)
        return None

    @property
    def filter_trans(self):
        """Read-only view of photometric filter transmission curves.

        Delegates to ``self.observation.photometry.filter_trans`` if available.
        Returns None if no photometry is configured.

        Notes
        -----
        **Deprecated**: Access filters through ``self.observation.photometry``
        directly. Direct assignment is discouraged.
        """
        if self.observation is not None and self.observation.can_do_photometry:
            return list(self.observation.photometry.filter_trans)
        return None

    @property
    def wave_obs(self):
        """Configured observed-frame spectroscopy wavelength grid, or ``None``.

        Reports the grid the model predicts spectra on: an explicitly cached
        ``_wave_obs`` if present, otherwise the configured
        ``observation.spectroscopy.wave_obs`` (the source of truth, #389/#620).
        Returns ``None`` only when no spectroscopy grid is configured anywhere.

        Returns
        -------
        ndarray or None
            Observed-frame wavelength grid [Angstrom], shape ``(n_pix,)``.
        """
        cached = getattr(self, "_wave_obs", None)
        if cached is not None:
            return cached
        obs = self.observation
        if obs is not None and getattr(obs, "spectroscopy", None) is not None:
            return getattr(obs.spectroscopy, "wave_obs", None)
        return None

    @property
    def has_fixedz_photometry_precompute(self) -> bool:
        """Whether this is a fixed-z photometry model (vmap-batch / fast-path eligible).

        Replaces the legacy ``model.precomputed.photometry is not None`` proxy
        (the ``PrecomputedData`` container is being retired, #620). The legacy
        container's ``photometry`` slot was populated exactly when a fixed
        redshift and filters were configured, so this reproduces that boolean
        without the container. The LUT fast path itself is opt-in via
        ``approx=WavePrecomp()`` and lives in the component chain.
        """
        return self._z_fixed is not None and self.filter_waves is not None

    @property
    def hybrid(self):
        """Container of hybrid (precomputed × on-the-fly) kernels, or ``None``.

        Public accessor for the internal ``_hybrid`` attribute. Returns
        ``None`` when no hybrid kernels were built (e.g. when the model
        is constructed without ``precompute=True``). Slots (``photometry``,
        ``spectroscopy``) on the returned container are individually
        ``None`` when that channel's hybrid path is unavailable.
        """
        return getattr(self, "_hybrid", None)

    @property
    def z_fixed(self):
        """Fixed redshift value if redshift is not a free parameter, else ``None``.

        Public accessor for the internal ``_z_fixed`` attribute. Set at
        construction from ``spec.get_fixed_values().get('redshift')``.
        """
        return getattr(self, "_z_fixed", None)

    @property
    def dl_cm_fixed(self):
        """Fixed luminosity distance [cm] when redshift is fixed, else ``None``.

        Public accessor for the internal ``_dl_cm_fixed`` attribute. Used
        by inference to detect a redshift-fixed forward model eligible
        for the fast precomputed-photometry path.
        """
        return getattr(self, "_dl_cm_fixed", None)

    @property
    def n_grid(self):
        """PSD-grid resolution for stochastic SFH, else ``0``.

        Public accessor for the internal ``_n_grid`` attribute. Non-zero
        only when the model uses a stochastic SFH (correlated-field
        prior on the SFH); used by inference to size the latent grid.
        """
        return getattr(self, "_n_grid", 0)

    @property
    def uses_stochastic_sfh(self) -> bool:
        """``True`` if the SFH is a stochastic correlated-field model.

        Public accessor for the internal ``_uses_stochastic_sfh`` flag.
        Stochastic SFH adds an additional ``psd_xi`` latent of shape
        ``(n_grid,)`` to the free-parameter set.
        """
        return bool(getattr(self, "_uses_stochastic_sfh", False))

    @property
    def Observables(self) -> type:
        """Return the per-model :class:`Observables` NamedTuple class.

        Returns
        -------
        type
            A :class:`typing.NamedTuple` subclass whose fields match the
            configured observation sub-blocks. Synthesized at construction
            time by :func:`build_observables_class`.

        Raises
        ------
        ValueError
            If no observation is configured.

        Notes
        -----
        Each model gets its own NamedTuple class, with fields (and
        magnitude properties) appearing only when the corresponding
        observation sub-block is configured.
        """
        if self._Observables is None:
            raise ValueError(
                "Observables requires an Observation. Build the model with observation= set."
            )
        return self._Observables

    @staticmethod
    def _init_observation(spec, filters, observation):
        """Resolve observation/filters into a canonical Observation + spec."""
        if filters is not None and observation is not None:
            raise ValueError(
                "Cannot specify both filters= and observation=. "
                "Use observation=Observation(photometry=...) instead."
            )

        if observation is not None or filters is not None:
            from tengri.observation.observation import Observation

        if observation is not None:
            if not isinstance(observation, Observation):
                observation = SEDModel._wrap_as_observation(observation)
            obs_params = observation.get_all_params()
            if obs_params:
                spec = spec.with_params(**obs_params)
        elif filters is not None:
            from tengri.observation.photometry_config import Photometry

            observation = Observation(photometry=Photometry.from_filter_set(filters))

        return observation, spec

    @staticmethod
    def _wrap_as_observation(observation):
        """Wrap a lone observation component in an :class:`Observation`.

        ``observation=`` expects an :class:`Observation`, but a fresh user
        naturally reaches for the component constructors the discovery API
        advertises, e.g. ``list_filters()`` suggests
        ``Photometry.from_names([...])``. A bare ``Photometry`` (or
        ``Spectroscopy``/``LineFluxData``/``SpectralIndexData``/``LineRatioData``)
        maps unambiguously onto a single :class:`Observation` slot, so wrap it
        rather than reject it. The ``filters=`` path already auto-wraps the
        same way.

        Parameters
        ----------
        observation: object
            The value passed as ``observation=``. Expected to be one of the
            single-slot observation component types.

        Returns
        -------
        Observation
            ``Observation(<slot>=observation)`` for the matching slot.

        Raises
        ------
        TypeError
            If ``observation`` is not an :class:`Observation` and not one of
            the wrappable component types. The message names the accepted
            types and the explicit wrap.
        """
        from tengri.observation.line_flux_data import LineFluxData
        from tengri.observation.line_ratio_data import LineRatioData
        from tengri.observation.observation import Observation
        from tengri.observation.photometry_config import Photometry
        from tengri.observation.spectral_indices import SpectralIndexData
        from tengri.observation.spectroscopy import Spectroscopy

        # type -> Observation constructor keyword for the single-slot map.
        wrap_slots = (
            (Photometry, "photometry"),
            (Spectroscopy, "spectroscopy"),
            (LineFluxData, "line_fluxes"),
            (SpectralIndexData, "spectral_indices"),
            (LineRatioData, "line_ratios"),
        )
        for component_type, slot in wrap_slots:
            if isinstance(observation, component_type):
                return Observation(**{slot: observation})

        accepted = ", ".join(t.__name__ for t, _ in wrap_slots)
        raise TypeError(
            f"observation= must be an Observation (or a single component of type "
            f"{accepted}), got {type(observation).__name__}. Wrap it explicitly, "
            f"e.g. observation=Observation(photometry=my_photometry), or pass "
            f"filters=[...] for a photometry-only model."
        )

    def _init_ssp(self, spec, ssp_data, csp_integration):
        """Set up SSP grid, CSP integration, and log-age grid."""
        self._met_interp = getattr(spec, "met_interp", "linear")
        self._lgmet_scatter = float(getattr(spec, "lgmet_scatter", 0.1))
        # Redshift-table interpolation mode for free-z inference.
        # "linear" → piecewise-linear (C^0 gradient, kinks at grid nodes).
        # "smooth" → triweight kernel (C^2 gradient), recommended for NUTS/HMC
        # when redshift is a free parameter. See `interpolate_ztable_smooth`
        # in components/sps/precompute.py.
        self._z_interp = getattr(spec, "z_interp", "linear")

        self.ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        self.ssp_ages_yr = 10.0**self.ssp_log_ages_yr

        if csp_integration not in _VALID_CSP_INTEGRATION:
            raise ValueError(
                f"csp_integration must be one of {_VALID_CSP_INTEGRATION}, got {csp_integration!r}"
            )
        if csp_integration != _DEFAULT_CSP_INTEGRATION:
            # Say it at construction, where the user can act on it. The stellar
            # component builds age weights with cloud-in-cell for every
            # configuration (sps_backend="dsps" is the only backend), so this
            # cannot reach the SED -- photometry is bit-identical across all five
            # values. It used to change _predict_sfh_quantities alone, which meant
            # the reported stellar mass came from a different integration than the
            # spectrum it was fitted to (#1500).
            warnings.warn(
                f"csp_integration={csp_integration!r} has no effect and is deprecated. "
                "The stellar component integrates the CSP with the kernel named by "
                "sfh={'age_kernel': ...} (cloud-in-cell by default), which this "
                "argument does not feed, so the predicted SED, "
                "photometry and derived quantities are identical to "
                f"{_DEFAULT_CSP_INTEGRATION!r}. It previously changed the reported "
                "stellar mass without changing the SED (0.32% for 'log_interp' / "
                "'dsps_native', NaN for 'dsps_met_table'); that inconsistency is "
                "fixed, and the argument will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=3,
            )
        self._csp_integration = csp_integration
        if csp_integration == "log_interp":
            from tengri.components.stellar.sps.dsps_wrapper import csp_log_interp_matrix

            self._csp_matrix = jnp.array(csp_log_interp_matrix(self.ssp_ages_yr))
            self._csp_age_dt = None
        elif csp_integration in ("dsps_native", "dsps_met_table"):
            self._csp_age_dt = None
            self._csp_matrix = None
        else:
            self._csp_age_dt = csp_age_dt(self.ssp_ages_yr, csp_integration)
            self._csp_matrix = None

        # Honor the user-set ``n_grid`` for every SFH type, not just
        # stochastic. ``spec.n_grid`` defaults to 256, so the parametric default
        # is unchanged; setting it (``SEDModel.build(..., n_grid=N)``) now takes
        # effect for parametric SFHs too. The parametric stellar SED is in fact
        # n_grid-invariant (the SFH×SSP integral converges by ~64 points, see
        # the #499 quadrature check), so this is a control/perf knob, not a
        # correctness change.
        n_grid = spec.n_grid
        self.log_age_grid = make_log_age_grid(n_grid)
        self.d_log_age = grid_spacing(self.log_age_grid)
        self.age_yr = log_age_to_age_yr(self.log_age_grid)
        self._n_grid = n_grid

    def _init_sfh(self, spec):
        """Resolve SFH from registry and return the base param_map delta.

        Returns
        -------
        dict[str, tuple[str, float, float]]
            Parameter map entries for this component:
            public_name -> (internal_name, scale, offset).
        """
        # Forward the (optional) non-parametric bin edges through to
        # resolve_sfh so prospector_beta / continuity / ... use the
        # user-supplied edges in the forward pass (#337).
        _sfh_kwargs = {}
        _bin_edges = getattr(spec, "bin_edges_gyr", None)
        if _bin_edges is not None:
            _sfh_kwargs["bin_edges_gyr"] = _bin_edges
        sfh_fn, _sfh_params, sfh_param_map, sfh_settings = resolve_sfh(
            spec.mean_sfh_type, **_sfh_kwargs
        )
        self._sfh_fn = sfh_fn
        self._sfh_internal_names = {v[0] for v in sfh_param_map.values()}
        # Per-spec public SFH param names (sfh_X_*). The composer
        # dispatches per-component on these to avoid collisions when two
        # additive SFHs share an internal kwarg (e.g. ``log_total_mass``).
        # See ``composed_fn`` in components/stellar/sfh/registry.py and #372.
        self._sfh_public_names = set(sfh_param_map.keys())
        self._sfh_settings = sfh_settings
        self._uses_stochastic_sfh = spec.stochastic
        self._gp_kernel = sfh_settings.get("sfh_field_model", "drw")
        # GP-field parameterization (#1355): 1.0 is the shipped non-centered
        # map. Read once here so every ``compute_field_gp`` call site in this
        # class shares one value, the knob was previously reachable by none.
        self._field_centering = float(getattr(spec, "field_centering", 1.0))

        # Warn if any burst-width SFH parameter is narrower than the
        # local SSP grid spacing at the burst peak, see #299. The
        # forward model interpolates SFR(t) at SSP grid points (not a
        # bin-integral), so narrow bursts alias as a staircase in
        # age-sensitive observables.
        from tengri.components.stellar.sfh._aliasing_warning import (
            maybe_warn_burst_aliasing,
        )

        maybe_warn_burst_aliasing(spec, self.ssp_ages_yr)

        # Return the base param_map delta (built param_map + dust-model selection)
        return _build_param_map(
            spec.mean_sfh_type,
            dust_model=getattr(spec, "dust_model", "two_component"),
        )

    def _init_metallicity(self, spec):
        """Configure metallicity mode and evolving alpha-enhancement.

        Returns
        -------
        dict[str, tuple[str, float, float]]
            Parameter map deltas for metallicity handling.

        Raises
        ------
        NotImplementedError
            If alpha_fe_evolving=True, which is currently not supported (#1767).
        """
        self._met_mode = getattr(spec, "met_mode", "delta")
        # _met_mode checked directly: "ramp" for evolving, "chem_evol" for chemical evolution

        from tengri.components.stellar.sfh.met_registry import resolve_met

        _, _, met_param_map, _ = resolve_met(self._met_mode)
        delta = {}

        # If not delta mode, exclude met_logzsol and use met_param_map instead
        if self._met_mode != "delta":
            delta.update(met_param_map)
        else:
            delta.update(met_param_map)

        self._alpha_fe_evolving = getattr(spec, "alpha_fe_evolving", False)
        if self._alpha_fe_evolving:
            # #1767: Per-age alpha-enhancement requires wiring compute_alpha_fe_evolving
            # through the stellar component's SED production pipeline. Currently the
            # mechanism accepts only a scalar met_alpha_fe, which is applied uniformly
            # to all ages (either via SSP grid interpolation or effective-metallicity
            # calculation). Supporting a per-age ramp would require architectural
            # changes: propagating per-age arrays through the age-weight loop and
            # modifying effective_metallicity / interpolate_alpha_only to work per-age.
            raise NotImplementedError(
                "alpha_fe_evolving=True is not yet supported (#1767). "
                "The stellar component currently accepts only a scalar "
                "met_alpha_fe [alpha/Fe] applied uniformly to all ages. "
                "To model alpha enhancement, use alpha_fe_evolving=False "
                "(the default) and set met_alpha_fe to a Fixed or free scalar value. "
                "Per-age alpha ramping from alpha_fe_old to alpha_fe_young requires "
                "architectural extensions currently under development."
            )

        return delta

    # Names of every public-API metallicity parameter that resolves to a
    # log10(Z/Zsun) lookup on the SSP grid. Each lives on the same grid
    # axis (``ssp_data.ssp_lgmet``) so the bounds check is identical.
    _MET_LOGZSOL_PARAM_NAMES = (
        "met_logzsol",
        "met_logzsol_0",
        "met_logzsol_final",
        "met_logzsol_old",
        "met_logzsol_young",
        "met_logzsol_burst",
        "met_logzsol_base",
    )

    def _validate_metallicity_bounds(self, spec, ssp_data):
        """Warn / raise if any ``met_logzsol*`` value escapes the SSP grid.

        The forward model interpolates log10(Z/Zsun) onto ``ssp_data.ssp_lgmet``
        with ``jnp.clip`` at the grid edges and ``jnp.searchsorted`` for the
        bracket, so an out-of-range value silently clamps to the edge and
        produces a smooth-but-wrong SED. A MAP/MCMC chain wandering to a
        prior edge would interpret that plateau as a likelihood maximum
        (issue #442).

        Catch this at construction time rather than at forward-pass time:
        the JIT'd predict path can't raise Python exceptions, but build()
        can.

        Both ``Fixed`` out-of-grid values and ``Uniform`` priors with
        out-of-grid bounds emit a :class:`UserWarning`. The forward pass
        still produces a numerical SED (via ``jnp.clip``), but the
        warning makes the silent-clip path visible. A strict raise was
        considered but rejected: synthetic / non-Zsun-offset SSPs (e.g.
        fixture grids in unit tests) sometimes ship lgmet values that
        live in log10(Z/Zsun) directly rather than absolute log10(Z),
        and we don't want to lock them out.
        """
        if ssp_data is None:
            return  # synthetic / placeholder construction path

        lgmet = np.asarray(ssp_data.ssp_lgmet)
        if lgmet.size == 0:
            return

        # SSP grid stores absolute log10(Z); user-facing params are
        # log10(Z/Zsun), which differs by LOG10_ZSUN.
        grid_lo_zsol = float(lgmet.min()) - LOG10_ZSUN
        grid_hi_zsol = float(lgmet.max()) - LOG10_ZSUN

        distributions = getattr(spec, "_distributions", {})
        for name in self._MET_LOGZSOL_PARAM_NAMES:
            dist = distributions.get(name)
            if dist is None:
                continue
            if dist.is_fixed:
                val = float(dist.value)
                if not (grid_lo_zsol <= val <= grid_hi_zsol):
                    warn_measured(
                        f"{name}={val:.3f} is outside the SSP grid metallicity "
                        f"range [{grid_lo_zsol:.3f}, {grid_hi_zsol:.3f}] "
                        f"log10(Z/Zsun) (absolute grid log10(Z) ∈ "
                        f"[{lgmet.min():.3f}, {lgmet.max():.3f}]). The forward "
                        f"model silently clips out-of-range values, producing "
                        f"a misleadingly smooth SED (issue #442). Either set "
                        f"{name} inside the grid, or load an SSP whose grid "
                        f"covers your target metallicity.",
                        UserWarning,
                        value=val,
                        grid_lo_zsol=grid_lo_zsol,
                        grid_hi_zsol=grid_hi_zsol,
                        stacklevel=3,
                    )
            else:
                lo, hi = float(dist.lo), float(dist.hi)
                if lo < grid_lo_zsol or hi > grid_hi_zsol:
                    warn_measured(
                        f"{name} prior bounds [{lo:.3f}, {hi:.3f}] extend "
                        f"beyond the SSP grid metallicity range "
                        f"[{grid_lo_zsol:.3f}, {grid_hi_zsol:.3f}] "
                        f"log10(Z/Zsun). Samples outside the grid will "
                        f"silently clip to the edge, a MAP/MCMC chain "
                        f"wandering there registers a fake likelihood "
                        f"maximum (issue #442). Tighten the prior to within "
                        f"the grid range or load an SSP with broader coverage.",
                        UserWarning,
                        prior_lo=lo,
                        prior_hi=hi,
                        grid_lo_zsol=grid_lo_zsol,
                        grid_hi_zsol=grid_hi_zsol,
                        stacklevel=3,
                    )

    def _validate_alpha_fe_identifiability(self, spec, ssp_data):
        """Warn if a free ``met_alpha_fe`` cannot be identified by this model.

        ``[alpha/Fe]`` is a real, independently-interpolated axis only when the
        SSP grid carries one (:func:`has_alpha_grid`, the 4D path of #226).
        Without it, ``met_alpha_fe`` reaches the SED solely through
        :func:`effective_metallicity`, and only from the ``"delta"`` metallicity
        branch. That leaves two distinct failures, both silent today:

        * **Any non-delta metallicity model** never reads ``met_alpha_fe`` at
          all. Measured on a 3D grid, sweeping 0.0 -> 0.6 under ``"ramp"`` and
          ``"two_step"``: ``numpy.array_equal`` returns ``True`` and
          ``d(sum SED)/d(alpha)`` is exactly ``0.0`` (issue #1764). The
          parameter is still accepted as free, so the reported posterior is the
          prior.
        * **Delta metallicity** folds it in as a pure additive shift,
          ``log_z_eff = met_logzsol + 0.75 * met_alpha_fe``, so freeing it
          alongside ``met_logzsol`` gives an exactly flat ridge (issue #1095).

        This check needs three facts that live in three objects, the grid
        (``ssp_data``), the metallicity mode (``self._met_mode``), and which
        parameters are free (``spec``), which is why it belongs here and not on
        the parameter declaration: :class:`Parameters` cannot see ``ssp_data``,
        so a guard placed there would fire falsely on every 4D grid.

        Warns rather than raises, matching :meth:`_validate_metallicity_bounds`:
        the forward model is correct in every case, and both configurations are
        legitimate if the user wants them.
        """
        if ssp_data is None:
            return  # synthetic / placeholder construction path

        distributions = getattr(spec, "_distributions", {})
        alpha_dist = distributions.get("met_alpha_fe")
        if alpha_dist is None or alpha_dist.is_fixed:
            return

        from tengri.components.stellar.sps.dsps_wrapper import (
            _ALPHA_TO_Z_COEFF,
            has_alpha_grid,
        )

        if has_alpha_grid(ssp_data):
            return  # 4D grid: [alpha/Fe] interpolates a real axis (#226)

        met_mode = str(self._met_mode)
        if met_mode != "delta":
            warn_measured(
                f"met_alpha_fe is free, but metallicity mode {met_mode!r} never "
                f"reads it. On an SSP grid with no [alpha/Fe] axis the "
                f"enhancement is applied only in the 'delta' branch, so "
                f"d(SED)/d(met_alpha_fe) is exactly 0.0: no gradient-based "
                f"sampler can move it and the reported posterior is the prior "
                f"(issue #1764). Use met_mode='delta', pin met_alpha_fe with "
                f"Fixed(...), or load an SSP grid carrying an [alpha/Fe] axis.",
                DeadGradientParameterWarning,
                gradient=0.0,
                stacklevel=3,
            )
            return

        logzsol_dist = distributions.get("met_logzsol")
        if logzsol_dist is not None and not logzsol_dist.is_fixed:
            warn_measured(
                f"met_alpha_fe and met_logzsol are both free, but this SSP grid "
                f"has no [alpha/Fe] axis, so [alpha/Fe] enters only as an "
                f"additive shift of the effective metallicity: log_z_eff = "
                f"met_logzsol + {_ALPHA_TO_Z_COEFF} * met_alpha_fe. The pair is "
                f"exactly degenerate, the likelihood is flat along "
                f"met_logzsol + {_ALPHA_TO_Z_COEFF} * met_alpha_fe = const, and "
                f"a Laplace fit assigns that direction the variance its "
                f"eigenvalue floor implies rather than a measured one (issues "
                f"#1095, #1515). Free one or the other, or load an SSP grid "
                f"carrying an [alpha/Fe] axis.",
                DegenerateParameterPairWarning,
                coefficient=_ALPHA_TO_Z_COEFF,
                stacklevel=3,
            )

    def _init_dust(self, spec):
        """Configure dust attenuation laws, nebular dust, and dust emission.

        Returns
        -------
        dict[str, tuple[str, float, float]]
            Parameter map deltas for dust components.
        """
        self._dust_model = getattr(spec, "dust_model", "two_component")
        self._dust_scheme = getattr(spec, "dust_approx", "fast")

        # WG00 (dust_type=3) structural selectors, static strings threaded into
        # the WG00 screen component via ``build_components`` (and the
        # ``compile_signature``). Defaults match the FSPS shell/MW/homogeneous case.
        self._wg00_dust_curve = getattr(spec, "dust_wg00_curve", "mw")
        self._wg00_geometry = getattr(spec, "dust_wg00_geometry", "shell")
        self._wg00_structure = getattr(spec, "dust_wg00_structure", "homogeneous")

        self._dust_law_bc = spec.dust_law_bc
        self._dust_law_diff = spec.dust_law_diff
        # Nebular birth-cloud law (None -> inherit the stellar birth cloud).
        # Decouples HII-region reddening from the stars while sharing the
        # diffuse ISM screen; consumed by ``DustSEDComponent`` via
        # ``build_components``.
        self._dust_law_neb = getattr(spec, "dust_law_neb", None)
        # Per-component law-parameter overrides ({'bc': {...}, 'diff': {...},
        # 'neb': {...}}), set by the builder when the user supplies
        # slope_bc / delta_diff / slope_neb / etc.
        self._dust_law_overrides = getattr(spec, "dust_law_overrides", None) or {}
        # Lyman-limit clip [Å]: zero the attenuation curve below this wavelength
        # (0.0 -> off; 912.0 -> CIGALE parity). Static, non-fittable.
        self._dust_lyman_cutoff_aa = float(getattr(spec, "dust_lyman_cutoff_aa", 0.0) or 0.0)
        # Whether ALL stellar LyC is absorbed by neb_fesc (FSPS/CIGALE) vs the
        # default young/birth-cloud-only (bagpipes). See DustSEDComponent.
        self._dust_lyc_absorb_all = bool(getattr(spec, "dust_lyc_absorb_all", False))
        # Include LyC in the dust energy-balance integral (FSPS/Prospector
        # parity, #961) vs the canonical LyC mask (#922). See DustSEDComponent.
        self._dust_eb_include_lyc = bool(getattr(spec, "dust_eb_include_lyc", False))

        # Dust law resolution. Skip for dust_model='off' or 'wg00' (wg00 has no
        # attenuation law; 'off' means no dust at all). Both store placeholder
        # power_law values that are never used, so we skip resolution to avoid
        # wasting compile time.
        if self._dust_model not in ("off", "wg00"):
            from tengri.components.dust.attenuation import resolve_dust_law

            self._dust_law_bc_fn = resolve_dust_law(self._dust_law_bc)
            if self._dust_model == "single_component":
                self._dust_law_diff_fn = self._dust_law_bc_fn
            else:
                self._dust_law_diff_fn = resolve_dust_law(self._dust_law_diff)

            self._neb_dust_mode = getattr(spec, "neb_dust", "bc")
            _neb_bc_law_name = self._dust_law_neb or getattr(spec, "neb_dust_law_bc", None)
            if _neb_bc_law_name is not None:
                from tengri.components.dust.attenuation import resolve_dust_law as _rdl

                self._neb_dust_law_bc_fn = _rdl(_neb_bc_law_name)
            else:
                self._neb_dust_law_bc_fn = self._dust_law_bc_fn
        else:
            # Placeholder functions for off/wg00 (never used, but kept for
            # attribute consistency).
            self._dust_law_bc_fn = None
            self._dust_law_diff_fn = None
            self._neb_dust_law_bc_fn = None
            self._neb_dust_mode = getattr(spec, "neb_dust", "bc")

        self._dust_emission_model = getattr(spec, "dust_emission", None)
        # Astrodust+PAH configuration: now always exists as a structural setting.
        self._astrodust_spinning_dust = bool(spec.astrodust_spinning_dust)
        self._astrodust_f_cnm = float(spec.astrodust_f_cnm)
        if self._dust_emission_model == "dl07_tabulated":
            warnings.warn(
                "'dl07_tabulated' is deprecated. Use 'draine_li2007' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._dust_emission_model = "draine_li2007"

        # Template-based dust emission models lazy-load HDF5 grids on first
        # call. If the first call happens inside a `@jax.jit` scope (the
        # common Fitter path), `jnp.array(...)` inside the loader creates
        # `DynamicJaxprTracer` objects that escape the loader's closure,
        # triggering `UnexpectedTracerError` on subsequent non-JIT calls.
        # Mirror the `_warm_grid_caches()` pattern used by `_build_precomputed_data`
        # and force-load at factory time (#390).
        _TEMPLATE_BASED_EMISSION_MODELS = frozenset(
            # NB: include both the canonical ``draine_li2014`` name and its
            # ``dl14`` alias, the canonical name was missing, so a model built
            # with emission='draine_li2014' was never preloaded and lazy-loaded
            # its templates inside the JIT trace (UnexpectedTracerError).
            {"draine_li2007", "dl14", "draine_li2014", "dale2014", "astrodust", "bosa", "themis"}
        )
        if self._dust_emission_model in _TEMPLATE_BASED_EMISSION_MODELS:
            from tengri.components.dust.emission import preload_emission_model

            # Same reasoning as the AGN backend warm below: this exists only to
            # pull templates into the registry before the JIT trace, since
            # loading them inside it raises UnexpectedTracerError. A load
            # failure here is recoverable, the exact path still works, but a
            # *bug* in the loader should not be. Narrowed to the
            # data/dependency family so it is not both.
            with contextlib.suppress(ImportError, OSError, KeyError):
                preload_emission_model(self._dust_emission_model)

        # Identity entries for dust-emission params now come from the
        # registry-driven auto-derive in ``_build_param_map`` (Step B,
        # ADR-deepening 2026-05-18). The conditional on the active
        # emission model is retained at the apply/predict layer; the
        # param-map can carry the entries unconditionally because
        # ``get_internal_params`` silently skips names absent from spec.
        return {}

    def _init_igm(self, spec):
        """Configure IGM absorption and DLA.

        The IGM model name is looked up in
        :data:`tengri.components.igm.IGM_TRANSMISSION_MODELS` via
        :func:`~tengri.components.igm.resolve_igm_model`, which also
        resolves the back-compat alias ``"inoue"`` -> ``"inoue14"``.
        """
        from tengri.components.igm import resolve_igm_model

        self._uses_igm = spec.apply_igm
        self._uses_dla = getattr(spec, "dla", False)
        self._igm_patchy = getattr(spec, "igm_patchy", False)
        self._igm_model = getattr(spec, "igm_model", "inoue")
        self._igm_fn = resolve_igm_model(self._igm_model)

    def _init_nebular(self, spec, ssp_data):
        """Configure nebular emission backend and return param_map entries.

        Returns
        -------
        dict[str, tuple[str, float, float]]
            Parameter map deltas for nebular components.
        """
        delta = {}

        if spec.nebular_mode in ("cloudy", "cue", "cb19"):
            # ``_NEBULAR_IDENTITY_PARAMS`` removed in Step B (ADR-deepening
            # 2026-05-18); the registry-driven auto-derive in
            # ``_build_param_map`` covers the identity entries. The
            # unit-converting ``neb_logZ_gas`` MUST stay here because
            # auto-derive only emits identity mappings.
            delta["neb_logZ_gas"] = ("neb_logZ_gas", 1.0, LOG10_ZSUN)

        self._nebular_backend = None
        # Track which nebular backend is active so the wavelength-extension
        # registry can route the build to ``native_wave_nebular(...)``. Only
        # backends that ship a tabulated native grid (Cue's ``cont_wav``)
        # currently extend the master grid; CLOUDY-grid / CB19 evaluate on
        # the consumer's grid and contribute nothing here.
        self._nebular_model = (
            spec.nebular_mode if spec.nebular_mode not in ("off", "ssp") else None
        )
        if spec.nebular_mode == "cue":
            from tengri.components.nebular import CueBackend

            # Cue-specific abundance + ionizing-spectrum free params. These
            # are validated by Parameters but were silently stripped by
            # translate.get_internal_params before being registered here.
            # See MISSING_FEATURES.md #16. Register only the ones the user
            # explicitly added to the spec, Parameters mirrors the same
            # conditional registration in _CUE_GAS_EXTRA_PARAMS / _CUE_IONSPEC_PARAMS.
            _user_params = getattr(spec, "_valid_param_names", frozenset())
            for name in _CUE_GAS_IDENTITY_PARAMS:
                if name in _user_params:
                    delta[name] = (name, 1.0, 0.0)
            for name in _CUE_IONSPEC_IDENTITY_PARAMS:
                if name in _user_params:
                    delta[name] = (name, 1.0, 0.0)
            self._nebular_backend = CueBackend(spec.cue_weights_path, ssp_data=ssp_data)
        elif spec.nebular_mode == "cloudy":
            from tengri.components.nebular import CloudyGridBackend

            self._nebular_backend = CloudyGridBackend(spec.cloudy_grid_path, ssp_data)
        elif spec.nebular_mode == "cb19":
            # Bug A in #361: ``neb={'type': 'cb19'}`` used to fall through
            # to the BakedIn ``else`` branch, leaving the user with a model
            # whose ``_nebular_backend`` was the wrong class, every line
            # accessor then returned NaN with no warning. Dispatch explicitly.
            from tengri.components.nebular import CB19Backend

            self._nebular_backend = CB19Backend(ssp_data=ssp_data)
        else:
            from tengri.components.nebular import BakedInBackend

            self._nebular_backend = BakedInBackend()

        return delta

    def _init_agn(self, spec):
        """Configure AGN model and detect parametric vs. fraction mode.

        Returns
        -------
        dict[str, tuple[str, float, float]]
            Parameter map deltas for AGN components.
        """
        self._agn_model = getattr(spec, "agn_model", None)
        # Static block selectors for the "composable" AGN recipe; default to
        # "none" so non-composable models receive harmless no-op selectors.
        self._agn_disc_block = getattr(spec, "agn_disc_block", "none")
        self._agn_nlr_block = getattr(spec, "agn_nlr_block", "none")
        self._agn_blr_block = getattr(spec, "agn_blr_block", "none")
        self._agn_feii_block = getattr(spec, "agn_feii_block", "none")
        self._agn_torus_block = getattr(spec, "agn_torus_block", "none")
        self._agn_attenuation_block = getattr(spec, "agn_attenuation_block", "none")
        self._agn_norm = getattr(spec, "agn_norm", "cigale_joint")
        self._agn_luminosity_mode = False

        delta = {}
        if self._agn_model:
            agn_dists = getattr(spec, "_distributions", {})
            agn_lbol_dist = agn_dists.get("agn_log_lbol")
            agn_frac_dist = agn_dists.get("agn_lum_ratio")
            lbol_is_free = agn_lbol_dist is not None and not agn_lbol_dist.is_fixed
            frac_is_free = agn_frac_dist is not None and not agn_frac_dist.is_fixed
            self._agn_luminosity_mode = lbol_is_free and not frac_is_free
            # Identity entries for agn_* now come from registry auto-derive
            # in _build_param_map (Step B).
            if self._agn_model == "skirtor":
                # Pre-warm the SKIRTOR template cache outside any JIT context.
                # Calling _load_skirtor_fn() lazily inside jit.trace causes a
                # tracer leak because create_skirtor_from_grid allocates jnp.array
                # objects that get captured as DynamicJaxprTracers.
                try:
                    from tengri.components.agn.unified import _load_skirtor_fn

                    _load_skirtor_fn()
                except Exception:
                    pass

            # Pre-warm the Synthesizer CLOUDY line-region grid singletons for
            # the composable ``nlr_synthesizer`` / ``blr_synthesizer`` line
            # blocks, for the same reason as SKIRTOR above (#390 class of bug):
            # ``SynthesizerNLRBackend.__init__`` reads the HDF5 grid and runs
            # ``jnp.sort`` / ``bool(axis[0] > axis[-1])`` on the grid axes to
            # pick interpolation direction. If that construction first happens
            # lazily inside ``predict_photometry`` / ``WavePrecomp`` (the JIT
            # path used for fitting), the eager ``bool(...)`` on what JAX has
            # lifted into the trace raises ``TracerBoolConversionError``. The
            # singleton must therefore be built once here, at factory time,
            # with the same grid path the forward resolves so the cached
            # instance is reused under trace. (The Gaussian ``nlr`` / ``blr``
            # blocks are JIT-safe and need no warming.)
            if self._agn_nlr_block in ("synthesizer", "synthesizer_spectra") or (
                self._agn_blr_block in ("synthesizer", "synthesizer_spectra")
            ):
                from tengri.components.agn.blocks.blr import (
                    _resolve_synthesizer_grid as _resolve_blr_grid,
                )
                from tengri.components.agn.blocks.nlr import (
                    _resolve_synthesizer_grid as _resolve_nlr_grid,
                )

                # Resolving the grid path is *outside* the suppress below, so a
                # grid that is not on disk fails the build instead of the first
                # ``predict`` (#1462). The suppress covers pre-warming, which is
                # an optimization: if the singleton cannot be constructed for
                # some other reason, the lazy path is still correct. A missing
                # file is not that, it guarantees ``predict`` raises, so
                # swallowing it here handed the user a model object that could
                # never produce a number, with the traceback arriving much later
                # and far from the ``nlr='synthesizer'`` that caused it.
                nlr_grid = blr_grid = None
                if self._agn_nlr_block in ("synthesizer", "synthesizer_spectra"):
                    nlr_grid = _resolve_nlr_grid("nlr")
                if self._agn_blr_block in ("synthesizer", "synthesizer_spectra"):
                    blr_grid = _resolve_blr_grid("blr")

                # Warming a cache: these backends load lazily at predict time
                # anyway, so a failure here costs latency, not correctness. The
                # failures worth tolerating are the data/dependency ones,                 #
                # Synthesizer absent, grid file missing or unreadable. Catching
                # everything also swallowed genuine bugs *inside* the loaders
                # (a TypeError from a changed signature, say), and the only
                # symptom was the first predict paying a cost this line existed
                # to remove.
                with contextlib.suppress(ImportError, OSError, KeyError):
                    from tengri.components.agn.nlr_cloudy import (
                        get_synthesizer_blr_backend,
                        get_synthesizer_nlr_backend,
                    )

                    if nlr_grid is not None:
                        get_synthesizer_nlr_backend(nlr_grid)
                    if blr_grid is not None:
                        get_synthesizer_blr_backend(blr_grid)

        return delta

    def _init_multiwavelength(self, spec, ssp_data):
        """Configure radio, X-ray, shock, and build wavelength grid.

        Returns
        -------
        dict[str, tuple[str, float, float]]
            Parameter map deltas for multiwavelength components.
        """
        self._uses_radio = getattr(spec, "radio", False)
        delta = {}

        if self._uses_radio:
            # Identity entries for radio_* now come from registry auto-derive
            # in _build_param_map (Step B).
            self._radio_include_freefree = getattr(spec, "radio_include_freefree", True)
            self._radio_sfr_mode = getattr(spec, "radio_sfr_mode", "bell2003")
            self._radio_agn_model = getattr(spec, "radio_agn_model", "powerlaw")

        self._uses_xray = getattr(spec, "xray", False)
        self._xray_model = getattr(spec, "xray_model", "yang20")

        # ── Master rest-wavelength grid (issue #463) ─────────────────
        #
        # Build the rest-frame wavelength grid as the sorted union of:
        #   (a) the SSP grid (fine UV–NIR sampling),
        #   (b) every attached component's native template grid (dust IR
        #       templates, AGN torus/disc libraries, …), and
        #   (c) analytic radio / X-ray extension wings for components that
        #       have no template grid but operate at extreme wavelengths.
        #
        # The native-grid registry lives in
        # ``tengri.forward.wavelength_extension``. Components that don't
        # declare a native grid (analytic dust models, IGM transmission, …)
        # contribute nothing and are correctly evaluated on whatever master
        # grid the orchestrator hands them. See ADR-comment in #463.
        from tengri.forward.wavelength_extension import collect_native_wavelength_grids
        from tengri.utils.wavelength import (
            RADIO_WAVE_MAX,
            XRAY_WAVE_MIN,
            make_union_grid,
        )

        component_grids = collect_native_wavelength_grids(
            dust_emission_model=getattr(self, "_dust_emission_model", None),
            nebular_model=getattr(self, "_nebular_model", None),
            agn_model=getattr(self, "_agn_model", None),
            agn_torus_block=getattr(self, "_agn_torus_block", None),
            agn_disc_block=getattr(self, "_agn_disc_block", None),
        )

        # Analytic radio/X-ray wings: only used when those components are
        # enabled AND nothing else already covers the extreme end of the
        # spectrum. ``make_union_grid`` deduplicates overlap so it's safe to
        # add these unconditionally when the flag is set.
        extra_wings: list[np.ndarray] = []
        ssp_min = float(np.asarray(ssp_data.ssp_wave).min())
        ssp_max = float(np.asarray(ssp_data.ssp_wave).max())

        if self._uses_xray and ssp_min > XRAY_WAVE_MIN:
            n_dec = np.log10(ssp_min) - np.log10(XRAY_WAVE_MIN)
            n_pts = max(int(n_dec * 20), 2)
            extra_wings.append(
                np.logspace(np.log10(XRAY_WAVE_MIN), np.log10(ssp_min), n_pts, endpoint=False)
            )

        if self._uses_radio:
            # Pick the longest wavelength reached by any template; extend the
            # radio wing past that point so the synchrotron tail has node
            # coverage even when dust templates don't already cover it.
            template_max = max((float(g.max()) for g in component_grids), default=ssp_max)
            radio_min = max(template_max, ssp_max)
            if radio_min < RADIO_WAVE_MAX:
                n_dec = np.log10(RADIO_WAVE_MAX) - np.log10(radio_min)
                n_pts = max(int(n_dec * 20), 2)
                extra_wings.append(
                    np.logspace(
                        np.log10(radio_min),
                        np.log10(RADIO_WAVE_MAX),
                        n_pts,
                        endpoint=True,
                    )[1:]
                )

        if component_grids or extra_wings:
            self._rest_wavelength = make_union_grid(
                ssp_data.ssp_wave,
                *component_grids,
                *extra_wings,
            )
        else:
            self._rest_wavelength = ssp_data.ssp_wave

        # Canonicalize to the session's working float dtype (#1206, #1439).
        # ``make_union_grid`` already builds at the working precision, but the
        # ``else`` branch hands back the SSP loader's float64 array verbatim,         # so under
        # ``jax.enable_x64(False)`` the grid's dtype depended on
        # whether some component happened to contribute a wing. That is not
        # cosmetic: thirteen precision gates in ``components/`` (AGN disc x6,
        # X-ray x2, radio, shock, ...) ask ``wave.dtype == jnp.float32`` to
        # decide whether to take their float32-safe log-domain path, and a
        # float64 grid makes every one of them fail *open* at once, the
        # float64 branch runs while the arithmetic is float32. Measured: a
        # composable AGN with no torus (nothing contributes a wing, so the
        # grid stays float64) evaluated the multicolor disc at the true
        # ``10**11 * L_sun`` = 3.8e44, past float32's 3.4e38, and returned
        # ``sed_agn`` NaN at every one of 5994 points. Adding a SKIRTOR torus
        # forced a union grid and the same model was clean, the bug was
        # reachable only through the component list, which is why no float32
        # test had caught it.
        #
        # Under x64 this is a no-op (``result_type(float)`` is float64 and the
        # grid already is), so float64 behavior is unchanged by construction.
        self._rest_wavelength = jnp.asarray(self._rest_wavelength, dtype=jnp.result_type(float))

        # Identity entries for shock_* and xray_* now come from registry
        # auto-derive in _build_param_map (Step B).
        self._uses_shock = getattr(spec, "shock", False)
        # Composable-shock config (#851): normalization mode + categorical
        # MAPPINGS knobs, threaded to the ShockNebular component config.
        self._shock_norm = getattr(spec, "shock_norm", "frac")
        self._shock_abundance = getattr(spec, "shock_abundance", "solar")
        self._shock_component = getattr(spec, "shock_component", "combined")

        return delta

    @staticmethod
    def _calibration_param_map(observation):
        """Identity param-map entries for spectroscopic calibration coefficients.

        Returns ``{cal_cN: (cal_cN, 1.0, 0.0)}`` for each coefficient the
        spectroscopy config declares (``calibration_order`` of them); empty when
        there is no spectroscopy or no calibration. The polynomial consumes the
        coefficients directly, so the mapping is a pure identity.
        """
        if observation is None or not observation.can_do_spectroscopy:
            return {}
        cal_params = observation.spectroscopy.get_calibration_params()
        return {name: (name, 1.0, 0.0) for name in cal_params}

    def _init_instrument(self, spec, observation):
        """Configure velocity dispersion and LSF settings."""
        self._has_sigma_v = spec.has_param("sigma_v") if hasattr(spec, "has_param") else False
        if not self._has_sigma_v:
            try:
                spec.get_distribution("sigma_v")
                self._has_sigma_v = True
            except KeyError:
                self._has_sigma_v = False

        if observation is not None and observation.can_do_spectroscopy:
            sc = observation.spectroscopy
            self._sigma_lib_kms = sc.sigma_lib_kms
            self._lsf_resolution = sc.resolution
            self._lsf_n_bins = sc.lsf_n_bins
        else:
            self._sigma_lib_kms = getattr(spec, "sigma_lib_kms", 0.0)
            self._lsf_resolution = getattr(spec, "lsf_resolution", None)
            self._lsf_n_bins = getattr(spec, "lsf_n_bins", 16)

    def _init_cosmology(self, spec):
        """Precompute luminosity distance if redshift is fixed."""
        redshift_dist = spec.get_distribution("redshift")
        if redshift_dist.is_fixed and self._catalog_z_range is None:
            self._dl_cm_fixed = luminosity_distance(redshift_dist.bounds[0])
            self._z_fixed = redshift_dist.bounds[0]
        else:
            # Catalog-fit mode (Approach A): even though redshift is Fixed
            # in the spec, treat it as a runtime input so different
            # per-galaxy values reuse the same compiled kernel. The
            # cosmology + IGM + filter-λ_eff paths fall back to their
            # already-existing free-redshift runtime branches.
            self._dl_cm_fixed = None
            self._z_fixed = None

    def _validate_and_freeze_param_map(self, param_map_deltas):
        """Merge param_map deltas, validate, and freeze the result.

        Parameters
        ----------
        param_map_deltas: list[dict[str, tuple[str, float, float]]]
            List of parameter map deltas from each _init_* method, in order.
            Each delta is a mapping: public_name -> (internal_name, scale, offset).

        Raises
        ------
        ParameterMapError
            If validation fails: missing free params, conflicting (scale, offset), etc.
        """
        # Merge all deltas in order (later entries override earlier ones for same key)
        merged = {}
        for delta in param_map_deltas:
            for public_name, (internal_name, scale, offset) in delta.items():
                if public_name in merged:
                    # Check for conflicting (scale, offset) claims
                    old_internal, old_scale, old_offset = merged[public_name]
                    if (old_internal, old_scale, old_offset) != (
                        internal_name,
                        scale,
                        offset,
                    ):
                        raise ParameterMapError(
                            f"Parameter '{public_name}' has conflicting mappings: "
                            f"({old_internal}, {old_scale}, {old_offset}) vs "
                            f"({internal_name}, {scale}, {offset})"
                        )
                merged[public_name] = (internal_name, scale, offset)

        # Validate: every free param in spec has an entry in the map
        free_params = self.spec.free_params
        missing = set(free_params) - set(merged.keys())
        if missing:
            raise ParameterMapError(
                f"The following free parameters in spec have no entry in the "
                f"parameter map: {sorted(missing)}. This indicates a mismatch "
                f"between what the spec declares as free and what the model "
                f"components registered."
            )

        # Freeze the merged map using MappingProxyType
        self._param_map = types.MappingProxyType(merged)

    def _warm_grid_caches(self) -> None:
        """Warm @functools.cache loaders to avoid tracer leaks from HDF5 grids.

        Some HDF5 grid loaders (@functools.cache decorators) construct jnp.array
        objects at load time. If the first call happens inside a JAX JIT trace,
        the jnp.array calls create DynamicJaxprTracers, which get cached and
        permanently leak into downstream code. This method calls each loader once
        OUTSIDE a JIT context so the cache stores concrete arrays instead.

        A grid file may legitimately be absent, so a failed warm degrades to the
        lazy path rather than blocking construction, but it warns, because the
        lazy path is precisely what raises ``UnexpectedTracerError`` later, far
        from this cause.
        """
        # MAPPINGS shock emission grids (nebular/shock.py:_load_mappings_grids)
        if self._uses_shock:
            try:
                from tengri.components.nebular.shock import _load_mappings_grids

                _load_mappings_grids()
            except (OSError, ImportError) as exc:
                _warn_grid_warm_failed("MAPPINGS shock emission", exc)

        # CAT3D-Wind AGN torus grids (agn/cat3d_wind.py:_load_cat3d_default)
        if self._agn_model == "cat3d_wind":
            try:
                from tengri.components.agn.cat3d_wind import _load_cat3d_default

                _load_cat3d_default()
            except (OSError, ImportError) as exc:
                _warn_grid_warm_failed("CAT3D-Wind AGN torus", exc)

        # Astrodust+PAH emission grid (dust/emission/templates/astrodust.py).
        # The faithful HD23 implementation self-loads its grid in load()/predict();
        # warm the process cache OUTSIDE any trace so an in-trace lazy load hits
        # concrete arrays rather than leaking a tracer (UnexpectedTracerError).
        # The grammar builds the component with the default template path (None).
        if self._dust_emission_model == "astrodust":
            try:
                from tengri.components.dust.emission.templates.astrodust import (
                    _cached_astrodust_grid,
                )

                _cached_astrodust_grid(None)
            except (OSError, ImportError) as exc:
                _warn_grid_warm_failed("Astrodust+PAH emission", exc)

    def _get_internal_params(self, params):
        """Translate public param dict to internal names with unit conversion.

        Thin wrapper around :func:`tengri.parameters.translate.get_internal_params`.
        """
        return get_internal_params(params, self._param_map, self.spec, self._uses_stochastic_sfh)

    def _get_redshift(self, params):
        """Get redshift value from params or fixed value."""
        if "redshift" in params:
            return params["redshift"]
        if self._z_fixed is not None:
            return self._z_fixed
        raise KeyError("Redshift not in params and not fixed in spec")

    def _get_dl_cm(self, params):
        """Get luminosity distance from params or precomputed value."""
        if self._dl_cm_fixed is not None:
            return self._dl_cm_fixed
        z = self._get_redshift(params)
        return luminosity_distance(z)

    def _get_sigma_v_kms(self, params):
        """Get stellar velocity dispersion sigma_v_kms from params.

        Returns a *traceable* value when ``sigma_v_kms`` is in the
        params dict (typical for spec fits with sigma_v as a free
        param) or the JAX/Python scalar from the spec's fixed
        distribution otherwise. Falls back to 0.0 when the parameter
        is absent. ``apply_lsf`` clamps via ``jnp.maximum`` so traced
        values flow through without breaking JIT.
        """
        if "sigma_v_kms" in params:
            return params["sigma_v_kms"]
        try:
            dist = self.spec.get_distribution("sigma_v_kms")
        except KeyError:
            return 0.0
        if dist.is_fixed:
            return float(dist.bounds[0])
        return 0.0

    # ── Core physics (SFH → SED pipeline) ─────────────────────────────

    def _compute_sfr(self, p):
        """Compute SFR via the composed SFH function.

        Single dispatch point for all SFH computation, replaces
        the old stochastic/parametric if/else branches.

        Parameters
        ----------
        p: dict
            Internal parameter dict from _get_internal_params().

        Returns
        -------
        array, shape (n_grid,)
            SFR(t) in Msun/yr on the log-age grid.
        """
        # Build kwargs for the composed SFH function
        kw = {
            k: v
            for k, v in p.items()
            if k in self._sfh_internal_names or k in self._sfh_public_names
        }

        # If field is present, compute GP and pass to composed fn
        if self._uses_stochastic_sfh and "xi" in p:
            gp_x, k0_half = compute_field_gp(
                xi=p["xi"],
                psd_sigma=p["psd_sigma"],
                psd_tau_yr=p["psd_tau_yr"],
                n_grid=self._n_grid,
                d_log_age=float(self.d_log_age),
                field_model=self._gp_kernel,
                log_age_grid=self.log_age_grid,
                centering=self._field_centering,
            )
            kw["gp_x"] = gp_x
            kw["k0_half"] = k0_half

        return self._sfh_fn(self.age_yr, **kw)

    def _compute_sfr_mean_and_full(self, p):
        """Compute both mean (no GP) and full (with GP) SFR.

        Used by predict_sfh which needs to return both.

        Returns
        -------
        sfr_mean: array
            SFR without GP modulation.
        sfr_full: array
            SFR with GP modulation (same as sfr_mean if no field).
        """
        kw = {
            k: v
            for k, v in p.items()
            if k in self._sfh_internal_names or k in self._sfh_public_names
        }
        sfr_mean = self._sfh_fn(self.age_yr, **kw)

        if self._uses_stochastic_sfh and "xi" in p:
            gp_x, k0_half = compute_field_gp(
                xi=p["xi"],
                psd_sigma=p["psd_sigma"],
                psd_tau_yr=p["psd_tau_yr"],
                n_grid=self._n_grid,
                d_log_age=float(self.d_log_age),
                field_model=self._gp_kernel,
                log_age_grid=self.log_age_grid,
                centering=self._field_centering,
            )
            kw["gp_x"] = gp_x
            kw["k0_half"] = k0_half
            sfr_full = self._sfh_fn(self.age_yr, **kw)
        else:
            sfr_full = sfr_mean

        return sfr_mean, sfr_full

    # ── Clone with a different approximation policy ────────────────────

    @property
    def approx(self) -> ApproxState:
        """The effective approximation state of this model.

        Answers "is a build-time look-up table live on this model?", the same
        question, spelled the same way, on
        :class:`~tengri.forward.forward_model.ForwardModel`.

        Returns
        -------
        ApproxState
            Frozen summary of the LUTs that resolved and activated. Falsy for an
            exact wave-grid model.

        Examples
        --------
        >>> model.approx.wave_precomp  # doctest: +SKIP
        True
        >>> if not model.approx:  # doctest: +SKIP
        ...     ...  # exact path

        Notes
        -----
        Reads the same lowered ``_approx`` flags the forward pipeline itself
        consumes, so it reports what the code *does*, not what was requested,         a
        ``SpectrumPrecomp`` that fell back to the exact path reports
        ``spectrum_precomp=False``. Deriving it from any other source would
        make this a third spelling of the question and free it to drift.

        Not JIT-relevant: Python-side introspection, never traced.
        """
        flags = self._approx or {}
        return ApproxState(
            wave_precomp=bool(flags.get("wave_precomp")),
            spectrum_precomp=bool(flags.get("spectrum_precomp")),
            feature_precomp=self._approx_config_feature is not None,
            ztable=bool(flags.get("ztable")),
            n_subbands=int(flags.get("n_subbands", 0) or 0),
        )

    def _has_modern_approx(self) -> bool:
        """Whether a build-time ``approx=`` LUT is active on this model.

        ``True`` when any of the WavePrecomp / SpectrumPrecomp / FeaturePrecomp
        precompute paths resolved and activated at construction; ``False`` for
        the exact wave-grid model (including a SpectrumPrecomp that fell back to
        the exact path because the spectral resolution was too high).
        """
        return (
            self._approx_config is not None
            or self._approx_config_spec is not None
            or self._approx_config_feature is not None
        )

    @property
    def approx_configs(self) -> tuple:
        """The **config objects** currently active, in ``approx=`` tuple form.

        Companion to :attr:`approx`, which reports *whether* each LUT is live as
        booleans. This returns the configs themselves, so a caller can add one
        family without discarding another's settings, rebuilding from
        ``WavePrecomp()`` because ``approx.wave_precomp`` was ``True`` would
        silently drop a configured ``catalog_z_range``, which is a behavioral
        change wearing a speedup's clothes.

        Returns
        -------
        tuple
            Active configs (``WavePrecomp`` / ``SpectrumPrecomp`` /
            ``FeaturePrecomp``), suitable to pass straight back to
            :meth:`with_approx`. Empty for an exact model.
        """
        return tuple(
            cfg
            for cfg in (
                self._approx_config_wave,
                self._approx_config_spec,
                self._approx_config_feature,
            )
            if cfg is not None
        )

    def with_approx(self, approx, *, observation=None):
        """Return a copy of this model built with a different ``approx`` policy.

        Parameters
        ----------
        approx: WavePrecomp or SpectrumPrecomp or FeaturePrecomp or tuple or None
            Approximation policy for the clone, with the same grammar as the
            ``approx=`` constructor argument: ``None`` for the exact wave-grid
            path, a single precompute config for one LUT family, or a composite
            tuple (at most one of each) such as ``(WavePrecomp(), FeaturePrecomp())``.
        observation: Observation, optional
            Observation for the clone. Defaults to this model's own. Passing a
            different one rebuilds the LUT against *its* filters, the seam
            :meth:`ForwardModel.build` uses to make its authoritative
            observation win (#1367, spec §5).

        Returns
        -------
        SEDModel
            A new model sharing this model's ``spec``, ``ssp_data``,
            ``observation`` and build settings, differing only in ``approx``
            (and ``observation`` when given). Returns ``self`` unchanged when
            ``approx=None`` is requested on a model that is already exact and
            no observation override is given (a no-op).

        Notes
        -----
        Building the clone only rebuilds the approximation LUT, which the
        ``tengri_precomp`` cache persists (content-hashed on SSP grid, filters,
        and z-grid), so repeat clones of the same combination are cheap. The
        inference layer uses this to fit on the fast LUT path while leaving the
        user's (exact) model untouched. **JIT-compatible**: build-time only.
        """
        if approx is None and observation is None and not self._has_modern_approx():
            return self
        return SEDModel(
            self.spec,
            self.ssp_data,
            observation=self.observation if observation is None else observation,
            forward_dtype=str(self._forward_dtype),
            csp_integration=str(self._csp_integration),
            wave_chunk_size=self._wave_chunk_size,
            agn_config=self._agn_config,
            compile=str(self._compile_mode),
            approx=approx,
        )

    # ── Predictions (public API) ──────────────────────────────────────

    def predict_sfh(self, params, n_linear=1000, grid="linear"):
        """Compute SFH on a uniform linear-time grid (plots) or the native log-age grid.

        Evaluates the SFH parameterization at ``n_linear`` evenly-spaced
        points in lookback time, returning both the smooth parametric
        component (``sfr_mean``) and the full SFH including GP-field
        modulation (``sfr_full``, if stochastic SFH enabled).

        **Raw forward-pass output** intended for plotting. For SFH-derived
        scalars (stellar mass, recent SFR, age), see
        ``model.predict(params).sfh.*`` or :meth:`predict_properties`
        for the JIT-compatible form.

        Parameters
        ----------
        params: dict
            Parameter values using public parameter names.
        n_linear: int, optional
            Number of output grid points, evenly spaced in lookback time.
            Default 1000 (sufficient for smooth visualization). Ignored when
            ``grid="native"``.
        grid: {"linear", "native"}, optional
            ``"linear"`` (default, backward compatible) resamples onto a uniform
            lookback-time grid for plotting. ``"native"`` returns the SFH on the
            model's own ``log_age_grid`` nodes, unresampled, use this for any
            QUANTITATIVE work (residuals, coverage, chi2 against a truth).

        Returns
        -------
        dict with keys:

            - ``"t_gyr"``: ndarray, shape (n_linear,) or (n_grid,).
              Lookback time [Gyr], from 0 (now) to ~13.8 (Big Bang).
            - ``"sfr_mean"``: ndarray, shape (n_linear,) or (n_grid,).
              Parametric mean SFR [M☉/yr] (no GP modulation).
            - ``"sfr_full"``: ndarray, shape (n_linear,) or (n_grid,).
              Full SFH including GP field [M☉/yr]. Identical to ``sfr_mean``
              if stochastic SFH not enabled.

        Notes
        -----
        **JIT-compatible**: no, uses Python-side interpolation. For
        JIT-compatible SFH evaluation, use :meth:`predict_properties`
        to get integrated quantities (stellar mass, age, etc.).

        **Time grid**: with ``grid="linear"`` the output is resampled onto a
        uniform linear-time (lookback) grid, not the internal log-age grid. This
        makes visualization cleaner, but it is **lossy at young ages and must not
        be used for quantitative scoring**. The step is
        ``age_max / n_linear``, at the default ``n_linear=1000`` and a 13.8 Gyr
        span that is 13.8 Myr, so a 16-node log-age grid whose five youngest
        nodes all lie below 15 Myr collapses into ~2 samples there. Resampling
        also interpolates *linearly between log-age nodes*, so a log-axis plot
        shows corners at the nodes; that is the interpolant, not the model.

        Scoring an SFH residual on the linear grid silently reweights it: every
        megayear counts equally, so 15-500 Myr swamps the <15 Myr bins where
        emission lines carry nearly all of their information. Measured on the
        field-SFH recovery study, that reweighting turned a real +54% improvement
        from adding line fluxes into an apparent 0%. Pass ``grid="native"`` for
        residuals, coverage, or any comparison against a known truth.

        **SFH mean vs. full**: When correlated-field (stochastic) SFH is enabled,
        ``sfr_mean`` shows the smooth parametric trend (e.g., exponential
        decline), while ``sfr_full`` adds GP modulation for realistic burstiness.
        If parametric-only SFH is used, they are identical.

        **Physical units**: Output SFR is in M☉/yr. Lookback time is in Gyr
        (cosmic time before today).

        Examples
        --------
        >>> sfh = model.predict_sfh(params)
        >>> print(sfh.keys())
        dict_keys(['t_gyr', 'sfr_mean', 'sfr_full'])
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(sfh["t_gyr"], sfh["sfr_mean"], label="Smooth")
        >>> if "sfr_full" in sfh:
        ...     plt.plot(sfh["t_gyr"], sfh["sfr_full"], alpha=0.5, label="With bursts")

        See Also
        --------
        predict_properties: Integrated SFH quantities, JIT/vmap-safe.
        predict: Lazy access to SFH and all derived quantities.
        """
        if grid not in ("linear", "native"):
            raise ValueError(f"grid must be 'linear' or 'native', got {grid!r}")

        p = self._get_internal_params(params)
        sfr_mean, sfr_full = self._compute_sfr_mean_and_full(p)

        if grid == "native":
            return {
                "t_gyr": jnp.asarray(10.0**self.log_age_grid) / 1e9,
                "sfr_mean": sfr_mean,
                "sfr_full": sfr_full,
            }

        t_gyr_mean, sfr_mean_lin = interpolate_to_linear_time(
            self.log_age_grid, sfr_mean, n_linear
        )
        _, sfr_full_lin = interpolate_to_linear_time(self.log_age_grid, sfr_full, n_linear)

        return {
            "t_gyr": t_gyr_mean,
            "sfr_mean": sfr_mean_lin,
            "sfr_full": sfr_full_lin,
        }

    def predict_rest_sed(self, params, wave=None):
        """Deprecated. Use ``model.predict(params).rest_sed()``.

        .. deprecated:: 2026-07
           Superseded by the property catalog and the ``Prediction`` surface
           (#1043 contract §2). The body is unchanged, this shim is bit-exact
           with the method it replaces, so migrating changes no number.
           Will be removed in tengri v1.0.

        Returns
        -------
        Same as :meth:`_predict_rest_sed`.
        """
        warnings.warn(
            "predict_rest_sed() is deprecated, use model.predict(params).rest_sed() "
            "instead (cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._predict_rest_sed(params, wave=wave)

    def _predict_rest_sed(self, params, wave=None):
        """Compute rest-frame panchromatic SED luminosity spectrum.

        Evaluates all stellar populations, emission (nebular, AGN), and
        multi-wavelength (radio, X-ray) components in rest-frame coordinates.
        Returns the total SED integrated across the age distribution set by
        the SFH and stellar mass parameters.

        **Raw forward-pass output.** Returns ``(wavelength, sed)``.
        For interactive exploration with cached derived quantities (stellar
        mass, SFR, indices, line ratios), use :meth:`predict` and access
        ``pred.sed`` properties.

        Parameters
        ----------
        params: dict
            Parameter values using public parameter names.
        wave: array, optional
            Custom rest-frame wavelength grid [Angstrom]. If None,
            uses the model's default: SSP wavelength grid
            (``ssp_data.ssp_wave``), or auto-extended grid if
            ``radio=True`` or ``xray=True`` in spec.

        Returns
        -------
        SEDResult
            NamedTuple with:

            - ``wavelength``: array, shape (n_wave,). Rest-frame wavelength [Ångstrom]
            - ``sed``: array, shape (n_wave,). Spectral luminosity density [erg/s/Hz]

        Notes
        -----
        **JIT-compatible**: no, computes SED components via the
        orchestrator path (:meth:`predict_state`) which is not
        JIT'd. For JIT-compatible SED access, use
        :meth:`predict_sed_quantities` instead.

        **Physical units**:

        - Wavelength: rest-frame Ångstrom (not redshifted)
        - SED: erg/s/Hz (L_ν), normalized to the total stellar mass
          implied by the SFH

        **SED components**: Total SED is the sum of:

        - Stellar continuum (CSP from SSP integration)
        - Nebular continuum (if nebular_mode ≠ 'baked-in')
        - Nebular emission lines (if ``neb_*`` params free)
        - AGN continuum (if ``agn_model`` set)
        - Dust attenuation (applied to stellar + AGN)
        - Dust emission (re-radiated IR, if dust_emission_model set)
        - Shock emission (if ``shock=True``)
        - Radio/X-ray (if ``radio=True`` or ``xray=True``)

        **Attenuation**: Applied via two-component (birth cloud + diffuse ISM)
        or single-screen dust law, parameterized by age-dependent optical depth.
        See ``components.dust`` for available laws.

        Examples
        --------
        >>> sed = model._predict_rest_sed(params)
        >>> import matplotlib.pyplot as plt
        >>> plt.loglog(sed.wavelength, sed.sed)
        >>> plt.xlabel("Rest-frame wavelength (Angstrom)")
        >>> plt.ylabel("SED (erg/s/Hz)")

        See Also
        --------
        predict_obs_sed: Observed-frame SED (redshifted + IGM).
        predict_sed_quantities: JIT-compatible SED-derived quantities.
        """
        from tengri.forward.result import SEDResult

        state = self.predict_state(params)
        if wave is None:
            # Use ``state.wave`` (the orchestrator's runtime wavelength
            # grid, which may differ from ``self._rest_wavelength``,             # e.g. when
            # radio/xray extends the SSP grid panchromatically
            # but the orchestrator hasn't been wired to that extension
            # yet). Mismatched shapes would otherwise break boolean
            # masking on (wavelength, sed) pairs in test_panchromatic_*.
            return SEDResult(wavelength=state.wave, sed=state.sed_intrinsic)
        # Custom rest-frame wavelength grid: interpolate the orchestrator's
        # SED onto it. Pure post-processing, keeps the orchestrator's
        # internal grid contract (state.wave / state.derived[...]) clean,
        # at the same accuracy a user gets from
        # ``np.interp(custom_wave, ssp_wave, sed)``.
        wave_target = jnp.asarray(wave)
        sed_interp = jnp.interp(wave_target, state.wave, state.sed_intrinsic)
        return SEDResult(wavelength=wave_target, sed=sed_interp)

    def predict_obs_sed(self, params, wave=None):
        """Deprecated. Use ``model.predict(params).obs_sed()``.

        .. deprecated:: 2026-07
           Superseded by the property catalog and the ``Prediction`` surface
           (#1043 contract §2). The body is unchanged, this shim is bit-exact
           with the method it replaces, so migrating changes no number.
           Will be removed in tengri v1.0.

        Returns
        -------
        Same as :meth:`_predict_obs_sed`.
        """
        warnings.warn(
            "predict_obs_sed() is deprecated, use model.predict(params).obs_sed() "
            "instead (cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._predict_obs_sed(params, wave=wave)

    def _predict_obs_sed(self, params, wave=None):
        """Compute observed-frame SED (redshifted + IGM + DLA transmission).

        Evaluates the rest-frame SED, redshifts to observed frame
        (wavelength × (1+z)), and applies IGM and DLA absorption where
        configured. At z=0, identical to :meth:`predict_rest_sed`.

        **Raw forward-pass output.** Returns ``(wavelength, sed)`` in the
        observed frame. For interactive use with derived quantities, see
        :meth:`predict`.

        Parameters
        ----------
        params: dict
            Parameter values using public parameter names.
        wave: array, optional
            Custom rest-frame wavelength grid [Angstrom] before redshifting.
            If None, uses model default.

        Returns
        -------
        SEDResult
            NamedTuple with:

            - ``wavelength``: array, shape (n_wave,).
              Observed-frame wavelength [Ångstrom]
            - ``sed``: array, shape (n_wave,).
              Observed-frame spectral luminosity density [erg/s/Hz]

        Notes
        -----
        **JIT-compatible**: no, delegates to :meth:`predict_rest_sed`.

        **IGM absorption**: Applies transmission via
        :math:`T_{\\mathrm{IGM}}(\\lambda_{\\mathrm{obs}}, z)` when ``igm=True`` in spec.
        Uses Inoue+2014 [1]_ mean IGM with optional extensions for:

        - Reionization epoch: CGM damping wing (Asada+2025 [2]_)
        - Patchy reionization: parameterized neutral fraction (Mason+2018 [3]_)

        **CRITICAL GOTCHA**: IGM transmission takes **observed-frame** wavelengths
        as input. The redshifted ``wavelength`` in this SED is already in observed
        frame, so ``igm_transmission(wave_obs, z)`` is called correctly.

        **DLA absorption**: Applies Lyman-series damping wing when ``dla=True``.
        Parameterized by neutral column density log₁₀(N_HI) and temperature.
        See :func:`~tengri.components.igm.dla.dla_transmission_obs`.

        **Physical units**:

        - Wavelength: observed-frame Ångstrom (redshifted)
        - SED: erg/s/Hz (same as rest-frame), but now at redshifted
          wavelengths and reduced intensity by :math:`(1+z)` factor from
          cosmological redshift

        Examples
        --------
        >>> sed_obs = model._predict_obs_sed(params)
        >>> # IGM and redshift already applied
        >>> print(f"z={params['redshift']}: wavelength {sed_obs.wavelength[0]:.0f} Å")

        See Also
        --------
        predict_rest_sed: Rest-frame SED (before redshift/IGM).
        predict_photometry: Filter-integrated observed flux (uses this internally).

        References
        ----------
        .. [1] A. K. Inoue et al., "An updated analytic model for attenuation
           by the intergalactic medium," MNRAS, 442, 1805 (2014).
           arXiv:1402.0677. https://doi.org/10.1093/mnras/stu936
        .. [2] Y. Asada et al., "Improving Photometric Redshifts of Epoch of
           Reionization Galaxies: A New Empirical Transmission Curve with
           Neutral Hydrogen Damping Wing Ly-alpha Absorption," ApJL, 983, L2
           (2025). arXiv:2410.21543.
           https://doi.org/10.3847/2041-8213/adc388
        .. [3] C. A. Mason et al., "The Universe Is Reionizing at z ~ 7:
           Bayesian Inference of the IGM Neutral Fraction Using Ly-alpha
           Emission from Galaxies," ApJ, 856, 2 (2018).
           https://doi.org/10.3847/1538-4357/aab0a7
        """
        from tengri.forward.result import SEDResult

        rest_result = self._predict_rest_sed(params, wave=wave)
        z = self._get_redshift(params)
        wave_obs = rest_result.wavelength * (1.0 + z)
        sed_obs = rest_result.sed
        if self._uses_igm or self._uses_dla:
            from tengri.components.igm.igm import igm_absorption

            # One flat dispatch: mean-IGM model (or 'none' when only a DLA is
            # requested) plus the optional DLA absorber. This is the same call
            # the IGMSEDComponent projection makes, so predict_obs_sed and
            # photometry/spectroscopy stay consistent (#932). Transmission is
            # all-ones at z=0, so no z>0 comparison is needed under JIT.
            transmission = igm_absorption(
                wave_obs,
                z,
                igm_x_HI=params.get("igm_x_HI", 0.0),
                igm_bubble_mpc=params.get("igm_bubble_mpc", 10.0),
                igm_patchy=getattr(self, "_igm_patchy", False),
                igm_model=self._igm_model if self._uses_igm else "none",
                use_dla=self._uses_dla,
                dla_z=params.get("dla_z", 0.0),
                dla_log_n_hi=params.get("dla_log_n_hi", 20.0),
                dla_temp=params.get("dla_temp", 1e4),
                dla_b_turb=params.get("dla_b_turb", 0.0),
            )
            sed_obs = sed_obs * transmission
        # MW foreground screen (#297), final transformation on the
        # observed-frame SED, independent of host-galaxy dust. Applied
        # at observed-frame wavelengths so it works for any source
        # redshift. Skipped when ebmv_mw=0 (the default).
        if getattr(self.spec, "foreground_ebmv_mw", 0.0) > 0.0:
            from tengri.components.dust.attenuation import cardelli

            ebmv = jnp.asarray(self.spec.foreground_ebmv_mw)
            rv = jnp.asarray(self.spec.foreground_rv)
            # Cardelli returns A(λ)/A(V); A_V = R_V * E(B-V).
            # T(λ) = 10^(-0.4 * A_V * k(λ))
            k_lambda = cardelli(wave_obs, dust_Rv=rv)
            a_v = rv * ebmv
            sed_obs = sed_obs * jnp.power(10.0, -0.4 * a_v * k_lambda)
        return SEDResult(wavelength=wave_obs, sed=sed_obs)

    def predict(self, params):
        """Create a lazy prediction object for all derived physical quantities.

        Returns a :class:`Prediction` object that computes and caches
        derived quantities on first access. This is the recommended API
        for interactive exploration of a single galaxy's properties,
        trading speed for convenience.

        For batch computation over posterior chains or mock catalogs, use
        :meth:`predict_properties`, the one JIT/vmap-safe surface for
        derived quantities, with :func:`jax.vmap` instead (up to 1000×
        faster for large batches).

        Parameters
        ----------
        params: dict
            Parameter values using public parameter names.

        Returns
        -------
        Prediction
            Lazy caching wrapper with property groups:

            - ``.sfh``: SFH-derived quantities (stellar mass, SFR, age, metallicity)
            - ``.sed``: SED-derived quantities (luminosities, colors, indices)
            - ``.lines``: Emission line properties (luminosities, fluxes, ratios)
            - ``.radio``: Radio SED properties (if ``radio=True``)
            - ``.xray``: X-ray SED properties (if ``xray=True``)
            - ``.ionizing``: Ionizing photon budget properties

        Notes
        -----
        **Not JIT-compatible**: Uses Python-side caching and object
        attribute access. Useful for interactive exploration, not
        for inference loops. For inference, use
        :meth:`predict_photometry` (the hot path) or
        :meth:`predict_properties` with :func:`jax.vmap`.

        **Lazy evaluation**: Quantities are computed only when accessed.
        Repeated access to the same property reuses cached results.
        This is transparent to the user.

        **NaN handling**: Some quantities (e.g., ``stellar_mass_surviving``,
        ``l_dust_absorbed``) may return NaN if required data/parameters
        unavailable (e.g., no mass-remaining table, dust_model='none').
        The Prediction object handles NaN gracefully (returns None when
        data required to compute the quantity is absent).

        Examples
        --------
        **Single-galaxy exploration (lazy, on-demand):**

        >>> pred = model.predict(params)
        >>> pred.sfh.stellar_mass  # triggers SFH computation, caches result
        Array(1.23e10, dtype=float64)
        >>> pred.sfh.mass_weighted_age_gyr  # reuses cached SFH
        Array(2.34, dtype=float64)
        >>> pred.sed.l_bol  # triggers SED computation
        Array(2.5e10, dtype=float64)
        >>> pred.sed.uv_slope_beta  # reuses cached SED
        Array(-1.8, dtype=float64)
        >>> pred.lines.halpha  # triggers nebular computation
        Array(4.23e-15, dtype=float64)

        **Batch computation (JIT-compatible, faster for large N):**

        >>> import jax
        >>> params_batch = spec.sample(jax.random.PRNGKey(0), n=10000)
        >>> sfh_fn = jax.vmap(lambda p: model.predict_properties(p, names=("stellar_mass",)))
        >>> sfh_batch = sfh_fn(params_batch)
        >>> sfh_batch["stellar_mass"].shape  # (10000,)
        >>> sfh_batch["stellar_mass"].mean()

        See Also
        --------
        predict_properties: JIT/vmap-safe derived quantities for batch.
        :meth:`Prediction.rest_sed`: Full rest-frame SED for custom analysis.
        :attr:`Prediction.lines`: Emission-line luminosities.
        """
        from collections.abc import Mapping

        from tengri.forward.prediction import Prediction

        if not isinstance(params, Mapping):
            raise TypeError(
                f"model.predict() expects a params dict (e.g. from "
                f"spec.sample(key)), got {type(params).__name__}."
            )
        # Validate eagerly, matching predict_photometry, so a typo'd or
        # missing free parameter raises a helpful error here instead of a bare
        # KeyError deferred to the first accessor of the lazy Prediction. Both
        # asymmetries were flagged as silent-wrong footguns in a fresh-user audit.
        check_unknown_params(params, self._param_map)
        check_missing_free_params(params, self.spec, self._param_map)
        return Prediction(self, params)

    def compile_signature(self) -> tuple:
        """Return a hashable signature identifying JIT-graph shape and structure.

        Two SEDModel instances with the same compile_signature() will produce
        identical XLA compilation graphs (for identical Fitter configurations),
        enabling cross-galaxy engine reuse in PopulationFitter and CatalogFitter.

        The signature captures every JIT-affecting field: SSP array shapes,
        filter grid dimensions, dust/AGN/nebular model identities, and all
        configuration flags that determine the control flow during inference.

        Returns
        -------
        tuple
            Hashable immutable signature. Entries are immutable types
            (int, str, tuple, bool, None) or tuples thereof.

        Notes
        -----
        This signature is used by Fitter._get_or_build_engine to key the
        module-level _SHARED_ENGINE_CACHE. Changes to SEDModel initialization
        that affect JIT graph shape MUST be added to this method to avoid
        silent miscompilation.

        "Structure" includes **precision**. The structural-kernel cache in
        :meth:`_get_or_build_predict_observables_jit` returns a closure that
        captured ``self``, so a signature collision hands one model's compiled
        kernel, and its wavelength grid, to another. A float64/float32 collision
        used to reach the components as a float64 ``wave`` under
        ``jax.enable_x64(False)``, which switched off every dtype-keyed float32
        path downstream and produced NaN gradients with nothing raised (#1392).
        See ``build_precision`` below.
        """
        # SSP grid shapes (n_met, n_age, n_wave)
        ssp_flux_shape = tuple(self.ssp_data.ssp_flux.shape)
        ssp_lgmet_shape = tuple(self.ssp_data.ssp_lgmet.shape)

        # SSP metallicity grid VALUES (not just shape).
        # Hybrid and compositional kernels close over actual ssp_lgmet values,
        # so two models with same shape but different grids must have different signatures.
        ssp_lgmet_array = np.asarray(self.ssp_data.ssp_lgmet)
        ssp_lgmet_id = (
            int(ssp_lgmet_array.tobytes().__hash__())
            if hasattr(ssp_lgmet_array, "tobytes")
            else hash(tuple(map(float, ssp_lgmet_array)))
        )

        # SSP flux grid CONTENT (not just shape/lgmet).
        # The inference closure bakes the SSP grid, so two models with same
        # shape/lgmet but different ssp_flux must have different signatures
        # (#1973). Two SSP grids "of identical shape" is the scenario that
        # _get_or_build_engine's docstring advertised as safe sharing; it is
        # the scenario that produces +1 dex stellar mass errors when the second
        # model silently runs the first model's physics.
        # Content-hashed once per grid and cached on the SSPData instance;
        # cost is 17-100 ms first call (depending on grid size), zero for
        # subsequent calls on the same object.
        from tengri.components.stellar.sps.dsps_wrapper import get_ssp_content_hash

        ssp_flux_id = get_ssp_content_hash(self.ssp_data)

        # Alpha-Fe enhancement presence
        has_alpha_fe = hasattr(self.ssp_data, "ssp_alpha_fe")

        # Filter grid dimensions
        n_filters = len(self.filter_waves) if self.filter_waves is not None else 0
        filter_wave_shape = tuple(self.filter_waves[0].shape) if self.filter_waves else ()
        filter_trans_dtype = str(self.filter_trans[0].dtype) if self.filter_trans else "none"

        # Filter transmission VALUES (not just dtype).
        # Hybrid kernels close over actual filter_trans curves, so two models with
        # same dtype but different filter profiles must have different signatures.
        if self.filter_trans is not None and self.filter_trans:
            filter_trans_id = hash(tuple(np.asarray(t).tobytes() for t in self.filter_trans))
        else:
            filter_trans_id = "none"

        # Filter-convolution convention (ADR-0017). The photometry channel
        # closes over it, so models that differ only in convention must not
        # share a compiled observables closure.
        _phot = getattr(self.observation, "photometry", None)
        phot_convention = str(getattr(_phot, "convention", FilterConvention.BESSELL))

        # Dust configuration
        dust_model = str(self._dust_model)
        dust_scheme = str(self._dust_scheme)
        dust_emission_model = str(self._dust_emission_model or "none")

        # Astrodust+PAH (HD23) configuration: spinning dust (AME) enable flag
        # and cold-neutral-medium filling fraction. These affect the emitted
        # SED shape without changing the graph structure, so they must be
        # keyed to prevent silent cache collisions (#1093).
        astrodust_spinning_dust = bool(getattr(self, "_astrodust_spinning_dust", False))
        astrodust_f_cnm = float(getattr(self, "_astrodust_f_cnm", 0.28))

        # WG00 (dust_type=3) structural selectors. Different geometry / dust
        # curve / local structure tabulate distinct attenuation curves, so each
        # combination must get its own compiled kernel. "none" when unused.
        wg00_selectors = (
            (
                str(getattr(self, "_wg00_dust_curve", "mw")),
                str(getattr(self, "_wg00_geometry", "shell")),
                str(getattr(self, "_wg00_structure", "homogeneous")),
            )
            if dust_model == "wg00"
            else ("none",)
        )

        # Dust law functions (by name to avoid closure capture)
        dust_law_bc_fn_name = self._dust_law_bc_fn.__name__ if self._dust_law_bc_fn else "none"
        dust_law_diff_fn_name = (
            self._dust_law_diff_fn.__name__ if self._dust_law_diff_fn else "none"
        )
        # Nebular birth-cloud law (None -> inherits bc). It reddens only the
        # nebular continuum, so a change is invisible to the stellar graph
        # shape; it MUST enter the signature or the kernel cache leaks one
        # model's nebular reddening into another (color-leak).
        dust_law_neb_name = str(getattr(self, "_dust_law_neb", None) or "inherit_bc")
        # Per-component law-parameter overrides change the baked-in chain
        # constants (e.g. birth-cloud n_slope) but not its graph shape, so two
        # models that differ only here MUST get distinct signatures or the
        # kernel cache leaks one's attenuation into the other (color-leak).
        _ovr = getattr(self, "_dust_law_overrides", None) or {}
        dust_law_overrides_sig = tuple(
            (comp, tuple(sorted((k, float(v)) for k, v in (_ovr.get(comp) or {}).items())))
            for comp in ("bc", "diff", "neb")
        )
        # Lyman-limit clip zeros the FUV curve but leaves the graph shape
        # unchanged, so two models that differ only here MUST get distinct
        # signatures or the kernel cache leaks one's FUV attenuation into the
        # other (color-leak), exactly like ``dust_law_overrides_sig`` above.
        dust_lyman_cutoff_sig = float(getattr(self, "_dust_lyman_cutoff_aa", 0.0))
        # Young-only vs absorb-all stellar LyC changes the baked below-912 chain
        # output but not its graph shape -> must enter the signature (color-leak).
        dust_lyc_absorb_all_sig = bool(getattr(self, "_dust_lyc_absorb_all", False))
        # LyC-in-energy-balance (FSPS parity, #961) rescales L_IR without
        # changing the graph shape -> must enter the signature (color-leak).
        dust_eb_include_lyc_sig = bool(getattr(self, "_dust_eb_include_lyc", False))

        # Nebular backend (by class name)
        nebular_backend_name = (
            type(self._nebular_backend).__name__ if self._nebular_backend is not None else "none"
        )

        # IGM configuration
        uses_igm = bool(self._uses_igm)
        igm_model = str(self._igm_model or "none")
        uses_dla = bool(self._uses_dla)

        # AGN configuration.
        #
        # ``agn_model`` carries no discriminating power on its own: the
        # composable surface is the only non-deprecated one, so
        # ``list_agn_models()`` returns exactly one selectable entry and every
        # composable model hashes to the same string. The six block selectors
        # ARE the AGN axis, and each one swaps the emitting physics (a torus
        # library, a disc SED, an NLR/BLR line set) without changing the graph
        # shape, so omitting them left the entire axis unkeyed and the
        # first-built kernel won, exactly like the fixed-z case below (#1450).
        # Measured: torus='skirtor' vs 'cat3d_wind' agreed bit-for-bit within a
        # process and disagreed by 60% in W4 across processes, depending only
        # on build order.
        agn_model = str(self._agn_model or "none")
        agn_luminosity_mode = bool(self._agn_luminosity_mode)
        agn_blocks = (
            str(getattr(self, "_agn_disc_block", "none") or "none"),
            str(getattr(self, "_agn_torus_block", "none") or "none"),
            str(getattr(self, "_agn_nlr_block", "none") or "none"),
            str(getattr(self, "_agn_blr_block", "none") or "none"),
            str(getattr(self, "_agn_feii_block", "none") or "none"),
            str(getattr(self, "_agn_attenuation_block", "none") or "none"),
        )
        # Cross-block normalization policy (#556). 'cigale_joint' ties
        # disc/torus/polar to one energy-conserving reference; 'independent'
        # puts each on its own luminosity scale. Same graph, different emitted
        # SED, a signature entry, not a flag.
        agn_norm = str(getattr(self, "_agn_norm", "cigale_joint") or "cigale_joint")

        # Radio and X-ray
        uses_radio = bool(self._uses_radio)
        uses_xray = bool(self._uses_xray)
        # WHICH X-ray model, not merely whether one is attached. ``_xray_model``
        # was stored at construction but never keyed, so `agn_xray_corona` and
        # `xray_aird` shared a compiled kernel and the first one built won,         # the same
        # class as the AGN block selectors (#1450) and the radio
        # models beside it, which do carry their selector. The collision is
        # invisible in optical/IR photometry because X-ray emission lands at
        # keV, which is why a flux-based sweep reads this axis as "inert"
        # rather than unkeyed; the signature shows it directly (#1462).
        xray_model = str(getattr(self, "_xray_model", "none") or "none") if uses_xray else "none"
        uses_shock = bool(self._uses_shock)
        # Shock normalization + categorical knobs change the emitted SED, so
        # they are part of the structural fingerprint (#851).
        shock_cfg = (
            str(getattr(self, "_shock_norm", "frac")),
            str(getattr(self, "_shock_abundance", "solar")),
            str(getattr(self, "_shock_component", "combined")),
        )

        # SFH configuration
        mean_sfh_type = str(self.spec.mean_sfh_type)
        met_mode = str(self._met_mode)
        stochastic = bool(self.spec.stochastic)
        n_grid = int(self._n_grid)
        # SFH→SSP age-weight kernel (#964). "cic" and "dsps" produce different
        # age weights on the SAME graph shape, so without this entry two models
        # differing only in ``age_kernel`` share a compiled kernel and the
        # second silently returns the first's photometry.
        age_kernel = str(getattr(self.spec, "age_kernel", None) or "auto")
        # Non-parametric SFH bin edges (#1975). Exactly the ``age_kernel``
        # hazard: custom edges change the age weights on the SAME graph shape,
        # so without this entry two models differing only in their bin layout
        # share a compiled kernel and the second silently returns the first's
        # photometry. Hashed by value; "default" is the model's own ladder.
        _bin_edges = getattr(self.spec, "bin_edges_gyr", None)
        sfh_bin_edges = (
            "default"
            if _bin_edges is None
            else hash(tuple(map(float, np.asarray(_bin_edges).ravel())))
        )
        # GP-field parameterization (#1355). Same hazard as ``age_kernel``: a
        # different ``centering`` changes the xi -> SFH map without changing the
        # graph shape, so without this entry two models differing only in
        # ``field_centering`` share a compiled kernel and the second returns the
        # first's photometry, which would make the A/B this knob exists for
        # report a null result.
        field_centering = round(float(getattr(self.spec, "field_centering", 1.0)), 8)

        # Alpha-Fe evolution
        alpha_fe_evolving = bool(self._alpha_fe_evolving)

        # Redshift configuration. The actual fixed-z value is part of the
        # structural fingerprint because the compiled kernels close over
        # ``_dl_cm_fixed``, ``_igm_fn`` precomputed tables, and effective
        # rest wavelengths, all derived from ``_z_fixed`` at construction.
        # Without the value, two models at different fixed z would share
        # a cached kernel and produce identical photometry (the kernel
        # built first wins). Float is rounded to a stable hash key.
        z_fixed = (
            ("fixed", round(float(self._z_fixed), 8)) if self._z_fixed is not None else ("free",)
        )
        # Catalog-fit reuse: the explicit range is part of the signature
        # so two models with different catalog ranges don't share a
        # compiled kernel (their ztable shape can differ).
        catalog_z_range = (
            ("catalog", round(self._catalog_z_range[0], 8), round(self._catalog_z_range[1], 8))
            if self._catalog_z_range is not None
            else ("none",)
        )

        # Instrument/spectroscopy
        has_spectroscopy = self.observation is not None and self.observation.can_do_spectroscopy
        if has_spectroscopy:
            spec_wave_shape = tuple(self.observation.spectroscopy.wave_obs.shape)
            sigma_lib_kms = float(self._sigma_lib_kms)
            lsf_resolution = self._lsf_resolution
            # The calibration order is structural: the compiled kernel closes over
            # an ``Observation`` whose projector reads ``cal_c1..cN`` out of the
            # param dict. Two models differing ONLY in ``calibration_order`` must
            # not share a cache slot, the second would inherit the first's
            # coefficient lookup and either apply a calibration it was never given
            # or raise ``KeyError: 'cal_c1'`` on a dict that rightly has no such key.
            calibration_order = int(self.observation.spectroscopy.calibration_order)
            # The RESOLVED resample decision (#1166): the spectrum projector closes
            # over whether it point-samples or flux-conservingly integrates the
            # model onto the pixels. Two models differing only in ``resample`` (or
            # in an ``"auto"`` decision that lands differently for their grids) must
            # NOT share a compiled kernel, otherwise the second silently inherits
            # the first's resampler. Keyed on the resolved bool, not the mode
            # string, so ``"auto"`` collides only with an explicit mode that
            # actually resamples the same way.
            spec_resample_conserving = bool(
                self.observation.spectroscopy.resolve_conserving(self.wavelengths)
            )
            # The banded resolution matrix (#1163) is structural: the spectrum
            # projector closes over whether it applies ``R @ model`` or the
            # Gaussian ``apply_lsf``. Two models differing only in
            # ``resolution_matrix`` must NOT share a compiled kernel, else the
            # second silently inherits the first's projector and drops (or
            # wrongly reuses) the matrix. Keyed on presence + band shape, which
            # is all the structural cache needs. Same cache-collision class as
            # #1135/#1149/#1166.
            _rm = self.observation.spectroscopy.resolution_matrix
            spec_resolution_matrix = (
                tuple(jnp.asarray(_rm.data).shape) if _rm is not None else None
            )
        else:
            spec_wave_shape = ()
            sigma_lib_kms = 0.0
            lsf_resolution = None
            calibration_order = 0
            spec_resample_conserving = False
            spec_resolution_matrix = None

        # csp_integration is deliberately NOT part of the signature. Every value
        # produces an identical program (#1500), so including it split the compile
        # cache five ways and recompiled the whole model to compute the same
        # numbers. Measured: 5 distinct signatures, 0 differing outputs.

        # ``forward_dtype`` is deliberately NOT part of this key (#1433). It is
        # retired and casts nothing, so two models differing only in it compute
        # bit-identical results, keying on it bought a second compile of an
        # identical kernel and nothing else. Anyone who wires it must put it back
        # here in the same change, or the two precisions will share a kernel.

        # Effective build precision (#1392). ``forward_dtype`` stays
        # "float64" in a **pure** float32 run (which is entered with
        # ``jax.enable_x64(False)``, not with that knob), so on its own it cannot
        # separate a float64 model from a float32 one, and since it casts nothing
        # (#1433) it could not do so at any setting.
        # It must: ``_get_or_build_predict_observables_jit`` caches a closure that
        # captured ``self``, keyed on this signature, so without a precision entry
        # a float32 model is handed the float64 model's kernel, carrying that
        # model's float64 wave grid. Every float32 gate downstream keys on a dtype
        # and so switches itself off, silently, producing NaN gradients rather than
        # an error (observed in the AGN block: #1392).
        build_precision = (
            str(self._rest_wavelength.dtype),
            bool(jax.config.jax_enable_x64),
        )

        # Metallicity interpolation mode
        met_interp = str(self._met_interp)
        z_interp = str(self._z_interp)

        # Radio-specific flags
        radio_include_freefree = (
            bool(self._radio_include_freefree)
            if hasattr(self, "_radio_include_freefree")
            else False
        )
        radio_sfr_mode = str(self._radio_sfr_mode) if hasattr(self, "_radio_sfr_mode") else "none"
        radio_agn_model = (
            str(self._radio_agn_model) if hasattr(self, "_radio_agn_model") else "powerlaw"
        )

        # Velocity dispersion
        has_sigma_v = bool(self._has_sigma_v)

        # Compile mode
        compile_mode = str(self._compile_mode)

        # Approximation settings, resolved and sorted.
        # 2026-05-20: include the resolved WavePrecomp configuration
        # so two models with different ztable sampling (n_z / z_min / z_max)
        # get distinct cache slots. Without this, ``WavePrecomp(n_z=100)`` and
        # ``WavePrecomp(n_z=200)`` would collide and the second galaxy would
        # reuse the first's stale compiled LUT.
        approx_resolved_flags = tuple(
            sorted((k, bool(v)) for k, v in (self._approx or {}).items() if isinstance(v, bool))
        )

        # The sub-band quadrature order changes the compiled kernel and the numbers
        # it produces (#1122). It is an int, so, unlike ``taylor_correction``, it
        # is NOT picked up by ``approx_resolved_flags`` above, which filters on
        # ``isinstance(v, bool)``. Without it, WavePrecomp(n_subbands=3) and
        # (n_subbands=8) collide and the second silently reuses the first's kernel.
        #
        # Keyed off the *resolved* value rather than re-derived from the config
        # object: the resolution rule (photometry knobs come from a WavePrecomp,
        # never from a SpectrumPrecomp) lives in one place, and a signature that
        # re-derives it can silently disagree with the physics it is caching.
        approx_n_subbands = int((self._approx or {}).get("n_subbands", 0))

        # ...and the same hazard generalized. ``band_integration`` is a *string*,
        # so it is invisible to ``approx_resolved_flags`` (bools only) and to
        # ``approx_n_subbands`` (that one key). It currently distinguishes kernels
        # only *incidentally*, because resolving it writes n_subbands and
        # taylor_correction to values that differ per scheme, which is exactly
        # the kind of accident that stops holding the moment someone adds a
        # scheme that leaves those two alone.
        #
        # Capturing every non-bool field generically means the next knob added to
        # ApproxPolicy is covered on the day it is added, rather than after two
        # models silently share a kernel. Cheap: the policy has 8 fields.
        approx_scalar_fields = tuple(
            sorted(
                (k, v)
                for k, v in (self._approx or {}).items()
                if not isinstance(v, bool) and isinstance(v, (str, int, float, type(None)))
            )
        )

        # FeaturePrecomp leaves NO trace in ``self._approx``, it sets
        # ``_fast_line_measurement`` instead, so neither ``approx_resolved_flags``
        # nor ``approx_n_subbands`` above can see it, and two models differing only
        # in FeaturePrecomp produced an IDENTICAL signature. Whichever was built
        # first won the JIT cache and the second silently reused its gradient:
        # measured 12.4 ms vs 0.5 ms for the same objective (~25x), and the loser
        # was whichever came second, not whichever was slower.
        #
        # Worse than the lost speed, it is a correctness hazard: two models with
        # different approximations sharing one compiled gradient means the second
        # computes the FIRST's approximation. Benign only while the two happen to
        # be bit-identical, which is luck, not a contract.
        #
        # Keyed off the same resolved state the PUBLIC ``model.approx`` reports
        # (``ApproxState.feature_precomp``), so what a user is shown and what the
        # cache keys on cannot drift apart, they disagreed here, which is exactly
        # how this survived.
        approx_feature_precomp = bool(getattr(self, "_fast_line_measurement", False))

        def _cfg_key(cfg):
            if cfg is None:
                return None
            return (
                ("n_z", int(cfg.n_z)),
                ("z_min", None if cfg.z_min is None else round(float(cfg.z_min), 12)),
                ("z_max", None if cfg.z_max is None else round(float(cfg.z_max), 12)),
            )

        if self._approx_config is not None or self._approx_config_spec is not None:
            # Key BOTH configs so a composite ``(WavePrecomp, SpectrumPrecomp)``
            # model (#610) gets a distinct slot from either single-LUT model and
            # from a composite with different ztable sampling.
            approx_resolved = (
                approx_resolved_flags,
                ("primary", _cfg_key(self._approx_config)),
                ("spec", _cfg_key(self._approx_config_spec)),
                ("n_subbands", approx_n_subbands),
                approx_scalar_fields,
            )
        else:
            # The scalar fields ride BOTH branches. Carrying them only on the
            # first would make the band-integration scheme invisible to the
            # signature on exactly the models that took the other path.
            approx_resolved = (approx_resolved_flags, approx_scalar_fields)

        # 2026-05-20: drop fixed-parameter VALUES from the
        # cache key. Keep names + types-of-fixed only. Two SEDModels with
        # the same physics + same SSP + same filters + same WavePrecomp
        # config + same FREE-parameter shape and same set of FIXED names
        # now share a compile slot. Their actual fixed VALUES are threaded
        # as a runtime JIT input (see ``_get_or_build_predict_observables_jit``
        # below) so the compiled function uses the correct per-galaxy
        # values at call time. Shape-affecting fixed config
        # (mean_sfh_type, met_mode, dust_model, agn_model, etc.) already
        # has its own dedicated signature entries above and stays distinct.
        spec_fixed_id = tuple(sorted(self.spec.fixed_params))

        # Fast nebular grid (#950): when attached via ``enable_fast_nebular`` the
        # nebular photometry + line channels reconstruct from a per-Q_H grid and
        # the Cue forward is pruned. That is a DIFFERENT compiled graph AND the
        # kernel closes over the grid arrays, so a fast model must not share a
        # slot with the exact model, nor with a fast model over different
        # ionization axes / grid values (would silently reuse a stale kernel,         # the
        # color-leak failure mode this signature exists to prevent).
        _grid = getattr(self, "_nebular_grid_table", None)
        if _grid is not None:
            nebular_grid_sig = (
                "grid",
                tuple(_grid.axis_names),
                int(np.asarray(_grid.log_line_per_qh).tobytes().__hash__()),
            )
        else:
            nebular_grid_sig = ("none",)

        return (
            ssp_flux_shape,
            ssp_lgmet_shape,
            ssp_lgmet_id,
            ssp_flux_id,
            has_alpha_fe,
            n_filters,
            filter_wave_shape,
            filter_trans_dtype,
            filter_trans_id,
            phot_convention,
            dust_model,
            dust_scheme,
            dust_emission_model,
            dust_law_bc_fn_name,
            dust_law_diff_fn_name,
            dust_law_neb_name,
            wg00_selectors,
            dust_law_overrides_sig,
            dust_lyman_cutoff_sig,
            dust_lyc_absorb_all_sig,
            dust_eb_include_lyc_sig,
            astrodust_spinning_dust,
            astrodust_f_cnm,
            nebular_backend_name,
            uses_igm,
            igm_model,
            uses_dla,
            agn_model,
            agn_luminosity_mode,
            agn_blocks,
            agn_norm,
            uses_radio,
            uses_xray,
            xray_model,
            uses_shock,
            shock_cfg,
            mean_sfh_type,
            met_mode,
            stochastic,
            n_grid,
            age_kernel,
            sfh_bin_edges,
            field_centering,
            alpha_fe_evolving,
            z_fixed,
            catalog_z_range,
            has_spectroscopy,
            spec_wave_shape,
            sigma_lib_kms,
            lsf_resolution,
            calibration_order,
            spec_resample_conserving,
            spec_resolution_matrix,
            build_precision,
            met_interp,
            z_interp,
            radio_include_freefree,
            radio_sfr_mode,
            radio_agn_model,
            has_sigma_v,
            compile_mode,
            approx_resolved,
            approx_feature_precomp,
            spec_fixed_id,
            nebular_grid_sig,
        )

    def predict_photometry(self, params, *, ssp_data=None, template_data=None):
        """Compute observed photometric flux densities through all filters.

        Convolves the SED (redshifted and IGM-absorbed) through filter
        transmission curves, returning flux densities in the AB system
        at the source. Routes through :meth:`predict_observables_jit`,
        the JIT-safe orchestrator with SSP threading.

        **Raw forward-pass output.** For interactive use with cached
        derived quantities, see ``model.predict(params).photometry()``.
        For batched photometry over posterior chains, use
        :meth:`predict_photometry_batch`.

        Parameters
        ----------
        params: dict
            Parameter values using public parameter names (e.g.,
            ``sfh_tsnorm_log_total_mass``, ``met_logzsol``, ``redshift``).
            See :class:`Parameters` for canonical names.
        ssp_data: SSPData | None, keyword-only, optional
            SSP grid to thread in as a traced argument. ``None`` (default) uses
            ``self.ssp_data``, which is correct for every ordinary call. Pass it
            explicitly **only when you wrap this method in your own JAX
            transform**, see the JIT note below.
        template_data: Any | None, keyword-only, optional
            Template arrays (nebular grids, dust IR LUTs, AGN libraries) to thread
            in. ``None`` (default) uses :meth:`_template_data_for_jit`. Same
            rationale as ``ssp_data``.

        Returns
        -------
        flux_density: array, shape (n_filters,)
            Observed flux densities in erg/s/cm²/Hz (AB system, rest-frame
            reference frame corrected for luminosity distance and (1+z)
            redshift factor).

        Raises
        ------
        ValueError
            If no filters configured in the model (pass ``filters`` or
            ``observation=`` to constructor).

        Notes
        -----
        **JIT-compatible**: yes. Safe inside :func:`jax.grad` for
        parameter gradients.

        **Threading across a JIT boundary you own (#1753).** This method is
        already self-JIT'd and structurally cached, and it threads the SSP grid
        as an argument, so tengri's own compiled programs never bake it. That
        guarantee does **not** survive being wrapped in a caller's transform::

            predict = jax.jit(model.predict_photometry)  # grid is BAKED

        The inner jit inlines into the outer trace and ``self.ssp_data``, read as
        a concrete array, becomes a ``Constant`` of your computation. On a real
        SSP that is 66.89 MB inlined, and the persistent-cache entry grows from
        0.23 MB to 58.82 MB, a factor of 256, the mechanism behind the 141 GB
        cache in #1507. Pass the grid in to keep it an invar instead::

            predict = jax.jit(lambda ssp, p: model.predict_photometry(p, ssp_data=ssp))
            flux = predict(model.ssp_data, params)

        Only the exact wave-grid path pays: under ``approx=WavePrecomp()`` the
        cube is dead code and XLA eliminates it before codegen. And if you are
        not composing this into a larger jitted program, do not wrap it at all,         the plain
        call is already compiled and cached.

        **Approximation accuracy**: Driven by the build-time ``approx=``
        policy. :class:`WavePrecomp` swaps in the SSP×filter LUT, which is
        ~0.4 % accurate for the *stellar* photometry (Zacharegkas+2025 [1]_)
        **but re-applies dust as a first-order Taylor projection across each
        filter (#617)**. That linear-in-λ model is accurate in the optical/IR,
        where the attenuation curve is smooth across a band, but biases bands
        sampling the **rest-UV** (steep, extrapolated attenuation), by an
        order of magnitude for far-UV bands at moderate/high redshift. Such
        configurations emit a build-time ``UserWarning``; use ``approx=None``
        for unbiased blue-band photometry, or validate against it.
        ``SpectrumPrecomp`` applies attenuation per pixel and is unaffected.
        The orchestrator path is itself bit-exact for the configured policy.

        **Filter wavelengths**: All filters loaded via :func:`load_filter_set`
        or :class:`Photometry` are assumed to be in observed frame (redshifted).
        The model auto-redshifts rest-frame SED by :math:`(1+z)` before
        filter integration.

        See Also
        --------
        predict: Lazy prediction object for all derived quantities.
        predict_spectrum: Spectral flux at arbitrary wavelengths.
        :meth:`Prediction.magnitudes`: AB magnitudes (uses photometry internally).

        Examples
        --------
        >>> flux = model.predict_photometry(params)
        >>> mags = model.predict(params).magnitudes()
        >>> # For the fast LUT path, build with ``approx=WavePrecomp()``.

        References
        ----------
        .. [1] A. Zacharegkas et al., "Fast Photometry with Precomputed
           Stellar Population Grids," ApJ, (2025).
        """
        if self.filter_waves is None:
            raise ValueError("No filters set. Pass filters or observation= to SEDModel().")
        return self.predict_observables_jit(
            params, ssp_data=ssp_data, template_data=template_data
        ).phot_fnu

    # There is deliberately no ``_refuse_on_fast_nebular`` here any more.
    #
    # #950 and #1665 both fixed the same defect by *refusing* every rest-frame-SED
    # consumer whenever a per-Q_H nebular grid was attached, because the fast path
    # zeroed ``sed_nebular`` and those consumers measured a gutted SED (worst: all
    # 13 spectral indices off the exact path, ``HgA`` by +1733%).
    #
    # #1673 removed the cause instead of the symptom: ``predict_state`` now
    # materializes the nebular component by default (``materialized_chain``), so the
    # forward state a rich consumer reads is complete. Measured on a
    # dust-free Cue model at ``approx=(WavePrecomp(), FeaturePrecomp())``, fast
    # versus exact: ``sed_intrinsic`` and ``sed_nebular`` **rel 0.0, bit-exact**,
    # and so are ``pred.rest_sed()``, ``pred.obs_sed()``, ``predict_spectrum`` and
    # ``pred.lines``. Only ``predict_photometry`` still reads the grid (the LUT's
    # own ~9e-04 bias), which is the whole point, the hot path keeps its speed and
    # the rich path is correct.
    #
    # So a refusal here would reject a computation that is now bit-exact, and its
    # advice ("use approx=WavePrecomp() alone") would push users off the config
    # every fit surface resolves to by default since #1683. Two mechanisms for one
    # defect also drift apart; this is the one that returns an answer.
    #
    # If you are about to re-add a refusal: measure the consumer against an
    # ``approx=None`` model first. If it is bit-exact, the cause is already fixed.

    def predict_spectrum(
        self,
        params,
        wave_obs=None,
        wave_chunk_size=None,
        *,
        ssp_data=None,
        template_data=None,
    ):
        """Compute observed spectrum at given wavelengths with LSF convolution.

        Evaluates the full SED at custom wavelengths in observed frame,
        applies velocity dispersion broadening (if ``sigma_v`` in spec),
        convolves with instrument line-spread function, and optionally
        applies multiplicative Chebyshev calibration polynomial.

        **Raw forward-pass output.** For interactive use, see
        ``model.predict(params).spectrum``. For batched spectra, use
        :meth:`predict_spectrum_batch`.

        Parameters
        ----------
        params: dict
            Parameter values using public parameter names.
        wave_obs: array, optional
            Observed-frame wavelength grid [Angstrom]. If None, uses:

            1. The grid bound at construction from
               ``observation.spectroscopy.wave_obs``
            2. Raises ValueError if no grid is available

        wave_chunk_size: int, optional
            If specified, split observed-frame wavelength axis into chunks of
            this size and evaluate via ``jax.lax.map`` to reduce per-chunk HLO
            size for XLA compilation. Default None (no chunking, exact behavior).
            For spectroscopy with R~500 at N≥64 galaxies, typical value is 32–64
            to avoid XLA compilation wall-clock.
        ssp_data, template_data: Any | None, keyword-only, optional
            The JIT-threading channel, see :meth:`predict_photometry` for what it
            is for and what baking costs (#1753). Honored on the configured-
            spectroscopy route (the inference hot path, taken when ``wave_obs`` is
            ``None`` and the model has a spectroscopy channel). An explicit
            ``wave_obs`` grid routes through ``_predict_obs_sed`` instead, which
            does not yet carry the channel, that route still closure-captures.

        Returns
        -------
        flux: array, shape (n_pix,)
            Observed spectral flux density [erg/s/cm²/Hz] in the AB system
            at the specified wavelengths.

        Raises
        ------
        ValueError
            If ``wave_obs`` is None and no precomputed wavelength grid available.

        Notes
        -----
        **JIT-compatible**: yes, routes through
        :meth:`predict_observables_jit` (the JIT-safe orchestrator).

        **Velocity dispersion**: When ``sigma_v`` is in free params,
        applies line-of-sight broadening via Gaussian convolution at
        FWHM = ``2.355 × sigma_v``. Implemented as wavelength-space
        Gaussian convolution (valid for linear pixels; use
        :func:`~tengri.observation.spectrum.apply_lsf` for
        log-wavelength pixels).

        **Line-spread function**: Composition of:

        - Velocity dispersion broadening (σ_v-dependent)
        - Instrument LSF (resolution R-dependent, Gaussian approximation)
        - Chebyshev multiplicative calibration (optional)

        All three are convolved in the forward model.

        **Precomputed wavelength grid**: When the model is built with
        ``Observation(spectroscopy=Spectroscopy(wave_obs=...))`` the SSP is
        resampled onto that fixed grid at construction, so each forward
        spectrum is a cached weighted sum (~1 ms warm) and ``wave_obs`` need
        not be passed on every call.

        **Wavelength-axis chunking**: Set ``wave_chunk_size`` to split the
        observed-frame wavelength axis into ~N/chunk_size chunks and evaluate
        independently via lax.map. Each chunk's HLO is ~1/K of the full HLO
        (K = chunk_size / min_chunk_width), reducing XLA compile-time
        superlinearly. Numerical output is bitwise-identical to unchunked.
        Typical runtime overhead: +5–20% per galaxy due to map overhead.

        Examples
        --------
        >>> wave_obs = np.linspace(4000, 5500, 1000)  # observed frame [Å]
        >>> flux = model.predict_spectrum(params, wave_obs)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(wave_obs, flux)
        >>> plt.xlabel("Wavelength (Å)")
        >>> plt.ylabel("Flux (erg/s/cm²/Hz)")

        For large spectroscopy sets with many galaxies, use chunking::

            >>> flux = model.predict_spectrum(params, wave_obs, wave_chunk_size=64)

        See Also
        --------
        predict_photometry: Filter-integrated flux (simpler, faster).
        predict: Lazy access to all SED and SFH quantities.
        predict_photometry: Filter-integrated flux (simpler, faster).
        """
        # Fast-nebular guard (#950): the fast path is for photometry + line
        # fluxes only. Shared with every other rest-SED consumer (#1665).

        # A caller-supplied ``wave_obs`` is evaluated directly on that grid,
        # independent of any configured spectroscopy channel or the cached
        # ``predict_observables`` (which is photometry-only on a photometry
        # model and previously raised ``'Observables' has no attribute
        # 'spec_fnu'`` after a ``predict_photometry``/fit call). This makes the
        # documented ``wave_obs`` argument do what it says, give me the model
        # spectrum on this grid, on any model and in any call order
        # (suchethac/tengri#707).
        if wave_obs is not None:
            return self._predict_spectrum_on_grid(params, jnp.asarray(wave_obs), wave_chunk_size)

        # No explicit grid: a configured spectroscopy channel routes through the
        # orchestrator, the JIT/grad/LUT-friendly cached path the Fitter relies
        # on (honors the SpectrumPrecomp LUT, LSF and any calibration). This is
        # the inference hot path and must stay on predict_observables.
        if (
            self.observation is not None
            and getattr(self.observation, "spectroscopy", None) is not None
            and getattr(self.observation.spectroscopy, "wave_obs", None) is not None
        ):
            del wave_obs, wave_chunk_size
            return self.predict_observables_jit(
                params, ssp_data=ssp_data, template_data=template_data
            ).spec_fnu

        # No spectroscopy channel but a manually attached grid (``model._wave_obs``),
        # evaluate directly so photometry-only models with an ad-hoc grid work
        # regardless of the predict_observables cache state (#707).
        manual_grid = getattr(self, "_wave_obs", None)
        if manual_grid is not None:
            return self._predict_spectrum_on_grid(
                params, jnp.asarray(manual_grid), wave_chunk_size
            )
        raise ValueError(
            "No wavelength grid. Pass wave_obs, "
            "or attach an Observation with spectroscopy.wave_obs set."
        )

    def _predict_spectrum_on_grid(self, params, wave_obs, wave_chunk_size=None):
        """Evaluate the observed-frame model spectrum on an arbitrary grid.

        Self-contained projector that does **not** depend on the model having a
        spectroscopy channel or on the ``predict_observables`` cache: it builds
        the observed-frame SED (rest SED + IGM/DLA/MW via :meth:`predict_obs_sed`)
        and resamples it onto ``wave_obs`` with the same kernel
        (:func:`~tengri.observation.spectrum.project_spectrum`) the configured
        spectroscopy path uses. The instrument LSF is applied only when the
        attached observation declares a spectroscopic resolution. Underpins the
        ``wave_obs`` argument of :meth:`predict_spectrum` (suchethac/tengri#707).

        Parameters
        ----------
        params: dict
            Public-name parameter values.
        wave_obs: array_like, shape (n_pix,)
            Observed-frame wavelength grid [Angstrom].
        wave_chunk_size: int, optional
            Currently advisory on this direct path, the per-pixel projection is
            cheap and LSF convolution couples pixels, so the grid is evaluated in
            one pass. Chunking remains active on the configured-grid orchestrator
            path.

        Returns
        -------
        ndarray, shape (n_pix,)
            Observed spectral flux density [erg/s/cm^2/Hz].
        """
        del wave_chunk_size  # see Parameters note
        from tengri.cosmology import luminosity_distance
        from tengri.observation.spectrum import project_spectrum

        sed_obs = self._predict_obs_sed(params)
        z = self._get_redshift(params)
        dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
        wave_rest = sed_obs.wavelength / (1.0 + z)

        spectroscopy = (
            getattr(self.observation, "spectroscopy", None) if self.observation else None
        )
        resolution = (
            getattr(spectroscopy, "resolution", None) if spectroscopy is not None else None
        )
        sigma_lib_kms = (
            getattr(spectroscopy, "sigma_lib_kms", 0.0) if spectroscopy is not None else 0.0
        )
        cal_coeffs = spectroscopy.calibration_coeffs(params) if spectroscopy is not None else None
        cal_wave_range = spectroscopy.calibration_wave_range if spectroscopy is not None else None
        # Static (pre-trace) resolution of the resample mode (#1166): the model
        # grid is fixed, so this is a Python bool baked into the trace, not a
        # branch on the sampled redshift.
        conserving = (
            spectroscopy.resolve_conserving(self.wavelengths)
            if spectroscopy is not None
            else False
        )
        resolution_matrix = (
            getattr(spectroscopy, "resolution_matrix", None) if spectroscopy is not None else None
        )

        flux = project_spectrum(
            sed_obs.sed,
            wave_rest,
            wave_obs,
            z,
            dl_cm,
            resolution=resolution,
            sigma_lib_kms=sigma_lib_kms,
            sigma_v_kms=params.get("sigma_v_kms", 0.0),
            cal_coeffs=cal_coeffs,
            cal_wave_range=cal_wave_range,
            conserving=conserving,
            resolution_matrix=resolution_matrix,
        )
        return flux

    def predict_magnitudes(self, params):
        """Deprecated. Use ``model.predict(params).magnitudes()``.

        .. deprecated:: 2026-07
           Superseded by the property catalog and the ``Prediction`` surface
           (#1043 contract §2). The body is unchanged, this shim is bit-exact
           with the method it replaces, so migrating changes no number.
           Will be removed in tengri v1.0.

        Returns
        -------
        Same as :meth:`_predict_magnitudes`.
        """
        warnings.warn(
            "predict_magnitudes() is deprecated, use model.predict(params).magnitudes() "
            "instead (cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._predict_magnitudes(params)

    def _predict_magnitudes(self, params):
        """Compute observed AB magnitudes through all filters.

        **Raw forward-pass output.** For interactive use, see
        ``model.predict(params).magnitudes``.

        Parameters
        ----------
        params: dict
            Parameter values using public parameter names.

        Returns
        -------
        magnitudes: ndarray, shape (n_filters,)
            Observed AB magnitudes [mag].

        Notes
        -----
        **JIT-compatible**: yes (routes through :meth:`predict_photometry`).

        Derived from :meth:`predict_photometry` via the AB definition
        :math:`m_\\mathrm{AB} = -2.5 \\log_{10}(F_\\nu) - 48.6`. Issue
        #436: routing instead through :func:`dsps.calc_obs_mag` used a
        different filter-convolution convention (Bessell & Murphy 2012
        photon-counting form :math:`\\int T F_\\nu \\, d\\lambda/\\lambda`)
        than ``predict_photometry`` (Tokunaga & Vacca 2005
        :math:`\\int \\lambda T F_\\nu \\, d\\lambda`), giving 5–40 mmag
        zero-point offsets in SDSS bands. Both APIs must use the same
        convention; deriving the magnitude from the flux is the only
        choice that is correct by construction.
        """
        if self.filter_waves is None:
            raise ValueError("No filters set.")
        flux = self.predict_photometry(params)
        return ab_mag_from_flux(flux)

    def predict_luminosity(self, params):
        """Compute rest-frame luminosity SED in solar units.

        **Raw forward-pass output.** For interactive use with derived
        scalars (L_bol, L_uv, L_ir), see ``model.predict(params).sed.*``.

        Parameters
        ----------
        params: dict
            Parameter values using public parameter names.

        Returns
        -------
        array, shape (n_wave,)
            Rest-frame luminosity [L_sun/Hz].

        Notes
        -----
        **JIT-compatible**: no, wraps :meth:`predict_rest_sed`.

        Divides rest-frame SED by :math:`L_{\\odot} = 3.828 \\times 10^{33}` erg/s
        (IAU 2015 definition).

        .. deprecated:: 2026-07 (cleanup PR-2)
            Interactive getter moved to the lazy Prediction wrapper:
            ``model.predict(params).sed.l_bol`` (one cached forward pass shared across
            all derived quantities). Removed in tengri v1.0.
        """
        warnings.warn(
            "predict_luminosity() is deprecated, use "
            "model.predict(params).sed.l_bol instead "
            "(cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from tengri.utils.physics_constants import L_SUN

        sed_erg = self._predict_rest_sed(params).sed
        return sed_erg / L_SUN

    def _has_line_catalog(self):
        """Whether the nebular backend publishes a discrete line catalog.

        ``True`` for photoionization backends (Cue / CloudyGrid) →
        :meth:`predict_line_fluxes` works. ``False`` for BakedIn / shock → line
        fluxes must be *measured* from the spectrum (:meth:`measure_line_fluxes`).
        """
        backend = getattr(self, "_nebular_backend", None)
        return backend is not None and hasattr(backend, "predict_nebular_line_luminosities")

    def _attenuate_line_catalog(self, params, line_waves, line_lums):
        """Dust-redden a discrete nebular line catalog at its wavelengths.

        THE single source of nebular-line reddening (Charlot & Fall 2000 birth-
        cloud + diffuse), used by :meth:`predict_line_fluxes` AND the interactive
        ``model.predict(params).lines`` catalog, so no public line path is
        silently intrinsic while carrying an "observed" contract. Backends publish
        INTRINSIC ``line_lums``; this applies the same operator the continuum
        nebular SED gets. JIT-safe (pure ``jnp`` via ``attenuate_emission``).
        """
        from tengri.forward.emission_helpers import attenuate_emission

        _is_single = self._dust_model == "single_component"
        return attenuate_emission(
            line_lums,
            line_waves,
            self._neb_dust_mode,
            jnp.asarray(params.get("dust_tau_bc", params.get("dust_tau_v", 0.0))),
            jnp.asarray(params.get("dust_tau_diff", 0.0)),
            self._dust_law_bc_fn,
            self._dust_law_bc_fn if _is_single else self._dust_law_diff_fn,
            neb_bc_fn=self._neb_dust_law_bc_fn,
            dust_slope=jnp.asarray(params.get("dust_slope", -0.7)),
            dust_bump_strength=jnp.asarray(params.get("dust_bump_strength", 0.0)),
        )

    def predict_line_fluxes(
        self, params, target_wavelengths=None, tolerance_aa=5.0, *, redden=True, state=None
    ):
        """Predict observed emission line fluxes (dust-reddened by default).

        The **model's** nebular line emission: calls the photoionization backend
        (Cue / CloudyGrid) for line luminosities, applies dust attenuation at each
        line wavelength (Charlot & Fall birth-cloud + diffuse), and converts to
        observed flux ``L / (4 pi d_L^2)`` [erg/s/cm^2].

        See also :meth:`measure_line_fluxes`, which instead *measures* line fluxes
        off the model spectrum the way a spectroscopic pipeline measures data
        (continuum-subtract + integrate), works for any backend (including
        baked-in wNE) and carries the stellar Balmer absorption self-consistently.
        The distinction: ``predict_line_fluxes`` = what the galaxy emits;
        ``measure_line_fluxes`` = what a pipeline would extract from its spectrum.

        For interactive access to individual named lines (luminosities, ratios,
        BPT diagnostics), see ``model.predict(params).lines.halpha`` etc.

        Parameters
        ----------
        params: dict
            Parameter values (public names).
        target_wavelengths: array, shape (n_target,), optional
            Rest-frame vacuum wavelengths (Angstrom) of lines to predict.
            Each wavelength is matched to the nearest backend line.
            If None, returns all lines from the nebular backend.
        tolerance_aa: float or None, default 5.0
            Maximum allowed wavelength delta [Angstrom] between a requested
            target and the matched catalog line. Raises ``ValueError`` on
            any miss, listing the offending targets. Pass ``None`` to disable
            (recovers legacy nearest-line-no-matter-what behavior).
        redden: bool, default True
            Apply dust attenuation to the lines (HII regions see birth-cloud +
            diffuse dust; Charlot & Fall 2000). ``True`` returns **observed**
            fluxes comparable to a raw catalog; set ``False`` for **intrinsic**
            (un-reddened) fluxes, e.g. when fitting extinction-corrected
            catalog line fluxes. (Before 2026-07 this was always intrinsic,             silently
            omitting the line reddening; ``redden=True`` is the fix.)

        Returns
        -------
        fluxes: array, shape (n_target,) or (n_all_lines,)
            Observed line fluxes in erg/s/cm^2.

        Raises
        ------
        ValueError
            If no nebular backend is configured.

        Notes
        -----
        **JIT-compatible**: no, delegates to nebular backend.

        **Shock component limitation** (#927): When a model includes both
        an active shock component and a photoionized backend (Cue/CloudyGrid),
        the shock's discrete line luminosities are **not** published to the
        returned catalog. MAPPINGS V bakes shock lines into the continuum SED
        only (``sed_shock``), which is invisible to this catalog-reading surface.
        The shock will constrain only through continuum (broadband/spectrum), with
        zero gradient from line-flux channels. Use :meth:`measure_line_fluxes`
        (pipeline-style flux extraction from the spectrum) to measure shock lines
        self-consistently with the stellar continuum.

        Observed flux is calculated from luminosity via:

        .. math::

            F = \\frac{L}{4\\pi d_L^2}

        where :math:`L` is the line luminosity [erg/s] and :math:`d_L` is
        the luminosity distance [cm].
        """
        backend = self._nebular_backend
        if backend is None or not hasattr(backend, "predict_nebular_line_luminosities"):
            raise ValueError(
                "No nebular backend with line prediction configured. Cannot compute line fluxes."
            )

        # Warn if shock component is active with photoionized backend (#927): shock
        # lines are baked into the continuum SED, not published to the discrete
        # catalog that predict_line_fluxes reads, so shock parameters get zero
        # gradient from line-flux fitting.
        if self._uses_shock:
            backend_name = type(backend).__name__.lower()
            is_photoionized = any(name in backend_name for name in ("cue", "cloudygrid"))
            if is_photoionized:
                from tengri.config.exceptions import ShockPhotoionizedMixedWarning

                warnings.warn(
                    "Shock component present with photoionized nebular backend "
                    "(Cue/CloudyGrid). Shock's discrete line emission is **not** "
                    "included in predict_line_fluxes output; shock lines are baked "
                    "into the continuum SED (sed_shock) only. Shock parameters "
                    "(shock_frac, shock_log_lhalpha, etc.) will have zero gradient "
                    "from line-flux fitting. "
                    "Remedy: use measure_line_fluxes() (pipeline-style extraction), "
                    "fit shock through continuum only, or use BakedIn backend.",
                    ShockPhotoionizedMixedWarning,
                    stacklevel=2,
                )

        # Read the discrete line catalog published by
        # NebularSEDComponent. The orchestrator's nebular adapter calls
        # ``predict_nebular_line_luminosities`` with SSP-derived
        # ``ssp_weights`` + ``ssp_log_ages_yr`` and the canonical
        # neb_logZ_gas → absolute-log10(Z) translation.
        #
        grid = getattr(self, "_nebular_grid_table", None)
        if grid is not None:
            # FAST path (#950): reconstruct intrinsic line luminosities from the
            # per-Q_H grid, no Cue forward. Q_H is the stellar-published ``nion``
            # (from the passed state when available, else the SED-free
            # ``compute_nion``); the grid supplies ``L_line / Q_H``. The shared
            # redden + target-match + cosmology tail below is unchanged.
            from tengri.components.nebular.nebular_grid_precompute import (
                reconstruct_nebular_line_lums,
            )

            if state is not None and "nion" in state.derived:
                nion = state.derived["nion"]
            else:
                nion = self._compute_nion(params)
            nion = jnp.sum(nion) if jnp.ndim(nion) else nion
            all_waves = jnp.asarray(grid.wavelengths)
            all_lums = reconstruct_nebular_line_lums(nion, params, grid)
        else:
            # ``state`` may be supplied by a caller that has already run the
            # forward (e.g. the joint loss deriving line fluxes + ratios +
            # indices from ONE ``predict_state``, see
            # ``loss_functions._build_prediction``) so the full-grid forward
            # is not recomputed once per feature channel.
            if state is None:
                state = self.predict_state(params)
            if "line_waves" not in state.derived or "line_lums" not in state.derived:
                raise ValueError(
                    "Configured nebular backend did not publish a discrete "
                    "line catalog to state.derived (expected keys "
                    "'line_waves' and 'line_lums'). The BakedIn backend bakes "
                    "lines into the SSP grid; ShockBackend publishes a "
                    "continuous line SED instead. Switch to Cue or CloudyGrid."
                )
            all_waves = jnp.asarray(state.derived["line_waves"])
            all_lums = jnp.asarray(state.derived["line_lums"])

        # Dust-redden the lines at their wavelengths. Reads the catalog the dust
        # component published (#1867) rather than computing its own, so this
        # surface and the `.lines` / `predict_properties` / `predict_line_ratios`
        # surfaces are on ONE screen. "Single-sourced" is what the previous
        # comment here claimed; it was not, and the two differed.
        #
        # `_attenuate_line_catalog` routes through
        # `emission_helpers.attenuate_emission`, whose signature names only
        # `dust_slope` and `dust_bump_strength`, it cannot thread `dust_delta`
        # or `dust_Rv` at all, and forces the bump to the spec's Fixed(0.0)
        # over any law's own default (#1858). Measured on the Balmer decrement,
        # property surface against this one: `calzetti` (which reads no shape
        # parameter) agreed to 4e-15, while `narayanan_z` (bump 1.0, delta -0.2)
        # disagreed by 1.1e-3 rising to 2.5e-3. The law that cannot see the
        # defect agreeing to machine precision is what identifies the shape
        # parameters as the whole of it.
        #
        # The fallback keeps `redden=True` meaningful for a chain that publishes
        # no attenuated catalog, no dust component, or a backend with no
        # discrete lines. The FAST line-LUT path (#1477) runs stateless by design
        # and takes the fallback screen.
        if redden:
            _log_atten = (
                state.derived.get("log_line_lums_attenuated") if state is not None else None
            )
            if _log_atten is None:
                all_lums = self._attenuate_line_catalog(params, all_waves, all_lums)
            else:
                from tengri.utils.scale import pow10

                # The published catalog is indexed on ``state.derived['line_waves']``,
                # the backend's FULL line list, while the fast branch above set
                # ``all_waves`` to ``grid.wavelengths``, which holds only the lines
                # the observation asked for. Taking the luminosities without the
                # wavelengths that index them pairs two different catalogs, and the
                # target match below then reads the first ``n_target`` entries of the
                # full list: for Cue those are the far-UV 923-937 A lines, returned
                # under the labels Halpha / Hbeta / [OIII] and low by ~2.3e4 (#1943).
                #
                # ``tolerance_aa`` cannot catch that, the WAVELENGTHS match the
                # targets exactly; only the luminosities come from the wrong array.
                # Which is why it went unnoticed on the default ``approx='auto'``
                # path for every dusty fit with a discrete-catalog backend.
                #
                # Only a dusty chain reaches here with a grid: dust sets
                # ``must_materialize_sed``, which disarms ``use_grid`` and so leaves
                # the nebular component publishing the attenuated catalog (#1281).
                # A dust-free model publishes none, takes the fallback screen above,
                # and was never affected.
                all_lums = pow10(jnp.asarray(_log_atten))
                all_waves = jnp.asarray(state.derived["line_waves"])

        if target_wavelengths is not None:
            target_wavelengths = jnp.asarray(target_wavelengths)
            deltas = jnp.abs(all_waves[None, :] - target_wavelengths[:, None])
            indices = jnp.argmin(deltas, axis=1)
            min_deltas = deltas[jnp.arange(target_wavelengths.shape[0]), indices]
            # Tolerance check: if a target has no nearby line in the catalog,
            # argmin silently returns whatever is closest. Catch that here so
            # callers don't accidentally read wrong-line fluxes (e.g. asking
            # for vacuum 5008.24 when the catalog is in air at 5006.84 is
            # within 1.4 Aa and OK; asking for a missing 6300 [OI] line could
            # match Halpha 264 Aa away). ``tolerance_aa=None`` disables.
            # The guard needs concrete values; under a jitted loss
            # (NUTS/HMC) ``min_deltas`` is a Tracer, so skip it there,             # line matching
            # is structural (static catalog × static
            # targets), and any eager call on the same model (mock
            # generation, prediction, the first Fitter setup) runs the
            # loud check for the identical matching.
            if tolerance_aa is not None and not isinstance(min_deltas, jax.core.Tracer):
                import numpy as _np

                bad = _np.asarray(min_deltas) > float(tolerance_aa)
                if bad.any():
                    tw = _np.asarray(target_wavelengths)
                    mw = _np.asarray(all_waves[indices])
                    md = _np.asarray(min_deltas)
                    misses = "\n".join(
                        f"  target={tw[i]:.3f} Aa  closest={mw[i]:.3f} Aa  delta={md[i]:.3f} Aa"
                        for i in _np.where(bad)[0]
                    )
                    raise ValueError(
                        f"predict_line_fluxes: {int(bad.sum())} target line(s) "
                        f"have no match within tolerance_aa={tolerance_aa} Aa.\n"
                        f"{misses}\n"
                        f"Pass tolerance_aa=None to disable, or pick a backend "
                        f"that publishes the missing line(s)."
                    )
            selected_lums = all_lums[indices]
        else:
            selected_lums = all_lums

        dl_cm = self._get_dl_cm(params)
        # ``line_lums`` are published in erg/s (DerivedKey contract in
        # NebularSEDComponent), no L_sun conversion here. Multiplying by
        # L_SUN was a 33.6-dex unit error that made every joint
        # photometry+line-flux fit unusable against real data.
        log10_scale = -log10_four_pi_dl2(dl_cm)
        flux = apply_log10_scale(selected_lums, log10_scale)
        return flux

    def enable_fast_nebular(self, target_wavelengths, *, n_grid=16, ranges=None):
        r"""Attach a per-Q_H nebular grid so lines + photometry skip the Cue forward.

        Builds the adaptive-axis nebular grid
        (:func:`~tengri.components.nebular.nebular_grid_precompute.precompute_nebular_grid`)
        from **this** model, running the Cue forward once per grid point at build
        time, then attaches it to the nebular component. Afterwards:

        * :meth:`predict_photometry` reconstructs the nebular broadband
          contribution as :math:`Q_H \times \mathrm{interp}(\text{grid})`, and the
          Cue forward becomes dead for the photometry channel (XLA prunes it);
        * :meth:`predict_line_fluxes` reconstructs the emission lines the same way.

        Both are **SED-free in** :math:`Q_H` (the stellar-published ``nion``). The
        grid axes are whichever of ``met_logzsol`` / ``neb_logU`` /
        ``neb_logZ_gas`` are FREE; fixed ionization params are baked.

        Parameters
        ----------
        target_wavelengths: array_like, shape (n_lines,)
            Rest-frame vacuum line wavelengths [Angstrom] the grid tabulates and
            :meth:`predict_line_fluxes` serves.
        n_grid: int or dict, default 16
            Grid points per free ionization axis. Denser → tighter interpolation.
            A dict ``{axis_name: n}`` resolves ``met_logzsol`` / ``neb_logU`` /
            ``neb_logZ_gas`` independently; omitted axes take 16 and an
            unrecognized key raises (#1311). Build cost is the product over free
            axes.
        ranges: dict, optional
            Override ``{param: (lo, hi)}`` grid bounds (defaults to each free
            param's prior support).

        Returns
        -------
        SEDModel
            ``self`` (configured in place), for chaining.

        Raises
        ------
        ValueError
            If no Q_H-linear nebular backend (Cue) is configured.

        Notes
        -----
        **Approximation.** Reconstruction is exact at grid nodes and node-exact
        PCHIP between them. Photometry error stays ~1 % on the total (nebular is
        subdominant in broadband). Line-flux accuracy depends strongly on which
        ionization params are free (#950 convergence study):

        * **Gas axes** (``logU`` + ``neb_logZ_gas``) drive smooth per-Q_H line
          changes and converge with ``n_grid``.
        * **Free ``met_logzsol``** used to be the pathological case, and is no
          longer. The old text here (retained in spirit because the reasoning is
          still worth knowing) said the exact Cue forward was *discontinuous* in
          metallicity, it took the ionizing-spectrum shape from a single
          ``argmax``-chosen age bin, so [OIII] stepped ~33 % whenever the dominant
          bin flipped, and no interpolant crosses a jump: a dense sweep gave [OIII]
          worst-case ~10-23 % at any ``n_grid``. **#1019 removed that argmax.** The
          shape is now a luminosity-additive mix over every ionizing age bin
          (``cue.py``), so the forward is smooth in metallicity and the grid
          converges. Measured on the shipped bare SSP: 0.42 % between nodes, 0.21 %
          at a node, identical on linux/x86 and macOS/arm64.

        .. warning::

           That stale ~10-23 % figure is a **trap**. It happens to bracket both the
           20.9 % photometry and 14.7 % line-flux drifts reported in #1154, so it
           offers a ready-made, and wrong, explanation for them. It cost a full
           debugging session. When a number in a docstring matches your bug
           suspiciously well, check that the mechanism behind it still exists before
           you believe it: this one was removed by #1019.

        **Caveat that #1019 introduced.** Making the shape a luminosity-weighted mix
        also made it depend on the **SFH**, the old ``argmax`` forced
        ``d(shape)/d(SFH) = 0``, and removing it was the point. But the SFH is *not*
        a grid axis here (only ``met_logzsol`` / ``logU`` / ``neb_logZ_gas`` are), so
        ``L = Q_H * l(...)`` assumes a shape-independence the exact forward no longer
        has. Q_H carries the *number* of ionizing photons, not their *hardness*. In
        practice the residual is small (the 0.42 % above is measured across free-SFH
        draws), but it is an approximation, not an identity.

        Validate accuracy with a **dense sweep strictly inside the grid range**,         random
        parameter draws under-sample structure and report optimistic bounds.

        For the photometry channel the model must be built with
        ``approx=WavePrecomp()`` (so the grid can capture the intrinsic
        filter-integrated nebular ``L_nu``); without it only the line channel is
        reconstructed and photometry stays on the exact path.

        **JIT-compatible**: the resulting :meth:`predict_photometry` /
        :meth:`predict_line_fluxes` are JIT- and gradient-safe; the one-time grid
        build is eager.
        """
        import dataclasses

        from tengri.components.nebular.component import NebularSEDComponent
        from tengri.components.nebular.nebular_grid_precompute import precompute_nebular_grid

        if self._nebular_backend is None or not hasattr(
            self._nebular_backend, "predict_nebular_line_luminosities"
        ):
            raise ValueError(
                "enable_fast_nebular requires a Q_H-linear nebular backend with "
                "line prediction (Cue). The configured backend is "
                f"{type(self._nebular_backend).__name__ if self._nebular_backend else 'none'}."
            )

        target_wavelengths = jnp.asarray(target_wavelengths)
        table = precompute_nebular_grid(self, target_wavelengths, n_grid=n_grid, ranges=ranges)
        self._nebular_grid_table = table
        # Rebuild the chain from scratch (exact, no grid) and swap in the
        # grid-carrying nebular component so ``apply`` takes the fast branch.
        # compile_signature() now differs (nebular_grid_sig), so the next
        # predict_* builds a fresh kernel over this chain, no stale reuse.
        chain = self._build_component_chain()
        # Whether the grid may also serve the photometry channel. It may only
        # when nothing downstream reads the continuum, because serving
        # photometry from the grid requires zeroing ``sed_nebular``, and the
        # dust energy balance reads it to size the absorbed budget. Asked of
        # the chain rather than assumed, so registering a new consumer is a
        # one-line ``inputs()`` declaration and nothing here goes stale.
        #
        # The census sees the component contract, and only that. A reader that
        # takes ``sed_nebular`` off ``state.derived`` without declaring an
        # input is invisible to it, ``state_to_sed_components`` does exactly
        # that, so ``sed_components()`` on a dust-free Cue model still reports
        # a zero nebular continuum (#1673).
        sed_consumers = _nebular_continuum_consumers(chain)
        self._cached_component_chain = [
            dataclasses.replace(c, grid_table=table, must_materialize_sed=bool(sed_consumers))
            if isinstance(c, NebularSEDComponent)
            else c
            for c in chain
        ]
        return self

    def _compute_nion(self, params):
        """SED-free ionizing photon rate :math:`Q_H` from the stellar component.

        Slices ``params`` to the stellar prefixes and delegates to
        :meth:`StellarSEDComponent.compute_nion`, the ionizing-slice integral
        that skips the full-wavelength SED. Used by the fast nebular line path so
        it never runs :meth:`predict_state`.
        """
        from tengri.components.stellar.component import StellarSEDComponent
        from tengri.forward.orchestrator import slice_params_for_component

        chain = getattr(self, "_cached_component_chain", None)
        if chain is None:
            chain = self._cached_component_chain = self._build_component_chain()
        stellar = next((c for c in chain if isinstance(c, StellarSEDComponent)), None)
        if stellar is None:
            raise ValueError("No StellarSEDComponent in the chain, cannot compute Q_H.")
        sliced = slice_params_for_component(stellar, params)
        return stellar.compute_nion(sliced, ssp_data=self.ssp_data)

    def predict_line_ratios(self, params, line_ratio_data, *, state=None):
        """Predict emission line ratios for a :class:`LineRatioData` set.

        Computes the model flux ratio ``F(numerator) / F(denominator)`` for
        each requested pair, in the same space (linear or log10) as the data.
        Runs the forward chain **once** and selects both the numerator and
        denominator lines from the published catalog, so the likelihood
        loop pays a single chain evaluation per step, not two.

        Works identically on the exact and SpectrumPrecomp paths: line
        luminosities are grid-independent and survive the LUT path.

        Parameters
        ----------
        params: dict
            Parameter values (public names).
        line_ratio_data: LineRatioData
            The observed ratio set; supplies ``numerator_waves`` /
            ``denominator_waves`` for matching and the ``log_space`` flag.

        Returns
        -------
        ndarray, shape (n_ratios,)
            Model ratios (``log10`` when ``line_ratio_data.log_space``).

        Raises
        ------
        ValueError
            If no nebular backend publishes a discrete line catalog.

        Notes
        -----
        **JIT-compatible**: no, delegates to the nebular backend via
        :meth:`predict_state`.
        """
        # Diagnose the fast-nebular case FIRST (#1665). The grid path skips the
        # discrete line-catalog publish, so a Cue model would otherwise fall
        # through to the backend message below and be told to "use Cue" -- advice
        # the user has already taken, naming a cause that is not theirs.

        if state is None:
            state = self.predict_state(params)
        if "line_waves" not in state.derived or "line_lums" not in state.derived:
            raise ValueError(
                "Configured nebular backend did not publish a discrete line "
                "catalog ('line_waves'/'line_lums'). Use Cue or CloudyGrid; "
                "BakedIn bakes lines into the SSP and cannot report ratios."
            )
        all_waves = jnp.asarray(state.derived["line_waves"])
        # Dust-reddened catalog when a dust component published one (#1867).
        # This surface used to read the INTRINSIC catalog while the data it is
        # fitted against, observed Balmer decrements, BPT positions, are
        # reddened, so a line-ratio fit compared two different things. The
        # comment here said "same fix as predict_line_fluxes"; that method's
        # unit hygiene was copied across and its reddening was not.
        #
        # Reading the published key rather than calling
        # ``_attenuate_line_catalog`` keeps this on the same screen as the
        # ``balmer_decrement`` / ``bpt_nii`` properties, which is the agreement
        # #1867 exists to establish.
        _log_atten = state.derived.get("log_line_lums_attenuated")
        if _log_atten is None:
            all_lums = jnp.asarray(state.derived["line_lums"])
        else:
            from tengri.utils.scale import pow10

            all_lums = pow10(jnp.asarray(_log_atten))
        dl_cm = self._get_dl_cm(params)
        # ``line_lums`` are erg/s (DerivedKey contract), same fix as
        # ``predict_line_fluxes``. The scale cancels in every ratio, so
        # this is unit hygiene, not a behavior change.
        log10_scale = -log10_four_pi_dl2(dl_cm)

        def _match(targets):
            targets = jnp.asarray(targets)
            deltas = jnp.abs(all_waves[None, :] - targets[:, None])
            idx = jnp.argmin(deltas, axis=1)
            return apply_log10_scale(all_lums[idx], log10_scale)

        num_flux = _match(line_ratio_data.numerator_waves)
        den_flux = _match(line_ratio_data.denominator_waves)
        return line_ratio_data.model_ratio(num_flux, den_flux)

    def predict_spectral_indices(
        self, params, index_defs, *, state=None, approx=False, fast=UNSET
    ):
        """Predict spectral index values from the model SED.

        Generates a rest-frame spectrum covering the index wavelength
        ranges and measures each index (EW or break ratio). Suitable
        for JIT/batch loops; for interactive use, access individual
        indices via ``model.predict(params).sed.dn4000`` etc.

        Parameters
        ----------
        params: dict
            Parameter values (public names).
        index_defs: tuple of SpectralIndexDef
            Index definitions to measure.
        state: ForwardState, optional
            A pre-computed forward state to measure on (shares one
            ``predict_state`` across channels). Ignored when ``approx=True``.
        approx: bool, default False
            Route through the FeaturePrecomp window-LUT path
            (:meth:`_feature_fast_indices`): contract precomputed SSP window
            integrals with SED-free SFH weights and the model's per-age dust
            screen, instead of reconstructing the full-grid SED. ~17x faster
            per evaluation (measured, wNE grid) and bit-exact for the supported
            configuration, **stellar + two-component (or no) dust + baked-in
            (or no) nebular, delta metallicity, parametric non-field SFH**. Any
            other configuration (additive nebular, AGN, non-delta metallicity,
            GP-field SFH, alpha-Fe grid) **raises** ``ValueError`` rather than
            silently falling back, because ``approx=True`` is an explicit opt-in;
            use ``approx=False`` there. Slope indices are filled from the exact
            SED (they are not window-LUT-expressible).

            Named for the build-time ``approx=FeaturePrecomp(...)`` it selects.
            Spelled ``fast`` until 2026-08.
        fast: bool, optional
            Deprecated spelling of `approx`. Removed in v1.0.

        Returns
        -------
        jnp.ndarray, shape (n_indices,)
            Predicted index values.

        Notes
        -----
        **JIT-compatible**: yes, both paths are pure ``jnp``. The ``approx`` path
        builds its window LUT once (cached on the model) from concrete SSP data,
        so it is safe to call under ``jax.jit``.

        Measures spectral indices (equivalent width or break ratio) from a
        rest-frame spectrum covering all wavelength ranges in ``index_defs``.
        """
        from tengri.forward.result import SEDResult
        from tengri.observation.spectral_indices import measure_index_jax

        approx = resolve_renamed_flag(
            approx,
            fast,
            old_name="fast",
            new_name="approx",
            caller="SEDModel.predict_spectral_indices",
        )

        # Indices are measured off the rest-frame SED, which the fast-nebular
        # grid path gutted (#1665). Same guard as predict_spectrum, this
        # consumer was simply missing from that census.

        if approx:
            return self._feature_fast_indices(params, tuple(index_defs))

        # Spectral indices (D4000 / Balmer break / Lick EW) are rest-frame
        # quantities measured on the attenuated galaxy SED. Evaluate the
        # rest-frame SED on the model's native (SSP-resolution) grid and
        # measure each index there, the same source as ``pred.sed.dn4000``.
        #
        # This works on both the exact and SpectrumPrecomp paths: the dust
        # components set ``state.sed_intrinsic`` to the full attenuated SED in
        # all cases (the spectrum LUT only *additionally* publishes per-pixel
        # transmission), so the rest-frame SED is the attenuated SED
        # regardless of ``approx``.
        #
        # ``state`` may be supplied by a caller that already ran the forward
        # (the joint loss shares ONE ``predict_state`` across the line-flux,
        # line-ratio, and index channels), ``predict_rest_sed`` reads exactly
        # ``(state.wave, state.sed_intrinsic)`` on the native grid, so deriving
        # ``rest`` from a shared state is bit-identical to recomputing it.
        if state is None:
            rest = self._predict_rest_sed(params)
        else:
            rest = SEDResult(wavelength=state.wave, sed=state.sed_intrinsic)
        wave_rest, flux_rest = rest.wavelength, rest.sed

        indices = [measure_index_jax(wave_rest, flux_rest, idx_def) for idx_def in index_defs]
        return jnp.array(indices)

    # ── FeaturePrecomp fast path for spectral indices (#950) ──────────

    def _feature_chain(self):
        """The component chain, reusing the construction-time cache when present."""
        chain = getattr(self, "_cached_component_chain", None)
        if chain is None:
            chain = self._build_component_chain()
            self._cached_component_chain = chain
        return chain

    def _index_window_precomp(self, index_defs):
        """Build (and memoize) the SSP window-integral LUT for ``index_defs``.

               Depends only on the model's (concrete) SSP grid and the index windows,
               so it is built once per distinct index set and reused across evaluations,
        the FeaturePrecomp analog of the WavePrecomp SSP x filter LUT. Built
               from concrete SSP data (not traced params), and forced to eager evaluation
               so the cached LUT is a true compile-time constant (see below).
        """
        from tengri.observation.spectral_indices import precompute_index_windows

        cache = getattr(self, "_index_window_lut_cache", None)
        if cache is None:
            cache = {}
            self._index_window_lut_cache = cache
        key = tuple(index_defs)
        pc = cache.get(key)
        if pc is None:
            # ``jax.ensure_compile_time_eval`` forces concrete evaluation even when
            # first reached inside a jit trace; otherwise the cached jnp LUT holds
            # trace-tied tracers that leak into a later trace (UnexpectedTracerError
            # under a joint-then-standalone call order). Same fix as
            # :meth:`_line_window_precomp`.
            with jax.ensure_compile_time_eval():
                pc = precompute_index_windows(
                    self.ssp_data.ssp_wave, self.ssp_data.ssp_flux, index_defs
                )
            cache[key] = pc
        return pc

    def _line_window_precomp(self, line_defs):
        """Build (and memoize) the SSP line-window LUT for ``line_defs``.

        The line-flux analog of :meth:`_index_window_precomp`, same concrete-
        SSP, cached-per-line-set contract (JIT-safe).
        """
        from tengri.observation.line_measurement import precompute_line_windows

        cache = getattr(self, "_line_window_lut_cache", None)
        if cache is None:
            cache = {}
            self._line_window_lut_cache = cache
        key = tuple(line_defs)
        pc = cache.get(key)
        if pc is None:
            # Force eager (concrete) evaluation even when first reached inside a
            # jit trace. Without this, ``precompute_line_windows`` runs its jnp
            # ops abstractly, the cached ``pc`` holds DynamicJaxprTracers tied to
            # that trace, and reusing the cache in a later trace raises
            # UnexpectedTracerError. Inputs are the concrete build-time SSP grid,
            # so this is a genuine compile-time constant.
            with jax.ensure_compile_time_eval():
                pc = precompute_line_windows(
                    self.ssp_data.ssp_wave, self.ssp_data.ssp_flux, line_defs
                )
            cache[key] = pc
        return pc

    def _require_feature_fast_eligible(self, chain, *, caller="predict_spectral_indices"):
        r"""Return the stellar component, or raise if the chain is unsupported.

               The window LUT reconstructs the measurement from ``scale * sum(jw * SSP * T)``,
        the baked-in stellar+dust-screen SED only. Any component that adds
               rest-frame flux the LUT does not model (additive nebular, AGN, radio, …)
               would make the fast measurement silently wrong, so those raise. IGM is
               rest-frame-neutral (observer-frame, applied after redshifting) and is
               allowed.

               Dust *emission* is admitted for line fluxes but not for indices, and the
               asymmetry is physical rather than a concession. A line flux is measured as
               the window integral minus a continuum fitted from the sidebands. Dust IR
               emission contributes a smooth continuum that is common to the window and
               its sidebands, so it cancels in that subtraction: measured bias on the ten
               DESI optical lines stays below :math:`10^{-7}` even at
               :math:`\tau_{\rm bc}=4,\ \tau_{\rm diff}=3`, where the IR term is already
               3% of the continuum *level*. A break index is a flux **ratio** of two
               bands, with no such subtraction, so the same smooth offset does not cancel
               and the exclusion stands.

               Parameters
               ----------
               chain: list
                   The component chain to validate.
               caller: str, default "predict_spectral_indices"
                   Name used in the error messages, and the switch for the dust-emission
                   rule: ``"measure_line_fluxes"`` admits a dust-IR component.

               Returns
               -------
               StellarSEDComponent
                   The chain's stellar component.
        """
        from tengri.components.dust.emission._component_base import EmissionComponent
        from tengri.components.dust.two_component import DustSEDComponent
        from tengri.components.igm.component import IGMSEDComponent
        from tengri.components.nebular.component import NebularSEDComponent
        from tengri.components.stellar.component import StellarSEDComponent

        allowed = [StellarSEDComponent, DustSEDComponent, NebularSEDComponent, IGMSEDComponent]
        if caller == "measure_line_fluxes":
            allowed.append(EmissionComponent)
        allowed = tuple(allowed)
        stellar = next((c for c in chain if isinstance(c, StellarSEDComponent)), None)
        if stellar is None:
            raise ValueError(f"{caller}(approx=True) requires a stellar component.")
        for c in chain:
            if not isinstance(c, allowed):
                raise ValueError(
                    f"{caller}(approx=True) does not support a "
                    f"{type(c).__name__} in the chain: it adds rest-frame flux the window "
                    f"LUT does not model. Use approx=False for this model."
                )
        neb = next((c for c in chain if isinstance(c, NebularSEDComponent)), None)
        if neb is not None and getattr(neb.config, "backend", None) != "baked_in":
            raise ValueError(
                f"{caller}(approx=True) supports baked-in nebular only "
                f"(chain has backend={getattr(neb.config, 'backend', None)!r}); an additive "
                f"backend's emission is not in the SSP window integrals. Use approx=False."
            )
        return stellar

    def _feature_fast_indices(self, params, index_defs):
        """FeaturePrecomp window-LUT measurement of ``index_defs`` (``approx=True``).

        Contracts the precomputed SSP window integrals with SED-free SFH+met
        weights (:meth:`StellarSEDComponent.compute_joint_weights`) and the
        model's per-age two-component dust screen
        (:meth:`DustSEDComponent.compute_transmission`), no full-grid SED. See
        :meth:`predict_spectral_indices` for the supported-configuration
        contract; unsupported chains raise here (never silently wrong).
        """
        from tengri.components.dust.two_component import DustSEDComponent
        from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S
        from tengri.observation.spectral_indices import (
            measure_index_jax,
            measure_indices_from_window_lut,
        )

        chain = self._feature_chain()
        stellar = self._require_feature_fast_eligible(chain)

        # SED-free (met, age) weights, raises on unsupported SFH / metallicity.
        joint_weights, total_mass, ssp_ages_yr = stellar.compute_joint_weights(params)
        scale = total_mass * LSUN_ERG_PER_S  # physical window means; cancels for ratios

        pc = self._index_window_precomp(index_defs)

        # per-age transmission at the window centers, from the model's own dust
        # (single-sourced with the forward), or unity when there is no dust.
        dust = next((c for c in chain if isinstance(c, DustSEDComponent)), None)
        if dust is None:
            transmission = jnp.ones((ssp_ages_yr.shape[0], pc.window_centers.shape[0]))
        else:
            transmission = dust.compute_transmission(params, pc.window_centers, ssp_ages_yr)

        values = measure_indices_from_window_lut(joint_weights, scale, transmission, pc)

        # Slope indices are not a single-window functional → the LUT leaves NaN
        # in those slots; fill them from one exact rest-frame SED measurement.
        if pc.has_slope:
            rest = self._predict_rest_sed(params)
            slots = pc.index_slots
            values = jnp.stack(
                [
                    measure_index_jax(rest.wavelength, rest.sed, d)
                    if slots[i][0] == "slope"
                    else values[i]
                    for i, d in enumerate(index_defs)
                ]
            )
        return values

    def measure_line_fluxes(self, params, line_defs=None, *, approx=False, state=None, fast=UNSET):
        r"""Emission-line fluxes **measured from the model spectrum**, catalog-style.

        The counterpart to :meth:`predict_line_fluxes`: where ``predict_*`` returns
        what the galaxy *emits* (the backend's nebular line luminosity → flux),
        ``measure_*`` applies the operator a spectroscopic pipeline applies to
        *data*, estimate a local continuum from side-bands, subtract it, integrate
        the emission, to the model's own rest-frame SED. It therefore works for
        **any** nebular backend (Cue additive, baked-in wNE, …), yields a quantity
        directly comparable to a catalog's continuum-subtracted line flux, and
        carries the stellar Balmer absorption under the line self-consistently.

        Parameters
        ----------
        params: dict
            Parameter values (public names). ``redshift`` sets the luminosity
            distance for the observed flux.
        line_defs: sequence of LineDef, optional
            Lines + continuum windows to measure. Defaults to the lines this
            model's ``observation`` declares (``Observation.line_fluxes``), and
            only to :data:`tengri.observation.line_measurement.DESI_LINES` when
            nothing declares a set. Before #1500 the DESI list was used
            unconditionally, so a model built with an eight-line
            :class:`~tengri.observation.LineFluxData` silently returned **five**
            fluxes, for different lines, in a different order.
        approx: bool, default False
            Route through the window-LUT path
            (:func:`~tengri.observation.line_measurement.measure_line_fluxes_from_window_lut`):
            SED-free SFH weights × precomputed SSP line-window integrals × the
            per-age dust screen, no full-grid SED. Bit-exact with the exact path
            for the supported configuration (stellar + two-component/no dust +
            baked-in/no nebular, delta metallicity, parametric non-field SFH) and
            **raises** otherwise (same contract as
            :meth:`predict_spectral_indices` ``approx=True``). An **additive** Cue
            backend is *not* eligible, its emission is not in the SSP window
            integrals, so use ``approx=False`` for Cue.

            Named for the build-time ``approx=FeaturePrecomp(...)`` it selects.
            Spelled ``fast`` until 2026-08.
        state: ForwardState, optional
            Pre-computed forward state to measure on (exact path only).
        fast: bool, optional
            Deprecated spelling of `approx`. Removed in v1.0.

        Returns
        -------
        jnp.ndarray, shape (n_line,)
            Observed emission-line fluxes [erg/s/cm^2], in ``line_defs`` order.

        Notes
        -----
        **JIT-compatible / differentiable**: yes.

        **Continuum caveat.** The side-band continuum carries the stellar
        absorption around the line; for baked-in nebular the Balmer flux is
        continuum-sensitive exactly as a real pipeline's is. See the
        :mod:`tengri.observation.line_measurement` module docstring.
        """
        from tengri.cosmology import luminosity_distance
        from tengri.forward.result import SEDResult
        from tengri.observation.line_measurement import (
            measure_line_flux_jax,
            measure_line_fluxes_from_window_lut,
            resolve_line_defs,
        )

        approx = resolve_renamed_flag(
            approx,
            fast,
            old_name="fast",
            new_name="approx",
            caller="SEDModel.measure_line_fluxes",
        )
        # Omitting ``line_defs`` used to mean DESI_LINES unconditionally, ignoring
        # the model's own Observation: a model built with an eight-line
        # LineFluxData returned FIVE fluxes, for different lines, in a different
        # order. Nothing raised -- the shape is only wrong downstream (#1500).
        line_defs = resolve_line_defs(line_defs, getattr(self, "observation", None))
        # Resolve the redshift through the spec, not out of the dict. A Fixed
        # redshift is legitimately absent from ``params``, and reading it back with
        # a 0.0 default put the galaxy at 10 pc, 1e17 too bright, silently
        # (#1127). ``_get_redshift`` lets an explicit value win, falls back to the
        # fixed one, and raises if the model has neither.
        z = jnp.asarray(self._get_redshift(params))
        dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
        # log10, never the linear divisor: 4 pi d_L^2 is ~1e57 (and ~1.2e40 even
        # at the 10-pc z=0 convention) against a float32 ceiling of 3.4e38, so the
        # linear form is ``inf`` at every distance and the flux ``nan`` (#1859).
        log10_4pi_dl2 = log10_four_pi_dl2(dl_cm)

        if approx:
            from tengri.components.dust.two_component import DustSEDComponent
            from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S

            chain = self._feature_chain()
            stellar = self._require_feature_fast_eligible(chain, caller="measure_line_fluxes")
            joint_weights, total_mass, ssp_ages_yr = stellar.compute_joint_weights(params)
            scale = total_mass * LSUN_ERG_PER_S
            pc = self._line_window_precomp(line_defs)
            dust = next((c for c in chain if isinstance(c, DustSEDComponent)), None)
            if dust is None:
                transmission = jnp.ones((ssp_ages_yr.shape[0], pc.window_centers.shape[0]))
            else:
                transmission = dust.compute_transmission(params, pc.window_centers, ssp_ages_yr)
            return measure_line_fluxes_from_window_lut(
                joint_weights, scale, transmission, pc, log10_4pi_dl2
            )

        if state is None:
            rest = self._predict_rest_sed(params)
        else:
            rest = SEDResult(wavelength=state.wave, sed=state.sed_intrinsic)
        return jnp.stack(
            [
                measure_line_flux_jax(rest.wavelength, rest.sed, ld, log10_4pi_dl2)
                for ld in line_defs
            ]
        )

    def predict_hbeta(self, params: dict) -> float:
        """Predict Hβ luminosity for use with CLOUDY-informed emission line priors.

        Required by ``marginalize_emission_lines_cloudy()`` as the ``l_hbeta``
        argument, which scales CLOUDY's ratio-relative-to-Hβ priors to physical
        units.

        **Raw forward-pass output** (single scalar). For interactive access
        to Balmer lines and ratios, see ``model.predict(params).lines.hbeta``
        / ``.lines.balmer_decrement``.

        Hβ luminosity is computed via the Case B recombination approximation
        (Leitherer et al. 1999):

        .. math::

            L_{H\\beta} \\approx 5.22 \\times 10^7 \\times \\text{SFR}_{10} \\; [L_\\odot]

        where :math:`\\text{SFR}_{10}` is the SFR averaged over the last 10 Myr
        (the ionizing-photon relevant timescale), derived from
        Q_H ≈ 4.2 × 10⁵³ × SFR [photons/s] and
        L_Hβ = 4.76 × 10⁻¹³ × Q_H erg/s converted to L_sun.

        Parameters
        ----------
        params: dict
            Model parameters (from ``spec.sample()`` or a ``Posterior``).

        Returns
        -------
        float
            Hβ luminosity [Lsun].

        Examples
        --------
        >>> l_hbeta = model.predict_hbeta(params)
        >>> ln_L = marginalize_emission_lines_cloudy(
        ...     residual,
        ...     noise,
        ...     A,
        ...     log_z=params["met_logzsol"],
        ...     neb_logU=-3.0,
        ...     l_hbeta=l_hbeta,
        ... )

        Notes
        -----
        **JIT-compatible**: no, wraps :meth:`predict_sfh_quantities`.

        Uses Case B recombination coefficients (Leitherer et al. 1999 [1]_).
        If SFH computation fails (e.g., invalid params), returns safe fallback of 1 L_sun.

        See Also
        --------
        predict_sfh_quantities: JIT-compatible SFH quantities including sfr_10myr.

        References
        ----------
        .. [1] C. Leitherer et al., "Starburst99: Synthesis Models for Galaxies
           with Active Star Formation," ApJS, 123, 3 (1999).
           arXiv:astro-ph/9807340.

        .. deprecated:: 2026-07 (cleanup PR-2)
            Interactive getter moved to the lazy Prediction wrapper:
            ``model.predict(params).lines.hbeta`` (one cached forward pass shared across
            all derived quantities). Removed in tengri v1.0.
        """
        warnings.warn(
            "predict_hbeta() is deprecated, use "
            "model.predict(params).lines.hbeta instead "
            "(cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Case B: L_Hbeta [Lsun] = 4.76e-13 * Q_H, Q_H = 4.2e53 * SFR [Msun/yr]
        # => L_Hbeta = 4.76e-13 * 4.2e53 / 3.828e33 * SFR ≈ 5.22e7 * SFR
        _L_HBETA_PER_SFR = 5.22e7  # Lsun per Msun/yr (Leitherer+1999)
        try:
            sfh_q = self._predict_sfh_quantities(params)
            sfr_10 = float(sfh_q.sfr_10myr)
            sfr_10 = max(sfr_10, 1e-10)
            return float(_L_HBETA_PER_SFR * sfr_10)
        except (AttributeError, TypeError, ValueError):
            # AttributeError: predict_sfh_quantities doesn't exist or sfr_10myr missing
            # TypeError: float() conversion failed (JAX tracer or wrong type)
            # ValueError: invalid params
            return 1.0  # 1 Lsun safe fallback

    def predict_derived(self, params):
        """Deprecated. Use ``model.predict(params).properties``.

        .. deprecated:: 2026-07
           Superseded by the property catalog and the ``Prediction`` surface
           (#1043 contract §2). The body is unchanged, this shim is bit-exact
           with the method it replaces, so migrating changes no number.
           Will be removed in tengri v1.0.

        Returns
        -------
        Same as :meth:`_predict_derived`.
        """
        warnings.warn(
            "predict_derived() is deprecated, use model.predict(params).properties "
            "instead (cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._predict_derived(params)

    def _predict_derived(self, params):
        """Compute derived physical quantities as a flat dict.

        Convenience wrapper around :meth:`predict` that extracts the key
        SFH-derived scalars into a plain dict. Use :meth:`predict` for
        lazy on-demand access to all quantities, or
        :meth:`predict_sfh_quantities` for JIT-compatible batch computation.

        Parameters
        ----------
        params: dict
            Parameter values.

        Returns
        -------
        dict with keys:
            "stellar_mass": total mass formed [M_sun]
            "stellar_mass_surviving": surviving mass in living stars +
                remnants [M_sun] or None if mass-remaining table not loaded.
            "sfr_100myr": SFR averaged over last 100 Myr [M_sun/yr]
            "sfr_10myr": SFR averaged over last 10 Myr [M_sun/yr]
            "ssfr": specific SFR [yr^-1], uses surviving mass if
                available, else formed mass.

        Notes
        -----
        **JIT-compatible**: no, wraps :meth:`predict`.

        Convenience wrapper around the lazy :meth:`predict` object.
        For batch operations, use :meth:`predict_sfh_quantities` directly
        with :func:`jax.vmap`.
        """
        pred = self.predict(params)
        mass_surv = pred.sfh.stellar_mass_surviving
        # Return None (not NaN) when mass-remaining table is absent
        mass_surv_out = None if jnp.isnan(mass_surv) else mass_surv
        return {
            "stellar_mass": pred.sfh.stellar_mass,
            "stellar_mass_surviving": mass_surv_out,
            "sfr_100myr": pred.sfh.sfr_100myr,
            "sfr_10myr": pred.sfh.sfr_10myr,
            "ssfr": pred.sfh.ssfr,
        }

    @property
    def available_properties(self) -> tuple[str, ...]:
        """Names of all properties available for this model.

        Returns
        -------
        tuple[str, ...]
            Sorted tuple of property names that can be queried via
            :meth:`predict_properties` or accessed as attributes on
            a :class:`Prediction` object.

        Raises
        ------
        ValueError
            If two active components declare the same property name.
            Each property can be owned by at most one active component.

        Notes
        -----
        This property triggers assembly of the property catalog on first
        access and caches the result in ``_property_catalog`` for subsequent
        calls. The collision check runs at this point.

        Examples
        --------
        >>> model = SEDModel.build(...)
        >>> model.available_properties
        ('stellar_mass', 'stellar_mass_surviving', 'sfr_10myr', ...)
        """
        return tuple(sorted(self._ensure_property_catalog().keys()))

    def _ensure_property_catalog(self):
        """Assemble the model's property catalog once, and return it.

        The catalog is built lazily from the active component chain. This is the
        **only** place that assembles it: it used to be inlined in both
        ``available_properties`` and ``predict_properties``, while
        ``Prediction.properties`` read ``_property_catalog`` directly, so on a
        fresh model, ``pred.properties["stellar_mass"]`` raised ``AttributeError``
        unless the user happened to touch one of the other two first (#1131).
        """
        if not hasattr(self, "_property_catalog"):
            from tengri.forward.properties import assemble_available_properties

            chain = getattr(self, "_cached_component_chain", None)
            if chain is None:
                chain = self._build_component_chain()
            active_names = {c.name for c in chain}
            self._property_catalog = assemble_available_properties(active_names)
        return self._property_catalog

    def predict_properties(self, params, names=None, *, ssp_data=None, template_data=None):
        """Compute derived properties from the forward state.

        Properties are computed from the same orchestrator :class:`ForwardState`
        that powers all other predictions, so their values are consistent with
        photometry, spectrum, and SED quantities. This method routes through
        :meth:`predict_state`, enabling JIT/vmap/grad by treating property
        names as static arguments.

        Parameters
        ----------
        params: dict
            Parameter values using public parameter names.
        names: tuple[str] or list[str], optional
            Property names to compute. If None, computes all available
            properties. Each name must be in :attr:`available_properties`,
            else :exc:`KeyError` is raised.
        ssp_data, template_data: Any | None, keyword-only, optional
            The JIT-threading channel, forwarded to :meth:`predict_state`. Pass
            these only when wrapping this method in your own ``jax.jit`` /
            ``vmap`` / ``grad``, where closure-captured grids would otherwise
            bake into your compiled program as constants, see
            :meth:`predict_photometry` for the measured cost (#1753). ``None``
            (default) uses the model's own arrays.

        Returns
        -------
        dict[str, scalar]
            Mapping of property name to JAX scalar value [various units].
            Returned as a plain dict with JAX array values.

        Raises
        ------
        KeyError
            If any name in ``names`` is not in :attr:`available_properties`.
            The error message lists the available names.

        Notes
        -----
        **JIT/vmap/grad-compatible**: :func:`jax.jit` and :func:`jax.vmap`
        handle this method correctly provided ``names`` is treated as a
        static argument (the function parameter itself is not JIT-traced).
        Wrap the call like::

            @jax.jit
            def compute_stellar_mass(p):
                return model.predict_properties(p, names=("stellar_mass",))["stellar_mass"]

        The returned dict has JAX-array values, so downstream operations on
        the scalars are fully differentiable.

        **Single-galaxy interactive use**: for exploration, use
        :meth:`predict` and access the ``Prediction.properties`` catalog
        with attribute syntax (e.g., ``pred.stellar_mass``) to share cached
        state across multiple property accesses.

        **Legacy compatibility**: :meth:`predict_sfh_quantities` and
        :meth:`predict_sed_quantities` remain unchanged and use independent
        computation paths; agreement is documented per test.

        Examples
        --------
        **Compute one property:**

        >>> model = SEDModel.build(...)
        >>> params = {...}
        >>> props = model.predict_properties(params, names=("stellar_mass",))
        >>> print(props["stellar_mass"])

        **Compute all properties:**

        >>> all_props = model.predict_properties(params)
        >>> print(f"Stellar mass: {all_props['stellar_mass']} Msun")
        >>> print(f"SSFR: {all_props['ssfr']} yr^-1")

        **Under jax.jit with vmap:**

        >>> vmap_props = jax.vmap(lambda p: model.predict_properties(p, names=("stellar_mass",)))(
        ...     params_batch
        ... )
        >>> print(vmap_props["stellar_mass"].shape)  # (N,)

        See Also
        --------
        available_properties: List of properties available in this model.
        predict: Lazy Prediction object with attribute-access syntax.
        """
        self._ensure_property_catalog()

        # Resolve names to compute (default = all)
        if names is None:
            names_to_compute = tuple(sorted(self._property_catalog.keys()))
        else:
            names_to_compute = tuple(names)  # Ensure tuple for consistency

        # Validate all names are known
        unknown = set(names_to_compute) - set(self._property_catalog.keys())
        if unknown:
            from tengri.forward.properties import missing_property_message

            raise KeyError(
                missing_property_message(*sorted(unknown), available=self._property_catalog)
            )

        # A 'lines' property on a backend with no per-line catalog is NaN. Say
        # so here too: pred.lines.* has warned since #361 but this surface --
        # the documented jit/vmap one -- returned the same NaN in silence.
        # Only when the caller asked by name; `names=None` means "everything
        # the model has" and would warn on every default call.
        if names is not None:
            from tengri.forward.properties import warn_if_lines_are_unavailable

            warn_if_lines_are_unavailable(self, names_to_compute)

        # main carries a ``_refuse_on_fast_nebular`` guard on this line (#1665).
        # It is deliberately NOT taken here, for the reason recorded in full at
        # ``predict_photometry``: #1673 fixed the cause rather than the symptom,
        # so ``predict_state`` materializes the nebular component and every
        # property below is bit-exact on the fast path (measured rel 0.0). The
        # method it calls no longer exists on this branch, and
        # ``_FAST_NEBULAR_UNSAFE_PROPERTIES`` survives as the census of
        # nebular-dependent properties worth CHECKING, not a refusal list --
        # see ``test_sed_derived_properties_are_exact_on_the_fast_path``, which
        # calls this surface on the fast path and asserts equality with the
        # exact model. Reinstating the refusal would make that test raise
        # instead of compare.

        # Compute the state once
        state = self.predict_state(params, ssp_data=ssp_data, template_data=template_data)

        # Evaluate each property
        result = {}
        for name in names_to_compute:
            entry = self._property_catalog[name]
            result[name] = entry.fn(state, params)

        return result

    def predict_sfh_quantities(self, params):
        """Deprecated. Use ``model.predict_properties(params, names=(...))``.

        .. deprecated:: 2026-07
           Superseded by the property catalog and the ``Prediction`` surface
           (#1043 contract §2). The body is unchanged, this shim is bit-exact
           with the method it replaces, so migrating changes no number.
           Will be removed in tengri v1.0.

        Returns
        -------
        Same as :meth:`_predict_sfh_quantities`.
        """
        warnings.warn(
            "predict_sfh_quantities() is deprecated, use "
            "model.predict_properties(params, names=(...)) instead "
            "(cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._predict_sfh_quantities(params)

    def _predict_sfh_quantities(self, params):
        """Compute SFH-derived quantities in JIT-compatible form.

        Integrates the SFH to compute stellar mass, recent SFR, specific SFR,
        and mass-weighted age/metallicity. Returns a :class:`SFHQuantities`
        NamedTuple that is fully JIT-compatible and vmap-ready for batch
        inference over posterior chains or mock catalogs.

        **Use this method for** JIT/batch loops (``jax.vmap``, ``jit``,
        ``grad``). **For interactive single-galaxy exploration**, use
        :meth:`predict` and access ``pred.sfh.stellar_mass`` etc., same
        quantities, with Python-side caching.

        Parameters
        ----------
        params: dict
            Parameter values using public parameter names.

        Returns
        -------
        SFHQuantities
            NamedTuple with fields:

            - ``stellar_mass``: float. Total stellar mass formed [M☉]
            - ``stellar_mass_surviving``: float. Mass in living stars + remnants [M☉],
              or NaN if SSP mass-remaining tables not loaded.
            - ``sfr_100myr``: float. SFR time-averaged over last 100 Myr [M☉/yr]
            - ``sfr_10myr``: float. SFR time-averaged over last 10 Myr [M☉/yr]
            - ``ssfr``: float. Specific SFR (SFR/M_surv or SFR/M_formed) [yr⁻¹]
            - ``mass_weighted_age_gyr``: float. Mass-weighted age [Gyr]
            - ``mass_weighted_metallicity``: float. Mass-weighted log₁₀(Z/Z☉)
              [dex], the same convention ``met_logzsol`` is set in. The old
              "or absolute log₁₀(Z) depending on metallicity mode" hedge was
              wrong twice over: the value was absolute in *every* mode, and
              the computation has no metallicity-mode branch to vary on (#1703)

        Notes
        -----
        **JIT-compatible**: yes, all operations use ``jnp`` primitives.
        Safe inside :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`.

        **Gradient-safe**: yes, all quantities are differentiable w.r.t.
        SFH and metallicity parameters.

        **Surviving mass**: Requires SSP grid with ``ssp_mass_remaining``
        (e.g., FSPS grids). If unavailable, returns NaN. :meth:`predict`
        handles NaN gracefully when the quantity is unavailable.

        **SFR averaging**: Time-weighted mean over lookback-time window:

        .. math::

            \\langle\\mathrm{SFR}\\rangle_T =
                \\frac{\\sum_i \\mathrm{SFR}_i \\Delta t_i}{\\sum_i \\Delta t_i}

        where :math:`i` ranges over all ages :math:`\\leq T`. Uses symmetric
        bin widths (``jnp.gradient``) to avoid trapezoid boundary artifacts.

        **Mass-weighted age**: Computed as

        .. math::

            t_\\mathrm{mw} = \\frac{\\sum_i w_i t_i}{\\sum_i w_i}

        where :math:`w_i` are stellar population weights (age-integrated SFR).

        Examples
        --------
        **Single galaxy:**

        >>> sfh = model._predict_sfh_quantities(params)
        >>> sfh.stellar_mass
        Array(1.23e10, dtype=float64)

        **Batch over 10,000 posterior samples:**

        >>> import jax
        >>> sfh_fn = jax.vmap(model.predict_sfh_quantities)
        >>> sfh_batch = sfh_fn(params_batch)
        >>> sfh_batch.stellar_mass  # shape (10000,)
        >>> print(sfh_batch.stellar_mass.mean())

        See Also
        --------
        predict: Lazy prediction for single-galaxy exploration (non-JIT).
        predict_sfh: SFH on linear-time grid for visualization.
        predict_sed_quantities: JIT-compatible SED quantities.
        """
        from tengri.forward.prediction import SFHQuantities
        from tengri.utils.sed_quantities import (
            compute_mass_weighted_age,
            compute_mass_weighted_metallicity,
        )

        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)

        # ONE definition of the age weights: the ones the SED was built from.
        #
        # This used to branch on ``csp_integration`` and compute its own weights
        # for 'log_interp' / 'dsps_native' / 'dsps_met_table'. Since the stellar
        # component always integrates with cloud-in-cell (``sps_backend="dsps"``
        # is the only backend), those branches changed the reported mass without
        # changing the spectrum it was fitted to: 0.32% for 'log_interp' and
        # 'dsps_native', and NaN for 'dsps_met_table' -- which ``Posterior`` then
        # vmapped over every sample. Measured in #1500.
        #
        # The default already routed here "so predict_sfh_quantities returns the
        # same stellar_mass / weights as predict_derived ... was 4.1% apart with
        # the legacy rectangle rule". That consistency fix was applied to the
        # values someone happened to test; the rest kept the divergent path. Now
        # there is no path to diverge.
        state_orch = self.predict_state(params)
        weights = jnp.asarray(state_orch.derived["age_weights"])
        mass_formed = jnp.sum(weights)

        # Surviving mass
        if self.ssp_data.ssp_mass_remaining is not None:
            from tengri.components.stellar.sps.dsps_wrapper import (
                compute_surviving_mass,
                interpolate_mass_remaining,
            )

            log_z = p.get("log_z_abs", 0.0)
            mr_at_met = interpolate_mass_remaining(
                self.ssp_data.ssp_mass_remaining,
                self.ssp_data.ssp_lgmet,
                log_z,
            )
            mass_surviving = compute_surviving_mass(weights, mr_at_met)
        else:
            mass_surviving = jnp.array(jnp.nan)

        # SFR averages, time-weighted mean over a lookback-time window.
        # <SFR>_T = sum(SFR_i * dt_i) / sum(dt_i)  for all age_i <= T.
        # Use jnp.gradient for symmetric bin widths; avoids the trapezoid boundary
        # artifact where zeroing SFR outside the window but keeping the full age
        # axis creates a phantom half-bin contribution at the window edge.
        dt = jnp.gradient(self.age_yr)
        mask_100 = self.age_yr <= 1e8
        numerator_100 = jnp.sum(jnp.where(mask_100, sfr * dt, 0.0))
        denom_100 = jnp.maximum(jnp.sum(jnp.where(mask_100, dt, 0.0)), 1.0)
        sfr_100myr = jnp.where(jnp.sum(mask_100) > 1, numerator_100 / denom_100, sfr[0])

        mask_10 = self.age_yr <= 1e7
        numerator_10 = jnp.sum(jnp.where(mask_10, sfr * dt, 0.0))
        denom_10 = jnp.maximum(jnp.sum(jnp.where(mask_10, dt, 0.0)), 1.0)
        sfr_10myr = jnp.where(jnp.sum(mask_10) > 1, numerator_10 / denom_10, sfr[0])

        # sSFR
        mass_for_ssfr = jnp.where(jnp.isnan(mass_surviving), mass_formed, mass_surviving)
        ssfr = sfr_100myr / jnp.maximum(mass_for_ssfr, 1.0)

        # Mass-weighted age and metallicity
        mw_age = compute_mass_weighted_age(weights, self.ssp_ages_yr)
        # The helper works in the grid's absolute log10(Z) (its inputs are the
        # ``log_z_abs*`` params); convert on the way out, as the property
        # catalog does, so this deprecated surface reports the same number.
        from tengri.utils.conversions import log_z_abs_to_logzsol

        mw_z = log_z_abs_to_logzsol(
            compute_mass_weighted_metallicity(
                weights,
                self.ssp_ages_yr,
                p.get("log_z_abs", 0.0),
                log_z_initial=p.get("log_z_abs_initial"),
                log_z_final=p.get("log_z_abs_final"),
            )
        )

        return SFHQuantities(
            stellar_mass=mass_formed,
            stellar_mass_surviving=mass_surviving,
            sfr_100myr=sfr_100myr,
            sfr_10myr=sfr_10myr,
            ssfr=ssfr,
            mass_weighted_age_gyr=mw_age,
            mass_weighted_metallicity=mw_z,
        )

    def predict_sed_quantities(self, params):
        """Deprecated. Use ``model.predict_properties(params, names=(...))``.

        .. deprecated:: 2026-07
           Superseded by the property catalog and the ``Prediction`` surface
           (#1043 contract §2). The body is unchanged, this shim is bit-exact
           with the method it replaces, so migrating changes no number.
           Will be removed in tengri v1.0.

        Returns
        -------
        Same as :meth:`_predict_sed_quantities`.
        """
        warnings.warn(
            "predict_sed_quantities() is deprecated, use "
            "model.predict_properties(params, names=(...)) instead "
            "(cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._predict_sed_quantities(params)

    def _predict_sed_quantities(self, params):
        """Compute SED-derived quantities in JIT-compatible form.

        Evaluates the full forward model and computes UV slope, spectral
        indices (D4000, Balmer break), bolometric/IR luminosities, dust
        attenuation, and luminosity-weighted age/metallicity. Returns
        a :class:`SEDQuantities` NamedTuple that is fully JIT-compatible
        and vmap-ready for batch inference.

        **Use this method for** JIT/batch loops (``jax.vmap``, ``jit``,
        ``grad``). **For interactive single-galaxy exploration**, use
        :meth:`predict` and access ``pred.sed.dn4000``, ``pred.sed.uv_slope``
        etc., same quantities, with Python-side caching.

        Parameters
        ----------
        params: dict
            Parameter values using public parameter names.

        Returns
        -------
        SEDQuantities
            NamedTuple with fields:

            - ``l_bol``: float. Bolometric luminosity [L☉]
            - ``l_tir``: float. Total infrared (8–1000 μm) luminosity [L☉]
            - ``l_dust_absorbed``: float. Dust-absorbed luminosity [L☉]
              (intrinsic − attenuated), or NaN if intrinsic SED unavailable.
            - ``irx``: float. Infrared excess := L_TIR / L_UV(1600 Å).
              Common probe of dust obscuration (Dale et al. 2001).
            - ``uv_slope_beta``: float. UV slope (power-law index) in
              f_λ ∝ λ^β for 1200–2600 Å.
            - ``dn4000``: float. D_n(4000) break ratio: flux average
              at 3750–3950 Å / 4050–4250 Å. Indicator of stellar age.
            - ``balmer_break``: float. Balmer break: flux ratio
              ~3700 Å / ~4000 Å. Old stellar population signature.
            - ``m_uv``: float. Absolute magnitude at 1500 Å
              (M_1500, standard reionization-era indicator).
            - ``fuv_flux``: float. Flux at 1500 Å [erg/s/cm²]
            - ``nuv_flux``: float. Flux at 2300 Å [erg/s/cm²]
            - ``fuv_flux_intrinsic``: float. FUV flux, dust-free
              (intrinsic SED). NaN if unavailable.
            - ``nuv_flux_intrinsic``: float. NUV flux, dust-free. NaN
              if unavailable.
            - ``rest_uv_color``: float. Rest-frame UV color (f_1500 − f_2300).
            - ``luminosity_weighted_age_gyr``: float. Luminosity-weighted
              age [Gyr] (∫L_λ age dλ / ∫L_λ dλ).
            - ``luminosity_weighted_metallicity``: float. Luminosity-weighted
              log₁₀(Z/Z☉) or absolute log₁₀(Z).

        Notes
        -----
        **JIT-compatible**: yes, all operations use ``jnp`` primitives.
        Safe inside :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`.

        **Gradient-safe**: yes, all quantities are differentiable w.r.t.
        SFH, metallicity, and dust parameters.

        **Spectral indices**: Computed directly on the rest-frame SED
        (not broadband-filtered). All wavelengths defined in rest frame.

        **Dust-absorbed luminosity**: Defined as L_dust = L_intrinsic − L_attenuated
        (i.e., the energy re-radiated in the IR). Requires the forward model
        to track both intrinsic and attenuated SEDs internally. Returns NaN if
        ``dust_model="none"`` or intrinsic SED not available.

        **Luminosity-weighted quantities**: Computed as:

        .. math::

            \\langle Q \\rangle_L = \\frac{\\int L_\\lambda(\\lambda) Q(\\lambda) d\\lambda}
                                        {\\int L_\\lambda(\\lambda) d\\lambda}

        More sensitive to young, UV-bright populations than mass-weighted age.

        Examples
        --------
        **Single galaxy:**

        >>> sed_q = model._predict_sed_quantities(params)
        >>> sed_q.l_bol
        Array(2.5e10, dtype=float64)
        >>> sed_q.dn4000
        Array(1.42, dtype=float64)
        >>> sed_q.irx
        Array(1.87, dtype=float64)

        **Batch over posterior samples:**

        >>> import jax
        >>> sed_fn = jax.vmap(model.predict_sed_quantities)
        >>> sed_batch = sed_fn(params_batch)
        >>> sed_batch.m_uv  # shape (n_samples,)
        >>> sed_batch.dn4000.mean()

        **Computing IRX − β relation:**

        >>> sed_q = sed_fn(params_batch)
        >>> irx = sed_q.irx
        >>> beta = sed_q.uv_slope_beta
        >>> # Compare to Meurer et al. (1999) IRX-β calibration

        See Also
        --------
        predict: Lazy prediction for single-galaxy exploration.
        predict_sfh_quantities: JIT-compatible SFH quantities.
        predict_rest_sed: Full rest-frame SED (for custom analysis).
        """
        # Dispatch to the orchestrator-backed bridge. Same semantics shift
        # PR 5a applied to ``predict_rest_sed``: the orchestrator's
        # stellar adapter uses DSPS-canonical (lognormal-MDF) CSP
        # integration unconditionally. For ``csp_integration='dsps_native'``
        # the legacy path produces identical results (sub-0.1% drift on
        # every published field). For the legacy default
        # ``csp_integration='trapz'``, the only field that drifts
        # noticeably (~12%) is ``luminosity_weighted_age_gyr``, the
        # orchestrator integrates the actual ``lnu_age`` cube whose
        # sum-over-age IS ``sed_intrinsic``, while legacy's per-bin
        # luminosity reconstruction has a hidden DSPS-joint-weight discrepancy
        # under trapz. The orchestrator value is the physically correct
        # one (energy-conserving by construction).
        from tengri.forward import state_to_sed_quantities

        return state_to_sed_quantities(self.predict_state(params))

    # ── Component orchestrator path ───────────────────────────────────

    def predict_sfh_quantities_components(self, params):
        """Deprecated alias of :meth:`predict_sfh_quantities`.

        .. deprecated:: 2026-07 (cleanup PR-2)
            The migration-era A/B twin is gone, the canonical method is
            orchestrator-routed. Removed in tengri v1.0.
        """
        warnings.warn(
            "predict_sfh_quantities_components() is deprecated, use "
            "state_to_sfh_quantities(model.predict_state(params)) for these "
            "exact numerics, or predict_sfh_quantities(params) / "
            "model.predict(params).sfh for the canonical surface. "
            "Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from tengri.forward import state_to_sfh_quantities

        return state_to_sfh_quantities(self.predict_state(params))

    def predict_sed_quantities_components(self, params):
        """Deprecated alias of :meth:`predict_sed_quantities`.

        .. deprecated:: 2026-07 (cleanup PR-2)
            The migration-era A/B twin is gone, the canonical method is
            orchestrator-routed. Removed in tengri v1.0.
        """
        warnings.warn(
            "predict_sed_quantities_components() is deprecated, use "
            "predict_sed_quantities(params) (same orchestrator numerics). "
            "Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._predict_sed_quantities(params)

    def predict_radio_quantities(self, params):
        """Orchestrator-path radio quantities.

        Returns
        -------
        RadioQuantities
            ``l_1p4ghz``, ``l_thermal``, ``l_nonthermal``, ``q_ir``.
            Fields are NaN if the configured chain has no
            :class:`RadioSEDComponent`.

        .. deprecated:: 2026-07 (cleanup PR-2)
            Interactive getter moved to the lazy Prediction wrapper:
            ``model.predict(params).radio`` (one cached forward pass shared across
            all derived quantities). Removed in tengri v1.0.
        """
        warnings.warn(
            "predict_radio_quantities() is deprecated, use "
            "model.predict(params).radio instead "
            "(cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from tengri.forward import state_to_radio_quantities

        return state_to_radio_quantities(self.predict_state(params))

    def predict_xray_quantities(self, params):
        """Orchestrator-path X-ray quantities.

        Returns
        -------
        XRayQuantities
            ``l_x_xrb``, ``l_x_agn``, ``l_x_total``.

        .. deprecated:: 2026-07 (cleanup PR-2)
            Interactive getter moved to the lazy Prediction wrapper:
            ``model.predict(params).xray`` (one cached forward pass shared across
            all derived quantities). Removed in tengri v1.0.
        """
        warnings.warn(
            "predict_xray_quantities() is deprecated, use "
            "model.predict(params).xray instead "
            "(cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from tengri.forward import state_to_xray_quantities

        return state_to_xray_quantities(self.predict_state(params))

    def predict_ionizing_quantities(self, params):
        """Orchestrator-path ionizing-photon quantities.

        Returns
        -------
        IonizingQuantities
            ``q_h``, ``xi_ion``.

        .. deprecated:: 2026-07 (cleanup PR-2)
            Interactive getter moved to the lazy Prediction wrapper:
            ``model.predict(params).ionizing`` (one cached forward pass shared across
            all derived quantities). Removed in tengri v1.0.
        """
        warnings.warn(
            "predict_ionizing_quantities() is deprecated, use "
            "model.predict(params).ionizing instead "
            "(cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from tengri.forward import state_to_ionizing_quantities

        return state_to_ionizing_quantities(self.predict_state(params))

    def _photometry_via_state(self, params):
        """Photometry through the orchestrator path (internal).

        Runs the SEDComponent chain on the model's configuration,
        then projects the resulting rest-frame SED through every
        filter in :attr:`self.observation.photometry`. Returns flux
        densities in the AB system at the source.

        Parameters
        ----------
        params: Mapping
            Free-parameter dict (same shape as
            :meth:`predict_state`).

        Returns
        -------
        flux_density: ndarray, shape (n_filters,)
            Observed flux densities [erg/s/cm²/Hz].

        Raises
        ------
        ValueError
            If no photometric filters are configured on the
            observation.

        Notes
        -----
        **JIT-compatible**: yes, uses :func:`jax.jit`-friendly
        :func:`tengri.observation.photometry.compute_flux_density`
        per filter.

        Differs from the legacy :meth:`predict_photometry`: this
        path goes through the SEDComponent orchestrator (no fused
        kernel dispatch); for inference workflows where you compile
        once and run thousands of times, the warm latency is
        equivalent (~2 ms). For one-shot photometry the legacy path
        with its tier-1/tier-2 fast paths is still faster.
        """
        if not self.observation.can_do_photometry:
            raise ValueError(
                "Photometry prediction requires photometric "
                "filters configured on the observation. Construct the "
                "model with ``filters=`` or pass an Observation that "
                "carries a Photometry instance."
            )
        state = self.predict_state(params)
        full = {**self.spec.get_fixed_values(), **params}
        return self.observation.predict(state, full)["phot_fnu"]

    def _spectrum_via_state(self, params, wave_obs=None):
        """Spectrum through the orchestrator path (internal).

        Runs the SEDComponent chain, applies the cosmological redshift +
        luminosity-distance projection, interpolates onto ``wave_obs``,
        and (if configured) applies the instrument LSF + velocity-dispersion
        broadening. Mirrors the contract of the legacy
        :meth:`predict_spectrum`'s observed-frame output but goes through
        the SEDComponent chain rather than the fused kernel.

        Parameters
        ----------
        params: Mapping
            Free-parameter dict (same shape as
            :meth:`predict_state`).
        wave_obs: array_like, shape (n_pix,), optional
            Observed-frame wavelength grid [Angstrom]. If ``None``,
            falls back to the precomputed grid (`self._wave_obs` or
            `self._precomputed.spectroscopy.wave_obs_pixels`).

        Returns
        -------
        flux: ndarray, shape (n_pix,)
            Observed-frame spectral flux density [erg/s/cm^2/Hz].

        Raises
        ------
        ValueError
            If no ``wave_obs`` grid is supplied or precomputed.

        Notes
        -----
        **JIT-compatible**: yes, :func:`run_components`, the rest→obs
        projection in :func:`~tengri.observation.spectrum.project_spectrum`, and
        LSF convolution are all JIT-compatible.

        **The flux calibration IS applied here** (since #1086). This routes through
        :meth:`Observation.predict`, which passes ``cal_c1..cN`` into
        :func:`~tengri.observation.spectrum.project_spectrum`. This note previously
        said the opposite, that callers should compose the calibration on top,         which would
        now apply the polynomial twice.
        """
        # (legacy dead ``self._precomputed.spectroscopy`` tier removed, #620)
        if wave_obs is None and hasattr(self, "_wave_obs"):
            wave_obs = self._wave_obs
        elif (
            wave_obs is None
            and self.observation is not None
            and getattr(self.observation, "spectroscopy", None) is not None
            and getattr(self.observation.spectroscopy, "wave_obs", None) is not None
        ):
            wave_obs = self.observation.spectroscopy.wave_obs
        elif wave_obs is None:
            raise ValueError(
                "Spectrum prediction requires a wave_obs grid "
                "(pass it explicitly, or build with "
                "Observation(spectroscopy=Spectroscopy(wave_obs=...)))."
            )

        state = self.predict_state(params)
        full = {**self.spec.get_fixed_values(), **params}
        return self.observation.predict(
            state,
            full,
            wave_obs=wave_obs,
            sigma_v_kms=self._get_sigma_v_kms(params),
            lsf_resolution=self._lsf_resolution,
            lsf_sigma_lib_kms=self._sigma_lib_kms,
            lsf_n_bins=self._lsf_n_bins,
        )["spec_fnu"]

    def predict_photometry_components(self, params):
        """Deprecated alias of the orchestrator photometry path.

        .. deprecated:: 2026-07 (cleanup PR-2)
            The legacy-vs-orchestrator A/B split is gone, every public
            predict method routes through the component chain now. Call
            :meth:`predict_photometry` instead. Removed in tengri v1.0.
        """
        warnings.warn(
            "predict_photometry_components() is deprecated, the orchestrator "
            "is the only forward path now; use predict_photometry(params). "
            "Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._photometry_via_state(params)

    def predict_spectrum_components(self, params, wave_obs=None):
        """Deprecated alias of the orchestrator spectrum path.

        .. deprecated:: 2026-07 (cleanup PR-2)
            The legacy-vs-orchestrator A/B split is gone, every public
            predict method routes through the component chain now. Call
            :meth:`predict_spectrum` instead. Removed in tengri v1.0.
        """
        warnings.warn(
            "predict_spectrum_components() is deprecated, the orchestrator "
            "is the only forward path now; use predict_spectrum(params). "
            "Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Deprecated, but it reaches the rest SED by its own route rather than
        # through predict_spectrum, so it needs the census guard too (#1665).
        return self._spectrum_via_state(params, wave_obs=wave_obs)

    def predict_emission_lines(self, params):
        """Orchestrator-path emission-line luminosities.

        Returns
        -------
        EmissionLines
            11 headline survey-diagnostic lines (``lya``, ``civ_1549``,
            ``oii``, ``hbeta``, ``oiii_4959/5007``, ``nii_6548/6584``,
            ``halpha``, ``sii_6717/6731``) plus the full backend catalog
            via ``all_waves`` / ``all_lums``. See
            :meth:`EmissionLines.get` for nearest-wavelength access to
            species the headline NamedTuple does not name (HeII 1640,
            [O III] 4363, ...). All luminosities in erg/s.

        Raises
        ------
        NotImplementedError
            When the active nebular backend does not publish a discrete
            line catalog (BakedIn or shock). Switch to ``neb={'type':
            'cue', ...}`` or ``neb={'type': 'cloudy_grid', ...}`` for
            discrete line predictions, or read the continuous nebular
            SED from ``model._predict_rest_sed(params).sed`` directly.

        Notes
        -----
        Dust attenuation is applied to the line luminosities in the
        attenuation regime selected by ``_neb_dust_mode`` (default
        ``"bc"``, birth-cloud + diffuse, Charlot & Fall 2000 [1]_).
        The line-attenuated values match the continuum treatment in
        :meth:`predict_rest_sed`, so Balmer decrement, BPT, and other
        line-ratio diagnostics behave correctly under a dust sweep
        (regression: issue #313).

        References
        ----------
        .. [1] S. Charlot & S. Fall, "A Simple Model for the Absorption of
           Starlight by Dust in Galaxies," ApJ 539, 718 (2000).

        .. deprecated:: 2026-07 (cleanup PR-2)
            Interactive getter moved to the lazy Prediction wrapper:
            ``model.predict(params).lines`` (one cached forward pass shared across
            all derived quantities). Removed in tengri v1.0.
        """
        warnings.warn(
            "predict_emission_lines() is deprecated, use "
            "model.predict(params).lines instead "
            "(cached, one forward pass). Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from tengri.components.nebular import BakedInBackend

        if isinstance(self._nebular_backend, BakedInBackend):
            raise NotImplementedError(
                "predict_emission_lines is not supported for the BakedIn "
                "nebular backend: emission is baked into the SSP grid and "
                "no discrete line catalog is published. To predict line "
                "luminosities, build the model with a photoionization "
                "backend, e.g. neb={'type': 'cue', 'all_params': FIXED} (requires "
                "a bare-stellar SSP) or neb={'type': 'cloudy_grid', ...}. "
                "For a quick narrow-band measurement on the BakedIn SED, "
                "integrate model._predict_rest_sed(params).sed across the "
                "line wavelength range yourself."
            )
        from tengri.forward import state_to_emission_lines
        from tengri.forward.emission_helpers import attenuate_emission

        state = self.predict_state(params)
        lines = state_to_emission_lines(state)
        if lines.all_waves.size == 0:
            # No discrete catalog; nothing to attenuate.
            return lines

        # Prefer the catalog the dust component published (#1867), so this
        # deprecated surface is on the SAME screen as `.lines`,
        # `predict_properties`, `predict_line_fluxes` and `predict_line_ratios`.
        # A deprecated method that disagrees with its replacement is worse than
        # one that is merely old, and this is the surface #313's regression
        # test drives, so it would have gone on asserting the right physics via
        # the one path that still had #1858's defect.
        _log_atten = state.derived.get("log_line_lums_attenuated")
        if _log_atten is not None:
            from tengri.utils.scale import pow10

            atten_lums = pow10(jnp.asarray(_log_atten))
        else:
            # Fallback for a chain that published no attenuated catalog. Charlot
            # & Fall 2000: lines from young populations (HII regions) experience
            # BC + diffuse; single-component dust applies the BC law twice
            # (degenerate fallback).
            tau_bc = jnp.asarray(params.get("dust_tau_bc", params.get("dust_tau_v", 0.0)))
            tau_diff = jnp.asarray(params.get("dust_tau_diff", 0.0))
            dust_kw = dict(
                dust_slope=jnp.asarray(params.get("dust_slope", -0.7)),
                dust_bump_strength=jnp.asarray(params.get("dust_bump_strength", 0.0)),
            )
            _is_single = self._dust_model == "single_component"
            atten_lums = attenuate_emission(
                lines.all_lums,
                lines.all_waves,
                self._neb_dust_mode,
                tau_bc,
                tau_diff,
                self._dust_law_bc_fn,
                self._dust_law_diff_fn if not _is_single else self._dust_law_bc_fn,
                neb_bc_fn=self._neb_dust_law_bc_fn,
                **dust_kw,
            )

        # Re-extract the headline scalars from the attenuated catalog
        # so EmissionLines.halpha / .hbeta / etc. reflect dust.
        from tengri.forward.prediction import EmissionLines
        from tengri.utils.sed_quantities import KEY_LINES, extract_line_luminosity

        def _at(name):
            return extract_line_luminosity(lines.all_waves, atten_lums, KEY_LINES[name])

        return EmissionLines(
            lya=_at("lya"),
            civ_1549=_at("civ_1549"),
            oii=_at("oii"),
            hbeta=_at("hbeta"),
            oiii_4959=_at("oiii_4959"),
            oiii_5007=_at("oiii_5007"),
            nii_6548=_at("nii_6548"),
            halpha=_at("halpha"),
            nii_6584=_at("nii_6584"),
            sii_6717=_at("sii_6717"),
            sii_6731=_at("sii_6731"),
            all_waves=lines.all_waves,
            all_lums=atten_lums,
        )

    def declared_parameters(self):
        """Free-parameter declarations for this SED chain.

        Returns
        -------
        list of :class:`tengri.protocols.ParamDeclaration`
            One entry per free parameter, lifted from ``self.spec``.

        Notes
        -----
        Satisfies :class:`tengri.protocols.SubModel`.
        """
        from tengri.protocols.component import ParamDeclaration

        spec = self.spec
        decls: list[ParamDeclaration] = []
        for pname in spec.free_params:
            prior = spec.get_distribution(pname)
            decls.append(ParamDeclaration(name=pname, prior=prior, description="", units=""))
        return decls

    def run(self, state, params, *, ssp_data=None, template_data=None):
        """Run the SED forward chain. Pure JAX.

        SED is the head of the per-population orchestration; in the
        tracer-bullet single-population path, ``state`` is an empty
        :class:`tengri.protocols.ForwardState` with just the wavelength
        grid. The method delegates to :meth:`predict_state` for the
        actual physics.

        Parameters
        ----------
        state: ForwardState
            Incoming state (empty for SED as the head of the chain).
        params: Mapping
            Free parameter values.

        Returns
        -------
        ForwardState
            State with SED contributions populated.

        Notes
        -----
        Satisfies :class:`tengri.protocols.SubModel`. Threading non-empty
        upstream state is reserved for a future ``ResolvedSEDModel`` mode
        that needs SED to read spatial keys; today the contract is
        "incoming state ignored, output state freshly built."

        ``ssp_data``/``template_data`` are the JIT-threading channel, forwarded
        to :meth:`predict_state`; both keyword-only and both defaulting to
        ``None``, so the ``SubModel`` call shape ``run(state, params)`` is
        unchanged for every existing caller (#1753).
        """
        return self.predict_state(params, ssp_data=ssp_data, template_data=template_data)

    def _full_state_chain(self):
        """The component chain with every publication shortcut disabled.

        Derived from the observables chain by
        :func:`~tengri.forward.orchestrator.materialized_chain`, and memoized
        against that chain's *identity* so any rebuild (``enable_fast_nebular``
        swaps the whole list) invalidates it automatically rather than leaving
        a stale copy behind.

        Returns
        -------
        list of SEDComponent
            The chain :meth:`predict_state` runs unless the caller declares it
            reads only the projected observables.
        """
        from tengri.forward.orchestrator import materialized_chain

        chain = getattr(self, "_cached_component_chain", None)
        if chain is None:
            chain = self._cached_component_chain = self._build_component_chain()
        cached = getattr(self, "_cached_full_state_chain", None)
        if cached is not None and cached[0] is chain:
            return cached[1]
        full = list(materialized_chain(chain))
        self._cached_full_state_chain = (chain, full)
        return full

    def predict_state(
        self,
        params,
        fixed_values=None,
        ssp_data=None,
        template_data=None,
        *,
        observables_only=False,
    ):
        """Forward pass via the SEDComponent orchestrator.

        Builds a component chain from this model's structural settings
        (``self.spec`` + ``self.ssp_data`` + dust / nebular / AGN / radio
        / X-ray / IGM flags) and threads ``params`` through
        :func:`tengri.forward.run_components`. Returns the final
        :class:`tengri.protocols.ForwardState`, **not** a legacy
        :class:`Prediction`, callers wanting the legacy shape should
        keep using :meth:`predict_photometry`/:meth:`predict_spectrum`
        until the full integration adapter ships.

        **Not a public surface, despite what this docstring used to say**
        (#1736). The prediction contract, ``docs/dev/NAMING_CONTRACT.md``
        §4b, binding on code, docs, notebooks and examples, names three:
        :meth:`predict`, :meth:`predict_photometry` and
        :meth:`predict_properties`. ``predict_state`` is not among them.
        It is the internal bridge from this model's configuration surface
        to the orchestrator, called by :meth:`predict_observables_jit` and
        by ``ForwardModel.predict_observables``; its return type and the
        keys on it carry no stability guarantee and may change without
        notice.

        It is **not** on the removal path either, it has production
        callers and is classified ``UNSANCTIONED`` rather than retired
        (``tests/contract/test_public_api_surface.py``). Do not describe
        it with the d-word: ``test_predict_surface_classification.py``
        derives that label by searching this docstring for the substring,
        so the wording here *is* the classification.

        **If you reached for this to get per-component SEDs, use**
        :attr:`pred.sed.components
        <tengri.forward.prediction.SEDProperties.components>` **instead**::

            comp = model.predict(params).sed.components
            comp["sed_intrinsic"]  # stellar, pre-dust  [erg/s/Hz]
            comp["sed_dust_ir"]  # and sed_nebular, sed_agn, sed_total, ...

        That is the supported decomposition, it reuses the prediction's
        cached forward state (no second forward pass), and it returns a
        plain dict of named arrays rather than a pipeline object. Reaching
        through ``predict_state`` instead hands you a
        :class:`~tengri.protocols.ForwardState`, whose ``derived`` dict
        CLAUDE.md explicitly documents as internal, so the convenient
        path out of this method leads directly into an object the
        conventions tell readers not to touch.

        Parameters
        ----------
        params: Mapping
            Free parameters keyed by canonical name (``sfh_*``,
            ``met_*``, ``dust_*``, ``agn_*``, ``radio_*``, ``xray_*``,
            ``igm_*``, ``redshift``).
        fixed_values: Mapping | None, optional
            Fixed parameter values. When provided, overrides
            ``self.spec.get_fixed_values()``. Used by :meth:`predict_observables_jit`
            to thread per-galaxy fixed values as JIT runtime inputs.
        ssp_data: Any | None, optional
            SSP grid. When provided, passed to components that need it as a
            JIT runtime input instead of using closure capture.
            Defaults to ``None``, which causes components to use their
            internal ``self.ssp_data``.
        template_data: Any | None, optional
            Nebular backend grids and weights. When provided, passed to
            components as JIT runtime inputs instead of closure capture.
            Defaults to ``None``, which causes components to use their
            internal template data.
        observables_only: bool, keyword-only, optional
            Declare that the caller reads only the *projected observables*,             photometry
            and spectra off the LUT, and never the SED arrays or
            ``derived`` publications. Components may then take publication
            shortcuts: the per-Q_H nebular grid zeroes ``sed_nebular`` because
            skipping the Cue forward is the whole saving (#1596).

            Default ``False``, so the returned state is complete. Only
            :meth:`predict_observables_jit` sets it, and that is the point:
            correctness is what you get by default and the optimization is the
            thing that has to be asked for. Setting it while reading the SED
            costs the entire nebular continuum, measured at 97 % of the peak
            on a dust-free Cue model, in float64 and silently (#1673).

        Returns
        -------
        ForwardState
            Threaded state after the chain runs. ``sed_intrinsic`` is
            the rest-frame total SED in erg/s/Hz; ``sed_observed`` is
            populated when an IGM component is present; ``derived``
            carries every cross-component publication (``L_ir``,
            ``L_agn_bol``, ``log_mstar``, ``lnu_age``, etc.).

        Notes
        -----
        **JIT-compatible**: yes, :func:`run_components` and every
        adapter's ``apply`` are pure JAX.

        ``self.spec.mean_sfh_type`` is a list (e.g. ``["tsnorm"]``,
        ``["dpl", "field"]``); the first entry is the mean SFH model,
        and ``"field"`` anywhere in the list enables the GP-field
        branch. Anything else (``burst``, etc.) is currently unmapped
        and will raise downstream.
        """
        from tengri.forward import build_components, run_components
        from tengri.protocols.component import ForwardState

        # 2026-05-20: cache the built chain on the model. Chain
        # construction runs each component's ``precompute()``, which for the
        # stellar component with ``wave_precomp=True`` calls
        # ``preintegrate_grid``, a numpy-level routine with Python ``float()``
        # calls that can't be re-traced under ``jax.jit``. Building the chain
        # once at first call (or earlier) makes subsequent ``predict_state``
        # invocations pure: they just thread ``params`` through the cached
        # chain via ``run_components``. The chain depends only on structural
        # config (spec, ssp_data, filters, approx), all of which are immutable
        # after ``__init__``.
        # Validate param keys at the orchestrator entry, silent drops
        # of typo'd or stale override keys produce plausible-looking but
        # wrong physics. Covers the dict-merge code path that
        # ``predict_observables_jit`` already guards at line ~4242
        # (#314). Skip in JIT-runtime threading mode where the JIT
        # entry point has already validated the caller's dict.
        if fixed_values is None:
            check_unknown_params(params, self._param_map)
            # ...and the converse: a free param with no value would survive
            # the {**fixed_values, **params} merge below and surface deep in
            # a component as a bare KeyError.
            check_missing_free_params(params, self.spec, self._param_map)

        cached = getattr(self, "_cached_component_chain", None)
        if cached is None:
            cached = self._build_component_chain()
            self._cached_component_chain = cached
        # The observables kernel reads only what the LUT projects, so it may
        # run the chain with its publication shortcuts intact. Every other
        # caller gets a complete state: this method's contract is the full
        # ForwardState, and a component that silently withholds one of its
        # published keys breaks it (#1673).
        chain = cached if observables_only else self._full_state_chain()
        # Initialize the chain on the panchromatic-extended grid when
        # radio/xray is configured. RadioSEDComponent / XRaySEDComponent
        # populate ``state.derived["sed_radio"]`` / ``["sed_xray"]``
        # over the full ``state.wave`` range; downstream consumers
        # (``predict_rest_sed.wavelength`` for the panchromatic SED,
        # FIR–radio q ratio, X-ray luminosity diagnostics) need pixels
        # below ~10 Å and above ~1e7 Å. Without the extension the
        # chain runs on the SSP grid only (typically 91–100000 Å) and
        # the multiwavelength contributions are confined to that range.
        wave = self._rest_wavelength
        state0 = ForwardState(wave=wave, sed_observed=jnp.ones_like(wave))
        del build_components  # silence unused-import warning; used in helper

        # Inject Fixed values from spec for parameters absent from
        # ``params``. Matches the legacy ``get_internal_params``
        # convention so callers using ``predict_rest_sed`` and
        # ``predict_state`` can pass the same params dict.
        #
        # ``fixed_values`` is an optional override. When provided
        # (typically from ``predict_observables_jit`` threading it as a
        # JIT runtime input), use it instead of ``self.spec.get_fixed_values()``.
        # That decouples per-galaxy fixed values from the closure so two
        # SEDModels with the same structure but different fixed values
        # share one compiled function.
        if fixed_values is None:
            fixed_values = self.spec.get_fixed_values()
        full_params = {**fixed_values, **params}

        # Thread ssp_data and template_data (nebular grids) as JIT inputs.
        # A None default makes components fall back to their
        # closure-captured copies (self.ssp_data / internal templates).
        return run_components(
            chain, state0, full_params, ssp_data=ssp_data, template_data=template_data
        )

    def predict_observables(self, params, *, ssp_data=None, template_data=None):
        """Project the orchestrator state into every configured observable.

        Single bit-exact entry point: runs the SEDComponent chain and
        delegates to :meth:`Observation.predict` for projection. Returns
        an :class:`Observables` NamedTuple with one field per configured
        observation sub-block (``phot_fnu``, ``phot_rest_fnu``, ``spec_fnu``).

        .. warning::

           This is the **exact** wave-grid path: it bypasses the WavePrecomp
           LUT even when the model was built with ``approx=WavePrecomp(...)``.
           If you only need photometry, :meth:`predict_photometry` returns the
           same ``phot_fnu`` through the LUT at roughly **16.5x** the speed.
           Reach for this one when you need several channels from one pass, or
           when you specifically want the exact path.

        Parameters
        ----------
        params: Mapping
            Free-parameter dict.

        Returns
        -------
        Observables
            NamedTuple with fields keyed by configured sub-blocks:
            ``phot_fnu`` [erg/s/cm²/Hz] shape ``(n_filters,)``,
            ``phot_rest_fnu`` [erg/s/cm²/Hz] shape ``(n_filters,)``,
            ``spec_fnu`` [erg/s/cm²/Hz] shape ``(n_pixels,)``.

        Notes
        -----
        **JIT-compatible**: yes. Not self-JIT'd, wrap with
        :func:`jax.jit` for hot loops, or call
        :meth:`predict_observables_jit` for the pre-cached version.

        Synthesized per-model at :meth:`__init__` from observation
        contents; missing channels raise ``AttributeError`` on access.
        """
        if self.observation is None:
            raise ValueError(
                "predict_observables requires an Observation. Build the "
                "model with ``observation=`` set."
            )

        # Eager (non-JIT) forward + projection. Runs the SAME ``_impl`` closure
        # that :meth:`predict_observables_jit` wraps in ``jax.jit``, one
        # implementation, so the two are bit-identical by construction (the old
        # dual implementation was how the spectrum LUT silently diverged). Use
        # this for one-off / interactive evaluation where the ~7–12 s JIT
        # trace+compile isn't worth amortizing; use :meth:`predict_observables_jit`
        # (what the Fitter calls) for repeated/inference evaluation, where the
        # fused kernel is structurally + persistently cached. Both honor the
        # build-time ``approx=`` (exact / WavePrecomp / SpectrumPrecomp / joint).
        from tengri.inference._model_cache import _default_owner

        self._get_or_build_predict_observables_jit()  # ensure cache is populated
        cache = _default_owner.get_structural_kernel(self.compile_signature())
        impl = cache["predict_observables_impl"]
        return impl(
            params,
            self.spec.get_fixed_values(),
            *self._resolve_threaded_data(ssp_data, template_data),
        )

    def predict_observables_jit(self, params, *, ssp_data=None, template_data=None):
        """Self-JIT'd, structurally-cached version of :meth:`predict_observables`.

        Bit-exact with :meth:`predict_observables` (same orchestrator
        chain, same :meth:`Observation.predict` projection). The compiled
        function is cached on :meth:`compile_signature`, so two SEDModel
        instances with identical structure (same physics, same filter
        set, same observation shape) share one compile across galaxies.

        Parameters
        ----------
        params: Mapping
            Free-parameter dict.

        Returns
        -------
        Observables
            NamedTuple with fields keyed by configured sub-blocks.

        Notes
        -----
        **JIT-compatible**: yes, this method IS the JIT entry point.

        2026-05-20: ``self.spec.get_fixed_values()`` is now
        passed as a JIT runtime input rather than closure-captured. Two
        SEDModels with the same structural config but different per-galaxy
        fixed values (e.g., ``redshift=Fixed(0.1)`` vs ``redshift=Fixed(0.5)``)
        share a :meth:`compile_signature` and reuse the same compiled
        function, per-galaxy values flow through at call time.

        For per-galaxy fixed redshifts, build with
        ``approx=WavePrecomp(z_min=catalog_z_min, z_max=catalog_z_max)`` so
        the ztable covers the catalog range; runtime ``params['redshift']``
        is then a fast interpolation lookup.

        See Also
        --------
        predict_observables: un-JIT'd version (debug / one-shot).
        compile_signature: structural fingerprint controlling cache reuse.

        2026-05-20: ``self.ssp_data`` is now passed as a JIT
        runtime input rather than closure-captured. The SSP grid becomes a
        ``Parameter`` op in the compiled HLO instead of a ``Constant`` op,
        reducing compile size and time.

        2026-05-21: nebular backend grids and weights are now
        passed as JIT runtime inputs. Backend grids become ``Parameter`` ops
        instead of ``Constant`` ops, reducing compile size for Cue and CloudyGrid.
        """
        # Validate param keys before entering JIT, silent drops of unknown
        # override keys produce plausible-looking but wrong physics (issue #314).
        check_unknown_params(params, self._param_map)
        # ...and free params with no value, which would otherwise surface as
        # a bare KeyError deep inside a component.
        check_missing_free_params(params, self.spec, self._param_map)
        return self._get_or_build_predict_observables_jit()(
            params,
            self.spec.get_fixed_values(),
            *self._resolve_threaded_data(ssp_data, template_data),
        )

    def _get_or_build_predict_observables_jit(self):
        """Return (and cache) the JIT'd predict_observables closure."""
        from tengri.inference._model_cache import _default_owner

        cache = _default_owner.get_structural_kernel(self.compile_signature())
        fn = cache.get("predict_observables_jit")
        if fn is not None:
            return fn

        # Capture the model's per-instance LSF + wave_obs into the closure.
        # These are part of compile_signature when spectroscopy is configured,
        # so caching is safe across instances with identical structure.
        # ``fixed_values`` is no longer closure-captured, it
        # comes through as a JIT runtime input from ``predict_observables_jit``.
        observation = self.observation
        sigma_v_getter = self._get_sigma_v_kms
        lsf_resolution = self._lsf_resolution
        sigma_lib_kms = self._sigma_lib_kms
        lsf_n_bins = self._lsf_n_bins
        wave_obs = (
            getattr(self, "_wave_obs", None)
            if observation is None or not observation.can_do_spectroscopy
            else (
                self._wave_obs if hasattr(self, "_wave_obs") else observation.spectroscopy.wave_obs
            )
        )
        observables_type = self._Observables
        # Route the photometry channel through the LUT projection
        # when the model was built with ``approx=WavePrecomp(...)``. The
        # routing decision is closure-captured per-model and baked into
        # ``compile_signature`` (via the resolved ``approx`` config tuple),
        # so structurally-equal models share a compile across the two
        # routings without colliding. Spectrum stays exact, no spectrum
        # LUT yet.
        use_lut = bool(self._approx.get("wave_precomp")) and not observation.can_do_spectroscopy
        # Part B: spectrum LUT inside the fused JIT kernel. ``predict_state``
        # (called within ``_impl``) publishes ``spec_eff_waves`` + the per-pixel
        # ``*_spec_lnu_precomp`` family via the cached chain, so the projector is
        # fully JAX-traceable, no eager fallback needed. ``phot_lut`` lets a
        # JOINT model also project photometry in the same kernel (Part A).
        spec_lut = bool(self._approx.get("spectrum_precomp")) and observation.can_do_spectroscopy
        phot_lut = bool(self._approx.get("wave_precomp"))

        # Warm the component-chain cache OUTSIDE the JIT trace. The chain
        # build runs each component's ``precompute()``, which for the
        # stellar component with ``wave_precomp=True`` calls
        # ``preintegrate_grid``, a numpy-level routine with Python
        # ``float()`` calls that can't be traced. After this warmup,
        # ``predict_state`` inside the JIT reuses the cached chain.
        if getattr(self, "_cached_component_chain", None) is None:
            self._cached_component_chain = self._build_component_chain()

        def _impl(params, fixed_values, ssp_data, template_data):
            # The one caller that may take the publication shortcuts: this
            # kernel returns projected observables and never exposes the state,
            # so a zeroed sed_nebular is invisible to it by construction.
            state = self.predict_state(
                params,
                fixed_values=fixed_values,
                ssp_data=ssp_data,
                template_data=template_data,
                observables_only=True,
            )
            full = {**fixed_values, **params}
            # Part B: spectrum LUT (and, for a joint model, photometry LUT)
            # projected inside the fused kernel. Each projector sums its own
            # ``*_spec_lnu_precomp`` / ``*_phot_lnu_precomp`` family; collect
            # dicts and build the per-model Observables once so phot_fnu and
            # spec_fnu coexist. (Velocity dispersion / LSF are not applied on the
            # per-pixel continuum LUT, that is SpectrumPrecomp's documented
            # low-to-medium-R domain.)
            if spec_lut:
                out: dict = {}
                out.update(
                    observation.predict_spectrum_via_precomp(state, full, observables_type=None)
                )
                if observation.can_do_photometry and phot_lut:
                    out.update(observation.predict_via_precomp(state, full, observables_type=None))
                avail = {k: v for k, v in out.items() if k in observables_type._fields}
                return observables_type(**avail)
            if observation.can_do_spectroscopy:
                return observation.predict(
                    state,
                    full,
                    wave_obs=wave_obs,
                    sigma_v_kms=sigma_v_getter(full),
                    lsf_resolution=lsf_resolution,
                    lsf_sigma_lib_kms=sigma_lib_kms,
                    lsf_n_bins=lsf_n_bins,
                    observables_type=observables_type,
                )
            if use_lut:
                return observation.predict_via_precomp(
                    state, full, observables_type=observables_type
                )
            return observation.predict(state, full, observables_type=observables_type)

        jit_fn = jax.jit(_impl)
        cache["predict_observables_jit"] = jit_fn
        # Cache the un-jitted closure too, so :meth:`predict_observables` can run
        # the IDENTICAL forward+projection logic eagerly (no XLA compile) without
        # a second implementation to keep in sync. Same code → bit-identical.
        cache["predict_observables_impl"] = _impl
        return jit_fn

    def _resolve_threaded_data(self, ssp_data, template_data):
        """Resolve the JIT-threading channel: caller's arrays, else this model's own.

        The one place the override policy lives, so every public surface that
        accepts ``ssp_data=``/``template_data=`` resolves them identically.

        Threading matters only across a JIT boundary the *caller* owns. Inside
        :meth:`predict_observables_jit` the grids already ride in as arguments to a
        structurally-cached ``jax.jit``, so tengri's own programs never bake them.
        But a caller who writes ``jax.jit(model.predict_photometry)`` inlines that
        inner jit into their trace, and ``self.ssp_data``, read here as a concrete
        array, becomes a ``Constant`` of *their* computation. Passing the grid in
        makes it an invar of their trace instead. Measured on a real SSP: the
        persistent-cache entry goes 0.23 MB → 58.82 MB when it bakes (#1753, #1507).

        Parameters
        ----------
        ssp_data: Any | None
            Caller-supplied SSP grid, or ``None`` to use ``self.ssp_data``.
        template_data: Any | None
            Caller-supplied template arrays, or ``None`` to use
            :meth:`_template_data_for_jit`.

        Returns
        -------
        tuple
            ``(ssp_data, template_data)`` ready to hand to the impl closure.
        """
        return (
            self.ssp_data if ssp_data is None else ssp_data,
            self._template_data_for_jit() if template_data is None else template_data,
        )

    def _template_data_for_jit(self):
        """Collect template grids/weights for JIT threading (nebular + dust IR + AGN).

        Walks the cached component chain (built by predict_state warmup)
        to collect template arrays that should be threaded as JIT runtime
        inputs instead of closure-captured, so they appear as JAX
        ``Parameter`` ops rather than baked-in HLO ``Constant`` ops.

        **Nebular templates**: duck-typed on backend ``.grid``
        / ``.weights`` attributes (Cue, CloudyGrid, etc.).

        **Dust IR**: emission components (Astrodust, PAHspec, Dale, …) self-load
        their HDF5 grids in ``EmissionComponent.load``/``predict``; only the
        build-time energy-balance LUT and per-filter band response are threaded
        here (added below).

        **AGN templates**: extracted from the
        AGNSEDComponent's cached state (SKIRTOR templates).

        Returns
        -------
        dict[str, Any] | None
            Nested dict with namespace keys (``"nebular"``, ``"dust_ir"``,
            ``"agn"``) carrying the threaded template data for that
            subsystem. Returns ``None`` if no components need threading.
        """
        # Build the chain if it is not cached yet, rather than bailing out.
        #
        # Returning None here made threading work exactly ONCE per process. The
        # chain is only pre-built in ``__init__`` under spectrum_precomp /
        # wave_precomp; otherwise the first ``predict_state`` warmup populates
        # it. On a SECOND model with the same compile signature the structural
        # kernel cache hits, that warmup never runs, and this returned None, so
        # every template fell back to its in-block load and baked. Measured on
        # ``torus='skirtor'``: 0.05 MB on build 1, **29.94 MB on builds 2+**.
        #
        # Every caller of this method assembles arguments *before* tracing (see
        # ``predict_observables`` / ``predict_observables_jit`` / the fitter), so
        # building the chain here runs in the same eager context that build 1
        # used. Matching the lazy pattern already used by ``_qh_at`` and friends.
        cached = getattr(self, "_cached_component_chain", None)
        if cached is None:
            cached = self._cached_component_chain = self._build_component_chain()

        from tengri.components.agn.blocks._protocol import collect_block_templates
        from tengri.components.agn.component import AGNSEDComponent
        from tengri.components.nebular.component import NebularSEDComponent

        result = {}

        # ── Nebular backend threading ──
        for component in cached:
            if not isinstance(component, NebularSEDComponent):
                continue
            backend = getattr(component, "backend", None)
            if backend is None:
                continue
            # Duck-type: prefer .weights (NN-backed like Cue),
            # then .grid (interpolation-table like CloudyGrid/CB19/MAPPINGS/AGN NLR).
            # Falls back to skipping for backends with no separate template data
            # (BakedIn, Shock).
            for attr in ("weights", "grid"):
                template = getattr(backend, attr, None)
                if template is not None:
                    result["nebular"] = template
                    break
            break

        # Dust IR emission components (Astrodust, PAHspec, Dale, …) self-load their
        # HDF5 grids in ``EmissionComponent.load``/``predict``, no adapter-state
        # threading is needed here. The build-time energy-balance LUT and
        # per-filter band response below are the only dust-IR data threaded.

        # ── AGN template threading ──
        for component in cached:
            if not isinstance(component, AGNSEDComponent):
                continue
            agn_templates = {}
            skirtor = component._state.skirtor_templates if component._state is not None else None
            # The exact path does not run AGN precompute, so ``_state`` carries
            # no template. Load the SKIRTOR grid arrays here so the data always
            # threads through jit (small compile) rather than baking into the
            # trace as a constant (#1198). Guarded to the monolithic SKIRTOR
            # model; composable blocks are handled by the recipe walk below.
            if skirtor is None and getattr(component.config, "model", None) == "skirtor":
                try:
                    from tengri.components.agn.skirtor import _load_skirtor_default_grid

                    skirtor = _load_skirtor_default_grid()
                except Exception:
                    skirtor = None
            if skirtor is not None:
                agn_templates["skirtor"] = skirtor

            # Composable-block template libraries, keyed "<category>/<name>".
            # Driven by the *resolved recipe*, so any block that declares a
            # ``template_loader`` threads, including ones added later. This
            # replaces a gate on ``config.model == "skirtor"``, which published
            # nothing for ``composable`` (the build-grammar default) and so let
            # every torus library bake into the graph (#1383).
            blocks = component._state.block_templates if component._state is not None else None
            if blocks is None:
                blocks = collect_block_templates(component.block_recipe()) or None
            if blocks:
                agn_templates["blocks"] = blocks

            if agn_templates:
                result["agn"] = agn_templates
            break

        # ── Dust energy-balance LUT (build-time, memoized) ──
        # When the attenuation-curve shape is fixed (only tau_bc/tau_diff vary),
        # the bolometric absorbed luminosity that feeds L_ir is a smooth
        # function of (tau_bc, tau_diff); precompute it so the per-call
        # full-wavelength stellar cube is no longer needed for dust IR emission.
        eb_lut = self._energy_balance_lut(cached)
        if eb_lut is not None:
            result.setdefault("dust_ir", {})["energy_balance_lut"] = eb_lut

        band_response = self._dust_emission_band_response(cached)
        if band_response is not None:
            result.setdefault("dust_ir", {})["emission_band_response"] = band_response

        # ── Component template libraries, keyed [namespace][component name] ──
        #
        # A template-backed component that reads its library inside ``predict``
        # freezes the whole thing into the graph as ``Constant`` ops, 66.6 MB
        # for Draine & Li 2014, 39.4 for THEMIS, 3.7 for the MAPPINGS V shock
        # grid, against a bare-stellar floor of 0.05 MB (#1649, #1694).
        # Publishing the bundle here, loaded eagerly, lets
        # ``SEDModelComponent.apply`` hand it to ``predict`` as a traced
        # argument instead.
        #
        # Driven by the component's own ``accepts_threaded_templates`` flag
        # rather than a list of classes, so a template-backed component added
        # later threads the day it lands, it does not have to be named here.
        # This walk used to be gated on ``isinstance(component, EmissionComponent)``,
        # which is why the shock grid went on baking after the whole dust
        # subsystem was fixed.
        for component in cached:
            if not getattr(component, "accepts_threaded_templates", False):
                continue
            resolve = getattr(component, "templates_for_threading", None)
            if resolve is None:
                # Every component registered by this package inherits the seam
                # (pinned by tests/contract/test_component_template_threading.py),
                # so this only fires for a class registered from outside it. Say
                # what to do rather than surfacing a bare AttributeError.
                raise TypeError(
                    f"Component {component.name!r} sets accepts_threaded_templates "
                    f"but does not inherit TemplateThreading, so it has no "
                    f"templates_for_threading(). Add "
                    f"tengri.components.template_threading.TemplateThreading as a "
                    f"base class."
                )
            bundle = resolve()
            if bundle is None:
                continue
            namespace = getattr(component, "template_namespace", "") or component.name
            slot = result.setdefault(namespace, {})
            if not isinstance(slot, dict):
                # Some namespaces predate the name-keyed layout and hold a bare
                # bundle (``result["nebular"]`` is the backend's grid itself,
                # read by ``NebularSEDComponent`` as ``template_data["nebular"]``).
                # Writing a name key into one would either raise deep inside a
                # NamedTuple or silently corrupt a dict-like grid, and the damage
                # would surface as wrong physics far from here.
                raise ValueError(
                    f"Component {component.name!r} declares "
                    f"template_namespace={namespace!r}, but that namespace already "
                    f"holds a bare {type(slot).__name__} bundle rather than a "
                    f"name-keyed dict. Choose a distinct namespace (leaving "
                    f"``template_namespace`` unset uses the component name, which "
                    f"is unique by registry construction)."
                )
            slot[component.name] = bundle

        # The other additive emitters (X-ray, radio) are sums of rank-1 terms, so
        # they get a response *per term* rather than the single L_ir * R that dust's
        # one-term SED admits. Same exactness, same build-time integral.
        for emitter in ("xray", "radio"):
            term_response = self._additive_term_band_response(cached, emitter)
            if term_response is not None:
                result.setdefault(emitter, {})["term_band_response"] = term_response

        return result if result else None

    #: dust_* params that change the *emission* template only (not the
    #: attenuation curve), so they may be free without invalidating the
    #: energy-balance (tau_bc, tau_diff) LUT.
    _EB_EMISSION_PARAMS = frozenset(
        {
            "dust_T",
            "dust_beta_ir",
            "dust_alpha_dale",
            "dust_lgU",  # astrodust + draine2021_pah starlight-intensity knob
            "dust_umin",
            "dust_qpah",
            "dust_gamma_dl",
            "dust_alpha_dl14",
            "dust_alpha_mir",
            "dust_qhac",
            "dust_alpha",
            "dust_frac_agn",
            "dust_f_pah",
            "dust_epsilon_mbb",
            "dust_f_cold",
            "dust_L_agn_ir",
            "dust_T_warm",
            "dust_T_cold",
            "dust_beta_warm",
            "dust_beta_cold",
        }
    )
    #: dust attenuation params that may be free (tau axes + linear eta scaling).
    _EB_ATTEN_FREE_OK = frozenset({"dust_tau_bc", "dust_tau_diff", "dust_eta_balance"})

    def _energy_balance_lut(self, chain):
        """Build (and memoize) the two-component energy-balance LUT, or ``None``.

        Returns ``None`` unless the model uses ``approx=WavePrecomp()`` with a
        two-component :class:`DustSEDComponent` that re-emits IR, the SSP needs
        no per-call alpha interpolation, and every *free* ``dust_*`` parameter
        is either an optical depth / eta scaling or an emission-shape knob, so
        the attenuation *curve* is fixed and the absorbed luminosity is a smooth
        function of ``(tau_bc, tau_diff)`` alone.
        """
        cached = getattr(self, "_energy_balance_lut_cache", "unset")
        if cached != "unset":
            return cached

        from tengri.components.dust.attenuation import resolve_bc_diff_law_params
        from tengri.components.dust.energy_balance_precompute import (
            build_energy_balance_lut,
        )
        from tengri.components.dust.two_component import DustSEDComponent

        lut = None
        dust = next((c for c in chain if isinstance(c, DustSEDComponent)), None)
        free = set(self.spec.free_params)
        unsafe_free = {
            p
            for p in free
            if p.startswith("dust_")
            and p not in self._EB_ATTEN_FREE_OK
            and p not in self._EB_EMISSION_PARAMS
        }
        # Detect dust emission: either old path (DustSEDComponent.emission_model)
        # or new path (separate dust emission component in the pipeline).
        # After the switchover, dust_emission_model is set from the spec even
        # when using separate components, so we check that or the old emission_model path.
        has_dust_emission = (
            dust is not None and getattr(dust.config, "emission_model", None) is not None
        ) or self._dust_emission_model is not None
        # Dust emission is not the only consumer of L_ir. Radio reads it too, the
        # FIR-radio correlation sets the SF synchrotron amplitude, so a radio model
        # with no dust *emission* block still needs the LUT, and without it the
        # full-grid energy-balance integral (and the stellar cube behind it) stays
        # alive: 33.3M FLOPs against 368k with it. Ask the declared cross-component
        # contract (ADR-0009) who consumes L_ir rather than hardcoding a list, so a
        # future L_ir consumer inherits the LUT instead of silently forfeiting it.
        needs_l_ir = has_dust_emission or _chain_consumes(chain, "L_ir")
        if (
            needs_l_ir
            and dust is not None
            and self._approx.get("wave_precomp")
            and not bool(getattr(self.spec, "alpha_fe_evolving", False))
            and not unsafe_free
            and self.ssp_data is not None
        ):
            fixed = self.spec.get_fixed_values()
            # Same narrowing as the component's own apply() (#1833), read off
            # the component that is actually in the chain, so the LUT cannot
            # bake a different curve from the one the direct path evaluates.
            bc_params, diff_params = resolve_bc_diff_law_params(
                fixed,
                dict(dust.config.bc_law_overrides),
                dict(dust.config.diff_law_overrides),
                dust.config.live_shape_params,
            )
            ssp_ages_yr = (10.0**self.ssp_data.ssp_lg_age_gyr) * 1e9

            def _grid(name):
                if name in free:
                    dist = self.spec.get_distribution(name)
                    lo, hi = float(dist.bounds[0]), float(dist.bounds[1])
                    return jnp.linspace(lo, hi, 24)
                return jnp.asarray([float(fixed.get(name, 0.0))])

            lut = build_energy_balance_lut(
                jnp.asarray(self.ssp_data.ssp_flux),
                jnp.asarray(self.ssp_data.ssp_wave),
                jnp.asarray(ssp_ages_yr),
                law_bc=dust.config.law_bc,
                law_diff=dust.config.law_diff,
                f_obscuration=float(fixed.get("dust_f_obscuration", 0.0)),
                t_birth_yr=dust.config.t_birth_yr,
                transition_width_dex=dust.config.transition_width_dex,
                bc_params={k: float(v) for k, v in bc_params.items()},
                diff_params={k: float(v) for k, v in diff_params.items()},
                lyman_cutoff_aa=dust.config.lyman_cutoff_aa,
                eb_include_lyc=dust.config.eb_include_lyc,
                tau_bc_grid=_grid("dust_tau_bc"),
                tau_diff_grid=_grid("dust_tau_diff"),
            )

        self._energy_balance_lut_cache = lut
        return lut

    def _dust_emission_band_response(self, chain):
        """Build-time filter-integrated dust-IR response per unit ``L_ir``.

        CIGALE's ``dl2014``/``dale2014`` normalize the IR template to unit
        luminosity and scale by ``dust.luminosity`` (emission = ``L_dust ×
        template``); the band fluxes are therefore ``L_ir × R`` with ``R`` the
        template's per-filter integral. When the emission *shape* (``dust_T``,
        ``dust_beta_ir``, ``dust_epsilon_mbb``) and ``redshift`` are fixed, ``R``
        is a build-time constant, returned here (shape ``(n_filter,)``) so
        :meth:`DustSEDComponent.apply` replaces the per-call dense filter
        integral (#622) with the exact ``L_ir × R``. Returns ``None`` otherwise
        (free shape/z → keep the per-call integral, or ``fast_dust_emission``).
        """
        cached = getattr(self, "_dust_band_response_cache", "unset")
        if cached != "unset":
            return cached

        response = None
        emitter = next(
            (c for c in chain if getattr(c, "name", "") == "dust_emission"),
            None,
        )
        stellar = next((c for c in chain if c.name == "stellar"), None)
        st = getattr(stellar, "_state", None)
        fw_pad = getattr(st, "phot_fw_padded", None)
        ft_pad = getattr(st, "phot_ft_padded", None)

        # ALLOWLIST, not a denylist. R is a build-time constant only if the emission
        # *shape* is fixed. Gating on "no free param from a known shape-param set"
        # fails DANGEROUS when a param is missing from that set (dust_log_ssfr was);
        # gating on "no free dust_* param outside the known attenuation knobs" fails
        # SAFE, an unrecognized free parameter simply disables the optimization.
        free = set(self.spec.free_params)
        free_dust = {p for p in free if p.startswith("dust_")}
        shape_free = bool(free_dust - self._EB_ATTEN_FREE_OK) or ("redshift" in free)

        if (
            emitter is not None
            and self._approx.get("wave_precomp")
            and not shape_free
            and fw_pad is not None
            and ft_pad is not None
            and hasattr(emitter, "predict")
            and hasattr(emitter, "slice_params")
        ):
            from tengri.observation.photometry import lnu_filter_integral_batch

            fixed = dict(self.spec.get_fixed_values())
            wave = self._rest_wavelength
            # Direct lookup, not .get(..., 0.0): ``shape_free`` above already
            # required ``"redshift" not in free``, and redshift is always in
            # exactly one of free/fixed (it defaults to Fixed when omitted), so
            # a fixed value is guaranteed here. A 0.0 fallback would be
            # unreachable, and if the gate above is ever weakened, it would
            # silently build R at z=0 instead of failing (#1432).
            z = jnp.asarray(fixed["redshift"])
            # Slice with the component's OWN rule, the same one apply() uses. A
            # precompute that slices differently silently builds R from default
            # template parameters and returns confidently wrong IR photometry.
            p = emitter.slice_params({k: jnp.asarray(v) for k, v in fixed.items()})

            # HOMOGENEITY CHECK. The band response is exact only because an additive
            # emitter is linear (degree-1 homogeneous) in its luminosity:
            #
            #     sed(L) = L * S_unit(lambda)   =>   int sed(L) R_f = L * int S_unit R_f
            #
            # Not every IR model obeys this. BOSA parameterizes its template by
            # (L_TIR, sSFR), so its *shape* is a function of L_ir, probing it at
            # L_ir = 1 erg/s samples a template ~44 dex from anything physical and
            # builds a response that is wrong by ~13% in W4. Luminosity-dependent
            # shapes (L-T relations) are common in IR SED models, so verify the
            # property rather than maintaining a list of which models have it:
            # probe at two luminosities and require the SED to scale.
            lo, _ = emitter.predict(p, jnp.zeros_like(wave), wave, L_ir=1.0)
            hi, _ = emitter.predict(p, jnp.zeros_like(wave), wave, L_ir=_L_IR_PROBE)
            if not bool(jnp.allclose(hi, _L_IR_PROBE * lo, rtol=1e-10)):
                self._dust_band_response_cache = None
                return None

            response = lnu_filter_integral_batch(lo, wave, fw_pad, ft_pad, z)

        self._dust_band_response_cache = response
        return response

    def _additive_term_band_response(self, chain, name):
        r"""Build-time per-filter response of each rank-1 term of an additive emitter.

        Generalizes :meth:`_dust_emission_band_response` from one term to many. An
        additive emitter is a *sum of rank-1 terms*, each a scalar amplitude times a
        spectral shape that depends only on the emitter's own (fixed) shape parameters:

        .. math::

            L_\nu(\lambda) = \sum_k A_k(\text{inputs}) \, S_k(\lambda; \text{shape})

        The filter integral is linear, so each term's band flux factorizes and

        .. math::

            \int \Big[\sum_k A_k S_k(\lambda)\Big] R_f(\lambda)\, d\lambda
                = \sum_k A_k \underbrace{\int S_k(\lambda) R_f(\lambda)\, d\lambda}_{R_{kf}}

        with :math:`R_{kf}` a build-time constant. This is **exact**, not an
        approximation: the true filter transmission is still integrated on the full
        wavelength grid with the identical quadrature the dense path uses
        (:func:`~tengri.observation.photometry.lnu_filter_integral_batch`), just once
        instead of on every call. Contrast ``fast_emission``, which samples the emitter
        at one effective wavelength, and the stellar WavePrecomp Taylor projection (#617).

        Dust IR happens to be the :math:`k=1` case (``sed = L_ir * S_unit``). X-ray is
        :math:`k=4` (HMXB, LMXB, hot gas, corona) and radio :math:`k=3` (SF synchrotron,
        free-free, AGN jet). Their *summed* SEDs are **not** rank-1, HMXB and LMXB carry
        different photon indices, so the HMXB/LMXB mix shifts with SFR and stellar mass,
        which is why the terms must be integrated separately rather than as a total.

        At runtime the amplitudes come back from a single-wavelength evaluation,
        :math:`A_k = \text{term}_k(\lambda^{\rm ref}_k) / S_k(\lambda^{\rm ref}_k)`. That
        needs no knowledge of *which* input carries the amplitude, nor that it enters
        linearly, :math:`\alpha_{\rm ox}` is famously nonlinear in :math:`L_{2500}`, and
        it does not matter, because it is still a scalar.

        Parameters
        ----------
        chain: sequence
            The cached component chain.
        name: str
            Component name to look for (``"xray"``, ``"radio"``).

        Returns
        -------
        dict or None
            ``{"R": (n_terms, n_filters), "lam_ref": (n_terms,), "S_ref": (n_terms,)}``
            [erg/s/Hz per unit amplitude, Å, erg/s/Hz], or ``None`` when the fast path
            is refused, in which case the caller keeps the exact per-call dense filter
            integral. Term order is the emitter's ``emission_terms`` dict order.

        Notes
        -----
        **Gate.** The response is a constant only while every one of the emitter's own
        parameters and ``redshift`` are fixed: a free *shape* parameter (a photon index,
        a spectral index, a turnover frequency) would move :math:`S_k` under the LUT, and
        a free redshift would move :math:`R_f`. Gating on *all* the emitter's parameters
        rather than on a known set of shape parameters fails **safe**, an unrecognized
        free parameter simply disables the optimization. Denylisting known shape knobs
        would fail *dangerous* the day a new one is added and forgotten (#1107 shipped
        exactly that hole with ``dust_log_ssfr``).

        **Rank-1 probe.** Being a sum of rank-1 terms is a *property*, not a promise, so
        it is verified rather than declared: each term is built twice from deliberately
        distant input draws (``EMITTER_PROBE_INPUTS``) and must come back proportional.
        A term whose *shape*, not just amplitude, responds to a runtime input fails,
        and the whole emitter drops to the dense path. This is the BOSA lesson from
        #1107: an emitter whose template shape tracked its luminosity sailed through a
        band response built at ``L_ir = 1`` and returned fluxes 13 % wrong, silently.

        **Zero terms.** A term that is identically zero under both probes is zero for a
        *structural* reason, a Python-level switch (``include_freefree=False``) or a
        fixed zero parameter (``radio_loudness = 0``, the default), never because a
        probe happened to zero it: every probe input is nonzero by construction. Under
        the all-parameters-fixed gate it is therefore zero at runtime too, so
        :math:`R_k = 0` is correct rather than a silent drop.

        **JIT.** Runs at build time on concrete values; the returned arrays are threaded
        into the JIT as ``template_data``.
        """
        cache_attr = f"_{name}_term_response_cache"
        cached = getattr(self, cache_attr, "unset")
        if cached != "unset":
            return cached

        response = None
        comp = next((c for c in chain if getattr(c, "name", "") == name), None)
        stellar = next((c for c in chain if getattr(c, "name", "") == "stellar"), None)
        st = getattr(stellar, "_state", None)
        fw_pad = getattr(st, "phot_fw_padded", None)
        ft_pad = getattr(st, "phot_ft_padded", None)

        free = set(self.spec.free_params)
        prefix = f"{name}_"
        gate_ok = not any(p.startswith(prefix) for p in free) and "redshift" not in free

        if (
            comp is not None
            and gate_ok
            and self._approx.get("wave_precomp")
            and fw_pad is not None
            and ft_pad is not None
            and hasattr(comp, "emission_terms")
            and hasattr(comp, "EMITTER_PROBE_INPUTS")
        ):
            from tengri.observation.photometry import lnu_filter_integral_batch

            fixed = {k: jnp.asarray(v) for k, v in self.spec.get_fixed_values().items()}
            wave = self._rest_wavelength
            # Direct lookup, not .get(..., 0.0): ``gate_ok`` above already
            # required ``"redshift" not in free``, and redshift is always in
            # exactly one of free/fixed, so a fixed value is guaranteed. See the
            # matching note on the band-response path (#1432).
            z = jnp.asarray(fixed["redshift"])

            probe_a, probe_b = comp.EMITTER_PROBE_INPUTS
            terms_a = comp.emission_terms(fixed, wave, **probe_a)
            terms_b = comp.emission_terms(fixed, wave, **probe_b)

            rows, lam_ref, s_ref = [], [], []
            for key, s_a in terms_a.items():
                s_b = terms_b[key]
                peak = int(jnp.argmax(jnp.abs(s_a)))
                s_at_ref = s_a[peak]

                if not bool(jnp.abs(s_at_ref) > 0.0):
                    # Structurally off under the all-fixed gate (see Notes). A zero
                    # response contributes nothing; S_ref = 1 keeps A = term/1 = 0
                    # finite instead of 0/0.
                    rows.append(jnp.zeros(fw_pad.shape[0], dtype=wave.dtype))
                    lam_ref.append(wave[peak])
                    s_ref.append(jnp.asarray(1.0, dtype=wave.dtype))
                    continue

                # RANK-1 CHECK: the second draw must be proportional to the first.
                scale = s_b[peak] / s_at_ref
                if not bool(jnp.allclose(s_b, scale * s_a, rtol=_RANK1_RTOL, atol=0.0)):
                    setattr(self, cache_attr, None)
                    return None

                rows.append(lnu_filter_integral_batch(s_a, wave, fw_pad, ft_pad, z))
                lam_ref.append(wave[peak])
                s_ref.append(s_at_ref)

            response = {
                "R": jnp.stack(rows),
                "lam_ref": jnp.stack(lam_ref),
                "S_ref": jnp.stack(s_ref),
            }

        setattr(self, cache_attr, response)
        return response

    #: Provenance tags that mean a caller asked for this parameter's value.
    #: ``registry_default`` and ``wildcard_fixed`` are deliberately absent:
    #: neither expresses a request, and for those an attenuation law's own
    #: published default must stand rather than the shared spec default.
    _REQUESTED_PROVENANCE = frozenset({"user_prior", "user_fixed", "user_free", "wildcard_free"})

    def _requested_law_shape_params(self, *laws: str | None) -> frozenset[str]:
        """Shape parameters of the selected attenuation law(s) a caller asked for.

        Parameters
        ----------
        *laws: str or None
            Attenuation-law registry keys in play. ``None`` entries are
            skipped. Defaults to the diffuse law alone, which is the single
            screen's only law.

        Returns
        -------
        frozenset of str
            Flat names whose provenance says somebody asked. Empty for a law
            that reads no shape parameter (the default ``calzetti`` among
            them), which keeps the build-time cached ``k(lambda)`` and the
            single screen's fast path unchanged.

        Notes
        -----
        **JIT-compatible**: no, build-time provenance lookup.

        The union is over every law in play, not just the diffuse one: the
        two-component screen evaluates ``law_bc``, ``law_diff`` and ``law_neb``,
        and a parameter read only by the birth-cloud law would otherwise be
        dropped as unrequested while the user had plainly requested it. See
        :attr:`DustAttenuationSEDComponentConfig.live_shape_params` (#1808) and
        :attr:`DustSEDComponentConfig.live_shape_params` (#1833).
        """
        from tengri.parameters.groups import _law_shape_params

        names = laws or (
            getattr(self, "_dust_law_diff", None) or getattr(self.spec, "dust_law_diff", None),
        )
        reads: set[str] = set()
        for law in names:
            if law is None:
                continue
            try:
                reads |= set(_law_shape_params(law))
            except Exception:  # pragma: no cover - law not registered
                continue
        if not reads:
            return frozenset()
        provenance = getattr(self.spec, "_group_provenance", None) or {}
        return frozenset(
            name
            for name in reads
            # ``_grid`` suffixes mark a declared free prior intersected with a
            # template grid; still a request, so match on the stem.
            if str(provenance.get(name, "registry_default")).removesuffix("_grid")
            in self._REQUESTED_PROVENANCE
        )

    def _build_component_chain(self):
        """Construct the orchestrator chain from ``self``'s settings.

        Reads ``self.spec`` and the ``_dust_*``/``_nebular_backend``/
        ``_agn_model``/``_uses_*`` attributes set in :meth:`__init__`
        and produces a list of :class:`SEDComponent` adapters in the
        canonical pipeline order.
        """
        from tengri.components.stellar.sfh.registry import apply_compositor_swap
        from tengri.forward.component_factory import build_components

        # Mean SFH: first entry of mean_sfh_type, with "field" flag if
        # the GP modulator is composed in.
        #
        # The compositor swap must be applied HERE too, not just in
        # ``resolve_sfh`` (#1074). Composing dense_basis with field renames its
        # public parameters ``sfh_db_*`` → ``sfh_dbp_*``; handing the component
        # the pre-swap name made it resolve the wrong spec, miss every
        # ``sfh_dbp_*`` the user set, and silently fall back to registry
        # defaults, so tx_frac_* moved predict_sfh but never the photometry.
        mean_types = apply_compositor_swap(list(getattr(self.spec, "mean_sfh_type", ["tsnorm"])))
        mean_model = next((m for m in mean_types if m != "field"), "tsnorm")
        field_on = "field" in mean_types

        # Nebular backend mapping. SEDModel's ``_nebular_backend`` is
        # either ``None`` (off) or a backend instance (BakedIn, Cue,
        # CloudyGrid, …); the factory takes a string + optional
        # instance.
        neb_inst = getattr(self, "_nebular_backend", None)
        if neb_inst is None:
            neb_backend_name = None
            neb_backend_instance = None
        else:
            cls_name = type(neb_inst).__name__.lower()
            if "bakedin" in cls_name:
                neb_backend_name = "baked_in"
            elif "cb19" in cls_name:
                neb_backend_name = "cb19"
            elif "cloudygrid" in cls_name:
                neb_backend_name = "cloudy_grid"
            elif "mappings" in cls_name:
                neb_backend_name = "mappings"
            elif "cue" in cls_name:
                neb_backend_name = "cue"
            elif "shock" in cls_name:
                neb_backend_name = "shock"
            else:
                neb_backend_name = "baked_in"  # fallback
            neb_backend_instance = neb_inst

        # Which attenuation-law shape parameters did somebody actually ask for?
        #
        # The single_component screen used to call its law with no arguments,
        # so dust_slope / dust_delta / dust_bump_strength were unfittable
        # (#1808). Passing the spec's values unconditionally is not the fix
        # either: the spec declares ONE shared dust_delta / dust_bump_strength,
        # both Fixed(0.0), while each law carries its paper's value in its own
        # signature (kriek_conroy bump=1.0, narayanan_z delta=-0.2), and
        # overriding those collapses three distinct published laws onto one
        # curve, measured.
        #
        # Provenance separates the two cases, and it can only be read here: the
        # component sees a plain params dict, and deciding at call time would
        # mean branching on a traced value. registry_default and wildcard_fixed
        # mean "nobody asked", so the law's own default stands.
        #
        # #1833: the two-component screen needs the same treatment and the union
        # over the three laws it evaluates. It was passing the shared spec values
        # unconditionally, so kriek_conroy lost its 2175 A bump and narayanan_z /
        # tea their delta = -0.2, the exact outcome rejected above, on the path
        # every shipped recipe builds.
        _dust_model = getattr(self, "_dust_model", "two_component")
        if _dust_model == "single_component":
            dust_live_shape_params = self._requested_law_shape_params()
        else:
            dust_live_shape_params = self._requested_law_shape_params(
                getattr(self, "_dust_law_bc", None),
                getattr(self, "_dust_law_diff", None),
                getattr(self, "_dust_law_neb", None),
            )

        chain = build_components(
            ssp_data=self.ssp_data,
            dust_live_shape_params=dust_live_shape_params,
            sfh_model=mean_model,
            field=field_on,
            metallicity_model=getattr(self, "_met_mode", "delta"),
            n_grid=int(getattr(self.spec, "n_grid", 256)),
            lgmet_scatter=float(getattr(self, "_lgmet_scatter", 0.2)),
            age_kernel=getattr(self.spec, "age_kernel", None),
            sfh_bin_edges_gyr=getattr(self.spec, "bin_edges_gyr", None),
            field_centering=float(getattr(self.spec, "field_centering", 1.0)),
            nebular_backend=neb_backend_name,
            nebular_backend_instance=neb_backend_instance,
            cue_full_catalog=bool(getattr(self.spec, "cue_full_catalog", False)),
            agn_model=getattr(self, "_agn_model", None),
            agn_disc_block=getattr(self, "_agn_disc_block", "none"),
            agn_nlr_block=getattr(self, "_agn_nlr_block", "none"),
            agn_blr_block=getattr(self, "_agn_blr_block", "none"),
            agn_feii_block=getattr(self, "_agn_feii_block", "none"),
            agn_torus_block=getattr(self, "_agn_torus_block", "none"),
            agn_attenuation_block=getattr(self, "_agn_attenuation_block", "none"),
            agn_norm=getattr(self, "_agn_norm", "cigale_joint"),
            dust_law_bc=getattr(self, "_dust_law_bc", "power_law"),
            dust_law_diff=getattr(self, "_dust_law_diff", "power_law"),
            dust_law_neb=getattr(self, "_dust_law_neb", None),
            dust_law_overrides=getattr(self, "_dust_law_overrides", None),
            dust_lyman_cutoff_aa=getattr(self, "_dust_lyman_cutoff_aa", 0.0),
            dust_lyc_absorb_all=getattr(self, "_dust_lyc_absorb_all", False),
            dust_eb_include_lyc=getattr(self, "_dust_eb_include_lyc", False),
            dust_emission_model=getattr(self, "_dust_emission_model", None),
            astrodust_spinning_dust=bool(getattr(self, "_astrodust_spinning_dust", False)),
            astrodust_f_cnm=float(getattr(self, "_astrodust_f_cnm", 0.28)),
            use_dust=(getattr(self, "_dust_model", "two_component") != "off"),
            dust_model=getattr(self, "_dust_model", "two_component"),
            wg00_dust_curve=getattr(self, "_wg00_dust_curve", "mw"),
            wg00_geometry=getattr(self, "_wg00_geometry", "shell"),
            wg00_structure=getattr(self, "_wg00_structure", "homogeneous"),
            use_radio=bool(getattr(self, "_uses_radio", False)),
            radio_sfr_mode=getattr(self, "_radio_sfr_mode", "bell2003"),
            radio_agn_model=getattr(self, "_radio_agn_model", "powerlaw"),
            use_xray=bool(getattr(self, "_uses_xray", False)),
            xray_model=getattr(self, "_xray_model", "yang20"),
            use_igm=bool(getattr(self, "_uses_igm", False)),
            igm_model=getattr(self, "_igm_model", "inoue"),
            igm_patchy=bool(getattr(self, "_igm_patchy", False)),
            use_dla=bool(getattr(self, "_uses_dla", False)),
            use_shock=bool(getattr(self, "_uses_shock", False)),
            shock_norm=getattr(self, "_shock_norm", "frac"),
            shock_abundance=getattr(self, "_shock_abundance", "solar"),
            shock_component=getattr(self, "_shock_component", "combined"),
        )

        # SINGLE BUILD-TIME PRECOMPUTATION PASS: resolve all component data before
        # first predict, unconditionally at build time (not gated on approx flags).
        # This ensures that components load their template data exactly once,
        # fixing #1278 where precompute() never ran on the default path.
        #
        # Strategy: iterate once over the chain. For each component:
        # 1. Call precompute() with basic args (load data if needed)
        # 2. If wave_precomp is enabled, augment precomputed state with LUT data
        # 3. If spectrum_precomp is enabled, augment precomputed state with spec LUT data
        #
        # This unifies the four separate isinstance scans (AGN/Nebular/IGM ×2)
        # into one loop that handles all specializations in order.
        from dataclasses import replace

        from tengri.components.agn.component import AGNSEDComponent
        from tengri.components.igm.component import IGMSEDComponent
        from tengri.components.nebular.component import NebularSEDComponent
        from tengri.components.stellar.component import StellarSEDComponent

        # Precompute-config state: extracted once, reused for all components
        wave_precomp_enabled = (
            self._approx.get("wave_precomp")
            and self.observation is not None
            and hasattr(self.observation, "photometry")
            and self.observation.photometry is not None
        )
        spec_precomp_enabled = (
            self._approx.get("spectrum_precomp")
            and self.observation is not None
            and self.observation.can_do_spectroscopy
        )

        # Build filter/redshift specs once (reused by all components)
        filters = None
        redshift_spec = None
        spec_wave_obs = None

        if wave_precomp_enabled or spec_precomp_enabled:
            # Determine redshift spec (shared by both paths)
            try:
                redshift_dist = self.spec.get_distribution("redshift")
                is_fixed = redshift_dist.is_fixed
                z_bounds = redshift_dist.bounds
            except (AttributeError, KeyError):
                is_fixed = True
                z_bounds = (0.0,)

            if wave_precomp_enabled:
                # Build filter tuple
                filters = tuple(
                    zip(
                        self.observation.photometry.filter_waves,
                        self.observation.photometry.filter_trans,
                        strict=False,
                    )
                )

                # Redshift spec for photometry
                if is_fixed and self._catalog_z_range is None:
                    redshift_spec = {"mode": "fixed", "value": float(z_bounds[0])}
                else:
                    cfg = self._approx_config or WavePrecomp()
                    if self._catalog_z_range is not None:
                        z_lo, z_hi = self._catalog_z_range
                        pad = 0.0
                    elif z_bounds is None or len(z_bounds) < 2:
                        z_lo, z_hi = 0.001, 3.0
                        pad = 0.0
                    else:
                        z_lo, z_hi = float(z_bounds[0]), float(z_bounds[1])
                        pad = 0.01 * (z_hi - z_lo)
                    redshift_spec = {
                        "mode": "free",
                        "z_min": (cfg.z_min if cfg.z_min is not None else max(0.001, z_lo - pad)),
                        "z_max": cfg.z_max if cfg.z_max is not None else z_hi + pad,
                        "n_z": cfg.n_z,
                    }

            if spec_precomp_enabled:
                spec_wave_obs = self.observation.spectroscopy.wave_obs
                if not is_fixed:
                    cfg = self._approx_config or SpectrumPrecomp()
                    if z_bounds is None or len(z_bounds) < 2:
                        z_lo, z_hi, pad = 0.001, 3.0, 0.0
                    else:
                        z_lo, z_hi = float(z_bounds[0]), float(z_bounds[1])
                        pad = 0.01 * (z_hi - z_lo)
                    redshift_spec = {
                        "mode": "free",
                        "z_min": (
                            cfg.z_min
                            if getattr(cfg, "z_min", None) is not None
                            else max(0.001, z_lo - pad)
                        ),
                        "z_max": (
                            cfg.z_max if getattr(cfg, "z_max", None) is not None else z_hi + pad
                        ),
                        "n_z": getattr(cfg, "n_z", 100),
                    }
                else:
                    redshift_spec = {"mode": "fixed", "value": float(z_bounds[0])}

        # MAIN PRECOMPUTATION LOOP: every component, one pass
        for idx, comp in enumerate(chain):
            requires_template_data = getattr(comp, "requires_template_data", True)
            if not requires_template_data:
                continue

            # Some components resolve their own templates inside ``predict``,
            # against the traced ``wave``. Caching those here would replace
            # traced values with concrete ones, which are then closure-captured
            # and baked into the graph -- measured at 4.97 MB for astrodust, and
            # at 29.94 MB on builds 2+ in the case recorded on
            # ``_template_data_for_jit``. ``astrodust.predict`` documents the
            # same trap from the tracer-leak side.
            #
            # This is deliberately its own flag rather than a reuse of
            # ``accepts_threaded_templates``. That one already means "I publish a
            # bundle from ``templates_for_threading()``", and
            # ``test_every_opted_in_component_can_resolve_a_bundle`` enforces it;
            # borrowing it here made astrodust declare a bundle it cannot
            # produce, which is the advertise-without-delivering shape #1738 is
            # about.
            if getattr(comp, "resolves_templates_at_trace_time", False):
                continue

            # Base precompute: load data, set up _state
            state = comp.precompute(
                ssp_data=comp.ssp_data if hasattr(comp, "ssp_data") else None,
                wave_grid=self.wavelengths,
                approx=self._approx,
            )

            # Wave_precomp augmentation: build LUTs for photometry
            if wave_precomp_enabled and len(chain) > 0 and chain[0].name == "stellar":
                if isinstance(comp, StellarSEDComponent) and filters is not None:
                    state = comp.precompute(
                        ssp_data=comp.ssp_data,
                        wave_grid=None,
                        approx=self._approx,
                        filters=filters,
                        redshift_spec=redshift_spec,
                    )
                elif (isinstance(comp, AGNSEDComponent) and filters is not None) or (
                    isinstance(comp, NebularSEDComponent) and filters is not None
                ):
                    state = comp.precompute(
                        ssp_data=None,
                        wave_grid=None,
                        approx=self._approx,
                        filters=filters,
                    )
                elif isinstance(comp, IGMSEDComponent):
                    igm_state = comp.precompute_band_factors(
                        wave_rest=self.wavelengths,
                        photometry=self.observation.photometry,
                        filters=filters,
                        redshift_spec=redshift_spec,
                    )
                    # The band factors ARE the precompute product for IGM: its own
                    # precompute() is a documented no-op, so ``state`` here is an
                    # empty marker. An earlier version of this pass kept that
                    # marker and copied only the two spec_* fields across, which
                    # dropped band_zgrid/band_table -- so predict_via_precomp
                    # re-evaluated the full Inoue+2014 curve on every call and the
                    # #1135 fold stopped being free (12.7 MFLOPs, caught by
                    # test_the_fold_is_free_at_runtime). The richer state wins.
                    state = igm_state
                    # Fold IGM transmission into stellar subbands
                    if isinstance(chain[0], StellarSEDComponent):
                        chain[0] = replace(
                            chain[0],
                            _state=_fold_igm_into_subbands(comp, chain[0]._state),
                        )

            # Spectrum_precomp augmentation: build spectrum LUTs
            if spec_precomp_enabled and len(chain) > 0 and chain[0].name == "stellar":
                if isinstance(comp, StellarSEDComponent) and spec_wave_obs is not None:
                    spec_state = comp.precompute(
                        ssp_data=comp.ssp_data,
                        wave_grid=None,
                        approx=self._approx,
                        spec_wave_obs=spec_wave_obs,
                        redshift_spec=redshift_spec,
                    )
                    # Merge spectrum LUT into existing state
                    if state is not None:
                        state = replace(
                            state,
                            ssp_spec_lut=spec_state.ssp_spec_lut,
                            ssp_spec_ztable=spec_state.ssp_spec_ztable,
                        )
                    else:
                        state = spec_state
                elif isinstance(comp, IGMSEDComponent) and spec_wave_obs is not None:
                    spec_igm = comp.precompute_spec_factors(
                        wave_rest=self.wavelengths,
                        spec_wave_obs=spec_wave_obs,
                        redshift_spec=redshift_spec,
                    )
                    if state is not None:
                        state = replace(
                            state,
                            spec_zgrid=spec_igm.spec_zgrid,
                            spec_table=spec_igm.spec_table,
                        )
                    else:
                        state = spec_igm

            # Replace component with precomputed state if it has _state field.
            # CRITICAL: never clobber an existing populated state with an empty marker.
            # The specialization blocks (wave_precomp, spectrum_precomp) run AFTER this
            # pass and populate _state further. If this pass overwrites with an empty
            # marker, silent wrong-physics results ensue. Check if old state has content.
            if hasattr(comp, "_state"):
                old_state = comp._state
                # State is "empty" if it's just a bare marker with no data.
                # A populated state has at least one content field.
                old_has_content = _state_has_content(old_state)
                new_has_content = _state_has_content(state)
                # Only replace if: (1) new state has content, or (2) old state was empty
                if new_has_content or not old_has_content:
                    chain[idx] = replace(comp, _state=state)

        # Bake the fast-dust-emission routing flag onto the dust component so
        # ``apply`` can branch on it statically (structural, not a runtime arg).
        if self._approx.get("fast_dust_emission"):
            from dataclasses import replace as _replace

            from tengri.components.dust.two_component import DustSEDComponent

            for _i, _c in enumerate(chain):
                if isinstance(_c, DustSEDComponent):
                    chain[_i] = _replace(_c, fast_emission=True)
                    break

        return chain

    # ── Batch operations ──────────────────────────────────────────────

    def predict_photometry_batch(self, params_batch, *, ssp_data=None, template_data=None):
        """Compute photometry for a batch of parameter sets via jax.vmap.

        **Use this method for** posterior chains / mock catalogs (batched
        forward pass). **For interactive single-galaxy use**, access
        ``model.predict(params).photometry()``.

        Parameters
        ----------
        params_batch: dict of arrays
            Each value has shape (N, ...) with leading batch dimension.

        Returns
        -------
        array, shape (N, n_filters)
            Photometric flux for each galaxy.

        Notes
        -----
        **JIT-compatible**: yes, uses :func:`jax.vmap` over
        :meth:`predict_photometry`.

        Examples
        --------
        >>> import jax
        >>> key = jax.random.PRNGKey(0)
        >>> params_batch = {
        ...     k: jnp.tile(v[None], (100,) + (1,) * (len(v.shape)))
        ...     for k, v in posterior.samples.items()
        ... }
        >>> flux_batch = model.predict_photometry_batch(params_batch)
        """
        from tengri.forward.convenience import predict_photometry_batch as _fn

        return _fn(self, params_batch, ssp_data=ssp_data, template_data=template_data)

    def predict_spectrum_batch(self, params_batch, *, ssp_data=None, template_data=None):
        """Compute spectra for a batch of parameter sets via jax.vmap.

        **Use this method for** batched spectra over posterior chains.
        **For interactive single-galaxy use**, access
        ``model.predict(params).spectrum``.

        Parameters
        ----------
        params_batch: dict of arrays
            Each value has leading batch dimension.

        Returns
        -------
        array, shape (N, n_pix)
            Spectral flux for each galaxy.

        Notes
        -----
        **JIT-compatible**: yes, uses :func:`jax.vmap` over
        :meth:`predict_spectrum`.

        Examples
        --------
        >>> params_batch = {
        ...     k: jnp.tile(v[None], (1000,) + (1,) * (len(v.shape)))
        ...     for k, v in posterior.samples.items()
        ... }
        >>> flux_batch = model.predict_spectrum_batch(params_batch)
        >>> flux_batch.shape
        (1000, n_pix)
        """
        from tengri.forward.convenience import predict_spectrum_batch as _fn

        return _fn(self, params_batch, ssp_data=ssp_data, template_data=template_data)

    @classmethod
    def from_config(
        cls,
        ssp,
        sfh=...,
        dust=...,
        nebular=...,
        agn=...,
        redshift=...,
        filters: list[str] | None = None,
        wave_obs=None,
        priors: dict | None = None,
        **model_kwargs,
    ) -> SEDModel:
        """Build a SEDModel from a grouped configuration dict.

        For the common case: instead of constructing
        ``Parameters``, ``SSPData``, ``Observation``, and ``SEDModel`` separately,
        provide a single grouped config and receive a fully configured ``SEDModel``.

        Parameters
        ----------
        ssp: str or SSPData
            Path to SSP HDF5 file, or a pre-loaded ``SSPData`` instance.
        sfh: str
            SFH family name, e.g. ``"tsnorm"``, ``"dpl"``, ``"dpl+field"``.
        dust: str
            Dust attenuation law. ``"charlot_fall"`` (default), ``"calzetti"``, etc.
        nebular: str or None
            Nebular emission backend. ``"baked_in"``, ``"cloudy_grid"``, ``"cb19"``,
            ``"mappings"``, ``"cue"``, ``"shock"``, or None.
        agn: str or None
            AGN model. None (disabled) or any AGN model name.
        redshift: float or str
            Fixed redshift (float), or ``"free"`` to add a free redshift parameter.
        filters: list of str, optional
            Filter names for photometry, e.g. ``["sdss_u", "sdss_g", "sdss_r"]``.
        wave_obs: array, optional
            Observed-frame wavelength array for spectroscopy.
        priors: dict, optional
            Parameter priors. Keys may be short names (``"log_total_mass"``),
            universal short names (``"logzsol"``), or full prefixed names.
            Short names are expanded automatically.
        **model_kwargs
            Forwarded to ``SEDModel.__init__()``.

        Returns
        -------
        SEDModel
            Fully initialized model ready for prediction or fitting.

        Notes
        -----
        Ellipsis (``...``) placeholders in optional parameters map to
        defaults from ``defaults.toml``. For example, ``dust=...`` uses
        the default dust attenuation law.

        Examples
        --------
        >>> model = tengri.SEDModel.from_config(
        ...     ssp="data/ssp.h5",
        ...     sfh="dense_basis",
        ...     filters=["sdss_u", "sdss_g", "sdss_r"],
        ...     redshift=0.1,
        ...     priors=dict(
        ...         log_total_mass=tengri.Uniform(8, 12),
        ...         log_sfr_inst=tengri.Uniform(-2, 3),
        ...         logzsol=tengri.Uniform(-2, 0.2),
        ...     ),
        ... )
        """
        from tengri.forward.convenience import build_model_from_config
        from tengri.parameters.defaults import UNSET

        # Map Ellipsis (signature placeholder) → UNSET so build_model_from_config
        # knows to fall back to defaults.toml instead of hard-coded values.
        def _r(v):
            """Convert ellipsis to UNSET sentinel for optional config parameters."""
            return UNSET if v is ... else v

        return build_model_from_config(
            cls,
            ssp,
            sfh=_r(sfh),
            dust=_r(dust),
            nebular=_r(nebular),
            agn=_r(agn),
            redshift=_r(redshift),
            filters=filters,
            wave_obs=wave_obs,
            priors=priors,
            **model_kwargs,
        )

    @classmethod
    def build(
        cls,
        ssp_data,
        *,
        sfh=None,
        met=None,
        dust_attenuation=None,
        dust_emission=None,
        neb=None,
        shock=None,
        agn=None,
        igm=None,
        radio=None,
        xray=None,
        foreground=None,
        redshift=None,
        filters=None,
        observation=None,
        **model_kwargs,
    ) -> SEDModel:
        """Build an SEDModel from the nested-dict grammar.

        The primary way to construct a model. Organizes physics components
        into one dict per block (sfh, dust_attenuation, neb, etc.), each
        declaring a structural type, which parameters are free, and per-parameter
        values. Translates this grammar to a ``Parameters`` spec via
        :func:`tengri.parse_groups`, then constructs the model.

        The grammar has three kinds of keys:

        1. **Structural keys** configure the component variant and behavior:
           ``'type'`` (required; e.g., ``'type': 'dpl'`` for SFH), and
           component-specific settings (``'law'`` for dust, ``'norm'`` for AGN, etc.).
        2. **``'all_params'`` wildcard** sets free/fixed status for all parameters
           in the group not explicitly overridden. Accepts :data:`~tengri.FREE` or
           :data:`~tengri.FIXED` (default). The only valid wildcard spelling.
        3. **Parameter keys** are bare or full-prefixed names that override the
           wildcard or default. Use short forms (``'beta'`` in sfh, ``'tau_bc'`` in
           dust) for readability.

        **Core groups with defaults:** ``sfh``, ``met``, ``dust_attenuation``,
        ``dust_emission``, ``redshift``.

        **Optional groups (OFF by default):** ``neb``, ``shock``, ``agn``, ``igm``,
        ``radio``, ``xray``, ``foreground``. Activate by providing a dict with
        ``'type'``; omit or pass ``{'type': 'none'}`` to keep off.

        Parameters
        ----------
        ssp_data: SSPData
            Pre-loaded SSP grid (from :func:`load_ssp_data`).
        sfh : dict, optional
            Star-formation history. Keys: ``'type'`` (required; e.g., ``'dpl'``,
            ``'delayed_tau'``, ``'field'``), ``'age_kernel'`` ('cic' or 'dsps'),
            ``'bin_edges_gyr'`` (for non-parametric), ``'all_params'``, and
            per-parameter overrides. Menu: :func:`tengri.list_sfh_models`.
        met : dict, optional
            Metallicity mode. Keys: ``'type'`` (required; e.g., ``'table'``,
            ``'ramp'``), ``'all_params'``, parameters. Menu:
            :func:`tengri.list_metallicity_modes`.
        dust_attenuation : dict, optional
            Dust attenuation screen. Keys: ``'type'`` (required; ``'single_component'``,
            ``'two_component'``, ``'wg00'``), ``'law'`` or ``'law_bc'``/``'law_diff'``
            (required depending on type), ``'all_params'``, parameters. Menu:
            :func:`tengri.list_dust_laws`. On two-component, you must provide
            either a single ``'law'`` for both screens or both ``'law_bc'`` and
            ``'law_diff'``.
        dust_emission : dict, optional
            Dust infrared emission. Keys: ``'type'`` (required; ``'dale2014'``,
            ``'draine2016'``, etc.), ``'eta_balance'``, ``'all_params'``, parameters.
            Menu: :func:`tengri.list_dust_emission_models`.
        neb : dict, optional
            Nebular emission. Keys: ``'type'`` (required; ``'cue'``, ``'cloudy'``, ``'none'``),
            ``'full_catalog'``, ``'grid'`` (CLOUDY), ``'all_params'``, parameters.
            Default: off. Menu: :func:`tengri.list_nebular_backends`.
            Metallicity (``'logZ_gas'`` or ``'neb_logz'``) is **independent** from
            ``met=``; default is ``-0.3`` (solar).
        shock : dict, optional
            Shock nebular emission. Keys: ``'type'`` (required; ``'mappings'``,
            ``'none'``), ``'norm'`` (``'frac'``, ``'lhalpha'``, ``'component'``),
            ``'abundance'``, ``'all_params'``, parameters. Default: off.
            Composes with ``neb`` when both are on.
        agn : dict, optional
            AGN emission. Keys: ``'type'`` (required; ``'composable'``, ``'legacy'``,
            ``'none'``), ``'norm'`` (``'cigale_joint'`` or ``'independent'``), and
            six optional sub-blocks (``'disc'``, ``'torus'``, ``'nlr'``, ``'blr'``,
            ``'feii'``, ``'atten'``), each with ``'type'`` and parameters. Default: off.
            Each sub-block follows the same grammar. Menu: :func:`tengri.list_agn_models`,
            :func:`tengri.list_agn_blocks`.
        igm : dict, optional
            IGM absorption. Keys: ``'type'`` (required; ``'inoue'``, ``'madau'``,
            ``'meiksin06'``, ``'none'``), ``'patchy'``, optional ``'dla'`` sub-block,
            ``'all_params'``, parameters. Default: off. Menu: :func:`tengri.list_igm_models`.
        radio : dict, optional
            Radio emission (star-formation and/or AGN). Keys: ``'type'``
            (required; ``'sfonly'``, ``'agn'``, ``'sf_agn'``, ``'none'``), optional
            ``'sf'`` and ``'agn'`` sub-blocks, ``'all_params'``, parameters.
            Default: off. Menu: :func:`tengri.list_radio_models`.
        xray : dict, optional
            X-ray emission. Keys: ``'type'`` (required; ``'yang22'``, ``'lehmer'``,
            ``'none'``), ``'all_params'``, parameters. Default: off.
            Menu: :func:`tengri.list_xray_models`.
        foreground : dict, optional
            Milky Way foreground reddening. Keys: ``'ebmv_mw'`` (E(B-V) in mag),
            ``'law'`` (dust law; e.g., ``'mw_rv31'``), ``'rv'`` (RV override).
            No ``'type'`` (not a registry). Default: off (no foreground reddening).
        redshift : scalar, Distribution, or sentinel
            Source redshift. **REQUIRED**. Omitting it raises ``ParameterError``.
            Specify as one of:

            - ``Fixed(z)`` for a known redshift (e.g., ``Fixed(0.05)``)
            - ``Uniform(lo, hi)`` for a photo-z fit
            - A bare scalar (auto-converts to ``Fixed``, e.g., ``redshift=0.05``)
            - Any other ``Distribution`` (e.g., ``Normal(mu, sigma)``)

            With ``approx=WavePrecomp()``, a free redshift adds ~9 s to the
            build (for IGM z-table folding); a fixed z adds ~0.4 s. See
            :doc:`/performance/compilation`.
        filters: list of str, optional
            Filter names; forwarded to ``__init__``.
        observation : Observation, optional
            Observation object (photometry, spectroscopy); forwarded to
            ``__init__``.
        **model_kwargs
            Additional keywords forwarded to :meth:`__init__` (e.g.,
            ``approx``, ``precompute``). ``forward_dtype`` is retired and
            ignored (#1433).

        Returns
        -------
        SEDModel
            Fully initialized model, equivalent to
            ``SEDModel(parse_groups(**groups), ssp_data, ...)``.

        Raises
        ------
        ParameterError
            If redshift is omitted, if unknown group keys are provided, if
            required structural keys (like ``'law'`` for dust) are missing,
            or if ``'all_params': FREE`` has no effect.

        Notes
        -----
        **Grammar rules:**

        - Every group dict must have a ``'type'`` key (except foreground, which
          has no registry).
        - ``'all_params'`` is the only wildcard. The retired ``'*'`` raises
          ``TypeError``.
        - Parameter names auto-resolve to their full prefixed forms. Short
          forms (``'beta'`` in sfh, ``'tau_bc'`` in dust) are preferred.
        - Unknown keys raise with suggestions via ``difflib``.
        - A group dict with keys but no ``'type'`` raises.
        - ``'all_params': FREE`` on a group with no free-parameter models raises
          (e.g., ``radio={'type': 'sfonly', 'all_params': FREE}``). Use explicit
          priors instead: ``radio={'q10': Uniform(...)}``.

        The resolved model configuration can be inspected and edited via:

        >>> config = model.spec.to_groups()  # dict with all groups
        >>> model.spec.summary()  # print with provenance tags

        See Also
        --------
        tengri.parse_groups: The underlying nested-dict parser that returns
            a :class:`Parameters` spec.
        tengri.recipes : Pre-built configuration dicts for common cases
            (star-forming galaxies, AGN, high-z, etc.).
        tengri.FREE, tengri.FIXED : Sentinel values for the ``'all_params'``
            wildcard.
        SEDModel.from_dict : Load a model from a serialized config dict.
        SEDModel.from_file : Load a model from a config file (JSON or YAML).
        model_configuration : Reference documentation for the grammar.

        Examples
        --------
        **Minimal star-forming galaxy (fixed redshift, photometry only):**

        >>> from tengri import SEDModel, FREE, FIXED, Uniform, Fixed
        >>> model = SEDModel.build(
        ...     ssp_data=ssp,
        ...     sfh={"type": "dpl", "all_params": FREE},
        ...     dust_attenuation={
        ...         "type": "two_component",
        ...         "law": "calzetti",
        ...         "all_params": FIXED,
        ...         "tau_bc": 0.5,
        ...         "tau_diff": 0.3,
        ...     },
        ...     dust_emission={"type": "dale2014", "all_params": FIXED},
        ...     neb={"type": "cue", "all_params": FIXED},
        ...     redshift=Fixed(0.05),
        ...     filters=["sdss_u", "sdss_g", "sdss_r"],
        ... )

        **High-z stochastic SFH with free metallicity and photo-z:**

        >>> model = SEDModel.build(
        ...     ssp_data=ssp,
        ...     sfh={"type": "dpl", "all_params": FREE},
        ...     met={"type": "ramp", "all_params": FREE},
        ...     dust_attenuation={
        ...         "type": "two_component",
        ...         "law": "calzetti",
        ...         "tau_bc": Uniform(0, 1),
        ...         "tau_diff": Uniform(0, 0.5),
        ...     },
        ...     neb={"type": "cue", "logZ_gas": Uniform(-1, 0)},
        ...     redshift=Uniform(2, 4),  # photo-z
        ...     observation=obs,
        ... )

        **Composable AGN with selective freedom:**

        >>> model = SEDModel.build(
        ...     ssp_data=ssp,
        ...     sfh={"type": "dpl", "all_params": FIXED, "alpha": Uniform(0.5, 3)},
        ...     agn={
        ...         "type": "composable",
        ...         "disc": {"type": "powerlaw", "all_params": FIXED},
        ...         "torus": {"type": "skirtor", "all_params": FREE},
        ...         "nlr": {"type": "cue", "logZ_gas": -0.3},
        ...         "norm": "cigale_joint",
        ...     },
        ...     igm={"type": "inoue"},
        ...     redshift=Fixed(0.3),
        ...     observation=obs,
        ... )

        **Recipe + tweak (start with a built-in template, customize):**

        >>> from tengri import recipes
        >>> config = recipes.star_forming_photometry()
        >>> config["dust_attenuation"]["tau_bc"] = 0.8  # override default
        >>> config["neb"]["logZ_gas"] = Uniform(-0.5, 0)  # photo-z on gas metallicity
        >>> model = SEDModel.build(ssp_data=ssp, observation=obs, **config)
        """
        # Validate that redshift is provided
        if redshift is None:
            raise ValueError(
                "redshift is required. Specify one of:\n"
                "  - redshift=Fixed(z) for a known redshift\n"
                "  - redshift=Uniform(lo, hi) for a photo-z fit\n"
                "  - redshift=<any Distribution> for other priors"
            )

        # Reject the retired apply_igm parameter
        if "apply_igm" in model_kwargs:
            raise ValueError(
                "apply_igm is retired. IGM activation is now derived from the igm dict: "
                "pass igm={'type': 'inoue'} (or 'madau', 'meiksin06') to enable IGM, or "
                "omit the igm dict (or pass igm={'type': 'none'}) to disable it."
            )

        groups = {
            k: v
            for k, v in dict(
                sfh=sfh,
                met=met,
                dust_attenuation=dust_attenuation,
                dust_emission=dust_emission,
                neb=neb,
                shock=shock,
                agn=agn,
                igm=igm,
                radio=radio,
                xray=xray,
                foreground=foreground,
                redshift=redshift,
            ).items()
            if v is not None
        }
        from tengri.parameters.groups import parse_groups

        # Anything in ``**model_kwargs`` that ``__init__`` does not declare is
        # grammar input, and ``parse_groups`` is the only thing that can judge
        # it. Forwarding it to the constructor instead loses two error channels
        # that already exist and are correct: the removed-group translations
        # (``stellar=`` names its ``met=`` replacement) and difflib's suggestion
        # on a misspelled group (``dsut=`` -> "Did you mean: dust?"). Both
        # degrade to a bare ``__init__() got an unexpected keyword argument``,
        # which names no replacement and no suggestion.
        #
        # PR #518 diagnosed exactly this and fixed it for the four keys in
        # ``_TOP_LEVEL_SETTINGS`` (``n_grid`` and friends), which this rule
        # subsumes, they are not ``__init__`` parameters, so they route here.
        # Every other keyword kept the old behavior, which is how ``stellar=``
        # came to die this way in five reproduction notebooks after #1720
        # removed it (#1776-#1781).
        _init_kw = _init_keywords(cls)
        for _key in [k for k in model_kwargs if k not in _init_kw]:
            groups[_key] = model_kwargs.pop(_key)

        # Auto-propagate the emission-line velocity mode from a Spectroscopy
        # observation so the line-velocity params (eline_sigma_kms,
        # eline_delta_v_kms) register without the user setting eline_mode twice
        # (#653). An explicit eline_mode in the build kwargs wins.
        if "eline_mode" not in groups and observation is not None:
            _spec_obs = getattr(observation, "spectroscopy", None)
            _obs_eline = getattr(_spec_obs, "eline_mode", None)
            if _obs_eline is not None and _obs_eline != "off":
                groups["eline_mode"] = _obs_eline

        spec = parse_groups(**groups)
        _validate_fracagn_requires_dust(spec)
        _validate_dale2014_requires_no_sf_radio(spec)
        _warn_agn_dust_double_count(spec)
        _warn_dead_gradient_params(spec)
        return cls(
            spec,
            ssp_data,
            filters=filters,
            observation=observation,
            **model_kwargs,
        )

    @classmethod
    def from_dict(
        cls, config: dict, ssp_data, *, filters=None, observation=None, **model_kwargs
    ) -> SEDModel:
        """Build an SEDModel from a serialized config dict."""
        from tengri.config.serialize import deserialize_config

        deserialized = deserialize_config(config)
        return cls.build(
            ssp_data, filters=filters, observation=observation, **deserialized, **model_kwargs
        )

    @classmethod
    def from_file(
        cls, path: str | pathlib.Path, ssp_data, *, filters=None, observation=None, **model_kwargs
    ) -> SEDModel:
        """Build an SEDModel from a config file (JSON or YAML)."""
        from tengri.config.serialize import load_config_from_file

        config = load_config_from_file(path)
        return cls.from_dict(
            config, ssp_data, filters=filters, observation=observation, **model_kwargs
        )

    @classmethod
    def from_yaml(
        cls, yaml_str: str, ssp_data, *, filters=None, observation=None, **model_kwargs
    ) -> SEDModel:
        """Build an SEDModel from a YAML string."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "from_yaml requires pyyaml. Install with: pip install pyyaml"
            ) from None
        from tengri.config.exceptions import ConfigError

        try:
            config = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML: {e}") from e
        if not isinstance(config, dict):
            raise ConfigError(f"YAML must deserialize to a dict, got {type(config)}")
        return cls.from_dict(
            config, ssp_data, filters=filters, observation=observation, **model_kwargs
        )

    @classmethod
    def from_json(
        cls, json_str: str, ssp_data, *, filters=None, observation=None, **model_kwargs
    ) -> SEDModel:
        """Build an SEDModel from a JSON string."""
        import json as json_module

        from tengri.config.exceptions import ConfigError

        try:
            config = json_module.loads(json_str)
        except json_module.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON: {e}") from e
        if not isinstance(config, dict):
            raise ConfigError(f"JSON must deserialize to a dict, got {type(config)}")
        return cls.from_dict(
            config, ssp_data, filters=filters, observation=observation, **model_kwargs
        )

    @property
    def config(self) -> dict:
        """Return the model configuration as a nested-dict (fully resolved)."""
        from tengri.parameters.groups import parameters_to_groups

        return parameters_to_groups(self.spec)

    def to_yaml(self, path: str | pathlib.Path | None = None) -> str:
        """Serialize the model configuration to YAML format."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "to_yaml requires pyyaml. Install with: pip install pyyaml"
            ) from None
        from tengri.config.serialize import serialize_config

        config_dict = self.config
        serialized = serialize_config(config_dict)
        yaml_str = yaml.dump(serialized, default_flow_style=False, sort_keys=False)
        if path is not None:
            with open(path, "w") as f:
                f.write(yaml_str)
            return ""
        return yaml_str

    def to_json(self, path: str | pathlib.Path | None = None) -> str:
        """Serialize the model configuration to JSON format."""
        import json

        from tengri.config.serialize import serialize_config

        config_dict = self.config
        serialized = serialize_config(config_dict)
        json_str = json.dumps(serialized, indent=2)
        if path is not None:
            with open(path, "w") as f:
                f.write(json_str)
            return ""
        return json_str

    def prior_predictive(self, n: int = 500, seed: int = 42) -> PriorPredictive:
        """Sample from the prior and evaluate forward model on each draw.

        Parameters
        ----------
        n: int
            Number of prior samples. Default 500.
        seed: int
            Random seed. Default 42.

        Returns
        -------
        PriorPredictive
            Object containing flux, SFH, and parameter draws with model reference.

        Notes
        -----
        Useful for prior predictive checks: visualizing what the model
        predicts under the prior without conditioning on data.

        Examples
        --------
        >>> pp = model.prior_predictive(n=100, seed=42)
        >>> # Access photometry, SFH, and parameters from the prior
        """
        from tengri.forward.convenience import prior_predictive as _fn

        return _fn(self, n=n, seed=seed)

    def fit(
        self,
        data=None,
        noise=None,
        method: str = DEFAULT_METHOD,
        data_type: str | None = None,
        *,
        photometry: tuple | None = None,
        spectrum: tuple | None = None,
        init: str | None = None,
        **kwargs,
    ):
        """Fit observed data with a convenient one-liner.

        A Bagpipes-style sugar over :class:`ForwardModel.fit`; equivalent to::

            forward = ForwardModel.build(sed=self, observation=...)
            result = forward.fit(data, noise, method=method, ...)

        For full control, a custom likelihood, per-fit parameter overrides,
        iterative refinement, or anything with a non-trivial output shape,         build the
        :class:`ForwardModel` yourself and call
        :meth:`ForwardModel.fit`, the canonical inference surface. For many
        independent galaxies, use :class:`~tengri.Catalog`.

        Parameters
        ----------
        data: array, optional
            Observed flux array (photometry or spectroscopy). For joint fitting,
            leave as ``None`` and use ``photometry=`` / ``spectrum=`` instead.
        noise: array, optional
            1-sigma uncertainties matching ``data``.
        method: str
            Inference method. Default ``"vi"`` (geoVI variational inference).
            Any canonical name accepted by ``Fitter.run()`` works here:
            ``"vi"``, ``"vi_linear"``, ``"mcmc"``, ``"mcmc_raytrace"``,
            ``"mcmc_nuts"``, ``"map"``, ``"laplace"``, ``"auto"``, etc.
        data_type: str or None
            ``"photometry"``, ``"spectroscopy"``, or ``"joint"``.
            When ``None`` (default), inferred from the model's ``observation``
            or from whether ``photometry=`` / ``spectrum=`` kwargs are used.
        photometry: tuple of (flux, noise), optional
            Photometric data for joint fitting. Pass alongside ``spectrum=``.
        spectrum: tuple of (flux, noise), optional
            Spectroscopic data for joint fitting. Pass alongside ``photometry=``.
        init: str or None
            Initialization strategy. ``"map"`` runs MAP optimization first, then
            uses the result to warm-start the requested method. ``None`` (default)
            uses the method's own default initialization.
        **kwargs
            Forwarded to the inference method (e.g. ``n_warmup``,
            ``n_samples`` for MCMC).

        Returns
        -------
        Posterior
            Inference results. ``.refine()`` continues or refines the fit.

        Notes
        -----
        Sugar over :meth:`ForwardModel.fit`, which stays the canonical
        inference surface. The engine underneath is an internal detail: all
        of its expensive caches are model-keyed, so it holds no state a
        fresh instance lacks and there is never a reason to reach for it
        directly.

        Examples
        --------
        >>> result = model.fit(flux_obs, noise)
        >>> result = model.fit(flux_obs, noise, method="mcmc")
        >>> result = model.fit(photometry=(flux_p, noise_p), spectrum=(flux_s, noise_s))
        >>> result = model.fit(flux_obs, noise, init="map")
        >>> result = model.fit(flux_obs, noise).refine("mcmc_raytrace")
        """
        from tengri.forward.convenience import fit_model

        return fit_model(
            self,
            data=data,
            noise=noise,
            method=method,
            data_type=data_type,
            photometry=photometry,
            spectrum=spectrum,
            init=init,
            **kwargs,
        )

    def fit_batch(
        self,
        catalog,
        flux_cols: list[str],
        err_cols: list[str],
        redshift_col: str | None = None,
        method: str = "vi",
        n_workers: int = 1,
        verbose: bool = True,
        output_dir: str | None = None,
        id_col: str | None = None,
        **kwargs,
    ) -> list:
        """Fit a batch of galaxies from a catalog (DataFrame, Table, or list of dicts).

        Parameters
        ----------
        catalog: DataFrame, Table, or list of dict
            Input catalog.
        flux_cols: list of str
            Column names for per-band flux values.
        err_cols: list of str
            Column names for per-band 1-sigma uncertainties.
        redshift_col: str or None
            If provided, use this column as per-row redshift.
        method: str
            Inference method. Default ``"vi"``.
        n_workers: int
            Currently ignored (reserved for multiprocessing). Default 1.
        verbose: bool
            Print per-galaxy progress. Default True.
        output_dir: str or None
            If provided, save each Posterior to ``{output_dir}/{id}.h5``.
        id_col: str or None
            Column name for galaxy identifiers in checkpoint filenames.
        **kwargs
            Forwarded to Fitter.run().

        Returns
        -------
        list of Posterior
            One result per galaxy in catalog.

        Notes
        -----
        Sequential fitting (no parallelization yet). For 1000+ galaxies,
        consider using :meth:`fit` in a loop with a multiprocessing pool.

        Examples
        --------
        >>> import pandas as pd
        >>> cat = pd.read_csv("catalog.csv")
        >>> results = model.fit_batch(
        ...     cat,
        ...     flux_cols=["f_u", "f_g", "f_r", "f_i", "f_z"],
        ...     err_cols=["e_u", "e_g", "e_r", "e_i", "e_z"],
        ...     redshift_col="z",
        ... )
        """
        from tengri.forward.convenience import fit_batch as _fn

        return _fn(
            self,
            catalog,
            flux_cols,
            err_cols,
            redshift_col=redshift_col,
            method=method,
            n_workers=n_workers,
            verbose=verbose,
            output_dir=output_dir,
            id_col=id_col,
            **kwargs,
        )

    def fit_population(
        self,
        observations_list: list,
        method: str = "vi",
        population_prior: dict | None = None,
        **kwargs,
    ):
        """Fit a population of galaxies with shared PSD hyperparameters.

        Parameters
        ----------
        observations_list: list
            Each element is a (flux, noise) tuple or dict with flux_obs/noise keys.
        method: str
            Hierarchical inference method. Default ``"vi"``.
        population_prior: dict or None
            Hyperpriors on shared PSD parameters.
        **kwargs
            Forwarded to PopulationFitter.run().

        Returns
        -------
        PopulationPosterior
            Hierarchical inference results with population-level and per-galaxy posteriors.

        Notes
        -----
        Enables population-level constraints on shared PSD hyperparameters
        (e.g., shared burst timescale across a sample). All galaxies must
        use the same model configuration.

        Examples
        --------
        >>> obs_list = [(flux1, noise1), (flux2, noise2), ...]
        >>> result = model.fit_population(obs_list, method="vi")
        """
        from tengri.forward.convenience import fit_population as _fn

        return _fn(
            self,
            observations_list,
            method=method,
            population_prior=population_prior,
            **kwargs,
        )

    def mock(self, params, snr=20.0, key=None):
        """Generate mock photometric observation with noise.

        Parameters
        ----------
        params: dict
            Parameter values.
        snr: float
            Signal-to-noise ratio. Default 20.0.
        key: PRNGKey, optional
            Random key for noise. If None, returns noiseless.

        Returns
        -------
        MockData
            Mock photometric observation.

        Notes
        -----
        Requires model to have filters configured (``filters=`` or
        ``observation=`` in constructor).

        Examples
        --------
        >>> key = jax.random.PRNGKey(0)
        >>> mock = model.mock(params, snr=15.0, key=key)
        >>> print(mock.flux.shape)  # (n_filters,)
        """
        from tengri.forward.convenience import mock as _fn

        return _fn(self, params, snr=snr, key=key)

    def mock_spectrum(self, params, wave_obs, snr=30.0, key=None):
        """Generate mock spectroscopic observation with noise.

        Parameters
        ----------
        params: dict
            Parameter values.
        wave_obs: array
            Observed wavelength grid [Angstrom].
        snr: float
            Signal-to-noise ratio per pixel. Default 30.0.
        key: PRNGKey, optional
            Random key for noise. If None, returns noiseless.

        Returns
        -------
        MockData
            Mock spectroscopic observation.

        Notes
        -----
        Noise is drawn from Gaussian distribution with standard deviation = flux/snr.

        Examples
        --------
        >>> wave_obs = np.linspace(4000, 5500, 1000)
        >>> mock = model.mock_spectrum(params, wave_obs, snr=10.0, key=key)
        >>> print(mock.flux.shape)  # (1000,)
        """
        from tengri.forward.convenience import mock_spectrum as _fn

        return _fn(self, params, wave_obs, snr=snr, key=key)

    def mock_batch(self, params_batch, snr=20.0, key=None):
        """Generate batch of mock photometric observations.

        Parameters
        ----------
        params_batch: dict of arrays
            Each value has leading batch dimension.
        snr: float
            Signal-to-noise ratio. Default 20.0.
        key: PRNGKey, optional
            Random key for noise. If None, returns noiseless.

        Returns
        -------
        MockData
            Mock observations with shape (N, n_filters).

        Notes
        -----
        Uses :func:`jax.vmap` over :meth:`mock` for vectorized generation.

        Examples
        --------
        >>> params_batch = {
        ...     k: jnp.tile(v[None], (1000,) + (1,) * (len(v.shape)))
        ...     for k, v in posterior.samples.items()
        ... }
        >>> mocks = model.mock_batch(params_batch, snr=15.0, key=key)
        """
        from tengri.forward.convenience import mock_batch as _fn

        return _fn(self, params_batch, snr=snr, key=key)

    def plot_sfh_posterior(
        self, posterior, true_params=None, ax=None, n_draws=50, color="C0", label="Posterior"
    ):
        """Plot posterior SFH with percentile fill and sample lines.

        Parameters
        ----------
        posterior: Posterior
            Inference results with samples (if available) or params.
        true_params: dict, optional
            True parameter values (if known) to overlay on plot.
        ax: matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure.
        n_draws: int
            Number of posterior samples to show as thin lines. Default 50.
        color: str
            Color for posterior lines. Default "C0" (first color in style).
        label: str
            Label for posterior. Default "Posterior".

        Returns
        -------
        ax: matplotlib.axes.Axes
            The matplotlib Axes object with the plot.

        Notes
        -----
        Shows 16th and 84th percentiles as filled region, with individual
        sample curves in light color. If ``true_params`` provided, shows
        truth in black with dashed line for smooth SFH (parametric part).

        Examples
        --------
        >>> result = model.fit(flux, noise)
        >>> ax = model.plot_sfh_posterior(result)
        >>> ax.set_yscale("log")
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        if posterior.samples is None:
            sfh = self.predict_sfh(posterior.params)
            ax.plot(sfh["t_gyr"], sfh["sfr_mean"], color=color, lw=2, label=label)
        else:
            n_total = len(next(iter(posterior.samples.values())))
            sfh_draws = []
            for i in range(n_total):
                s_i = {k: posterior.samples[k][i] for k in posterior.samples}
                sfh_i = self.predict_sfh(s_i)
                key = "sfr_full" if self.spec.stochastic else "sfr_mean"
                sfh_draws.append(sfh_i[key])

            import numpy as np

            sfh_arr = np.array(sfh_draws)
            t_gyr = np.array(self.predict_sfh(posterior.params)["t_gyr"])

            lo = np.percentile(sfh_arr, 16, axis=0)
            hi = np.percentile(sfh_arr, 84, axis=0)
            ax.fill_between(t_gyr, lo, hi, color=color, alpha=0.2)

            n_show = min(n_draws, n_total)
            indices = np.linspace(0, n_total - 1, n_show, dtype=int)
            for idx in indices:
                ax.plot(t_gyr, sfh_arr[idx], color=color, alpha=0.1, lw=0.4)

            sfh_mean = self.predict_sfh(posterior.params)
            key = "sfr_full" if self.spec.stochastic else "sfr_mean"
            ax.plot(t_gyr, sfh_mean[key], color=color, lw=2, label=label)

        if true_params is not None:
            sfh_true = self.predict_sfh(true_params)
            key = "sfr_full" if self.spec.stochastic else "sfr_mean"
            ax.plot(sfh_true["t_gyr"], sfh_true[key], "k-", lw=2.5, label="Truth", zorder=10)
            if self.spec.stochastic:
                ax.plot(sfh_true["t_gyr"], sfh_true["sfr_mean"], "k--", lw=1, alpha=0.3)

        ax.set_xlabel("Lookback time (Gyr)")
        ax.set_ylabel(r"SFR (M$_{\odot}$/yr)")
        ax.set_xlim(0, 13.5)
        ax.legend(fontsize=9)
        return ax

    # ── Utilities ─────────────────────────────────────────────────────

    @property
    def wavelengths(self):
        """Rest-frame wavelength grid (Angstrom).

        Returns the SSP grid by default, or the extended panchromatic grid
        when radio or X-ray emission is enabled.

        Returns
        -------
        ndarray, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].

        Notes
        -----
        This is the grid used by :meth:`Prediction.rest_sed` by default when
        no custom ``wave`` is passed. Updated when radio/X-ray components
        are added to the model.

        Examples
        --------
        >>> print(model.wavelengths[0], model.wavelengths[-1])
        >>> # Default SSP range, e.g. 91.2 to 160000 Å
        """
        return self._rest_wavelength

    @staticmethod
    def _t_universe_gyr(z):
        """Age of the universe at redshift z in Gyr.

        Thin wrapper around age_at_z.

        Parameters
        ----------
        z: float or jnp.ndarray
            Redshift.

        Returns
        -------
        float
            Age of universe in Gyr.
        """
        return age_at_z(z)

    def _method_recommendation(self) -> tuple[str, str]:
        """Return (method_name, reason) for the recommended inference method."""
        from tengri.config.display import method_recommendation

        return method_recommendation(self)

    def tree(self) -> str:
        """Return a human-readable physics tree showing the model hierarchy.

        Shows the active sub-models at each physical layer (SFH, SPS, Dust,
        Nebular, AGN, Observation), the free parameters at each layer, and
        the recommended inference method.

        Returns
        -------
        str
            Multi-line formatted tree string.

        Notes
        -----
        Useful for inspecting model configuration before fitting or inference.

        Examples
        --------
        >>> print(model.tree())
        Model  [D=7, stochastic=False]
        ...
        """
        from tengri.config.display import tree as _tree

        return _tree(self)

    def recommend_method(self) -> str:
        """Return the recommended inference method string for this model.

        Returns
        -------
        str
            Canonical method name for ``Fitter.run()`` or ``model.fit()``.

        Notes
        -----
        Based on model dimensionality, complexity, and available precomputation.
        Use as input to ``model.fit(method=model.recommend_method())``.

        Examples
        --------
        >>> method = model.recommend_method()
        >>> result = model.fit(flux, noise, method=method)
        """
        method, _ = self._method_recommendation()
        return method

    def summary(self) -> str:
        """Return a human-readable summary of the model configuration.

        Returns
        -------
        str
            Formatted summary showing SSP grid, filters, precomputation,
            fused kernel status, and enabled components.

        Notes
        -----
        Similar to :meth:`tree` but focuses on computational configuration
        and precomputation status rather than physics parameters.

        Examples
        --------
        >>> print(model.summary())
        """
        from tengri.config.display import summary as _summary

        return _summary(self)


# Backward-compatibility alias
