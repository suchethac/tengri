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

Comments in this module reference historical "Phase" labels from the 2026
precompute rollout. The decoder ring: Phase 2 = approx settings owned by
components; Phase 3b/3c = the WavePrecomp SSP x filter photometry LUT
(fixed-z scalar -> free-z ztable); Phase 3d = the typed
``approx=WavePrecomp(...)`` construction contract (dict/bool/str forms
removed); Phase 5 = the SpectrumPrecomp per-pixel LUT; Phase 6 = removal of
the legacy hybrid kernels (``predict_observables_jit`` is the single
forward path).
"""

from __future__ import annotations

import contextlib
import dataclasses
import types
import warnings
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.stellar.sfh.registry import compute_field_gp, resolve_sfh
from tengri.components.stellar.sps.dsps_wrapper import csp_age_dt
from tengri.config.exceptions import ParameterMapError
from tengri.cosmology import age_at_z, luminosity_distance
from tengri.forward.pipeline import (
    interp_metallicity,
    interp_metallicity_evolving,
)
from tengri.forward.sed_model_types import (
    CompositionalKernels,
    HybridKernels,
    MockData,
    PriorPredictive,
    SEDModelState,
)
from tengri.observation.photometry import ab_mag_from_flux
from tengri.parameters.translate import (
    _CUE_GAS_IDENTITY_PARAMS,
    _CUE_IONSPEC_IDENTITY_PARAMS,
    _EVOLVING_ALPHA_PARAM_MAP,
    LOG10_ZSUN,
    _build_param_map,
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

# Re-export supporting types for backwards compatibility
__all__ = [
    "CompositionalKernels",
    "HybridKernels",
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
    n_z : int, default 100
        Number of grid points in the ztable. Higher → finer redshift
        interpolation, slower precompute.
    z_min : float or None, default None
        Lower bound of the ztable grid. ``None`` → pull from the redshift
        prior with 1 % padding. Ignored when redshift is ``Fixed`` unless
        ``catalog_z_range`` is set.
    z_max : float or None, default None
        Upper bound of the ztable grid. ``None`` → pull from the redshift
        prior with 1 % padding. Ignored when redshift is ``Fixed`` unless
        ``catalog_z_range`` is set.
    catalog_z_range : tuple of float or None, default None
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
    >>> # Catalog fit: 10⁴ galaxies at per-galaxy Fixed(z), one compile.
    >>> model = SEDModel.build(
    ...     ...,
    ...     redshift=Fixed(0.0),  # placeholder; injected per call
    ...     approx=WavePrecomp(catalog_z_range=(0.05, 1.5), n_z=200),
    ... )
    >>> for row in catalog:
    ...     posterior = model.fit(row.data, params={"redshift": row.z})
    """

    n_z: int = 100
    z_min: float | None = None
    z_max: float | None = None
    catalog_z_range: tuple[float, float] | None = None
    taylor_correction: bool = True
    """First-order spectral-moment (Ψ) correction to the effective-wavelength dust
    attenuation (Zacharegkas+2025). ``True`` (default) applies ``A(λ_eff)·Φ +
    A'(λ_eff)·Ψ`` for both single- and two-component dust, cutting the attenuation
    residual to ~0.3%. Set ``False`` for the bare ``A(λ_eff)·Φ`` (flat) form —
    ~0.8–1.5% on the birth-cloud layer, but well below typical SSP/dust template
    systematics, and slightly cheaper (the Ψ tensor is not built)."""

    fast_dust_emission: bool = False
    """Approximate the dust IR re-emission *band projection* when its exact, fast
    form is unavailable. For a ``modified_blackbody`` with fixed shape (``dust_T``,
    ``dust_beta_ir``, ``dust_epsilon_mbb``) and fixed ``redshift``, the template is
    linear in ``L_ir`` and its per-band integral ``R`` is precomputed once
    (CIGALE ``dl2014`` style): the exact ``L_ir × R`` projection is then used
    automatically — fastest *and* exact, no flag needed. This flag only bites when
    that constant ``R`` cannot be formed (free emission shape / redshift, or
    structured templates): ``True`` then samples the self-normalizing template at
    the filter effective wavelength instead of integrating it through each
    bandpass — much cheaper than the exact per-band integral (#622), at a
    band-shape approximation (smooth on a modified blackbody; up to a few percent
    on a band crossing a steep IR rise or a PAH complex). The energy balance is
    unaffected either way."""


@dataclasses.dataclass(frozen=True)
class SpectrumPrecomp:
    """Configuration for spectrum-grid LUT precomputation (Phase 5).

    Pass this to :class:`SEDModel` via ``approx=`` to enable spectroscopic
    LUT precomputation. The SSP × dust × IGM stack is precomputed at
    spectrum pixel centers (effective wavelengths in the galaxy rest frame)
    and cached per redshift. This is analogous to the photometric LUT path
    (Phase 3b/3c) but for spectroscopy.

    The pixel grid is inherited from ``Observation.spectroscopy.wave_obs``.
    For a free redshift, a redshift table is built so the rest-frame pixel
    grid ``wave_obs / (1 + z)`` can be interpolated at runtime; ``n_z`` /
    ``z_min`` / ``z_max`` tune that table (mirroring :class:`WavePrecomp`).
    When redshift is Fixed, these are ignored.

    Parameters
    ----------
    n_z : int, default 100
        Number of grid points in the free-z redshift table.
    z_min, z_max : float or None
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
    >>> # Joint fit — accelerate both channels with independent LUT configs:
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


def _warn_agn_dust_double_count(spec) -> None:
    """Warn when composable AGN and Dale2014 ``dust_frac_agn`` both inject AGN IR.

    The composable AGN's ``agn_fracAGN`` (CIGALE-joint tie) and Dale2014's
    embedded quasar template ``dust_frac_agn`` are two distinct AGN surfaces,
    both keyed off the same stellar ``L_absorbed`` (component_factory.py:346,
    ADR-0018 §5, issue #721). With both > 0 the AGN mid/far-IR is double-counted
    — the SKIRTOR/torus block already models AGN IR, so Dale2014's fracAGN should
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

    if not (_positive_active("dust_frac_agn") and _positive_active("agn_fracAGN")):
        return

    from tengri.config.exceptions import AGNDustDoubleCountWarning

    warnings.warn(
        "Both AGN surfaces are active: the composable AGN (agn_fracAGN > 0) and "
        "Dale2014 dust emission (dust_frac_agn > 0). Both inject AGN-heated IR "
        "from the same stellar L_absorbed, so AGN mid/far-IR is DOUBLE-COUNTED. "
        "Use one surface: set dust_frac_agn=0 and let the composable AGN torus "
        "(e.g. SKIRTOR) own the AGN IR — recommended when a torus block is "
        "configured — or drop the composable AGN and use Dale2014's embedded "
        "quasar template alone. See ADR-0018 §5 / issue #721. Filter "
        "AGNDustDoubleCountWarning if the overlap is deliberate.",
        AGNDustDoubleCountWarning,
        stacklevel=3,
    )


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
    spec : Parameters
        Parameter specification from ``tengri.Parameters``. Defines
        free/fixed parameters and their priors.
    ssp_data : SSPData
        Pre-loaded SSP templates (from ``load_ssp_data()``). Contains
        absolute SSP grid in ``log10(Z)`` absolute, age array, and
        optional mass-remaining tables for stellar mass surviving
        constraints.
    filters : list or tuple, optional
        Filter transmission curves for photometric prediction. Accepts either:

        - 3-tuple from :func:`load_filter_set`: ``(filter_waves, filter_trans, filter_curves)``
        - List of :class:`FilterCurve` namedtuples

        If provided, enables photometry prediction and automatic precomputation
        at initialization. Either ``filters`` or ``observation`` may be passed,
        not both.
    observation : Observation, optional
        Unified observation config (photometry + spectroscopy + emission lines).
        Mutually exclusive with ``filters``.
    precompute : bool, optional
        **Legacy / largely superseded.** Builds the pre-Phase-3
        ``PrecomputedData`` container (fixed-z SSP photometry/spectroscopy grid
        defaults). This predates — and is NOT — the fast LUT path: the
        Zacharegkas+2025 fast-photometry / spectroscopy speedup is selected at
        build time via ``approx=WavePrecomp()`` / ``approx=SpectrumPrecomp()``
        (see ``approx`` below), which builds its own LUTs through the component
        chain and does not consume ``PrecomputedData``. ``precompute`` now only
        supplies a couple of grid fallbacks for the exact path and is scheduled
        to be folded into ``approx`` (tracked in the precompute-naming cleanup
        issue). Default True; leave it unless you know you need the legacy
        container. Set False to skip building it.
    forward_dtype : str or jnp.dtype, optional
        Dtype for forward model computation. Default ``"float64"`` preserves
        full precision. ``"float32"`` halves memory and gives ~1.5× speedup
        with <0.1% accuracy loss for photometry.

        Affects both fused (photometry + precomputation) and exact paths:

        - **Fused path**: captured arrays (SSP grid, dust weights, effective
          wavelengths) cast to ``forward_dtype`` at kernel build; outputs
          always cast back to float64 for cosmological distance scaling.
        - **Exact path** (spectroscopy, non-precomputed AGN): three largest intermediates
          — metallicity-interpolated SSP ``(n_age, n_wave)``, dust attenuation
          ``(n_age, n_wave)``, dust age weights ``(n_age,)`` — computed in
          ``forward_dtype``, halving the 4.5 MB memory traffic that dominates
          exact-path dust cost.

        Cosmological distances always use float64 (float32 overflows at z > 0.01).
    approx : dict or bool, optional
        Control which approximations enter the component chain. Default True enables
        all approximations (fastest). False disables all (forces exact path
        everywhere). A dict enables selective control:

        - ``"ztable"``: SSP × filter lookup table indexed on redshift grid (True, default)
        - ``"wave_precomp"``: SSP × filter lookup table on fixed wavelength grid (False, default)

        Approximation dependencies (resolved at build time):

        - ``wave_precomp=True`` with free redshift auto-enables ``ztable=True``.
        - ``ztable=True`` requires ``wave_precomp=True``.
        - Unknown flag names raise ``ValueError`` with list of legal flags.

    compile : str, optional
        JIT-wrapping strategy for the forward pass. Default ``"per_component"``
        wraps each :class:`SEDComponent.apply` independently for faster cold-starts
        in notebooks; ``"fused"`` compiles the entire ``observation.predict ∘
        run_components`` chain at once for hot inference loops; ``"auto"`` is a
        stub that currently resolves to ``"per_component"``.

        **Legal values:** ``"per_component"`` (default), ``"fused"``, ``"auto"``.
        Invalid values raise ``ValueError``.

    csp_integration : str, optional
        CSP age integration scheme. Default ``"trapz"`` (trapezoidal on
        linear time). Options: ``"log_trapz"``, ``"log_interp"`` (Dopita+2005
        interpolation), ``"dsps_native"`` (DSPS trapezoidal with automatic
        metallicity marginalization), ``"dsps_met_table"`` (time-evolving
        metallicity table). See Appendix A of the forward model paper [2]_.

    Attributes
    ----------
    observation : Observation or None
        Attached observation object containing photometry and/or spectroscopy
        configuration. Set by constructor if filters or observation= passed.
    spec : Parameters
        Parameter specification defining all free/fixed parameters and their priors.
    ssp_data : SSPData
        Pre-loaded stellar population synthesis templates (from ``load_ssp_data()``).
    config : ModelConfig
        Frozen model configuration (immutable after init).

    Notes
    -----
    **JIT-compatible**: yes — all prediction methods (except
    :meth:`predict` for lazy evaluation) are fully JAX differentiable
    and can be called inside :func:`jax.jit` and :func:`jax.vmap`.

    **Gradient-safe**: yes — all physical parameters are differentiable
    for inference via HMC, VI, and score-based methods.

    **Approximation scheme**: All prediction methods route through the
    single JIT-safe orchestrator :meth:`predict_observables_jit` (Phase
    4-B, with SSP threading). Historical mode-cascade strategies
    (compositional / hybrid / exact) collapsed into this one path in
    Phase 6-prep (2026-05-20); the orchestrator itself remains XLA-fused
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
    # See docs/dev/forward-model-architecture.md §4. SEDModel directly
    # satisfies tengri.protocols.SubModel; ForwardModel's per-population
    # orchestration consumes the `run` and `declared_parameters` methods.

    name: str = "sed"

    # Default approximation settings (immutable — used as template only)
    # Phase 2: owned by components, per the unification plan.
    # "wave_precomp" = SSP × filter LUT on fixed wavelength grid (stellar component)
    # "ztable" = SSP × filter LUT indexed on redshift grid, requires wave_precomp
    # ztable is auto-enabled when wave_precomp=True and redshift is free.
    # "igm" = pre-compute IGM transmission at filter effective wavelengths for
    # the hybrid kernel at fixed z. Default True matches the historic behavior
    # before the Phase 3 ``approx=`` flag was introduced (``_build_precomputed_data``
    # always computed ``igm_eff`` when ``_uses_igm`` and ``_z_fixed`` were set).
    _DEFAULT_APPROX: ClassVar[dict] = {
        "wave_precomp": False,
        "ztable": False,
        "spectrum_precomp": False,
        "igm": True,
        "taylor_correction": True,
        "fast_dust_emission": False,
    }

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
        self._approx["wave_precomp"] = True

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
            self._approx["spectrum_precomp"] = False
            self._approx_config_spec = None
        else:
            self._approx["spectrum_precomp"] = True
            self._approx_config_spec = cfg

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
        # ``strategy`` is accepted for backwards-compat signature but ignored —
        # the kernel-selection strategy machinery was removed in Phase 6
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
        self._wave_chunk_size = wave_chunk_size

        # ── Observables NamedTuple (Phase 2) ─────────────────────
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
        # Contract (Phase 3d, 2026-05-20):
        #   * ``approx=None`` (default)        — exact wave-grid integration.
        #   * ``approx=WavePrecomp(...)``      — opt into the precomputed
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
        if approx is None:
            self._approx = dict(self._DEFAULT_APPROX)
            self._approx["wave_precomp"] = False
            self._approx["ztable"] = False
        else:
            # Accept a single config or a composite ``(WavePrecomp, SpectrumPrecomp)``
            # sequence (order-independent, at most one of each).
            configs = list(approx) if isinstance(approx, (tuple, list)) else [approx]
            wave_cfgs = [c for c in configs if isinstance(c, WavePrecomp)]
            spec_cfgs = [c for c in configs if isinstance(c, SpectrumPrecomp)]
            unknown = [c for c in configs if not isinstance(c, (WavePrecomp, SpectrumPrecomp))]
            if unknown or len(wave_cfgs) > 1 or len(spec_cfgs) > 1 or not configs:
                raise TypeError(
                    f"approx={approx!r} is not a legal value. Legal forms: "
                    "None (default — exact wave-grid), "
                    "WavePrecomp() for the SSP × filter LUT path, "
                    "SpectrumPrecomp() for the spectrum LUT path, or a composite "
                    "(WavePrecomp(...), SpectrumPrecomp()) tuple for a joint fit "
                    "(at most one of each). The pre-3d dict / bool / string forms "
                    "(e.g. approx={'wave_precomp': True}, approx=True, "
                    "approx='wave_precomp') were removed."
                )
            self._approx = dict(self._DEFAULT_APPROX)
            if wave_cfgs:
                self._resolve_wave_precomp(wave_cfgs[0])
            if spec_cfgs:
                self._resolve_spectrum_precomp(spec_cfgs[0], observation)
            # Primary config: WavePrecomp drives the shared z-table / Taylor /
            # catalog knobs when present (back-compat); else the SpectrumPrecomp
            # (but only when it actually activated — an R-fallback leaves it None).
            if wave_cfgs:
                self._approx_config = wave_cfgs[0]
            else:
                self._approx_config = self._approx_config_spec

        # Thread the Taylor-correction toggle (default True) from the config into
        # the approx dict, so the stellar precompute knows whether to build the Ψ
        # moment and predict_via_precomp whether to apply it (#617). When the
        # exact path is selected (config is None) the default True is harmless.
        if self._approx_config is not None:
            self._approx["taylor_correction"] = getattr(
                self._approx_config, "taylor_correction", True
            )
            self._approx["fast_dust_emission"] = getattr(
                self._approx_config, "fast_dust_emission", False
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
            self._approx["wave_precomp"] = True
            self._approx["spectrum_precomp"] = True

        # Free-redshift ztable auto-extension. ``ztable`` is an internal
        # extension of ``wave_precomp`` (free-z interpolation on the same LUT),
        # not a user flag — it switches on transparently when the method is
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
            # ``catalog_z_range`` is a WavePrecomp-only knob; under a joint
            # model the config may be a SpectrumPrecomp (no such field).
            cz = getattr(self._approx_config, "catalog_z_range", None)
            if cz is not None:
                if redshift_dist.is_fixed:
                    self._approx["ztable"] = True
                    self._catalog_z_range = (float(cz[0]), float(cz[1]))
                # Free-redshift case: catalog_z_range is harmless (ztable already on)
                # but record it so the compile_signature still distinguishes ranges.
                else:
                    self._approx["ztable"] = True
                    self._catalog_z_range = (float(cz[0]), float(cz[1]))
            elif not redshift_dist.is_fixed:
                self._approx["ztable"] = True

        # ── Stellar populations ───────────────────────────────────
        self._init_ssp(spec, ssp_data, csp_integration)

        # ── Collect parameter map deltas from each _init_* method ──
        param_map_deltas = []

        # ── Star formation history ────────────────────────────────
        param_map_deltas.append(self._init_sfh(spec))

        # ── Metallicity ───────────────────────────────────────────
        param_map_deltas.append(self._init_metallicity(spec))
        self._validate_metallicity_bounds(spec, ssp_data)

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

        # ── Cosmology (luminosity distance) ───────────────────────
        self._init_cosmology(spec)

        # ── Validate and freeze parameter map ─────────────────────
        self._validate_and_freeze_param_map(param_map_deltas)

        # ── Warm HDF5 grid caches BEFORE any JIT compilation ──────
        # (tracer-leak prevention; formerly the side-effect at the top of the
        # retired ``_build_precomputed_data``, #620). ``precompute`` is now an
        # accepted-but-ignored legacy kwarg — the fast path is opt-in via
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
        # config values — not lazily on the first predict_state, which may
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
            try:
                chain = self._build_component_chain()
                self._cached_component_chain = chain
                self._energy_balance_lut(chain)
                self._dust_emission_band_response(chain)
            except Exception as e:
                # Any build-time failure leaves the LUT disabled — the exact
                # full-wave energy-balance path is the correct fallback, but
                # it forfeits the speedup the astronomer opted into, so say so.
                warnings.warn(
                    f"WavePrecomp energy-balance LUT precompute failed ({e!r}); "
                    "falling back to the exact energy-balance path (correct, "
                    "but without the precomputed-LUT speedup).",
                    UserWarning,
                    stacklevel=2,
                )
                self._energy_balance_lut_cache = None
                self._dust_band_response_cache = None

        # Build-time accuracy guard (#617): the photometry LUT bakes the
        # SSP×filter integral at zero dust and re-applies attenuation as a
        # first-order Taylor projection about each filter's effective
        # wavelength. That linear-in-λ model breaks down where the attenuation
        # curve is steep across the bandpass — the rest-UV — so blue bands at
        # moderate/high z are biased silently (the far-UV by >10×). Warn loudly
        # so no fit is biased without the astronomer knowing. Cheap: no SED
        # evaluation, so it is safe on every build.
        self._warn_if_wave_precomp_dust_blue_bias()

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

        The photometry LUT re-applies dust as a first-order Taylor expansion of
        the attenuation about each filter's effective wavelength. That linear
        model is accurate where the attenuation curve is smooth across the
        bandpass (optical/IR) but biases bands sampling the rest-UV, where the
        curve is steep and extrapolated — silently, and by an order of magnitude
        for far-UV bands at moderate/high redshift. The exact path
        (``approx=None``) and ``SpectrumPrecomp`` (per-pixel attenuation; only a
        small intra-pixel error) are unaffected.

        Configuration-level heuristic — fires only when a photometry LUT, a
        non-trivial dust screen, and at least one rest-UV band coincide. Does
        **not** evaluate the SED, so it is cheap on every build. It flags the
        *risk*; quantify the actual per-band bias by comparing against
        ``approx=None``.
        """
        if not self._approx.get("wave_precomp"):
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
        warnings.warn(
            "WavePrecomp applies dust as a first-order Taylor projection across "
            f"each filter (#617); at z~{z_rep:.2f} these rest-UV band(s) are "
            f"biased versus the exact path: {bands}. The bias grows steeply "
            "toward the far-UV (>10x for the bluest bands at moderate/high z) "
            "and with optical depth. For unbiased blue-band photometry use "
            "approx=None, or validate against it; SpectrumPrecomp is unaffected. "
            "See docs/known_limitations.md.",
            UserWarning,
            stacklevel=2,
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
            # Don't actually set the attribute — it's a property now
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
        Phase 2 of forward-projection unification. Each model gets its own
        NamedTuple class, with fields (and magnitude properties) appearing
        only when the corresponding observation sub-block is configured.
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
                raise TypeError(
                    f"observation must be an Observation instance, got {type(observation)}"
                )
            obs_params = observation.get_all_params()
            if obs_params:
                spec = spec.with_params(**obs_params)
        elif filters is not None:
            from tengri.observation.photometry_config import Photometry

            observation = Observation(photometry=Photometry.from_filter_set(filters))

        return observation, spec

    def _init_ssp(self, spec, ssp_data, csp_integration):
        """Set up SSP grid, CSP integration, and log-age grid."""
        self._met_interp = getattr(spec, "met_interp", "linear")
        self._lgmet_scatter = float(getattr(spec, "lgmet_scatter", 0.1))
        # Redshift-table interpolation mode for free-z inference.
        # "linear" → piecewise-linear (C^0 gradient, kinks at grid nodes).
        # "smooth" → triweight kernel (C^2 gradient) — recommended for NUTS/HMC
        # when redshift is a free parameter. See `interpolate_ztable_smooth`
        # in components/sps/precompute.py.
        self._z_interp = getattr(spec, "z_interp", "linear")

        self.ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        self.ssp_ages_yr = 10.0**self.ssp_log_ages_yr

        _valid_csp = ("trapz", "log_trapz", "log_interp", "dsps_native", "dsps_met_table")
        if csp_integration not in _valid_csp:
            raise ValueError(
                f"csp_integration must be one of {_valid_csp}, got {csp_integration!r}"
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
        # n_grid-invariant (the SFH×SSP integral converges by ~64 points — see
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

        # Warn if any burst-width SFH parameter is narrower than the
        # local SSP grid spacing at the burst peak — see #299. The
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
            delta.update(_EVOLVING_ALPHA_PARAM_MAP)

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
        bracket — so an out-of-range value silently clamps to the edge and
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
                    import warnings

                    warnings.warn(
                        f"{name}={val:.3f} is outside the SSP grid metallicity "
                        f"range [{grid_lo_zsol:.3f}, {grid_hi_zsol:.3f}] "
                        f"log10(Z/Zsun) (absolute grid log10(Z) ∈ "
                        f"[{lgmet.min():.3f}, {lgmet.max():.3f}]). The forward "
                        f"model silently clips out-of-range values, producing "
                        f"a misleadingly smooth SED (issue #442). Either set "
                        f"{name} inside the grid, or load an SSP whose grid "
                        f"covers your target metallicity.",
                        UserWarning,
                        stacklevel=3,
                    )
            else:
                lo, hi = float(dist.lo), float(dist.hi)
                if lo < grid_lo_zsol or hi > grid_hi_zsol:
                    import warnings

                    warnings.warn(
                        f"{name} prior bounds [{lo:.3f}, {hi:.3f}] extend "
                        f"beyond the SSP grid metallicity range "
                        f"[{grid_lo_zsol:.3f}, {grid_hi_zsol:.3f}] "
                        f"log10(Z/Zsun). Samples outside the grid will "
                        f"silently clip to the edge — a MAP/MCMC chain "
                        f"wandering there registers a fake likelihood "
                        f"maximum (issue #442). Tighten the prior to within "
                        f"the grid range or load an SSP with broader coverage.",
                        UserWarning,
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

        # WG00 (dust_type=3) structural selectors — static strings threaded into
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

        self._dust_emission_model = getattr(spec, "dust_emission", None)
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
            # ``dl14`` alias — the canonical name was missing, so a model built
            # with emission='draine_li2014' was never preloaded and lazy-loaded
            # its templates inside the JIT trace (UnexpectedTracerError).
            {"draine_li2007", "dl14", "draine_li2014", "dale2014", "astrodust", "bosa", "themis"}
        )
        if self._dust_emission_model in _TEMPLATE_BASED_EMISSION_MODELS:
            from tengri.components.dust.emission import preload_emission_model

            with contextlib.suppress(Exception):
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
            # explicitly added to the spec — Parameters mirrors the same
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
            # whose ``_nebular_backend`` was the wrong class — every line
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
            agn_frac_dist = agn_dists.get("agn_frac")
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
                with contextlib.suppress(Exception):
                    from tengri.components.agn.blocks.blr_blocks import (
                        _resolve_synthesizer_grid as _resolve_blr_grid,
                    )
                    from tengri.components.agn.blocks.nlr_blocks import (
                        _resolve_synthesizer_grid as _resolve_nlr_grid,
                    )
                    from tengri.components.agn.nlr_cloudy import (
                        get_synthesizer_blr_backend,
                        get_synthesizer_nlr_backend,
                    )

                    if self._agn_nlr_block in ("synthesizer", "synthesizer_spectra"):
                        get_synthesizer_nlr_backend(_resolve_nlr_grid("nlr"))
                    if self._agn_blr_block in ("synthesizer", "synthesizer_spectra"):
                        get_synthesizer_blr_backend(_resolve_blr_grid("blr"))

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

        # Identity entries for shock_* and xray_* now come from registry
        # auto-derive in _build_param_map (Step B).
        self._uses_shock = getattr(spec, "shock", False)
        # Composable-shock config (#851): normalization mode + categorical
        # MAPPINGS knobs, threaded to the ShockNebular component config.
        self._shock_norm = getattr(spec, "shock_norm", "frac")
        self._shock_abundance = getattr(spec, "shock_abundance", "solar")
        self._shock_component = getattr(spec, "shock_component", "combined")

        return delta

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
        param_map_deltas : list[dict[str, tuple[str, float, float]]]
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

    def _precompute_dust_ir_photometry(self):
        """Precompute dust IR template photometry for fast hybrid kernel lookup.

        Delegates to the Precompute Protocol adapter at
        :mod:`tengri.components.dust.dust_emission_precompute` (for template-based
        models) or :mod:`tengri.components.dust.dust_analytic_precompute` (for
        analytic models), which handle template loading / model evaluation, filter
        preintegration, and (per the Protocol) auto-collapse-on-Fixed for any
        ``AXIS_PARAMS`` marked :class:`Fixed` in ``self.spec``.  Returns ``None``
        when template data is not available on disk — callers fall back to
        full-wavelength evaluation.

        Returns
        -------
        tuple (lookup, grid_arrays) or (None, None)
            Tuple of (JIT-compiled lookup, JIT-traceable grid arrays).
            For template-based models, grid_arrays is a tuple of arrays that
            can be passed as traced inputs to the lookup function.
            For analytic models or data-missing cases, returns (None, None).
        """
        import warnings

        model_name = self._dust_emission_model

        # Try template-based models first (DL07, Dale2014, etc.)
        try:
            from tengri.components.dust.dust_emission_precompute import (
                build_lookup as build_lookup_template,
                extract_grid_arrays,
                precompute_for_model,
            )

            precomp = precompute_for_model(
                model_name,
                filter_waves=self.filter_waves,
                filter_trans=self.filter_trans,
                redshift=float(self._z_fixed) if self._z_fixed is not None else 0.0,
                parameters=self.spec,
            )
            if precomp is not None:
                lookup = build_lookup_template(precomp, model_name=model_name)
                grid_arrays = extract_grid_arrays(precomp, model_name=model_name)
                return lookup, grid_arrays
        except Exception as e:
            warnings.warn(
                f"Failed to precompute dust IR photometry (template path) for "
                f"{model_name}: {e}. Trying analytic path.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Try analytic models (modified_blackbody, casey2012, pah_drude)
        if model_name in ("modified_blackbody", "casey2012", "pah_drude"):
            try:
                from tengri.components.dust.dust_analytic_precompute import (
                    build_lookup as build_lookup_analytic,
                    precompute,
                )

                precomp = precompute(
                    self.filter_waves,
                    self.filter_trans,
                    redshift=float(self._z_fixed) if self._z_fixed is not None else 0.0,
                    parameters=self.spec,
                    model=model_name,
                )
                if precomp is not None:
                    lookup = build_lookup_analytic(precomp, model=model_name)
                    return lookup, None  # Analytic models don't have grid arrays
            except Exception as e:
                warnings.warn(
                    f"Failed to precompute dust IR photometry (analytic path) for "
                    f"{model_name}: {e}. Falling back to full-wavelength evaluation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None, None

        return None, None

    def _precompute_dust_analytic_photometry(self, model_name: str):
        """Build preintegrated lookup for a specific analytic dust model.

        Parameters
        ----------
        model_name : str
            One of "modified_blackbody", "casey2012", "pah_drude".

        Returns
        -------
        object or None
            JIT-compiled lookup callable or None if precompute unavailable.
        """
        import warnings

        if model_name not in ("modified_blackbody", "casey2012", "pah_drude"):
            return None

        try:
            from tengri.components.dust.dust_analytic_precompute import (
                build_lookup as build_lookup_analytic,
                precompute,
            )

            precomp = precompute(
                self.filter_waves,
                self.filter_trans,
                redshift=float(self._z_fixed) if self._z_fixed is not None else 0.0,
                parameters=self.spec,
                model=model_name,
            )
            if precomp is not None:
                return build_lookup_analytic(precomp, model=model_name)
        except Exception as e:
            warnings.warn(
                f"Failed to precompute dust analytic photometry for {model_name}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        return None

    def _warm_grid_caches(self) -> None:
        """Warm @functools.cache loaders to avoid tracer leaks from HDF5 grids.

        Some HDF5 grid loaders (@functools.cache decorators) construct jnp.array
        objects at load time. If the first call happens inside a JAX JIT trace,
        the jnp.array calls create DynamicJaxprTracers, which get cached and
        permanently leak into downstream code. This method calls each loader once
        OUTSIDE a JIT context so the cache stores concrete arrays instead.

        Wrapped in try/except because grid files are optional — missing files
        should not block SEDModel construction.
        """
        # MAPPINGS shock emission grids (nebular/shock.py:_load_mappings_grids)
        if self._uses_shock:
            try:
                from tengri.components.nebular.shock import _load_mappings_grids

                _load_mappings_grids()
            except Exception:
                pass

        # CAT3D-Wind AGN torus grids (agn/cat3d_wind.py:_load_cat3d_default)
        if self._agn_model == "cat3d_wind":
            try:
                from tengri.components.agn.cat3d_wind import _load_cat3d_default

                _load_cat3d_default()
            except Exception:
                pass

        # Astrodust+PAH emission grid (dust/emission/templates/astrodust.py).
        # The faithful HD23 port self-loads its grid in load()/predict(); warm
        # the process cache OUTSIDE any trace so an in-trace lazy load hits
        # concrete arrays rather than leaking a tracer (UnexpectedTracerError).
        # The grammar builds the port with the default template path (None).
        if getattr(self, "_dust_emission_model", None) == "astrodust":
            try:
                from tengri.components.dust.emission.templates.astrodust import (
                    _cached_astrodust_grid,
                )

                _cached_astrodust_grid(None)
            except Exception:
                pass

    def ensure_photometry_precomputed(self) -> bool:
        """Deprecated no-op (the legacy ``PrecomputedData`` container was retired, #620).

        The photometry fast path is now opt-in at build time via
        ``approx=WavePrecomp()`` and is built through the component chain, not
        lazily here. Retained as a no-op so existing callers (e.g. the Fitter,
        whose return value was already discarded) keep working.

        Returns
        -------
        bool
            Always ``False`` (nothing is built lazily).
        """
        return False

    def _get_internal_params(self, params):
        """Translate public param dict to internal names with unit conversion.

        Thin wrapper around :func:`tengri._param_translate.get_internal_params`.
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

        Single dispatch point for all SFH computation — replaces
        the old stochastic/parametric if/else branches.

        Parameters
        ----------
        p : dict
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
            )
            kw["gp_x"] = gp_x
            kw["k0_half"] = k0_half

        return self._sfh_fn(self.age_yr, **kw)

    def _compute_sfr_mean_and_full(self, p):
        """Compute both mean (no GP) and full (with GP) SFR.

        Used by predict_sfh which needs to return both.

        Returns
        -------
        sfr_mean : array
            SFR without GP modulation.
        sfr_full : array
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
            )
            kw["gp_x"] = gp_x
            kw["k0_half"] = k0_half
            sfr_full = self._sfh_fn(self.age_yr, **kw)
        else:
            sfr_full = sfr_mean

        return sfr_mean, sfr_full

    def _get_non_stellar_kwargs(self, p):
        """Extract non-stellar kwargs from internal params for hybrid kernel."""
        kw = {}
        # Nebular
        if self._nebular_backend is not None and getattr(
            self._nebular_backend, "has_free_params", False
        ):
            kw["neb_logU"] = p.get("neb_logU", -3.0)
            kw["neb_logZ_gas"] = p.get("neb_logZ_gas", None)
            kw["neb_fesc"] = p.get("neb_fesc", 0.0)
            kw["neb_fesc_lya"] = p.get("neb_fesc_lya", 0.0)
        # Shock
        if self._uses_shock:
            kw["shock_frac"] = p.get("shock_frac", 0.0)
            kw["shock_velocity"] = p.get("shock_velocity", 300.0)
            kw["shock_log_density"] = p.get("shock_log_density", 0.0)
            kw["shock_b_over_sqrt_n"] = p.get("shock_b_over_sqrt_n", 1.0)
        # Dust emission (all models, not just MBB)
        if self._dust_emission_model is not None:
            kw["dust_T"] = p.get("dust_T", 35.0)
            kw["dust_beta_ir"] = p.get("dust_beta_ir", 1.6)
            kw["dust_eta_balance"] = p.get("dust_eta_balance", 1.0)
            kw["dust_alpha_mir"] = p.get("dust_alpha_mir", 2.0)
            kw["dust_alpha_dale"] = p.get("dust_alpha_dale", 2.0)
            kw["dust_umin"] = p.get("dust_umin", 1.0)
            kw["dust_gamma_dl"] = p.get("dust_gamma_dl", 0.01)
            kw["dust_qpah"] = p.get("dust_qpah", 2.5)
            kw["dust_lgU"] = p.get("dust_lgU", 0.0)
            # THEMIS qhac + radiation-field slope alpha, Dale AGN fraction,
            # Schreiber PAH fraction and MBB epsilon — forwarded so they reach
            # the emission kernel (CIGALE parity, 2026-06). Schreiber temperature
            # rides the canonical ``dust_T`` (forwarded above); ``dust_f_pah`` is
            # the canonical PAH-fraction name (#849).
            kw["dust_qhac"] = p.get("dust_qhac", 0.17)
            kw["dust_alpha"] = p.get("dust_alpha", 2.0)
            kw["dust_frac_agn"] = p.get("dust_frac_agn", 0.0)
            kw["dust_f_pah"] = p.get("dust_f_pah", 0.05)
            kw["dust_epsilon_mbb"] = p.get("dust_epsilon_mbb", 1.0)
        # AGN (full params for exact evaluation)
        if self._agn_model is not None:
            kw["agn_polar_ebv"] = p.get("agn_polar_ebv", 0.0)
            kw["agn_cos_inc"] = p.get("agn_cos_inc", 0.5)
            kw["agn_polar_oa"] = p.get("agn_polar_oa", 45.0)
            kw["agn_frac"] = p.get("agn_frac", 1.0)
            kw["agn_a_spin"] = p.get("agn_a_spin", 0.0)
            kw["agn_log_mbh"] = p.get("agn_log_mbh", 7.0)
            kw["agn_log_ledd"] = p.get("agn_log_ledd", -1.0)
            # K&D 3-zone disc params
            kw["agn_f_hard"] = p.get("agn_f_hard", 0.02)
            kw["agn_gamma_warm"] = p.get("agn_gamma_warm", 2.5)
            kw["agn_kt_warm"] = p.get("agn_kt_warm", 0.2)
            kw["agn_gamma_hard"] = p.get("agn_gamma_hard", 1.8)
            kw["agn_kt_hot"] = p.get("agn_kt_hot", 100.0)
            kw["agn_r_warm_ratio"] = p.get("agn_r_warm_ratio", 2.0)
            # Two-temperature torus
            kw["agn_T_hot"] = p.get("agn_T_hot", 1200.0)
            kw["agn_T_warm"] = p.get("agn_T_warm", 300.0)
            kw["agn_frac_hot"] = p.get("agn_frac_hot", 0.3)
            # SKIRTOR torus
            kw["agn_tau_skirtor"] = p.get("agn_tau_skirtor", 7.0)
            kw["agn_p_skirtor"] = p.get("agn_p_skirtor", 1.0)
            kw["agn_q_skirtor"] = p.get("agn_q_skirtor", 1.0)
            kw["agn_oa_skirtor"] = p.get("agn_oa_skirtor", 40.0)
            # SKIRTOR_mean_3p (AGNfitter-rX averaged) torus
            kw["agn_incl_skirtor"] = p.get("agn_incl_skirtor", 30.0)
            kw["agn_tv_skirtor"] = p.get("agn_tv_skirtor", 7.0)
            # CAT3D-Wind clumpy torus (Hönig & Kishimoto 2017)
            kw["agn_a_cat3d"] = p.get("agn_a_cat3d", -2.0)
            kw["agn_fwd_cat3d"] = p.get("agn_fwd_cat3d", 0.2)
            # Silva+04 obscured-torus column density
            kw["agn_log_nh_silva"] = p.get("agn_log_nh_silva", 23.0)
            # Fritz+2006 torus (CIGALE fritz2006) — forwarded so the 6D grid
            # block sees its parameters under the composable AGN path (#347).
            kw["agn_fritz_r_ratio"] = p.get("agn_fritz_r_ratio", 60.0)
            kw["agn_fritz_tau"] = p.get("agn_fritz_tau", 1.0)
            kw["agn_fritz_beta"] = p.get("agn_fritz_beta", -0.5)
            kw["agn_fritz_gamma"] = p.get("agn_fritz_gamma", 4.0)
            kw["agn_fritz_oa"] = p.get("agn_fritz_oa", 60.0)
            kw["agn_fritz_psy"] = p.get("agn_fritz_psy", 0.001)
            # Nenkova+2008 CLUMPY torus (FSPS/Prospector)
            kw["agn_tau"] = p.get("agn_tau", 30.0)
        # Radio
        if self._uses_radio:
            kw["radio_loudness"] = p.get("radio_loudness", 0.0)
            kw["log_mstar"] = jnp.log10(jnp.maximum(p.get("mstar", 1e10), 1e-10))
        return kw

    # ── Predictions (public API) ──────────────────────────────────────

    def predict_sfh(self, params, n_linear=1000):
        """Compute SFH on uniform linear-time grid for visualization.

        Evaluates the SFH parameterization at ``n_linear`` evenly-spaced
        points in lookback time, returning both the smooth parametric
        component (``sfr_mean``) and the full SFH including GP-field
        modulation (``sfr_full``, if stochastic SFH enabled).

        **Raw forward-pass output** intended for plotting. For SFH-derived
        scalars (stellar mass, recent SFR, age), see
        ``model.predict(params).sfh.*`` or :meth:`predict_sfh_quantities`
        for the JIT-compatible form.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.
        n_linear : int, optional
            Number of output grid points, evenly spaced in lookback time.
            Default 1000 (sufficient for smooth visualization).

        Returns
        -------
        dict with keys:

            - ``"t_gyr"`` : array, shape (n_linear,).
              Lookback time [Gyr], from 0 (now) to ~13.8 (Big Bang).
            - ``"sfr_mean"`` : array, shape (n_linear,).
              Parametric mean SFR [M☉/yr] (no GP modulation).
            - ``"sfr_full"`` : array, shape (n_linear,).
              Full SFH including GP field [M☉/yr]. Identical to ``sfr_mean``
              if stochastic SFH not enabled.

        Notes
        -----
        **JIT-compatible**: no — uses Python-side interpolation. For
        JIT-compatible SFH evaluation, use :meth:`predict_sfh_quantities`
        to get integrated quantities (stellar mass, age, etc.).

        **Time grid**: Output is on a uniform linear-time (lookback) grid,
        not the internal log-age grid. This makes visualization cleaner
        and suitable for plotting.

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
        predict_sfh_quantities : Integrated SFH quantities (JIT-compatible).
        predict : Lazy access to SFH and all derived quantities.
        """
        p = self._get_internal_params(params)
        sfr_mean, sfr_full = self._compute_sfr_mean_and_full(p)

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
        params : dict
            Parameter values using public parameter names.
        wave : array, optional
            Custom rest-frame wavelength grid [Angstrom]. If None,
            uses the model's default: SSP wavelength grid
            (``ssp_data.ssp_wave``), or auto-extended grid if
            ``radio=True`` or ``xray=True`` in spec.

        Returns
        -------
        SEDResult
            NamedTuple with:

            - ``wavelength`` : array, shape (n_wave,). Rest-frame wavelength [Ångstrom]
            - ``sed`` : array, shape (n_wave,). Spectral luminosity density [erg/s/Hz]

        Notes
        -----
        **JIT-compatible**: no — computes SED components via the
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
        >>> sed = model.predict_rest_sed(params)
        >>> import matplotlib.pyplot as plt
        >>> plt.loglog(sed.wavelength, sed.sed)
        >>> plt.xlabel("Rest-frame wavelength (Angstrom)")
        >>> plt.ylabel("SED (erg/s/Hz)")

        See Also
        --------
        predict_obs_sed : Observed-frame SED (redshifted + IGM).
        predict_sed_quantities : JIT-compatible SED-derived quantities.
        """
        from tengri.forward.result import SEDResult

        state = self.predict_state(params)
        if wave is None:
            # Use ``state.wave`` (the orchestrator's runtime wavelength
            # grid, which may differ from ``self._rest_wavelength`` —
            # e.g. when radio/xray extends the SSP grid panchromatically
            # but the orchestrator hasn't been wired to that extension
            # yet). Mismatched shapes would otherwise break boolean
            # masking on (wavelength, sed) pairs in test_panchromatic_*.
            return SEDResult(wavelength=state.wave, sed=state.sed_intrinsic)
        # Custom rest-frame wavelength grid: interpolate the orchestrator's
        # SED onto it. Pure post-processing — keeps the orchestrator's
        # internal grid contract (state.wave / state.derived[...]) clean,
        # at the same accuracy a user gets from
        # ``np.interp(custom_wave, ssp_wave, sed)``.
        wave_target = jnp.asarray(wave)
        sed_interp = jnp.interp(wave_target, state.wave, state.sed_intrinsic)
        return SEDResult(wavelength=wave_target, sed=sed_interp)

    def predict_obs_sed(self, params, wave=None):
        """Compute observed-frame SED (redshifted + IGM + DLA transmission).

        Evaluates the rest-frame SED, redshifts to observed frame
        (wavelength × (1+z)), and applies IGM and DLA absorption where
        configured. At z=0, identical to :meth:`predict_rest_sed`.

        **Raw forward-pass output.** Returns ``(wavelength, sed)`` in the
        observed frame. For interactive use with derived quantities, see
        :meth:`predict`.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.
        wave : array, optional
            Custom rest-frame wavelength grid [Angstrom] before redshifting.
            If None, uses model default.

        Returns
        -------
        SEDResult
            NamedTuple with:

            - ``wavelength`` : array, shape (n_wave,).
              Observed-frame wavelength [Ångstrom]
            - ``sed`` : array, shape (n_wave,).
              Observed-frame spectral luminosity density [erg/s/Hz]

        Notes
        -----
        **JIT-compatible**: no — delegates to :meth:`predict_rest_sed`.

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
        >>> sed_obs = model.predict_obs_sed(params)
        >>> # IGM and redshift already applied
        >>> print(f"z={params['redshift']}: wavelength {sed_obs.wavelength[0]:.0f} Å")

        See Also
        --------
        predict_rest_sed : Rest-frame SED (before redshift/IGM).
        predict_photometry : Filter-integrated observed flux (uses this internally).

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

        rest_result = self.predict_rest_sed(params, wave=wave)
        z = self._get_redshift(params)
        wave_obs = rest_result.wavelength * (1.0 + z)
        sed_obs = rest_result.sed
        if self._uses_igm:
            from tengri.forward.emission_helpers import igm_absorption

            # Always apply IGM when enabled — igm_transmission returns
            # all-ones at z=0. Avoid z>0 comparison which fails under JIT.
            igm_trans = igm_absorption(
                wave_obs,
                z,
                igm_x_HI=params.get("igm_x_HI", 0.0),
                igm_bubble_mpc=params.get("igm_bubble_mpc", 10.0),
                igm_patchy=getattr(self, "_igm_patchy", False),
                igm_model=self._igm_model,
            )
            sed_obs = sed_obs * igm_trans
        if self._uses_dla:
            from tengri.components.igm.dla import dla_transmission_obs

            z_dla = params.get("dla_z", 0.0)
            z_dla = jnp.where(z_dla > 0.0, z_dla, z)
            sed_obs = sed_obs * dla_transmission_obs(
                wave_obs,
                z_dla=z_dla,
                log_n_hi=params.get("dla_log_n_hi", 20.0),
                temp=params.get("dla_temp", 1e4),
                b_turb_kms=params.get("dla_b_turb", 0.0),
            )
        # MW foreground screen (#297) — final transformation on the
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

        For batch computation over posterior chains or mock catalogs,
        use the JIT-compatible methods :meth:`predict_sfh_quantities`,
        :meth:`predict_sed_quantities`, or :meth:`predict_line_luminosities`
        with :func:`jax.vmap` instead (up to 1000× faster for large batches).

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.

        Returns
        -------
        Prediction
            Lazy caching wrapper with property groups:

            - ``.sfh`` : SFH-derived quantities (stellar mass, SFR, age, metallicity)
            - ``.sed`` : SED-derived quantities (luminosities, colors, indices)
            - ``.lines`` : Emission line properties (luminosities, fluxes, ratios)
            - ``.radio`` : Radio SED properties (if ``radio=True``)
            - ``.xray`` : X-ray SED properties (if ``xray=True``)
            - ``.ionizing`` : Ionizing photon budget properties

        Notes
        -----
        **Not JIT-compatible**: Uses Python-side caching and object
        attribute access. Useful for interactive exploration, not
        for inference loops. For inference, use
        :meth:`predict_sfh_quantities`, :meth:`predict_sed_quantities`,
        etc. with :func:`jax.vmap`.

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
        >>> sfh_fn = jax.vmap(model.predict_sfh_quantities)
        >>> sfh_batch = sfh_fn(params_batch)
        >>> sfh_batch.stellar_mass  # shape (10000,)
        >>> sfh_batch.stellar_mass.mean()

        See Also
        --------
        predict_sfh_quantities : JIT-compatible SFH quantities for batch.
        predict_sed_quantities : JIT-compatible SED quantities for batch.
        predict_line_luminosities : JIT-compatible emission lines for batch.
        predict_rest_sed : Full rest-frame SED for custom analysis.
        """
        from tengri.forward.prediction import Prediction

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

        # Nebular backend (by class name)
        nebular_backend_name = (
            type(self._nebular_backend).__name__ if self._nebular_backend is not None else "none"
        )

        # IGM configuration
        uses_igm = bool(self._uses_igm)
        igm_model = str(self._igm_model or "none")
        uses_dla = bool(self._uses_dla)

        # AGN configuration
        agn_model = str(self._agn_model or "none")
        agn_luminosity_mode = bool(self._agn_luminosity_mode)

        # Radio and X-ray
        uses_radio = bool(self._uses_radio)
        uses_xray = bool(self._uses_xray)
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

        # Alpha-Fe evolution
        alpha_fe_evolving = bool(self._alpha_fe_evolving)

        # Redshift configuration. The actual fixed-z value is part of the
        # structural fingerprint because the compiled kernels close over
        # ``_dl_cm_fixed``, ``_igm_fn`` precomputed tables, and effective
        # rest wavelengths — all derived from ``_z_fixed`` at construction.
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
        else:
            spec_wave_shape = ()
            sigma_lib_kms = 0.0
            lsf_resolution = None

        # CSP integration method
        csp_integration = str(self._csp_integration)

        # Forward dtype
        forward_dtype = str(self._forward_dtype)

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

        # Compile mode (Phase 2)
        compile_mode = str(self._compile_mode)

        # Approximation settings (Phase 2), resolved and sorted.
        # Phase 3d (2026-05-20): include the resolved WavePrecomp configuration
        # so two models with different ztable sampling (n_z / z_min / z_max)
        # get distinct cache slots. Without this, ``WavePrecomp(n_z=100)`` and
        # ``WavePrecomp(n_z=200)`` would collide and the second galaxy would
        # reuse the first's stale compiled LUT.
        approx_resolved_flags = tuple(
            sorted((k, bool(v)) for k, v in (self._approx or {}).items() if isinstance(v, bool))
        )

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
            )
        else:
            approx_resolved = approx_resolved_flags

        # Fixed-parameter values from spec. The compositional/hybrid kernels
        # capture self via closure at build time, so two models with identical
        # *structural* signature but different Fixed defaults must NOT share
        # a cached kernel — otherwise the first-built kernel's Fixed values
        # leak into the second model's predict_photometry({}) call. See
        # https://github.com/<repo>/issues/<n> for the starburst-vs-quenched
        # color-leak symptom (the second-built model returned the first
        # model's u-g color at z=0.05).
        def _fixed_value_id(name: str):
            dist = self.spec.get_distribution(name)
            val = dist.bounds[0] if dist.bounds is not None else getattr(dist, "value", None)
            if val is None:
                return ("none",)
            if isinstance(val, str):
                return ("str", val)
            try:
                return ("num", round(float(val), 12))
            except (TypeError, ValueError):
                return ("repr", repr(val))

        # Phase 4-A (2026-05-20): drop fixed-parameter VALUES from the
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

        return (
            ssp_flux_shape,
            ssp_lgmet_shape,
            ssp_lgmet_id,
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
            nebular_backend_name,
            uses_igm,
            igm_model,
            uses_dla,
            agn_model,
            agn_luminosity_mode,
            uses_radio,
            uses_xray,
            uses_shock,
            shock_cfg,
            mean_sfh_type,
            met_mode,
            stochastic,
            n_grid,
            alpha_fe_evolving,
            z_fixed,
            catalog_z_range,
            has_spectroscopy,
            spec_wave_shape,
            sigma_lib_kms,
            lsf_resolution,
            csp_integration,
            forward_dtype,
            met_interp,
            z_interp,
            radio_include_freefree,
            radio_sfr_mode,
            radio_agn_model,
            has_sigma_v,
            compile_mode,
            approx_resolved,
            spec_fixed_id,
        )

    def predict_photometry(self, params):
        """Compute observed photometric flux densities through all filters.

        Convolves the SED (redshifted and IGM-absorbed) through filter
        transmission curves, returning flux densities in the AB system
        at the source. Routes through :meth:`predict_observables_jit`,
        the JIT-safe orchestrator with SSP threading (Phase 4-B).

        **Raw forward-pass output.** For interactive use with cached
        derived quantities, see ``model.predict(params).photometry``.
        For batched photometry over posterior chains, use
        :meth:`predict_photometry_batch`.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names (e.g.,
            ``sfh_tsnorm_log_total_mass``, ``met_logzsol``, ``redshift``).
            See :class:`Parameters` for canonical names.

        Returns
        -------
        flux_density : array, shape (n_filters,)
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

        **Approximation accuracy**: Driven by the build-time ``approx=``
        policy. :class:`WavePrecomp` swaps in the SSP×filter LUT, which is
        ~0.4 % accurate for the *stellar* photometry (Zacharegkas+2025 [1]_)
        **but re-applies dust as a first-order Taylor projection across each
        filter (#617)**. That linear-in-λ model is accurate in the optical/IR,
        where the attenuation curve is smooth across a band, but biases bands
        sampling the **rest-UV** (steep, extrapolated attenuation) — by an
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
        predict : Lazy prediction object for all derived quantities.
        predict_spectrum : Spectral flux at arbitrary wavelengths.
        predict_magnitudes : AB magnitudes (uses photometry internally).

        Examples
        --------
        >>> flux = model.predict_photometry(params)
        >>> mags = model.predict_magnitudes(params)
        >>> # For the fast LUT path, build with ``approx=WavePrecomp()``.

        References
        ----------
        .. [1] A. Zacharegkas et al., "Fast Photometry with Precomputed
           Stellar Population Grids," ApJ, (2025).
        """
        if self.filter_waves is None:
            raise ValueError("No filters set. Pass filters or observation= to SEDModel().")
        return self.predict_observables_jit(params).phot_fnu

    def predict_spectrum(
        self,
        params,
        wave_obs=None,
        wave_chunk_size=None,
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
        params : dict
            Parameter values using public parameter names.
        wave_obs : array, optional
            Observed-frame wavelength grid [Angstrom]. If None, uses:

            1. The grid bound at construction from
               ``observation.spectroscopy.wave_obs``
            2. Raises ValueError if no grid is available

        wave_chunk_size : int, optional
            If specified, split observed-frame wavelength axis into chunks of
            this size and evaluate via ``jax.lax.map`` to reduce per-chunk HLO
            size for XLA compilation. Default None (no chunking, exact behavior).
            For spectroscopy with R~500 at N≥64 galaxies, typical value is 32–64
            to avoid XLA compilation wall-clock.

        Returns
        -------
        flux : array, shape (n_pix,)
            Observed spectral flux density [erg/s/cm²/Hz] in the AB system
            at the specified wavelengths.

        Raises
        ------
        ValueError
            If ``wave_obs`` is None and no precomputed wavelength grid available.

        Notes
        -----
        **JIT-compatible**: yes — routes through
        :meth:`predict_observables_jit` (Phase 4-B orchestrator).

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
        predict_photometry : Filter-integrated flux (simpler, faster).
        predict : Lazy access to all SED and SFH quantities.
        predict_photometry : Filter-integrated flux (simpler, faster).
        """
        # A caller-supplied ``wave_obs`` is evaluated directly on that grid,
        # independent of any configured spectroscopy channel or the cached
        # ``predict_observables`` (which is photometry-only on a photometry
        # model and previously raised ``'Observables' has no attribute
        # 'spec_fnu'`` after a ``predict_photometry``/fit call). This makes the
        # documented ``wave_obs`` argument do what it says — give me the model
        # spectrum on this grid — on any model and in any call order
        # (suchethac/tengri#707).
        if wave_obs is not None:
            return self._predict_spectrum_on_grid(params, jnp.asarray(wave_obs), wave_chunk_size)

        # No explicit grid: a configured spectroscopy channel routes through the
        # orchestrator — the JIT/grad/LUT-friendly cached path the Fitter relies
        # on (honors the SpectrumPrecomp LUT, LSF and any calibration). This is
        # the inference hot path and must stay on predict_observables.
        if (
            self.observation is not None
            and getattr(self.observation, "spectroscopy", None) is not None
            and getattr(self.observation.spectroscopy, "wave_obs", None) is not None
        ):
            del wave_obs, wave_chunk_size
            return self.predict_observables_jit(params).spec_fnu

        # No spectroscopy channel but a manually attached grid (``model._wave_obs``)
        # — evaluate directly so photometry-only models with an ad-hoc grid work
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
        (:func:`~tengri.observation.spectrum.compute_spectrum`) the configured
        spectroscopy path uses. The instrument LSF is applied only when the
        attached observation declares a spectroscopic resolution. Underpins the
        ``wave_obs`` argument of :meth:`predict_spectrum` (suchethac/tengri#707).

        Parameters
        ----------
        params : dict
            Public-name parameter values.
        wave_obs : array_like, shape (n_pix,)
            Observed-frame wavelength grid [Angstrom].
        wave_chunk_size : int, optional
            Currently advisory on this direct path — the per-pixel projection is
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
        from tengri.observation.spectrum import apply_lsf, compute_spectrum

        sed_obs = self.predict_obs_sed(params)
        z = self._get_redshift(params)
        dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
        wave_rest = sed_obs.wavelength / (1.0 + z)
        flux = compute_spectrum(sed_obs.sed, wave_rest, wave_obs, z, dl_cm)

        spectroscopy = (
            getattr(self.observation, "spectroscopy", None) if self.observation else None
        )
        if spectroscopy is not None and getattr(spectroscopy, "resolution", None) is not None:
            flux = apply_lsf(
                flux,
                wave_obs,
                spectroscopy.resolution,
                sigma_lib_kms=spectroscopy.sigma_lib_kms,
                sigma_v_kms=params.get("sigma_v_kms", 0.0),
            )
        return flux

    def predict_magnitudes(self, params):
        """Compute observed AB magnitudes through all filters.

        **Raw forward-pass output.** For interactive use, see
        ``model.predict(params).magnitudes``.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.

        Returns
        -------
        magnitudes : ndarray, shape (n_filters,)
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
        params : dict
            Parameter values using public parameter names.

        Returns
        -------
        array, shape (n_wave,)
            Rest-frame luminosity [L_sun/Hz].

        Notes
        -----
        **JIT-compatible**: no — wraps :meth:`predict_rest_sed`.

        Divides rest-frame SED by :math:`L_{\\odot} = 3.828 \\times 10^{33}` erg/s
        (IAU 2015 definition).
        """
        from tengri.utils.physics_constants import L_SUN

        sed_erg = self.predict_rest_sed(params).sed
        return sed_erg / L_SUN

    def predict_line_fluxes(self, params, target_wavelengths=None, tolerance_aa=5.0):
        """Predict observed emission line fluxes.

        Calls the nebular backend to compute line luminosities,
        selects target lines by wavelength matching, and converts
        from luminosity (Lsun) to observed flux (erg/s/cm^2).

        **Raw forward-pass output.** For interactive access to individual
        named lines (with luminosities, ratios, and BPT diagnostics), see
        ``model.predict(params).lines.halpha`` etc.

        Parameters
        ----------
        params : dict
            Parameter values (public names).
        target_wavelengths : array, shape (n_target,), optional
            Rest-frame vacuum wavelengths (Angstrom) of lines to predict.
            Each wavelength is matched to the nearest backend line.
            If None, returns all lines from the nebular backend.
        tolerance_aa : float or None, default 5.0
            Maximum allowed wavelength delta [Angstrom] between a requested
            target and the matched catalog line. Raises ``ValueError`` on
            any miss, listing the offending targets. Pass ``None`` to disable
            (recovers legacy nearest-line-no-matter-what behavior).

        Returns
        -------
        fluxes : array, shape (n_target,) or (n_all_lines,)
            Observed line fluxes in erg/s/cm^2.

        Raises
        ------
        ValueError
            If no nebular backend is configured.

        Notes
        -----
        **JIT-compatible**: no — delegates to nebular backend.

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

        # Read the discrete line catalog published by
        # NebularSEDComponent. The orchestrator's nebular adapter calls
        # ``predict_nebular_line_luminosities`` with SSP-derived
        # ``ssp_weights`` + ``ssp_log_ages_yr`` and the canonical
        # neb_logZ_gas → absolute-log10(Z) translation.
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
            # (NUTS/HMC) ``min_deltas`` is a Tracer, so skip it there —
            # line matching is structural (static catalog × static
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
        # NebularSEDComponent) — no L_sun conversion here. Multiplying by
        # L_SUN was a 33.6-dex unit error that made every joint
        # photometry+line-flux fit unusable against real data.
        flux = selected_lums / (4.0 * jnp.pi * dl_cm**2)
        return flux

    def predict_line_ratios(self, params, line_ratio_data):
        """Predict emission line ratios for a :class:`LineRatioData` set.

        Computes the model flux ratio ``F(numerator) / F(denominator)`` for
        each requested pair, in the same space (linear or log10) as the data.
        Runs the forward chain **once** and selects both the numerator and
        denominator lines from the published catalog — so the likelihood
        loop pays a single chain evaluation per step, not two.

        Works identically on the exact and SpectrumPrecomp paths: line
        luminosities are grid-independent and survive the LUT path.

        Parameters
        ----------
        params : dict
            Parameter values (public names).
        line_ratio_data : LineRatioData
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
        **JIT-compatible**: no — delegates to the nebular backend via
        :meth:`predict_state`.
        """
        state = self.predict_state(params)
        if "line_waves" not in state.derived or "line_lums" not in state.derived:
            raise ValueError(
                "Configured nebular backend did not publish a discrete line "
                "catalog ('line_waves'/'line_lums'). Use Cue or CloudyGrid; "
                "BakedIn bakes lines into the SSP and cannot report ratios."
            )
        all_waves = jnp.asarray(state.derived["line_waves"])
        all_lums = jnp.asarray(state.derived["line_lums"])
        dl_cm = self._get_dl_cm(params)
        # ``line_lums`` are erg/s (DerivedKey contract) — same fix as
        # ``predict_line_fluxes``. The scale cancels in every ratio, so
        # this is unit hygiene, not a behavior change.
        scale = 1.0 / (4.0 * jnp.pi * dl_cm**2)

        def _match(targets):
            targets = jnp.asarray(targets)
            deltas = jnp.abs(all_waves[None, :] - targets[:, None])
            idx = jnp.argmin(deltas, axis=1)
            return all_lums[idx] * scale

        num_flux = _match(line_ratio_data.numerator_waves)
        den_flux = _match(line_ratio_data.denominator_waves)
        return line_ratio_data.model_ratio(num_flux, den_flux)

    def predict_spectral_indices(self, params, index_defs):
        """Predict spectral index values from the model SED.

        Generates a rest-frame spectrum covering the index wavelength
        ranges and measures each index (EW or break ratio). Suitable
        for JIT/batch loops; for interactive use, access individual
        indices via ``model.predict(params).sed.dn4000`` etc.

        Parameters
        ----------
        params : dict
            Parameter values (public names).
        index_defs : tuple of SpectralIndexDef
            Index definitions to measure.

        Returns
        -------
        jnp.ndarray, shape (n_indices,)
            Predicted index values.

        Notes
        -----
        **JIT-compatible**: yes — routes through
        :meth:`predict_spectrum` → :meth:`predict_observables_jit`.

        Measures spectral indices (equivalent width or break ratio) from a
        rest-frame spectrum covering all wavelength ranges in ``index_defs``.
        """
        from tengri.observation.spectral_indices import measure_index_jax

        # Spectral indices (D4000 / Balmer break / Lick EW) are rest-frame
        # quantities measured on the attenuated galaxy SED. Evaluate the
        # rest-frame SED on the model's native (SSP-resolution) grid and
        # measure each index there — the same source as ``pred.sed.dn4000``.
        #
        # This works on both the exact and SpectrumPrecomp paths: the dust
        # components set ``state.sed_intrinsic`` to the full attenuated SED in
        # all cases (the spectrum LUT only *additionally* publishes per-pixel
        # transmission), so ``predict_rest_sed`` returns the attenuated SED
        # regardless of ``approx``.
        rest = self.predict_rest_sed(params)
        wave_rest, flux_rest = rest.wavelength, rest.sed

        indices = [measure_index_jax(wave_rest, flux_rest, idx_def) for idx_def in index_defs]
        return jnp.array(indices)

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
        params : dict
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
        **JIT-compatible**: no — wraps :meth:`predict_sfh_quantities`.

        Uses Case B recombination coefficients (Leitherer et al. 1999 [1]_).
        If SFH computation fails (e.g., invalid params), returns safe fallback of 1 L_sun.

        See Also
        --------
        predict_sfh_quantities : JIT-compatible SFH quantities including sfr_10myr.

        References
        ----------
        .. [1] C. Leitherer et al., "Starburst99: Synthesis Models for Galaxies
           with Active Star Formation," ApJS, 123, 3 (1999).
           arXiv:astro-ph/9807340.
        """
        # Case B: L_Hbeta [Lsun] = 4.76e-13 * Q_H, Q_H = 4.2e53 * SFR [Msun/yr]
        # => L_Hbeta = 4.76e-13 * 4.2e53 / 3.828e33 * SFR ≈ 5.22e7 * SFR
        _L_HBETA_PER_SFR = 5.22e7  # Lsun per Msun/yr (Leitherer+1999)
        try:
            sfh_q = self.predict_sfh_quantities(params)
            sfr_10 = float(sfh_q.sfr_10myr)
            sfr_10 = max(sfr_10, 1e-10)
            return float(_L_HBETA_PER_SFR * sfr_10)
        except (AttributeError, TypeError, ValueError):
            # AttributeError: predict_sfh_quantities doesn't exist or sfr_10myr missing
            # TypeError: float() conversion failed (JAX tracer or wrong type)
            # ValueError: invalid params
            return 1.0  # 1 Lsun safe fallback

    def predict_derived(self, params):
        """Compute derived physical quantities as a flat dict.

        Convenience wrapper around :meth:`predict` that extracts the key
        SFH-derived scalars into a plain dict. Use :meth:`predict` for
        lazy on-demand access to all quantities, or
        :meth:`predict_sfh_quantities` for JIT-compatible batch computation.

        Parameters
        ----------
        params : dict
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
        **JIT-compatible**: no — wraps :meth:`predict`.

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

    def predict_sfh_quantities(self, params):
        """Compute SFH-derived quantities in JIT-compatible form.

        Integrates the SFH to compute stellar mass, recent SFR, specific SFR,
        and mass-weighted age/metallicity. Returns a :class:`SFHQuantities`
        NamedTuple that is fully JIT-compatible and vmap-ready for batch
        inference over posterior chains or mock catalogs.

        **Use this method for** JIT/batch loops (``jax.vmap``, ``jit``,
        ``grad``). **For interactive single-galaxy exploration**, use
        :meth:`predict` and access ``pred.sfh.stellar_mass`` etc. — same
        quantities, with Python-side caching.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.

        Returns
        -------
        SFHQuantities
            NamedTuple with fields:

            - ``stellar_mass`` : float. Total stellar mass formed [M☉]
            - ``stellar_mass_surviving`` : float. Mass in living stars + remnants [M☉],
              or NaN if SSP mass-remaining tables not loaded.
            - ``sfr_100myr`` : float. SFR time-averaged over last 100 Myr [M☉/yr]
            - ``sfr_10myr`` : float. SFR time-averaged over last 10 Myr [M☉/yr]
            - ``ssfr`` : float. Specific SFR (SFR/M_surv or SFR/M_formed) [yr⁻¹]
            - ``mass_weighted_age_gyr`` : float. Mass-weighted age [Gyr]
            - ``mass_weighted_metallicity`` : float. Mass-weighted log₁₀(Z/Z☉) or
              absolute log₁₀(Z) depending on metallicity mode

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.
        Safe inside :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`.

        **Gradient-safe**: yes — all quantities are differentiable w.r.t.
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

        >>> sfh = model.predict_sfh_quantities(params)
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
        predict : Lazy prediction for single-galaxy exploration (non-JIT).
        predict_sfh : SFH on linear-time grid for visualization.
        predict_sed_quantities : JIT-compatible SED quantities.
        """
        from tengri.forward.prediction import SFHQuantities
        from tengri.utils.sed_quantities import (
            compute_mass_weighted_age,
            compute_mass_weighted_metallicity,
        )

        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)

        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        if self._csp_integration == "log_interp":
            weights = self._csp_matrix @ sfr_on_ssp
        elif self._csp_integration == "dsps_native":
            # For stellar_mass(), only age_weights matter (not ssp_flux_at_z).
            from tengri.components.stellar.sps.dsps_wrapper import compute_dsps_native_weights

            z_val = p.get("redshift", 0.1)
            t_obs_gyr = self._t_universe_gyr(z_val)
            lgmet = p.get("log_z_abs", -1.8477)
            lgmet_scatter = float(p.get("lgmet_scatter", self._lgmet_scatter))
            weights, _ = compute_dsps_native_weights(
                sfr_on_ssp,
                self.ssp_ages_yr,
                self.ssp_data.ssp_lgmet,
                self.ssp_data.ssp_lg_age_gyr,
                self.ssp_data.ssp_flux,
                t_obs_gyr,
                lgmet,
                lgmet_scatter,
            )
        elif self._csp_integration == "dsps_met_table":
            from tengri.components.stellar.sps.dsps_wrapper import compute_dsps_met_table_weights

            z_val = p.get("redshift", 0.1)
            t_obs_gyr = self._t_universe_gyr(z_val)
            lgmet_scatter = float(p.get("lgmet_scatter", self._lgmet_scatter))
            if self._met_mode == "ramp":
                from tengri.components.stellar.sps.dsps_wrapper import compute_log_z_evolving

                lgmet_per_age = compute_log_z_evolving(
                    self.ssp_data.ssp_lg_age_gyr,
                    p["log_z_abs_initial"],
                    p["log_z_abs_final"],
                    t_obs_gyr,
                )
            else:
                lgmet_per_age = jnp.full_like(self.ssp_ages_yr, p.get("log_z_abs", -1.8477))
            weights, _ = compute_dsps_met_table_weights(
                sfr_on_ssp,
                lgmet_per_age,
                self.ssp_ages_yr,
                self.ssp_data.ssp_lgmet,
                self.ssp_data.ssp_lg_age_gyr,
                self.ssp_data.ssp_flux,
                t_obs_gyr,
                lgmet_scatter,
            )
        else:
            # Closure-A consistency: route through the orchestrator so
            # ``predict_sfh_quantities`` returns the same stellar_mass /
            # weights as ``predict_derived`` (which uses
            # ``predict_state`` internally via
            # ``Prediction._ensure_sfh``). Was 4.1% apart with the
            # legacy rectangle rule (``sfr_on_ssp * _csp_age_dt``).
            # See ``tests/integration/test_derived_quantities.py::
            # test_mstar_consistent_between_methods``.
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

        # SFR averages — time-weighted mean over a lookback-time window.
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
        mw_z = compute_mass_weighted_metallicity(
            weights,
            self.ssp_ages_yr,
            p.get("log_z_abs", 0.0),
            log_z_initial=p.get("log_z_abs_initial"),
            log_z_final=p.get("log_z_abs_final"),
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
        """Compute SED-derived quantities in JIT-compatible form.

        Evaluates the full forward model and computes UV slope, spectral
        indices (D4000, Balmer break), bolometric/IR luminosities, dust
        attenuation, and luminosity-weighted age/metallicity. Returns
        a :class:`SEDQuantities` NamedTuple that is fully JIT-compatible
        and vmap-ready for batch inference.

        **Use this method for** JIT/batch loops (``jax.vmap``, ``jit``,
        ``grad``). **For interactive single-galaxy exploration**, use
        :meth:`predict` and access ``pred.sed.dn4000``, ``pred.sed.uv_slope``
        etc. — same quantities, with Python-side caching.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.

        Returns
        -------
        SEDQuantities
            NamedTuple with fields:

            - ``l_bol`` : float. Bolometric luminosity [L☉]
            - ``l_tir`` : float. Total infrared (8–1000 μm) luminosity [L☉]
            - ``l_dust_absorbed`` : float. Dust-absorbed luminosity [L☉]
              (intrinsic − attenuated), or NaN if intrinsic SED unavailable.
            - ``irx`` : float. Infrared excess := L_TIR / L_UV(1600 Å).
              Common probe of dust obscuration (Dale et al. 2001).
            - ``uv_slope_beta`` : float. UV slope (power-law index) in
              f_λ ∝ λ^β for 1200–2600 Å.
            - ``dn4000`` : float. D_n(4000) break ratio: flux average
              at 3750–3950 Å / 4050–4250 Å. Indicator of stellar age.
            - ``balmer_break`` : float. Balmer break: flux ratio
              ~3700 Å / ~4000 Å. Old stellar population signature.
            - ``m_uv`` : float. Absolute magnitude at 1500 Å
              (M_1500, standard reionization-era indicator).
            - ``fuv_flux`` : float. Flux at 1500 Å [erg/s/cm²]
            - ``nuv_flux`` : float. Flux at 2300 Å [erg/s/cm²]
            - ``fuv_flux_intrinsic`` : float. FUV flux, dust-free
              (intrinsic SED). NaN if unavailable.
            - ``nuv_flux_intrinsic`` : float. NUV flux, dust-free. NaN
              if unavailable.
            - ``rest_uv_color`` : float. Rest-frame UV color (f_1500 − f_2300).
            - ``luminosity_weighted_age_gyr`` : float. Luminosity-weighted
              age [Gyr] (∫L_λ age dλ / ∫L_λ dλ).
            - ``luminosity_weighted_metallicity`` : float. Luminosity-weighted
              log₁₀(Z/Z☉) or absolute log₁₀(Z).

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.
        Safe inside :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`.

        **Gradient-safe**: yes — all quantities are differentiable w.r.t.
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

        >>> sed_q = model.predict_sed_quantities(params)
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
        predict : Lazy prediction for single-galaxy exploration.
        predict_sfh_quantities : JIT-compatible SFH quantities.
        predict_rest_sed : Full rest-frame SED (for custom analysis).
        """
        # Dispatch to the orchestrator-backed bridge. Same semantics shift
        # PR 5a applied to ``predict_rest_sed``: the orchestrator's
        # stellar adapter uses DSPS-canonical (lognormal-MDF) CSP
        # integration unconditionally. For ``csp_integration='dsps_native'``
        # the legacy path produces identical results (sub-0.1% drift on
        # every published field). For the legacy default
        # ``csp_integration='trapz'``, the only field that drifts
        # noticeably (~12%) is ``luminosity_weighted_age_gyr`` — the
        # orchestrator integrates the actual ``lnu_age`` cube whose
        # sum-over-age IS ``sed_intrinsic``, while legacy's
        # ``compute_per_bin_luminosity(weights, ssp_flux_at_z)``
        # reconstruction has a hidden DSPS-joint-weight discrepancy
        # under trapz. The orchestrator value is the physically correct
        # one (energy-conserving by construction).
        return self.predict_sed_quantities_components(params)

    # ── Component orchestrator path (opt-in) ──────────────────────────

    def predict_sfh_quantities_components(self, params):
        """Drop-in replacement for :meth:`predict_sfh_quantities`.

        Routes through the orchestrator and converts the resulting
        :class:`ForwardState` to a legacy :class:`SFHQuantities`
        NamedTuple via :func:`tengri.forward.state_to_sfh_quantities`.
        Same return shape as the legacy method, computed via the
        SEDComponent path.

        Returns
        -------
        SFHQuantities
            7-field NamedTuple matching the legacy contract.
        """
        from tengri.forward import state_to_sfh_quantities

        return state_to_sfh_quantities(self.predict_state(params))

    def predict_sed_quantities_components(self, params):
        """Drop-in replacement for :meth:`predict_sed_quantities`.

        Returns
        -------
        SEDQuantities
            15-field NamedTuple matching the legacy contract.
        """
        from tengri.forward import state_to_sed_quantities

        return state_to_sed_quantities(self.predict_state(params))

    def predict_radio_quantities(self, params):
        """Orchestrator-path radio quantities.

        Returns
        -------
        RadioQuantities
            ``l_1p4ghz``, ``l_thermal``, ``l_nonthermal``, ``q_ir``.
            Fields are NaN if the configured chain has no
            :class:`RadioSEDComponent`.
        """
        from tengri.forward import state_to_radio_quantities

        return state_to_radio_quantities(self.predict_state(params))

    def predict_xray_quantities(self, params):
        """Orchestrator-path X-ray quantities.

        Returns
        -------
        XRayQuantities
            ``l_x_xrb``, ``l_x_agn``, ``l_x_total``.
        """
        from tengri.forward import state_to_xray_quantities

        return state_to_xray_quantities(self.predict_state(params))

    def predict_ionizing_quantities(self, params):
        """Orchestrator-path ionizing-photon quantities.

        Returns
        -------
        IonizingQuantities
            ``q_h``, ``xi_ion``.
        """
        from tengri.forward import state_to_ionizing_quantities

        return state_to_ionizing_quantities(self.predict_state(params))

    def predict_photometry_components(self, params):
        """Photometry through the orchestrator path.

        Runs the SEDComponent chain on the model's configuration,
        then projects the resulting rest-frame SED through every
        filter in :attr:`self.observation.photometry`. Returns flux
        densities in the AB system at the source.

        Parameters
        ----------
        params : Mapping
            Free-parameter dict (same shape as
            :meth:`predict_state`).

        Returns
        -------
        flux_density : ndarray, shape (n_filters,)
            Observed flux densities [erg/s/cm²/Hz].

        Raises
        ------
        ValueError
            If no photometric filters are configured on the
            observation.

        Notes
        -----
        **JIT-compatible**: yes — uses :func:`jax.jit`-friendly
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
                "predict_photometry_components requires photometric "
                "filters configured on the observation. Construct the "
                "model with ``filters=`` or pass an Observation that "
                "carries a Photometry instance."
            )
        state = self.predict_state(params)
        full = {**self.spec.get_fixed_values(), **params}
        return self.observation.predict(state, full)["phot_fnu"]

    def predict_spectrum_components(self, params, wave_obs=None):
        """Spectrum through the orchestrator path.

        Runs the SEDComponent chain, applies the cosmological redshift +
        luminosity-distance projection, interpolates onto ``wave_obs``,
        and (if configured) applies the instrument LSF + velocity-dispersion
        broadening. Mirrors the contract of the legacy
        :meth:`predict_spectrum`'s observed-frame output but goes through
        the SEDComponent chain rather than the fused kernel.

        Parameters
        ----------
        params : Mapping
            Free-parameter dict (same shape as
            :meth:`predict_state`).
        wave_obs : array_like, shape (n_pix,), optional
            Observed-frame wavelength grid [Angstrom]. If ``None``,
            falls back to the precomputed grid (`self._wave_obs` or
            `self._precomputed.spectroscopy.wave_obs_pixels`).

        Returns
        -------
        flux : ndarray, shape (n_pix,)
            Observed-frame spectral flux density [erg/s/cm^2/Hz].

        Raises
        ------
        ValueError
            If no ``wave_obs`` grid is supplied or precomputed.

        Notes
        -----
        **JIT-compatible**: yes — :func:`run_components`, the rest→obs
        projection in :func:`observe_spectrum_from_rest_sed`, and
        :func:`apply_lsf` are all JIT-friendly. No calibration polynomial
        is applied; callers that need calibration should compose it on
        top via the user-likelihood Protocol path.
        """
        # (legacy dead ``self._precomputed.spectroscopy`` tier removed — #620)
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
                "predict_spectrum_components requires a wave_obs grid "
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
            [O III] 4363, ...). All luminosities in Lsun.

        Raises
        ------
        NotImplementedError
            When the active nebular backend does not publish a discrete
            line catalog (BakedIn or shock). Switch to ``neb={'type':
            'cue', ...}`` or ``neb={'type': 'cloudy_grid', ...}`` for
            discrete line predictions, or read the continuous nebular
            SED from ``model.predict_rest_sed(params).sed`` directly.

        Notes
        -----
        Dust attenuation is applied to the line luminosities in the
        attenuation regime selected by ``_neb_dust_mode`` (default
        ``"bc"`` — birth-cloud + diffuse, Charlot & Fall 2000 [1]_).
        The line-attenuated values match the continuum treatment in
        :meth:`predict_rest_sed`, so Balmer decrement, BPT, and other
        line-ratio diagnostics behave correctly under a dust sweep
        (regression: issue #313).

        References
        ----------
        .. [1] S. Charlot & S. Fall, "A Simple Model for the Absorption of
           Starlight by Dust in Galaxies," ApJ 539, 718 (2000).
        """
        from tengri.components.nebular import BakedInBackend

        if isinstance(self._nebular_backend, BakedInBackend):
            raise NotImplementedError(
                "predict_emission_lines is not supported for the BakedIn "
                "nebular backend: emission is baked into the SSP grid and "
                "no discrete line catalog is published. To predict line "
                "luminosities, build the model with a photoionization "
                "backend, e.g. neb={'type': 'cue', '*': FIXED} (requires "
                "a bare-stellar SSP) or neb={'type': 'cloudy_grid', ...}. "
                "For a quick narrow-band measurement on the BakedIn SED, "
                "integrate model.predict_rest_sed(params).sed across the "
                "line wavelength range yourself."
            )
        from tengri.forward import state_to_emission_lines
        from tengri.forward.emission_helpers import attenuate_emission

        state = self.predict_state(params)
        lines = state_to_emission_lines(state)
        if lines.all_waves.size == 0:
            # No discrete catalog; nothing to attenuate.
            return lines

        # Apply dust attenuation at line wavelengths in the same regime
        # the continuum sees. Charlot & Fall 2000: lines from young
        # populations (HII regions) experience BC + diffuse; single-
        # component dust applies the BC law twice (degenerate fallback).
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

    def run(self, state, params):
        """Run the SED forward chain. Pure JAX.

        SED is the head of the per-population orchestration; in the
        tracer-bullet single-population path, ``state`` is an empty
        :class:`tengri.protocols.ForwardState` with just the wavelength
        grid. The method delegates to :meth:`predict_state` for the
        actual physics.

        Parameters
        ----------
        state : ForwardState
            Incoming state (empty for SED as the head of the chain).
        params : Mapping
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
        """
        return self.predict_state(params)

    def predict_state(self, params, fixed_values=None, ssp_data=None, template_data=None):
        """Forward pass via the SEDComponent orchestrator.

        Builds a component chain from this model's structural settings
        (``self.spec`` + ``self.ssp_data`` + dust / nebular / AGN / radio
        / X-ray / IGM flags) and threads ``params`` through
        :func:`tengri.forward.run_components`. Returns the final
        :class:`tengri.protocols.ForwardState`, **not** a legacy
        :class:`Prediction` — callers wanting the legacy shape should
        keep using :meth:`predict_photometry`/:meth:`predict_spectrum`
        until the full integration adapter ships.

        This is the public bridge between :class:`SEDModel`'s
        configuration surface and the orchestrator: it lets users with
        an existing ``SEDModel`` go through ``run_components`` without
        re-typing the chain at every call site.

        Parameters
        ----------
        params : Mapping
            Free parameters keyed by canonical name (``sfh_*``,
            ``met_*``, ``dust_*``, ``agn_*``, ``radio_*``, ``xray_*``,
            ``igm_*``, ``redshift``).
        fixed_values : Mapping | None, optional
            Fixed parameter values. When provided, overrides
            ``self.spec.get_fixed_values()``. Used by :meth:`predict_observables_jit`
            to thread per-galaxy fixed values as JIT runtime inputs (Phase 4-A).
        ssp_data : Any | None, optional
            SSP grid. When provided, passed to components that need it as a
            JIT runtime input instead of using closure capture (Phase 4-B).
            Defaults to ``None``, which causes components to use their
            internal ``self.ssp_data``.
        template_data : Any | None, optional
            Nebular backend grids and weights. When provided, passed to
            components as JIT runtime inputs instead of closure capture (Phase 4-C).
            Defaults to ``None``, which causes components to use their
            internal template data.

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
        **JIT-compatible**: yes — :func:`run_components` and every
        adapter's ``apply`` are pure JAX.

        ``self.spec.mean_sfh_type`` is a list (e.g. ``["tsnorm"]``,
        ``["dpl", "field"]``); the first entry is the mean SFH model,
        and ``"field"`` anywhere in the list enables the GP-field
        branch. Anything else (``burst``, etc.) is currently unmapped
        and will raise downstream.
        """
        from tengri.forward import build_components, run_components
        from tengri.protocols.component import ForwardState

        # Phase 3d-5 (2026-05-20): cache the built chain on the model. Chain
        # construction runs each component's ``precompute()``, which for the
        # stellar component with ``wave_precomp=True`` calls
        # ``preintegrate_grid`` — a numpy-level routine with Python ``float()``
        # calls that can't be re-traced under ``jax.jit``. Building the chain
        # once at first call (or earlier) makes subsequent ``predict_state``
        # invocations pure: they just thread ``params`` through the cached
        # chain via ``run_components``. The chain depends only on structural
        # config (spec, ssp_data, filters, approx), all of which are immutable
        # after ``__init__``.
        # Validate param keys at the orchestrator entry — silent drops
        # of typo'd or stale override keys produce plausible-looking but
        # wrong physics. Covers the dict-merge code path that
        # ``predict_observables_jit`` already guards at line ~4242
        # (#314). Skip in JIT-runtime threading mode where the JIT
        # entry point has already validated the caller's dict.
        if fixed_values is None:
            check_unknown_params(params, self._param_map)

        cached = getattr(self, "_cached_component_chain", None)
        if cached is None:
            cached = self._build_component_chain()
            self._cached_component_chain = cached
        chain = cached
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
        # Phase 4-A: ``fixed_values`` is an optional override. When provided
        # (typically from ``predict_observables_jit`` threading it as a
        # JIT runtime input), use it instead of ``self.spec.get_fixed_values()``.
        # That decouples per-galaxy fixed values from the closure so two
        # SEDModels with the same structure but different fixed values
        # share one compiled function.
        if fixed_values is None:
            fixed_values = self.spec.get_fixed_values()
        full_params = {**fixed_values, **params}

        # Phase 4-B: thread ssp_data as JIT input. Defaults to None,
        # which causes components to use their closure-captured self.ssp_data.
        # Phase 4-C: thread template_data (nebular grids) as JIT input.
        return run_components(
            chain, state0, full_params, ssp_data=ssp_data, template_data=template_data
        )

    def predict_observables(self, params):
        """Project the orchestrator state into every configured observable.

        Single bit-exact entry point: runs the SEDComponent chain and
        delegates to :meth:`Observation.predict` for projection. Returns
        an :class:`Observables` NamedTuple with one field per configured
        observation sub-block (``phot_fnu``, ``phot_rest_fnu``, ``spec_fnu``).

        Parameters
        ----------
        params : Mapping
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
        **JIT-compatible**: yes. Not self-JIT'd — wrap with
        :func:`jax.jit` for hot loops, or call
        :meth:`predict_observables_jit` for the pre-cached version.

        **Phase 2 of forward-projection unification.** Synthesized per-model
        at :meth:`__init__` from observation contents; missing channels
        raise ``AttributeError`` on access.
        """
        if self.observation is None:
            raise ValueError(
                "predict_observables requires an Observation. Build the "
                "model with ``observation=`` set."
            )

        # Eager (non-JIT) forward + projection. Runs the SAME ``_impl`` closure
        # that :meth:`predict_observables_jit` wraps in ``jax.jit`` — one
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
            params, self.spec.get_fixed_values(), self.ssp_data, self._template_data_for_jit()
        )

    def predict_observables_jit(self, params):
        """Self-JIT'd, structurally-cached version of :meth:`predict_observables`.

        Bit-exact with :meth:`predict_observables` (same orchestrator
        chain, same :meth:`Observation.predict` projection). The compiled
        function is cached on :meth:`compile_signature`, so two SEDModel
        instances with identical structure (same physics, same filter
        set, same observation shape) share one compile across galaxies.

        Parameters
        ----------
        params : Mapping
            Free-parameter dict.

        Returns
        -------
        Observables
            NamedTuple with fields keyed by configured sub-blocks.

        Notes
        -----
        **JIT-compatible**: yes — this method IS the JIT entry point.

        Phase 4-A (2026-05-20): ``self.spec.get_fixed_values()`` is now
        passed as a JIT runtime input rather than closure-captured. Two
        SEDModels with the same structural config but different per-galaxy
        fixed values (e.g., ``redshift=Fixed(0.1)`` vs ``redshift=Fixed(0.5)``)
        share a :meth:`compile_signature` and reuse the same compiled
        function — per-galaxy values flow through at call time.

        For per-galaxy fixed redshifts, build with
        ``approx=WavePrecomp(z_min=catalog_z_min, z_max=catalog_z_max)`` so
        the ztable covers the catalog range; runtime ``params['redshift']``
        is then a fast interpolation lookup.

        See Also
        --------
        predict_observables : un-JIT'd version (debug / one-shot).
        compile_signature : structural fingerprint controlling cache reuse.

        Phase 4-B (2026-05-20): ``self.ssp_data`` is now passed as a JIT
        runtime input rather than closure-captured. The SSP grid becomes a
        ``Parameter`` op in the compiled HLO instead of a ``Constant`` op,
        reducing compile size and time.

        Phase 4-C (2026-05-21): nebular backend grids and weights are now
        passed as JIT runtime inputs. Backend grids become ``Parameter`` ops
        instead of ``Constant`` ops, reducing compile size for Cue and CloudyGrid.
        """
        # Validate param keys before entering JIT — silent drops of unknown
        # override keys produce plausible-looking but wrong physics (issue #314).
        check_unknown_params(params, self._param_map)
        return self._get_or_build_predict_observables_jit()(
            params, self.spec.get_fixed_values(), self.ssp_data, self._template_data_for_jit()
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
        # Phase 4-A: ``fixed_values`` is no longer closure-captured — it
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
        # Phase 3d-5: route the photometry channel through the LUT projection
        # when the model was built with ``approx=WavePrecomp(...)``. The
        # routing decision is closure-captured per-model and baked into
        # ``compile_signature`` (via the resolved ``approx`` config tuple),
        # so structurally-equal models share a compile across the two
        # routings without colliding. Spectrum stays exact — no spectrum
        # LUT yet.
        use_lut = bool(self._approx.get("wave_precomp")) and not observation.can_do_spectroscopy
        # Part B: spectrum LUT inside the fused JIT kernel. ``predict_state``
        # (called within ``_impl``) publishes ``spec_eff_waves`` + the per-pixel
        # ``*_spec_lnu_precomp`` family via the cached chain, so the projector is
        # fully JAX-traceable — no eager fallback needed. ``phot_lut`` lets a
        # JOINT model also project photometry in the same kernel (Part A).
        spec_lut = bool(self._approx.get("spectrum_precomp")) and observation.can_do_spectroscopy
        phot_lut = bool(self._approx.get("wave_precomp"))

        # Warm the component-chain cache OUTSIDE the JIT trace. The chain
        # build runs each component's ``precompute()``, which for the
        # stellar component with ``wave_precomp=True`` calls
        # ``preintegrate_grid`` — a numpy-level routine with Python
        # ``float()`` calls that can't be traced. After this warmup,
        # ``predict_state`` inside the JIT reuses the cached chain.
        if getattr(self, "_cached_component_chain", None) is None:
            self._cached_component_chain = self._build_component_chain()

        def _impl(params, fixed_values, ssp_data, template_data):
            state = self.predict_state(
                params,
                fixed_values=fixed_values,
                ssp_data=ssp_data,
                template_data=template_data,
            )
            full = {**fixed_values, **params}
            # Part B: spectrum LUT (and, for a joint model, photometry LUT)
            # projected inside the fused kernel. Each projector sums its own
            # ``*_spec_lnu_precomp`` / ``*_phot_lnu_precomp`` family; collect
            # dicts and build the per-model Observables once so phot_fnu and
            # spec_fnu coexist. (Velocity dispersion / LSF are not applied on the
            # per-pixel continuum LUT — that is SpectrumPrecomp's documented
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

    def _template_data_for_jit(self):
        """Collect template grids/weights for JIT threading (nebular + dust IR + AGN).

        Walks the cached component chain (built by predict_state warmup)
        to collect template arrays that should be threaded as JIT runtime
        inputs instead of closure-captured, so they appear as JAX
        ``Parameter`` ops rather than baked-in HLO ``Constant`` ops.

        **Nebular templates** (Phase 4-C): duck-typed on backend ``.grid``
        / ``.weights`` attributes (Cue, CloudyGrid, etc.).

        **Dust IR**: emission ports (Astrodust, PAHspec, Dale, …) self-load
        their HDF5 grids in ``EmissionPort.load``/``predict``; only the
        build-time energy-balance LUT and per-filter band response are threaded
        here (added below).

        **AGN templates** (Phase 4-D-C): extracted from the
        AGNSEDComponent's cached state (SKIRTOR templates).

        Returns
        -------
        dict[str, Any] | None
            Nested dict with namespace keys (``"nebular"``, ``"dust_ir"``,
            ``"agn"``) carrying the threaded template data for that
            subsystem. Returns ``None`` if no components need threading.
        """
        cached = getattr(self, "_cached_component_chain", None)
        if cached is None:
            return None

        from tengri.components.agn.component import AGNSEDComponent
        from tengri.components.nebular.component import NebularSEDComponent

        result = {}

        # ── Nebular backend threading (Phase 4-C) ──
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

        # Dust IR emission ports (Astrodust, PAHspec, Dale, …) self-load their
        # HDF5 grids in ``EmissionPort.load``/``predict`` — no adapter-state
        # threading is needed here. The build-time energy-balance LUT and
        # per-filter band response below are the only dust-IR data threaded.

        # ── AGN template threading (Phase 4-D-C) ──
        for component in cached:
            if not isinstance(component, AGNSEDComponent):
                continue
            agn_templates = {}
            if component._state is not None and component._state.skirtor_templates is not None:
                agn_templates["skirtor"] = component._state.skirtor_templates
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
        is either an optical depth / eta scaling or an emission-shape knob — so
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
        # or new path (separate dust emission port in the pipeline).
        # After the switchover, dust_emission_model is set from the spec even
        # when using separate ports, so we check that or the old emission_model path.
        has_dust_emission = (
            dust is not None and getattr(dust.config, "emission_model", None) is not None
        ) or self._dust_emission_model is not None
        if (
            has_dust_emission
            and dust is not None
            and self._approx.get("wave_precomp")
            and not bool(getattr(self.spec, "alpha_fe_evolving", False))
            and not unsafe_free
            and self.ssp_data is not None
        ):
            fixed = self.spec.get_fixed_values()
            bc_params, diff_params = resolve_bc_diff_law_params(
                fixed, dict(dust.config.bc_law_overrides), dict(dust.config.diff_law_overrides)
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
        is a build-time constant — returned here (shape ``(n_filter,)``) so
        :meth:`DustSEDComponent.apply` replaces the per-call dense filter
        integral (#622) with the exact ``L_ir × R``. Returns ``None`` otherwise
        (free shape/z → keep the per-call integral, or ``fast_dust_emission``).
        """
        cached = getattr(self, "_dust_band_response_cache", "unset")
        if cached != "unset":
            return cached

        from tengri.components.dust.two_component import DustSEDComponent

        response = None
        dust = next(
            (
                c
                for c in chain
                if isinstance(c, DustSEDComponent)
                and getattr(c.config, "emission_model", None) == "modified_blackbody"
            ),
            None,
        )
        free = set(self.spec.free_params)
        shape_params = ("dust_T", "dust_beta_ir", "dust_epsilon_mbb", "redshift")
        stellar = next((c for c in chain if c.name == "stellar"), None)
        st = getattr(stellar, "_state", None)
        fw_pad = getattr(st, "phot_fw_padded", None)
        ft_pad = getattr(st, "phot_ft_padded", None)
        if (
            dust is not None
            and self._approx.get("wave_precomp")
            and not (free & set(shape_params))
            and fw_pad is not None
            and ft_pad is not None
        ):
            from tengri.components.dust.emission import modified_blackbody
            from tengri.observation.photometry import lnu_filter_integral_batch

            fixed = self.spec.get_fixed_values()
            wave = self._rest_wavelength
            z = jnp.asarray(fixed.get("redshift", 0.0))
            unit_sed_ir = modified_blackbody(
                wave,
                L_absorbed=1.0,
                dust_T=jnp.asarray(fixed.get("dust_T", 35.0)),
                dust_beta_ir=jnp.asarray(fixed.get("dust_beta_ir", 1.6)),
                redshift=z,
                dust_epsilon_mbb=jnp.asarray(fixed.get("dust_epsilon_mbb", 1.0)),
            )
            response = lnu_filter_integral_batch(unit_sed_ir, wave, fw_pad, ft_pad, z)

        self._dust_band_response_cache = response
        return response

    def _build_component_chain(self):
        """Construct the orchestrator chain from ``self``'s settings.

        Reads ``self.spec`` and the ``_dust_*``/``_nebular_backend``/
        ``_agn_model``/``_uses_*`` attributes set in :meth:`__init__`
        and produces a list of :class:`SEDComponent` adapters in the
        canonical pipeline order.
        """
        from tengri.forward.component_factory import build_components

        # Mean SFH: first entry of mean_sfh_type, with "field" flag if
        # the GP modulator is composed in.
        mean_types = list(getattr(self.spec, "mean_sfh_type", ["tsnorm"]))
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

        chain = build_components(
            ssp_data=self.ssp_data,
            sfh_model=mean_model,
            field=field_on,
            metallicity_model=getattr(self, "_met_mode", "delta"),
            n_grid=int(getattr(self.spec, "n_grid", 256)),
            lgmet_scatter=float(getattr(self, "_lgmet_scatter", 0.2)),
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
            dust_emission_model=getattr(self, "_dust_emission_model", None),
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
            use_shock=bool(getattr(self, "_uses_shock", False)),
            shock_norm=getattr(self, "_shock_norm", "frac"),
            shock_abundance=getattr(self, "_shock_abundance", "solar"),
            shock_component=getattr(self, "_shock_component", "combined"),
        )

        # Phase 3b/3c: Eager precompute stellar photometry LUT when wave_precomp is enabled.
        # Phase 3b: fixed-z LUT when redshift is Fixed.
        # Phase 3c-1: free-z ztable when redshift is Free (Uniform prior).
        if (
            self._approx.get("wave_precomp")
            and self.observation is not None
            and hasattr(self.observation, "photometry")
            and self.observation.photometry is not None
            and len(chain) > 0
            and chain[0].name == "stellar"
        ):
            from dataclasses import replace

            from tengri.components.stellar.component import StellarSEDComponent

            stellar = chain[0]
            if isinstance(stellar, StellarSEDComponent):
                # Build filter tuple from observation photometry.
                filters = tuple(
                    zip(
                        self.observation.photometry.filter_waves,
                        self.observation.photometry.filter_trans,
                        strict=False,
                    )
                )

                # Determine redshift spec: fixed or free. Phase 3c-1 dispatch.
                try:
                    redshift_dist = self.spec.get_distribution("redshift")
                    is_fixed = redshift_dist.is_fixed
                    z_bounds = redshift_dist.bounds
                except (AttributeError, KeyError):
                    # Fallback: assume fixed at 0.0 if redshift not in spec
                    is_fixed = True
                    z_bounds = (0.0,)

                # Catalog-fit reuse (Approach A): when the user passes
                # ``WavePrecomp(catalog_z_range=...)``, route through the
                # free-z ztable branch even when redshift is Fixed in the
                # spec — different per-galaxy ``Fixed(z)`` values then share
                # the same compile.
                if is_fixed and self._catalog_z_range is None:
                    # Phase 3b: fixed-z LUT
                    redshift_spec = {"mode": "fixed", "value": float(z_bounds[0])}
                else:
                    # Phase 3c-1: free-z ztable. User can override n_z / z_min /
                    # z_max via ``approx=WavePrecomp(...)``; otherwise pull from
                    # the redshift prior with 1 % padding and use n_z=100.
                    cfg = self._approx_config or WavePrecomp()
                    if self._catalog_z_range is not None:
                        z_lo, z_hi = self._catalog_z_range
                        pad = 0.0  # explicit range — no padding
                    elif z_bounds is None or len(z_bounds) < 2:
                        # Fixed spec falling through (no bounds for a Fixed dist);
                        # default to a generous photo-z range.
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

                # Call precompute to build the LUT or ztable.
                state = stellar.precompute(
                    ssp_data=stellar.ssp_data,
                    wave_grid=None,
                    approx=self._approx,
                    filters=filters,
                    redshift_spec=redshift_spec,
                )
                # Replace the stellar component with one carrying the precomputed state.
                chain[0] = replace(stellar, _state=state)

                # Phase 3c-3d-agn: AGN component also needs filter passbands
                # so its apply() can publish ``agn_phot_lnu_precomp``. Find the
                # AGN component in the chain and re-precompute it with filters.
                from tengri.components.agn.component import AGNSEDComponent

                for idx, comp in enumerate(chain):
                    if isinstance(comp, AGNSEDComponent):
                        agn_state = comp.precompute(
                            ssp_data=None,
                            wave_grid=None,
                            approx=self._approx,
                            filters=filters,
                        )
                        chain[idx] = replace(comp, _state=agn_state)
                        break

                # Phase 3c-3d-neb: non-BakedIn nebular component caches
                # filters too, for its filter-integrated precompute publish.
                from tengri.components.nebular.component import NebularSEDComponent

                for idx, comp in enumerate(chain):
                    if isinstance(comp, NebularSEDComponent):
                        neb_state = comp.precompute(
                            ssp_data=None,
                            wave_grid=None,
                            approx=self._approx,
                            filters=filters,
                        )
                        chain[idx] = replace(comp, _state=neb_state)
                        break

        # Phase 5: SpectrumPrecomp — pre-rebin the SSP grid to the spectrum
        # pixel centers. Only the stellar component needs a build-time LUT;
        # downstream SEDModelComponents (dust / AGN / IGM / nebular continuum)
        # route through their ``_apply_spec_precomp`` branch at runtime once
        # the stellar component publishes ``spec_eff_waves``.
        if (
            self._approx.get("spectrum_precomp")
            and self.observation is not None
            and self.observation.can_do_spectroscopy
            and len(chain) > 0
            and chain[0].name == "stellar"
        ):
            from dataclasses import replace

            from tengri.components.stellar.component import StellarSEDComponent

            stellar = chain[0]
            if isinstance(stellar, StellarSEDComponent):
                spec_wave_obs = self.observation.spectroscopy.wave_obs

                # Fixed vs free redshift (mirror the wave_precomp dispatch).
                try:
                    redshift_dist = self.spec.get_distribution("redshift")
                    is_fixed = redshift_dist.is_fixed
                    z_bounds = redshift_dist.bounds
                except (AttributeError, KeyError):
                    is_fixed = True
                    z_bounds = (0.0,)

                if is_fixed:
                    redshift_spec = {"mode": "fixed", "value": float(z_bounds[0])}
                else:
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

                spec_state = stellar.precompute(
                    ssp_data=stellar.ssp_data,
                    wave_grid=None,
                    approx=self._approx,
                    spec_wave_obs=spec_wave_obs,
                    redshift_spec=redshift_spec,
                )
                # Part A (joint): MERGE the spectrum LUT into the stellar state
                # rather than overwriting it — the photometry-LUT block above
                # may already have populated ssp_phot_lut/ztable for a joint
                # model. Spec-only models have no prior phot state, so the merge
                # is a no-op on the phot fields.
                prev = stellar._state
                if prev is not None:
                    merged = replace(
                        prev,
                        ssp_spec_lut=spec_state.ssp_spec_lut,
                        ssp_spec_ztable=spec_state.ssp_spec_ztable,
                    )
                else:
                    merged = spec_state
                chain[0] = replace(stellar, _state=merged)

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

    def predict_photometry_batch(self, params_batch):
        """Compute photometry for a batch of parameter sets via jax.vmap.

        **Use this method for** posterior chains / mock catalogs (batched
        forward pass). **For interactive single-galaxy use**, access
        ``model.predict(params).photometry``.

        Parameters
        ----------
        params_batch : dict of arrays
            Each value has shape (N, ...) with leading batch dimension.

        Returns
        -------
        array, shape (N, n_filters)
            Photometric flux for each galaxy.

        Notes
        -----
        **JIT-compatible**: yes — uses :func:`jax.vmap` over
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

        return _fn(self, params_batch)

    def predict_spectrum_batch(self, params_batch):
        """Compute spectra for a batch of parameter sets via jax.vmap.

        **Use this method for** batched spectra over posterior chains.
        **For interactive single-galaxy use**, access
        ``model.predict(params).spectrum``.

        Parameters
        ----------
        params_batch : dict of arrays
            Each value has leading batch dimension.

        Returns
        -------
        array, shape (N, n_pix)
            Spectral flux for each galaxy.

        Notes
        -----
        **JIT-compatible**: yes — uses :func:`jax.vmap` over
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

        return _fn(self, params_batch)

    # ── Private prediction dispatch ───────────────────────────────────

    @staticmethod
    def _jit_safe_params(params):
        """Strip string-typed entries so the params dict is safe to pass into JIT.

        String-typed Fixed parameters (e.g. ``shock_abundance="solar"``,
        ``shock_component="combined"``) are config enums, not values that
        flow through the gradient computation. Including them in the dict
        passed to a ``jax.jit``'d function makes ``tree_flatten`` reject the
        input with ``TypeError: ... <class 'str'> ... at path params['<name>']``.
        Strip them here; downstream code that needs them must read from
        ``self.spec``'s fixed values, not from the JIT params dict.
        """
        return {k: v for k, v in params.items() if not isinstance(v, str)}

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
        ssp : str or SSPData
            Path to SSP HDF5 file, or a pre-loaded ``SSPData`` instance.
        sfh : str
            SFH family name, e.g. ``"tsnorm"``, ``"dpl"``, ``"dpl+field"``.
        dust : str
            Dust attenuation law. ``"charlot_fall"`` (default), ``"calzetti"``, etc.
        nebular : str or None
            Nebular emission backend. ``"baked_in"``, ``"cloudy_grid"``, ``"cb19"``,
            ``"mappings"``, ``"cue"``, ``"shock"``, or None.
        agn : str or None
            AGN model. None (disabled) or any AGN model name.
        redshift : float or str
            Fixed redshift (float), or ``"free"`` to add a free redshift parameter.
        filters : list of str, optional
            Filter names for photometry, e.g. ``["sdss_u", "sdss_g", "sdss_r"]``.
        wave_obs : array, optional
            Observed-frame wavelength array for spectroscopy.
        priors : dict, optional
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
        stellar=None,
        dust=None,
        neb=None,
        shock=None,
        agn=None,
        igm=None,
        radio=None,
        xray=None,
        foreground=None,
        redshift=None,
        apply_igm=None,
        filters=None,
        observation=None,
        **model_kwargs,
    ) -> SEDModel:
        """Build an SEDModel from the Bagpipes-style nested-dict form.

        Convenience constructor that translates grouped dicts (one per physics
        block) into a ``Parameters`` via
        :func:`tengri.parse_groups`, then constructs the
        ``SEDModel``. Anything left unspecified auto-fills from the registry.

        The ``from_*`` namespace is reserved for future deserialization
        entry points (``from_file``, ``from_yaml``, ``from_dict``); use
        ``build`` to construct a model from in-memory physics-component dicts.

        Parameters
        ----------
        ssp_data : SSPData
            Pre-loaded SSP grid (from :func:`load_ssp_data`).
        sfh, dust, neb, agn, igm, radio, xray : dict, optional
            Per-component nested dicts. Each may carry ``'type'``, ``'*'``
            (wildcard set to :data:`~tengri.FREE` or :data:`~tengri.FIXED`),
            and per-parameter overrides. See
            :func:`tengri.parameters.parse_groups` for the full grammar.
        redshift, apply_igm : scalar, Distribution, or sentinel, optional
            Top-level kwargs forwarded into the parameter resolution.
        filters : list of str, optional
            Filter names; forwarded to ``__init__``.
        observation : Observation, optional
            Observation object; forwarded to ``__init__``.
        **model_kwargs
            Additional keywords forwarded to :meth:`__init__` (e.g.
            ``precompute``, ``forward_dtype``, ``approx``).

        Returns
        -------
        SEDModel
            Fully initialized model, identical to one built via
            ``SEDModel(parse_groups(**groups), ssp_data, ...)``.

        See Also
        --------
        tengri.parse_groups : The underlying nested-dict parser that returns
            a :class:`Parameters` spec.
        SEDModel.from_config : String-based grouped configuration with
            defaults from ``defaults.toml``.

        Examples
        --------
        >>> from tengri import SEDModel, FREE, FIXED, Uniform, Fixed
        >>> model = SEDModel.build(
        ...     ssp_data=ssp,
        ...     sfh={"type": "dpl", "*": FREE, "beta": Uniform(1, 3)},
        ...     dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED, "tau_bc": 0.5},
        ...     neb={"type": "cue", "*": FIXED},
        ...     redshift=Fixed(0.05),
        ...     filters=["sdss_u", "sdss_g", "sdss_r"],
        ... )
        """
        groups = {
            k: v
            for k, v in dict(
                sfh=sfh,
                stellar=stellar,
                dust=dust,
                neb=neb,
                shock=shock,
                agn=agn,
                igm=igm,
                radio=radio,
                xray=xray,
                foreground=foreground,
                redshift=redshift,
                apply_igm=apply_igm,
            ).items()
            if v is not None
        }
        from tengri.parameters.groups import _TOP_LEVEL_SETTINGS, parse_groups

        # Top-level parameter settings (e.g. ``n_grid``) belong to
        # ``parse_groups`` / ``Parameters``, not ``__init__``. Pull any that
        # arrived via ``**model_kwargs`` over to the group dict so that, e.g.,
        # ``SEDModel.build(..., n_grid=128)`` works instead of raising a
        # confusing ``__init__() got an unexpected keyword argument`` error.
        for _key in _TOP_LEVEL_SETTINGS:
            if _key in model_kwargs:
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
        _warn_agn_dust_double_count(spec)
        return cls(
            spec,
            ssp_data,
            filters=filters,
            observation=observation,
            **model_kwargs,
        )

    def prior_predictive(self, n: int = 500, seed: int = 42) -> PriorPredictive:
        """Sample from the prior and evaluate forward model on each draw.

        Parameters
        ----------
        n : int
            Number of prior samples. Default 500.
        seed : int
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
        method: str = "vi",
        data_type: str | None = None,
        *,
        photometry: tuple | None = None,
        spectrum: tuple | None = None,
        init: str | None = None,
        **kwargs,
    ):
        """Fit observed data. Deprecated — prefer ``ForwardModel.fit`` for new code.

        .. deprecated:: 0.x
            Inference is canonically through :class:`ForwardModel`
            (issue #211). Replace::

                result = sed.fit(data, noise, method="vi")

            with::

                forward = ForwardModel.build(sed=sed, observation=obs)
                result = forward.fit(data, noise, method="vi")

            or the equivalent ``Fitter(forward, data, noise).run("vi")``.

            ``SEDModel.fit`` keeps working until tengri v1.0; this method
            is a thin shim around :func:`tengri.forward.convenience.fit_model`
            and emits a one-shot DeprecationWarning.

        Parameters
        ----------
        data : array, optional
            Observed flux array (photometry or spectroscopy). For joint fitting,
            leave as ``None`` and use ``photometry=`` / ``spectrum=`` instead.
        noise : array, optional
            1-sigma uncertainties matching ``data``.
        method : str
            Inference method. Default ``"vi"`` (geoVI variational inference).
            Any canonical name accepted by ``Fitter.run()`` works here:
            ``"vi"``, ``"vi_linear"``, ``"mcmc"``, ``"mcmc_raytrace"``,
            ``"mcmc_nuts"``, ``"map"``, ``"laplace"``, ``"auto"``, etc.
        data_type : str or None
            ``"photometry"``, ``"spectroscopy"``, or ``"joint"``.
            When ``None`` (default), inferred from the model's ``observation``
            or from whether ``photometry=`` / ``spectrum=`` kwargs are used.
        photometry : tuple of (flux, noise), optional
            Photometric data for joint fitting. Pass alongside ``spectrum=``.
        spectrum : tuple of (flux, noise), optional
            Spectroscopic data for joint fitting. Pass alongside ``photometry=``.
        init : str or None
            Initialization strategy. ``"map"`` runs MAP optimization first, then
            uses the result to warm-start the requested method. ``None`` (default)
            uses the method's own default initialization.
        **kwargs
            Forwarded to ``Fitter.run()``.

        Returns
        -------
        Posterior
            Inference results.  ``._fitter`` is set so ``.refine()`` works.
            After this call, ``self.fitter_`` holds the ``Fitter`` instance.

        Notes
        -----
        Convenience wrapper around :class:`Fitter`. For advanced usage
        (custom loss, multiple refinement steps), use ``Fitter`` directly.

        Examples
        --------
        >>> result = model.fit(flux_obs, noise)
        >>> result = model.fit(flux_obs, noise, method="mcmc")
        >>> result = model.fit(photometry=(flux_p, noise_p), spectrum=(flux_s, noise_s))
        >>> result = model.fit(flux_obs, noise, init="map")
        >>> result = model.fit(flux_obs, noise).refine("mcmc_raytrace")
        """
        import warnings

        warnings.warn(
            "SEDModel.fit is deprecated and will be removed in tengri v1.0. "
            "Use ForwardModel.fit instead: "
            "forward = ForwardModel.build(sed=sed, observation=obs); "
            "result = forward.fit(data, noise, method=...). "
            "See issue #211.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        catalog : DataFrame, Table, or list of dict
            Input catalog.
        flux_cols : list of str
            Column names for per-band flux values.
        err_cols : list of str
            Column names for per-band 1-sigma uncertainties.
        redshift_col : str or None
            If provided, use this column as per-row redshift.
        method : str
            Inference method. Default ``"vi"``.
        n_workers : int
            Currently ignored (reserved for multiprocessing). Default 1.
        verbose : bool
            Print per-galaxy progress. Default True.
        output_dir : str or None
            If provided, save each Posterior to ``{output_dir}/{id}.h5``.
        id_col : str or None
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
        observations_list : list
            Each element is a (flux, noise) tuple or dict with flux_obs/noise keys.
        method : str
            Hierarchical inference method. Default ``"vi"``.
        population_prior : dict or None
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
        params : dict
            Parameter values.
        snr : float
            Signal-to-noise ratio. Default 20.0.
        key : PRNGKey, optional
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
        params : dict
            Parameter values.
        wave_obs : array
            Observed wavelength grid [Angstrom].
        snr : float
            Signal-to-noise ratio per pixel. Default 30.0.
        key : PRNGKey, optional
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
        params_batch : dict of arrays
            Each value has leading batch dimension.
        snr : float
            Signal-to-noise ratio. Default 20.0.
        key : PRNGKey, optional
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
        posterior : Posterior
            Inference results with samples (if available) or params.
        true_params : dict, optional
            True parameter values (if known) to overlay on plot.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure.
        n_draws : int
            Number of posterior samples to show as thin lines. Default 50.
        color : str
            Color for posterior lines. Default "C0" (first color in style).
        label : str
            Label for posterior. Default "Posterior".

        Returns
        -------
        ax : matplotlib.axes.Axes
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
        This is the grid used by :meth:`predict_rest_sed` by default when
        no custom ``wave=`` is passed. Updated when radio/X-ray components
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
        z : float or jnp.ndarray
            Redshift.

        Returns
        -------
        float
            Age of universe in Gyr.
        """
        return age_at_z(z)

    def _interp_metallicity(self, log_z):
        """Dispatch metallicity interpolation (single Z value)."""
        return interp_metallicity(self, log_z)

    def _interp_metallicity_evolving(self, log_z_per_age):
        """Dispatch evolving metallicity interpolation (per-age Z)."""
        return interp_metallicity_evolving(self, log_z_per_age)

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
