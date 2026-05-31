# SPDX-License-Identifier: BSD-3-Clause
"""StellarSEDComponent: composite stellar population assembly from SFH and SSP.

Merges the SFH and SSP sub-modules into a single ``SEDComponent``.

Currently supported models:

- ``sfh_model="tsnorm"`` (truncated skew-normal SFH, no GP field)
- ``metallicity_model="delta"`` (single ``met_logzsol`` scalar)
- ``sps_backend="dsps"`` (DSPS native CSP integration)
- ``field=False``

The component publishes derived quantities (stellar mass, SFR history, etc.)
in ``state.derived`` that downstream components (dust, nebular, radio, X-ray)
read to compute their own emission.

Architectural note: the SSP grid is held on the component instance
(constructor field) and treated as a fixed input baked in at construction time,
not an output of a separate precompute step.

See ``docs/dev/20260404-refactor.md`` for the migration plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.stellar.sfh.gp_sfh import log_age_grid_step, make_log_age_grid
from tengri.components.stellar.sfh.metallicity_history import (
    massmap_box_metallicity,
    massmap_lin_metallicity,
    metallicity_bins_continuity_on_ssp_grid,
    metallicity_bins_on_ssp_grid,
    psb_two_step_metallicity,
    tabulated_metallicity_on_ssp_grid,
    two_step_metallicity,
)
from tengri.components.stellar.sps.dsps_wrapper import (
    LSUN_ERG_PER_S,
    SSPData,
    compute_log_z_evolving,
    compute_surviving_mass,
    effective_metallicity,
    has_alpha_grid,
    interpolate_alpha_only,
    interpolate_mass_remaining,
)

# Default time bins for ``metallicity_model="bins"`` /
# ``"bins_continuity"`` — log-spaced from 1 Myr to 13.7 Gyr,
# 7 edges → 6 bins, matching ``MET_REGISTRY``'s
# ``_N_MET_BINS_DEFAULT``.
_DEFAULT_MET_BIN_EDGES_LOG_YR = jnp.array([6.0, 7.5, 8.5, 9.0, 9.5, 9.9, 10.14])
from tengri.parameters.translate import LOG10_ZSUN
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.utils.physics_constants import C_AA, H_PLANCK

__all__ = [
    "StellarSEDComponent",
    "StellarSEDComponentConfig",
    "StellarSEDComponentState",
]

# Lyman limit — wavelengths below this contribute to the ionising
# photon rate (matches :mod:`tengri.components.nebular.ionizing_spectrum`).
_HI_LIMIT_AA: float = 911.76


@dataclass(frozen=True)
class StellarSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for :class:`StellarSEDComponent`.

    Parameters
    ----------
    name : str
        Diagnostic identifier.
    sfh_model : str
        Registered SFH name. Currently supports ``"tsnorm"``, ``"dpl"``,
        ``"continuity"``, ``"dirichlet"``, ``"dense_basis"``, and several
        parametric/bursty variants.
    field : bool
        If ``True``, applies stochastic log-normal GP modulation to the mean SFH.
        Default ``False`` (no field).
    n_grid : int
        Lookback-time grid resolution for SFH evaluation and the published
        ``state.derived["sfh_grid_lbt_yr"]`` array.
    metallicity_model : str
        Metallicity evolution model. Currently supports ``"delta"`` (constant Z),
        ``"ramp"`` (linear Z(t)), ``"two_step"`` (step function), ``"bins"``
        (piecewise-constant per age bin), ``"table"`` (user-provided), and
        ``"chem_evol"`` (closed-box chemical evolution).
    sps_backend : str
        Stellar population synthesis backend. Currently supports ``"dsps"``
        (DSPS native triweight-MDF CSP integration).
    use_alpha_grid : bool
        Whether the SSP grid carries an α/Fe axis. Currently ``False``.
    lgmet_scatter : float
        Gaussian scatter in log10(Z) (dex) for the DSPS triweight kernel.
        Default 0.2 dex matches Prospector / DSPS convention.
    """

    name: str = "stellar"
    sfh_model: str = "tsnorm"
    field: bool = False
    n_grid: int = 256
    metallicity_model: str = "delta"
    sps_backend: str = "dsps"
    use_alpha_grid: bool = False
    lgmet_scatter: float = 0.2
    # Number of bins for ``metallicity_model="bins"`` /
    # ``"bins_continuity"``. Defaults to 6 to match
    # ``MET_REGISTRY``'s ``_N_MET_BINS_DEFAULT`` and the
    # ``met_bin_<i>`` / ``met_d_log_z_<i>`` parameter declarations.
    met_n_bins: int = 6
    # Bin edges in ``log10(age/yr)``, sorted ascending. Used by the
    # ``"bins"`` and ``"bins_continuity"`` metallicity modes.
    # ``None`` falls back to :data:`_DEFAULT_MET_BIN_EDGES_LOG_YR`
    # (log-spaced from 1 Myr to 13.7 Gyr).
    met_bin_edges_log_yr: Any = None
    # User-provided Z(t) table for ``metallicity_model="table"``.
    # ``met_table_log_age_yr`` is the table's age axis in log10(age/yr),
    # sorted ascending; ``met_table_log_z_abs`` is absolute log10(Z) at
    # each table age. Both required if and only if
    # ``metallicity_model == "table"``.
    met_table_log_age_yr: Any = None
    met_table_log_z_abs: Any = None


@dataclass(frozen=True)
class StellarSEDComponentState(SEDComponentState):
    """Marker state. SSP tensors are held on the component instance.

    The ``precompute`` method returns an empty marker or (when wave_precomp
    is enabled) a state carrying the pre-computed SSP×filter LUT (fixed-z)
    or ztable (free-z). When ``approx=SpectrumPrecomp()`` is set, it instead
    carries the pre-rebinned SSP×pixel LUT (``ssp_spec_lut``, fixed-z) or
    its redshift table (``ssp_spec_ztable``, free-z) — the spectroscopic
    analogue of the photometric LUT.
    """

    name: str = "stellar"
    ssp_phot_lut: Any | None = None
    ssp_phot_ztable: Any | None = None
    # Phase 5 (SpectrumPrecomp): SSP flux pre-rebinned to spectrum pixel
    # centres in the galaxy rest frame. ``ssp_spec_lut`` is a
    # :class:`SpectroscopicPrecomputation` (fixed-z); ``ssp_spec_ztable``
    # is a :class:`SpectroscopicZTable` (free-z).
    ssp_spec_lut: Any | None = None
    ssp_spec_ztable: Any | None = None


@dataclass(frozen=True)
class StellarSEDComponent:
    """SEDComponent adapter for stellar emission.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX. The ``SSPData``
    NamedTuple registers as a JAX pytree, so ``self.ssp_data`` is a
    leaf-set of traced arrays under JIT.

    **Pipeline ordering**: stellar runs **first** in any chain. It
    writes ``state.sed_intrinsic`` from scratch and publishes the full
    set of stellar quantities other components consume.

    Construction
    ------------
    ``ssp_data`` is required at construction time. The component is a
    frozen dataclass; build it once at session start and reuse::

        ssp = load_ssp_data("data/ssp_miles.h5")
        stellar = StellarSEDComponent(ssp_data=ssp)
        result = run_components([stellar, ...], state, params)

    Cross-component publications (``state.derived``)
    ------------------------------------------------
    These keys are the stable contract every downstream component relies
    on.

    - ``log_mstar`` (scalar, dex) — log10(surviving stellar mass / Msun).
      Falls back to ``log_mstar_formed`` when the SSP grid lacks a
      ``ssp_mass_remaining`` table.
    - ``log_mstar_formed`` (scalar, dex) — log10(formed mass / Msun).
    - ``sfr`` (scalar, Msun/yr) — SFR at lookback ≈ 0 (i.e. the youngest
      grid point of ``sfr_history``).
    - ``sfr_10myr`` (scalar, Msun/yr) — time-weighted SFR over the last
      10 Myr of the SFH on the lookback grid.
    - ``sfr_100myr`` (scalar, Msun/yr) — same for 100 Myr.
    - ``L_age`` (ndarray, shape ``(n_age,)``, erg/s) — bolometric L per
      SSP age bin (∫ L_ν dν).
    - ``lnu_age`` (ndarray, shape ``(n_age, n_wave)``, erg/s/Hz) —
      per-age L_nu cube. Memory cost ~3 MB for n_age=140, n_wave=2700.
    - ``nion`` (scalar, photons/s) — ionising photon production rate
      (∫_{λ<911.76 Å} L_ν / (hν) dν, total over all ages).
    - ``sfh_grid_lbt_yr`` (ndarray, shape ``(n_grid,)``, yr) — SFH
      lookback-time grid (log-spaced, 1e5 yr → AGEMAX_YR).
    - ``sfr_history`` (ndarray, shape ``(n_grid,)``, Msun/yr) — SFR on
      the SFH grid.
    - ``log_metallicity_history`` (ndarray, shape ``(n_grid,)``, dex) —
      per-time-bin metallicity (constant for ``metallicity_model="delta"``).
    - ``stellar_phot_lnu_precomp`` (ndarray, shape ``(n_filter,)``, erg/s/Hz) —
      stellar contribution to photometry from the LUT. Published only when
      ``approx=WavePrecomp()`` is set at model construction.
    """

    config: StellarSEDComponentConfig = field(default_factory=StellarSEDComponentConfig)
    ssp_data: SSPData | None = None
    name: str = "stellar"
    parameter_prefix: tuple[str, ...] = ("sfh_", "met_", "chem_")
    _state: StellarSEDComponentState | None = None

    def citations(self) -> tuple[str, ...]:
        """The stellar component is structurally built on DSPS; SFH-family
        and SSP-grid citations are config-driven via
        :mod:`tengri.citations.associations`."""
        return ("dsps",)

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns.

        Pulled from :data:`tengri.components.stellar.sfh.registry.SFH_REGISTRY`
        for the configured ``sfh_model`` plus a metallicity block keyed
        by ``metallicity_model``. Field parameters are added when
        ``config.field`` is ``True``.
        """
        # Lazy-import the registries so this module remains importable even
        # if the SFH registry temporarily fails to build.
        from tengri.components.stellar.sfh.met_registry import MET_REGISTRY
        from tengri.components.stellar.sfh.registry import SFH_REGISTRY

        if self.config.sfh_model not in SFH_REGISTRY:
            raise ValueError(
                f"sfh_model={self.config.sfh_model!r} not in SFH_REGISTRY. "
                f"Available: {list(SFH_REGISTRY.keys())}"
            )
        if self.config.metallicity_model not in MET_REGISTRY:
            raise ValueError(
                f"metallicity_model={self.config.metallicity_model!r} not in MET_REGISTRY. "
                f"Available: {list(MET_REGISTRY.keys())}"
            )

        decls: list[ParamDeclaration] = []
        sfh_spec = SFH_REGISTRY[self.config.sfh_model]
        for pname, pdef in sfh_spec.params.items():
            decls.append(ParamDeclaration(pname, pdef.default, pdef.description))

        met_spec = MET_REGISTRY[self.config.metallicity_model]
        for pname, pdef in getattr(met_spec, "params", {}).items():
            decls.append(ParamDeclaration(pname, pdef.default, pdef.description))

        if self.config.field:
            field_spec = SFH_REGISTRY.get("field")
            if field_spec is not None:
                for pname, pdef in field_spec.params.items():
                    decls.append(ParamDeclaration(pname, pdef.default, pdef.description))

        return decls

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this stellar component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.
        """
        return (
            DerivedKey("log_mstar", "dex", "log10(surviving stellar mass / Msun)"),
            DerivedKey("log_mstar_formed", "dex", "log10(formed stellar mass / Msun)"),
            DerivedKey("sfr", "Msun/yr", "SFR at lookback ~ 0"),
            DerivedKey("sfr_10myr", "Msun/yr", "Time-weighted SFR over last 10 Myr"),
            DerivedKey("sfr_100myr", "Msun/yr", "Time-weighted SFR over last 100 Myr"),
            DerivedKey("L_age", "erg/s", "Bolometric L per SSP age bin"),
            DerivedKey("lnu_age", "erg/s/Hz", "Per-age L_nu cube, shape (n_age, n_wave)"),
            DerivedKey("ssp_ages_yr", "yr", "SSP age axis"),
            DerivedKey("age_weights", "Msun", "CSP mass weights per SSP age bin"),
            DerivedKey("nion", "photons/s", "Ionizing photon rate (lambda < 911.76 A)"),
            DerivedKey("sfh_grid_lbt_yr", "yr", "SFH lookback-time grid"),
            DerivedKey("sfr_history", "Msun/yr", "SFR on SFH grid"),
            DerivedKey("log_metallicity_history", "dex", "log10(Z) per SFH time bin"),
            DerivedKey(
                "stellar_phot_lnu_precomp",
                "erg/s/Hz",
                "stellar contribution to photometry from LUT (approx.wave_precomp only)",
            ),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
        redshift_spec: dict[str, Any] | None = None,
        spec_wave_obs: jnp.ndarray | None = None,
    ) -> StellarSEDComponentState:
        """Build SSP×filter LUT (WavePrecomp) or SSP×pixel LUT (SpectrumPrecomp).

        Reads the ``wave_precomp`` / ``spectrum_precomp`` flags from
        ``approx``. For ``wave_precomp`` (with ``filters``), calls
        :func:`precompute_photometry` (fixed-z, Phase 3b) or
        :func:`precompute_photometry_ztable` (free-z, Phase 3c-1). For
        ``spectrum_precomp`` (with ``spec_wave_obs``), calls
        :func:`precompute_spectroscopy` (fixed-z) or
        :func:`precompute_spectroscopy_ztable` (free-z), pre-rebinning the
        SSP grid to the spectrum pixel centres in the galaxy rest frame
        (Phase 5). Otherwise returns an empty state marker.

        Parameters
        ----------
        approx : Mapping[str, bool] | None
            Approximation flags. Reads ``"wave_precomp"`` and
            ``"spectrum_precomp"``.
        filters : tuple of (filter_wave_obs, filter_trans) pairs, optional
            Required when ``wave_precomp=True``. Tuple shape:
            ((fw_0, ft_0), (fw_1, ft_1), ...) where each pair is a pair of
            1-D arrays. The filter_wave is observed-frame.
        redshift_spec : dict[str, Any] | None
            Redshift specification for precomputation. If None or
            mode="fixed", builds a fixed-z LUT (Phase 3b).
            - mode="fixed", value=float: builds LUT at that fixed z.
            - mode="free", z_min=float, z_max=float, n_z=int: builds
              ztable via precompute_photometry_ztable with the given grid.
        spec_wave_obs : array_like, shape (n_pix,), optional
            Observed-frame spectrum pixel wavelengths [Angstrom]. Required
            when ``spectrum_precomp=True``.
        """
        del wave_grid
        approx = approx or {}

        # Phase 5: SpectrumPrecomp — pre-rebin SSP to spectrum pixel centres.
        if approx.get("spectrum_precomp"):
            return self._precompute_spectrum(spec_wave_obs, redshift_spec)

        if not approx.get("wave_precomp"):
            return StellarSEDComponentState(name=self.name)

        # Phase 3b/3c: requires filters at construction.
        if filters is None or self.ssp_data is None:
            # Can't build LUT without filters or SSP grid. Fall back to no-op.
            return StellarSEDComponentState(name=self.name)

        filter_waves, filter_trans = zip(*filters, strict=False)
        filter_list = [jnp.asarray(fw) for fw in filter_waves]
        filter_trans_list = [jnp.asarray(ft) for ft in filter_trans]

        # Dispatch: fixed-z (Phase 3b) or free-z (Phase 3c-1)
        if redshift_spec is None or redshift_spec.get("mode") == "fixed":
            # Phase 3c-3a: build the fixed-z LUT at the source's z so the
            # filter passband is correctly redshifted into the rest frame.
            # This aligns fixed-mode with free-mode semantics — both LUTs
            # carry the filter integral at the source's z. Cosmology
            # ``(1+z)/(4π·dl²)`` is applied in :meth:`Observation.predict_via_precomp`.
            from tengri.components.stellar.sps.precompute import precompute_photometry

            z_source = redshift_spec.get("value", 0.0) if redshift_spec else 0.0
            lut = precompute_photometry(
                ssp_data=self.ssp_data,
                filter_waves=filter_list,
                filter_trans=filter_trans_list,
                redshift=z_source,
                dl_cm=1.0,  # placeholder; cosmology applied at projection time
                taylor_correction=True,  # Phase 3c-3c: enables Ψ moment for dust LUT
            )
            return StellarSEDComponentState(name=self.name, ssp_phot_lut=lut)

        else:  # mode == "free"
            # Phase 3c-1 path: build ztable for free-z interpolation.
            from tengri.components.stellar.sps.precompute import (
                precompute_photometry_ztable,
            )

            ztable = precompute_photometry_ztable(
                ssp_data=self.ssp_data,
                filter_waves=filter_list,
                filter_trans=filter_trans_list,
                z_min=redshift_spec.get("z_min", 0.001),
                z_max=redshift_spec.get("z_max", 3.0),
                n_z=redshift_spec.get("n_z", 100),
                apply_igm=False,
                taylor_correction=True,  # Phase 3c-3c-v: Ψ moment for dust LUT
            )
            return StellarSEDComponentState(name=self.name, ssp_phot_ztable=ztable)

    def _precompute_spectrum(
        self,
        spec_wave_obs: jnp.ndarray | None,
        redshift_spec: dict[str, Any] | None,
    ) -> StellarSEDComponentState:
        """Build the SSP×pixel LUT for ``approx=SpectrumPrecomp()`` (Phase 5).

        Pre-rebins the SSP flux cube to the spectrum pixel centres in the
        galaxy rest frame. Unlike the photometric LUT, **no Taylor moment
        is needed**: a spectrum pixel is a single wavelength, so dust
        attenuation ``A(λ_pix)`` evaluated at the pixel centre is exact —
        there is no wide-kernel integral to factorise.

        Fixed-z builds a single :class:`SpectroscopicPrecomputation`; free-z
        builds a :class:`SpectroscopicZTable` so the rest-frame pixel grid
        ``wave_obs / (1 + z)`` can be interpolated at runtime.
        """
        if spec_wave_obs is None or self.ssp_data is None:
            # No grid or no SSP — fall back to the full-grid path.
            return StellarSEDComponentState(name=self.name)

        spec_wave_obs = jnp.asarray(spec_wave_obs)

        if redshift_spec is None or redshift_spec.get("mode") == "fixed":
            from tengri.components.stellar.sps.precompute import precompute_spectroscopy

            z_source = redshift_spec.get("value", 0.0) if redshift_spec else 0.0
            lut = precompute_spectroscopy(
                ssp_data=self.ssp_data,
                wave_obs_pixels=spec_wave_obs,
                redshift=z_source,
                dl_cm=1.0,  # placeholder; cosmology applied at projection time
            )
            return StellarSEDComponentState(name=self.name, ssp_spec_lut=lut)

        # mode == "free": build the redshift table.
        from tengri.components.stellar.sps.precompute import precompute_spectroscopy_ztable

        ztable = precompute_spectroscopy_ztable(
            ssp_data=self.ssp_data,
            wave_obs_pixels=spec_wave_obs,
            z_min=redshift_spec.get("z_min", 0.001),
            z_max=redshift_spec.get("z_max", 3.0),
            n_z=redshift_spec.get("n_z", 100),
        )
        return StellarSEDComponentState(name=self.name, ssp_spec_ztable=ztable)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        """Compute stellar SED and publish derived quantities.

        Assembles the composite stellar population by convolving the
        star formation history with SSP templates via DSPS. Publishes
        stellar mass, ionizing photon rate, and age-dependent quantities
        for downstream components.

        Parameters
        ----------
        state : ForwardState
            Initial pipeline state. Carries ``wave`` (rest-frame Å); the
            component reads ``redshift`` from ``params`` (allowlist).
        params : mapping
            Receives ``sfh_*``, ``met_*``, ``chem_*`` keys plus the bare
            ``redshift`` from :data:`BARE_NAME_ALLOWLIST`.
        ssp_data : Any | None, optional
            SSP data passed as a JIT runtime input (Phase 4-B threading).
            When provided, uses this instead of ``self.ssp_data``. Enables
            SSP arrays to be ``Parameter`` ops in compiled code rather than
            ``Constant`` ops, reducing HLO size and compile time.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` set and 13 derived keys
            published.
        """
        # Phase 4-B: use ssp_data if threaded as JIT input, otherwise fall
        # back to the closure (for non-JIT paths).
        ssp = ssp_data if ssp_data is not None else self.ssp_data
        if ssp is None:
            raise ValueError(
                "StellarSEDComponent.apply requires ssp_data set on the component. "
                "Pass it at construction: StellarSEDComponent(ssp_data=ssp)."
            )
        # SFH models are routed through SFH_REGISTRY's internal_param_map.
        # Each model is validated against legacy DSPS path via
        # tests/integration/test_stellar_integration.py.
        _SUPPORTED_SFH = (
            "tsnorm",
            "dpl",
            "continuity",
            "dirichlet",
            "dense_basis",
            "lnorm",
            "snorm",
            "snorm_burst",
            "tsnorm_burst",
            "norm",
            "const",
            "const_exp",
            "continuity_flex",
            "psb",
            "delayed_bq",
            "dense_basis_pure",
            "exp",
            "dexp",
            "tau",
            "delayed",
            "periodic",
            "buat08",
        )
        if self.config.sfh_model not in _SUPPORTED_SFH:
            raise NotImplementedError(
                f"sfh_model={self.config.sfh_model!r} not yet validated "
                f"against legacy DSPS. Supported modes: {_SUPPORTED_SFH}."
            )
        _SUPPORTED_MET = (
            "delta",
            "ramp",
            "chem_evol",
            "two_step",
            "psb_two_step",
            "bins",
            "bins_continuity",
            "table",
        )
        if self.config.metallicity_model not in _SUPPORTED_MET:
            raise NotImplementedError(
                f"metallicity_model={self.config.metallicity_model!r} not in "
                f"{_SUPPORTED_MET}. Add a branch in StellarSEDComponent.apply() "
                f"per docs/dev/20260506-met-mode-wiring-blueprint.md."
            )
        ssp_ages_yr = (10.0**ssp.ssp_lg_age_gyr) * 1e9
        n_grid = self.config.n_grid

        # ── 1. SFH lookback-time grid ───────────────────────────────────
        # Use the SAME grid construction as the legacy SEDModel path
        # (forward/sed_model.py:467). ``make_log_age_grid`` returns a
        # uniform grid in log10(age/yr) over [6.0, 10.14] (1 Myr →
        # 13.8 Gyr). This is critical for ``field=True`` parity:
        # ``compute_field_gp`` keys on n_grid + d_log_age to build
        # the GP correlation kernel, so both paths must construct
        # the grid identically. See tests/integration/test_stellar_integration.py.
        log_age_grid = make_log_age_grid(n_grid)
        sfh_lbt_grid = 10.0**log_age_grid

        # ── 2. Evaluate mean SFH on grid (registry-driven) ──────────────
        # Translate user-facing public params → SFH-function kwargs via
        # the registry's ``internal_param_map``: each entry is
        # ``(internal_name, scale, offset)`` and the conversion is
        # ``internal = public * scale + offset``. This ensures both this
        # component and legacy SEDModel paths see the same units and naming.
        from tengri.components.stellar.sfh.registry import SFH_REGISTRY

        sfh_spec = SFH_REGISTRY[self.config.sfh_model]
        sfh_kwargs = {}
        for public_name, (internal_name, scale, offset) in sfh_spec.internal_param_map.items():
            if public_name in params:
                raw = params[public_name]
            else:
                # Fall back to the registry default for any declared parameter
                # the caller omitted. Required so callers that pass a partial
                # params dict (e.g. the low-level run_components path, or flat-
                # kwarg specs predating the dpl/lnorm ``age`` anchor of #514)
                # still resolve every positional argument of the SFH callable.
                # Mirrors the dense_basis age_universe injection just below.
                pdef = sfh_spec.params.get(public_name)
                default_scalar = pdef.default.default if pdef is not None else None
                if default_scalar is None:
                    continue
                raw = default_scalar
            sfh_kwargs[internal_name] = jnp.asarray(raw) * scale + offset

        # Mode-specific settings that are NOT free parameters.
        # ``dense_basis`` needs an explicit ``age_universe_yr`` derived
        # from the configured cosmology; default of 13.47 Gyr matches
        # the registry setting (FlatLambdaCDM, H0=70, Omega_m=0.3, z=0).
        if self.config.sfh_model == "dense_basis":
            age_universe_gyr = sfh_spec.settings.get("sfh_db_age_universe_gyr", 13.47)
            sfh_kwargs["age_universe_yr"] = float(age_universe_gyr) * 1e9

        sfr_history = sfh_spec.fn(sfh_lbt_grid, **sfh_kwargs)

        # ── 2b. GP-field modulation ───────────────────────────────────
        # Multiplicative log-normal modulation: SFR_total = SFR_mean ×
        # exp(x(t) - K(0)/2), where x(t) is a PSD-governed Gaussian
        # process and K(0)/2 is the lognormal bias correction so the
        # ensemble mean equals SFR_mean. ``compute_field_gp`` lives in
        # the SFH registry next to the prior on ``sfh_field_xi``.
        if self.config.field:
            from tengri.components.stellar.sfh.registry import compute_field_gp

            psd_sigma = jnp.asarray(params["sfh_field_psd_sigma"])
            psd_tau_myr = jnp.asarray(params["sfh_field_psd_tau_myr"])
            xi = jnp.asarray(params.get("sfh_field_xi", jnp.zeros(n_grid)))
            psd_tau_yr = psd_tau_myr * 1e6
            # JIT-safe: ``log_age_grid`` is traced under jit so indexing +
            # float() would raise ConcretizationTypeError. ``log_age_grid_step``
            # recomputes the step from static ``n_grid`` and module constants.
            d_log_age = log_age_grid_step(n_grid)
            gp_x, k0_half = compute_field_gp(
                xi, psd_sigma, psd_tau_yr, n_grid, d_log_age, field_model="drw"
            )
            sfr_history = sfr_history * jnp.exp(gp_x - k0_half)

        # ── 3. Resample to SSP age grid for CSP integration ─────────────
        # For deterministic (non-GP) parametric SFHs, evaluate the analytic
        # shape on ``ssp_ages_yr`` directly. The SSP age grid is linear-spaced
        # at 1 Myr cadence (13700 bins over 1 Myr → 13.7 Gyr), so SF-onset
        # cutoffs at ``t_lookback = age`` land on grid points instead of
        # being smeared across the coarser ~3 % log-spaced bins of
        # ``sfh_lbt_grid``. The closed form also self-normalises through
        # ``_renormalize_to_mass``, so total mass formed = ``10**log_total_mass``
        # exactly. See suchethac/tengri#385.
        #
        # The GP-field path still goes through the log grid: ``compute_field_gp``
        # builds its DRW kernel keyed on ``n_grid`` and ``d_log_age``, so the
        # GP draw lives on the lookback grid by construction.
        if self.config.field:
            sfr_on_ssp = jnp.interp(ssp_ages_yr, sfh_lbt_grid, sfr_history)
        else:
            sfr_on_ssp = sfh_spec.fn(ssp_ages_yr, **sfh_kwargs)

        # ── 5. Cosmology: t_obs from redshift ───────────────────────────
        # ``age_at_z`` is JIT-compatible (pure JAX under the hood); keep
        # everything as JAX arrays so the whole apply() stays traceable.
        from tengri.cosmology import age_at_z as _age_at_z

        z = jnp.asarray(params.get("redshift", 0.0))
        t_obs_gyr = jnp.asarray(_age_at_z(z)).reshape(())

        # ── 4. Metallicity history Z(t) on SFH grid + per-SSP-age ───────
        # delta: scalar absolute log10(Z), constant in time.
        # ramp: linear interpolation between two endpoints.
        # chem_evol: closed-box gas regulator — Z(t) derived from SFH self-
        # consistently. Mirrors legacy sed_model.py:3578-3592.
        # 4D α-enhanced SSPs: collapse the [α/Fe] axis to a single
        # plane once, here, then pass the resulting 3D ssp_flux to the
        # downstream DSPS kernel (closes #226). The Z marginalisation
        # remains the standard lognormal MDF for every met_mode, so the
        # 4D and 3D paths share the same Z bookkeeping — only the
        # ``ssp_flux`` that DSPS sees differs.
        _alpha_collapse_active = has_alpha_grid(ssp)
        if _alpha_collapse_active:
            _alpha_fe_value = jnp.asarray(params.get("met_alpha_fe", 0.0))
            ssp_flux_for_csp = interpolate_alpha_only(
                ssp.ssp_flux, ssp.ssp_alpha_fe, _alpha_fe_value
            )
        else:
            ssp_flux_for_csp = ssp.ssp_flux

        if self.config.metallicity_model == "delta":
            # Apply alpha-Fe enhancement via effective_metallicity for
            # 3D SSP grids (no native α axis). Mirrors the legacy
            # ``interp_met_alpha_dispatch`` fallback in
            # ``forward/pipeline.py:239``: when no α grid is available,
            # the α-shift is folded into log_z via Salaris+05 / DSPS
            # canonical relation. For 4D α-grid SSPs the α axis has
            # already been collapsed above, so we use ``met_logzsol``
            # directly without the effective-Z approximation.
            alpha_fe = jnp.asarray(params.get("met_alpha_fe", 0.0))
            if _alpha_collapse_active:
                log_z_abs_scalar = jnp.asarray(params["met_logzsol"]) + LOG10_ZSUN
            else:
                log_z_eff = effective_metallicity(jnp.asarray(params["met_logzsol"]), alpha_fe)
                log_z_abs_scalar = log_z_eff + LOG10_ZSUN
            log_metallicity_history = jnp.full(n_grid, log_z_abs_scalar)
            lgmet_on_ssp_ages = jnp.full_like(ssp_ages_yr, log_z_abs_scalar)
            log_z_for_mr = log_z_abs_scalar
        elif self.config.metallicity_model == "ramp":
            log_z_init_abs = jnp.asarray(params["met_logzsol_0"]) + LOG10_ZSUN
            log_z_final_abs = jnp.asarray(params["met_logzsol_final"]) + LOG10_ZSUN
            # Build the per-age metallicity ramp on both grids (SFH grid for
            # diagnostics + SSP grid for the CSP integral).
            sfh_lg_age_gyr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0)) - 9.0
            log_metallicity_history = compute_log_z_evolving(
                sfh_lg_age_gyr, log_z_init_abs, log_z_final_abs, t_obs_gyr
            )
            lgmet_on_ssp_ages = compute_log_z_evolving(
                ssp.ssp_lg_age_gyr, log_z_init_abs, log_z_final_abs, t_obs_gyr
            )
            # For mass-remaining interpolation use the present-day metallicity
            # (newest stars dominate the mass-loss correction).
            log_z_for_mr = log_z_final_abs
        elif self.config.metallicity_model == "two_step":
            # Sigmoid-smoothed step at ``met_step_age_gyr``. Stars older than
            # the step get ``met_logzsol_old``, younger get ``met_logzsol_young``.
            log_z_old_abs = jnp.asarray(params["met_logzsol_old"]) + LOG10_ZSUN
            log_z_young_abs = jnp.asarray(params["met_logzsol_young"]) + LOG10_ZSUN
            step_age_gyr = jnp.asarray(params["met_step_age_gyr"])
            lgmet_on_ssp_ages = two_step_metallicity(
                ssp.ssp_lg_age_gyr, log_z_old_abs, log_z_young_abs, step_age_gyr
            )
            sfh_lg_age_gyr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0)) - 9.0
            log_metallicity_history = two_step_metallicity(
                sfh_lg_age_gyr, log_z_old_abs, log_z_young_abs, step_age_gyr
            )
            # Present-day Z (youngest SSP age, lookback ≈ 0).
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "psb_two_step":
            # Step tied to the PSB SFH burst onset
            # (``sfh_psb_burstage_gyr``). Pre-burst stars get
            # ``met_logzsol_old``, burst-and-younger get
            # ``met_logzsol_burst``.
            log_z_old_abs = jnp.asarray(params["met_logzsol_old"]) + LOG10_ZSUN
            log_z_burst_abs = jnp.asarray(params["met_logzsol_burst"]) + LOG10_ZSUN
            burstage_gyr = jnp.asarray(params.get("sfh_psb_burstage_gyr", 1.0))
            lgmet_on_ssp_ages = psb_two_step_metallicity(
                ssp.ssp_lg_age_gyr, log_z_old_abs, log_z_burst_abs, burstage_gyr
            )
            sfh_lg_age_gyr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0)) - 9.0
            log_metallicity_history = psb_two_step_metallicity(
                sfh_lg_age_gyr, log_z_old_abs, log_z_burst_abs, burstage_gyr
            )
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "bins":
            # Piecewise-constant Z per age bin. Bin edges from config
            # (defaults to log-spaced 1 Myr → 13.7 Gyr); per-bin
            # metallicities from ``met_bin_<i>`` params (i = 0..N-1).
            n_bins = self.config.met_n_bins
            bin_edges_log_yr = (
                self.config.met_bin_edges_log_yr
                if self.config.met_bin_edges_log_yr is not None
                else _DEFAULT_MET_BIN_EDGES_LOG_YR
            )
            metallicities_abs = (
                jnp.stack([jnp.asarray(params[f"met_bin_{i}"]) for i in range(n_bins)])
                + LOG10_ZSUN
            )
            lgmet_on_ssp_ages = metallicity_bins_on_ssp_grid(
                ssp.ssp_lg_age_gyr, jnp.asarray(bin_edges_log_yr), metallicities_abs
            )
            # SFH-grid history: same primitive applied to sfh_lbt_grid.
            sfh_lg_age_yr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0))
            log_metallicity_history = metallicity_bins_on_ssp_grid(
                sfh_lg_age_yr - 9.0, jnp.asarray(bin_edges_log_yr), metallicities_abs
            )
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "bins_continuity":
            # Cumulative delta-log-Z steps from oldest bin to youngest.
            # ``met_logzsol_base`` is the oldest bin; ``met_d_log_z_<i>``
            # are the N-1 steps. Reuses the binning primitive with
            # convolved metallicities.
            n_bins = self.config.met_n_bins
            bin_edges_log_yr = (
                self.config.met_bin_edges_log_yr
                if self.config.met_bin_edges_log_yr is not None
                else _DEFAULT_MET_BIN_EDGES_LOG_YR
            )
            log_z_base_abs = jnp.asarray(params["met_logzsol_base"]) + LOG10_ZSUN
            d_log_z = jnp.stack(
                [jnp.asarray(params[f"met_d_log_z_{i}"]) for i in range(n_bins - 1)]
            )
            lgmet_on_ssp_ages = metallicity_bins_continuity_on_ssp_grid(
                ssp.ssp_lg_age_gyr, jnp.asarray(bin_edges_log_yr), log_z_base_abs, d_log_z
            )
            sfh_lg_age_yr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0))
            log_metallicity_history = metallicity_bins_continuity_on_ssp_grid(
                sfh_lg_age_yr - 9.0,
                jnp.asarray(bin_edges_log_yr),
                log_z_base_abs,
                d_log_z,
            )
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "table":
            # User-provided Z(t) table on the component's config (settings,
            # not JAX params — the table is constructor-time data).
            if self.config.met_table_log_age_yr is None or self.config.met_table_log_z_abs is None:
                raise ValueError(
                    "metallicity_model='table' requires met_table_log_age_yr "
                    "and met_table_log_z_abs on StellarSEDComponentConfig "
                    "(both arrays in absolute log10(Z) and log10(age/yr))."
                )
            met_log_age_yr = jnp.asarray(self.config.met_table_log_age_yr)
            met_log_z_abs = jnp.asarray(self.config.met_table_log_z_abs)
            lgmet_on_ssp_ages = tabulated_metallicity_on_ssp_grid(
                ssp.ssp_lg_age_gyr, met_log_age_yr, met_log_z_abs
            )
            sfh_lg_age_yr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0))
            log_metallicity_history = tabulated_metallicity_on_ssp_grid(
                sfh_lg_age_yr - 9.0, met_log_age_yr, met_log_z_abs
            )
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "massmap_lin":
            # Linear metallicity tied to cumulative stellar mass formed
            # (ProSpect Bellstedt+2020 massmap_lin model).
            log_z_start_abs = jnp.asarray(params["met_logzsol_start"]) + LOG10_ZSUN
            log_z_final_abs = jnp.asarray(params["met_logzsol_final"]) + LOG10_ZSUN
            # Per-age metallicity on the SSP grid
            lgmet_on_ssp_ages = massmap_lin_metallicity(
                ssp.ssp_lg_age_gyr, ssp_ages_yr, sfr_on_ssp, log_z_start_abs, log_z_final_abs
            )
            # Z(t) on the SFH grid for diagnostics
            sfh_lg_age_gyr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0)) - 9.0
            log_metallicity_history = massmap_lin_metallicity(
                sfh_lg_age_gyr, sfh_lbt_grid, sfr_history, log_z_start_abs, log_z_final_abs
            )
            # Mass-remaining interpolation: use present-day Z (youngest SSP age).
            log_z_for_mr = lgmet_on_ssp_ages[0]
        elif self.config.metallicity_model == "massmap_box":
            # Closed-box chemical evolution tied to cumulative stellar mass formed
            # (ProSpect Bellstedt+2020 massmap_box model).
            log_z_start_abs = jnp.asarray(params["met_logzsol_start"]) + LOG10_ZSUN
            log_z_final_abs = jnp.asarray(params["met_logzsol_final"]) + LOG10_ZSUN
            yield_rho = jnp.asarray(params.get("yield", 0.03))
            # Per-age metallicity on the SSP grid
            lgmet_on_ssp_ages = massmap_box_metallicity(
                ssp.ssp_lg_age_gyr,
                ssp_ages_yr,
                sfr_on_ssp,
                log_z_start_abs,
                log_z_final_abs,
                yield_rho,
            )
            # Z(t) on the SFH grid for diagnostics
            sfh_lg_age_gyr = jnp.log10(jnp.maximum(sfh_lbt_grid, 1.0)) - 9.0
            log_metallicity_history = massmap_box_metallicity(
                sfh_lg_age_gyr,
                sfh_lbt_grid,
                sfr_history,
                log_z_start_abs,
                log_z_final_abs,
                yield_rho,
            )
            # Mass-remaining interpolation: use present-day Z (youngest SSP age).
            log_z_for_mr = lgmet_on_ssp_ages[0]
        else:  # chem_evol
            from tengri.components.stellar.sfh.chemical_evolution import (
                chem_evol_metallicity_on_ssp_grid,
                closed_box_metallicity,
            )

            yield_y = float(params.get("chem_yield", 0.03))
            eta_outflow = float(params.get("chem_eta_outflow", 0.0))
            f_gas_init = float(params.get("chem_f_gas_init", 0.9))
            return_frac = float(params.get("chem_return_frac", 0.4))

            # Per-age metallicity on the SSP grid — mirrors legacy
            # sed_model.py:3583. Uses log10(age/yr) on both grids; the
            # SSP grid is ssp.ssp_lg_age_gyr + 9.0.
            ssp_log_ages_yr = ssp.ssp_lg_age_gyr + 9.0
            lgmet_on_ssp_ages = chem_evol_metallicity_on_ssp_grid(
                ssp_log_ages_yr,
                log_age_grid,
                sfr_history,
                yield_y=yield_y,
                eta_outflow=eta_outflow,
                f_gas_init=f_gas_init,
                return_frac=return_frac,
            )
            # Z(t) on the SFH grid for diagnostics — closed_box_metallicity
            # returns log10(Z/Zsun); add LOG10_ZSUN for absolute log10(Z).
            log_metallicity_history = (
                closed_box_metallicity(
                    sfh_lbt_grid,
                    sfr_history,
                    yield_y=yield_y,
                    eta_outflow=eta_outflow,
                    f_gas_init=f_gas_init,
                    return_frac=return_frac,
                )
                + LOG10_ZSUN
            )
            # Mass-remaining interpolation: use present-day Z (youngest SSP age).
            log_z_for_mr = lgmet_on_ssp_ages[0]

        # ── 6. CSP integral via DSPS ────────────────────────────────────
        # We call DSPS directly and use ``result.weights`` — the JOINT
        # (n_met, n_age) probability distribution — instead of the
        # separable approximation in compute_dsps_native_weights. The
        # separable form (lgmet_w × age_w) gave the right marginals but
        # the wrong product for non-trivial age-metallicity correlations,
        # over-scaling the CSP SED by orders of magnitude.
        from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_lognormal_mdf

        # NaN-safe cosmic-time prep mirroring
        # :func:`compute_dsps_age_weights`: when SSP ages exceed
        # ``t_obs`` (typical at z>0 with old SSPs), the implied cosmic
        # time is negative. Bare ``jnp.clip(min=1e-3)`` collapses
        # multiple such bins to the same boundary value, producing a
        # degenerate ``gal_t_table`` that DSPS NaNs on. Instead, we
        # build a strictly-monotonic ramp at the invalid end and zero
        # the SFR there so those bins contribute nothing.
        ssp_age_gyr = ssp_ages_yr / 1e9
        T_TABLE_MIN = 0.01  # Gyr; matches dsps.constants.T_TABLE_MIN
        t_cosmic_raw = t_obs_gyr - ssp_age_gyr
        n_ssp_for_ramp = ssp_ages_yr.shape[0]
        t_cosmic_floor = jnp.maximum(t_cosmic_raw, T_TABLE_MIN)
        valid = t_cosmic_raw > 0.0
        valid_asc = valid[::-1]
        t_cosmic_asc_raw = t_cosmic_floor[::-1]
        sfr_asc_raw = sfr_on_ssp[::-1]
        n_invalid = jnp.sum(~valid_asc)
        idx_pos = jnp.arange(n_ssp_for_ramp)
        is_invalid_pos = idx_pos < n_invalid
        ramp = T_TABLE_MIN + (T_TABLE_MIN * 0.5) * (idx_pos + 1) / jnp.maximum(n_invalid, 1)
        t_cosmic_asc = jnp.where(is_invalid_pos, ramp, t_cosmic_asc_raw)
        sfr_asc = jnp.where(is_invalid_pos, 0.0, sfr_asc_raw)
        total_mass = jnp.maximum(jnp.trapezoid(sfr_asc, t_cosmic_asc * 1e9), 0.0)

        if self.config.metallicity_model == "delta":
            dsps_result = calc_rest_sed_sfh_table_lognormal_mdf(
                gal_t_table=t_cosmic_asc,
                gal_sfr_table=sfr_asc,
                gal_lgmet=log_z_abs_scalar,
                gal_lgmet_scatter=self.config.lgmet_scatter,
                ssp_lgmet=ssp.ssp_lgmet,
                ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
                ssp_flux=ssp_flux_for_csp,
                t_obs=t_obs_gyr,
            )
        else:  # ramp / chem_evol — per-age metallicity table
            from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_met_table

            dsps_result = calc_rest_sed_sfh_table_met_table(
                gal_t_table=t_cosmic_asc,
                gal_sfr_table=sfr_asc,
                gal_lgmet_table=lgmet_on_ssp_ages[::-1],
                gal_lgmet_scatter=self.config.lgmet_scatter,
                ssp_lgmet=ssp.ssp_lgmet,
                ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
                ssp_flux=ssp_flux_for_csp,
                t_obs=t_obs_gyr,
            )

        # ``dsps_result.weights`` is the joint (n_met, n_age) probability
        # distribution (sums to 1) over SSP grid points. The age axis is
        # already aligned with tengri's ssp_flux ordering (ascending
        # lookback age) — no flip needed; DSPS handles the cosmic-time
        # bookkeeping internally before storing weights against the
        # SSP grid.
        joint_weights = dsps_result.weights  # (n_met, n_age)
        # Per-age × per-Msun-formed weighted SSP flux (Lsun/Hz/Msun):
        ssp_flux_at_age = jnp.einsum("ma,maw->aw", joint_weights, ssp_flux_for_csp)
        # Per-age "mass" for downstream per-age operations (dust BC mask).
        # This is the marginalised age distribution × total_mass.
        age_weights = joint_weights.sum(axis=0) * total_mass  # (n_age,) Msun

        # ── 7. Stellar SED in erg/s/Hz ──────────────────────────────────
        # Use DSPS's own ``rest_sed`` (= ``sed_unit_mstar × mstar_obs`` in
        # Lsun/Hz) rather than reconstructing it as
        # ``total_mass × Σ(weights × ssp_flux)``. The two paths are
        # mathematically identical when ``total_mass == mstar_obs`` (both
        # integrate the same SFH), but DSPS computes ``mstar_obs`` via a
        # cumulative-SFH interpolation while tengri's ``total_mass`` is a
        # trapezoid integral over the ramp-zeroed ``sfr_asc`` grid. Using
        # DSPS's value keeps the SED self-consistent with the kernel's
        # own normalisation (closes #394). The per-age cube ``lnu_age``
        # below is retained for downstream per-age operations (dust BC
        # mask, bolometric L_age) and continues to use ``total_mass`` —
        # the per-age sum is consumed before any wavelength-resolved
        # quantity reads it, so any residual ``total_mass / mstar_obs``
        # discrepancy does not propagate into observables.
        sed_intrinsic = dsps_result.rest_sed * LSUN_ERG_PER_S
        lnu_age = total_mass * ssp_flux_at_age * LSUN_ERG_PER_S

        # ── 8. Mass quantities ──────────────────────────────────────────
        log_mstar_formed = jnp.log10(jnp.maximum(jnp.sum(age_weights), 1e-30))
        if ssp.ssp_mass_remaining is not None:
            mr_at_met = interpolate_mass_remaining(
                ssp.ssp_mass_remaining, ssp.ssp_lgmet, log_z_for_mr
            )
            mstar_surv = compute_surviving_mass(age_weights, mr_at_met)
            log_mstar = jnp.log10(jnp.maximum(mstar_surv, 1e-30))
        else:
            log_mstar = log_mstar_formed

        # ── 9. SFR averages on the SFH grid ─────────────────────────────
        sfr_now = sfr_history[0]
        sfr_10myr = _time_weighted_sfr(sfr_history, sfh_lbt_grid, 1e7)
        sfr_100myr = _time_weighted_sfr(sfr_history, sfh_lbt_grid, 1e8)

        # ── 10. Bolometric L per SSP age bin ────────────────────────────
        # ν = c/λ ⟹ |dν| = c/λ² dλ. Trapezoid in wavelength with the
        # frequency Jacobian gives ∫ L_ν dν per age.
        wave = ssp.ssp_wave
        nu_jac = C_AA / (wave**2)
        L_age = jnp.trapezoid(lnu_age * nu_jac[None, :], wave, axis=1)

        # ── 11. Ionising photon production rate (λ < 911.76 Å) ──────────
        # photons/s = ∫_{ν > c/λ_HI} L_ν / (hν) dν, summed over all ages.
        # Mirrors components/nebular/ionizing_spectrum.py:299.
        #
        # Partial-bin correction (#537): when 911.76 Å falls between two
        # grid points (true for BC03's 10 Å sampling: 905 and 915 Å are
        # the bracketing points), a hard ``wave < 911.76`` mask drops
        # the 905 → 911.76 portion of the boundary bin entirely. SSP
        # spectra have a near-discontinuous Lyman drop at 911.76 Å:
        # ionising flux is well-defined right up to the limit, then
        # drops to zero. Linear interpolation between 905 and 915 Å
        # would under-estimate the boundary value (a half-value of the
        # ionising side); the correct partial-bin contribution treats
        # ``L_ν`` as constant from the last ionising grid point up to
        # 911.76 Å — a rectangle, not a trapezium. This matches the
        # physical Lyman discontinuity and produces a Q_H consistent
        # with CIGALE's tabulated ``stellar.n_ly`` to within numerical
        # noise at any SSP grid spacing.
        nu = C_AA / wave
        nu_edge = C_AA / _HI_LIMIT_AA
        integrand = sed_intrinsic / (H_PLANCK * nu)
        ionizing_mask = wave < _HI_LIMIT_AA
        integrand_masked = jnp.where(ionizing_mask, integrand, 0.0)
        # Find the last ionising grid point (wave_below < 911.76) and the
        # first non-ionising one (wave_above ≥ 911.76).
        idx_below = jnp.argmax(jnp.where(ionizing_mask, jnp.arange(len(wave)), -1))
        idx_above = idx_below + 1
        nu_below = nu[idx_below]
        nu_above = nu[idx_above]
        integrand_below = integrand[idx_below]
        # Bulk trapezoid integrates over every adjacent pair (including
        # the (905, 915) boundary bin where one side has ``integrand``
        # and the other is masked to zero — yielding a triangle of
        # area ½·integrand_below·|ν_below − ν_above|). For SSPs with a
        # sharp Lyman discontinuity the *correct* contribution from
        # that bin is a rectangle from ν_edge to ν_below
        # (= integrand_below · |ν_below − ν_edge|). Subtract the
        # triangle, add the rectangle:
        triangle_overcount = 0.5 * integrand_below * jnp.abs(nu_below - nu_above)
        rectangle_correct = integrand_below * jnp.abs(nu_below - nu_edge)
        nion_bulk = jnp.abs(jnp.trapezoid(integrand_masked, nu))
        nion = nion_bulk - triangle_overcount + rectangle_correct

        # ── 11b. Project to pipeline wavelength grid ────────────────
        # When the pipeline runs on a panchromatic grid (radio/X-ray
        # extension via ``make_panchromatic_grid``), ``state.wave`` is
        # wider than ``ssp.ssp_wave``. Both ``sed_intrinsic`` and the
        # per-age cube ``lnu_age`` MUST live on ``state.wave`` so
        # downstream additive emitters (radio, X-ray) and per-age
        # transforms (dust two-component) can broadcast. Linear interp
        # is exact at SSP grid points (panchromatic preserves them) and
        # zero is the physically correct extrapolation outside the SSP
        # range — the SSP templates carry no information there.
        #
        # The shape comparison is Python-level (both arrays exist at
        # trace time), so the no-extension case incurs zero JIT cost.
        if state.wave.shape[0] != wave.shape[0]:
            target = state.wave
            ssp_wave_arr = wave
            outside = (target < ssp_wave_arr[0]) | (target > ssp_wave_arr[-1])
            sed_intrinsic_proj = jnp.where(
                outside, 0.0, jnp.interp(target, ssp_wave_arr, sed_intrinsic)
            )
            from jax import vmap

            lnu_age_proj = vmap(lambda row: jnp.interp(target, ssp_wave_arr, row))(lnu_age)
            lnu_age_proj = jnp.where(outside[None, :], 0.0, lnu_age_proj)
            sed_intrinsic = sed_intrinsic_proj
            lnu_age = lnu_age_proj

        # ── 12b. Stellar photometry LUT (Phase 3b) ─────────────────────
        # When eager precomputation is enabled and the LUT is available,
        # compute stellar_phot_lnu_precomp and publish it to derived.
        derived_overrides = dict(
            log_mstar=log_mstar,
            log_mstar_formed=log_mstar_formed,
            sfr=sfr_now,
            sfr_10myr=sfr_10myr,
            sfr_100myr=sfr_100myr,
            L_age=L_age,
            lnu_age=lnu_age,
            # CSP mass weights (Msun per SSP age bin), summed
            # over the metallicity axis. Published so downstream
            # nebular backends (Cue, CloudyGrid) can call their
            # high-level ``predict_nebular_*(ssp_weights=...)``
            # entry points and derive Q_H + ionising spectrum
            # from the SSP, matching legacy parity.
            age_weights=age_weights,
            nion=nion,
            sfh_grid_lbt_yr=sfh_lbt_grid,
            sfr_history=sfr_history,
            log_metallicity_history=log_metallicity_history,
            # Published for downstream (dust two-component attenuation
            # needs the SSP age axis to apply the BC/diffuse split).
            ssp_ages_yr=ssp_ages_yr,
        )

        if self._state is not None and self._state.ssp_phot_lut is not None:
            # Fixed-z path (Phase 3b) — LUT built at source's z in precompute()
            ssp_phot = self._state.ssp_phot_lut.ssp_phot
            # (n_met, n_age, n_filt) in Lsun/Hz/Msun; sum over metallicity and
            # age axes weighted by joint distribution × total mass.
            # Convert to erg/s/Hz to match sed_intrinsic units.
            stellar_phot_lnu_precomp_rest = (
                total_mass * jnp.einsum("ma,maf->f", joint_weights, ssp_phot) * LSUN_ERG_PER_S
            )
            derived_overrides["stellar_phot_lnu_precomp"] = stellar_phot_lnu_precomp_rest
            # Phase 3c-3c-iv-a: age-resolved per-filter LUT for two-component
            # dust attenuation. Marginalise over metallicity only; preserve
            # the age axis. Shape (n_age, n_filter). Sum over age == the
            # marginalised stellar_phot_lnu_precomp above.
            stellar_phot_lnu_per_age = (
                total_mass * jnp.einsum("ma,maf->af", joint_weights, ssp_phot) * LSUN_ERG_PER_S
            )
            derived_overrides["stellar_phot_lnu_per_age_precomp"] = stellar_phot_lnu_per_age
            # Phase 3c-3c: Taylor moment Ψ — same einsum, units erg/s/Hz × Å.
            ssp_phot_moment = self._state.ssp_phot_lut.ssp_phot_moment
            if ssp_phot_moment is not None:
                stellar_phot_moment_precomp = (
                    total_mass
                    * jnp.einsum("ma,maf->f", joint_weights, ssp_phot_moment)
                    * LSUN_ERG_PER_S
                )
                derived_overrides["stellar_phot_moment_precomp"] = stellar_phot_moment_precomp
                stellar_phot_moment_per_age = (
                    total_mass
                    * jnp.einsum("ma,maf->af", joint_weights, ssp_phot_moment)
                    * LSUN_ERG_PER_S
                )
                derived_overrides["stellar_phot_moment_per_age_precomp"] = (
                    stellar_phot_moment_per_age
                )
            # Phase 3c-3c-ii: publish filter pivot wavelengths so the dust LUT
            # (and future per-filter consumers like AGN and IGM) can use them.
            derived_overrides["filter_eff_waves"] = jnp.asarray(
                self._state.ssp_phot_lut.effective_wavelengths_rest
            )

        elif self._state is not None and self._state.ssp_phot_ztable is not None:
            # Free-z path (Phase 3c-1 + Phase 3c-3c-v) — smooth triweight
            # interp of the ztable at runtime z. Publishes the same derived
            # keys as the fixed-z path: stellar_phot_lnu_precomp,
            # stellar_phot_moment_precomp, stellar_phot_lnu_per_age_precomp,
            # stellar_phot_moment_per_age_precomp, filter_eff_waves.
            #
            # The original linear z-interp was O(h^2) and non-monotonic in
            # n_z at fixed test redshifts: doubling the grid can shift a
            # test point into a less-favourable cell and raise the error.
            # The triweight kernel (Hearin et al. 2023) is the canonical
            # smooth-grid interpolant used throughout tengri for SSP, CLOUDY,
            # and SKIRTOR grids — C²-continuous, kernel-supported on the
            # 3-bandwidth neighbourhood. See issue #438.
            from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

            ztable = self._state.ssp_phot_ztable
            z = jnp.asarray(params.get("redshift", 0.0))
            z_grid = ztable.z_grid
            z_edges = edges_for_grid(z_grid)
            # Match grid-cell width for the kernel bandwidth (Hearin 2023
            # convention): smooth across one neighbour on each side.
            z_scatter = 0.5 * (z_grid[1] - z_grid[0])
            w_z = compute_grid_weights(z, z_grid, scatter=z_scatter, edges=z_edges)

            def _interp(table):
                # table: (n_z, ...). Contract axis 0 with kernel weights.
                return jnp.tensordot(w_z, table, axes=([0], [0]))

            # ssp_phot_table: (n_z, n_met, n_age, n_filt); interp along axis 0.
            ssp_phot_at_z = _interp(ztable.ssp_phot_table)
            # Marginalised + age-resolved LUTs (Phase 3c-3c-iv-a parity).
            stellar_phot_lnu_precomp_rest = (
                total_mass * jnp.einsum("ma,maf->f", joint_weights, ssp_phot_at_z) * LSUN_ERG_PER_S
            )
            stellar_phot_lnu_per_age = (
                total_mass
                * jnp.einsum("ma,maf->af", joint_weights, ssp_phot_at_z)
                * LSUN_ERG_PER_S
            )
            derived_overrides["stellar_phot_lnu_precomp"] = stellar_phot_lnu_precomp_rest
            derived_overrides["stellar_phot_lnu_per_age_precomp"] = stellar_phot_lnu_per_age
            # Phase 3c-3c-v: Taylor moment Ψ at runtime z. Interpolate the
            # moment table the same way and publish marginalised + per-age.
            if ztable.ssp_phot_moment_table is not None:
                ssp_moment_at_z = _interp(ztable.ssp_phot_moment_table)
                stellar_phot_moment_precomp = (
                    total_mass
                    * jnp.einsum("ma,maf->f", joint_weights, ssp_moment_at_z)
                    * LSUN_ERG_PER_S
                )
                stellar_phot_moment_per_age = (
                    total_mass
                    * jnp.einsum("ma,maf->af", joint_weights, ssp_moment_at_z)
                    * LSUN_ERG_PER_S
                )
                derived_overrides["stellar_phot_moment_precomp"] = stellar_phot_moment_precomp
                derived_overrides["stellar_phot_moment_per_age_precomp"] = (
                    stellar_phot_moment_per_age
                )
            # Interpolate effective rest-frame wavelengths and publish for
            # downstream consumers (dust LUT, AGN, IGM).
            eff_waves_at_z = _interp(ztable.eff_waves_rest_table)
            derived_overrides["filter_eff_waves"] = eff_waves_at_z

        # ── 12c. Stellar spectrum LUT (Phase 5; SpectrumPrecomp) ────────
        # Pre-rebinned SSP × pixel LUT: the continuum at the spectrum pixel
        # centres in the galaxy rest frame. Publishes:
        #   - ``stellar_spec_lnu_precomp`` (n_pix,) — rest-frame Lν [erg/s/Hz]
        #   - ``spec_eff_waves`` (n_pix,) — rest-frame pixel wavelengths [Å]
        # The latter routes downstream SEDModelComponents (dust / AGN / IGM /
        # nebular continuum) through their spectrum-LUT branch, mirroring how
        # ``filter_eff_waves`` drives the photometry LUT path.
        if self._state is not None and self._state.ssp_spec_lut is not None:
            ssp_on_pixels = self._state.ssp_spec_lut.ssp_on_pixels  # (n_met, n_age, n_pix)
            stellar_spec_lnu = (
                total_mass * jnp.einsum("ma,map->p", joint_weights, ssp_on_pixels) * LSUN_ERG_PER_S
            )
            derived_overrides["stellar_spec_lnu_precomp"] = stellar_spec_lnu
            # Age-resolved per-pixel LUT for two-component (Charlot & Fall)
            # dust attenuation at the pixel grid (sum over age == marginalised).
            stellar_spec_lnu_per_age = (
                total_mass
                * jnp.einsum("ma,map->ap", joint_weights, ssp_on_pixels)
                * LSUN_ERG_PER_S
            )
            derived_overrides["stellar_spec_lnu_per_age_precomp"] = stellar_spec_lnu_per_age
            derived_overrides["spec_eff_waves"] = jnp.asarray(
                self._state.ssp_spec_lut.wave_rest_pixels
            )

        elif self._state is not None and self._state.ssp_spec_ztable is not None:
            # Free-z: interpolate the SSP cube to the rest-frame pixel grid
            # ``wave_obs / (1 + z)`` at runtime. Exact (no z-grid interpolation
            # of absorption features) and differentiable in z. ``wave`` is
            # ``ssp.ssp_wave`` and ``ssp_flux_for_csp`` is the (n_met, n_age,
            # n_wave) cube already used for the full-grid CSP einsum above.
            from jax import vmap

            z = jnp.asarray(params.get("redshift", 0.0))
            wave_obs_pix = jnp.asarray(self._state.ssp_spec_ztable.wave_obs_pixels)
            wave_rest = wave_obs_pix / (1.0 + z)
            n_met_s, n_age_s = ssp_flux_for_csp.shape[0], ssp_flux_for_csp.shape[1]
            flat = ssp_flux_for_csp.reshape(n_met_s * n_age_s, -1)
            interp_flat = vmap(lambda row: jnp.interp(wave_rest, wave, row, left=0.0, right=0.0))(
                flat
            )
            ssp_on_pixels_at_z = interp_flat.reshape(n_met_s, n_age_s, -1)
            stellar_spec_lnu = (
                total_mass
                * jnp.einsum("ma,map->p", joint_weights, ssp_on_pixels_at_z)
                * LSUN_ERG_PER_S
            )
            stellar_spec_lnu_per_age = (
                total_mass
                * jnp.einsum("ma,map->ap", joint_weights, ssp_on_pixels_at_z)
                * LSUN_ERG_PER_S
            )
            derived_overrides["stellar_spec_lnu_precomp"] = stellar_spec_lnu
            derived_overrides["stellar_spec_lnu_per_age_precomp"] = stellar_spec_lnu_per_age
            derived_overrides["spec_eff_waves"] = wave_rest

        # ── 12. Assemble new state ──────────────────────────────────────
        return state.with_(
            sed_intrinsic=sed_intrinsic,
            derived=state.derived.with_(**derived_overrides),
        )


def _time_weighted_sfr(
    sfr_history: jnp.ndarray,
    sfh_lbt_grid: jnp.ndarray,
    window_yr: float,
) -> jnp.ndarray:
    """Time-weighted SFR over the last ``window_yr`` years.

    Thin wrapper around the canonical helper in
    :mod:`tengri.components.stellar.sfh.sfr_window`. Kept for
    StellarSEDComponent's existing call sites; new code should import
    :func:`time_weighted_sfr` from there directly.
    """
    from tengri.components.stellar.sfh.sfr_window import time_weighted_sfr

    return time_weighted_sfr(sfr_history, sfh_lbt_grid, window_yr)


# ─────────────────────────────────────────────────────────────────────
# JAX pytree registration
# ─────────────────────────────────────────────────────────────────────
#
# Register StellarSEDComponent as a JAX pytree so ``self.ssp_data``
# flows through ``jax.jit`` as a TRACED input rather than being baked
# into the XLA graph as a literal constant. The SSP grid is ~8 MB
# (15 × 93 × 5994 doubles); without this registration the cold-compile
# time explodes to ~900 ms because XLA inlines the entire grid as constants
# at every call site. With registration cold-compile drops by an
# order of magnitude.
#
# ``ssp_data`` is the only data field (it's a JAX-pytree-compatible
# NamedTuple with ndarray leaves). Everything else is structural
# (config, name, parameter_prefix) → meta.

from jax import tree_util as _tree_util

_tree_util.register_dataclass(
    StellarSEDComponent,
    data_fields=("ssp_data",),
    meta_fields=("config", "name", "parameter_prefix", "_state"),
)

del _tree_util
