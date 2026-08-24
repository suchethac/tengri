# SPDX-License-Identifier: BSD-3-Clause
"""DustAttenuationSEDComponent: screen-style dust attenuation as a SEDComponent.

The first SED component in the pipeline that **transforms** the SED rather than
adding to it: reads ``sed_intrinsic``, writes ``sed_attenuated``.

Scope (intentionally small)
---------------------------
Wraps a *single-component screen* attenuation law: picked at construction
time from the catalog in :mod:`tengri.components.dust.attenuation`
(default: Calzetti+2000). For two-component (birth-cloud + diffuse ISM)
attenuation that needs the per-age stellar luminosity grid, see the
sibling :class:`DustSEDComponent` in
:mod:`tengri.components.dust.two_component`.

Cross-component reads
---------------------

- ``sed_intrinsic`` (erg/s/Hz): produced by upstream emitters (stellar +
  AGN + radio + X-ray etc.). If ``None`` this adapter is a no-op.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.dust.attenuation import calzetti, resolve_dust_law
from tengri.components.template_threading import TemplateThreading
from tengri.parameters.priors import Fixed
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = ["DustAttenuationSEDComponent", "DustAttenuationSEDComponentConfig"]


@dataclass(frozen=True, kw_only=True)
class DustAttenuationSEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`DustAttenuationSEDComponent`.

    Attributes
    ----------
    law : str
        Attenuation law name resolved by
        :func:`tengri.components.dust.attenuation.resolve_dust_law`.
        Examples: ``"calzetti"``, ``"cardelli"``, ``"smc"``, ``"lmc"``,
        ``"prevot_smc"``, ``"li08"``, etc. Default ``"calzetti"``. This
        default is a low-level construction convenience only (component
        tests that build ``DustAttenuationSEDComponent()`` directly to
        exercise pipeline mechanics, not curve choice); the public grammar
        (``SEDModel.build`` / ``Parameters``) always resolves and passes
        ``law`` explicitly before reaching this dataclass, so it never
        observes this default.
    name : str
        Diagnostic identifier. Default ``"dust_attenuation"``.

    Notes
    -----
    The law is resolved eagerly into a callable ``k(λ)`` at construction
    time so :meth:`apply` does not perform a registry lookup inside the
    JIT scope.
    """

    law: str = "calzetti"
    name: str = "dust_attenuation"
    live_shape_params: frozenset[str] = frozenset()
    r"""Shape parameters somebody actually asked for, resolved at build time.

    The screen used to call its law with no arguments at all, so every law
    shape parameter was unreachable and unfittable (#1808). Passing the spec's
    values unconditionally is not the fix: the spec declares ONE shared
    ``dust_delta`` / ``dust_bump_strength``, both ``Fixed(0.0)``, while each law
    carries its paper's value in its own signature: ``kriek_conroy``
    ``dust_bump_strength = 1.0``, ``narayanan_z`` ``dust_delta = -0.2``.
    Overriding those collapses three distinct published laws onto one curve
    (measured).

    So only parameters with a provenance of ``user_fixed`` / ``user_prior`` /
    ``user_free`` / ``wildcard_free`` land here; ``registry_default`` and
    ``wildcard_fixed`` do not, and for those the law's own default stands.
    ``SEDModel._build_component_chain`` computes the set: it has the spec, and
    deciding once at build time keeps this a static Python branch rather than a
    comparison against a traced value inside ``apply()``.
    """


@dataclass(frozen=True)
class DustAttenuationSEDComponentState(SEDComponentState):
    r"""Cached attenuation curve evaluated on the pipeline wave grid.

    Attributes
    ----------
    k_lambda: jnp.ndarray, shape (n_wave,) | None
        Pre-evaluated normalized attenuation curve k(λ) (with k(5500 Å) = 1).
        ``None`` until :meth:`DustAttenuationSEDComponent.precompute` runs.
    """

    name: str = "dust_attenuation"
    k_lambda: jnp.ndarray | None = None


@dataclass(frozen=True)
class DustAttenuationSEDComponent(TemplateThreading):
    r"""SEDComponent adapter for a single-component screen attenuation law.

    Notes
    -----
    **JIT-compatible**: yes, :meth:`apply` is pure JAX once the
    attenuation curve is precomputed.
    **Transforms**: writes ``sed_attenuated = sed_intrinsic * exp(-tau_v * k(λ))``.
    Reads ``state.sed_intrinsic`` and is a no-op if it is ``None``.

    The single-component model is intentional; see module docstring.
    Two-component (Charlot & Fall 2000) attenuation will be a separate
    adapter once the stellar component publishes per-age luminosities.
    """

    config: DustAttenuationSEDComponentConfig = field(
        default_factory=DustAttenuationSEDComponentConfig
    )
    name: str = "dust_attenuation"
    parameter_prefix: str = "dust_"
    _state: DustAttenuationSEDComponentState | None = None

    def citations(self) -> tuple[str, ...]:
        """Attenuation-law citations (Calzetti, Cardelli, SMC, …) are
        config-driven via
        :data:`tengri.citations.associations.DUST_LAW_CITATIONS`."""
        return ()

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Returns ``dust_tau_v`` only: the law shape is fixed by
        :attr:`config.law` and is not a free parameter (changing the
        attenuation law mid-fit would require re-precomputing).
        """
        return [
            ParamDeclaration(
                "dust_tau_v",
                Fixed(0.3),
                "V-band optical depth at 5500 Å [dimensionless]",
            ),
        ]

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this dust component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.
        """
        return (
            DerivedKey("L_ir", "erg/s", "Integrated dust-absorbed luminosity"),
            DerivedKey("L_absorbed", "erg/s", "Alias for L_ir (energy balance)"),
            DerivedKey("log_L_ir", "dex", "log10(L_ir / (erg/s)); float32-safe form"),
            DerivedKey(
                "dust_attenuation_factor",
                "",
                "exp(-tau_v * k(lambda)) on pipeline wave grid",
            ),
            DerivedKey("sed_dust_attenuated", "erg/s/Hz", "Attenuated stellar SED"),
            DerivedKey(
                "log_line_lums_attenuated",
                "dex",
                "log10 of the discrete line catalog after this screen; absent when no "
                "photoionized backend published one (#1867)",
            ),
        )

    def optional_inputs(self) -> tuple[DerivedKey, ...]:
        """Nebular continuum read opportunistically, for ordering only.

        Declaring ``sed_nebular`` an optional input makes the pipeline
        topological sort (ADR-0006) place the nebular component *before*
        this single screen. The nebular component folds its continuum into
        ``sed_intrinsic``; running first means the screen reddens the
        nebular continuum together with the stellar light: the single law
        attenuates HII-region emission by the same curve as the stars,
        matching bagpipes/FSPS/CIGALE. Without this declaration the stable
        sort kept dust *before* nebular, leaving the continuum unattenuated
        (the single-screen analog of the two-component bug fixed in #668).

        BakedIn backends publish ``sed_nebular`` as zeros (emission is
        already in the SSP grid), so this is a no-op there. The screen does
        not read that key directly: it acts on the already-summed
        ``sed_intrinsic``: so it is purely an ordering edge.

        ``line_waves`` / ``line_lums`` ARE read directly: :meth:`apply`
        reddens the discrete catalog and publishes ``line_lums_attenuated``
        (#1867). The ``sed_nebular`` edge already sequences nebular first, so
        declaring them adds no new constraint: they are declared because
        ADR-0009 says a component states what it reads. An undeclared read
        works until someone reorders the pipeline, and then fails silently.
        """
        return (
            DerivedKey(
                "sed_nebular",
                "erg/s/Hz",
                "Nebular continuum folded into sed_intrinsic before the screen",
            ),
            DerivedKey(
                "line_waves",
                "Angstrom",
                "Discrete nebular line wavelengths (Cue/CloudyGrid); absent for BakedIn",
            ),
            DerivedKey(
                "log_line_lums",
                "dex",
                "INTRINSIC log10 line luminosities to redden (#1867); absent for BakedIn",
            ),
        )

    def _curve(self, params: Mapping[str, jnp.ndarray]):
        r"""``k(lambda)`` for the selected law, with requested shape parameters.

        Returns ONE bound callable for the whole of :meth:`apply`. The screen
        evaluates its curve at six places: the SED, the filter LUT, that LUT's
        finite-difference slope, the sub-band quadrature, the rest band and the
        spectroscopy pixels: and every one must use the same curve with the
        same parameters. Six independent call sites are six chances for one to
        keep the old behavior and put two different screens in one model.
        """
        from tengri.components.dust._apply import _TWO_COMPONENT_LAW_PARAMS

        law_fn = calzetti if self.config.law == "calzetti" else resolve_dust_law(self.config.law)
        live = self.config.live_shape_params
        if not live:
            return law_fn

        tabled = {flat for _, flat, _ in _TWO_COMPONENT_LAW_PARAMS}
        kwargs: dict[str, jnp.ndarray] = {
            law_kw: jnp.asarray(params[flat])
            for law_kw, flat, _ in _TWO_COMPONENT_LAW_PARAMS
            if flat in live and flat in params
        }
        # Shape parameters a law names directly rather than through the shared
        # table (``dust_bump_x0`` and friends).
        for flat in live - tabled:
            if flat in params:
                kwargs[flat] = jnp.asarray(params[flat])
        if not kwargs:
            return law_fn

        def _bound(wave: jnp.ndarray) -> jnp.ndarray:
            return law_fn(wave, **kwargs)

        return _bound

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> DustAttenuationSEDComponentState:
        r"""Evaluate the attenuation curve k(λ) on the pipeline wave grid.

        Parameters
        ----------
        ssp_data: object | None
            Unused (this adapter does not depend on SSP data). Kept in
            the signature to match the :class:`SEDComponent` Protocol.
        wave_grid: jnp.ndarray, shape (n_wave,) | None
            Rest-frame wavelength grid in Å. Required.

        Returns
        -------
        DustAttenuationSEDComponentState
            Holds the cached k(λ) tensor.
        """
        del ssp_data, filters
        if wave_grid is None:
            # Permissive path: contract tests call precompute() with no
            # args. Return an unprimed state; apply() will compute
            # k(λ) lazily on first call.
            return DustAttenuationSEDComponentState(name=self.name, k_lambda=None)
        if self.config.live_shape_params:
            # A curve cached here is frozen before any parameter value exists,
            # and apply() prefers the cached array: which is what made the
            # requested shape parameters unreachable (#1808). When somebody has
            # asked for one, leave the state unprimed and let apply() build the
            # curve from params. Laws with nothing requested (calzetti and the
            # parameter-free curves: the default and common case) keep the
            # cache and the fast path unchanged.
            return DustAttenuationSEDComponentState(name=self.name, k_lambda=None)
        if self.config.law == "calzetti":
            # Avoid a registry lookup on the most common path.
            k = calzetti(wave_grid)
        else:
            law_fn = resolve_dust_law(self.config.law)
            k = law_fn(wave_grid)
        return DustAttenuationSEDComponentState(name=self.name, k_lambda=k)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        r"""Apply screen attenuation to ``state.sed_intrinsic``.

        Parameters
        ----------
        state: ForwardState
            Must carry rest-frame ``wave``. If ``sed_intrinsic`` is
            ``None`` this method is a no-op (returns the input
            unchanged).
        params: mapping
            Receives ``dust_*`` keys plus ``redshift`` (unused here).

        Returns
        -------
        ForwardState
            New state with ``sed_attenuated`` populated.

        Notes
        -----
        Computes :math:`k(\lambda)` lazily inside :meth:`apply` if
        :meth:`precompute` was not called: this keeps the adapter
        usable in tests that drive components directly without an
        orchestrator that schedules precompute.
        """
        if state.sed_intrinsic is None:
            return state

        # One curve for the whole method; see :meth:`_curve`.
        curve = self._curve(params)
        if self._state is not None and self._state.k_lambda is not None:
            k = self._state.k_lambda
        else:
            k = curve(state.wave)

        tau_v = jnp.asarray(params["dust_tau_v"])
        attenuation = jnp.exp(-tau_v * k)
        attenuated = state.sed_intrinsic * attenuation

        # Energy balance: integrate the absorbed luminosity in frequency
        # space and publish for downstream consumers (dust emission components
        # re-emit it; RadioSEDComponent uses it to set the SF radio
        # amplitude via the FIR-radio correlation). LyC photons ionize H
        # rather than heat dust, so the canonical integral masks λ < 912 Å
        # (#922).
        from tengri.forward.energy_balance import bolometric_absorbed_log10, warn_if_corrupt
        from tengri.utils.physics_constants import C_AA
        from tengri.utils.scale import pow10

        nu = C_AA / state.wave  # Hz
        # Absorbed luminosities are ~1e43 erg/s: outside float32: so the
        # integral is done in log space and the linear form derived from it
        # (#1206). The sign only tracks grid orientation; the energy is |L|.
        log_l_ir, _ = bolometric_absorbed_log10(
            state.sed_intrinsic, attenuated, nu, wave=state.wave
        )
        warn_if_corrupt(log_l_ir, component=type(self).__name__)
        l_ir = pow10(log_l_ir)  # erg/s

        # Filter-level A(λ_eff) and A'(λ_eff) LUTs.
        # Published only when an upstream component (stellar) has put
        # ``filter_eff_waves`` into ``state.derived``: i.e. only when
        # ``approx=WavePrecomp()`` is set on SEDModel.
        derived_overrides = dict(
            dust_attenuation_factor=attenuation,
            L_ir=l_ir,
            L_absorbed=l_ir,
            log_L_ir=log_l_ir,
            sed_dust_attenuated=attenuated,
        )

        # Discrete emission-line catalog, reddened with this component's single
        # screen (#1867). The two-component component does the same in its §2c;
        # omitting it here would leave every single-screen model reading
        # INTRINSIC line luminosities from `pred.lines.*` and
        # `predict_properties`, which is the half-fix #1867 warns about.
        #
        # `curve(line_wave)`, never the cached `k_lambda`: that array is bound
        # to `state.wave` and means nothing at line wavelengths.
        #
        # Reads and publishes the LOG companion, never the linear `line_lums`,
        # which is `inf` in float32 at ~1e41 erg/s (#1534/#1837). `-tau*k` is a
        # log-domain quantity already; converting to dex is one division.
        _line_waves = state.derived.get("line_waves")
        _log_line_lums = state.derived.get("log_line_lums")
        if _line_waves is not None and _log_line_lums is not None:
            line_wave = jnp.asarray(_line_waves)
            log10_e = 1.0 / jnp.log(10.0)
            derived_overrides["log_line_lums_attenuated"] = (
                jnp.asarray(_log_line_lums) - tau_v * curve(line_wave) * log10_e
            )

        filter_eff = state.derived.get("filter_eff_waves")
        if filter_eff is not None:
            # Evaluate the attenuation law at the filter pivots and at a
            # finite-difference offset to compute the slope analytically.
            k_at = curve(filter_eff)
            a_lut = jnp.exp(-tau_v * k_at)
            # k'(λ) via central finite difference. δλ = 1 Å is small
            # compared with filter widths (~100–10000 Å) and gives an
            # accurate slope for smooth analytic dust laws.
            d_lambda = jnp.asarray(1.0)
            k_plus = curve(filter_eff + d_lambda)
            k_minus = curve(filter_eff - d_lambda)
            k_slope = (k_plus - k_minus) / (2.0 * d_lambda)
            # A'(λ) = d/dλ exp(-τ·k(λ)) = -τ · k'(λ) · A(λ).
            a_slope_lut = -tau_v * k_slope * a_lut
            derived_overrides["dust_attenuation_precomp"] = a_lut
            derived_overrides["dust_attenuation_slope_precomp"] = a_slope_lut

            # Sub-band quadrature (#1122). Same treatment the two-component screen
            # gets: EVALUATE the law at each sub-band's quadrature node instead of
            # extrapolating it from λ_eff. Wiring this here is not optional: the
            # quadrature supersedes ``taylor_correction``, so a single-component
            # model whose sub-bands were never published would fall back to the bare
            # ``A(λ_eff)·Φ`` form and be *worse* than before.
            sub_waves = state.derived.get("stellar_subband_waves_rest_precomp")
            if sub_waves is not None:
                k_sub = curve(sub_waves)
                derived_overrides["dust_attenuation_subband_precomp"] = jnp.exp(-tau_v * k_sub)

            # The same screen on the REST band (#1148). ``phot_rest_fnu`` projects at
            # z=0, so its filter samples rest λ_pivot, not rest λ_pivot/(1+z): a
            # different set of wavelengths, and the galaxy's own dust belongs THERE.
            _law = curve
            rb_eff = state.derived.get("filter_restband_eff_waves")
            if rb_eff is not None:
                derived_overrides["dust_restband_attenuation_precomp"] = jnp.exp(
                    -tau_v * _law(rb_eff)
                )
            rb_sub_waves = state.derived.get("stellar_restband_subband_waves_precomp")
            if rb_sub_waves is not None:
                derived_overrides["dust_restband_attenuation_subband_precomp"] = jnp.exp(
                    -tau_v * _law(rb_sub_waves)
                )

        # SpectrumPrecomp: per-pixel transmission. A spectrum pixel
        # is a single wavelength, so T(λ_pix) = exp(-τ·k(λ_pix)) is exact:
        # no Taylor slope needed (contrast the filter branch above).
        spec_eff = state.derived.get("spec_eff_waves")
        if spec_eff is not None:
            k_pix = curve(spec_eff)
            derived_overrides["dust_spec_transmission_precomp"] = jnp.exp(-tau_v * k_pix)

        return state.with_(
            sed_intrinsic=attenuated,
            sed_attenuated=attenuated,
            derived=state.derived.with_(**derived_overrides),
        )


# Register in the unified component dispatch table so the grammar type
# ``dust_attenuation={'type': 'single_component'}`` resolves via _resolve_registry_component
# (the single dispatch seam), not a hardcoded class in build_components (#844).
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["single_component"] = DustAttenuationSEDComponent
