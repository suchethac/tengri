# SPDX-License-Identifier: BSD-3-Clause
"""Composable shock nebular emission: the canonical shock ``SEDModelComponent``.

MAPPINGS V (3MdBs) fast-radiative-shock + precursor emission, added as a
*separate additive component* that composes with any photoionized nebular
backend (Cue / CloudyGrid / CB19 / baked-in). Shock and photoionized emission
are physically distinct regimes, so they are summed independently and the
shock contribution is published under its own ``sed_shock`` diagnostic key.

Activated by the top-level ``shock={...}`` grammar group (or the low-level
``Parameters(shock=True)`` escape hatch). The forward pipeline places this
component alongside the photoionized nebular component; its emission is
accumulated into ``sed_intrinsic`` before dust runs, so it is reddened by the
dust screen exactly like the rest of the intrinsic SED.

Normalization (two knobs, selected by the static ``norm`` config)
-----------------------------------------------------------------

* ``norm="frac"`` (**default**): *relative*: the shock Hα luminosity is a
  fraction ``shock_frac`` of the galaxy's approximate Hα
  (``L(Hα) ~ 1e-3 L_bol`` of the SED accumulated so far). Intuitive "how
  much of the line budget is shock-driven"; reproduces the legacy
  :func:`tengri.forward.emission_helpers.shock_emission` bit-for-bit.
* ``norm="lhalpha"``: *absolute*: the shock Hα luminosity is set directly by
  ``shock_log_lhalpha`` (``log10(L_Hα / [erg/s])``), decoupled from the star
  formation rate. Preferred for AGN narrow-line-region / outflow / SN-remnant
  shocks that are unrelated to the young-stellar Hα budget.

The kinematic/structural parameters (``shock_velocity``,
``shock_log_density``, ``shock_b_over_sqrt_n``) and the categorical knobs
(``abundance``, ``component``) are shared by both normalizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from tengri.components.nebular.shock import compute_shock_sed
from tengri.components.sed_model_component import SEDModelComponent
from tengri.protocols.component import SEDComponentConfig
from tengri.utils.physics_constants import C_AA
from tengri.utils.scale import apply_log10_scale, pow10 as _pow10

#: Reference Hα luminosity the float32 path carries the shock shape at, as log10
#: of erg/s (#1206). The shock SED is exactly linear in L_Hα, so ANY split of the
#: scale is exact: but the split point decides whether float32 survives. Carrying
#: at unit Hα pushes the whole ~41 dex onto ``apply_log10_scale``, whose Jacobian
#: is ``10**offset`` and therefore ``inf`` past 38.5 dex: the forward stayed finite
#: while the gradient became NaN. At 1e30 the carried array (~1e18) and the
#: derivative (~1e30) both sit well inside float32.
_SHOCK_LHA_LOG_REF: float = 30.0

__all__ = ["ShockNebular", "ShockNebularConfig"]


@dataclass(frozen=True)
class ShockNebularConfig(SEDComponentConfig):
    """Static (non-traced) configuration for :class:`ShockNebular`.

    Attributes
    ----------
    name: str
        Diagnostic identifier. Default ``"shock"``.
    norm: str
        Normalization mode: ``"frac"`` (relative to the galaxy's approximate
        Hα) or ``"lhalpha"`` (absolute ``shock_log_lhalpha``). Default
        ``"frac"``.
    abundance: str
        MAPPINGS abundance set: ``"solar"`` | ``"2xsolar"`` | ``"lmc"`` |
        ``"smc"`` | ``"dopita2005"``. Default ``"solar"``.
    component: str
        Emission component: ``"shock"`` (post-shock only) | ``"precursor"``
        (pre-shock photoionization) | ``"combined"`` (sum). Default
        ``"combined"``.
    """

    name: str = "shock"
    norm: str = "frac"
    abundance: str = "solar"
    component: str = "combined"


class ShockNebular(SEDModelComponent):
    r"""MAPPINGS V shock-driven nebular emission (composable, additive).

    Reads its free parameters from the ``shock_*`` bucket
    (``tengri.components.nebular._params.SHOCK_PARAMS``, registered by
    ``Parameters(shock=True)`` / the ``shock`` grammar group) rather than
    auto-declaring them, so it composes with: and never double-declares
    against: the photoionized nebular backend.

    Cross-component contract
    ------------------------
    Reads: nothing required. The ``norm="frac"`` path uses ``sed_in`` (the SED
    accumulated by upstream components) only to set the *normalization* of the
    shock template.
    Publishes: ``sed_shock`` (erg/s/Hz, the shock contribution) and
    ``L_shock`` (erg/s, its frequency integral).

    Notes
    -----
    **JIT-compatible**: yes. ``predict`` is pure JAX; the categorical knobs
    (``norm``, ``abundance``, ``component``) are static config, not traced.

    **Composability**: this component is *additive*. The photoionized nebular
    backend publishes ``sed_nebular``; this component publishes ``sed_shock``;
    the two are summed in ``sed_intrinsic`` and both are reddened by the dust
    screen. A model may run shock alone (``neb={'type':'none'}`` + ``shock``),
    photoionization alone, or both together.

    **Dust attenuation** (#1434): shock emission is attenuated by the two-component
    dust screen (young-limit: ``tau_bc·k_bc + tau_diff·k_diff``) the same way as
    nebular continuum. This is defensible for star-formation-linked shocks (``norm='frac'``
    mode), which originate in young stellar populations behind the birth cloud. For
    AGN outflow or NLR shocks (``norm='lhalpha'`` mode), the screen geometry is an
    approximation: MAPPINGS traces often originate in partially or fully unobscured
    AGN-driven outflows, so the birth-cloud dust in front of them is not the physical
    situation. Measure the effect with synthetic fixtures before using shock in
    conjunction with high-optical-depth dust models. Single-component dust models
    (single-screen attenuation) are unaffected: they screen the total ``sed_intrinsic``
    and have no separate shocked-gas treatment.

    References
    ----------
    .. [1] M. A. Allen et al., "The MAPPINGS III Library of Fast Radiative
       Shock Models," ApJS, 178, 20 (2008). https://doi.org/10.1086/589652
    .. [2] C. Alarie & C. Morisset, "Extensive Online Shock Model Database,"
       Rev. Mex. Astron. Astrofis., 55, 377 (2019).
       https://doi.org/10.22201/ia.01851101p.2019.55.02.21
    """

    config: ShockNebularConfig = ShockNebularConfig()
    name: str = "shock"
    parameter_prefix: str = "shock_"
    # Justified by design fallback: load() returns None when data/mappings_templates.h5
    # is absent, but predict() falls back to hardcoded Allen+2008 subset (see load()
    # docstring and shock.compute_shock_sed). Missing template data is not an error.
    requires_template_data: ClassVar[bool] = False

    # Params are supplied by the ``shock_*`` bucket, not auto-declared here.
    inputs: ClassVar[dict[str, str]] = {}
    outputs: ClassVar[dict[str, str]] = {"sed_shock": "erg/s/Hz"}

    #: The MAPPINGS V ratio cubes thread as a JIT argument rather than baking
    #: into the graph. Without this the selected cube was an XLA ``Constant`` on
    #: every compile: 3.73 MB against a 0.05 MB bare-stellar floor (#1694).
    accepts_threaded_templates: ClassVar[bool] = True

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Load the MAPPINGS V ratio cubes at build time.

        Parameters
        ----------
        wave: ndarray, optional
            Unused: the shock grid is wavelength-independent (it carries its
            own line wavelengths). Present for the :meth:`load` contract.

        Returns
        -------
        ShockTemplateGrid or None
            ``None`` when ``data/mappings_templates.h5`` is absent, in which
            case :meth:`predict` falls back to the hardcoded Allen+2008 subset
            exactly as before.

        Notes
        -----
        **JIT-compatible**: no, build-time only, which is the point.
        """
        from tengri.components.nebular.shock import load_shock_template_grid

        return load_shock_template_grid()

    def predict(
        self,
        p: dict[str, Any],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        templates: Any | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        r"""Add MAPPINGS V shock emission to the running SED.

        Parameters
        ----------
        p: dict[str, ndarray]
            Shock parameters, ``shock_`` prefix stripped: ``frac``,
            ``log_lhalpha``, ``velocity`` [km/s], ``log_density`` [dex cm^-3],
            ``b_over_sqrt_n`` [μG].
        sed_in: ndarray, shape (n_wave,)
            SED accumulated so far [erg/s/Hz]. Used for the ``norm="frac"``
            Hα normalization only.
        wave: ndarray, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].
        templates: ShockTemplateGrid, optional
            MAPPINGS V ratio cubes, supplied by :meth:`apply` from the threaded
            ``template_data`` so they arrive as a JIT argument rather than a
            baked constant (#1694). ``None`` falls back to the module-level
            cache: correct, but 3.73 MB per compile.

        Returns
        -------
        sed_out: ndarray, shape (n_wave,)
            ``sed_in`` plus the shock contribution [erg/s/Hz].
        published: dict
            ``{"sed_shock": ndarray}``: the shock contribution [erg/s/Hz].
            Its total luminosity is recoverable as :math:`-\int S_\nu\,d\nu`.

        Notes
        -----
        .. math::

            L_{\mathrm{H}\alpha}^{\mathrm{shock}} =
            \begin{cases}
              f_{\rm shock}\,\max(10^{-3} L_{\rm bol},\ \epsilon)
                & \text{norm=frac}\\
              10^{\,\log L_{\mathrm{H}\alpha}} & \text{norm=lhalpha}
            \end{cases}

        with :math:`L_{\rm bol} = -\int S_\nu\,d\nu` over ``sed_in`` [erg/s]
        and :math:`\epsilon = 10^{-30}` guarding the log. The shock template is
        then scaled to this Hα anchor by
        :func:`tengri.components.nebular.shock.compute_shock_sed`. The ``frac``
        branch reproduces
        :func:`tengri.forward.emission_helpers.shock_emission` exactly.
        """
        nu = C_AA / wave

        # Float32 (#1206): the shock Hα luminosity is ~1e41 erg/s: ``inf`` in
        # float32 (max 3.4e38): so ``l_shock_halpha * <line profile>`` becomes
        # ``inf * 0 = nan`` even though the resulting SED (~1e26 erg/s/Hz) is
        # perfectly representable. The shock SED is *exactly* linear in
        # ``l_shock_halpha`` (verified: ×10 per dex), so on the float32 path we
        # carry the shape at a REFERENCE Hα and re-apply only the residual as a
        # log10 offset. Float64 keeps the linear expressions, bit-identical.
        #
        # The reference matters. Evaluating at unit Hα forces the whole ~41 dex
        # onto ``apply_log10_scale``, whose Jacobian is exactly ``10**offset``:
        # 1e41, past the float32 maximum (3.4e38), so the FORWARD stayed finite
        # while the reverse pass produced NaN and broke float32 NUTS. Splitting
        # the scale (reference 1e30, residual ~11 dex) keeps the carried array
        # (~1e18) AND the derivative (~1e11) comfortably in range. The split is
        # exact: the shock SED is linear in L_Hα (verified x10/dex).
        # Gate on ``sed_in``: the SED actually being accumulated. This is the
        # quantity whose range is at stake, so it is the honest thing to key on.
        #
        # The justification here used to be that ``wave`` "arrives as float64 even
        # when the pipeline runs in float32", and that ``sed_in`` is float32 in the
        # mixed-precision ``forward_dtype="float32"`` mode as well. Both were
        # measured false (#1411, #1433): ``wave`` and ``sed_in`` carry the *same*
        # dtype at every gate-bearing component, in all four configurations
        # (exact/WavePrecomp x float64/pure float32), and ``forward_dtype`` casts
        # nothing so mixed mode leaves both in float64. Gating on ``sed_in`` and
        # gating on ``wave`` are therefore equivalent today: this one is kept
        # because it names the array whose magnitude the rescale exists to protect.
        _f32 = jnp.asarray(sed_in).dtype == jnp.float32

        if self.config.norm == "lhalpha":
            _log_l_shock_halpha = jnp.asarray(p["log_lhalpha"])
            l_shock_halpha = (
                _pow10(_log_l_shock_halpha - _SHOCK_LHA_LOG_REF)
                if _f32
                else jnp.power(10.0, p["log_lhalpha"])
            )
        else:
            # Relative: fraction of the galaxy's approximate Hα. Same
            # order-of-magnitude proxy (L(Hα) ~ 1e-3 L_bol) as the legacy
            # ``shock_emission`` helper, so the two paths agree bit-for-bit.
            if _f32:
                # ``l_bol`` ~1e44 erg/s overflows too: peak-factor the integrand
                # so only the O(1) residual is integrated, and carry the peak in
                # log10 along with the 1e-3 proxy and the user's fraction.
                # stop_gradient: factorization constant, added back as log10(_peak) (#1436).
                _peak = jax.lax.stop_gradient(jnp.max(jnp.abs(sed_in)))
                _peak = jnp.where(_peak > 0.0, _peak, 1.0)
                _hat_l_bol = -jnp.trapezoid(sed_in / _peak, nu)
                _log_l_shock_halpha = (
                    jnp.log10(_peak)
                    + jnp.log10(jnp.maximum(_hat_l_bol, 1e-30))
                    - 3.0
                    + jnp.log10(jnp.maximum(p["frac"], 1e-30))
                )
                l_shock_halpha = _pow10(_log_l_shock_halpha - _SHOCK_LHA_LOG_REF)
            else:
                l_bol = -jnp.trapezoid(sed_in, nu)
                l_halpha_approx = jnp.maximum(l_bol * 1e-3, 1e-30)
                l_shock_halpha = p["frac"] * l_halpha_approx
                _log_l_shock_halpha = None

        shock_sed = compute_shock_sed(
            wave,
            p["velocity"],
            l_shock_halpha,
            shock_log_density=p["log_density"],
            shock_b_over_sqrt_n=p["b_over_sqrt_n"],
            shock_abundance=self.config.abundance,
            shock_component=self.config.component,
            templates=templates,
        )
        if _f32:
            shock_sed = apply_log10_scale(shock_sed, _SHOCK_LHA_LOG_REF)
        return sed_in + shock_sed, {"sed_shock": shock_sed}

    def apply(
        self,
        state: Any,
        params: Any,
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> Any:
        """Add shock emission, publishing a filter-integrated LUT under WavePrecomp.

        Delegates to the inherited orchestration on the exact path, so
        ``approx=None`` behavior is unchanged by construction. Under
        ``approx=WavePrecomp()`` it takes over, because the inherited LUT is
        wrong for this component in two independent ways (#1375):

        1. **The normalization collapses.** The inherited
           :meth:`~tengri.components.sed_model_component.SEDModelComponent.predict_precomp`
           calls :meth:`predict` with a dummy SED of *zeros*. Under
           ``norm="frac"`` the shock Hα anchor is
           ``frac * max(1e-3 * L_bol, 1e-30)`` with ``L_bol`` the frequency
           integral of that SED, so ``L_bol = 0``, the ``1e-30`` epsilon guard
           fires, and the LUT lands near 1e-44 erg/s/Hz against a true ~1e29.
           It sums into the photometry as an exact no-op.
        2. **The sampling misses the lines.** Shock emission is line-dominated.
           Evaluating it at a handful of filter *effective wavelengths* samples
           the continuum between lines instead of integrating the lines through
           the filter response: wrong under ``norm="lhalpha"`` too, where the
           absolute normalization makes defect 1 inapplicable.

        Both are fixed the same way the photoionized backends handle
        ``nebular_phot_lnu_precomp``: evaluate the shock SED on the full
        wavelength grid (correct normalization *and* correct line sampling),
        then integrate it through the filter curves with
        :func:`~tengri.observation.photometry.lnu_filter_integral_batch`.

        Publishes ``shock_phot_lnu_precomp`` (observed band) and
        ``shock_restband_lnu_precomp`` (the same integral at ``z=0``; the rest
        band sits at its own pivot and reusing the observed value is what made
        the nebular LUT read 769 % high in ``des_g`` at z=0.5, #1148). Both are
        intrinsic rest-frame Lν: no dust, no cosmology:
        matching the precompute contract; ``predict_via_precomp`` applies the
        young-limit dust screen and the ``(1+z)/(4 pi d_L^2)`` dimming.

        Parameters
        ----------
        state: ForwardState
            Current state; ``state.wave`` and ``state.sed_intrinsic`` supply the
            full-grid context the LUT needs.
        params: mapping
            Full parameter dict (sliced by prefix here).
        ssp_data, template_data: object, optional
            Unused by this component; accepted for protocol conformance.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` advanced and the shock keys published.

        Raises
        ------
        KeyError
            If ``redshift`` is absent under WavePrecomp. The observed-band
            integral needs it, and defaulting to 0.0 would silently place the
            filters in the wrong frame.

        Notes
        -----
        **JIT-compatible**: yes. The branch is on grid *presence* (static build
        config), not on any traced value.
        """
        if state.derived.get("filter_eff_waves") is None:
            # Exact path (and spectrum-only LUT models): the inherited
            # orchestration is already correct, so do not duplicate it.
            return super().apply(state, params, ssp_data=ssp_data, template_data=template_data)

        p = self.slice_params(params)
        sed_in = (
            state.sed_intrinsic if state.sed_intrinsic is not None else jnp.zeros_like(state.wave)
        )
        # Full grid: identical call to the exact path, so the LUT is built from
        # the same shock SED the exact path adds to sed_intrinsic.
        #
        # ``templates=`` matters more here than on the exact path, not less:
        # ``approx="auto"`` resolves to WavePrecomp for every photometry fit, so
        # this branch is what an inference run actually compiles. Threading only
        # in ``super().apply`` would have fixed the forward path and left the
        # 3.73 MB constant in place everywhere it is paid most (#1694).
        sed_out, published = self.predict(
            p, sed_in, state.wave, templates=self.threaded_templates(template_data)
        )

        fw = state.derived.get("phot_filter_waves_padded")
        ft = state.derived.get("phot_filter_trans_padded")
        if fw is not None and ft is not None:
            if "redshift" not in p:
                raise KeyError(
                    "ShockNebular needs 'redshift' to filter-integrate its "
                    "WavePrecomp LUT into the observed band, but it is absent "
                    "from the parameter dict. Defaulting to 0.0 would put the "
                    "filters in the rest frame and silently mis-weight the "
                    "shock lines."
                )
            # Not routed through ``_band_projection``: shock has no band response
            # and no effective-wavelength fallback, so it has no cascade to share
            # -- only this branch. Going through the helper would mean passing two
            # placeholder arguments to reach the call below.
            from tengri.observation.photometry import lnu_filter_integral_batch

            z = jnp.asarray(p["redshift"])
            shock_sed = published["sed_shock"]
            published = {
                **published,
                "shock_phot_lnu_precomp": lnu_filter_integral_batch(
                    shock_sed, state.wave, fw, ft, z
                ),
                "shock_restband_lnu_precomp": lnu_filter_integral_batch(
                    shock_sed, state.wave, fw, ft, 0.0
                ),
            }

        new_derived = self._merge_published(state.derived, published)
        return state.with_(sed_intrinsic=sed_out, derived=new_derived)
