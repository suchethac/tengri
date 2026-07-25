# SPDX-License-Identifier: BSD-3-Clause
"""Composable shock nebular emission — the canonical shock ``SEDModelComponent``.

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

* ``norm="frac"`` (**default**) — *relative*: the shock Hα luminosity is a
  fraction ``shock_frac`` of the galaxy's approximate Hα
  (``L(Hα) ~ 1e-3 L_bol`` of the SED accumulated so far). Intuitive "how
  much of the line budget is shock-driven"; reproduces the legacy
  :func:`tengri.forward.emission_helpers.shock_emission` bit-for-bit.
* ``norm="lhalpha"`` — *absolute*: the shock Hα luminosity is set directly by
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

import jax.numpy as jnp

from tengri.components.nebular.shock import compute_shock_sed
from tengri.components.sed_model_component import SEDModelComponent
from tengri.protocols.component import SEDComponentConfig
from tengri.utils.physics_constants import C_AA

__all__ = ["ShockNebular", "ShockNebularConfig"]


@dataclass(frozen=True)
class ShockNebularConfig(SEDComponentConfig):
    """Static (non-traced) configuration for :class:`ShockNebular`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"shock"``.
    norm : str
        Normalization mode: ``"frac"`` (relative to the galaxy's approximate
        Hα) or ``"lhalpha"`` (absolute ``shock_log_lhalpha``). Default
        ``"frac"``.
    abundance : str
        MAPPINGS abundance set: ``"solar"`` | ``"2xsolar"`` | ``"lmc"`` |
        ``"smc"`` | ``"dopita2005"``. Default ``"solar"``.
    component : str
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
    (:data:`tengri.components.nebular._params.SHOCK_PARAMS`, registered by
    ``Parameters(shock=True)`` / the ``shock`` grammar group) rather than
    auto-declaring them, so it composes with — and never double-declares
    against — the photoionized nebular backend.

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

    # Params are supplied by the ``shock_*`` bucket, not auto-declared here.
    inputs: ClassVar[dict[str, str]] = {}
    outputs: ClassVar[dict[str, str]] = {"sed_shock": "erg/s/Hz"}

    def predict(
        self,
        p: dict[str, Any],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        r"""Add MAPPINGS V shock emission to the running SED.

        Parameters
        ----------
        p : dict[str, ndarray]
            Shock parameters, ``shock_`` prefix stripped: ``frac``,
            ``log_lhalpha``, ``velocity`` [km/s], ``log_density`` [dex cm^-3],
            ``b_over_sqrt_n`` [μG].
        sed_in : ndarray, shape (n_wave,)
            SED accumulated so far [erg/s/Hz]. Used for the ``norm="frac"``
            Hα normalization only.
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].

        Returns
        -------
        sed_out : ndarray, shape (n_wave,)
            ``sed_in`` plus the shock contribution [erg/s/Hz].
        published : dict
            ``{"sed_shock": ndarray}`` — the shock contribution [erg/s/Hz].
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

        # Float32 (#1206): the shock Hα luminosity is ~1e41 erg/s — ``inf`` in
        # float32 (max 3.4e38) — so ``l_shock_halpha * <line profile>`` becomes
        # ``inf * 0 = nan`` even though the resulting SED (~1e26 erg/s/Hz) is
        # perfectly representable. The shock SED is *exactly* linear in
        # ``l_shock_halpha`` (verified: ×10 per dex), so on the float32 path we
        # evaluate the shape at unit Hα and re-apply the true luminosity as a
        # log10 offset. Float64 keeps the linear expressions, bit-identical.
        # NB: gate on ``sed_in`` — the SED actually being accumulated — not on
        # ``wave``, which arrives as float64 even when the pipeline runs in
        # float32. ``sed_in`` is float32 for BOTH pure float32 and the
        # mixed-precision ``forward_dtype="float32"`` mode, which is exactly the
        # set of cases that need the log-space rescale.
        _f32 = jnp.asarray(sed_in).dtype == jnp.float32

        if self.config.norm == "lhalpha":
            _log_l_shock_halpha = jnp.asarray(p["log_lhalpha"])
            l_shock_halpha = jnp.asarray(1.0) if _f32 else jnp.power(10.0, p["log_lhalpha"])
        else:
            # Relative: fraction of the galaxy's approximate Hα. Same
            # order-of-magnitude proxy (L(Hα) ~ 1e-3 L_bol) as the legacy
            # ``shock_emission`` helper, so the two paths agree bit-for-bit.
            if _f32:
                # ``l_bol`` ~1e44 erg/s overflows too: peak-factor the integrand
                # so only the O(1) residual is integrated, and carry the peak in
                # log10 along with the 1e-3 proxy and the user's fraction.
                _peak = jnp.max(jnp.abs(sed_in))
                _peak = jnp.where(_peak > 0.0, _peak, 1.0)
                _hat_l_bol = -jnp.trapezoid(sed_in / _peak, nu)
                _log_l_shock_halpha = (
                    jnp.log10(_peak)
                    + jnp.log10(jnp.maximum(_hat_l_bol, 1e-30))
                    - 3.0
                    + jnp.log10(jnp.maximum(p["frac"], 1e-30))
                )
                l_shock_halpha = jnp.asarray(1.0)
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
        )
        if _f32:
            from tengri.utils.scale import apply_log10_scale

            shock_sed = apply_log10_scale(shock_sed, _log_l_shock_halpha)
        return sed_in + shock_sed, {"sed_shock": shock_sed}
