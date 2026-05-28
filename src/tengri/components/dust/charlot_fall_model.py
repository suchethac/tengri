# SPDX-License-Identifier: BSD-3-Clause
"""Charlot & Fall (2000) two-component dust attenuation — SEDModelComponent port.

Birth-cloud + diffuse-ISM attenuation, applied per stellar age. Young
populations (< t_birth) see both components; old populations see only
the diffuse component. This is the canonical "two-component"
attenuation used in Charlot & Fall (2000) and adopted by Prospector,
Bagpipes, FSPS, CIGALE.

This port coexists with the existing `DustSEDComponent` in
`src/tengri/components/dust/two_component.py`, which bundles
attenuation + IR re-emission into one adapter. The new
`SEDModelComponent` style separates concerns: `CharlotFall` handles
attenuation only and publishes `L_absorbed`; an IR re-emission
component (`ModifiedBlackbodySED`, `DL07IRSEDComponent`, …) consumes
that downstream.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tengri.components.dust.attenuation import two_component_dust
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Fixed, Uniform

__all__ = ["CharlotFall"]


_C_AA_PER_S = 2.99792458e18


def _trapz_freq(L_lambda: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    nu = _C_AA_PER_S / wave
    return jnp.abs(jnp.trapezoid(L_lambda, nu))


class CharlotFall(SEDModelComponent):
    r"""Charlot & Fall (2000) two-component dust attenuation.

    Birth-cloud + diffuse-ISM attenuation with a smooth age transition.
    Young populations (age < t_birth) see both components; old
    populations see only the diffuse component:

    .. math::

        T(\lambda, \tau) =
            \begin{cases}
                e^{-\tau_{\rm BC} k_{\rm BC}(\lambda)}
                    \cdot e^{-\tau_{\rm diff} k_{\rm diff}(\lambda)} & \tau < t_{\rm birth} \\
                e^{-\tau_{\rm diff} k_{\rm diff}(\lambda)} & \tau \geq t_{\rm birth}
            \end{cases}

    The transition between regimes is a smooth log-age sigmoid (the
    `transition_width` config field, in dex) so the transmission is
    differentiable in every parameter.

    Cross-component contract
    ------------------------
    Reads:
      * ``lnu_age`` — (n_age, n_wave) per-age stellar L_ν cube
      * ``ssp_ages_yr`` — (n_age,) SSP age grid in years
    Publishes:
      * ``L_absorbed`` — total absorbed luminosity (erg/s) for the IR
        re-emission downstream

    Notes
    -----
    **JIT-compatible**: yes. ``predict`` is pure JAX; the sigmoid age
    transition is smooth so gradients flow.

    **Physics**: the attenuated stellar SED is
    :math:`L^{\rm out}_\nu(\lambda) = \sum_a T(\lambda, \tau_a)\,L^{\rm in}_{\nu,a}(\lambda)`
    where :math:`L^{\rm in}_{\nu,a}` is the per-age intrinsic stellar L_ν
    published by :class:`~tengri.components.stellar.component.StellarSEDComponent`.

    The non-stellar contribution to ``sed_in`` (from AGN, nebular,
    radio, X-ray) is passed through unattenuated — stellar dust does
    not absorb AGN/nebular/radio/X-ray emission.

    References
    ----------
    .. [1] S. Charlot & S. M. Fall, "A Simple Model for the Absorption
       of Starlight by Dust in Galaxies," ApJ, 539, 718 (2000).
       https://doi.org/10.1086/309250
    """

    name = "charlot_fall"
    parameter_prefix = "dust_"

    # ─── Free parameters
    tau_bc = Uniform(
        0.0,
        4.0,
        default=1.0,
        description="birth-cloud V-band optical depth",
        units="dimensionless",
    )
    tau_diff = Uniform(
        0.0,
        4.0,
        default=0.3,
        description="diffuse-ISM V-band optical depth",
        units="dimensionless",
    )

    # ─── Slope/bump knobs (Fixed by default; user overrides per fit)
    slope = Fixed(-0.7, description="diffuse-ISM attenuation slope", units="dimensionless")
    delta = Fixed(0.0, description="UV slope deviation (Noll+2009)", units="dimensionless")
    bump_strength = Fixed(0.0, description="2175 Å bump strength", units="dimensionless")

    inputs: dict[str, str] = {  # noqa: RUF012
        "lnu_age": "erg/s/Hz",
        "ssp_ages_yr": "yr",
    }
    outputs: dict[str, str] = {"L_absorbed": "erg/s"}  # noqa: RUF012

    # Structural (not JAX-traced); future PR may move to a Config dataclass.
    LAW_BC: str = "power_law"
    LAW_DIFF: str = "power_law"
    T_BIRTH_YR: float = 1e7
    TRANSITION_WIDTH_DEX: float = 0.3

    def predict(
        self,
        p: dict[str, Any],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        lnu_age: jnp.ndarray,
        ssp_ages_yr: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Apply Charlot & Fall two-component attenuation per stellar age.

        Parameters
        ----------
        p : dict
            Parameter dict with prefix stripped: ``p["tau_bc"]``,
            ``p["tau_diff"]``, plus the Fixed slope/bump knobs.
        sed_in : ndarray, shape (n_wave,)
            Rest-frame L_ν from upstream — typically the bare stellar
            sum :math:`\\sum_a L^{\\rm in}_{\\nu,a}` plus any non-stellar
            contribution (AGN, nebular) that should pass through unattenuated.
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid (Å).
        lnu_age : ndarray, shape (n_age, n_wave)
            Per-age intrinsic stellar L_ν cube, published by Stellar.
        ssp_ages_yr : ndarray, shape (n_age,)
            SSP age grid in years.

        Returns
        -------
        sed_out : ndarray, shape (n_wave,)
            Attenuated rest-frame L_ν.
        published : dict
            ``{"L_absorbed": L_absorbed_erg_s}``.
        """
        transmission = two_component_dust(
            wavelength=wave,
            age_grid=ssp_ages_yr,
            tau_v1=p["tau_bc"],
            tau_v2=p["tau_diff"],
            law_bc=self.LAW_BC,
            law_diff=self.LAW_DIFF,
            t_birth=self.T_BIRTH_YR,
            transition_width=self.TRANSITION_WIDTH_DEX,
            n_slope=p["slope"],
            dust_delta=p["delta"],
            dust_bump_strength=p["bump_strength"],
        )  # (n_age, n_wave)

        sed_stellar_intrinsic = jnp.sum(lnu_age, axis=0)
        sed_stellar_attenuated = jnp.sum(lnu_age * transmission, axis=0)

        # Non-stellar contribution rides through unattenuated.
        non_stellar = sed_in - sed_stellar_intrinsic
        sed_out = sed_stellar_attenuated + non_stellar

        L_absorbed = _trapz_freq(sed_stellar_intrinsic - sed_stellar_attenuated, wave)
        return sed_out, {"L_absorbed": L_absorbed}
