# SPDX-License-Identifier: BSD-3-Clause
"""Two-temperature energy-balance dust emission as SEDModelComponent.

Wraps the pure closure :func:`~tengri.components.dust.emission.energy_balance_split`,
a MAGPHYS/Kokorev+2021-style warm+cold two-temperature split with an optional
AGN-heated IR contribution.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.resolve import require_redshift

__all__ = ["EnergyBalanceSplitIRSEDComponent"]


class EnergyBalanceSplitIRSEDComponent(EmissionComponent):
    """Two-temperature (warm + cold) energy-balance dust IR emission.

    Decomposes the re-radiated IR into a warm, SF-heated component and a cold,
    diffuse-ISM component (fraction ``f_cold``), and optionally adds an
    AGN-heated IR luminosity ``L_agn_ir`` on top of the strict stellar
    energy-balance budget.

    Notes
    -----
    **Parameters are declared globally**, not on this class: the warm/cold and
    AGN-IR knobs (``dust_T_warm``, ``dust_T_cold``, ``dust_f_cold``,
    ``dust_beta_warm``, ``dust_beta_cold``, ``dust_L_agn_ir``) live in
    :mod:`tengri.components.dust._params` because ``dust_eta_balance`` and the
    energy-balance bookkeeping are shared with the attenuator. Re-declaring them
    here would raise a duplicate-declaration error, so ``predict`` reads them
    from the sliced parameter dict instead.

    ``eta_balance`` is applied to ``L_ir`` upstream by the attenuator, so the
    closure is called with ``eta_balance=1.0`` here (the incoming ``L_ir`` is
    already the scaled budget).

    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    References
    ----------
    .. [1] Kokorev, V. I., Magdis, G. E., Davidzon, I., et al. 2021, ApJ, 921, 40.
    """

    name: str = "energy_balance_split"

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute the two-temperature + AGN-IR dust emission.

        Parameters
        ----------
        p : dict
            Parameters with the ``dust_`` prefix stripped. Reads the globally
            declared ``T_warm``, ``T_cold``, ``f_cold``, ``beta_warm``,
            ``beta_cold``, ``L_agn_ir`` (falling back to their defaults).
        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (typically zeros for an emission component).
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        L_ir : float
            Absorbed (eta-scaled) luminosity in erg/s.

        Returns
        -------
        tuple[ndarray, dict]
            ``(sed_out, {"sed_dust_ir": emission})`` with the emission in erg/s/Hz.
        """
        from tengri.components.dust.emission.emission import energy_balance_split as ebs_fn

        z = jnp.asarray(
            require_redshift(p, "components.dust.emission.analytic.energy_balance_split.predict")
        )
        sed = ebs_fn(
            wave,
            L_ir,
            L_agn_ir=p.get("L_agn_ir", 0.0),
            eta_balance=1.0,  # already applied to L_ir by the attenuator
            f_cold=p.get("f_cold", 0.5),
            dust_T_warm=p.get("T_warm", 45.0),
            dust_T_cold=p.get("T_cold", 20.0),
            dust_beta_warm=p.get("beta_warm", 1.5),
            dust_beta_cold=p.get("beta_cold", 2.0),
            redshift=z,
        )
        return sed_in + sed, {"sed_dust_ir": sed}
