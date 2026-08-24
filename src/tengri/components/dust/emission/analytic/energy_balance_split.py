# SPDX-License-Identifier: BSD-3-Clause
"""Two-temperature energy-balance dust emission as SEDModelComponent.

Wraps the pure closure :func:`~tengri.components.dust.emission.energy_balance_split`,
a MAGPHYS/Kokorev+2021-style warm+cold two-temperature split with an optional
AGN-heated IR contribution.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.parameters.resolve import require_redshift

__all__ = ["EnergyBalanceSplitIRSEDComponent"]


def _log10_nonneg(value: jnp.ndarray) -> jnp.ndarray:
    """log10 of a non-negative quantity, mapping 0 to ``-inf`` grad-safely.

    Parameters
    ----------
    value: array_like
        A non-negative luminosity [erg/s]; ``0`` denotes an exactly absent term.

    Returns
    -------
    ndarray
        ``log10(value)`` [dex], or ``-inf`` where ``value == 0``. The
        where-dummy keeps the backward pass free of NaN at ``value == 0``.
    """
    from tengri.utils.scale import log10_magnitude

    # Delegates so the corrupt/zero contract has one definition (#1527). The
    # hand-rolled version here tested ``value > 0``, which is False for NaN, so
    # a corrupt L_absorbed became -inf and powered back to exactly 0.0.
    return log10_magnitude(value)


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
    ``tengri.components.dust._params`` because ``dust_eta_balance`` and the
    energy-balance bookkeeping are shared with the attenuator. Re-declaring them
    here would raise a duplicate-declaration error, so ``predict`` reads them
    from the sliced parameter dict instead.

    ``eta_balance`` is applied to ``L_ir`` upstream by the attenuator, so the
    closure is called with ``eta_balance=1.0`` here (the incoming ``L_ir`` is
    already the scaled budget).

    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    References
    ----------
    .. [1] Kokorev, V. I., Magdis, G. E., Davidzon, I., et al. 2021, ApJ, 921, 40.
    """

    name: str = "energy_balance_split"

    #: The six knobs ``predict`` reads, declared in ``components/dust/_params.py``
    #: rather than on this class (see the class Notes: re-declaring them beside
    #: the attenuator's shared ``dust_eta_balance`` would raise a duplicate).
    #:
    #: Without this, ``_declared_param_names`` cannot tell an empty ``_priors``
    #: meaning "declared elsewhere" from one meaning "reads nothing", leaves the
    #: sub-block unnarrowed, and ``dust.emission {'*': FREE}`` hands the sampler
    #: the whole static union of every IR engine's parameters. Measured before
    #: this marker: **20 freed, 6 moving the prediction, 14 inert** -- fourteen
    #: flat directions (``dust_qpah``, ``dust_umin``, ``dust_alpha_dale``, ...)
    #: belonging to Draine & Li, Dale and MBB engines that are not even built.
    #: That is #1482, in the one engine #1482's own fix could not reach.
    #:
    #: ``dust_eta_balance`` is included even though ``predict`` never reads it
    #: -- it is applied to ``L_ir`` upstream by the attenuator, and this
    #: component is called with ``eta_balance=1.0``. Physically it is the
    #: attenuator's knob, and an earlier revision of this set left it out on
    #: exactly that reasoning. That was wrong, and measurably so: the grammar
    #: partitions it into **dust.emission**, not ``dust``
    #: (``dust_emission={'eta_balance': ...}`` via top-level group), so dropping
    #: it here left it freed by no wildcard at all --
    #: ``dust_emission={'all_params': FREE}`` does not reach it either. It is
    #: live (it scales ``L_ir``, so it moves the prediction), and orphaning a
    #: live parameter is the opposite failure to the one this marker fixes.
    #:
    #: ``dust_L_agn_ir`` is present although no wildcard can free it -- it
    #: declares no ``free_prior`` by design -- because this set states what the
    #: component reads, not what happens to be freeable.
    reads_parameters = frozenset(
        {
            "dust_T_warm",
            "dust_T_cold",
            "dust_f_cold",
            "dust_beta_warm",
            "dust_beta_cold",
            "dust_L_agn_ir",
            "dust_eta_balance",
        }
    )

    #: The menu row has credited Kokorev+2021 all along
    #: (``_DUST_EMISSION_METADATA``); the component itself credited nobody.
    citations = ("kokorev2021",)

    #: Affine, so it opts out of the generic ``apply``-level ``L_ir`` factoring
    #: (which re-applies a single scale after evaluating at unit luminosity:
    #: valid only for a proportional model). Instead this component assembles its
    #: two-term budget in log space *inside* :meth:`predict` and does its own
    #: single rescale, so ``apply`` must leave the scale alone. See the
    #: float32 discussion in :meth:`predict`.
    factors_l_ir: ClassVar[bool] = False

    #: The log budget ``log_L_ir`` [dex] published by the attenuator is the
    #: float32-safe form of the ~2.4e43 erg/s ``L_ir`` (``inf`` in float32).
    #: Declared alongside the inherited linear ``L_ir`` so both reach
    #: :meth:`predict`; only the log form is used.
    optional_inputs: ClassVar[dict[str, str]] = {"L_ir": "erg/s", "log_L_ir": "dex"}

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float = 0.0,
        log_L_ir: float = -jnp.inf,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        r"""Compute the two-temperature + AGN-IR dust emission.

        Parameters
        ----------
        p: dict
            Parameters with the ``dust_`` prefix stripped. Reads the globally
            declared ``T_warm``, ``T_cold``, ``f_cold``, ``beta_warm``,
            ``beta_cold``, ``L_agn_ir`` (falling back to their defaults).
        sed_in: ndarray, shape (n_wave,)
            Input SED in erg/s/Hz (typically zeros for an emission component).
        wave: ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        L_ir: float
            Absorbed (eta-scaled) luminosity [erg/s]. Ignored: ~2.4e43 and so
            ``inf`` in float32; the log form ``log_L_ir`` is used instead.
        log_L_ir: float
            ``log10(L_ir / (erg/s))`` [dex], the float32-safe budget published
            by the attenuator. ``-inf`` (its absent sentinel) means nothing was
            absorbed and the emission is exactly zero.

        Returns
        -------
        tuple[ndarray, dict]
            ``(sed_out, {"sed_dust_ir": emission})`` with the emission in erg/s/Hz.

        Notes
        -----
        **JIT-compatible**: yes, all operations are ``jnp`` primitives.

        The IR budget is *affine*, not proportional:

        .. math::

            L_{\rm IR}^{\rm tot} = L_{\rm IR} + L_{\rm AGN,IR}

        with both terms ~1e43 erg/s and so ``inf`` in float32. It is assembled
        in log space via :func:`~tengri.utils.scale.log10_add` so the sum is
        never materialized linearly, then applied *once* to the unit-luminosity
        two-temperature shape :math:`S(\lambda)` via
        :func:`~tengri.utils.scale.apply_log10_scale`. Because
        ``modified_blackbody`` is exactly linear in its ``L_absorbed``,
        :math:`{\rm ebs}(\lambda, 1, 0) = S(\lambda)` and
        :math:`{\rm ebs}(\lambda, L, A) = (L + A)\,S(\lambda)`, so this is exact
        in float64 (to ~1e-14 relative, the log round-trip) and finite in float32
        whenever the *net* budget is representable (#1206).

        The default ``dust_L_agn_ir = 0``: strict stellar energy balance: is
        fully float32-clean. A nonzero AGN-IR luminosity is a linear erg/s
        parameter and must itself be float32-representable (:math:`\lesssim
        3\times10^{38}` erg/s); a larger value stays out of range until
        ``dust_L_agn_ir`` moves to a log parameter (#1206 item 3).
        """
        from tengri.components.dust.emission.emission import energy_balance_split as ebs_fn
        from tengri.utils.scale import apply_log10_scale, log10_add

        z = jnp.asarray(
            require_redshift(p, "components.dust.emission.analytic.energy_balance_split.predict")
        )

        # Affine budget in log space: L_ir_total = L_ir + dust_L_agn_ir. Both
        # terms are ~1e43 erg/s (inf in float32); log10_add sums their log
        # magnitudes without ever forming the linear sum. -inf on either term
        # (absent) drops out exactly.
        log_total = log10_add(jnp.asarray(log_L_ir), _log10_nonneg(p.get("L_agn_ir", 0.0)))

        # Unit-luminosity two-temperature shape S(lambda), independent of the
        # total. Evaluated at L_absorbed_stellar=1, L_agn_ir=0 so ebs_fn returns
        # exactly S; the true scale is re-applied below.
        shape = ebs_fn(
            wave,
            1.0,
            L_agn_ir=0.0,
            eta_balance=1.0,  # already applied to L_ir by the attenuator
            f_cold=p.get("f_cold", 0.5),
            dust_T_warm=p.get("T_warm", 45.0),
            dust_T_cold=p.get("T_cold", 20.0),
            dust_beta_warm=p.get("beta_warm", 1.5),
            dust_beta_cold=p.get("beta_cold", 2.0),
            redshift=z,
        )
        sed = apply_log10_scale(shape, log_total)
        return sed_in + sed, {"sed_dust_ir": sed}
