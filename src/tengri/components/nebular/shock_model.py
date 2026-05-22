"""Shock nebular emission — SEDModelComponent port.

MAPPINGS V (3MdBs) shock + precursor emission lines on an arbitrary
wavelength grid. Used to model the line emission from shock-ionised
gas (AGN narrow-line regions, supernova remnants, galactic outflows).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from tengri.components.nebular.shock import compute_shock_sed
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Fixed, Uniform

__all__ = ["ShockNebular"]


class ShockNebular(SEDModelComponent):
    r"""MAPPINGS V shock-driven nebular emission.

    Places shock + precursor emission lines on the rest-frame wavelength
    grid using the 3MdBs MAPPINGS V grids (Alarie & Morisset 2019).
    The normalisation is the total Hα luminosity (``l_shock_halpha``),
    a flexible free parameter that maps directly to the inferred shock
    energetics.

    Cross-component contract
    ------------------------
    Reads: nothing — shock emission is parameterised directly by its
    own free parameters; SFH coupling (if desired) is left to the user
    via a fixed prior anchored to upstream quantities.
    Publishes: ``L_shock`` — total shock luminosity (erg/s).

    Notes
    -----
    **JIT-compatible**: yes. ``predict`` is pure JAX; the categorical
    knobs (``abundance``, ``component``) are class-level constants and
    not part of the JIT trace.

    **Component selector**. Set the class-level ``COMPONENT`` constant
    to one of ``"shock"`` (post-shock only), ``"precursor"`` (pre-shock
    photo-ionisation), or ``"combined"`` (sum). Default is ``"combined"``.

    References
    ----------
    .. [1] M. A. Allen et al., "The MAPPINGS III Library of Fast Radiative
       Shock Models," ApJS, 178, 20 (2008).
       https://doi.org/10.1086/589652
    .. [2] C. Alarie & C. Morisset, "Extensive Online Shock Model Database,"
       Rev. Mex. Astron. Astrofis., 55, 377 (2019).
       https://doi.org/10.22201/ia.01851101p.2019.55.02.21
    """

    name = "shock"
    parameter_prefix = "shock_"

    log_l_halpha = Uniform(38.0, 44.0, description="log Hα luminosity", units="dex (erg/s)")
    velocity = Uniform(150.0, 1000.0, description="shock velocity", units="km/s")
    log_density = Fixed(0.0, description="log pre-shock density", units="dex (cm^-3)")
    b_over_sqrt_n = Fixed(1.0, description="magnetic-field parameter", units="μG")
    line_sigma_aa = Fixed(0.0, description="Gaussian line width", units="Å")

    inputs: dict[str, str] = {}  # noqa: RUF012
    outputs: dict[str, str] = {"L_shock": "erg/s"}  # noqa: RUF012

    # Structural knobs (categorical — kept off the JIT trace).
    ABUNDANCE: str = "solar"
    COMPONENT: str = "combined"

    def predict(
        self,
        p: dict[str, Any],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        l_halpha = jnp.power(10.0, p["log_l_halpha"])
        addition = compute_shock_sed(
            wavelength=wave,
            shock_velocity=p["velocity"],
            l_shock_halpha=l_halpha,
            shock_log_density=p["log_density"],
            shock_b_over_sqrt_n=p["b_over_sqrt_n"],
            shock_abundance=self.ABUNDANCE,
            shock_component=self.COMPONENT,
            line_sigma_aa=p["line_sigma_aa"],
        )
        # ``L_shock`` is the total shock-driven luminosity — the
        # frequency integral of the published L_nu, not just the
        # Hα anchor. Downstream consumers (radio energy balance,
        # diagnostic plots) want the total, not a single line.
        c_aa_per_s = 2.99792458e18
        nu = c_aa_per_s / wave
        L_shock = jnp.abs(jnp.trapezoid(addition, nu))
        return sed_in + addition, {"L_shock": L_shock}
