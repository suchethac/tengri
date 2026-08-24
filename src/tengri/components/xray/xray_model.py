# SPDX-License-Identifier: BSD-3-Clause
"""X-ray emission model (Aird et al. 2017 + Lusso & Risaliti 2016)
SEDModelComponent.

Implements the X-ray emission model (SFR-X-ray scaling + AGN X-ray) on the
SEDModelComponent contract. Provides differentiable X-ray SED prediction.

Physical pipeline
-----------------
1. SFR → X-ray binaries (HMXB + LMXB)
2. M_* → additional XRB scaling
3. L_AGN → AGN X-ray corona
4. Combine into X-ray SED
5. Add to full SED

Cross-component contract
------------------------
Inputs: sfr (star-formation rate), log_mstar (stellar mass), L_agn_bol
(AGN bolometric luminosity) from upstream with fallbacks.
Outputs: sed_xray (X-ray continuum on pipeline wavelength grid).

Notes
-----
**JIT-compatible**: yes.

**Models**: Lehmer et al. (2010, 2016) for X-ray binaries + Lusso &
Risaliti (2016) for AGN X-ray corona. Both are analytic scalings with
no grid dependence.

**Fallbacks**: X-ray is fully functional without AGN component; it
gracefully defaults to XRB-only emission.

References
----------
.. [1] Lehmer et al. 2010, ApJ, 724, 559
.. [2] Lehmer et al. 2016, ApJ, 825, 7
.. [3] Lusso & Risaliti 2016, ApJ, 819, 154
.. [4] Aird et al. 2017, MNRAS, 465, 3390
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.sed_model_component import SEDModelComponent
from tengri.components.xray.xray import metallicity_from_history, xray_total
from tengri.parameters.priors import Uniform
from tengri.protocols.component import (
    ParamDeclaration,
    SEDComponentConfig,
)

__all__ = ["XRayAirdSEDComponent", "XRayAirdSEDComponentConfig"]


@dataclass(frozen=True)
class XRayAirdSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for XRayAirdSEDComponent.

    Attributes
    ----------
    name: str
        Diagnostic identifier. Default ``"xray_aird"``.
    """

    name: str = "xray_aird"


class XRayAirdSEDComponent(SEDModelComponent):
    """SEDComponent for X-ray emission (Lehmer+2016 / Yang+2020 canonical).

    Computes total X-ray continuum from X-ray binaries (HMXB + LMXB),
    hot gas, and AGN corona. Implements Lehmer et al. 2016 (metallicity-
    and age-dependent XRB scaling), Yang et al. 2020 (hot gas and AGN
    anisotropy), and Just et al. 2007 (α_OX–L_2500 relation).

    Free parameters (6):

    - xray_gamma_hmxb: HMXB spectral index
    - xray_gamma_lmxb: LMXB spectral index
    - xray_gamma_agn: AGN X-ray spectral index
    - xray_log_nh: line-of-sight equivalent hydrogen column density,
      applied to the AGN corona only as
      ``T_phabs(E, N_H) × T_cabs(N_H) × intrinsic + 0.01 × intrinsic``
      (Ricci+2017; Matsumoto+2026 Eq. B6). T_phabs uses
      Morrison & McCammon (1983) wabs cross-sections; T_cabs is
      Thomson down-scattering. Galactic absorption is not modeled
      (assume user provides intrinsic-frame fluxes).
    - xray_det_hmxb: HMXB luminosity offset (deviation from expected
      SFR relation in log-space; positive = brighter X-ray)
    - xray_det_lmxb: LMXB luminosity offset (deviation from expected
      mass relation in log-space; positive = brighter X-ray)

    The α_OX parameter is not a free parameter here; it is a PRIOR
    that couples AGN UV (L_2500) and X-ray emission self-consistently
    (Just+2007, Yang+2020). Offsets from the empirical relation can be
    passed at the function level via ``delta_alpha_ox``.

    Notes
    -----
    **JIT-compatible**: yes.

    **Optional inputs** (#1846, #1755, #1706 fixed): declares six cross-component
    keys (sfr, log_mstar, L_2500_30deg, age_weights, ssp_ages_yr, log_metallicity_history).
    The base ``apply()`` injects 0.0 when absent; ``predict`` distinguishes these
    injected fallbacks via zero-sum checks (age_weights) and direct presence logic
    (sfr, log_mstar, L_2500_30deg). Reads X with sensible defaults: SFR from stellar
    (→ 1.0 Msun/yr), stellar mass from stellar (→ 10 Msun), AGN UV from AGN
    (→ 0.0, no corona), mass-weighted age from stellar (→ 1.0 Gyr), and metallicity
    from stellar (→ Z_sun). Matches yang20's pattern
    (:class:`tengri.components.xray.component.XRaySEDComponent`).

    **Models**:

    - HMXB: Lehmer+2016 metallicity quartic, scaling with SFR
    - LMXB: Lehmer+2016 age quartic, scaling with M_star
    - Hot gas: Yang+2020, scaling with SFR
    - AGN corona: Just+2007 / Yang+2020 α_OX, scaling with L_2500

    """

    def __init__(self) -> None:
        """Initialize component with xray_aird config."""
        self.config = XRayAirdSEDComponentConfig()

    name: str = "xray_aird"
    parameter_prefix: str = "xray_"
    requires_template_data: ClassVar[bool] = False

    #: Cross-component reads with documented fallbacks. These are declared so
    #: the base ``apply`` can wire them from upstream publishers when present
    #: (see ADR-0009). The fallback chain distinguishes published 0.0 from
    #: the injected 0.0 when a key is absent: sfr/log_mstar/L_2500_30deg check
    #: against zero after injection (line 765 of sed_model_component.py),
    #: while stellar_age_gyr is computed as a mass-weighted average that requires
    #: its summand (age_weights) to distinguish absence. This fixes #1846, #1755,
    #: #1706 recurrence.
    optional_inputs: ClassVar[dict[str, str]] = {
        "sfr": "Msun/yr",
        "log_mstar": "dex",
        "L_2500_30deg": "erg/s/Hz",
        "age_weights": "Msun",
        "ssp_ages_yr": "yr",
        "log_metallicity_history": "dex",
    }

    #: Publish into the shared ``xray`` domain rather than under the registry
    #: key. ``DerivedState`` declares ``xray_phot_lnu_precomp`` /
    #: ``xray_spec_lnu_precomp`` / ``xray_restband_lnu_precomp``; keying them off
    #: ``name`` instead would emit ``xray_aird_*``, which are not fields, so they
    #: spill into ``_extras`` and the ADR-0007 guard raises ``ComponentIOError``
    #: on every WavePrecomp build. Only one X-ray component is ever built, so the
    #: two can share the domain without colliding.
    publish_name: ClassVar[str] = "xray"

    # Free parameters
    gamma_hmxb = Uniform(
        1.0,
        3.0,
        description="HMXB spectral index",
        units="dimensionless",
        default=2.0,
    )
    gamma_lmxb = Uniform(
        1.0,
        3.0,
        description="LMXB spectral index",
        units="dimensionless",
        default=1.7,
    )
    gamma_agn = Uniform(
        1.0,
        3.0,
        description="AGN X-ray spectral index",
        units="dimensionless",
        default=1.9,
    )
    log_nh = Uniform(
        20.0,
        26.0,
        description=(
            "Line-of-sight equivalent hydrogen column density; applied as "
            "T_phabs × T_cabs to AGN corona (Ricci+2017; Matsumoto+2026 "
            "Eq. B6). Compton-thick regime starts at log_nh = 24."
        ),
        units="log10(cm^-2)",
        default=21.0,
    )
    det_hmxb = Uniform(
        -2.0,
        2.0,
        description=(
            "Deviation from expected HMXB log L_X (Yang+2020 [1]_). "
            "Positive = brighter X-ray. Allows intrinsic scatter or evolution "
            "around the Lehmer+2016 SFR relation."
        ),
        units="dex",
        default=0.0,
    )
    det_lmxb = Uniform(
        -2.0,
        2.0,
        description=(
            "Deviation from expected LMXB log L_X (Yang+2020 [1]_). "
            "Positive = brighter X-ray. Allows intrinsic scatter or evolution "
            "around the Lehmer+2016 age/mass relation."
        ),
        units="dex",
        default=0.0,
    )
    # alpha_ox is deliberately NOT a free parameter here; PR #329 promotes it to
    # an empirical prior derived from L_2500 via alpha_ox_from_l2500()
    # (Just+2007 / Lusso–Risaliti). Offsets from the empirical value are exposed
    # via delta_alpha_ox in xray_agn_corona{,_from_disc}. See ADR-0015.

    # No required cross-component inputs (all have fallbacks)
    inputs: ClassVar[dict[str, str]] = {}
    outputs: ClassVar[dict[str, str]] = {
        "sed_xray": "erg/s/Hz",
    }

    def load(self, wave: jnp.ndarray | None = None) -> None:
        """No-op precomputation for analytic X-ray model."""
        return None

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Declare the 3 free parameters owned by X-ray."""
        return super().declared_parameters()

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Predict X-ray SED via Lehmer+Yang canonical path.

        Parameters
        ----------
        p: mapping[str, ndarray]
            Parameters with prefix stripped: gamma_hmxb, gamma_lmxb, gamma_agn,
            log_nh, det_hmxb, det_lmxb.
        sed_in: ndarray
            Input SED (stellar + nebular + AGN + radio).
        wave: ndarray
            Rest-frame wavelength grid in Angstrom.
        **inputs: ndarray
            Declared cross-component reads with documented fallback behavior:

            - sfr [Msun/yr], when stellar publishes, reads the value; when
              absent, the base apply() injects 0.0, yielding zero XRB luminosity
              (physically defensible: no SFH → no star-formation-driven X-rays).
            - log_mstar [dex], when stellar publishes, reads the value; when
              absent, injected 0.0 (10^0 = 1 Msun, a degenerate case but not
              used in practice since X-ray requires stellar component).
            - L_2500_30deg [erg/s/Hz], when AGN publishes L_2500, reads it;
              when absent, injected 0.0 (no AGN → no corona).
            - age_weights [Msun]: mass weights on SSP grid. Uses zero-sum check:
              if sum > 0, computes mass-weighted stellar age; if sum == 0 (absent
              or truly zero-weighted), falls back to 1.0 Gyr.
            - ssp_ages_yr [yr]; SSP age grid; paired with age_weights.
            - log_metallicity_history [dex, absolute log10(Z)]: stellar's SFH-
              indexed metallicity history. Present-day bin (index 0) drives the
              Lehmer+2016 HMXB quartic; if absent, metallicity_from_history()
              returns Z_SUN (0.0142).

        Returns
        -------
        tuple[ndarray, mapping]

            - sed_out: sed_in + X-ray continuum.
            - published: Dict with "sed_xray".

        Notes
        -----
        Implements the "declared-input" cross-component read pattern (ADR-0009):
        every key read here is declared in optional_inputs() so the validator can
        check units and the base apply() can wire them. Absent publishers are
        signaled by injected 0.0 from the base class; for most inputs (sfr,
        log_mstar, L_2500_30deg) this produces physically defensible zero-
        contribution SED. For the mass-weighted age, a zero-sum check
        (sum(age_weights) == 0) distinguishes "not published" from "truly zero-
        weighted" and enables the 1.0 Gyr fallback. Fixes #1846, #1755, #1706.
        """
        # SFR [Msun/yr]: read when stellar publishes, else use physical default 1.0
        sfr = jnp.asarray(inputs.get("sfr", 1.0))

        # Stellar mass [log10 Msun]: read when stellar publishes, else 10.0
        log_mstar = jnp.asarray(inputs.get("log_mstar", 10.0))
        stellar_mass = 10.0**log_mstar

        # Absolute log10(Z) per SFH bin; index 0 is the present-day value, so
        # this is the metallicity the young HMXB population was born with. The
        # same reduction the nebular component uses. When absent, metallicity_from_history
        # returns Z_SUN (0.0142). Before #1755, this read a "metallicity_z" key
        # that nothing publishes, so the fallback was the only value ever seen.
        metallicity_z = metallicity_from_history(inputs.get("log_metallicity_history"))

        # Stellar age [Gyr]: computed from age_weights and ssp_ages_yr when both
        # are published (age_weights sum > 0), else falls back to 1.0 Gyr.
        # This mirrors yang20's emitter_inputs logic (component.py:374-388).
        # The base apply() injects 0.0 when these keys are absent; we distinguish
        # that from published zeros by checking the weight sum.
        age_weights = jnp.asarray(inputs.get("age_weights", 0.0))
        ssp_ages_yr = jnp.asarray(inputs.get("ssp_ages_yr", 0.0))
        _w_sum = jnp.sum(age_weights)
        # Use where-safe denominator to avoid subnormal floor + where-grad hazard
        w_sum_safe = jnp.where(_w_sum > 0.0, _w_sum, 1.0)
        stellar_age_gyr = jnp.where(
            _w_sum > 0.0,
            jnp.sum(age_weights * ssp_ages_yr) / w_sum_safe / 1.0e9,
            1.0,
        )

        # AGN UV luminosity [erg/s/Hz]: read when AGN publishes, else 0.0
        l_2500_30deg = jnp.asarray(inputs.get("L_2500_30deg", 0.0))

        # Call xray_total with new signature
        L_xray = xray_total(
            wave,
            sfr=sfr,
            stellar_mass=stellar_mass,
            metallicity_z=metallicity_z,
            stellar_age_gyr=stellar_age_gyr,
            l_2500_30deg=l_2500_30deg,
            gamma_hmxb=jnp.asarray(p["gamma_hmxb"]),
            gamma_lmxb=jnp.asarray(p["gamma_lmxb"]),
            gamma_agn=jnp.asarray(p["gamma_agn"]),
            E_cut=300.0,  # fixed cutoff (E_cut is no longer a free parameter)
            delta_alpha_ox=0.0,  # no offset from the empirical α_ox prior
            cos_inc=1.0,  # face-on by default
            apply_anisotropy=False,  # XRayAirdSEDComponent stays inclination-agnostic
            a1=0.5,
            a2=0.0,
            log_nh=jnp.asarray(p["log_nh"]),
            # Default 0.0 keeps predict robust to partial param dicts that
            # predate the det_hmxb/det_lmxb XRB offsets (CIGALE parity 2026-06).
            log_L_hmxb_offset=jnp.asarray(p.get("det_hmxb", 0.0)),
            log_L_lmxb_offset=jnp.asarray(p.get("det_lmxb", 0.0)),
        )

        return sed_in + L_xray, {
            "sed_xray": L_xray,
        }
