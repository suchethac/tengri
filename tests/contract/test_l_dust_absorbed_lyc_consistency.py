# SPDX-License-Identifier: BSD-3-Clause
"""One absorbed-luminosity definition: compute_l_dust_absorbed == bolometric_absorbed_log10.

Two public spellings of "energy removed from the SED by dust attenuation"
used to disagree by the Lyman-continuum fraction (#922): ``compute_l_dust_absorbed``
(``tengri.utils.sed_quantities``) integrated the *whole* wavelength grid, while
``bolometric_absorbed_log10`` (``tengri.forward.energy_balance``, the pipeline's
own dust normalization, called from ``DustAttenuationSEDComponent.apply``/
``DustSEDComponent.apply``) masks :math:`\\lambda < 912` A (LyC photons ionize
hydrogen rather than heat dust). ``agnfitter_priors``'s ``energy_balance`` branch
called the unmasked utility to build ``l_gal_att`` while comparing it against
``l_sb_emit`` (the bolometric integral of the *masked*-normalized ``sed_dust_ir``),
so ``l_gal_att`` structurally exceeded ``l_sb_emit`` by the LyC fraction and
``prior_energy_balance`` returned ``AGNFITTER_HARD_REJECT`` for every
Calzetti-attenuated star-forming galaxy, independent of ``tau_v``/``dust_T``/
``dust_eta_balance``.

Measured before this fix (Task 9 re-review, 2026-09-06, on the reproduction
notebook's illustrative galaxy: z=0.5, schreiber2018 dust, Calzetti screen from
E(B-V)=0.3): unmasked integral 4.0899e43 erg/s vs masked (== the pipeline's own
``L_absorbed`` == ``sed_dust_ir``'s bolometric integral) 3.9443e43 erg/s, ratio
0.9644 < 1 -- always a hard reject.

Both functions now build their integrand through the SAME shared helper
(:func:`tengri.forward.energy_balance.absorbed_integrand`, masked by
:data:`tengri.forward.energy_balance.LYMAN_CUTOFF_AA`), so the two spellings
cannot silently disagree again.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
from tengri.forward.energy_balance import bolometric_absorbed_log10
from tengri.observation.photometry import FilterCurve
from tengri.parameters.agn_priors import AGNFITTER_HARD_REJECT, agnfitter_priors
from tengri.utils.physics_constants import C_AA, L_SUN
from tengri.utils.sed_quantities import compute_l_dust_absorbed

pytestmark = pytest.mark.contract


def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


def _build(ssp, dust_emission_type: str) -> SEDModel:
    """Calzetti-attenuated star-forming galaxy: single-component screen, z=0.5.

    ``tau_v = R_V * E(B-V) / 1.086`` with ``E(B-V) = 0.3``, ``R_V = 4.05``
    (Calzetti's normalization), matching the illustrative galaxy this bug was
    found on.
    """
    obs = Observation(
        photometry=Photometry(
            filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0, 9000.0, 1.0e6))
        )
    )
    tau_v = 4.05 * 0.3 / 1.086
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_v": tau_v,
            },
            dust_emission={"type": dust_emission_type, "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            redshift=Fixed(0.5),
        )


@pytest.mark.parametrize("dust_emission_type", ["schreiber2018", "dl07"])
def test_compute_l_dust_absorbed_matches_bolometric_absorbed_log10(
    synthetic_ssp_wide, dust_emission_type
):
    """The two public spellings of L_absorbed agree to 1e-10 relative.

    Conservation law: both are the same integral
    ``integral_{lambda >= 912 A} (L_nu_intrinsic - L_nu_attenuated) dnu`` and must
    return the identical energy once they share the masking convention.
    """
    model = _build(synthetic_ssp_wide, dust_emission_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = {**model.spec.get_fixed_values()}
        pred = model.predict(params)
    components = pred.sed.components
    wave = components["wavelength"]
    nu = C_AA / wave

    l_abs_via_sed_quantities_ergs = (
        float(
            compute_l_dust_absorbed(
                components["sed_intrinsic"], components["sed_attenuated"], wave
            )
        )
        * L_SUN
    )
    log_mag, _sign = bolometric_absorbed_log10(
        components["sed_intrinsic"], components["sed_attenuated"], nu, wave=wave
    )
    l_abs_via_energy_balance_ergs = float(10.0**log_mag)

    assert l_abs_via_sed_quantities_ergs > 0.0
    np.testing.assert_allclose(
        l_abs_via_sed_quantities_ergs,
        l_abs_via_energy_balance_ergs,
        rtol=1e-10,
        err_msg=(
            f"[{dust_emission_type}] compute_l_dust_absorbed "
            f"({l_abs_via_sed_quantities_ergs:.6e} erg/s) no longer matches "
            f"10**bolometric_absorbed_log10 ({l_abs_via_energy_balance_ergs:.6e} erg/s) "
            "-- the two public L_absorbed spellings have drifted apart again (#922)."
        ),
    )

    # Regression: the pre-fix unmasked call (include_lyc=True) must NOT equal
    # the masked default on this UV-bright-enough fixture -- the historical bug
    # (ratio 0.9644 on the notebook's illustrative galaxy) is the DEFAULT
    # disagreeing from the masked pipeline value, not a coincidence of this
    # particular SSP. If this assertion ever goes vacuous (unmasked == masked),
    # the SSP no longer has flux below 912 A and this regression stops testing
    # anything -- revisit the fixture rather than deleting the check.
    l_abs_unmasked_ergs = (
        float(
            compute_l_dust_absorbed(
                components["sed_intrinsic"],
                components["sed_attenuated"],
                wave,
                include_lyc=True,
            )
        )
        * L_SUN
    )
    assert l_abs_unmasked_ergs != pytest.approx(l_abs_via_energy_balance_ergs, rel=1e-6), (
        f"[{dust_emission_type}] unmasked ({l_abs_unmasked_ergs:.6e}) == masked "
        f"({l_abs_via_energy_balance_ergs:.6e}): this fixture no longer has flux "
        "below the Lyman limit, so it cannot exercise the historical bug."
    )


@pytest.mark.parametrize("dust_emission_type", ["schreiber2018", "dl07"])
def test_agnfitter_priors_energy_balance_is_not_hard_rejected(
    synthetic_ssp_wide, dust_emission_type
):
    """``energy_balance`` (flexible mode) is finite AND not the reject sentinel.

    Before the fix, ``l_gal_att`` (via the unmasked ``compute_l_dust_absorbed``)
    structurally exceeded ``l_sb_emit`` (the masked ``L_ir``'s bolometric
    integral) by the LyC fraction, so ``prior_energy_balance`` returned
    ``AGNFITTER_HARD_REJECT`` (-9999, itself finite) for every Calzetti-attenuated
    star-forming galaxy -- ``jnp.isfinite`` alone does not catch this, which is
    why the existing adapter contract tests stayed green through the bug.
    """
    model = _build(synthetic_ssp_wide, dust_emission_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = {**model.spec.get_fixed_values()}
        pred = model.predict(params)

    _total, breakdown = agnfitter_priors(
        pred,
        redshift=0.5,
        dlum=1.0e28,
        data_flux_1500=1.0e-28,
        enable_energy_balance=True,
        enable_agn_fraction=False,
    )
    energy_balance = float(breakdown["energy_balance"])
    assert jnp.isfinite(energy_balance)
    assert energy_balance != AGNFITTER_HARD_REJECT, (
        f"[{dust_emission_type}] energy_balance == AGNFITTER_HARD_REJECT "
        "({AGNFITTER_HARD_REJECT}): l_gal_att still exceeds l_sb_emit, so the "
        "adapter is still comparing masked to unmasked L_absorbed."
    )
    # Flexible mode: 0.0 exactly whenever l_sb_emit >= l_gal_att (eta=1.0 keeps
    # L_ir == L_absorbed on the attenuator, and every dust-emission engine used
    # here is proportional to L_ir -- test_dust_emission_l_ir_linearity.py pins
    # that -- so the emitted bolometric integral reproduces L_absorbed exactly).
    assert energy_balance == pytest.approx(0.0, abs=1e-6)

    components = pred.sed.components
    wave = components["wavelength"]
    l_absorbed = (
        float(
            compute_l_dust_absorbed(
                components["sed_intrinsic"], components["sed_attenuated"], wave
            )
        )
        * L_SUN
    )
    from tengri.parameters.agn_priors import _integrate_l_nu

    l_dust = float(_integrate_l_nu(wave, components["sed_dust_ir"]))
    ratio = l_dust / l_absorbed
    print(f"[{dust_emission_type}] L_dust/L_abs = {ratio:.6f}")
    assert ratio == pytest.approx(1.0, rel=1e-4)
