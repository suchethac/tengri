# SPDX-License-Identifier: BSD-3-Clause
"""``log_L_ir`` did two jobs: the re-emitted IR budget AND the absorbed energy.

Three readers wanted the ABSORBED stellar+nebular energy specifically --
``pred.l_dust_absorbed`` (via ``component_factory.py`` and
``stellar/component.py``) and the AGN CIGALE fracAGN torus coupling
(``components/agn/component.py``) -- but read ``log_L_ir`` as a stand-in.
That is only correct when ``dust_eta_balance == 1`` (today's universal
default) and no total-dust-IR-budget override is declared. Under a relaxed
``dust_eta_balance`` the three readers were already silently wrong by the
eta factor; under a declared ``dust_log_L_ir`` override they would be wrong
by the full replacement offset.

Every dust-attenuation publisher (``component.py``, ``two_component.py``,
``wg00_model.py``) now publishes a companion ``log_L_absorbed`` /
``L_absorbed`` pair that never moves with ``dust_eta_balance`` or a declared
override, and the three readers above were repointed to it. This file pins
the invariant the split exists to establish: reading the absorbed energy
must not depend on ``dust_eta_balance``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed
from tengri.components.stellar.sps.dsps_wrapper import SSPData

pytestmark = pytest.mark.regression_bug


def _synthetic_ssp() -> SSPData:
    """Small synthetic SSP spanning UV-optical-FIR (mirrors conftest's ``synthetic_ssp_wide``)."""
    n_age = 25
    wave = jnp.logspace(2.0, 7.0, 1600)
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


def _build(ssp, *, eta=1.0, log_l_ir=None, tau_bc=1.0, tau_diff=0.7):
    emission = {"type": "dale2014", "all_params": Fixed(DEFAULT), "eta_balance": Fixed(eta)}
    if log_l_ir is not None:
        emission["log_L_ir"] = Fixed(log_l_ir)
    return tengri.SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(tau_bc),
            "tau_diff": Fixed(tau_diff),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission=emission,
        redshift=Fixed(0.0),
    )


@pytest.fixture(scope="module")
def ssp():
    return _synthetic_ssp()


def test_l_dust_absorbed_invariant_under_eta(ssp):
    """``pred.l_dust_absorbed`` must NOT scale with ``dust_eta_balance``.

    Before the split, ``component_factory._l_dust_absorbed_fn`` and
    ``stellar/component.py``'s twin read ``log_L_ir``, which is
    ``log_L_absorbed + log10(eta)``: doubling eta silently doubled the
    "absorbed" answer too, even though the absorbed energy (the UV/optical
    photons the dust screen actually intercepted) cannot depend on how much
    of it the emission side chooses to re-radiate.
    """
    m1 = _build(ssp, eta=1.0)
    m2 = _build(ssp, eta=2.0)
    l_abs_1 = float(np.asarray(m1.predict({}).l_dust_absorbed))
    l_abs_2 = float(np.asarray(m2.predict({}).l_dust_absorbed))
    assert l_abs_1 > 0.0, "setup: expected a positive absorbed luminosity"
    np.testing.assert_allclose(l_abs_2, l_abs_1, rtol=1e-10)


def test_l_dust_absorbed_invariant_under_override(ssp):
    """``pred.l_dust_absorbed`` must NOT move when ``dust_log_L_ir`` is declared.

    The override replaces the RE-EMITTED budget (``log_L_ir``), never the
    ABSORBED one (``log_L_absorbed``): the dust screen intercepts the same
    UV/optical photons regardless of what the emission side is told to
    re-radiate.
    """
    m_plain = _build(ssp, eta=1.0)
    m_override = _build(ssp, log_l_ir=11.0)
    l_abs_plain = float(np.asarray(m_plain.predict({}).l_dust_absorbed))
    l_abs_override = float(np.asarray(m_override.predict({}).l_dust_absorbed))
    np.testing.assert_allclose(l_abs_override, l_abs_plain, rtol=1e-10)


def test_agn_torus_coupling_reads_absorbed_not_ir(ssp):
    """The CIGALE fracAGN torus coupling must use the ABSORBED energy, not ``log_L_ir``.

    The coupling in ``components/agn/component.py`` forms
    ``log10(L_absorbed) + log10(frac/(1-frac)) - agn_log_lbol - log10(Lsun)``
    only when ``agn_ir_frac > 0`` (CIGALE's skirtor2016 ``agn_power``
    bookkeeping). Before the split it read ``log_L_ir``, so a nonzero
    ``agn_ir_frac`` coupling silently absorbed the ``dust_eta_balance``
    factor into the torus normalization. Measured through the public
    predict surface, on a build with ``agn_ir_frac`` free (mirroring
    ``recipes.composable_agn()``, committed-data only): the AGN torus SED
    must be identical at eta=1 and eta=2.
    """

    def _build_agn(eta):
        return tengri.SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "calzetti",
                "tau_bc": Fixed(1.0),
                "tau_diff": Fixed(0.7),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={
                "type": "dale2014_cigale",
                "all_params": Fixed(DEFAULT),
                "eta_balance": Fixed(eta),
            },
            neb={"type": "cue", "all_params": Fixed(DEFAULT)},
            agn={
                "type": "composable",
                "disc": {"type": "multicolor"},
                "torus": {"type": "skirtor"},
                "norm": "cigale_joint",
                # No agn_log_lbol: with agn_ir_frac active the AGN power is
                # L_absorbed * f/(1 - f) (the CIGALE skirtor2016 coupling), and
                # a user-given agn_log_lbol beside it is refused at build
                # (#2069/#2210) because it would be computed over and discarded.
                "ir_frac": Fixed(0.3),
                "other_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.0),
        )

    try:
        m1 = _build_agn(1.0)
        m2 = _build_agn(2.0)
        state1 = m1.predict_state({})
        state2 = m2.predict_state({})
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"AGN torus / nebular grid not on disk: {exc}")

    sed_agn_1 = np.asarray(state1.derived.get("sed_agn"))
    sed_agn_2 = np.asarray(state2.derived.get("sed_agn"))
    assert sed_agn_1 is not None and np.any(sed_agn_1 != 0.0), "setup: expected nonzero AGN SED"
    np.testing.assert_allclose(sed_agn_2, sed_agn_1, rtol=1e-8, atol=0.0)
