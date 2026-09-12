# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for the ``agnfitter_priors`` public adapter.

Surface protected: ``tengri.parameters.agn_priors.agnfitter_priors`` (also
reachable as ``tengri.agn.priors.agnfitter_priors``), the task-7 brief's
change 2 (adapters) as scoped by review round 1 item 3: an adapter built on
the PUBLIC prediction surface (``model.predict(params).sed.components``,
``sed_intrinsic``/``sed_attenuated``/``sed_dust_ir``/``sed_agn``/``sed_xray``/
``sed_radio`` plus the wavelength axis), covering all eight priors via
explicit ``enable_*`` settings and (for the priors that compare the model to
actual data, not just to itself) explicit data keyword arguments.

Not this file's job: the physics-correctness of any individual prior formula
(``tests/crossval/test_agn_priors_vs_agnfitter.py``,
``tests/contract/test_agn_priors.py``) or the JIT-safe fitting hook
(``tests/contract/test_fitter_extra_log_prior.py``). This file only checks
that the adapter correctly wires ``pred.sed.components`` into the prior
functions: finiteness, that toggling each setting changes the total, and
that the per-prior breakdown equals calling the corresponding prior
function directly with the SAME inputs the adapter derived.

Uses the session-scoped ``synthetic_ssp_wide`` fixture (smooth UV-to-far-IR
continuum, no ``data/ssp_*.h5`` needed) so this runs in the default fast
tier.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
from tengri.observation.photometry import FilterCurve
from tengri.parameters.agn_priors import (
    agnfitter_priors,
    prior_agn_fraction,
    prior_energy_balance,
    prior_ir_syn_fraction,
    prior_ir_xrays,
    prior_low_agn_fraction,
    prior_midir_uv,
    prior_uv_xrays,
)
from tengri.utils.physics_constants import L_SUN
from tengri.utils.sed_quantities import compute_bolometric_luminosity, compute_l_dust_absorbed

pytestmark = pytest.mark.contract

_DLUM = 1.0e28  # cm, arbitrary (adapter self-consistency, not a cosmology test)
_REDSHIFT = 0.5


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


_PHOT = Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))


def _build_model(ssp):
    obs = Observation(photometry=_PHOT)
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        radio={
            "sf": {"type": "bell2003", "all_params": Fixed(DEFAULT)},
            "agn": {"type": "dpl", "all_params": Fixed(DEFAULT)},
        },
        agn={
            "type": "composable",
            "norm": "independent",
            "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
            "nlr": {"type": "none"},
            "blr": {"type": "none"},
            "feii": {"type": "none"},
            "atten": {"type": "none"},
        },
        redshift=Fixed(_REDSHIFT),
    )
    return model, obs


@pytest.fixture(scope="module")
def _pred(synthetic_ssp_wide):
    model, _obs = _build_model(synthetic_ssp_wide)
    assert model.spec.free_params == []  # every parameter Fixed -> one prediction to test
    params = {k: v[0] for k, v in model.spec.sample_batch(jax.random.PRNGKey(0), 1).items()}
    return model.predict(params)


def test_agnfitter_priors_is_finite(_pred):
    total, breakdown = agnfitter_priors(
        _pred, redshift=_REDSHIFT, dlum=_DLUM, data_flux_1500=1.0e-28
    )
    assert jnp.isfinite(total)
    assert breakdown  # default settings enable energy_balance + agn_fraction
    for name, value in breakdown.items():
        assert jnp.isfinite(value), f"{name} is not finite: {value}"


def test_default_breakdown_keys_match_default_settings(_pred):
    """Default enable_* mirror AGNFITTER_PRIOR_DEFAULTS (energy_balance + agn_fraction on)."""
    _total, breakdown = agnfitter_priors(
        _pred, redshift=_REDSHIFT, dlum=_DLUM, data_flux_1500=1.0e-28
    )
    assert set(breakdown.keys()) == {"energy_balance", "agn_fraction"}


@pytest.mark.parametrize(
    "flag, extra_kwargs",
    [
        # "restrictive" mode: the fixture's dust attenuation/emission
        # are exactly energy-balanced (eta=1.0, LyC-masked L_absorbed ==
        # L_ir), so "flexible" mode's floor legitimately returns 0.0 -- same
        # as the disabled baseline, so it cannot demonstrate the toggle
        # changed anything. "restrictive" adds a Gaussian log-density term
        # even at perfect balance (nonzero at its own mean), so it reliably
        # differs from the 0.0 baseline regardless of how well-balanced the
        # fixture is.
        ("enable_energy_balance", {"energy_balance_mode": "restrictive"}),
        ("enable_agn_fraction", {}),
        ("enable_low_agn_fraction", {}),
        ("enable_midir_uv", {}),
        ("enable_uv_xrays", {"log_l2kev_data": 28.0}),
        ("enable_ir_xrays", {"log_f2_10kev_data": 23.0}),
        (
            "enable_ir_syn_fraction",
            {
                "data_flux_rad": 1.0e-27,
                "data_nu_rad": 9.0,
                "data_flux_ir": 1.0e-27,
                "data_nu_ir": 12.5,
            },
        ),
    ],
)
def test_toggling_each_setting_changes_the_total(_pred, flag, extra_kwargs):
    base_kwargs = dict(
        redshift=_REDSHIFT,
        dlum=_DLUM,
        data_flux_1500=1.0e-28,
        enable_energy_balance=False,
        enable_agn_fraction=False,
    )
    baseline, _ = agnfitter_priors(_pred, **base_kwargs)

    kwargs = dict(base_kwargs)
    kwargs[flag] = True
    kwargs.update(extra_kwargs)
    total, breakdown = agnfitter_priors(_pred, **kwargs)

    assert jnp.isfinite(total)
    assert total != pytest.approx(float(baseline), abs=1e-9), f"{flag} did not change the total"
    assert flag.removeprefix("enable_") in breakdown


def test_per_prior_values_match_direct_prior_calls(_pred):
    """The breakdown must equal calling each prior function directly with
    the SAME inputs the adapter derives from pred.sed.components -- a
    wiring check, not a re-derivation of the prior math."""
    total, breakdown = agnfitter_priors(
        _pred,
        redshift=_REDSHIFT,
        dlum=_DLUM,
        data_flux_1500=1.0e-28,
        log_l2kev_data=28.0,
        log_f2_10kev_data=23.0,
        enable_energy_balance=True,
        enable_agn_fraction=True,
        enable_low_agn_fraction=True,
        enable_midir_uv=True,
        enable_uv_xrays=True,
        enable_ir_xrays=True,
    )
    components = _pred.sed.components
    wave = components["wavelength"]

    l_absorbed = (
        compute_l_dust_absorbed(components["sed_intrinsic"], components["sed_attenuated"], wave)
        * L_SUN
    )
    l_sb_emit = compute_bolometric_luminosity(components["sed_dust_ir"], wave) * L_SUN
    expected_energy_balance = prior_energy_balance(l_absorbed, l_sb_emit, mode="flexible")
    assert float(breakdown["energy_balance"]) == pytest.approx(
        float(expected_energy_balance), rel=1e-9, abs=1e-9
    )

    bbb_flux_1500 = jnp.interp(1500.0, wave, components["sed_agn"])
    gal_flux_1500 = jnp.interp(1500.0, wave, components["sed_attenuated"])
    expected_agn_fraction = prior_agn_fraction(
        bbb_flux_1500, gal_flux_1500, 1.0e-28, _DLUM, _REDSHIFT
    )
    assert float(breakdown["agn_fraction"]) == pytest.approx(
        float(expected_agn_fraction), rel=1e-9, abs=1e-9
    )
    expected_low_agn_fraction = prior_low_agn_fraction(
        bbb_flux_1500, gal_flux_1500, 1.0e-28, _DLUM, _REDSHIFT
    )
    assert float(breakdown["low_agn_fraction"]) == pytest.approx(
        float(expected_low_agn_fraction), rel=1e-9, abs=1e-9
    )

    from tengri.utils.physics_constants import C_AA

    l_nu_6um = jnp.interp(60000.0, wave, components["sed_agn"])
    nulnu_6um = (C_AA / 60000.0) * l_nu_6um
    log_l2500a_bbmodel = jnp.log10(jnp.interp(2500.0, wave, components["sed_agn"]))
    expected_midir_uv = prior_midir_uv(log_l2500a_bbmodel, nulnu_6um)
    assert float(breakdown["midir_uv"]) == pytest.approx(
        float(expected_midir_uv), rel=1e-9, abs=1e-9
    )

    log_l2500a_data = jnp.log10(jnp.interp(2500.0, wave, components["sed_agn"]))
    expected_uv_xrays = prior_uv_xrays(log_l2500a_data, 28.0)
    assert float(breakdown["uv_xrays"]) == pytest.approx(
        float(expected_uv_xrays), rel=1e-9, abs=1e-9
    )

    expected_ir_xrays = prior_ir_xrays(23.0, nulnu_6um)
    assert float(breakdown["ir_xrays"]) == pytest.approx(
        float(expected_ir_xrays), rel=1e-9, abs=1e-9
    )

    assert float(total) == pytest.approx(
        sum(float(v) for v in breakdown.values()), rel=1e-9, abs=1e-9
    )


def test_ir_syn_fraction_matches_direct_call(_pred):
    from tengri.utils.physics_constants import C_AA

    total, breakdown = agnfitter_priors(
        _pred,
        redshift=_REDSHIFT,
        dlum=_DLUM,
        data_flux_1500=1.0e-28,
        enable_energy_balance=False,
        enable_agn_fraction=False,
        enable_ir_syn_fraction=True,
        data_flux_rad=1.0e-27,
        data_nu_rad=9.0,
        data_flux_ir=1.0e-27,
        data_nu_ir=12.5,
    )
    components = _pred.sed.components
    wave = components["wavelength"]
    wave_ir_aa = C_AA / (10.0**12.5)
    sb_flux_ir = jnp.interp(wave_ir_aa, wave, components["sed_dust_ir"])
    syn_flux_ir = jnp.interp(wave_ir_aa, wave, components["sed_radio"])
    expected = prior_ir_syn_fraction(1.0e-27, 9.0, 1.0e-27, 12.5, sb_flux_ir, syn_flux_ir)
    assert float(breakdown["ir_syn_fraction"]) == pytest.approx(
        float(expected), rel=1e-9, abs=1e-9
    )
    assert float(total) == pytest.approx(float(expected), rel=1e-9, abs=1e-9)


def test_missing_required_data_raises_value_error(_pred):
    """Default settings enable_agn_fraction=True requires data_flux_1500."""
    with pytest.raises(ValueError, match="data_flux_1500"):
        agnfitter_priors(_pred, redshift=_REDSHIFT, dlum=_DLUM)


def test_stellar_mass_requires_ga(_pred):
    with pytest.raises(ValueError, match="ga="):
        agnfitter_priors(
            _pred,
            redshift=_REDSHIFT,
            dlum=_DLUM,
            data_flux_1500=1.0e-28,
            enable_stellar_mass=True,
        )
