# SPDX-License-Identifier: BSD-3-Clause
r"""``stellar_mass_scale`` must have a float32-safe log form (#1206).

``stellar_mass_scale = total_mass x L_sun`` [erg/s/Hz per unit SSP weight] is
~3.8e43 for a 1e10 Msun galaxy — five decades past the float32 ceiling of
3.4e38 — so it is ``inf`` in a pure-float32 forward pass.

Nothing about the SSP grid rescues this. The scale is ``total_mass`` times a
constant, with no SSP flux factor to keep it small, so it overflows for any
galaxy above ``3.4e38 / 3.828e33 ~ 9e4`` Msun. That is every galaxy anyone
fits.

The float32-safe contract is therefore ``log_stellar_mass_scale``, published
beside the linear key and exactly its base-10 logarithm.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri import FIXED, Fixed, SEDModel

pytestmark = pytest.mark.regression_bug

# float32 ceiling; the scale passes it for any galaxy above ~9e4 Msun.
_F32_MAX = 3.4028235e38


def _model(ssp, log_mass=10.0):
    """Minimal model at a given formed mass.

    The dust bounds are pinned explicitly: the default dust group leaves
    ``tau_bc``/``tau_diff`` free, so an empty ``predict_state({})`` would raise
    rather than exercise the scale.
    """
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(log_mass),
            "all_params": FIXED,
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": FIXED,
        },
        redshift=Fixed(0.0),
    )


def test_log_stellar_mass_scale_is_published(synthetic_ssp_wide):
    """The key exists and is exactly log10 of the linear one."""
    derived = _model(synthetic_ssp_wide).predict_state({}).derived
    assert "log_stellar_mass_scale" in derived, (
        "stellar_mass_scale has no float32-safe log form; it is ~1e43 and "
        "therefore inf in pure float32"
    )
    linear = np.float64(np.asarray(derived["stellar_mass_scale"]))
    log_form = np.float64(np.asarray(derived["log_stellar_mass_scale"]))
    assert_allclose(log_form, np.log10(linear), rtol=1e-12)


@pytest.mark.parametrize("log_mass", [7.0, 9.0, 10.0, 11.5])
def test_log_stellar_mass_scale_tracks_mass(synthetic_ssp_wide, log_mass):
    """The log scale moves dex-for-dex with the formed mass, at every mass."""
    derived = _model(synthetic_ssp_wide, log_mass).predict_state({}).derived
    linear = np.float64(np.asarray(derived["stellar_mass_scale"]))
    log_form = np.float64(np.asarray(derived["log_stellar_mass_scale"]))
    assert_allclose(log_form, np.log10(linear), rtol=1e-12)
    # log10(total_mass * L_sun) = log_total_mass + log10(L_sun) exactly.
    from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S

    assert_allclose(log_form, log_mass + np.log10(LSUN_ERG_PER_S), rtol=1e-9)


def test_log_stellar_mass_scale_finite_in_pure_float32(synthetic_ssp_wide):
    """Pure float32: the log form survives where the linear one overflows.

    The load-bearing assertion. The linear key must be proven to overflow
    here, otherwise this test would pass for the wrong reason.
    """
    ref = np.float64(
        np.asarray(_model(synthetic_ssp_wide).predict_state({}).derived["log_stellar_mass_scale"])
    )
    assert np.isfinite(ref)
    assert 10.0**ref > _F32_MAX, (
        f"setup: the linear scale 1e{ref:.2f} must exceed the float32 ceiling "
        "for this test to mean anything"
    )

    with jax.enable_x64(False):
        derived = _model(synthetic_ssp_wide).predict_state({}).derived
        log_form = np.asarray(derived["log_stellar_mass_scale"])
        linear = np.asarray(derived["stellar_mass_scale"])
        assert log_form.dtype == jnp.float32  # precondition: genuinely pure float32

    assert np.isfinite(log_form), f"log scale non-finite in float32: {log_form}"
    assert_allclose(float(log_form), ref, atol=1e-5)
    # The linear key overflowing is the whole reason the log key exists.
    assert not np.isfinite(linear), (
        f"expected the linear scale to overflow in float32, got {linear} — "
        "if this ever becomes finite the log form is no longer load-bearing"
    )


def test_dust_energy_balance_uses_the_published_log_scale(synthetic_ssp_wide):
    """The dust energy balance must read the published key, not re-derive it.

    ``two_component`` carried a ``jnp.log10(mass_scale)`` fallback while the
    producer had not published yet. Once it does, the fallback is dead code —
    and a fallback that silently recomputes an ``inf`` input is exactly the
    failure this task removes.
    """
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        met={"logzsol": Fixed(0.0), "all_params": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": FIXED,
        },
        dust_attenuation={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(1.0),
            "tau_diff": Fixed(0.7),
            "all_params": FIXED,
        },
        dust_emission={"type": "dale2014", "all_params": FIXED},
        redshift=Fixed(0.0),
    )
    derived = model.predict_state({}).derived
    assert "log_stellar_mass_scale" in derived
    # The energy balance still produces a finite, positive L_ir from it.
    assert np.isfinite(float(np.asarray(derived["log_L_ir"])))
    assert float(np.asarray(derived["L_ir"])) > 0.0
