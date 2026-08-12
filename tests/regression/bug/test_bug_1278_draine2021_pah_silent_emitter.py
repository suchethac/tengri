# SPDX-License-Identifier: BSD-3-Clause
"""Draine2021PAH dust emission is silent when precomputation is off (#1278).

Related to epic #1738: PAH emission precomputation.

Bug: `Draine2021PAHIRSEDComponent.predict()` declares outputs
`{"L_ir_emission": "erg/s"}` but returns an empty dict `{}` in all code paths.
The component also returns without warning when templates are unavailable
(the default for a cold build), making it silently inert on default builds.

Expected: The component should (1) publish `L_ir_emission` containing the
re-radiated luminosity when templates load, and (2) warn or return early
when data is missing. On a default build with no `approx=` precomputation,
dust emission components should re-emit the absorbed `L_ir`.

This test establishes the baseline: on a default build, the PAH component
does NOT publish its declared output, while sibling models (dale2014,
pah_drude) DO (control tests to verify the harness is sound).
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def test_draine2021_pah_publishes_its_declared_output(synthetic_ssp_wide, synthetic_tophat_obs):
    """On a default build (no approx=), the derived dict contains L_ir_emission.

    The component declares outputs={'L_ir_emission': 'erg/s'}, so predictions
    should publish a derived quantity matching that name. **This fails now** (#1278).
    """
    from tengri import SEDModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust={
                "law_bc": "calzetti",
                "tau_bc": 1.0,
                "tau_diff": 1.0,
                "emission": {"type": "draine2021_pah"},
            },
            sfh={"type": "dpl"},
        )

    params = model.spec.sample(jax.random.PRNGKey(0))
    state = model.predict_state(params)

    # The Draine2021PAH component declares outputs={'L_ir_emission': ...},
    # so it MUST publish that key when it successfully runs. This is the contract.
    assert hasattr(state.derived, "L_ir_emission"), (
        "Draine2021PAH declared L_ir_emission in outputs but it is not published "
        "in derived state. Expected state.derived.L_ir_emission (#1278)."
    )


def test_draine2021_pah_contributes_infrared_emission(synthetic_ssp_wide, synthetic_tophat_obs):
    """On a default build, sed_dust_ir is published and nonzero.

    When dust absorbs stellar photons, it must re-emit in the IR. A zero
    sed_dust_ir when dust has absorbed energy (L_ir > 0) means the component
    is silently inert (#1278). **This fails now**.
    """
    from tengri import SEDModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust={
                "law_bc": "calzetti",
                "tau_bc": 1.0,
                "tau_diff": 1.0,
                "emission": {"type": "draine2021_pah"},
            },
            sfh={"type": "dpl"},
        )

    params = model.spec.sample(jax.random.PRNGKey(0))
    state = model.predict_state(params)

    # The state.derived.sed_dust_ir must be published and nonzero when dust
    # absorbs energy (L_ir > 0).
    assert hasattr(state.derived, "sed_dust_ir"), (
        "sed_dust_ir not found in predict_state. "
        "Expected dust emission to contribute to the SED (#1278)."
    )

    sed_dust_ir = state.derived.sed_dust_ir
    assert sed_dust_ir is not None, (
        "sed_dust_ir is None (not published by component). "
        "Draine2021PAH should emit IR but has no output (#1278)."
    )

    sed_dust_ir_sum = jnp.sum(jnp.abs(sed_dust_ir))

    assert sed_dust_ir_sum > 0.0, (
        f"sed_dust_ir sums to zero (sum={float(sed_dust_ir_sum):.2e}). "
        "Draine2021PAH is not contributing IR emission even though dust absorbed energy. "
        "Component is silent (#1278)."
    )


@pytest.mark.parametrize("model_name", ["dale2014", "pah_drude"])
def test_sibling_emission_models_publish(synthetic_ssp_wide, synthetic_tophat_obs, model_name):
    """Sibling emission models (dale2014, pah_drude) DO publish on default builds.

    Control test: verify the harness is sound by checking that comparable
    components work. If dale2014 and pah_drude both publish but draine2021_pah
    does not, the failure is specific to Draine2021PAH, not a harness issue.
    """
    from tengri import SEDModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust={
                "law_bc": "calzetti",
                "tau_bc": 1.0,
                "tau_diff": 1.0,
                "emission": {"type": model_name},
            },
            sfh={"type": "dpl"},
        )

    params = model.spec.sample(jax.random.PRNGKey(0))
    state = model.predict_state(params)

    # Both dale2014 and pah_drude should publish sed_dust_ir
    assert hasattr(state.derived, "sed_dust_ir"), (
        f"{model_name} did not publish sed_dust_ir in predict_state. "
        f"Expected dust emission to contribute to the SED."
    )

    sed_dust_ir = jnp.asarray(state.derived.sed_dust_ir)
    sed_dust_ir_sum = jnp.sum(jnp.abs(sed_dust_ir))

    assert sed_dust_ir_sum > 0.0, (
        f"{model_name}: sed_dust_ir sums to zero (sum={float(sed_dust_ir_sum):.2e}). "
        f"Expected IR emission to contribute. Control test is broken."
    )
