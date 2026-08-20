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

from tengri import Fixed

pytestmark = pytest.mark.regression_bug


def _draine2021_available() -> bool:
    """Whether the published PAHspec grid is present on this machine.

    The 104 MB grid is not committed, so CI has no copy. Absent it the component
    warns and contributes nothing, which is the *designed* response and not the
    silent no-op #1278 is about — asserting emission there tests the data, not
    the fix. Mirrors the helper in tests/regression/test_dust_goldens_852.py.
    """
    from tengri.components.sed_model_component import _REGISTRY

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _REGISTRY["draine2021_pah_ir"]().load(jnp.logspace(3, 7, 64)) is not None
    except Exception:
        return False


#: Applied only to the assertions that need the grid. The two control tests
#: (dale2014, pah_drude) and the publish-contract test do not.
requires_pahspec = pytest.mark.skipif(
    not _draine2021_available(), reason="draine2021 PAHspec template grid not available"
)


@requires_pahspec
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
            dust_attenuation={
                "law": "calzetti",
                "tau_bc": 1.0,
                "tau_diff": 1.0,
            },
            dust_emission={"type": "draine2021_pah"},
            sfh={"type": "dpl"},
            redshift=Fixed(0.1),
        )

    params = model.spec.sample(jax.random.PRNGKey(0))
    state = model.predict_state(params)

    # The Draine2021PAH component declares outputs={'L_ir_emission': ...},
    # so it MUST publish that key when it successfully runs. This is the contract.
    #
    # NOT hasattr: the fix for #1278 promoted L_ir_emission to a typed field on
    # DerivedState, so the attribute exists on every prediction and defaults to
    # None. A hasattr assertion here passes whether or not the component
    # published anything -- it would have gone green against the very bug it
    # exists to pin. Assert the value.
    published = getattr(state.derived, "L_ir_emission", None)
    assert published is not None, (
        "Draine2021PAH declared L_ir_emission in outputs but it is not published "
        "in derived state. Expected state.derived.L_ir_emission (#1278)."
    )
    assert float(jnp.sum(jnp.abs(jnp.asarray(published)))) > 0.0, (
        "Draine2021PAH published L_ir_emission but it is zero — the component "
        "absorbed energy and re-radiated none of it (#1278)."
    )


@requires_pahspec
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
            dust_attenuation={
                "law": "calzetti",
                "tau_bc": 1.0,
                "tau_diff": 1.0,
            },
            dust_emission={"type": "draine2021_pah"},
            sfh={"type": "dpl"},
            redshift=Fixed(0.1),
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
            dust_attenuation={
                "law": "calzetti",
                "tau_bc": 1.0,
                "tau_diff": 1.0,
            },
            dust_emission={"type": model_name},
            sfh={"type": "dpl"},
            redshift=Fixed(0.1),
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
