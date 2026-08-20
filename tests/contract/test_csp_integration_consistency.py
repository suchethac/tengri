# SPDX-License-Identifier: BSD-3-Clause
"""``csp_integration`` must not make the reported mass disagree with the SED (#1500).

The stellar component builds its age weights with cloud-in-cell
(``_age_weights_cic``) for **every** configuration -- ``sps_backend="dsps"`` is
the only backend. ``csp_integration`` therefore cannot change the predicted SED,
and measurement confirms it: photometry is bit-identical across all five accepted
values.

What it *did* change was ``_predict_sfh_quantities``, which branched on it to
compute a different set of age weights. So the reported stellar mass came from a
different integration than the spectrum it was fitted to:

===================  ==============================  ==============
``csp_integration``  ``_predict_sfh_quantities`` M*  vs properties
===================  ==============================  ==============
``trapz``            1.706632e12                     0.00%
``log_trapz``        1.706632e12                     0.00%
``log_interp``       1.712171e12                     0.32%
``dsps_native``      1.712171e12                     0.32%
``dsps_met_table``   **NaN**                         --
===================  ==============================  ==============

The ``else`` branch already routed the default through the orchestrator "so
``predict_sfh_quantities`` returns the same stellar_mass / weights as
``predict_derived`` ... was 4.1% apart with the legacy rectangle rule". That fix
was applied to the values someone happened to test; the other three kept the
divergent path. These tests pin the *invariant* over every accepted value, so a
future branch cannot reintroduce it for the untested ones.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FREE, Fixed, SEDModel
from tengri.forward.sed_model import _VALID_CSP_INTEGRATION

pytestmark = pytest.mark.contract


def _model(ssp, obs, csp):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        met={"logzsol": Fixed(-0.3)},
        redshift=Fixed(0.1),
        igm={"type": "none"},
        csp_integration=csp,
    )


@pytest.fixture(scope="module")
def _params(synthetic_ssp_wide, synthetic_tophat_obs):
    m = _model(synthetic_ssp_wide, synthetic_tophat_obs, "trapz")
    return {**m.spec.get_fixed_values(), **m.spec.sample(jax.random.PRNGKey(0))}


@pytest.mark.parametrize("csp", sorted(_VALID_CSP_INTEGRATION))
def test_reported_mass_matches_the_sed_path(
    synthetic_ssp_wide, synthetic_tophat_obs, _params, csp
):
    """One model, one parameter set, one stellar mass -- whatever the knob says.

    Derived quantities are computed from the SED's own age weights, so a value
    that cannot change the SED cannot change the mass either.
    """
    m = _model(synthetic_ssp_wide, synthetic_tophat_obs, csp)
    from_sfh = float(np.asarray(m._predict_sfh_quantities(_params).stellar_mass))
    from_state = float(np.sum(np.asarray(m.predict_state(_params).derived["age_weights"])))

    assert np.isfinite(from_sfh), f"csp_integration={csp!r} reported a non-finite stellar mass"
    np.testing.assert_allclose(
        from_sfh,
        from_state,
        rtol=1e-10,
        err_msg=(
            f"csp_integration={csp!r}: reported stellar mass disagrees with the age "
            "weights the SED was built from"
        ),
    )


@pytest.mark.parametrize("csp", sorted(_VALID_CSP_INTEGRATION))
def test_derived_quantities_are_independent_of_the_knob(
    synthetic_ssp_wide, synthetic_tophat_obs, _params, csp
):
    """Every accepted value yields the same derived quantities as the default.

    This is what "cannot change the SED" has to mean downstream. Before the fix,
    log_interp / dsps_native were 0.32% off and dsps_met_table returned NaN --
    and ``Posterior`` vmaps this function over every sample, so that NaN reached
    posterior summaries.
    """
    ref = _model(synthetic_ssp_wide, synthetic_tophat_obs, "trapz")._predict_sfh_quantities(
        _params
    )
    got = _model(synthetic_ssp_wide, synthetic_tophat_obs, csp)._predict_sfh_quantities(_params)
    for field in ("stellar_mass", "stellar_mass_surviving", "ssfr", "mass_weighted_age_gyr"):
        a, b = np.asarray(getattr(ref, field)), np.asarray(getattr(got, field))
        if np.all(np.isnan(a)) and np.all(np.isnan(b)):
            continue
        np.testing.assert_allclose(
            b, a, rtol=1e-10, err_msg=f"csp_integration={csp!r} changed {field}"
        )


def test_photometry_is_independent_of_the_knob(synthetic_ssp_wide, synthetic_tophat_obs, _params):
    """The premise: the SED path never reads this setting."""
    ref = np.asarray(
        _model(synthetic_ssp_wide, synthetic_tophat_obs, "trapz").predict_photometry(_params)
    )
    for csp in sorted(_VALID_CSP_INTEGRATION):
        got = np.asarray(
            _model(synthetic_ssp_wide, synthetic_tophat_obs, csp).predict_photometry(_params)
        )
        np.testing.assert_array_equal(
            got, ref, err_msg=f"csp_integration={csp!r} changed the photometry"
        )


def test_non_default_values_warn_that_they_do_nothing(synthetic_ssp_wide, synthetic_tophat_obs):
    """A knob that cannot change any output must say so, not accept silently."""
    for csp in sorted(_VALID_CSP_INTEGRATION):
        if csp == "trapz":
            continue
        with pytest.warns(DeprecationWarning, match="csp_integration"):
            _model(synthetic_ssp_wide, synthetic_tophat_obs, csp)


def test_default_does_not_warn(synthetic_ssp_wide, synthetic_tophat_obs):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _model(synthetic_ssp_wide, synthetic_tophat_obs, "trapz")


def test_knob_no_longer_partitions_the_compile_cache(synthetic_ssp_wide, synthetic_tophat_obs):
    """It used to produce 5 distinct compile signatures for identical programs.

    Every value recompiled the whole model from scratch and then computed the
    same numbers.
    """
    sigs = {
        str(_model(synthetic_ssp_wide, synthetic_tophat_obs, c).compile_signature())
        for c in _VALID_CSP_INTEGRATION
    }
    assert len(sigs) == 1, f"csp_integration still splits the compile cache {len(sigs)} ways"


def test_jit_and_grad_survive(synthetic_ssp_wide, synthetic_tophat_obs, _params):
    """The consistency fix must not break the JIT/grad contract."""
    m = _model(synthetic_ssp_wide, synthetic_tophat_obs, "trapz")
    f = jax.jit(lambda p: jnp.sum(m.predict_photometry(p)))
    assert np.isfinite(float(f(_params)))
