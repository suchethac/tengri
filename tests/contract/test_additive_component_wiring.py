# SPDX-License-Identifier: BSD-3-Clause
"""Contract: additive components are wired for inference (no silent no-op).

Radio, X-ray, and shock each expose parameters that must actually move the
predicted SED with a *non-zero gradient*; IGM exposes an ``apply_igm`` gate that
must attenuate the rest-UV at high redshift. A parameter can be declared free,
appear in :attr:`spec.free_params`, and still never touch ``predict`` -- a
recurring footgun (a fittable parameter needs the prior, the param-map entry,
the forward-pass consumption, *and* the gradient path; miss one and it is a
silent no-op). These contracts build on the synthetic wide SSP so they run on
CI without the gitignored ``data/`` grids.

Shock uses the hardcoded Allen+2008 fallback when the MAPPINGS grid is absent
(CI-safe). The X-ray corona tilt (``xray_delta_alpha_ox``) only acts when an
AGN corona is present -- it is *correctly* inert without one, so that case is
tested with an AGN disc in the model.

See #926.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, FREE, Fixed, SEDModel, Uniform, builders

pytestmark = pytest.mark.contract


def _sed_response_grad(model, params, param_name):
    """Return (is_finite, is_nonzero, grad) of the total rest-SED w.r.t. a param.

    ``sum(predict_rest_sed.sed)`` is a scalar that reflects *any* change to the
    SED anywhere on the grid, so a non-zero gradient proves the parameter is
    consumed by the forward pass and differentiable -- i.e. fittable.
    """

    def objective(value):
        p = {**params, param_name: value}
        return jnp.sum(model.predict_rest_sed(p).sed)

    grad = float(jax.grad(objective)(jnp.asarray(float(params[param_name]))))
    return np.isfinite(grad), abs(grad) > 0.0, grad


def _base_kwargs():
    return dict(
        sfh={"type": "dpl", "*": FREE},
        dust={"type": "two_component", "law": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


def test_radio_q_ir_is_wired(synthetic_ssp_wide, synthetic_tophat_obs):
    """radio ``q_ir`` moves the SED with a non-zero gradient."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        radio={"type": "condon92", "q_ir": Uniform(2.0, 3.0)},
        **_base_kwargs(),
    )
    assert "radio_q_ir" in model.spec.free_params
    params = model.spec.sample(jax.random.PRNGKey(1))
    finite, nonzero, grad = _sed_response_grad(model, params, "radio_q_ir")
    assert finite, f"radio_q_ir gradient not finite: {grad}"
    assert nonzero, f"radio_q_ir is a silent no-op (zero gradient): {grad}"


def test_xray_delta_alpha_ox_is_wired_with_agn(synthetic_ssp_wide, synthetic_tophat_obs):
    """X-ray ``delta_alpha_ox`` moves the SED when an AGN corona is present."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        agn={"type": "composable", "disc": builders.agn.disc.multicolor(defaults=FREE)},
        xray={"type": "simple", "delta_alpha_ox": Uniform(-2.0, -1.0)},
        **_base_kwargs(),
    )
    assert "xray_delta_alpha_ox" in model.spec.free_params
    params = model.spec.sample(jax.random.PRNGKey(1))
    if "agn_log_lbol" in params:
        params["agn_log_lbol"] = jnp.asarray(12.0)  # a bright, unambiguous corona
    finite, nonzero, grad = _sed_response_grad(model, params, "xray_delta_alpha_ox")
    assert finite, f"xray_delta_alpha_ox gradient not finite: {grad}"
    assert nonzero, f"xray_delta_alpha_ox is a silent no-op with an AGN present: {grad}"


def test_shock_frac_is_wired(synthetic_ssp_wide, synthetic_tophat_obs):
    """shock ``frac`` scales the shock SED with a non-zero gradient."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        shock={"type": "mappings", "frac": Uniform(0.0, 1.0)},
        **_base_kwargs(),
    )
    assert "shock_frac" in model.spec.free_params
    params = model.spec.sample(jax.random.PRNGKey(1))
    params["shock_frac"] = jnp.asarray(0.5)  # interior of (0, 1) for a clean gradient
    finite, nonzero, grad = _sed_response_grad(model, params, "shock_frac")
    assert finite, f"shock_frac gradient not finite: {grad}"
    assert nonzero, f"shock_frac is a silent no-op (zero gradient): {grad}"


def test_igm_gate_attenuates_rest_uv_at_high_z(synthetic_ssp_wide, synthetic_tophat_obs):
    """``apply_igm`` suppresses the observed blue band at high redshift.

    IGM exposes no free parameter -- it is a redshift-dependent transmission
    applied in the *observed* frame -- so the wiring check is that toggling the
    gate changes the bluest broadband flux at a redshift where the IGM bites.
    At z=3 the 3500 A observed band samples rest ~875 A (below the Lyman limit),
    which the IGM must almost entirely remove.
    """
    common = dict(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "*": FREE},
        dust={"type": "two_component", "law": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(3.0),
    )
    m_on = SEDModel.build(igm={"type": "inoue"}, **common)
    m_off = SEDModel.build(igm={"type": "none"}, **common)
    params = m_on.spec.sample(jax.random.PRNGKey(1))
    sed_on = m_on.predict_obs_sed(params)
    sed_off = m_off.predict_obs_sed(params)
    # Observed wavelengths sampling rest < 912 A (the Lyman limit) at z=3 -> the
    # IGM Lyman-continuum absorption must remove most of the flux there.
    lyc = np.asarray(sed_on.wavelength) < 912.0 * (1.0 + 3.0)
    flux_on = float(jnp.sum(sed_on.sed[lyc]))
    flux_off = float(jnp.sum(sed_off.sed[lyc]))
    assert flux_off > 0.0, "control (igm off) has no rest-Lyman flux to attenuate"
    assert flux_on < 0.5 * flux_off, (
        f"IGM gate did not attenuate the rest-Lyman continuum at z=3: "
        f"on={flux_on:.3e} off={flux_off:.3e}"
    )
