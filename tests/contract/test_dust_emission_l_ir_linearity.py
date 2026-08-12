# SPDX-License-Identifier: BSD-3-Clause
r"""Dust IR emission must be exactly linear in ``L_ir`` — or say so (#1206).

Every dust emission model normalizes a template shape to the absorbed
luminosity: ``sed = (L_ir / integral) * shape``. That makes ``sed_dust_ir``
exactly proportional to ``L_ir``, which is what lets the float32 work factor
``L_ir`` out of the template and re-apply it in log space — ``L_ir`` is ~2.4e43
and therefore ``inf`` in pure float32, while ``log_L_ir`` is finite.

The proportionality is an *assumption* about every registered emission model,
and it is not free: ``energy_balance_split`` computes

.. math:: L_{\rm IR}^{\rm tot} = \eta L_{\rm stellar} + L_{\rm AGN,IR}

which is **affine, not linear** — doubling :math:`\eta` does not double the
output once :math:`L_{\rm AGN,IR}` is comparable to the stellar term. It looks
linear at default settings only because ``dust_L_agn_ir`` defaults to 0.

So this file pins the invariant per model rather than assuming it globally. A
new emission model that is not proportional to ``L_ir`` fails here, loudly,
instead of silently returning wrong fluxes once the log-domain migration lands.

``dust_eta_balance`` multiplies ``L_ir`` directly (``L_ir = eta * L_absorbed``),
so it is the cleanest end-to-end handle on the scaling — it exercises the real
wiring rather than poking a component's internals.
"""

import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.builders.dust import emission as emission_builders

pytestmark = pytest.mark.contract

#: Models whose output is NOT proportional to L_ir, with the reason. The
#: log-domain migration must handle these explicitly (combining the two terms
#: with ``log10_add``) rather than factoring a single scale out.
AFFINE_MODELS = {
    "energy_balance_split": "L_ir_total = eta * L_stellar + L_agn_ir (additive AGN term)",
}


def _model(ssp, model_name, eta, **emission_extra):
    """Two-component dust with one emission model at a given eta_balance."""
    emission = {"type": model_name, "*": FIXED, "eta_balance": Fixed(eta)}
    emission.update(emission_extra)
    return SEDModel.build(
        ssp_data=ssp,
        stellar={"logzsol": Fixed(0.0), "*": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "*": FIXED,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(1.0),
            "tau_diff": Fixed(0.7),
            "*": FIXED,
            "emission": emission,
        },
        redshift=Fixed(0.0),
    )


def _sed_dust_ir(ssp, model_name, eta, **extra):
    derived = _model(ssp, model_name, eta, **extra).predict_state({}).derived
    if "sed_dust_ir" not in derived:
        pytest.skip(f"{model_name} publishes no sed_dust_ir (see #1278)")
    return np.asarray(derived["sed_dust_ir"], dtype=np.float64)


@pytest.mark.parametrize("model_name", sorted(emission_builders.available()))
def test_sed_dust_ir_is_proportional_to_l_ir(synthetic_ssp_wide, model_name):
    """Doubling ``L_ir`` must exactly double ``sed_dust_ir``.

    This is the invariant the float32 log-domain migration rests on. Models
    listed in :data:`AFFINE_MODELS` are expected to violate it and are checked
    for the opposite.
    """
    try:
        s1 = _sed_dust_ir(synthetic_ssp_wide, model_name, 1.0)
        s2 = _sed_dust_ir(synthetic_ssp_wide, model_name, 2.0)
    except pytest.skip.Exception:
        raise
    except (ValueError, KeyError) as exc:
        pytest.skip(f"{model_name} not constructible here: {type(exc).__name__}: {exc}")

    nonzero = np.abs(s1) > 0
    if not nonzero.any():
        pytest.skip(f"{model_name} emits nothing at eta_balance=1")

    ratio = s2[nonzero] / s1[nonzero]
    deviation = float(max(abs(ratio.min() - 2.0), abs(ratio.max() - 2.0)))

    if model_name in AFFINE_MODELS:
        pytest.skip(f"{model_name} is affine by construction: {AFFINE_MODELS[model_name]}")
    assert deviation < 1e-10, (
        f"{model_name} is not proportional to L_ir (ratio range "
        f"[{ratio.min()!r}, {ratio.max()!r}], deviation {deviation:.3e}). "
        "The float32 log-domain migration factors L_ir out of the template and "
        "re-applies it in log space, which is only valid for a proportional "
        "model. Either make it proportional or add it to AFFINE_MODELS and "
        "handle it explicitly."
    )


@pytest.mark.parametrize("model_name", sorted(AFFINE_MODELS))
def test_affine_models_really_are_affine(synthetic_ssp_wide, model_name):
    """The affine exemption must be earned, not asserted.

    An entry in :data:`AFFINE_MODELS` that has since become proportional would
    silently keep its exemption, so prove the non-proportionality here: with the
    additive term comparable to the stellar term, the ratio must depart from 2.
    """
    stellar_l_ir = float(
        np.asarray(_model(synthetic_ssp_wide, model_name, 1.0).predict_state({}).derived["L_ir"])
    )
    assert stellar_l_ir > 0.0, "setup: expected a positive absorbed luminosity"

    # Additive term equal to the stellar term: L_tot = eta*L + L, so doubling
    # eta takes the ratio from 2 to (2L + L)/(L + L) = 1.5, not 2.
    s1 = _sed_dust_ir(synthetic_ssp_wide, model_name, 1.0, L_agn_ir=Fixed(stellar_l_ir))
    s2 = _sed_dust_ir(synthetic_ssp_wide, model_name, 2.0, L_agn_ir=Fixed(stellar_l_ir))
    nonzero = np.abs(s1) > 0
    ratio = s2[nonzero] / s1[nonzero]

    assert abs(float(ratio.max()) - 2.0) > 1e-6, (
        f"{model_name} is listed in AFFINE_MODELS but behaved proportionally "
        f"(ratio {ratio.max()!r}); if it is now linear, remove the exemption"
    )
    # The exact affine prediction, which also pins that L_agn_ir is applied.
    np.testing.assert_allclose(ratio, 1.5, rtol=1e-9)
