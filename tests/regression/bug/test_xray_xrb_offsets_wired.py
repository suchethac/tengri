# SPDX-License-Identifier: BSD-3-Clause
"""``xray_det_hmxb`` / ``xray_det_lmxb`` must reach the physics (#1706).

Both are declared free on every X-ray model and were read by none of them.
:func:`xray_xrb_terms` has supported ``log_L_hmxb_offset`` /
``log_L_lmxb_offset`` all along — ``XRaySEDComponent`` simply never passed
them, and :func:`xray_total_lopez24_terms` would have swallowed them in
``**_kwargs`` had it been passed them.

The offsets are pure multiplicative amplitudes on their own term, so the
assertion is a *ratio*, not "something moved": a ratio cannot pass by accident
the way ``max_diff > 0`` can.

Both ``approx`` paths are exercised. The band-response precompute derives its
amplitudes by calling ``emission_terms`` at reference wavelengths, so it should
inherit the fix — but running only the exact path would not prove that, and
``WavePrecomp`` is what every fitter resolves ``approx="auto"`` to for
photometry.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

jax.config.update("jax_enable_x64", True)

#: The synthetic SSP grid stores absolute log10(Z) = [-1.5, -0.5, 0.0], which is
#: log10(Z/Zsun) = [0.348, 1.348, 1.848] once LOG10_ZSUN is removed. Sit in the
#: middle of that rather than tripping the out-of-grid clip warning (#442).
_MET_IN_GRID = 1.348


def _xray_observation():
    """Three synthetic bands at 0.1, 1 and 10 keV (124, 12.4, 1.24 A)."""
    centers = np.array([124.0, 12.4, 1.24])
    widths = np.array([30.0, 3.0, 0.3])
    curves = []
    for i, (c, w) in enumerate(zip(centers, widths)):
        wv = np.linspace(max(c - 3 * w, 0.1), c + 3 * w, 64)
        curves.append(
            FilterCurve(wave=wv, trans=np.exp(-0.5 * ((wv - c) / w) ** 2), name=f"xray_{i}")
        )
    return Observation(photometry=Photometry(filters=tuple(curves)))


def _build(synthetic_ssp, *, xray_model=None, approx=None, precompute=False, **param_overrides):
    """Return ``(model, base_params)`` with every free parameter given a value."""
    kwargs = {
        "mean_sfh_type": "dpl",
        "sfh_dpl_alpha": Fixed(1.5),
        "sfh_dpl_beta": Fixed(2.0),
        "sfh_dpl_tau_gyr": Fixed(5.0),
        "sfh_dpl_log_total_mass": Fixed(0.0),
        "met_logzsol": Fixed(_MET_IN_GRID),
        "dust_tau_bc": Fixed(0.0),
        "dust_tau_diff": Fixed(0.0),
        "dust_slope": Fixed(-0.7),
        "redshift": Fixed(0.1),
        "xray": True,
    }
    if xray_model is not None:
        kwargs["xray_model"] = xray_model
    kwargs.update(param_overrides)

    spec = Parameters(**kwargs)
    model = SEDModel(
        spec,
        synthetic_ssp,
        observation=_xray_observation(),
        precompute=precompute,
        approx=approx,
    )
    # Every free parameter needs a value; sample once and override per-test.
    base = {k: float(v) for k, v in spec.sample(jax.random.PRNGKey(0)).items()}
    return model, base


# ── the offsets must reach the physics ────────────────────────────


@pytest.mark.parametrize("xray_model", ["yang20", "lopez24"])
@pytest.mark.parametrize("param", ["xray_det_hmxb", "xray_det_lmxb"])
def test_offset_reaches_the_sed(synthetic_ssp, xray_model, param):
    """Sweeping the offset must move the SED (#1706).

    ``xray_det_*`` sit outside the ``xray`` group wildcard (#1676), so they only
    become free by explicit prior — which is how a user would reach them.
    """
    model, base = _build(synthetic_ssp, xray_model=xray_model, **{param: Uniform(-2.0, 2.0)})

    sed_0 = np.asarray(model.predict({**base, param: 0.0}).rest_sed())
    sed_h = np.asarray(model.predict({**base, param: 0.5}).rest_sed())

    assert not np.array_equal(sed_0, sed_h), (
        f"{param} is inert on xray_model={xray_model!r}: sweeping it across its "
        f"declared support left the SED bit-identical, so nothing reads it"
    )


@pytest.mark.parametrize("xray_model", ["yang20", "lopez24"])
@pytest.mark.parametrize("accel", ["wave_precomp", "precompute_flag"])
def test_offset_is_live_on_every_accelerated_path(synthetic_ssp, xray_model, accel):
    """Every accelerated path must inherit the offsets (#1706).

    Fitters resolve ``approx="auto"`` to ``WavePrecomp`` for photometry, so a fix
    that worked only on the exact path would leave real fits sampling a perfectly
    flat dimension while a ``predict()`` test measured it as fixed. ``precompute=
    True`` is covered as well because it is the flag that builds the
    ``PreintegratedGrid`` family in ``xray_precompute.py``.
    """
    kwargs = {"approx": WavePrecomp()} if accel == "wave_precomp" else {"precompute": True}
    model, base = _build(
        synthetic_ssp,
        xray_model=xray_model,
        xray_det_hmxb=Uniform(-2.0, 2.0),
        **kwargs,
    )

    phot_0 = np.asarray(model.predict_photometry({**base, "xray_det_hmxb": 0.0}))
    phot_h = np.asarray(model.predict_photometry({**base, "xray_det_hmxb": 1.0}))

    assert np.any(phot_h != phot_0), (
        f"xray_det_hmxb is inert under {accel} on xray_model={xray_model!r}"
    )


# ── control: the arm is reachable on the same build ───────────────


@pytest.mark.parametrize("xray_model", ["yang20", "lopez24"])
def test_control_gamma_hmxb_is_live(synthetic_ssp, xray_model):
    """A sibling X-ray parameter must move the SED on the same build.

    Without this, the assertions above could pass on a fixture whose X-ray arm
    contributes nothing at all — an inert-parameter test needs a live control.
    """
    model, base = _build(synthetic_ssp, xray_model=xray_model, xray_gamma_hmxb=Uniform(1.7, 2.3))

    sed_lo = np.asarray(model.predict({**base, "xray_gamma_hmxb": 1.8}).rest_sed())
    sed_hi = np.asarray(model.predict({**base, "xray_gamma_hmxb": 2.2}).rest_sed())

    assert not np.array_equal(sed_lo, sed_hi), (
        f"control failed: xray_gamma_hmxb is itself inert on {xray_model!r}, so this "
        f"fixture cannot detect an inert offset"
    )
