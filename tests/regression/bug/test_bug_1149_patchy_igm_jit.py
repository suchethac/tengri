# SPDX-License-Identifier: BSD-3-Clause
r"""Patchy reionization was unreachable, and once reachable was 281 % wrong (#1149).

**The crash.** ``igm_absorption`` dispatched with

    elif igm_patchy and igm_x_HI > 0.0:

``igm_patchy`` is a static structural flag (from the build config), but
``igm_x_HI`` arrives from the params dict, which is *traced* under ``jax.jit`` —
even when FIXED at its 0.0 default, a fixed param is still a tracer inside the
compiled ``_impl``. Python's ``and`` short-circuits to evaluate the traced
``igm_x_HI > 0.0``, whose ``bool[]`` cannot be converted, so **every** patchy
configuration died at trace time with ``TracerBoolConversionError`` — on the
exact path and under ``WavePrecomp`` alike. The z > 6 damping-wing model, the one
you reach for exactly where the IGM dominates the photometry, could not be called.
The guard was also *redundant*: the damping-wing optical depth scales linearly
with ``x_HI``, so patchy reduces bit-for-bit to the mean Inoue+2014 model at
``x_HI = 0`` (``exp(-0) = 1``). Because ``igm_patchy`` is genuinely static the
dispatch collapses at trace time and needs no ``jnp.where``; the fix drops the
traced half of the condition.

**The covariance gap the crash was hiding.** Merely making it trace exposed a
worse failure: under ``WavePrecomp`` the mean IGM folds :math:`T` at the sub-band
quadrature nodes as a build-time constant (#1135), but patchy reads a free
parameter, so that fold was absent and the projector band-averaged
:math:`\langle T \rangle` over the whole flux — :math:`\langle S \rangle \langle
T \rangle` where the flux needs :math:`\langle S T \rangle`. Across jwst_f115w /
galex_fuv at z=7 that reached **+281 %**. The fix folds :math:`T` at the same
nodes at *runtime* (``IGMSEDComponent._fold_transmission_into_subbands``), so the
parametric IGM gets the same quadrature: patchy is now no less accurate under the
LUT than the mean IGM it extends.

Neuter-checks:
- Restore ``elif igm_patchy and igm_x_HI > 0.0:`` → tests 1-5 go red with
  ``TracerBoolConversionError`` on today's ``main``.
- Drop the ``_fold_transmission_into_subbands`` call → test 4 goes red (+281 %).

Every accuracy assertion compares the LUT against the **exact path**
(``approx=None``) — never against another preintegral.

References
----------
- Miralda-Escude 1998, ApJ, 501, 15 (integrated damping wing)
- Mason et al. 2018, ApJ, 856, 2 (patchy reionization neutral fraction)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp

pytestmark = pytest.mark.regression_bug

KEY = jax.random.PRNGKey(0)

#: jwst_f115w (11570 Å) lands on the peak of the z=7 damping wing (observed Lyα is
#: 9725 Å, the wing bites redward): the band that actually feels the neutral IGM.
WING_BAND = "jwst_f115w"
#: galex_fuv/des_g are the issue's own repro bands, positive at the low z where it
#: was first hit; at z=7 they sit deep in the Lyman forest.
REPRO_BANDS = ["galex_fuv", "des_g"]
BANDS = [*REPRO_BANDS, WING_BAND]


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


def _build(ssp, *, z, approx, bands=BANDS, patchy=True, x_HI=None):
    """A star-forming galaxy behind the patchy-reionization IGM.

    ``x_HI`` given as a ``Distribution`` frees the neutral fraction; given as a
    float pins it; ``patchy=False`` selects the plain mean Inoue+2014 IGM.
    """
    if patchy:
        igm = {"type": "inoue14", "patchy": True}
        if x_HI is not None:
            igm["x_HI"] = x_HI if hasattr(x_HI, "sample") else Fixed(float(x_HI))
    else:
        igm = {"type": "inoue14"}
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(bands)),
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.3,
        },
        redshift=Fixed(z),
        approx=approx,
        igm=igm,
    )


# ── 1. the bug: patchy must be reachable on BOTH paths ─────────────────────────


@pytest.mark.parametrize("approx", [None, WavePrecomp()], ids=["exact", "waveprecomp"])
def test_patchy_igm_is_reachable(ssp, approx):
    """It must simply trace and return finite, non-negative fluxes.

    Pre-fix this raised ``TracerBoolConversionError`` on both the exact path and
    under ``WavePrecomp`` — patchy reionization could not be called in any
    configuration. z=0.8 with optical/UV bands (the issue's own repro) keeps every
    band positive so the assertion is unambiguous.
    """
    m = _build(ssp, z=0.8, approx=approx, bands=REPRO_BANDS)
    flux = np.asarray(m.predict_photometry(dict(m.spec.sample(KEY))))
    assert np.all(np.isfinite(flux)), f"non-finite patchy photometry: {flux}"
    assert np.all(flux > 0.0), f"non-positive patchy photometry: {flux}"


# ── 2. the gradient the sampler will actually take ─────────────────────────────


@pytest.mark.gradient
def test_gradient_wrt_neutral_fraction_is_finite(ssp):
    """``igm_x_HI`` is the parameter the sampler moves and the one that was being
    branched on. ``jax.grad`` w.r.t. it must be finite — and non-zero, or the
    neutral fraction has been silently dropped from the graph rather than wired in.
    """
    m = _build(ssp, z=7.0, approx=None, x_HI=Uniform(0.0, 0.9))
    assert "igm_x_HI" in m.spec.free_params
    p = dict(m.spec.sample(KEY))

    def loss(x_HI):
        return jnp.sum(m.predict_photometry({**p, "igm_x_HI": x_HI}))

    g = float(jax.grad(loss)(0.5))
    assert np.isfinite(g), "gradient w.r.t. igm_x_HI is not finite"
    assert g != 0.0, "gradient w.r.t. igm_x_HI is exactly zero — the param is a no-op"


# ── 3. behavior preservation: the dropped guard was redundant ──────────────────


def test_patchy_at_zero_neutral_fraction_equals_the_mean_igm_exactly(ssp):
    """On the exact path, ``x_HI = 0`` patchy must reproduce the plain Inoue+2014
    model bit-for-bit — precisely why dropping ``and igm_x_HI > 0.0`` changes no
    physics. The registry maps ``"inoue14"`` to the very callable
    ``igm_transmission_patchy`` uses as its base, and the wing contributes
    ``exp(0) = 1``, so the two agree exactly, not merely to a tolerance.
    """
    m_patchy = _build(ssp, z=7.0, approx=None, x_HI=0.0)
    m_mean = _build(ssp, z=7.0, approx=None, patchy=False)
    p = dict(m_mean.spec.sample(KEY))  # identical free params (both dpl, dust FIXED)

    np.testing.assert_allclose(
        np.asarray(m_patchy.predict_photometry(p)),
        np.asarray(m_mean.predict_photometry(p)),
        rtol=0.0,
        atol=0.0,
        err_msg="patchy IGM at x_HI=0 must equal the mean Inoue+2014 IGM exactly",
    )


# ── 4. the covariance gap: patchy under the LUT is as accurate as the mean IGM ──


@pytest.mark.parametrize("x_HI", [0.0, 0.5, 0.8])
def test_waveprecomp_patchy_is_as_accurate_as_the_mean_igm(ssp, x_HI):
    """The runtime sub-band fold must close the covariance gap the crash was hiding.

    Without it, the WavePrecomp patchy path band-averages ⟨T⟩ over the whole flux
    (⟨S⟩·⟨T⟩), which ran +281 % / +176 % out in galex_fuv / des_g at z=7. Folding T
    at the quadrature nodes at runtime gives patchy the same ⟨S·T⟩ quadrature the
    mean IGM gets — so its LUT-vs-exact error must be no worse than the mean IGM's
    own (a self-calibrating bar: the residual that remains is the shared LUT
    accuracy for extreme-z forest bands, not a patchy-specific gap).

    Neuter-check: drop ``_fold_transmission_into_subbands`` and this goes red at
    every x_HI (+281 %).
    """
    p = dict(_build(ssp, z=7.0, approx=None, patchy=False).spec.sample(KEY))

    def err(patchy, x):
        exact = np.asarray(
            _build(ssp, z=7.0, approx=None, patchy=patchy, x_HI=x).predict_photometry(p)
        )
        lut = np.asarray(
            _build(ssp, z=7.0, approx=WavePrecomp(), patchy=patchy, x_HI=x).predict_photometry(p)
        )
        return np.abs(lut / exact - 1.0)

    worst_patchy = float(np.max(err(True, x_HI)))
    worst_mean = float(np.max(err(False, None)))
    assert worst_patchy <= worst_mean + 0.01, (
        f"patchy LUT is {worst_patchy * 100:.2f}% off exact at x_HI={x_HI}, "
        f"worse than the mean IGM's {worst_mean * 100:.2f}% — the fold did not close "
        f"the covariance gap"
    )


# ── 5. and it must not be a no-op: the wing actually absorbs, on both paths ─────


@pytest.mark.parametrize("approx", [None, WavePrecomp()], ids=["exact", "waveprecomp"])
def test_neutral_fraction_suppresses_the_damping_wing_band(ssp, approx):
    """A neutral IGM must dim the band that straddles the damping wing — and the
    runtime fold must carry that physics into the LUT, not just the exact path.

    Guards the opposite failure of #1149: a "fix" that traced by routing
    everything through the mean IGM would pass tests 1-4 while ignoring the neutral
    fraction. At z=7 the wing suppresses jwst_f115w (11570 Å, just redward of
    observed Lyα), so the x_HI=0.8 flux there must sit strictly below the mean-IGM
    flux on whichever path is under test.
    """
    wing = BANDS.index(WING_BAND)
    p = dict(_build(ssp, z=7.0, approx=approx, patchy=False).spec.sample(KEY))
    f_neutral = float(
        np.asarray(_build(ssp, z=7.0, approx=approx, x_HI=0.8).predict_photometry(p))[wing]
    )
    f_mean = float(
        np.asarray(_build(ssp, z=7.0, approx=approx, patchy=False).predict_photometry(p))[wing]
    )
    assert f_neutral < f_mean, (
        f"neutral IGM did not suppress {WING_BAND} on approx={approx!r}: "
        f"x_HI=0.8 -> {f_neutral:.3e}, mean IGM -> {f_mean:.3e}"
    )
