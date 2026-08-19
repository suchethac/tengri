# SPDX-License-Identifier: BSD-3-Clause
"""phot_rest_fnu meant different things on the exact and LUT paths (#1148).

``phot_rest_fnu`` — and so ``Observables.mag_absolute`` — is the SED reprojected at
z=0, d_L=10 pc: *the galaxy as it is*. The filter therefore sits in the **rest** frame
and samples the rest SED at its own pivot.

The LUT reused ``total_lnu``, the **observed**-band sum, which samples rest
λ_eff/(1+z). Those are different physical quantities. Against the exact path the LUT
ran **769 % out in des_g at z=0.5** and orders of magnitude out in the blue — so an
object's ABSOLUTE magnitude depended on its redshift, and on which ``approx`` you
passed. A speed knob must never change the physics.

Not a units bug, and it matters which: the discriminator is that the ratio tracks the
SED's own color between λ_eff and λ_eff/(1+z) with **no leftover constant** — verified
below by band-integrating the model's own rest SED both ways.

Every accuracy assertion compares the LUT against the **exact path** (approx=None) —
never against another preintegral.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp

pytestmark = pytest.mark.regression_bug

KEY = jax.random.PRNGKey(0)

#: The gap grows with redshift and is worst in the blue, where the SED is steepest.
REDSHIFTS = [0.0, 0.5, 1.0, 2.0, 3.0]

BANDS = ["galex_fuv", "galex_nuv", "des_g", "des_r", "wise_w1"]
FUV, NUV, G, R, W1 = range(5)


def _build(ssp, z: float, approx, *, tau_diff=0.5, tau_bc=1.0, bands=None) -> SEDModel:
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(bands or BANDS)),
        sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={"law": "power_law", "type": "two_component", "*": FIXED, "tau_diff": tau_diff, "tau_bc": tau_bc},
        redshift=Fixed(z),
        approx=approx,
    )


def _rest_lut_vs_exact(ssp, z: float, **kw):
    """Per-band LUT/exact - 1 for phot_rest_fnu. ONE shared param dict.

    ``spec.sample()`` returns FIXED params too, so a dict sampled from a
    differently-configured model would override this one's fixed dust.
    """
    m_exact = _build(ssp, z, None, **kw)
    p = dict(m_exact.spec.sample(KEY))
    exact = np.asarray(m_exact.predict_observables(p).phot_rest_fnu)
    lut = np.asarray(_build(ssp, z, WavePrecomp(), **kw).predict_observables(p).phot_rest_fnu)
    return lut / exact - 1.0


# ── 1. the bug: the two paths must agree ──────────────────────────────────────


@pytest.mark.parametrize("z", REDSHIFTS)
def test_rest_frame_photometry_agrees_between_the_paths(synthetic_ssp_wide, z):
    """A speed knob must not change what phot_rest_fnu MEANS.

    Pre-fix, with real filters: +769 % (des_g, z=0.5), +5.7e6 % (galex_fuv, z=1),
    -50 % (wise_w1, z=1) — the sign even flips, because there the LUT samples the
    bluer, brighter side of the stellar Rayleigh-Jeans tail.

    Neuter-check: restore ``phot_rest_fnu = total_lnu * cosmology_rest`` in
    ``predict_via_precomp`` and this goes red at every z > 0.
    """
    err = _rest_lut_vs_exact(synthetic_ssp_wide, z)
    worst = float(np.max(np.abs(err)))
    assert worst < 0.02, (
        f"LUT rest-frame photometry is {worst * 100:.2f}% off the exact path at z={z} "
        f"(per band: {np.round(err * 100, 2)})"
    )


def test_the_disagreement_was_the_frame_not_the_units(synthetic_ssp_wide):
    """Pins the DIAGNOSIS, not just the symptom.

    A units bug and a frame (K-correction) bug both look like "the paths disagree,
    and agree at z=0". They are told apart by band-integrating the model's own rest
    SED two ways with the SAME quadrature:

        A = filter at z=0  -> the filter's own λ IS the rest λ   (rest-frame band)
        B = filter at z    -> samples L_nu at λ_filter/(1+z)     (observed band)

    The exact path is A. The LUT *was* B. Both to machine precision, with no leftover
    constant — so there was never a stray (1+z) factor to find, and anyone who goes
    looking for one is wasting their time.

    This test asserts the exact path is A, which is the contract ``phot_rest_fnu``
    documents ("the SED reprojected at z=0, d_L=10 pc").
    """
    from tengri.observation.photometry import lnu_filter_integral_batch, pad_filters
    from tengri.utils.physics_constants import TEN_PC_CM

    z = 1.0
    m = _build(synthetic_ssp_wide, z, None)
    p = dict(m.spec.sample(KEY))
    exact = np.asarray(m.predict_observables(p).phot_rest_fnu)

    ph = m.observation.photometry
    fw, ft, _ = pad_filters(
        [jnp.asarray(w) for w in ph.filter_waves], [jnp.asarray(t) for t in ph.filter_trans]
    )
    l_nu = np.asarray(m.predict(p).rest_sed())
    w_rest = np.asarray(m.wavelengths)
    rest_scale = 1.0 / (4.0 * np.pi * TEN_PC_CM**2)

    band_at_z0 = (
        np.asarray(lnu_filter_integral_batch(l_nu, w_rest, fw, ft, 0.0, convention=ph.convention))[
            : len(BANDS)
        ]
        * rest_scale
    )
    band_at_z = (
        np.asarray(lnu_filter_integral_batch(l_nu, w_rest, fw, ft, z, convention=ph.convention))[
            : len(BANDS)
        ]
        * rest_scale
    )

    np.testing.assert_allclose(exact, band_at_z0, rtol=1e-6)
    # And the two references are genuinely different — otherwise this proves nothing.
    assert np.max(np.abs(band_at_z0 / band_at_z - 1.0)) > 0.5, (
        "the rest-frame and observed-frame band integrals coincide here, so this test "
        "cannot distinguish them — pick a band/redshift where the SED has color"
    )


# ── 2. the physical statement behind it ───────────────────────────────────────


def test_rest_frame_photometry_is_a_property_of_the_rest_sed_alone(synthetic_ssp_wide):
    """*The galaxy as it is.* phot_rest_fnu must be a functional of the rest SED and
    nothing else — in particular it must not know the source's redshift. That is the
    whole meaning of an ABSOLUTE magnitude, and it is what the LUT violated: its
    phot_rest_fnu tracked the OBSERVED band, so the same rest SED gave a different M
    just by being further away.

    Stated as "move the galaxy and M must not move", this would be untestable through
    ``redshift``: the SFH is normalized against cosmic time, so a model at z=2 has a
    younger universe and is a genuinely *different* galaxy. Instead, hold the rest SED
    fixed by construction — take the model's own rest SED at each z and check that
    phot_rest_fnu is exactly the z=0 band integral OF THAT SED, on both paths. If a
    path smuggles the redshift into the projection, this goes red.
    """
    from tengri.observation.photometry import lnu_filter_integral_batch, pad_filters
    from tengri.utils.physics_constants import TEN_PC_CM

    rest_scale = 1.0 / (4.0 * np.pi * TEN_PC_CM**2)
    for z in (0.5, 1.0, 2.0):
        m_exact = _build(synthetic_ssp_wide, z, None)
        p = dict(m_exact.spec.sample(KEY))

        ph = m_exact.observation.photometry
        fw, ft, _ = pad_filters(
            [jnp.asarray(w) for w in ph.filter_waves],
            [jnp.asarray(t) for t in ph.filter_trans],
        )
        l_nu = np.asarray(m_exact.predict(p).rest_sed())
        reference = (
            np.asarray(
                lnu_filter_integral_batch(
                    l_nu, np.asarray(m_exact.wavelengths), fw, ft, 0.0, convention=ph.convention
                )
            )[: len(BANDS)]
            * rest_scale
        )

        for approx in (None, WavePrecomp()):
            got = np.asarray(
                _build(synthetic_ssp_wide, z, approx).predict_observables(p).phot_rest_fnu
            )
            worst = float(np.max(np.abs(got / reference - 1.0)))
            assert worst < 0.02, (
                f"phot_rest_fnu is not the z=0 band integral of the rest SED "
                f"({worst * 100:.2f}% off) at z={z}, approx={approx!r} — the projection "
                f"is carrying the source's redshift into a rest-frame quantity"
            )


# ── 3. the fix must not disturb the observed frame ────────────────────────────


@pytest.mark.parametrize("z", [0.5, 1.0, 2.0])
def test_observed_photometry_is_untouched(synthetic_ssp_wide, z):
    """phot_fnu is the likelihood channel. This PR must not move it."""
    m_exact = _build(synthetic_ssp_wide, z, None)
    p = dict(m_exact.spec.sample(KEY))
    exact = np.asarray(m_exact.predict_photometry(p))
    lut = np.asarray(_build(synthetic_ssp_wide, z, WavePrecomp()).predict_photometry(p))
    worst = float(np.max(np.abs(lut / exact - 1.0)))
    assert worst < 0.02, f"observed photometry moved {worst * 100:.2f}% at z={z}"


def test_the_rest_band_does_not_pin_the_full_grid(synthetic_ssp_wide):
    """The rest-band tensors are build-time constants, so the LUT must stay far
    cheaper than the exact path (#1107's dead-code-elimination trap: anything that
    reads a full-grid array in the projector forfeits the entire WavePrecomp
    speedup).
    """
    m_exact = _build(synthetic_ssp_wide, 0.8, None)
    m_lut = _build(synthetic_ssp_wide, 0.8, WavePrecomp())
    p = dict(m_exact.spec.sample(KEY))

    def _flops(m):
        f = jax.jit(lambda q: m.predict_observables(q))
        a = f.lower(p).compile().cost_analysis()
        if isinstance(a, list):
            a = a[0]
        return int(a["flops"])

    ratio = _flops(m_exact) / _flops(m_lut)
    assert ratio >= 10.0, f"LUT is only {ratio:.1f}x cheaper than exact — the rest band pinned it"


# ── 4. the rest band is redshift-independent (why it is free) ─────────────────


def test_the_rest_band_lut_carries_no_redshift_axis(synthetic_ssp_wide):
    """At z=0 the filter always samples the same rest wavelengths, so the rest-band
    tensor is ONE build-time constant serving fixed-z and free-z alike — no z-table,
    no interpolation. If someone gives it a redshift axis, they have misunderstood it
    and paid for a table they do not need.
    """
    m_fixed = _build(synthetic_ssp_wide, 1.0, WavePrecomp())
    rb_fixed = m_fixed._cached_component_chain[0]._state.restband_lut
    assert rb_fixed is not None, "the rest-band LUT was not built"
    assert rb_fixed.ssp_restband_phot.ndim == 3, "(n_met, n_age, n_filter) — no z axis"

    m_free = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=Observation(photometry=Photometry.from_names(BANDS)),
        sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={"law": "power_law", "type": "two_component", "*": FIXED, "tau_diff": 0.5, "tau_bc": 1.0},
        redshift=Uniform(0.1, 2.0),
        approx=WavePrecomp(),
    )
    rb_free = m_free._cached_component_chain[0]._state.restband_lut
    assert rb_free is not None, "the free-z path did not get a rest-band LUT"
    np.testing.assert_allclose(
        np.asarray(rb_free.ssp_restband_phot),
        np.asarray(rb_fixed.ssp_restband_phot),
        rtol=0.0,
        atol=0.0,
        err_msg="the rest band must not depend on the model's redshift",
    )
