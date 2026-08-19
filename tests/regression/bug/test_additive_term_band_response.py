# SPDX-License-Identifier: BSD-3-Clause
"""Multi-term additive emitters must project through a precomputed band response, exactly.

#1107 fixed dust IR by assuming an additive emitter is ``sed = L * S_unit(lambda)`` — ONE
luminosity, ONE shape — so its band flux is ``L * R`` with ``R`` a build-time constant.
That is the k=1 case of the real structure. X-ray and radio are sums of *several* rank-1
terms:

    sed(lambda) = sum_k A_k(inputs) * S_k(lambda; shape params)

X-ray is k=4 (HMXB, LMXB, hot gas, corona) and radio k=3 (SF synchrotron, free-free, AGN
jet). The filter integral is linear, so each term factorizes and the band flux is
``sum_k A_k * R_kf`` — still exact, still one build-time integral through the true filter
transmission.

The SUM is emphatically NOT rank-1, which is the whole point: HMXB (Gamma=2.0) and LMXB
(Gamma=1.6) carry different photon indices, so the HMXB/LMXB mix — and hence the summed
spectral shape — shifts with SFR and stellar mass. A band response built from the *total*
would be silently wrong. ``test_the_summed_sed_is_not_rank1`` pins that, so nobody
"simplifies" the terms back into a total.

Pinned here:
  1. the fast path is actually FAST (compiled FLOPs near the emitter-free budget),
  2. it is EXACT against the dense per-call filter integral, and
  3. the rank-1 probe DISCRIMINATES — it does not wave everything through.

(2) is the one that matters, and it is checked with NON-DEFAULT shape parameters: the
response is built at build time from the fixed parameter values, so a precompute that
read *defaults* instead would return confidently wrong fluxes with no error at all.
Non-default shape knobs are what make that failure visible.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, SEDModel, Uniform, WavePrecomp
from tengri.observation.filters import load_tophat_filter
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.regression_bug

# The bands MUST include ones the emitter actually shines in, or the whole test is
# vacuous. X-ray emission is masked to lambda < 124 A and radio lives beyond ~1 mm, so
# an optical-only filter set cannot see either: perturbing the build-time shape by 10%
# then moves `predict_photometry` by exactly nothing, and the test passes while the
# precompute is broken. That is not hypothetical — this file's first draft did exactly
# that, and only a neuter check (break the fix on purpose, watch the test stay green)
# caught it. Hence the tophats below AND the explicit non-vacuity assertion in
# `_emitter_bands`, which fails loudly rather than silently proving nothing.
OPTICAL = ["sdss_g", "wise_w4"]

#: 2-10 keV -> lambda = 12.398 / E[keV] = 1.24-6.20 A. An X-CIGALE-style X-ray band.
XRAY_BAND = load_tophat_filter(3.72, 4.96, name="xray_2_10kev")
#: 1.4 GHz -> lambda = c/nu = 2.14e9 A (21 cm). The canonical radio continuum band.
RADIO_BAND = load_tophat_filter(2.14e9, 2.14e8, name="radio_1p4ghz")

#: Budget on the emitter's *incremental* compiled FLOPs — the model with it, minus the
#: same model without it. Incremental rather than absolute because the AGN block these
#: models carry still pays its own dense per-call integral (#1109 covers it separately,
#: and it dominates any absolute budget); the claim under test is that *this emitter*
#: adds no dense integral, and that claim is exactly the difference.
#:
#: With the term band response each emitter adds only sum_k A_k * R_kf — a few ops per
#: filter. Without it the emitter added 5.2e6 (X-ray) to 3.3e7 (radio). The budget has
#: to sit between those, and comfortably.
#:
#: Raised 2e5 -> 5e5 in #1146. That PR committed the SSP grids, which turned this test
#: ON in CI for the first time — it had been skipping for want of `data/`, so its budget
#: had never been checked on a CI box. It does not hold there:
#:
#:     xray_lopez24, same code, same 5994-wavelength grid
#:         macOS / arm64 : added = 148,600
#:         linux / x86-64: added = 217,712     <- over the old 2e5 budget
#:
#: XLA's cost analysis is platform-dependent; the old budget was calibrated on arm64.
#: The emitter itself is NOT regressing — the invariant under test is "no DENSE per-call
#: integral", i.e. no O(n_wave) term, and that still holds:
#:
#:     n_wave x5.0 (1200 -> 5994)  ->  added FLOPs x1.40   (dense would be x5)
#:
#: 5e5 clears the platform spread while staying ~10x below the dense regression this
#: test exists to catch (5.2e6). Do not raise it further without re-measuring the
#: scaling above: a budget that no longer separates 1e5 from 5e6 is not a guard.
MAX_EMITTER_FLOPS = 500_000

#: An AGN block, so the radio jet term and the X-ray corona are genuinely LIT. Without
#: it ``L_agn_bol = 0`` and both terms are identically zero — which is exactly how a
#: first pass at this measured "radio contributes nothing" and nearly concluded the
#: support could be truncated. A probe that omits the component under test confirms
#: whatever you already believed.
AGN = {
    "type": "composable",
    "torus": {"type": "skirtor"},
    "*": FIXED,
    "log_lbol": Fixed(45.5),
    "fracAGN": Fixed(0.3),
}

#: Non-default shape knobs per emitter, plus the free-parameter prior used to force the
#: gate to REFUSE (which is how we obtain an independent dense reference for the same
#: physics: the exact per-call filter integral the emitter falls back to).
EMITTERS = {
    "radio": {
        "group": {"type": "condon92", "*": FIXED, "alpha_sf": Fixed(0.9)},
        "free": ("radio_alpha_sf", Uniform(0.6, 1.0)),
        "band": RADIO_BAND,
    },
    "xray_yang20": {
        "group": {"type": "yang20", "*": FIXED, "gamma_agn": Fixed(2.1)},
        "free": ("xray_gamma_agn", Uniform(1.4, 2.2)),
        "block": "xray",
        "band": XRAY_BAND,
    },
    "xray_lopez24": {
        "group": {"type": "lopez24", "*": FIXED, "gamma_agn": Fixed(2.1)},
        "free": ("xray_gamma_agn", Uniform(1.4, 2.2)),
        "block": "xray",
        "band": XRAY_BAND,
    },
}


def _block(name):
    return EMITTERS[name].get("block", name)


def _photometry(name):
    """Optical bands plus the band this emitter actually shines in."""
    optical = Photometry.from_names(OPTICAL)
    return Photometry(filters=(*tuple(optical.filters), EMITTERS[name]["band"]))


def _model(name, *, approx, free_shape=False, bands=None):
    """Build a model carrying one additive emitter (plus an AGN, so its terms are lit).

    ``name=None`` builds the same model with no additive emitter at all — the reference
    for the incremental-FLOPs budget. It must carry the SAME filter set (``bands=``), or
    the difference picks up the AGN block's dense integral scaling with filter count
    instead of the emitter's cost.

    ``free_shape=True`` frees one of the emitter's shape parameters, which the gate must
    refuse — dropping the emitter to the dense per-call filter integral. That is the
    independent reference the fast path is checked against.
    """
    block = {}
    if name is not None:
        spec = EMITTERS[name]
        group = dict(spec["group"])
        if free_shape:
            pname, prior = spec["free"]
            group[pname.split("_", 1)[1]] = prior
        block = {_block(name): group}
    return SEDModel.build(
        ssp_data=pytest.importorskip("tengri").load_ssp(),
        observation=Observation(photometry=_photometry(bands or name)),
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "*": FIXED},
        dust={
            "law_diff": "calzetti",
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "emission": {"type": "dale2014", "*": FIXED},
        },
        agn=AGN,
        approx=approx,
        **block,
    )


def _params(m):
    p = {k: jnp.asarray(v) for k, v in m.spec.sample(jax.random.PRNGKey(0)).items()}
    p.update({k: jnp.asarray(float(v)) for k, v in m.spec.get_fixed_values().items()})
    return p


@pytest.mark.parametrize("name", sorted(EMITTERS))
def test_the_emitter_band_is_not_vacuous(name):
    """The emitter must DOMINATE its own band, or every other test here proves nothing.

    This is the guard the rest of the file leans on, and it exists because two earlier
    drafts of these tests were silently vacuous:

      1. optical-only filters — X-ray is identically zero above 124 A, so perturbing the
         build-time shape by 10 % moved ``predict_photometry`` by exactly nothing; and
      2. reading the precompute off ``predict_state``, which defaults ``template_data``
         to None and therefore never threads the band response at all — so the "fast" and
         "dense" arms were both dense, and the comparison compared dense to dense.

    Both passed. Only a neuter check (break the fix on purpose, watch the test stay green)
    exposed them. So: assert, explicitly, that removing the emitter changes its band.
    """
    with_em = np.asarray(
        _model(name, approx=WavePrecomp()).predict_photometry(
            _params(_model(name, approx=WavePrecomp()))
        )
    )
    ref = _model(None, approx=WavePrecomp(), bands=name)
    without = np.asarray(ref.predict_photometry(_params(ref)))

    # The emitter's band is the last one (see _photometry).
    got, base = with_em[-1], without[-1]
    assert got > 0.0, f"{name}: its own band is zero — the model does not reach it."
    assert abs(got / max(base, 1e-300) - 1.0) > 1.0, (
        f"{name}: removing the emitter changes its own band by only "
        f"{abs(got / max(base, 1e-300) - 1.0):.2e} — the band is dominated by something "
        "else, so an error in this emitter would be invisible here."
    )


@pytest.mark.parametrize("name", sorted(EMITTERS))
def test_band_response_is_exact_against_the_dense_filter_integral(name):
    """sum_k A_k * R_kf must equal the full per-call filter integral, to fp roundoff.

    Goes through ``predict_photometry`` — the ONLY entry point that threads
    ``template_data`` and therefore the only one that exercises the band response at all.
    (``predict_state`` defaults it to None, so the components there quietly take the dense
    path; correct, just slow — and useless as a test of the fast one.)

    The reference is the SAME model with one shape parameter freed, which makes the gate
    refuse the response and fall back to integrating the emitter's SED through the true
    filter transmission on the full wavelength grid. The freed parameter is then pinned
    back to the value the fast model has fixed, so the two are the same physics computed
    two independent ways. ``test_the_emitter_band_is_not_vacuous`` guarantees the emitter
    actually dominates the band being compared.
    """
    lut = _model(name, approx=WavePrecomp())
    dense = _model(name, approx=WavePrecomp(), free_shape=True)

    pname, _ = EMITTERS[name]["free"]
    p_dense = _params(dense)
    p_dense[pname] = jnp.asarray(float(lut.spec.get_fixed_values()[pname]))

    f_lut = np.asarray(lut.predict_photometry(_params(lut)))
    f_dense = np.asarray(dense.predict_photometry(p_dense))

    nz = f_dense != 0.0
    assert nz.any()
    # Ratios, not absolutes: these fluxes span 1e-27 to 1e-35, so an absolute tolerance
    # would be vacuously satisfied by anything at all. 1e-12 is ~4 orders looser than the
    # measured fp re-association (2.2e-16), ~10 orders tighter than any real shape error.
    rel = np.abs(f_lut[nz] / f_dense[nz] - 1.0).max()
    assert rel < 1e-12, f"{name}: band response differs from the dense integral by {rel:.3e}"


def _flops(m):
    return jax.jit(m.predict_photometry).lower(_params(m)).compile().cost_analysis()["flops"]


@pytest.mark.parametrize("name", sorted(EMITTERS))
def test_band_response_is_fast(name):
    """The emitter must not force a dense per-call filter integral over the full grid.

    Values stay correct either way when the response is missed, which is exactly why the
    regression in #1109 survived a green suite for months: nothing asserted on the *cost*
    of a compiled kernel. This does.
    """
    with_emitter = _flops(_model(name, approx=WavePrecomp()))
    without = _flops(_model(None, approx=WavePrecomp(), bands=name))
    added = with_emitter - without
    assert added < MAX_EMITTER_FLOPS, (
        f"{name}: adds {added:,.0f} compiled FLOPs — it is still paying a dense per-call "
        f"filter integral (budget {MAX_EMITTER_FLOPS:,})."
    )


@pytest.mark.parametrize("name", sorted(EMITTERS))
def test_a_free_shape_parameter_refuses_the_band_response(name):
    """A free shape parameter must DISABLE the fast path, not silently freeze the shape.

    The response is a constant only while the emitter's spectral shape is. Free a photon
    index or a spectral index and S_k moves under the LUT — so the gate must refuse and
    fall back to the dense integral. Fail-safe: an unrecognized free parameter costs
    speed, never correctness.
    """
    dense = _model(name, approx=WavePrecomp(), free_shape=True)
    _params(dense)  # force the build-time warmup that populates the cache
    assert getattr(dense, f"_{_block(name)}_term_response_cache", "unset") is None


def test_the_summed_sed_is_not_rank1():
    """The X-ray TOTAL is not rank-1, which is why the terms cannot be summed first.

    HMXB (Gamma=2.0) and LMXB (Gamma=1.6) have different photon indices, so the shape of
    their sum depends on the HMXB/LMXB mix — i.e. on SFR and stellar mass. A band response
    built from ``xray_total`` would therefore be a function of the runtime inputs, and
    freezing it at build time would be silently wrong. This test is the reason
    ``xray_total_terms`` exists; if someone "simplifies" it back to a total, this goes red.
    """
    from tengri.components.xray.xray import xray_total, xray_total_terms

    wave = jnp.asarray(np.geomspace(0.05, 1e4, 2000))
    shape = dict(gamma_hmxb=2.0, gamma_lmxb=1.6, gamma_agn=1.8, E_cut=300.0, log_nh=21.0)
    lo = dict(sfr=1.0, stellar_mass=1e10, stellar_age_gyr=1.0, l_2500_30deg=1e29)
    hi = dict(sfr=50.0, stellar_mass=3e11, stellar_age_gyr=8.0, l_2500_30deg=7e31)

    a = np.asarray(xray_total(wave, **lo, **shape))
    b = np.asarray(xray_total(wave, **hi, **shape))
    nz = (a != 0) & (b != 0)
    ratio = b[nz] / a[nz]
    # rank-1 would mean a CONSTANT ratio (pure amplitude rescaling). It is not.
    spread = np.abs(ratio / ratio[0] - 1.0).max()
    assert spread > 1e-3, (
        "the summed X-ray SED came back rank-1 — if that is now true, the per-term "
        f"machinery is unnecessary; got shape spread {spread:.3e}"
    )

    # ...while every individual term IS rank-1. That is the property the build-time probe
    # verifies, and the reason the factorization is exact.
    ta = xray_total_terms(wave, **lo, **shape)
    tb = xray_total_terms(wave, **hi, **shape)
    for key in ta:
        x, y = np.asarray(ta[key]), np.asarray(tb[key])
        m = (x != 0) & (y != 0)
        if not m.any():
            continue
        r = y[m] / x[m]
        assert np.abs(r / r[0] - 1.0).max() < 1e-12, f"term {key!r} is not rank-1"
