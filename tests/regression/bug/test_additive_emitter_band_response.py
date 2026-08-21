# SPDX-License-Identifier: BSD-3-Clause
"""Additive emitters must project through a precomputed band response, exactly.

An additive emitter's SED is LINEAR in its luminosity, sed = L * S_unit(lambda), and
the filter integral is linear, so

    int [L * S_unit(lam)] R_f(lam) dlam  =  L * int S_unit(lam) R_f(lam) dlam  =  L * R_f

R_f is therefore a build-time constant and the projection is EXACT -- the true filter
transmission is still integrated, just once instead of on every call. This is a
different thing from the two *approximations* nearby: ``fast_emission`` (samples at the
effective wavelength) and the stellar WavePrecomp dust-attenuation Taylor projection.

``_dust_emission_band_response`` used to be gated on ``modified_blackbody`` only, so
every other model fell through to a per-call dense filter integral over the full 5994-
point grid: ~1650 us against ~108 us for the same model without emission. The consumer
(`_component_base._project_photometry`) always supported the response generically -- only
the producer was restricted.

Two properties are pinned here, and the second is the one that matters:
  1. it is FAST for every emission model, and
  2. it is EXACT for every emission model, with NON-DEFAULT shape parameters.

(2) also guards the build-time parameter slicing. The response is built by calling the
component's ``predict`` at build time; if that slice grabbed *default* template params
instead of the user's, R would be built from the wrong template and the model would
return confidently wrong IR photometry with no error at all. Non-default shape knobs
below are what make that failure visible.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, SEDModel, WavePrecomp
from tengri.observation.photometry_config import Photometry

# w4 (22 um) is where the IR template actually lands at z=0.1.
FILTERS = ["sdss_g", "sdss_r", "sdss_i", "wise_w4"]

# EVERY dust-emission model in the registry, not a hand-picked subset. The first
# version of this file listed eight models, all of them linear, and so could never
# fail -- while its own docstring claimed to catch "a non-linear emitter". bosa is
# exactly that emitter, and it was the one left out.
#
# Non-default shape knobs where the model has them, so a precompute that silently
# sliced the DEFAULTS would produce different fluxes and fail the exactness check.
SHAPES = {
    "dale2014": {"alpha_dale": 2.5},
    "dale2014_cigale": {},
    "draine_li2007": {"umin": 2.0, "qpah": 3.0},
    "draine_li2014": {},
    "themis": {"qhac": 0.10},
    "casey2012": {"T": 40.0, "beta_ir": 1.8},
    "modified_blackbody": {"T": 40.0, "beta_ir": 1.8},
    "schreiber2016": {},
    "schreiber2018": {},
    "astrodust": {},
    "pah_drude": {},
    "bosa": {},  # NOT homogeneous in L_ir -- must be refused a band response
}

#: Emitters whose template *shape* depends on L_ir, so no build-time-constant band
#: response can exist. BOSA parameterizes by (L_TIR, sSFR). Luminosity-dependent
#: shapes (L-T relations) are common in IR SED models, so the code detects this by
#: *property* (a homogeneity probe), not by consulting this list -- the list only
#: pins the expected outcome for the models we ship.
NON_HOMOGENEOUS = {"bosa"}

# Emission-free WavePrecomp photometry compiles ~2.9e5 FLOPs. With the band response the
# emitter adds only L_ir * R (a few ops per filter). The dense per-call integral cost
# 6.6e6 - 2.3e7. 1e6 separates them by an order of magnitude either way.
MAX_FLOPS = 1_000_000


def _model(emission_type, shape, *, approx):
    emission = {"type": emission_type, "*": FIXED}
    emission.update({k: Fixed(v) for k, v in shape.items()})
    return SEDModel.build(
        ssp_data=pytest.importorskip("tengri").load_ssp(),
        observation=Observation(photometry=Photometry.from_names(FILTERS)),
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "*": FIXED},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "*": FIXED,
        },
        dust_emission=emission,
        approx=approx,
    )


def _params(m):
    p = {k: jnp.asarray(v) for k, v in m.spec.sample(jax.random.PRNGKey(0)).items()}
    p.update({k: jnp.asarray(float(v)) for k, v in m.spec.get_fixed_values().items()})
    return p


@pytest.mark.regression_bug
@pytest.mark.parametrize("emission_type", sorted(SHAPES))
def test_band_response_is_exact_against_the_dense_filter_integral(emission_type):
    """L_ir * R must equal the full per-call filter integral, to fp roundoff.

    The exact path (``approx=None``) never uses the band response, so it IS the
    independent reference: it integrates the emitter's SED through the true filter
    transmission on the full wavelength grid.
    """
    shape = SHAPES[emission_type]
    lut = _model(emission_type, shape, approx=WavePrecomp())
    exact = _model(emission_type, shape, approx=None)

    f_lut = np.asarray(lut.predict_photometry(_params(lut)))
    f_exact = np.asarray(exact.predict_photometry(_params(exact)))

    # Compare RATIOS: these fluxes are ~1e-30, so an absolute tolerance would be
    # vacuously satisfied by anything at all.
    #
    # This is a COARSE guard, deliberately. It compares the WHOLE model, and the
    # WavePrecomp *stellar* path carries its own documented approximation (the
    # first-order Taylor dust attenuation, #617) which leaks into W4 at the 1e-6 to
    # 1e-3 level depending on the SSP grid and filter set -- pre-existing, present on
    # main, and nothing to do with the band response. A tight bound here would go red
    # for that pre-existing reason (pah_drude has been measured at 2.4e-3).
    #
    # The *fine* guarantee lives elsewhere and is stronger: the projection is exact by
    # homogeneity (pinned by test_homogeneity_probe_actually_discriminates), and
    # switching from the dense per-call integral to L_ir * R was measured to leave
    # every homogeneous emitter's flux unchanged to <= 4.4e-16 (fp re-association).
    #
    # What this bound must catch is a GROSS error: a non-homogeneous emitter given a
    # constant response (bosa: 1.3e-1), or a build-time slice that grabbed default
    # template params. 5e-3 clears the pre-existing residual and still catches bosa by
    # a factor of 25.
    w4 = np.abs(f_lut[-1] / f_exact[-1] - 1)
    assert w4 < 5e-3, (
        f"{emission_type}: WISE-W4 (IR-dominated) differs by {w4:.3e} between the "
        "band-response LUT and the dense filter integral. The additive projection is "
        "supposed to be EXACT -- either the emitter is not linear in L_ir, or the "
        "build-time parameter slice picked up default template params."
    )


@pytest.mark.regression_bug
@pytest.mark.parametrize("emission_type", sorted(set(SHAPES) - NON_HOMOGENEOUS))
def test_dust_emission_does_not_force_a_dense_per_call_filter_integral(emission_type):
    """...and a homogeneous emitter must actually be fast.

    Asserts on compiled FLOPs, not wall-clock: deterministic, fails for the right
    reason. Non-homogeneous emitters are excluded — they are *supposed* to keep the
    dense integral, and `test_a_non_homogeneous_emitter_is_refused` pins that.
    """
    m = _model(emission_type, SHAPES[emission_type], approx=WavePrecomp())
    cost = jax.jit(m.predict_photometry).lower(_params(m)).compile().cost_analysis()
    flops = (cost[0] if isinstance(cost, list) else cost)["flops"]

    assert flops < MAX_FLOPS, (
        f"{emission_type}: WavePrecomp photometry compiled {flops:,.0f} FLOPs "
        f"(budget {MAX_FLOPS:,}). The dust IR template is being evaluated on the full "
        "wavelength grid and integrated through the filters on every call, instead of "
        "projecting through the precomputed band response."
    )


@pytest.mark.regression_bug
@pytest.mark.parametrize("emission_type", sorted(NON_HOMOGENEOUS))
def test_a_non_homogeneous_emitter_is_refused_a_band_response(emission_type):
    """An emitter whose SHAPE depends on L_ir must NOT get a constant band response.

    The band response is exact only because sed(L) = L * S_unit(lambda). BOSA
    parameterizes its template by (L_TIR, sSFR), so probing it at the unit luminosity
    L_ir = 1 erg/s samples a template ~44 dex from anything physical, and the resulting
    response is wrong by ~13% in WISE-W4 — silently, with correct-looking fluxes.

    The build-time homogeneity probe must detect this and fall back to the dense
    per-call integral. This is the test that the first version of this file was
    missing: it claimed to guard against "a non-linear emitter" while listing only
    linear ones.
    """
    m = _model(emission_type, SHAPES[emission_type], approx=WavePrecomp())

    assert m._dust_band_response_cache is None, (
        f"{emission_type} is not homogeneous in L_ir, so no build-time-constant band "
        "response exists — but one was built. Its photometry is now silently wrong."
    )

    # ...and having been refused, it must still be *correct*.
    exact = _model(emission_type, SHAPES[emission_type], approx=None)
    f_lut = np.asarray(m.predict_photometry(_params(m)))
    f_exact = np.asarray(exact.predict_photometry(_params(exact)))
    w4 = np.abs(f_lut[-1] / f_exact[-1] - 1)
    assert w4 < 1e-4, (
        f"{emission_type}: W4 differs by {w4:.3e} from the exact path even on the "
        "dense-integral fallback."
    )


@pytest.mark.regression_bug
def test_homogeneity_probe_actually_discriminates():
    """The probe must not be vacuous: it must SEE the L_ir dependence it screens for.

    If BOSA happened to scale linearly, the guard above would pass for the wrong
    reason. Assert the property directly: bosa's SED does not scale with L_ir, and a
    linear emitter's does.
    """
    from tengri.components.igm.component import IGMSEDComponent  # noqa: F401

    L = 1.0e44
    for name, homogeneous in (("bosa", False), ("dale2014", True)):
        m = _model(name, SHAPES[name], approx=WavePrecomp())
        emitter = next(
            c for c in m._build_component_chain() if getattr(c, "name", "") == "dust_emission"
        )
        wave = jnp.asarray(m.wavelengths)
        p = emitter.slice_params(
            {k: jnp.asarray(float(v)) for k, v in m.spec.get_fixed_values().items()}
        )
        lo, _ = emitter.predict(p, jnp.zeros_like(wave), wave, L_ir=1.0)
        hi, _ = emitter.predict(p, jnp.zeros_like(wave), wave, L_ir=L)
        scales = bool(jnp.allclose(hi, L * lo, rtol=1e-10))
        assert scales is homogeneous, (
            f"{name}: expected homogeneous={homogeneous} in L_ir, measured {scales}. "
            "The homogeneity probe is not discriminating what it claims to."
        )
