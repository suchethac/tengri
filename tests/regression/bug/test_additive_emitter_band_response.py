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

# Deliberately non-default shape knobs, so a precompute that silently sliced the
# DEFAULTS would produce different fluxes and fail the exactness check.
SHAPES = {
    "dale2014": {"alpha_dale": 2.5},
    "draine_li2007": {"umin": 2.0, "qpah": 3.0},
    "draine_li2014": {},
    "themis": {"qhac": 0.10},
    "casey2012": {"T": 40.0, "beta_ir": 1.8},
    "modified_blackbody": {"T": 40.0, "beta_ir": 1.8},
    "schreiber2016": {},
    "astrodust": {},
}

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
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "emission": emission,
        },
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
    # Bound is 1e-4, not machine epsilon, because this compares the WHOLE model: the
    # WavePrecomp *stellar* path carries its own documented approximation (the
    # first-order Taylor dust attenuation, #617), which leaks a little even into W4.
    # That residual is pre-existing and unrelated to the additive projection --
    # astrodust sits at 6e-6 here both before and after the band response.
    #
    # The exactness of the additive projection *itself* is what the emitter-linearity
    # argument guarantees, and it is measured directly: switching this projection from
    # the dense per-call filter integral to L_ir * R leaves every one of these eight
    # models' fluxes unchanged to <= 4.4e-16 (fp re-association). This assertion is the
    # coarse guard that catches a *gross* error -- a non-linear emitter, or a build-time
    # slice that grabbed default template parameters.
    w4 = np.abs(f_lut[-1] / f_exact[-1] - 1)
    assert w4 < 1e-4, (
        f"{emission_type}: WISE-W4 (IR-dominated) differs by {w4:.3e} between the "
        "band-response LUT and the dense filter integral. The additive projection is "
        "supposed to be EXACT -- either the emitter is not linear in L_ir, or the "
        "build-time parameter slice picked up default template params."
    )


@pytest.mark.regression_bug
@pytest.mark.parametrize("emission_type", sorted(SHAPES))
def test_dust_emission_does_not_force_a_dense_per_call_filter_integral(emission_type):
    """...and it must actually be fast. Asserts on compiled FLOPs, not wall-clock."""
    m = _model(emission_type, SHAPES[emission_type], approx=WavePrecomp())
    cost = jax.jit(m.predict_photometry).lower(_params(m)).compile().cost_analysis()
    flops = (cost[0] if isinstance(cost, list) else cost)["flops"]

    assert flops < MAX_FLOPS, (
        f"{emission_type}: WavePrecomp photometry compiled {flops:,.0f} FLOPs "
        f"(budget {MAX_FLOPS:,}). The dust IR template is being evaluated on the full "
        "wavelength grid and integrated through the filters on every call, instead of "
        "projecting through the precomputed band response."
    )
