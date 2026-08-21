# SPDX-License-Identifier: BSD-3-Clause
"""WavePrecomp must not evaluate the IGM curve on the full model grid (#932 perf).

``approx=WavePrecomp()`` does not build a smaller SED. The stellar component always
builds the full-resolution SED einsum, and the entire speedup is XLA *dead-code
eliminating* it because the LUT projector consumes only the precomputed per-filter
families. Anything that reads a full-grid array in ``predict_via_precomp`` pins the
whole grid alive and silently forfeits the speedup.

#932 fixed a real physics bug (photometry came back unattenuated at high z) but did
it by band-averaging the full-grid Inoue+2014 curve at *runtime*: a 5994-point
transmission evaluated on every call to produce n_filters numbers. That pinned the
grid and cost 12.1 MFLOPs per call — the stellar-only LUT kernel went from 103k
FLOPs (~60-110 us) to 12.4M (~1.8 ms), a 16x wall-clock regression, with every
value still correct. A green suite never noticed, because no test asserts on the
*cost* of the compiled kernel.

The band factor depends only on (z, filter, convention) — the transmission is
averaged alone, never weighted by the SED — so it is tabulated against z at build
time instead. These tests pin all three properties that makes safe: the kernel is
cheap again, the numbers are bit-identical, and the IGM is still actually applied.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, SEDModel, WavePrecomp
from tengri.observation.photometry_config import Photometry

FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]

# The stellar-only LUT kernel is ~2.7e5 FLOPs once the IGM curve is precomputed and
# the SED einsum is pruned. The regressed kernel was 1.24e7. Anything above this
# bound means a full-grid array is being consumed by the LUT projector again.
MAX_LUT_FLOPS = 1_000_000


def _model(*, approx=None, z=3.0, **extra):
    if "igm" not in extra:
        extra["igm"] = {"type": "inoue"}
    return SEDModel.build(
        ssp_data=pytest.importorskip("tengri").load_ssp(),
        observation=Observation(photometry=Photometry.from_names(FILTERS)),
        redshift=Fixed(z),
        sfh={"type": "dpl", "*": FIXED},
        approx=approx,
        **extra,
    )


def _params(model):
    p = {k: jnp.asarray(v) for k, v in model.spec.sample(jax.random.PRNGKey(0)).items()}
    p.update({k: jnp.asarray(float(v)) for k, v in model.spec.get_fixed_values().items()})
    return p


def _compiled_flops(model, params):
    cost = jax.jit(model.predict_photometry).lower(params).compile().cost_analysis()
    if isinstance(cost, list):
        cost = cost[0]
    return cost["flops"]


@pytest.mark.regression_bug
def test_igm_does_not_pin_the_full_grid_in_the_lut_kernel():
    """The LUT photometry kernel must not contain a full-grid IGM evaluation.

    Asserts on compiled FLOPs, not wall-clock: deterministic, and it fails for the
    right reason. Pre-fix this kernel compiled 12,396,433 FLOPs.
    """
    model = _model(approx=WavePrecomp())
    flops = _compiled_flops(model, _params(model))

    assert flops < MAX_LUT_FLOPS, (
        f"WavePrecomp photometry kernel compiled {flops:,.0f} FLOPs "
        f"(budget {MAX_LUT_FLOPS:,}). A full-grid array is being consumed by "
        "predict_via_precomp, which pins the model grid and defeats the dead-code "
        "elimination that IS the WavePrecomp speedup."
    )


@pytest.mark.regression_bug
def test_the_lut_kernel_is_far_cheaper_than_the_exact_one():
    """The whole point of the LUT: it must compile substantially less work."""
    exact, lut = _model(), _model(approx=WavePrecomp())
    f_exact = _compiled_flops(exact, _params(exact))
    f_lut = _compiled_flops(lut, _params(lut))

    assert f_lut * 10 < f_exact, (
        f"WavePrecomp compiled {f_lut:,.0f} FLOPs vs exact {f_exact:,.0f} "
        f"({f_exact / f_lut:.1f}x) — the LUT is not buying its keep."
    )


@pytest.mark.regression_bug
def test_band_factor_is_bit_identical_to_the_runtime_band_average():
    """Moving <T>_f to build time must not move a single bit at fixed z.

    The tabulated factor is the same lnu_filter_integral_batch quadrature the
    runtime used, evaluated at the same z — so it is exactly reproducible.
    """
    from tengri.components.igm.component import IGMSEDComponent
    from tengri.components.igm.igm import igm_absorption
    from tengri.observation.photometry import lnu_filter_integral_batch, pad_filters

    z = 3.0
    model = _model(approx=WavePrecomp(), z=z)
    phot = model.observation.photometry
    wave_rest = jnp.asarray(model.wavelengths)

    igm = next(c for c in model._build_component_chain() if isinstance(c, IGMSEDComponent))
    tabulated = igm._state.band_table[0]

    # Independent reference: the full-grid curve, band-averaged the old way.
    fw, ft, _ = pad_filters(list(phot.filter_waves), list(phot.filter_trans))
    trans = igm_absorption(wave_rest * (1.0 + z), z, igm_model=igm.config.igm_model)
    reference = lnu_filter_integral_batch(trans, wave_rest, fw, ft, z, convention=phot.convention)[
        : phot.n_filters
    ]

    np.testing.assert_array_equal(np.asarray(tabulated), np.asarray(reference))


@pytest.mark.regression_bug
def test_the_igm_is_still_actually_applied():
    """Guard the #932 bug itself: a fast path must not become a silent no-op.

    At z=3 the Lyman forest eats sdss_u. If the band factor were dropped, the LUT
    flux would match the IGM-free model instead of being strongly suppressed.
    """
    with_igm = _model(approx=WavePrecomp(), z=3.0)
    without = _model(approx=WavePrecomp(), z=3.0, igm={"type": "none"})

    f_with = np.asarray(with_igm.predict_photometry(_params(with_igm)))
    f_without = np.asarray(without.predict_photometry(_params(without)))

    # Dropping the factor would give a ratio of exactly 1. It is ~0.50 here: the
    # forest removes half the u-band flux at z=3.
    u_ratio = f_with[0] / f_without[0]
    assert u_ratio < 0.75, (
        f"sdss_u at z=3 is only suppressed by {1 - u_ratio:.1%} — the IGM band "
        "factor looks like it is not being applied on the LUT path (#932)."
    )
    # ...and the reddest band, far from the forest, must be essentially untouched.
    assert f_with[-1] / f_without[-1] > 0.99


@pytest.mark.regression_bug
def test_spectrum_lut_does_not_evaluate_the_igm_on_the_full_grid():
    """The SpectrumPrecomp twin: the LUT bought literally nothing before this.

    ``predict_spectrum_via_precomp`` sampled the full-grid Inoue+2014 curve at each
    pixel on every call, so the spectrum LUT ran at 2098 us against 2120 us exact --
    a 1.0x "speedup". A pixel's rest effective wavelength is wave_obs/(1+z) and the
    curve is T(wave_rest*(1+z), z), so the sample collapses to T at the FIXED observed
    instrument grid: tabulable against z at build time.

    Pinned via FLOPs: with the table in place the IGM-on and IGM-off spectrum kernels
    must compile the SAME work, because the transmission is no longer computed at all.
    """
    import numpy as np

    from tengri import SpectrumPrecomp
    from tengri.observation.spectroscopy import Spectroscopy

    ssp = pytest.importorskip("tengri").load_ssp()
    obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=jnp.asarray(np.linspace(4000.0, 9000.0, 2000)))
    )

    def build(**extra):
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(3.0),
            sfh={"type": "dpl", "*": FIXED},
            approx=SpectrumPrecomp(),
            **extra,
        )

    def flops(m):
        c = jax.jit(m.predict_spectrum).lower(_params(m)).compile().cost_analysis()
        return (c[0] if isinstance(c, list) else c)["flops"]

    on, off = flops(build()), flops(build(igm={"type": "none"}))
    # Applying the factor costs one multiply per pixel and nothing more; the
    # regressed path cost 6.65M FLOPs extra (a 5994-point Inoue+2014 evaluation).
    n_pix = 2000
    assert on - off < 10 * n_pix, (
        f"spectrum LUT with IGM compiles {on - off:,.0f} FLOPs more than without it "
        f"(budget {10 * n_pix:,} = a few ops per pixel) — the full-grid transmission "
        "is still being evaluated at runtime."
    )


@pytest.mark.regression_bug
def test_the_igm_is_still_applied_to_the_lut_spectrum():
    """...and the per-pixel factor must not have become a silent no-op either."""
    import numpy as np

    from tengri import SpectrumPrecomp
    from tengri.observation.spectroscopy import Spectroscopy

    ssp = pytest.importorskip("tengri").load_ssp()
    obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=jnp.asarray(np.linspace(4000.0, 9000.0, 2000)))
    )

    def build(**extra):
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(3.0),
            sfh={"type": "dpl", "*": FIXED},
            approx=SpectrumPrecomp(),
            **extra,
        )

    m, m0 = build(), build(igm={"type": "none"})
    p = _params(m)
    s = np.asarray(m.predict_spectrum(p))
    s0 = np.asarray(m0.predict_spectrum(p))

    # Blue pixels sit in the forest at z=3; the red end is untouched.
    assert s[0] / s0[0] < 0.9, "the bluest pixel is not being attenuated by the IGM"
    assert s[-1] / s0[-1] > 0.999


@pytest.mark.regression_bug
def test_free_redshift_band_factor_interpolation_stays_bounded():
    """Free z is the ONE case where the tabulated <T>_f is not exact — bound it.

    A fixed-z model gets a single node and is bit-identical. A free-z model
    interpolates <T>_f over a z-table, so it carries an interpolation error that main
    (which evaluated the transmission at the traced z) did not have. Every other test
    here pins z=3.0, so nothing covered this at all.

    The IGM factor is the steepest function of z in the model — the Lyman forest
    thickens fast — and the error grows toward z_max. Sharing the SSP ztable's coarse
    n_z put it at 3.0e-3 in sdss_u at z=3.98; the IGM table gets its own denser grid
    (_IGM_MIN_N_Z), which brings it to ~1e-4.

    Compares against the EXACT path, so this bounds the band factor's own error at low
    z where the LUT's band-average and Taylor-dust errors are still small. (Both of
    those explode in the rest-UV at high z -- the documented #617 blue bias -- which is
    why this checks the red bands at high z and all bands at low z.)
    """
    ssp = pytest.importorskip("tengri").load_ssp()
    obs = Observation(photometry=Photometry.from_names(FILTERS))

    def build(approx):
        from tengri import Uniform

        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Uniform(0.0, 4.0),
            sfh={"type": "dpl", "*": FIXED},
            dust_attenuation={"type": "single_component", "law": "calzetti", "*": FIXED},
            approx=approx,
        )

    lut, exact = build(WavePrecomp()), build(None)

    def at(m, z):
        p = _params(m)
        p["redshift"] = jnp.asarray(z)
        return np.asarray(m.predict_photometry(p))

    # Low z: the forest is thin, so every band is a fair test of the z-interpolation.
    for z in (0.5, 1.0, 2.0):
        dev = np.abs(at(lut, z) / at(exact, z) - 1).max()
        assert dev < 1e-2, (
            f"free-z WavePrecomp deviates by {dev:.3e} from the exact path at z={z}. "
            "The <T>_f z-table interpolation has regressed (or its grid was coarsened)."
        )

    # High z: the reddest band still samples the rest-optical, where the LUT's own
    # approximations are small, so it isolates the IGM interpolation.
    dev_red = abs(at(lut, 3.98)[-1] / at(exact, 3.98)[-1] - 1)
    assert dev_red < 1e-2, (
        f"free-z WavePrecomp deviates by {dev_red:.3e} in the reddest band at z=3.98."
    )


@pytest.mark.regression_bug
def test_dla_falls_back_to_the_exact_full_grid_path():
    """A DLA carries free parameters, so <T>_f is not a function of z alone.

    Those configs must NOT get a stale precomputed table — they fall back to the
    runtime band average and simply forfeit the speedup.
    """
    from tengri.components.igm.component import IGMSEDComponent

    model = _model(approx=WavePrecomp(), igm={"type": "inoue", "dla": True})
    igm = next(c for c in model._build_component_chain() if isinstance(c, IGMSEDComponent))

    assert igm._state is None or igm._state.band_table is None, (
        "A DLA model must not carry a precomputed IGM band table — the factor "
        "depends on dla_log_n_hi / dla_z, not on redshift alone."
    )
    flux = np.asarray(model.predict_photometry(_params(model)))
    assert np.all(np.isfinite(flux)) and np.all(flux > 0)
