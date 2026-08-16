"""The batched lognormal-MDF kernel must agree with DSPS's per-bin route.

``_lgmet_weights_parcels`` duplicates DSPS's formula in order to evaluate the
triweight CDF once per bin *edge* instead of twice (once as some bin's upper
edge, once as the next bin's lower edge). That is worth 1.48x the FLOPs of the
joint (met, age) weight builder, but only while the two routes stay the same
function. Nothing in the type system says they do, and ``_tw_cuml_kern`` /
``_get_bin_edges`` are private DSPS symbols that can be renamed or reworked
without notice.

These tests are that guard.

**Why parity is asserted at 1e-15 and not bit-for-bit.** The two routes feed
the identical expression identical operands, but at different *shapes*: DSPS
evaluates ``_tw_cuml_kern`` on scalars under ``vmap``, this kernel evaluates it
on an ``(n_parcel, n_met + 1)`` broadcast. XLA contracts the degree-7
polynomial into fused multiply-adds differently for the two shapes, which moves
the last bit. Measured across the production shape: 95.7 % of elements
bit-identical, worst absolute disagreement 3.3e-16 on weights bounded by 1 --
about 1.5 ULP. Anything larger than that is a real divergence, not rounding.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.component import (
    _lgmet_weights,
    _lgmet_weights_parcels,
)

pytestmark = pytest.mark.regression_bug

#: Ceiling on the FMA-contraction disagreement described in the module
#: docstring. Weights are bounded by 1, so this is an absolute tolerance.
ULP_TOL = 1e-15

#: The real ProGeny/MILES metallicity axis spans -4.0 .. -1.22 over 15 nodes.
#: Test grids stay in that neighborhood and stay near-evenly spaced: DSPS
#: builds bin edges as ``bin_mids - dbins / 2``, which is only monotone for
#: reasonably regular grids. A wildly uneven grid makes the edges cross and
#: *both* routes then return negative bin masses -- identically, but the
#: pathology has nothing to say about this kernel.
LGMET_LO, LGMET_HI = -4.0, -1.22


def _realistic_grid(n_met, rng, jitter=0.25):
    """A monotone, near-evenly-spaced metallicity axis of ``n_met`` nodes."""
    base = np.linspace(LGMET_LO, LGMET_HI, n_met)
    step = (LGMET_HI - LGMET_LO) / max(n_met - 1, 1)
    wobble = rng.uniform(-jitter, jitter, n_met) * step
    wobble[0] = wobble[-1] = 0.0
    return jnp.asarray(np.sort(base + wobble))


def _reference(log_z, scatter, ssp_lgmet):
    """DSPS's own route, vmapped over parcels.

    Calls upstream directly rather than through :func:`_lgmet_weights`, which
    now delegates to the kernel under test -- comparing them would be circular.
    """
    from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf

    return jax.vmap(lambda g: calc_lgmet_weights_from_lognormal_mdf(g, scatter, ssp_lgmet))(log_z)


def test_scalar_face_delegates_to_the_batched_kernel():
    """``_lgmet_weights`` is the same kernel, so the two cannot drift apart."""
    ssp_lgmet = jnp.linspace(LGMET_LO, LGMET_HI, 15)
    x = -2.4

    scalar = _lgmet_weights(x, 0.2, ssp_lgmet)
    batched = _lgmet_weights_parcels(jnp.asarray([x]), 0.2, ssp_lgmet)[0]

    assert scalar.shape == (15,)
    np.testing.assert_array_equal(np.asarray(scalar), np.asarray(batched))
    np.testing.assert_allclose(
        np.asarray(scalar),
        np.asarray(_reference(jnp.asarray([x]), 0.2, ssp_lgmet)[0]),
        rtol=0.0,
        atol=ULP_TOL,
    )


@pytest.mark.parametrize("n_met", [3, 7, 15, 22])
@pytest.mark.parametrize("scatter", [0.02, 0.2, 1.0])
def test_matches_dsps_per_bin_route(n_met, scatter):
    """Same weights as vmapping DSPS's kernel, across grid sizes and MDF widths."""
    rng = np.random.default_rng(n_met * 100 + int(scatter * 1000))
    ssp_lgmet = _realistic_grid(n_met, rng)
    log_z = jnp.asarray(rng.uniform(-5.0, -0.5, 512))

    got = _lgmet_weights_parcels(log_z, scatter, ssp_lgmet)
    want = _reference(log_z, scatter, ssp_lgmet)

    assert got.shape == want.shape == (512, n_met)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0.0, atol=ULP_TOL)


def test_production_shape_agreement_is_machine_precision():
    """On the shape the forward model actually uses, and on the real axis.

    The measured configuration of the tabulated-metallicity path: the
    ProGeny/MILES 15-node metallicity axis, and the 1682-parcel dense CIC
    integrand ``(93 - 1) * 16 + 1`` plus the #1522 old-age tail.
    """
    rng = np.random.default_rng(0)
    ssp_lgmet = jnp.asarray(np.linspace(-4.0, -1.2218487496163564, 15))
    log_z = jnp.asarray(rng.uniform(-3.9, -1.3, 1682))

    got = np.asarray(_lgmet_weights_parcels(log_z, 0.2, ssp_lgmet))
    want = np.asarray(_reference(log_z, 0.2, ssp_lgmet))

    worst = np.abs(got - want).max()
    assert worst <= ULP_TOL, (
        f"max |diff| {worst:.3e} exceeds {ULP_TOL:.0e}; "
        f"{100 * (got == want).mean():.4f}% of elements bit-identical"
    )


@pytest.mark.parametrize(
    "log_z",
    [
        pytest.param(-50.0, id="far_below_grid"),
        pytest.param(50.0, id="far_above_grid"),
        pytest.param(-4.0, id="on_lowest_node"),
        pytest.param(-1.22, id="on_highest_node"),
    ],
)
def test_off_grid_parcels_match(log_z):
    """The empty-weight fill (parcel entirely outside the grid) matches too.

    This is the branch ``_fill_empty_weights_singlepoint`` exists for, and the
    one a vectorized rewrite is most likely to get wrong: it is reached only
    when every bin weight underflows to exactly zero.
    """
    ssp_lgmet = jnp.linspace(LGMET_LO, LGMET_HI, 15)
    x = jnp.asarray([log_z])

    got = _lgmet_weights_parcels(x, 0.2, ssp_lgmet)
    want = _reference(x, 0.2, ssp_lgmet)

    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_allclose(float(jnp.sum(got)), 1.0, rtol=1e-12)


def test_weights_are_normalized_and_non_negative():
    """Physical contract on a well-formed grid, independent of DSPS."""
    rng = np.random.default_rng(7)
    ssp_lgmet = _realistic_grid(15, rng)
    log_z = jnp.asarray(rng.uniform(-6.0, 0.0, 256))

    w = _lgmet_weights_parcels(log_z, 0.25, ssp_lgmet)

    assert bool(jnp.all(w >= 0.0)), f"negative MDF weight: min {float(jnp.min(w)):.3e}"
    np.testing.assert_allclose(np.asarray(jnp.sum(w, axis=-1)), 1.0, rtol=1e-12)


def test_degenerate_grid_pathology_is_shared_with_dsps():
    """A pathological axis breaks both routes identically, not just this one.

    ``_get_bin_edges`` builds lower edges as ``bin_mids - dbins / 2``, which
    crosses when the axis is wildly uneven; the resulting bin masses go
    negative. Pinning that the two routes agree *there too* keeps a future
    reader from mistaking a DSPS quirk for a defect introduced here.
    """
    rng = np.random.default_rng(0)
    ssp_lgmet = jnp.asarray(np.sort(rng.uniform(-4.5, -1.3, 15)))
    log_z = jnp.asarray(rng.uniform(-4.0, -1.5, 512))

    got = np.asarray(_lgmet_weights_parcels(log_z, 0.2, ssp_lgmet))
    want = np.asarray(_reference(log_z, 0.2, ssp_lgmet))

    assert (want < 0).any(), "expected the reference to expose the edge-crossing pathology"
    np.testing.assert_allclose(got, want, rtol=0.0, atol=ULP_TOL)


def test_gradients_flow_and_are_finite():
    """The kernel sits inside the fit hot path, so it must differentiate."""
    ssp_lgmet = jnp.linspace(LGMET_LO, LGMET_HI, 15)

    def loss(log_z):
        return jnp.sum(_lgmet_weights_parcels(log_z, 0.2, ssp_lgmet) ** 2)

    g = jax.grad(loss)(jnp.linspace(-3.9, -1.3, 64))

    assert g.shape == (64,)
    assert bool(jnp.all(jnp.isfinite(g))), "non-finite gradient through the MDF"
    assert float(jnp.max(jnp.abs(g))) > 0.0, "gradient vanished entirely"


def test_float32_emits_no_mixed_dtype_scatter():
    """The float32 contract of #1448 survives the rewrite.

    The SSP metallicity axis is a cached host array and stays float64 even with
    x64 off, while parameters arrive as float32. Handing both to a DSPS kernel
    is what raises ``scatter inputs have incompatible types``; the operands are
    canonicalized first precisely to stop that. A rewrite that quietly dropped
    the canonicalization would still pass every float64 test here.
    """
    ssp_lgmet_f64 = np.linspace(-4.0, -1.2218487496163564, 15)  # cached, float64
    log_z = np.random.default_rng(0).uniform(-3.9, -1.3, 256)

    with jax.enable_x64(False):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            w = _lgmet_weights_parcels(
                jnp.asarray(log_z, dtype=jnp.float32),
                jnp.asarray(0.2, dtype=jnp.float32),
                jnp.asarray(ssp_lgmet_f64),
            )
            jax.block_until_ready(w)

        incompatible = [c for c in caught if "incompatible types" in str(c.message)]
        assert not incompatible, (
            "a DSPS kernel is being handed a cached float64 grid beside float32 "
            f"operands: {[str(c.message)[:120] for c in incompatible]}"
        )
        assert w.dtype == jnp.float32, f"float32 in, {w.dtype} out"
        assert bool(jnp.all(jnp.isfinite(w))), "non-finite float32 MDF weights"
        np.testing.assert_allclose(np.asarray(jnp.sum(w, axis=-1)), 1.0, atol=1e-6)


def test_float32_tracks_dsps_to_float32_epsilon():
    """In float32 the two routes still agree, at float32 precision.

    Also pins that the CDF-difference formulation's tiny negative weights are
    DSPS's behavior and not this kernel's: measured on this fixture, DSPS emits
    8 of them and this kernel 5, both around -6e-08. Clamping here would break
    parity with upstream, so the count is bounded rather than forbidden.
    """
    from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf

    ssp_lgmet_f64 = np.linspace(-4.0, -1.2218487496163564, 15)
    log_z = np.random.default_rng(0).uniform(-3.9, -1.3, 1682)

    with jax.enable_x64(False):
        ssp = jnp.asarray(ssp_lgmet_f64)
        lz = jnp.asarray(log_z, dtype=jnp.float32)
        sig = jnp.asarray(0.2, dtype=jnp.float32)

        got = np.asarray(_lgmet_weights_parcels(lz, sig, ssp))
        want = np.asarray(
            jax.vmap(lambda g: calc_lgmet_weights_from_lognormal_mdf(g, sig, ssp))(lz)
        )

    # float32 eps is 1.2e-07; weights are bounded by 1.
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-6)
    assert got.min() > -1e-6, f"float32 MDF weight too negative: {got.min():.3e}"
    assert (got < 0).sum() <= (want < 0).sum(), (
        f"this kernel produced more negative weights than DSPS: "
        f"{int((got < 0).sum())} vs {int((want < 0).sum())}"
    )
