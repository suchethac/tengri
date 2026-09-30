# SPDX-License-Identifier: BSD-3-Clause
r"""Masked grid interpolation must normalize over the populated cells (#2435).

``shock_line_ratios`` is documented to return line ratios **relative to Hb**, so
``Hb_4861A`` must come back as exactly 1.0. It returned **0.7135**.

``components/nebular/shock.py`` zeroed unpopulated cells and passed the mask to
``interp_nd_triweight``, whose ``_tensor_contract`` documented the argument as
"accepted for API compatibility but currently unused" and contracted with triweight
weights that still summed to one over the **whole** axis. Interpolating a field that
is 1.0 on populated cells and 0.0 elsewhere therefore returned the fraction of kernel
weight landing on populated cells -- and that fraction was being published as Hb.

Zeroing is not masking. Zeroing without renormalizing the weights is *dilution*, and
with only 3992 of 38850 cells (10.3%) populated it was the normal case, not an edge:

=========================================  ===========  ==============================
point (v = 400 km/s, solar, combined)      Hb returned  every line was low by
=========================================  ===========  ==============================
B = 1 uG, log n = 0                        0.7135       28.7%
B = 100 uG, log n = 0                      0.1132       88.7%
B = 142 uG, log n = 0                      0.00375      99.6%  (a factor of 267)
B = 100 uG, log n = -1                     -0.0         100%: every line silently zero
=========================================  ===========  ==============================

The grid itself was never wrong: ``mappings5/combined_ratios`` stores ``Hb_4861A``
identically 1.0 at all 3992 populated cells, with true Ha/Hb from 2.8923 to 4.8521
(median 2.9985) -- every cell above Case B.

Because the factor was common to all lines it **canceled** in line-to-line ratios,
which is why so little caught it: ``_shock_line_arrays`` anchors on Ha
(``L_line = (ratio / r_ha) * l_shock_halpha``), so fitted SEDs were unaffected. The
one check that did notice was
``tests/crossval/test_nebular_physics.py::TestShockLineRatioPhysics::test_halpha_hbeta_above_case_b``
-- red bit-identically (2.1432161950153708) every night from 2026-09-14 -- and
``tests/crossval/`` runs in no gate, so nothing reported it.

A Balmer decrement cannot sit below Case B: recombination alone floors Ha/Hb at 2.86
at 1e4 K and collisional excitation in shocks only raises it. 2.14 was not a
calibration disagreement, it was a value no shock model can produce.

The guard here is the invariant that was missing: **Hb_4861A == 1.0**. One line, at
the source, instead of three layers downstream in a suite nothing runs.

This does **not** fix #2066 (B-field flat at off-node points). Measured across 17
usable off-node B points, flat-before was 0 and flat-after 2; where flatness appears
it is because a single populated node is in range, which no normalization can undo.
The ``xfail(strict=True)`` for #2066 in
``tests/components/nebular/test_shock_interpolation.py`` still xfails.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("h5py", reason="h5py required for MAPPINGS grid tests")

pytestmark = pytest.mark.regression_bug

from tengri.components.nebular.shock import _load_mappings_grids, shock_line_ratios

#: Populated (B [uG], log10 density [cm^-3]) pairs, spanning the range over which the
#: dilution factor ran from 0.88 down to 0.0037.
_POPULATED = ((0.01, 0.0), (1.0, 0.0), (1.0, 2.0), (10.0, 1.0), (100.0, 0.0))

#: A (B, log n) pair with no populated cell in kernel range. Before the fix every line
#: here came back as -0.0; the honest answer is "no data", not "zero emission".
_UNPOPULATED = (100.0, -1.0)

_VELOCITIES = (200.0, 300.0, 500.0)

#: Case B recombination at 1e4 K. Shocks add collisional excitation, so the true
#: value only ever sits above this.
_CASE_B_HA_HB = 2.86
_CASE_B_HG_HB = 0.466


@pytest.fixture(scope="module")
def mappings_grid():
    """The real MAPPINGS V grid, or skip.

    Every assertion below is vacuous on the hardcoded Allen+2008 fallback, which
    returns ``Hb_4861A`` as a literal ``jnp.array(1.0)`` and would satisfy the central
    invariant without exercising the interpolator at all. The skip is narrow on
    purpose (``tools/check_test_skip_handlers.py``): a missing data file is the only
    thing it may excuse.
    """
    try:
        grids = _load_mappings_grids()
    except FileNotFoundError:
        pytest.skip("data/mappings_templates.h5 not present")
    if grids is None or "mappings5" not in grids:
        pytest.skip("MAPPINGS V HDF5 grid unavailable; the fallback path cannot test this")
    return grids["mappings5"]


def test_grid_really_stores_hbeta_as_one(mappings_grid):
    """Anti-vacuity: the invariant below is only meaningful if the table asserts it.

    If the stored Hb were not 1.0, ``Hb_4861A == 1.0`` downstream would be testing the
    interpolator against the wrong reference.
    """
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in mappings_grid["line_names"]]
    ratios = np.asarray(mappings_grid["combined_ratios"])
    hb = ratios[..., names.index("Hb_4861A")]
    populated = hb > 0.0
    assert populated.sum() > 0, "no populated cells at all — fixture is wrong"
    np.testing.assert_allclose(hb[populated], 1.0, rtol=0, atol=0)


@pytest.mark.parametrize("velocity", _VELOCITIES)
@pytest.mark.parametrize(("b_field", "log_density"), _POPULATED)
def test_hbeta_is_unity(mappings_grid, velocity, b_field, log_density):
    """The missing invariant. Hb-normalized means Hb == 1, everywhere it has data."""
    ratios = shock_line_ratios(
        velocity, shock_log_density=log_density, shock_b_over_sqrt_n=b_field
    )
    assert float(ratios["Hb_4861A"]) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("velocity", _VELOCITIES)
def test_balmer_decrement_clears_case_b(mappings_grid, velocity):
    """Ha/Hb > 2.86 -- the crossval assertion that was red every night since 09-14."""
    ratios = shock_line_ratios(velocity)
    ha_hb = float(ratios["HA_6563A"]) / float(ratios["Hb_4861A"])
    assert ha_hb > _CASE_B_HA_HB, (
        f"Ha/Hb = {ha_hb:.4f} at v={velocity} km/s is below Case B ({_CASE_B_HA_HB}); "
        "no shock model can produce a decrement below the recombination floor"
    )


@pytest.mark.parametrize("velocity", _VELOCITIES)
def test_line_to_line_ratios_are_unmoved(mappings_grid, velocity):
    """The fix changes amplitude, not shape.

    The defect scaled every line by one common factor, so line-to-line ratios were
    already correct and must stay correct -- a change here would mean the fix did
    something beyond removing the scale. Hg/Hb is checked against Case B (0.466)
    rather than a captured number so the assertion states physics rather than
    whatever the code currently emits; 3% accommodates the shock's collisional
    contribution and the grid's own interpolation, and is far tighter than the 41%
    the dilution moved it.
    """
    ratios = shock_line_ratios(velocity)
    hg_hb = float(ratios["Hg_4341A"]) / float(ratios["Hb_4861A"])
    assert hg_hb == pytest.approx(_CASE_B_HG_HB, rel=0.03)


def test_unpopulated_region_returns_nan_not_zero(mappings_grid):
    """No populated cell in range is "no data", and must say so.

    Returning zeros is a silent wrong answer: it is indistinguishable from a real
    prediction of no line emission, and it made ``ratio / r_ha`` in
    ``_shock_line_arrays`` a 0/0.
    """
    b_field, log_density = _UNPOPULATED
    ratios = shock_line_ratios(400.0, shock_log_density=log_density, shock_b_over_sqrt_n=b_field)
    values = jnp.stack([jnp.asarray(v) for v in ratios.values()])
    assert bool(jnp.all(jnp.isnan(values))), (
        "an entirely unpopulated query must be NaN; zeros read as a physical "
        "prediction of no emission"
    )


def test_gradient_is_finite_and_live_at_populated_points(mappings_grid):
    """The double-``where`` guard, asserted on both failure modes it has.

    A single ``where`` around ``num / W`` still evaluates the division in the dead
    branch and poisons the reverse-mode gradient with NaN even where W > 0. The
    opposite failure is just as real here and finiteness cannot see it: division by
    an interpolated denominator is exactly the kind of rewrite that can detach a
    gradient, and an identically-zero gradient is finite. Both halves are asserted
    (#2100 shipped a zero float32 photometry gradient past a finite-only check).
    """

    def o3_over_hb(b):
        ratios = shock_line_ratios(400.0, shock_log_density=0.0, shock_b_over_sqrt_n=b)
        return ratios["O3_5007A"] / ratios["Hb_4861A"]

    for b_field in (0.5, 1.0, 5.0, 50.0):
        grad = float(jax.grad(o3_over_hb)(jnp.asarray(b_field)))
        assert np.isfinite(grad), (
            f"gradient at B={b_field} uG is {grad}: the W == 0 branch leaked NaN into "
            "the backward pass, which is what the double-`where` exists to prevent"
        )
        assert grad != 0.0, (
            f"gradient at B={b_field} uG is identically zero, so the normalization "
            "detached it — [OIII]/Hb does vary with B here and must stay differentiable"
        )


def test_unmasked_interpolation_is_untouched():
    """``population_mask=None`` must contract exactly as before.

    ``components/agn/disc.py`` calls ``interp_nd_triweight`` without a mask, so the
    normalization must be entirely inert on that path.
    """
    from tengri.utils.grid_interp import edges_for_grid, interp_nd_triweight

    rng = np.random.default_rng(2435)
    axes = (jnp.linspace(0.0, 1.0, 5), jnp.linspace(0.0, 1.0, 4), jnp.linspace(0.0, 1.0, 3))
    edges = tuple(edges_for_grid(a) for a in axes)
    grid = jnp.asarray(rng.normal(size=(5, 4, 3, 7)))

    out = interp_nd_triweight(grid, axes, edges, (0.37, 0.61, 0.42))

    # A constant field interpolates to that constant when the weights partition unity,
    # which is the property the masked path has to reproduce by division.
    const = jnp.full((5, 4, 3, 1), 3.5)
    out_const = interp_nd_triweight(const, axes, edges, (0.37, 0.61, 0.42))

    assert out.shape == (7,)
    assert bool(jnp.all(jnp.isfinite(out)))
    np.testing.assert_allclose(float(out_const[0]), 3.5, rtol=1e-6)
