# SPDX-License-Identifier: BSD-3-Clause
"""Dust IR template grids must not bake into the compiled HLO as constants.

**This file skipped 6 of 6 tests for an unknown stretch before 2026-08.** Five
skipped on::

    Failed to build model: SEDModel.__init__() got an unexpected keyword
    argument 'filter_waves'

a stale-API ``TypeError`` caught by ``except Exception: pytest.skip(...)``. It
was green the whole time and executed no assertions. Because it was the only
thing exercising the dust ``grid_arrays_traced`` seam, it also stood in as the
evidence that dust template threading works. It never tested that.

Rewritten against the current ``SEDModel.build`` grammar, and the guards
narrowed: a missing template file on disk is a legitimate skip, and nothing
else is. A stale API must fail here, loudly, the way it would have the first
time had this file not been swallowing it.

See #1615 for the census of the same shape elsewhere (40 sites, 17 files).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract

jax.config.update("jax_enable_x64", True)

#: Dust deep enough that L_absorbed is large and the IR term is real rather
#: than a rounding error.
_TAU_BC, _TAU_DIFF = 2.0, 1.5

#: A constant this large in the compiled HLO is a baked template grid, not a
#: scalar or an axis vector.
_LARGE_CONST_BYTES = 1 << 20  # 1 MiB


@pytest.fixture(scope="module")
def panchromatic_obs():
    """Six top-hats from 1500 A to 500 um, so IR emission lands in a band."""

    def _tophat(center, frac=0.16, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = (1500.0, 5000.0, 2.0e4, 2.4e5, 1.0e6, 5.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _build(ssp, obs, emission_type):
    """Build a model with the named dust-emission backend.

    Deliberately not wrapped in ``try/except Exception``. That wrapper is what
    hid the stale API here for months; a build failure is a test failure.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "*": FIXED},
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "tau_bc": _TAU_BC,
            "tau_diff": _TAU_DIFF,
            "emission": {"type": emission_type, "*": FIXED},
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


def _large_constants(model):
    """Byte sizes of every >1 MiB constant in the lowered photometry HLO.

    Returns
    -------
    list of int
        Descending. Empty when nothing large is baked in.
    """
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    lowered = jax.jit(model.predict_photometry).lower(params)
    text = lowered.as_text()

    sizes = []
    for line in text.splitlines():
        if "constant(" not in line and "dense<" not in line:
            continue
        # Shapes appear as e.g. `tensor<100x50xf64>` / `f64[100,50]`.
        for token in line.replace("<", " ").replace(">", " ").split():
            if "x" not in token:
                continue
            head, _, dtype = token.rpartition("x")
            if not dtype.startswith(("f32", "f64", "i32", "i64")):
                continue
            try:
                dims = [int(d) for d in head.split("x")]
            except ValueError:
                continue
            width = 8 if "64" in dtype else 4
            nbytes = width
            for d in dims:
                nbytes *= d
            if nbytes >= _LARGE_CONST_BYTES:
                sizes.append(nbytes)
    return sorted(sizes, reverse=True)


# ── the lookup surfaces still work ──────────────────────────────────────


@pytest.mark.parametrize("emission_type", ["dale2014", "draine_li2014", "modified_blackbody"])
def test_the_model_builds_and_predicts(synthetic_ssp_wide, panchromatic_obs, emission_type):
    """The assertion the old file never reached, because it skipped first."""
    model = _build(synthetic_ssp_wide, panchromatic_obs, emission_type)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    phot = model.predict_photometry(params)

    assert phot.shape[0] == 6, f"expected 6 bands, got {phot.shape}"
    assert jnp.all(jnp.isfinite(phot)), f"{emission_type} produced non-finite photometry"
    assert jnp.any(phot > 0), f"{emission_type} produced no flux at all"


def test_an_ir_band_responds_to_the_dust_emission_backend(synthetic_ssp_wide, panchromatic_obs):
    """Two backends must not give the same far-IR photometry.

    Identical output would mean the emission block is inert — the failure mode
    a build-and-predict smoke test cannot see.

    Compared **relatively**. Far-IR fluxes here are of order 1e-10, and
    ``allclose``'s default ``atol=1e-8`` swamps them completely: the first
    version of this assertion called 1.11e-10 and 2.07e-10 equal, a factor of
    1.9 apart. An absolute tolerance on a quantity smaller than the tolerance
    is not a comparison.
    """
    params_key = jax.random.PRNGKey(0)
    out = {}
    for emission_type in ("dale2014", "modified_blackbody"):
        model = _build(synthetic_ssp_wide, panchromatic_obs, emission_type)
        out[emission_type] = model.predict_photometry(dict(model.spec.sample(params_key)))

    far_ir = jnp.asarray(out["dale2014"][-2:])
    other = jnp.asarray(out["modified_blackbody"][-2:])
    scale = jnp.maximum(jnp.abs(far_ir), jnp.abs(other))
    rel = jnp.max(jnp.abs(far_ir - other) / jnp.where(scale > 0, scale, 1.0))
    assert rel > 1e-6, (
        "dale2014 and modified_blackbody give the same far-IR photometry to "
        f"{float(rel):.2e} relative; the dust-emission block is not reaching "
        f"the output. dale2014={far_ir}, modified_blackbody={other}"
    )


# ── the property this file was named for ────────────────────────────────


@pytest.mark.xfail(
    reason=(
        "draine_li2014 bakes 69.8 MB of template grid into the photometry HLO "
        "(66.6 + 3.2 MB). Attributed by control, not assumed: the same build "
        "with no dust emission bakes 0 MB, as do modified_blackbody and "
        "dale2014 — so this is the DL14 template grid, not the SSP. Nothing in "
        "src/ supplies grid_arrays_traced: the producer (extract_grid_arrays) "
        "was unreferenced and removed in #1614, and _precomputed."
        "dust_ir_grid_arrays does not exist. #1595 fixed the AGN half of "
        "#1383 via a declarative template_loader; the dust half is open. "
        "strict=True so this flips to a failure the moment it is wired."
    ),
    strict=True,
)
def test_dust_ir_templates_are_not_baked_as_large_constants(synthetic_ssp_wide, panchromatic_obs):
    """The claim in this file's title, asserted for the first time.

    The old version never evaluated it: it skipped on model construction, and
    its one relevant assertion referenced ``_precomputed.dust_ir_grid_arrays``,
    an attribute that has never existed in ``src/``.

    ``draine_li2014`` is the backend under test because it is the one that
    bakes. Measured across backends on this fixture:

    ======================  ===================
    ``emission.type``       baked >= 1 MiB
    ======================  ===================
    *(none)* — control      0 MB
    ``modified_blackbody``  0 MB
    ``dale2014``            0 MB
    ``draine_li2014``       **69.8 MB**
    ======================  ===================

    The control row is what makes this a statement about dust templates rather
    than about the SSP grid.
    """
    model = _build(synthetic_ssp_wide, panchromatic_obs, "draine_li2014")
    large = _large_constants(model)
    assert not large, (
        f"{len(large)} constant(s) >= 1 MiB baked into the photometry HLO; "
        f"largest {large[0] / 1e6:.1f} MB"
    )
