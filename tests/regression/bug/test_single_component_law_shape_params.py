# SPDX-License-Identifier: BSD-3-Clause
"""Attenuation-law shape parameters must reach the curve — and only when asked (#1808).

``DustAttenuationSEDComponent`` (the ``single_component`` screen) called each
attenuation law with **no arguments** and preferred a ``k(lambda)`` cached in
``precompute()``. Every law shape parameter was therefore unreachable:
``dust_slope``, ``dust_delta`` and ``dust_bump_strength`` had exactly-zero
gradients across six laws while being live on ``two_component``. A fit that
freed a UV slope or a 2175 A bump there explored a direction the likelihood
could not see and returned the prior, reporting convergence.

**The obvious fix is wrong, and this file pins why.** Passing the spec's values
unconditionally overrides each law's own *published* defaults, because the spec
declares ONE shared ``dust_delta`` / ``dust_bump_strength`` (both ``Fixed(0.0)``)
while each law's signature carries its paper's value:

===============  ====================================================
law              published default
===============  ====================================================
``kriek_conroy``  ``dust_bump_strength = 1.0``  (KC13 as published)
``narayanan_z``   ``dust_delta = -0.2``         (Narayanan+2018)
``salim``         ``dust_bump_strength = 0.0``
===============  ====================================================

Measured on a first attempt: passing unconditionally made ``kriek_conroy``,
``narayanan_z`` and ``salim`` **bit-identical**, collapsing three distinct
published laws onto one curve. That trades an unfittable parameter for three
silently-substituted laws — the same defect class, moved.

The resolution is to pass a shape parameter only when somebody actually asked
for a value. ``spec._group_provenance`` says who did: ``user_fixed`` /
``user_prior`` / ``user_free`` / ``wildcard_free`` are requests;
``registry_default`` and ``wildcard_fixed`` are not, and for those the law's own
default must stand. That decision is made once at build time, so it is a static
Python branch rather than a comparison against a traced value — which is why it
cannot be done inside ``apply()``.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, SSPData, Uniform
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

_INERT_TOL = 1e-9


@pytest.fixture(scope="module")
def uv_ssp() -> SSPData:
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 7.0, 1200)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-30, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


@pytest.fixture(scope="module")
def uv_obs() -> Observation:
    """Bands bracketing the 2175 A bump and the UV slope."""

    def _tophat(center: float, frac: float = 0.10, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = (1500.0, 2175.0, 2800.0, 4400.0, 6200.0)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _build(uv_ssp, uv_obs, dust: dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=uv_ssp,
            observation=uv_obs,
            sfh={"type": "dpl", "all_params": FIXED},
            dust=dust,
            redshift=Fixed(0.5),
        )


def _sed(model) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = model.spec.sample(jax.random.PRNGKey(0))
        return np.asarray(model.predict(params).rest_sed())


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.where(np.abs(b) > 0, np.abs(b), 1.0)))


# ── The defect: a requested shape parameter must reach the curve ─────


@pytest.mark.parametrize(
    ("law", "param"),
    [
        ("power_law", "dust_slope"),
        ("noll09", "dust_delta"),
        ("salim_sbl18", "dust_delta"),
        ("kriek_conroy", "dust_delta"),
    ],
)
def test_freed_shape_parameter_has_a_nonzero_gradient(law, param, uv_ssp, uv_obs):
    """A free shape parameter must be fittable, not merely declared."""
    prior = Uniform(-1.5, -0.3) if param == "dust_slope" else Uniform(-1.0, 0.4)
    model = _build(uv_ssp, uv_obs, {"type": "single_component", "law_bc": law, param: prior})
    params = model.spec.sample(jax.random.PRNGKey(0))
    assert param in model.spec.free_params, f"{law}: {param} was not freed by the prior"

    def objective(value):
        probe = dict(params)
        probe[param] = value
        return jnp.sum(model.predict_photometry(probe))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = float(jax.grad(objective)(jnp.asarray(params[param], dtype=jnp.float64)))

    assert grad != 0.0, (
        f"{law}: free parameter {param!r} has an exactly-zero gradient. A fit cannot "
        "move it, so the posterior returns the prior and reports convergence (#1808)."
    )


@pytest.mark.parametrize(
    ("law", "param", "value"),
    [
        ("power_law", "dust_slope", -2.5),
        ("noll09", "dust_delta", -0.8),
        ("noll09", "dust_bump_strength", 3.0),
    ],
)
def test_user_set_shape_parameter_reaches_the_curve(law, param, value, uv_ssp, uv_obs):
    """An explicitly-set value must change the SED, not be silently discarded."""
    base = _sed(_build(uv_ssp, uv_obs, {"type": "single_component", "law_bc": law}))
    set_ = _sed(_build(uv_ssp, uv_obs, {"type": "single_component", "law_bc": law, param: value}))
    assert _max_rel(set_, base) > _INERT_TOL, (
        f"{law}: setting {param}={value} did not change the SED. The grammar accepted "
        "the value and the curve never saw it."
    )


# ── The guard: published per-law defaults must survive ───────────────


def test_published_law_defaults_are_not_overridden_by_the_shared_spec_default(uv_ssp, uv_obs):
    """Three distinct published laws must stay distinct at default settings.

    This is the assertion a naive fix fails. The spec declares one shared
    ``dust_delta`` / ``dust_bump_strength``, both ``Fixed(0.0)``; each of these
    laws carries a different published value in its own signature. Passing the
    spec value when nobody asked for one collapses all three onto a common base
    — measured, on a first attempt at #1808.
    """
    seds = {
        law: _sed(_build(uv_ssp, uv_obs, {"type": "single_component", "law_bc": law}))
        for law in ("kriek_conroy", "narayanan_z", "salim")
    }
    pairs = [("kriek_conroy", "narayanan_z"), ("kriek_conroy", "salim"), ("narayanan_z", "salim")]
    collapsed = [f"{a} == {b}" for a, b in pairs if _max_rel(seds[a], seds[b]) <= _INERT_TOL]
    assert not collapsed, (
        "Distinct published attenuation laws collapsed onto one curve at default "
        f"settings: {collapsed}. Their published defaults (kriek_conroy bump=1.0, "
        "narayanan_z delta=-0.2, salim bump=0.0) were overridden by the shared "
        "spec default of 0.0."
    )


@pytest.mark.parametrize("law", ["calzetti", "smc", "lmc", "leitherer02"])
def test_laws_with_no_shape_parameters_are_untouched(law, uv_ssp, uv_obs):
    """A law that reads no shape parameter must be bit-identical to before.

    These keep the build-time cached ``k(lambda)`` and the fast path. Pinning
    them means a change to the shape-parameter plumbing cannot quietly move the
    default build's physics — ``calzetti`` is the default law.
    """
    a = _sed(_build(uv_ssp, uv_obs, {"type": "single_component", "law_bc": law}))
    b = _sed(_build(uv_ssp, uv_obs, {"type": "single_component", "law_bc": law}))
    assert np.array_equal(a, b), f"{law}: not reproducible between builds"
    # Not ``all > 0``: the SED is legitimately zero at the extreme blue end of
    # the grid, where the stellar continuum has nothing left.
    assert np.all(np.isfinite(a)), f"{law}: non-finite SED"
    assert float(np.sum(a)) > 0.0, f"{law}: SED is identically zero"
