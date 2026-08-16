# SPDX-License-Identifier: BSD-3-Clause
r"""``two_component`` must not overwrite a dust law's published defaults (#1833).

At ``tau_bc = 0`` the two dust geometries are the same screen, so selecting a
law under ``dust={'type': 'two_component'}`` must give the same curve as
selecting it under ``single_component``. Measured on ``48d4b1efb``, nineteen of
twenty-two laws agreed to ~4e-16 and three did not:

===============  ==============
law              max rel diff
===============  ==============
``kriek_conroy``  **1.281372**
``narayanan_z``   **1.000000**
``tea``           **1.000000**
nineteen others   <= 5.6e-16
===============  ==============

The three are exactly the laws carrying a non-zero shape default in their own
signature (``kriek_conroy`` ``dust_bump_strength=1.0``; ``narayanan_z`` and
``tea`` ``dust_delta=-0.2``). ``resolve_bc_diff_law_params`` injected all four
shared law kwargs unconditionally, and the spec declares one shared
``dust_bump_strength`` / ``dust_delta``, both ``Fixed(0.0)`` -- so the paper
value was overwritten with zero on every build.

For ``kriek_conroy`` that removes the 2175 Å Drude bump entirely. Kriek &
Conroy (2013), ApJ 775, L16, Eqn 3:

.. math::

    k(\lambda) = \frac{A_\lambda}{A_V}
        = \frac{k_{\rm Cal}(\lambda)\,R_V + D(\lambda; \lambda_0, E_b)}{R_V}
          \left(\frac{\lambda}{5500\,\text{Å}}\right)^{\delta}

with :math:`D` the Drude profile at :math:`\lambda_0 = 2175` Å and
:math:`E_b` its amplitude [dimensionless]. ``dust_bump_strength`` multiplies
:math:`E_b`, so 0.0 deletes the term the law exists to add. A user selecting
Kriek & Conroy got Calzetti with a tilt.

This is the defect #1808 declined to introduce on ``single_component`` -- it
implemented "pass the spec's values unconditionally", measured that it
"collapsed three distinct published laws onto one curve", and rejected it. The
same three laws. The fix here is #1808's provenance rule applied to the second
caller: ``user_prior`` / ``user_fixed`` / ``user_free`` / ``wildcard_free`` are
requests and get passed; ``registry_default`` / ``wildcard_fixed`` are not, and
the law's own default stands.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, SSPData
from tengri.components.dust.laws._registry import list_laws
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

#: The repo's inertness threshold. Agreement below this is "the same curve".
_INERT_TOL = 1e-9

_TAU = 1.0

#: Laws whose signature carries a non-zero shape default -- the ones #1833 broke.
_PAPER_DEFAULT_LAWS = ("kriek_conroy", "narayanan_z", "tea")


def _all_laws() -> tuple[str, ...]:
    """Every registered law, from the live registry rather than a frozen list.

    #1482 and #1833 were both missed by suites carrying a hardcoded menu, so
    a law registered later is covered here without editing this file.
    """
    return tuple(sorted(row["name"] for row in list_laws(headline=False)))


@pytest.fixture(scope="module")
def dust_ssp() -> SSPData:
    """SSP with a Lyman break, so the UV is not dominated by an unphysical tail."""
    ages = jnp.linspace(-3.0, 1.14, 20)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 7.0, 700)
    base = ((5000.0 / wave) ** 2 * jnp.where(wave < 912.0, 1e-6, 1.0))[None, None, :]
    flux = (
        base
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-12, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


@pytest.fixture(scope="module")
def uv_obs() -> Observation:
    """Bands straddling the 2175 Å bump, where these laws differ."""

    def _tophat(center: float, n: int = 24) -> FilterCurve:
        wave = jnp.linspace(center * 0.8, center * 1.2, n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{center:.4g}")

    return Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (1500.0, 2800.0, 3500.0, 6200.0)))
    )


def _build(dust_ssp, uv_obs, dust: dict) -> SEDModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=dust_ssp,
            observation=uv_obs,
            sfh={"type": "dpl", "all_params": FIXED},
            dust=dust,
            redshift=Fixed(0.5),
        )


def _sed(model: SEDModel) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        return np.asarray(model.predict(params).rest_sed())


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.where(np.abs(b) > 0, np.abs(b), 1.0)))


def _single(law: str, **extra) -> dict:
    return {"type": "single_component", "law_bc": law, "tau_v": Fixed(_TAU), **extra}


def _two(law: str, **extra) -> dict:
    return {
        "type": "two_component",
        "law_bc": law,
        "law_diff": law,
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(_TAU),
        **extra,
    }


def test_the_fixture_can_see_dust_at_all(dust_ssp, uv_obs):
    """Live control: without it, every agreement below is vacuous.

    An all-inert result indicts the fixture; an inert result *beside a live
    one* indicts the scope. This is the live one.
    """
    off = _sed(_build(dust_ssp, uv_obs, _single("calzetti", tau_v=Fixed(0.0))))
    on = _sed(_build(dust_ssp, uv_obs, _single("calzetti")))
    moved = _rel(on, off)
    assert moved > 1e-3, (
        f"calzetti tau_v 0 -> {_TAU} moved the SED by only {moved:.3e}. Dust is not "
        "live on this fixture, so no agreement measured here means anything."
    )


@pytest.mark.parametrize("law", _all_laws())
def test_both_dust_geometries_agree_on_every_law(law, dust_ssp, uv_obs):
    """``single_component(tau_v=t)`` == ``two_component(tau_bc=0, tau_diff=t)``.

    At ``tau_bc = 0`` the birth-cloud screen contributes nothing, so the two
    are the same model and the same law must give the same curve. Which of the
    two is *right* is asserted separately below -- this only pins that one name
    cannot mean two curves.
    """
    a = _sed(_build(dust_ssp, uv_obs, _single(law)))
    b = _sed(_build(dust_ssp, uv_obs, _two(law)))
    diff = _rel(a, b)
    assert diff <= _INERT_TOL, (
        f"'{law}' gives different curves on the two dust geometries: max rel {diff:.6e}. "
        "At tau_bc=0 they are the same screen, so one of them is not the published "
        "law. #1833: two_component overwrote the law's own defaults with the shared "
        "spec Fixed(0.0)."
    )


@pytest.mark.parametrize("law", _PAPER_DEFAULT_LAWS)
def test_two_component_honors_the_published_default(law, dust_ssp, uv_obs):
    """The agreement must be at the law's value, not at the spec's zero.

    Both geometries agreeing is not enough -- they would also agree if the fix
    had pushed ``single_component`` down to ``0.0``. Stating the paper value
    explicitly must be a *no-op*, which is only true if it was already in use.
    """
    default_sed = _sed(_build(dust_ssp, uv_obs, _two(law)))
    explicit = {"kriek_conroy": {"bump_strength": Fixed(1.0)}}.get(law, {"delta": Fixed(-0.2)})
    stated_sed = _sed(_build(dust_ssp, uv_obs, _two(law, **explicit)))
    moved = _rel(stated_sed, default_sed)
    assert moved <= _INERT_TOL, (
        f"'{law}': stating its own published default {explicit} changed the SED by "
        f"{moved:.6e}, so the default in use is not the published one (#1833)."
    )


def test_kriek_conroy_keeps_its_2175_angstrom_bump(dust_ssp, uv_obs):
    """The physics #1833 deleted, asserted as physics rather than as agreement.

    ``dust_bump_strength=0`` is documented as removing the Drude term (KC13
    Eqn 3), so a build at the law's own default must differ from an explicit
    zero. If it does not, the bump is gone and the law is Calzetti with a tilt.
    """
    with_bump = _sed(_build(dust_ssp, uv_obs, _two("kriek_conroy")))
    no_bump = _sed(_build(dust_ssp, uv_obs, _two("kriek_conroy", bump_strength=Fixed(0.0))))
    moved = _rel(with_bump, no_bump)
    assert moved > 1e-3, (
        "kriek_conroy at its own default is indistinguishable from "
        f"bump_strength=0 (rel {moved:.3e}): the 2175 A bump the law exists to "
        "add is not there (#1833)."
    )


@pytest.mark.parametrize("value", [0.0, 0.5, 2.0])
def test_a_requested_shape_parameter_still_reaches_the_curve(value, dust_ssp, uv_obs):
    """Guard against over-correcting into ignoring the user.

    The fix omits a law kwarg when *nobody asked*. A user who does ask must
    still be honored -- including asking for exactly the value the shared spec
    default happens to be, which must not be mistaken for "unset".
    """
    base = _sed(_build(dust_ssp, uv_obs, _two("kriek_conroy")))
    stated = _sed(_build(dust_ssp, uv_obs, _two("kriek_conroy", bump_strength=Fixed(value))))
    moved = _rel(stated, base)
    if abs(value - 1.0) < 1e-12:  # the law's own default: stating it is a no-op
        assert moved <= _INERT_TOL
    else:
        assert moved > 1e-3, (
            f"bump_strength={value} was requested and did not move the SED "
            f"(rel {moved:.3e}); the narrowing has swallowed a user request (#1833)."
        )


def test_per_component_overrides_are_still_honored(dust_ssp, uv_obs):
    """``slope_bc`` / ``delta_diff`` are explicit requests, not defaults.

    They arrive as static config overrides rather than through the params
    dict, so a narrowing keyed only on provenance could drop them.
    """
    base = _sed(_build(dust_ssp, uv_obs, _two("noll09")))
    override = _sed(_build(dust_ssp, uv_obs, _two("noll09", delta_diff=-0.6)))
    moved = _rel(override, base)
    assert moved > 1e-3, (
        f"delta_diff=-0.6 was set per-component and did not move the SED (rel {moved:.3e}); "
        "the narrowing dropped an explicit override (#1833)."
    )


@pytest.mark.parametrize("law", ["kriek_conroy", "calzetti"])
def test_the_precompute_path_bakes_the_same_curve(law, dust_ssp, uv_obs):
    """The energy-balance LUT must not disagree with the direct path.

    ``resolve_bc_diff_law_params`` has three callers, and the third is
    ``SEDModel``'s energy-balance LUT builder for the ``WavePrecomp`` path --
    which every fitter turns on, since each resolves ``approx="auto"`` to it
    for photometry. Narrowing only the component would leave the LUT baking a
    curve *with* the bump while ``apply()`` evaluated one without it: one model
    carrying two screens depending on whether ``approx`` was on, and invisible
    to any test that exercises a single path (cf. #1665, #1434).

    ``calzetti`` is the control -- it reads no shape parameter, so the
    narrowing cannot reach it and this pins that the harness itself compares
    like with like.
    """
    from tengri import WavePrecomp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = SEDModel.build(
            ssp_data=dust_ssp,
            observation=uv_obs,
            sfh={"type": "dpl", "all_params": FIXED},
            dust=_two(law),
            redshift=Fixed(0.5),
        )
        precomp = SEDModel.build(
            ssp_data=dust_ssp,
            observation=uv_obs,
            sfh={"type": "dpl", "all_params": FIXED},
            dust=_two(law),
            redshift=Fixed(0.5),
            approx=WavePrecomp(n_z=8, z_min=0.0, z_max=2.0),
        )
        params = dict(exact.spec.sample(jax.random.PRNGKey(0)))
        a = np.asarray(exact.predict_photometry(params))
        b = np.asarray(precomp.predict_photometry(params))

    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), (
        f"'{law}': non-finite photometry (exact finite={np.all(np.isfinite(a))}, "
        f"precomp finite={np.all(np.isfinite(b))})"
    )
    # Band-averaging makes the two paths agree to a residual, not to the bit;
    # the defect this guards against is a whole missing Drude bump, which is
    # 128% -- orders above any quadrature difference.
    diff = _rel(a, b)
    assert diff < 0.05, (
        f"'{law}': exact and WavePrecomp photometry differ by {diff:.4e}. The LUT and "
        "apply() are not evaluating the same curve, so the model carries two screens "
        "depending on whether approx is on (#1833)."
    )
