# SPDX-License-Identifier: BSD-3-Clause
"""``xray_aird`` and ``agn_xray_corona`` must build their own component (#1684).

Both register their own ``SEDModelComponent`` subclass -- ``XRayAirdSEDComponent``
and ``AGNXRayCoronaSEDComponent``, each with its own config and ``predict`` --
and ``component_factory`` built neither. It resolved the registry key ``"xray"``
unconditionally and passed the *name* as a config field:

    _resolve_registry_component("xray", "xray",
                                config=XRaySEDComponentConfig(model=xray_model))

``XRaySEDComponent`` branches on ``config.model`` for ``lopez24`` and falls
through to the ``yang20`` corona otherwise, so both names produced a
bit-identical SED to ``yang20``.

This is the unfinished half of **#1120**, which closed after adding the names to
the grammar allowlist but not to the factory -- turning that issue's loud
``ValueError`` into silence, the outcome it explicitly called worse. PR #1676
recorded it as "filed, not fixed here".

**Fixture note.** The corona models tie their emission to the disc's
``L_2500`` through alpha_ox, so a build with no AGN has nothing for the
prescription to act on and every X-ray name measures identical -- for an honest
reason. A luminous AGN is required to tell the defect from the fixture; without
it, ``lopez24`` also measures identical, which is how it was once mis-recorded
as a fourth aliased name.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, SSPData
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

#: Corona prescriptions that must each produce their own SED.
_DISTINCT_FROM_YANG20 = ("xray_aird", "agn_xray_corona", "lopez24")

_LUMINOUS_AGN = {
    "type": "composable",
    "disc": {"type": "multicolor"},
    "torus": {"type": "skirtor"},
    "log_lbol": Fixed(13.0),
}


@pytest.fixture(scope="module")
def xray_ssp() -> SSPData:
    """SSP from 1 A, so the X-ray band exists on the grid."""
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(0.0, 7.0, 2000)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-12, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


@pytest.fixture(scope="module")
def xray_obs() -> Observation:
    """Bands in the X-ray plus an optical anchor."""

    def _tophat(center: float, frac: float = 0.25, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{center:.4g}")

    centers = (2.0, 10.0, 50.0, 3500.0, 6200.0)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _rest_sed(xray_ssp, xray_obs, name: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=xray_ssp,
            observation=xray_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn=_LUMINOUS_AGN,
            xray={"type": name},
            redshift=Fixed(0.5),
        )
        params = model.spec.sample(jax.random.PRNGKey(0))
        pred = model.predict(params)
        sed = np.asarray(pred.rest_sed())
        axis = np.asarray(pred.wave_rest)
    # judged in the X-ray band, not grid-wide
    return sed[(axis >= 0.1) & (axis <= 1.0e2)]


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.where(np.abs(b) > 0, np.abs(b), 1.0)))


@pytest.mark.parametrize("name", _DISTINCT_FROM_YANG20)
def test_corona_name_is_not_silently_yang20(name, xray_ssp, xray_obs):
    """Selecting a corona prescription must deliver that prescription."""
    yang20 = _rest_sed(xray_ssp, xray_obs, "yang20")
    other = _rest_sed(xray_ssp, xray_obs, name)

    assert _max_rel(other, yang20) > 1e-9, (
        f"xray={name!r} is indistinguishable from 'yang20' in the X-ray band. "
        "The name validates, appears in list_xray_models(), and delivers another "
        "model's physics (#1684, the unfinished half of #1120)."
    )


@pytest.mark.parametrize("name", ("yang20", *_DISTINCT_FROM_YANG20))
def test_every_corona_actually_emits(name, xray_ssp, xray_obs):
    """Each X-ray name must put real flux into ``sed_xray``.

    "Differs from yang20" does not imply this, and the gap is not theoretical:
    when ``agn_xray_corona`` was first routed to its own class it built
    correctly, separated from yang20 by max rel 1.0, and emitted **exactly
    zero** -- it reads its L_2500 anchor from ``inputs`` and declared no
    optional inputs, so every anchor fell back to 0.0. A silent component
    differs from an emitting one, so the sibling comparison passed while the
    fix had traded "silently delivers yang20's physics" for "silently delivers
    nothing".

    This asserts the thing the user actually wants: that selecting the name
    produces X-ray emission.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=xray_ssp,
            observation=xray_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn=_LUMINOUS_AGN,
            xray={"type": name},
            redshift=Fixed(0.5),
        )
        params = model.spec.sample(jax.random.PRNGKey(0))
        state = model.predict_state(params)

    sed_xray = getattr(state.derived, "sed_xray", None)
    assert sed_xray is not None, f"xray={name!r} published no sed_xray at all"
    total = float(np.sum(np.asarray(sed_xray)))
    assert total > 0.0, (
        f"xray={name!r} builds but emits nothing: sum(sed_xray) == {total}. "
        "A silent component still 'differs from yang20', so only this "
        "assertion can tell the two apart."
    )


def test_simple_is_still_the_declared_yang20_alias(xray_ssp, xray_obs):
    """``simple`` and ``yang20`` are one model under two names, by declaration.

    Pinned so a future change cannot quietly make them differ (which would be a
    silent physics change for every default build) and so this pair is never
    confused with the three above, which are defects.
    """
    assert (
        _max_rel(_rest_sed(xray_ssp, xray_obs, "simple"), _rest_sed(xray_ssp, xray_obs, "yang20"))
        <= 1e-9
    ), (
        "'simple' and 'yang20' are documented as the same physics "
        "(parameters/groups.py). They now differ -- either the alias was broken "
        "or the declaration is stale."
    )


@pytest.mark.parametrize("name", ("xray_aird", "agn_xray_corona"))
def test_factory_builds_the_named_component_class(name):
    """The factory must build the class the name registers, not the generic one.

    Asserted structurally as well as numerically: the SED comparison above can
    be satisfied by any change that makes the numbers differ, while this pins
    *which* component was constructed.
    """
    from tengri.components.sed_model_component import _REGISTRY

    assert name in _REGISTRY, f"{name} registers no component class"
    assert _REGISTRY[name] is not _REGISTRY["xray"], (
        f"{name} resolves to the generic XRaySEDComponent; it has its own class"
    )
