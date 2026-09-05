# SPDX-License-Identifier: BSD-3-Clause
"""``xray_aird`` must build under ``WavePrecomp``, not just on the exact path.

Wiring ``xray_aird`` to its own component (#1684) exposed a latent defect in
that component: it publishes its precompute keys from ``name``, which is the
*registry key*, so it emitted ``xray_aird_phot_lnu_precomp``. ``DerivedState``
declares ``xray_phot_lnu_precomp`` and siblings, not that, so the key spilled
into ``_extras`` and the ADR-0007 guard raised on every build::

    ComponentIOError: run_components: state.derived._extras is non-empty after
    the forward pass: ['xray_aird_phot_lnu_precomp']

This never fired before because ``component_factory`` never built the component.
The failure mode matters: it is a **crash on the fit path**. Every fitter
resolves ``approx="auto"`` to ``WavePrecomp`` for photometry, so without this
the #1684 fix would have traded a silent wrong answer for a hard failure the
moment anyone fitted with ``xray_aird`` — worse than what it replaced.

The fix gives ``SEDModelComponent`` a ``publish_name``, so a component's
registry key and its published domain can differ. Several components share one
domain by construction: only one X-ray component is ever built, so ``xray_aird``
and the shared ``xray`` component publish the same ``xray_*`` fields and are
never in a state together.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, SSPData, WavePrecomp
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

_LUMINOUS_AGN = {
    "type": "composable",
    "disc": {"type": "multicolor"},
    "torus": {"type": "skirtor"},
    "log_lbol": Fixed(13.0),
}


@pytest.fixture(scope="module")
def xray_ssp() -> SSPData:
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(0.0, 7.0, 1500)
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
    def _tophat(center: float, frac: float = 0.25, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{center:.4g}")

    return Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (2.0, 10.0, 3500.0, 6200.0)))
    )


@pytest.mark.parametrize("approx_label", ["exact", "wave_precomp"])
@pytest.mark.parametrize("xray_type", ["yang20", "xray_aird", "lopez24", "agn_xray_corona"])
def test_every_xray_type_builds_and_predicts_on_both_paths(
    xray_type, approx_label, xray_ssp, xray_obs
):
    """A production X-ray name must not raise on the path fits actually take."""
    approx = None if approx_label == "exact" else WavePrecomp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=xray_ssp,
            observation=xray_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn=_LUMINOUS_AGN,
            xray={"type": xray_type},
            redshift=Fixed(0.5),
            approx=approx,
        )
        params = model.spec.sample(jax.random.PRNGKey(0))
        phot = np.asarray(model.predict_photometry(params))

    assert np.all(np.isfinite(phot)), (
        f"xray={xray_type!r} on the {approx_label} path produced non-finite photometry"
    )


def test_xray_aird_publishes_into_the_xray_domain():
    """Structural: the registry key and the published domain are allowed to differ.

    Pinned separately from the build test because the build test would also pass
    if someone added ``xray_aird_*`` fields to ``DerivedState`` instead -- which
    works, but leaves each new registry name outside the ``sed_xray`` accounting
    until somebody remembers to add it too.
    """
    from tengri.components.sed_model_component import _REGISTRY
    from tengri.protocols.derived_state import DerivedState

    cls = _REGISTRY["xray_aird"]
    assert cls.publish_name == "xray", (
        "xray_aird must publish into the shared xray domain; keying its "
        "precompute publishes off `name` emits xray_aird_* keys, which are not "
        "DerivedState fields and trip the ADR-0007 spillover guard."
    )
    declared = set(getattr(DerivedState, "__dataclass_fields__", {}))
    assert "xray_phot_lnu_precomp" in declared
    assert "xray_aird_phot_lnu_precomp" not in declared, (
        "A per-registry-name precompute field was added. That is the alternative "
        "fix this test exists to rule out -- see the docstring."
    )
