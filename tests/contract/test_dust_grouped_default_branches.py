# SPDX-License-Identifier: BSD-3-Clause
"""The grouped default covers the branches that sit inside its path.

``age_mixture='grouped'`` is the two-component default, so every dust configuration that
reaches the grouped path must agree with explicit ``'per_node'`` in value and gradient.
``test_dust_grouped_default_exact.py`` covers the main matrix; this module adds the
branches it does not exercise: the Lyman-continuum escape geometries (``birth_cloud_holes``,
``clear``), ``lyc_reprocessed_by='all'``, a composable shock under a non-default
``shock_screen``, an AGN component under a non-default ``agn_screen``, and the
``(WavePrecomp, FeaturePrecomp)`` approximation on a Cue + two-component + Dale 2014 model.

Each case is built twice (the default, and ``age_mixture='per_node'``) and compared in
float64 at rtol 1e-10, value and every gradient component.
"""

from __future__ import annotations

import warnings

import jax
import pytest

from tengri import DEFAULT, FeaturePrecomp, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.components.dust import two_component
from tests.contract.test_dust_grouped_default_exact import (
    _PROPERTY_NAMES,
    RTOL,
    _assert_agree,
    _dust,
    _emission,
    _neb,
    _observation,
    _params,
    _sfh,
)

pytestmark = pytest.mark.contract

_FALLBACK_TEXT = "did not publish 'lnu_interval'"


@pytest.fixture(autouse=True)
def _no_interval_fallback():
    """Fail any test whose grouped build reaches the lnu_interval fallback."""
    two_component._GROUPED_FALLBACK_WARNED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    hits = [str(w.message) for w in caught if _FALLBACK_TEXT in str(w.message)]
    assert not hits, f"grouped path fell back to the per-age cube: {hits[0][:120]}"


def _agn_block():
    """Composable AGN with a multicolor disc, independent norm (required for agn_screen)."""
    return {
        "type": "composable",
        "all_params": Fixed(DEFAULT),
        "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
        "log_lbol": Fixed(15.0),
        "norm": "independent",
    }


def _build(ssp, obs, mode, *, dust, neb, extra=None, emission=False, approx=None, z=0.05):
    """One model with ``dust`` and ``extra`` groups; ``mode`` None leaves the default."""
    d = dict(dust)
    if mode is not None:
        d["age_mixture"] = mode
    kwargs = dict(extra or {})
    if approx is not None:
        kwargs["approx"] = approx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh=_sfh("delayed"),
            dust_attenuation=d,
            dust_emission=_emission(emission),
            neb=neb,
            redshift=Fixed(z),
            **kwargs,
        )


def _pair(ssp, obs, case, approx=None):
    kw = dict(
        dust=case["dust"],
        neb=case["neb"],
        extra=case.get("extra"),
        emission=case.get("emission", False),
        approx=approx,
    )
    return _build(ssp, obs, None, **kw), _build(ssp, obs, "per_node", **kw)


_CUE_FESC = _neb("cue", lyc_fesc=True)
#: Geometry holes require the young-only reprocessing, so Cue counts only the young ages.
_CUE_YOUNG_FESC = {**_CUE_FESC, "lyc_max_age_yr": 1.0e7}
_NO_NEB = {"type": "none"}

#: name -> spec. ``dust`` is the two-component (or shock/AGN-screened) dust group.
CASES = {
    "lyc_birth_cloud_holes": dict(
        dust={
            **_dust(width=0.3),
            "lyc_escape_geometry": "birth_cloud_holes",
            "lyc_reprocessed_by": "young",
        },
        neb=_CUE_YOUNG_FESC,
        kinds=("phot", "spec", "joint", "props"),
        spectroscopy=True,
        ssp="fsps",
    ),
    "lyc_clear": dict(
        dust={**_dust(width=0.3), "lyc_escape_geometry": "clear", "lyc_reprocessed_by": "young"},
        neb=_CUE_YOUNG_FESC,
        kinds=("phot", "spec", "joint", "props"),
        spectroscopy=True,
        ssp="fsps",
    ),
    "lyc_reprocessed_by_all": dict(
        dust={**_dust(width=0.3), "lyc_reprocessed_by": "all"},
        neb=_CUE_FESC,
        kinds=("phot", "spec", "joint", "props"),
        spectroscopy=True,
        ssp="fsps",
    ),
    "shock_screen_birth_cloud": dict(
        dust={**_dust(width=0.3), "shock_screen": "birth_cloud"},
        neb=_NO_NEB,
        extra={"shock": {"frac": Uniform(0.0, 1.0)}},
        kinds=("phot", "spec", "joint", "props"),
        spectroscopy=True,
        ssp="wide",
    ),
    "agn_screen_diffuse": dict(
        dust={**_dust(width=0.3), "agn_screen": "diffuse"},
        neb=_NO_NEB,
        extra={"agn": _agn_block()},
        kinds=("phot", "spec", "joint", "props"),
        spectroscopy=True,
        ssp="wide",
    ),
    "feature_precomp_dale_cue": dict(
        dust=_dust(width=0.3),
        neb=_CUE_FESC,
        emission=True,
        kinds=("phot", "props"),
        spectroscopy=False,
        ssp="fsps",
        approx=(WavePrecomp(n_z=8, z_min=0.0, z_max=0.2), FeaturePrecomp()),
    ),
}


def _ssp_for(case, ssp_wide, ssp_fsps):
    return ssp_fsps if case["ssp"] == "fsps" else ssp_wide


@pytest.fixture(scope="module")
def ssp_obs(synthetic_tophat_obs):
    return synthetic_tophat_obs


@pytest.fixture(scope="module")
def ssp_wide(synthetic_ssp_wide):
    return synthetic_ssp_wide


@pytest.fixture(scope="module")
def ssp_fsps(ssp_data_fsps):
    return ssp_data_fsps


@pytest.mark.parametrize("name", list(CASES))
def test_grouped_default_branch_matches_per_node(name, ssp_wide, ssp_fsps, ssp_obs):
    """Value and gradient, grouped vs per_node, for every requested output kind."""
    case = CASES[name]
    ssp = _ssp_for(case, ssp_wide, ssp_fsps)
    obs = _observation(ssp_obs, spectroscopy=case["spectroscopy"])
    grouped, per_node = _pair(ssp, obs, case, approx=case.get("approx"))
    names = _PROPERTY_NAMES
    if case["neb"].get("type") == "cue":
        names = (*names, "halpha")
    rels = []
    for i, key in enumerate((jax.random.PRNGKey(3), jax.random.PRNGKey(11))):
        p = _params(grouped, key)
        for kind in case["kinds"]:
            rels.append(
                _assert_agree(per_node, grouped, p, kind, f"{name}/{kind}/draw{i}", names=names)
            )
    assert max(rels) <= RTOL
