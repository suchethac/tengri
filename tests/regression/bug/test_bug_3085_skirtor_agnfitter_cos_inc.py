# SPDX-License-Identifier: BSD-3-Clause
r"""The SKIRTOR-AGNfitter tori take their sightline from ``agn_cos_inc`` (#3085).

The AGN has one inclination, ``agn_cos_inc`` (owner ruling, #2605). The AGNfitter-rX
``SKIRTOR_mean_1p``/``2p``/``3p`` reductions of SKIRTOR (Stalevski et al. 2016 [1]_;
Martinez-Ramirez et al. 2024 [2]_) tabulate the torus on the inclination
:math:`i` from the polar axis [deg] (0 face-on, 90 edge-on), and were keyed by their own
``agn_incl_skirtor`` while the disc screen read ``agn_cos_inc``: a composable model saw its
torus along one sightline and its disc along another. The library is now keyed at
:math:`i = \arccos(\cos i)`.

``agn_incl_skirtor`` is retired: a call that passes it is translated to
``agn_cos_inc = cos(radians(agn_incl_skirtor))`` with a ``DeprecationWarning``, a call that
passes both with different sightlines raises, and the model grammar refuses it with the
translated entry.

The reference is the library itself, interpolated at the inclination in degrees with the
same lookup the components use, so it does not pass through the conversion under test.

References
----------
.. [1] M. Stalevski et al., MNRAS, 458, 2288 (2016). arXiv:1602.06954.
.. [2] L. N. Martinez-Ramirez et al., A&A, 688, A46 (2024). arXiv:2405.12111.
"""

from __future__ import annotations

import importlib
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel, Uniform
from tengri._data_setup import find_data
from tengri.components.agn._template_grid import torus_lnu_from_grid
from tengri.components.agn._torus_inclination import torus_lnu_anisotropic
from tengri.components.agn.blocks.registry import composable
from tengri.components.agn.blocks.runner import composable_agn_l_nu

pytestmark = pytest.mark.regression_bug

_WAVE = np.geomspace(1.0e3, 1.0e7, 1500)
_KW = {"agn_log_lbol": 10.0, "agn_torus_frac": 0.4}

#: torus block -> (module, grid file, the other axes as (kwarg, value) in axis order, the
#: inclination's axis index).
_TORI = {
    "skirtor_agnfitter_1p": ("skirtor_agnfitter_1p", "skirtor_mean1p_torus_grid.h5", (), 0),
    "skirtor_agnfitter_2p": (
        "skirtor_agnfitter_2p",
        "skirtor_mean2p_torus_grid.h5",
        (("agn_oa_skirtor", 40.0),),
        1,
    ),
    "skirtor_agnfitter": (
        "skirtor_agnfitter",
        "skirtor_mean3p_torus_grid.h5",
        (("agn_oa_skirtor", 40.0), ("agn_tv_skirtor", 7.0)),
        1,
    ),
}

#: Two sightlines on either side of the 30 deg default, off the library's 10 deg nodes.
_MUS = (0.93, 0.31)


def _library(torus):
    module, filename, others, incl_axis = _TORI[torus]
    path = find_data(filename)
    if path is None:
        pytest.skip(f"template data unavailable ({filename})")
    mod = importlib.import_module(f"tengri.components.agn.{module}")
    return mod, getattr(mod, f"load_{module}_grid")(str(path)), dict(others), incl_axis


def _reference(torus, mu):
    """The library at i = degrees(arccos(mu)), interpolated directly on its degree axis."""
    mod, grid, others, incl_axis = _library(torus)
    coords = list(others.values())
    coords.insert(incl_axis, float(np.degrees(np.arccos(mu))))
    if torus == "skirtor_agnfitter_1p":
        return np.asarray(
            torus_lnu_anisotropic(
                grid, jnp.asarray(_WAVE), tuple(coords), mod.TORUS_INCLINATION, **_KW
            )
        )
    return np.asarray(torus_lnu_from_grid(grid, jnp.asarray(_WAVE), tuple(coords), **_KW))


def _sed(torus, **sightline):
    mod, grid, others, _ = _library(torus)
    return np.asarray(
        getattr(mod, f"{_TORI[torus][0]}_sed_from_grid")(
            grid, jnp.asarray(_WAVE), **others, **_KW, **sightline
        )
    )


@pytest.mark.parametrize("torus", sorted(_TORI))
@pytest.mark.parametrize("mu", _MUS)
def test_torus_is_keyed_at_the_inclination_of_agn_cos_inc(torus, mu):
    """The torus SED at ``agn_cos_inc = mu`` is the library at ``i = arccos(mu)`` (1e-12)."""
    sed = _sed(torus, agn_cos_inc=mu)
    ref = _reference(torus, mu)
    sig = ref > 0.0
    np.testing.assert_allclose(sed[sig], ref[sig], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("torus", sorted(_TORI))
def test_disc_and_torus_move_together_with_agn_cos_inc(torus):
    """In the composable runner one ``agn_cos_inc`` moves the disc and the torus at once."""
    _library(torus)
    pieces = {}
    for mu in _MUS:
        _, _, _, comps = composable_agn_l_nu(
            _WAVE,
            agn_log_lbol=_KW["agn_log_lbol"],
            agn_disc_block="multicolor",
            agn_torus_block=torus,
            agn_norm="conserving",
            agn_torus_frac=_KW["agn_torus_frac"],
            agn_cos_inc=mu,
            return_l2500=True,
            return_components=True,
        )
        pieces[mu] = {k: np.asarray(comps[k]) for k in ("disc", "torus")}
    face, edge = (pieces[mu] for mu in _MUS)
    assert not np.allclose(face["disc"], edge["disc"], rtol=1e-6), "disc ignores agn_cos_inc"

    # The torus piece follows the library between the two sightlines exactly as the library
    # does, whatever the runner's own (sightline-independent) scaling of the block.
    with np.errstate(divide="ignore", invalid="ignore"):
        ref_ratio = _reference(torus, _MUS[1]) / _reference(torus, _MUS[0])
    sig = (face["torus"] > 0.0) & np.isfinite(ref_ratio) & (ref_ratio > 0.0)
    assert np.max(np.abs(ref_ratio[sig] - 1.0)) > 1e-3, "the library must differ between mus"
    np.testing.assert_allclose(edge["torus"][sig] / face["torus"][sig], ref_ratio[sig], rtol=1e-9)


@pytest.mark.parametrize("torus", sorted(_TORI))
def test_retired_incl_skirtor_warns_and_translates(torus):
    """``agn_incl_skirtor`` warns and equals ``agn_cos_inc = cos(radians(.))`` bit for bit."""
    incl = 63.0
    with pytest.warns(DeprecationWarning, match="agn_incl_skirtor.*agn_cos_inc"):
        old = _sed(torus, agn_incl_skirtor=incl)
    new = _sed(torus, agn_cos_inc=float(np.cos(np.radians(incl))))
    np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("torus", sorted(_TORI))
def test_retired_incl_skirtor_conflicting_with_agn_cos_inc_raises(torus):
    """Two different sightlines in one call raise; agreeing ones warn and run."""
    with pytest.raises(ValueError, match="two different sightlines"):
        _sed(torus, agn_incl_skirtor=60.0, agn_cos_inc=0.9)
    with pytest.warns(DeprecationWarning):
        _sed(torus, agn_incl_skirtor=60.0, agn_cos_inc=float(np.cos(np.radians(60.0))))


def test_runner_translates_the_retired_angle_for_the_disc_too():
    """``composable(..., agn_incl_skirtor=i)`` is ``composable(..., agn_cos_inc=cos i)``."""
    _library("skirtor_agnfitter_1p")
    base = dict(
        agn_disc_block="multicolor",
        agn_torus_block="skirtor_agnfitter_1p",
        agn_norm="conserving",
        agn_torus_frac=_KW["agn_torus_frac"],
    )
    with pytest.warns(DeprecationWarning, match="agn_incl_skirtor"):
        old = np.asarray(composable(_WAVE, _KW["agn_log_lbol"], agn_incl_skirtor=72.0, **base))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new = np.asarray(
            composable(
                _WAVE, _KW["agn_log_lbol"], agn_cos_inc=float(np.cos(np.radians(72.0))), **base
            )
        )
    assert not [w for w in caught if "agn_incl_skirtor" in str(w.message)]
    np.testing.assert_array_equal(old, new)


@pytest.fixture(scope="module")
def _model_inputs(synthetic_ssp_wide, synthetic_tophat_obs):
    return synthetic_ssp_wide, synthetic_tophat_obs


def _model(inputs, agn_top=None, **torus):
    ssp, obs = inputs
    agn = {
        "type": "composable",
        "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor_agnfitter_1p", **torus},
        "all_params": Fixed(DEFAULT),
        **(agn_top or {}),
    }
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        agn=agn,
        redshift=Fixed(0.1),
    )


@pytest.mark.parametrize(
    ("value", "translation"),
    [
        (Fixed(60.0), r"'cos_inc': Fixed\(0\.5\)"),
        (Uniform(10.0, 80.0), r"'cos_inc': Uniform\(0\.173648, 0\.984808\)"),
        (FREE, r"'cos_inc': FREE\}"),
        (Fixed(DEFAULT), r"'cos_inc': Fixed\(DEFAULT\)\}"),
    ],
)
@pytest.mark.parametrize("key", ["incl_skirtor", "agn_incl_skirtor"])
def test_grammar_refuses_the_retired_angle_with_its_translation(
    _model_inputs, key, value, translation
):
    """The model grammar refuses ``incl_skirtor`` and spells the ``agn_cos_inc`` entry."""
    with pytest.raises(ValueError, match=rf"retired \(#3085\)(.|\n)*{translation}"):
        _model(_model_inputs, **{key: value})


def test_model_frees_one_inclination_for_disc_and_torus(_model_inputs):
    """A free inclination is ``agn_cos_inc``; the model carries no second angle."""
    model = _model(_model_inputs, agn_top={"cos_inc": Uniform(0.0, 1.0)})
    assert "agn_cos_inc" in model.spec.free_params
    names = [*model.spec.free_params, *model.spec.fixed_params]
    assert not any("incl_skirtor" in name for name in names)


def test_predict_refuses_the_retired_angle_with_its_translation(_model_inputs):
    """``model.predict({'agn_incl_skirtor': ...})`` names ``agn_cos_inc``, not a guess."""
    from tengri.config.exceptions import UnknownParameterError

    model = _model(_model_inputs, agn_top={"cos_inc": Uniform(0.0, 1.0)})
    with pytest.raises(UnknownParameterError, match=r"retired \(#3085\)(.|\n)*agn_cos_inc"):
        model.predict({"agn_cos_inc": 0.5, "agn_incl_skirtor": 40.0})
