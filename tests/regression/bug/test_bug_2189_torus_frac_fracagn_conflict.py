# SPDX-License-Identifier: BSD-3-Clause
"""#2189 (RULING R15): agn_torus_frac silently discarded under fracAGN.

``AGNSEDComponent.apply`` (``components/agn/component.py``) overrides
whatever ``agn_torus_frac`` a caller supplies with a value derived from the
dust-absorbed stellar luminosity (the CIGALE skirtor2016
``agn_power = L_absorbed x fracAGN/(1-fracAGN)`` coupling) whenever fracAGN
(``agn_ir_frac``) is active. The override computation reads only
``agn_ir_frac``/``agn_torus_frac``/the absorbed luminosity -- it runs BEFORE
the runner's separate ``agn_norm`` branch and never references it -- so it
applies regardless of ``agn_norm``. Measured on this branch (5 torus types,
``sfh=delayed``, ``dust_attenuation=two_component``, ``dust_emission=dl07``,
WISE/2MASS/Spitzer bands, ``norm='independent'`` explicit): varying
``agn_torus_frac`` 0.05 -> 0.95 with fracAGN left at its registry default
(0.0, inactive) changes photometry by 8.7x-17.1x (max relative diff); with
fracAGN explicitly active (measured under the DEFAULT ``agn_norm``,
``'cigale_joint'``) it changes photometry by EXACTLY 0.0 -- bit-identical,
for every torus type.

Two guards close this silent-override trap:

1. A build-time ``ConfigError`` (``sed_model.py::
   _validate_torus_frac_fracagn_conflict``) when a caller explicitly names
   BOTH ``agn_torus_frac`` (a prior or ``Fixed`` value) and an active fracAGN
   -- a contradiction the caller should be told about, not silently resolved
   in fracAGN's favor.
2. The ``agn.torus`` sub-block's own ``'all_params': FREE`` wildcard never
   frees ``agn_torus_frac`` when fracAGN is active
   (``parameters.groups._agn_ir_frac_explicit_and_active``, consulted while
   computing wildcard scopes) -- silently narrower, not an error, since a
   wildcard asks for "whatever varies", not a specific name.

This file asserts BOTH the refusal (guard 1) and the measured LIVE case
(fracAGN inactive) -- the refusal alone would not prove agn_torus_frac
actually works when it is safe to use it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel
from tengri.config.exceptions import ConfigError

pytestmark = [pytest.mark.regression_bug, pytest.mark.contract]

#: Torus types measured; F1's own probe covered these five.
_TORUS_TYPES = ("fritz", "cat3d_wind", "nenkova", "simple", "skirtor")

#: Measured (this branch) max relative photometry diff for agn_torus_frac
#: 0.05 -> 0.95, fracAGN inactive, norm='independent' explicit. Every value
#: is well above the _LIVE_FLOOR the regression test asserts, with margin.
_MEASURED_LIVE_REL_DIFF = {
    "fritz": 8.7394,
    "cat3d_wind": 17.130,
    "nenkova": 16.948,
    "simple": 11.460,
    "skirtor": 16.448,
}
_LIVE_FLOOR = 5.0
assert all(v > _LIVE_FLOOR for v in _MEASURED_LIVE_REL_DIFF.values()), (
    "measured numbers must exceed the floor the regression test asserts"
)

_BANDS = (
    "SLOAN_SDSS_g",
    "2MASS_2MASS_Ks",
    "WISE_WISE_W1",
    "WISE_WISE_W2",
    "WISE_WISE_W3",
    "WISE_WISE_W4",
    "Spitzer_IRAC_I4",
    "Spitzer_MIPS_24mu",
)


@pytest.fixture(scope="module")
def ssp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tengri.load_ssp()


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(list(_BANDS)))


def _build(ssp, obs, torus_type: str, *, torus_frac, ir_frac=None):
    """One composable-AGN build with the given torus type, 'norm':
    'independent' explicit (no cross-block energy coupling to confound the
    measurement), and an explicit ``torus_frac``. ``ir_frac`` is left unset
    (registry default 0.0, inactive) unless given.
    """
    agn = {
        "type": "composable",
        "disc": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "torus": {"type": torus_type, "all_params": Fixed(DEFAULT), "torus_frac": torus_frac},
        "agn_log_lbol": Fixed(12.0),
        "norm": "independent",
    }
    if ir_frac is not None:
        agn["ir_frac"] = ir_frac
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dl07", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            agn=agn,
            redshift=Fixed(0.1),
        )


def _photometry(model) -> np.ndarray:
    p = model.spec.sample_batch(__import__("jax").random.PRNGKey(0), 1)
    p = {k: v[0] for k, v in p.items()}
    return np.asarray(model.predict_photometry(p))


@pytest.mark.parametrize("torus_type", _TORUS_TYPES)
def test_explicit_torus_frac_and_active_fracagn_raises(ssp, obs, torus_type):
    """Guard 1: naming both agn_torus_frac AND an active fracAGN explicitly
    must raise ConfigError naming both keys and the two ways out."""
    with pytest.raises(ConfigError) as excinfo:
        _build(ssp, obs, torus_type, torus_frac=Fixed(0.3), ir_frac=Fixed(0.5))
    msg = str(excinfo.value)
    assert "agn_torus_frac" in msg
    assert "fracAGN" in msg or "agn_ir_frac" in msg
    assert "#2189" in msg


@pytest.mark.parametrize("torus_type", _TORUS_TYPES)
def test_explicit_torus_frac_is_live_when_fracagn_inactive(ssp, obs, torus_type):
    """The LIVE case (not merely the refusal): with fracAGN left at its
    registry default, agn_torus_frac 0.05 -> 0.95 must change photometry by
    at least _LIVE_FLOOR (max relative diff) -- well below every measured
    value in _MEASURED_LIVE_REL_DIFF, so this is a floor, not a repeat of
    the exact measurement."""
    a = _photometry(_build(ssp, obs, torus_type, torus_frac=Fixed(0.05)))
    b = _photometry(_build(ssp, obs, torus_type, torus_frac=Fixed(0.95)))
    rel = float(np.max(np.abs(b - a) / np.maximum(np.abs(a), 1e-300)))
    assert rel > _LIVE_FLOOR, (
        f"{torus_type}: agn_torus_frac 0.05->0.95 changed photometry by only "
        f"{rel:.4e} (floor {_LIVE_FLOOR}) with fracAGN inactive -- expected "
        f"live (measured {_MEASURED_LIVE_REL_DIFF[torus_type]:.4e} on this branch)"
    )


@pytest.mark.parametrize("torus_type", _TORUS_TYPES)
def test_wildcard_never_frees_torus_frac_when_fracagn_active(ssp, obs, torus_type):
    """Guard 2: agn.torus={'all_params': FREE} must NOT free agn_torus_frac
    when fracAGN is explicitly active -- it silently narrows (no error,
    unlike guard 1's explicit-vs-explicit case), since freeing a parameter
    AGNSEDComponent.apply() is about to override would hand the sampler a
    dead dimension. Every OTHER declared torus parameter must still be
    freed normally (the narrowing is targeted, not a wholesale wildcard
    failure)."""
    agn = {
        "type": "composable",
        "disc": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "torus": {"type": torus_type, "all_params": FREE},
        "agn_log_lbol": Fixed(12.0),
        "norm": "independent",
        "ir_frac": Fixed(0.5),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dl07", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            agn=agn,
            redshift=Fixed(0.1),
        )
    free = set(model.spec.free_params)
    assert "agn_torus_frac" not in free, (
        f"{torus_type}: agn.torus={{'all_params': FREE}} froze agn_torus_frac "
        f"as free even though fracAGN is explicitly active -- it should be "
        f"silently narrowed out (guard 2)"
    )
    from tengri.parameters.groups import _agn_subblock_declared_params

    declared = _agn_subblock_declared_params("torus", torus_type) or frozenset()
    other_declared = declared - {"agn_torus_frac"}
    if other_declared:
        assert other_declared <= free, (
            f"{torus_type}: narrowing agn_torus_frac out of the wildcard scope "
            f"also dropped OTHER declared torus parameters: "
            f"{sorted(other_declared - free)}"
        )
