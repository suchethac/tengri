# SPDX-License-Identifier: BSD-3-Clause
"""Measured-liveness contract: dust_eta_balance must never be free-but-dead.

``dust_eta_balance`` ("L_IR = eta * L_absorbed") is partitioned into the
``dust_emission`` group, whose wildcard (``_wildcard_scopes`` in
``parameters/groups.py``) frees it unconditionally across every
``dust_attenuation`` type. That union is only correct if EVERY attenuation
type actually reads ``dust_eta_balance`` when computing ``L_ir`` -- otherwise
the wildcard frees a parameter whose posterior always equals its prior, the
exact "declared free, silently dead" defect this file's fix (and the earlier
single-component fix) both close.

Regression: WG00 (``src/tengri/components/dust/wg00_model.py``) was exactly
this case. The wildcard-scope fix (unconditional union) landed before
``wg00_model.py::apply()`` read ``dust_eta_balance`` at all, so a
WG00-attenuated model reported ``dust_eta_balance`` in ``free_params`` with
``d(photometry)/d(eta) == 0`` exactly -- confirmed live by review. This test
enumerates every ``dust_attenuation`` type from the registry (``_VALID_DUST_TYPES``,
the same set ``SEDModel.build`` validates against) rather than allow-listing
which types to check, so a future attenuation type that frees the parameter
without reading it fails here immediately, the same way WG00 did.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel
from tengri.observation.photometry import FilterCurve
from tengri.parameters.groups import _VALID_DUST_TYPES

pytestmark = pytest.mark.contract

#: Structural kwargs each dust_attenuation type requires beyond 'type' (a law
#: selector for single/two-component; wg00 needs none -- its curve/geometry/
#: structure selectors all have defaults). Syntactic requirements of the
#: grammar, not an allow-list of which types are expected to pass the check
#: below: every type in _VALID_DUST_TYPES is exercised identically.
_ATTEN_STRUCTURAL_KWARGS: dict[str, dict] = {
    "single_component": {"law": "calzetti"},
    "two_component": {"law": "calzetti"},
    "wg00": {},
}


def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


def _obs() -> Observation:
    """UV through far-IR, so the dust IR re-emission lands somewhere observable."""
    centers = (1500.0, 3500.0, 6200.0, 2.4e5, 5.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _build(ssp, dust_type: str) -> SEDModel:
    extra = _ATTEN_STRUCTURAL_KWARGS.get(dust_type, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=_obs(),
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": dust_type, "all_params": FREE, **extra},
            dust_emission={"type": "schreiber2018", "all_params": FREE},
            neb={"type": "none"},
            redshift=Fixed(0.5),
        )


def _central_params(spec) -> dict:
    """Free parameters at their declared prior medians, via ``unstandardize(0)``.

    Mirrors ``tengri.inference.fitter._central_params``: ``unstandardize(0.0)``
    is the declared prior's median for every distribution class (#1651), the
    single source of truth every hierarchical sampler standardizes through,
    so this reuses it rather than inventing a second "central value" notion.
    """
    return {name: spec.get_distribution(name).unstandardize(0.0) for name in spec.free_params}


@pytest.mark.parametrize("dust_type", sorted(_VALID_DUST_TYPES))
def test_dust_eta_balance_free_implies_live(synthetic_ssp_wide, dust_type):
    """If dust_eta_balance is freed by any wildcard, it must move photometry.

    Measured, not asserted by type: every ``dust_attenuation`` type is built
    the same way and checked the same way. A type that frees the parameter
    but never reads it fails here -- no per-type allow-list carves out an
    exception.
    """
    model = _build(synthetic_ssp_wide, dust_type)
    if "dust_eta_balance" not in model.spec.free_params:
        pytest.skip(f"{dust_type}: dust_eta_balance not freed by this build; nothing to measure")

    params = _central_params(model.spec)
    eta0 = float(params["dust_eta_balance"])

    def loss(eta):
        p = {**params, "dust_eta_balance": eta}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return jnp.sum(model.predict_photometry(p))

    grad = jax.grad(loss)(eta0)
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0, (
        f"{dust_type}: dust_eta_balance is in free_params (freed by the dust_emission "
        f"wildcard) but d(photometry)/d(eta) == 0 at the prior median (eta={eta0}) -- "
        "a declared free parameter whose posterior always equals its prior. This "
        f"dust_attenuation type does not read dust_eta_balance when computing L_ir."
    )
