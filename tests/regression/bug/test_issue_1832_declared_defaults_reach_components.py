# SPDX-License-Identifier: BSD-3-Clause
"""A params dict built from declarations cannot fall behind them (#1832).

``75bedcb4e`` (#1706/#1771) wired ``xray_det_hmxb`` / ``xray_det_lmxb`` into the
physics, indexing them as ``params["xray_det_hmxb"]`` rather than
``.get(..., 0.0)`` — deliberately, because a neutral default is exactly what let
the missing handoff look wired for as long as it did.

That is right, and it broke fifty integration tests, because those two
parameters differ from every sibling ``xray_*`` in **how they are declared**:

    xray_gamma_hmxb   Fixed(2.0)                    -> spec.get_fixed_values()
    xray_det_hmxb     Uniform(-2, 2, default=0.0)   -> free; the caller supplies it

Eight fixtures hand-rolled a "complete" ``xray_*`` dict. They were complete when
written. A literal cannot follow the declaration it copies, so the component now
reads past the end of them.

The guard is therefore not "these eight dicts have these two keys" — that is the
same copy one generation later. It is that a dict *derived* from the declarations
covers every declared parameter, for every component in the registry, so the next
parameter addition is a no-op here.

See :func:`tengri.forward.orchestrator.default_params_dict`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

from tengri.components.xray.component import XRaySEDComponent
from tengri.forward.orchestrator import (
    default_params_dict,
    merge_declared_parameters,
    run_components,
)
from tengri.protocols import ForwardState


def _xray_state():
    """A minimal state carrying what the X-ray component reads."""
    wave = jnp.linspace(1e0, 1e3, 1024)  # X-ray range: 1 to 1000 Angstrom
    return ForwardState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        derived={"sfr": 5.0, "log_mstar": jnp.log10(5e10), "L_agn_bol": 1e44},
    )


def test_declared_defaults_drive_the_xray_pipeline():
    """The reproduction: #1832's ``KeyError: 'xray_det_hmxb'``, on the derived dict."""
    component = XRaySEDComponent()
    params = default_params_dict([component], overrides={"redshift": 0.5})

    final = run_components([component], _xray_state(), params)

    assert np.all(np.isfinite(np.asarray(final.sed_intrinsic)))
    assert float(jnp.sum(final.sed_intrinsic)) > 0.0, "X-ray component emitted nothing"


def test_the_offsets_reach_the_physics_through_the_derived_dict():
    """The derived dict must carry the *value*, not merely the key.

    A helper that supplied a placeholder would satisfy the test above and
    reintroduce #1706 — the offsets present and inert. ``xray_det_hmxb`` is a
    log10 luminosity offset in dex, so +0.5 dex must brighten the X-ray SED.
    """
    component = XRaySEDComponent()
    base = default_params_dict([component], overrides={"redshift": 0.5})

    assert float(base["xray_det_hmxb"]) == 0.0, "declared default is a null offset"

    brighter = {**base, "xray_det_hmxb": jnp.asarray(0.5)}
    sed_0 = np.asarray(run_components([component], _xray_state(), base).sed_intrinsic)
    sed_up = np.asarray(run_components([component], _xray_state(), brighter).sed_intrinsic)

    assert np.sum(sed_up) > np.sum(sed_0), (
        "xray_det_hmxb is inert through default_params_dict — the offset reached "
        "the dict but not the physics (#1706)"
    )


def test_every_declared_parameter_is_supplied():
    """No declared parameter may be missing from its own derived dict.

    This is the property the eight hand-rolled literals could not hold. The
    registry-wide version of it lives in ``tests/contract``.
    """
    component = XRaySEDComponent()
    declared = merge_declared_parameters([component])
    supplied = default_params_dict([component])

    missing = sorted(set(declared) - set(supplied))
    assert not missing, f"{component.name!r} declares but does not supply: {missing}"
