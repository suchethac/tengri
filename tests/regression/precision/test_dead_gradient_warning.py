# SPDX-License-Identifier: BSD-3-Clause
r"""Freeing a gradient-dead parameter must say so (#1206).

``agn_kt_warm`` is declared ``Uniform(0.1, 0.5)`` by the KD18 disc, so a user
can free it — but it reaches the SED only through ``_nthcomp_lnu_interp``, whose
``custom_jvp`` supplies a ``gamma`` tangent and discards the ``kTe`` one. The
rule returns exactly ``0.0`` where a central difference gives
``d ln f / d ln kTe`` ~ -0.24.

The consequence is not a wrong number, it is an *unfalsifiable* one: every
gradient backend leaves the parameter at its initial value, so the posterior is
the prior — which is exactly what an unconstrained-but-honestly-fitted parameter
also looks like. Nothing downstream can tell the two apart, which is why this
has to be said at build time.

**Why the existing anti-laziness suite cannot catch this.**
``tests/crossval/test_anti_laziness.py::test_kd_all_unique_params_matter``
already asserts that ``agn_kt_warm`` changes the SED — it evaluates
``kubota_done_disc`` at 0.1 and 0.5 and requires a relative difference above
1e-4. That test passes, and is right to: the *forward* sensitivity is real. It
is the *derivative* that is missing. A parameter can be forward-live and
gradient-dead at the same time, and a suite that varies values can never
distinguish them — which is why this file differentiates instead.
"""

import warnings

import jax
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.config.exceptions import DeadGradientParameterWarning

pytestmark = pytest.mark.regression_bug

_SFH = {
    "type": "delayed",
    "all_params": FIXED,
    "log_total_mass": Uniform(9.0, 11.0),
    "tau_gyr": 1.0,
    "age_gyr": 5.0,
}


def _build(ssp, *, free_kt_warm):
    """A KD18-disc AGN model, with ``kt_warm`` freed or pinned."""
    disc = {"type": "kubota_done", "all_params": FIXED}
    disc["kt_warm"] = Uniform(0.1, 0.5) if free_kt_warm else Fixed(0.2)
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_r", "wise_w1"])),
        redshift=Fixed(0.1),
        sfh=_SFH,
        agn={
            "type": "composable",
            "all_params": FIXED,
            "disc": disc,
            "log_lbol": Uniform(9.0, 12.0),
        },
    )


def test_freeing_kt_warm_warns(ssp_bare):
    """The warning must fire, and name the parameter."""
    with pytest.warns(DeadGradientParameterWarning, match="agn_kt_warm"):
        _build(ssp_bare, free_kt_warm=True)


def test_pinning_kt_warm_does_not_warn(ssp_bare):
    """The other half — a pinned parameter is a normal, correct configuration.

    Without this, the warning could be firing unconditionally and the test above
    would still pass.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeadGradientParameterWarning)
        _build(ssp_bare, free_kt_warm=False)


def test_the_gradient_really_is_zero(ssp_bare):
    """Pins the defect itself, not only the announcement.

    If the ``kTe`` tangent is ever supplied, this fails and whoever did it is
    told to drop the warning — otherwise the warning would outlive the problem
    and start lying in the other direction.
    """
    from tengri.components.agn._nthcomp import nthcomp_lnu_interp

    nu = np.logspace(14.5, 18.5, 200)

    def total(kte):
        import jax.numpy as jnp

        return jnp.sum(nthcomp_lnu_interp(jnp.asarray(nu), 2.37, kte, 0.05))

    import jax.numpy as jnp

    _, tangent = jax.jvp(total, (jnp.asarray(0.2),), (jnp.asarray(1.0),))
    h = 1e-4
    central = float((total(0.2 + h) - total(0.2 - h)) / (2 * h))

    assert float(tangent) == 0.0, (
        f"the nthcomp rule now returns d/d(kTe) = {float(tangent):.5e} rather than 0.0. "
        "If the kTe tangent was deliberately added, remove 'agn_kt_warm' from "
        "_DEAD_GRADIENT_PARAMS in forward/sed_model.py — the warning is now false"
    )
    assert abs(central) > 0.0, (
        "setup: the central difference is also zero, so this configuration cannot "
        "demonstrate the dropped derivative — pick another (gamma, kTe)"
    )
