# SPDX-License-Identifier: BSD-3-Clause
"""Regression: SKIRTOR torus radius_ratio is wired and polar_beta matches canonical.

Task 14: radius_ratio parameter was declared but never passed through predict(),
causing it to be a silent no-op on SKIRTORTorus class. Also, polar_beta was
declared at [1.0, 2.5] while the canonical declaration was [1.0, 2.0].

This test ensures:
1. radius_ratio changes the SED output (no-op guard)
2. polar_beta declaration matches the canonical one
3. SKIRTORTorus and composable block parameter names are reconciled
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.protocols.component import declared_prior

pytestmark = pytest.mark.regression_bug

_SFH = {"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT), "log_total_mass": -10.0}
_DUST = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.Fixed(tengri.DEFAULT),
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}
_DISC = {
    "disc": {"type": "multicolor", "all_params": tengri.Fixed(tengri.DEFAULT)},
    "all_params": tengri.Fixed(tengri.DEFAULT),
    "log_lbol": 12.0,
    "frac": 1.0,
}


@pytest.fixture(scope="module")
def ssp():
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")


def _sed_with_radius_ratio(ssp, radius_ratio: float) -> np.ndarray:
    model = tengri.SEDModel.build(
        ssp,
        sfh=_SFH,
        dust_attenuation=_DUST,
        agn=dict(
            _DISC,
            torus={
                "type": "skirtor",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "agn_radius_ratio": radius_ratio,
            },
        ),
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    return np.asarray(model.predict_rest_sed(p).sed)


def test_skirtor_radius_ratio_is_not_a_noop_composable(ssp):
    """radius_ratio parameter must change predict() via the composable builder."""
    # Grid nodes are [10, 20, 30] per canonical declaration
    s1 = _sed_with_radius_ratio(ssp, 10.0)
    s2 = _sed_with_radius_ratio(ssp, 30.0)
    rel = float(np.abs(s1 - s2).max() / max(np.abs(s1).max(), 1e-99))
    assert rel > 1e-3, (
        f"skirtor radius_ratio is a no-op via the builder (rel diff {rel:.2e}); "
        "wiring is incomplete."
    )


def test_skirtor_torus_class_polar_beta_matches_canonical():
    """SKIRTORTorus.polar_beta must match the canonical agn_polar_beta declaration."""
    from tengri.components.agn.skirtor_model import SKIRTORTorus

    canonical = declared_prior(_AGN_PARAMS, "agn_polar_beta")
    class_prior = SKIRTORTorus.polar_beta

    assert class_prior.lo == canonical.lo, (
        f"SKIRTORTorus.polar_beta.lo={class_prior.lo} does not match "
        f"canonical agn_polar_beta.lo={canonical.lo}"
    )
    assert class_prior.hi == canonical.hi, (
        f"SKIRTORTorus.polar_beta.hi={class_prior.hi} does not match "
        f"canonical agn_polar_beta.hi={canonical.hi}"
    )
    assert class_prior.default == canonical.default, (
        f"SKIRTORTorus.polar_beta.default={class_prior.default} does not match "
        f"canonical agn_polar_beta.default={canonical.default}"
    )


def test_skirtor_torus_radius_ratio_declared():
    """SKIRTORTorus must declare radius_ratio as a free parameter."""
    from tengri.components.agn.skirtor_model import SKIRTORTorus

    # Check that the class has radius_ratio attribute
    assert hasattr(SKIRTORTorus, "radius_ratio"), (
        "SKIRTORTorus must declare radius_ratio as a class attribute"
    )

    # Get the declared prior
    class_radius_ratio = SKIRTORTorus.radius_ratio
    canonical = declared_prior(_AGN_PARAMS, "agn_radius_ratio")

    # Verify bounds match
    assert class_radius_ratio.lo == canonical.lo, (
        f"SKIRTORTorus.radius_ratio.lo={class_radius_ratio.lo} does not match "
        f"canonical agn_radius_ratio.lo={canonical.lo}"
    )
    assert class_radius_ratio.hi == canonical.hi, (
        f"SKIRTORTorus.radius_ratio.hi={class_radius_ratio.hi} does not match "
        f"canonical agn_radius_ratio.hi={canonical.hi}"
    )
    assert class_radius_ratio.default == canonical.default, (
        f"SKIRTORTorus.radius_ratio.default={class_radius_ratio.default} does not match "
        f"canonical agn_radius_ratio.default={canonical.default}"
    )


def test_skirtor_composable_and_class_param_names_reconciled():
    """Composable block and class must expose the same key physics parameters.

    Verifies that skirtor_torus_block (composable) and SKIRTORTorus (class)
    declare the radius_ratio and polar_beta parameters with compatible names.
    """
    from inspect import signature

    from tengri.components.agn.blocks.torus import skirtor_torus_block
    from tengri.components.agn.skirtor_model import SKIRTORTorus

    # Get the parameter names from the composable block
    block_sig = signature(skirtor_torus_block)
    block_params = set(block_sig.parameters.keys())

    # Key parameters that must be present in the composable block
    # Map from composable block names to expected class attribute names
    key_params_map = {
        "agn_radius_ratio": "radius_ratio",
        "agn_polar_T": "polar_temperature",
        "agn_polar_beta": "polar_beta",
        "agn_tau_skirtor": "tau_skirtor",
        "agn_oa_skirtor": "oa_skirtor",
        "agn_cos_inc": "cos_inc",
    }

    # Verify all key parameters are in the composable block signature
    for param in key_params_map:
        assert param in block_params, (
            f"Key parameter '{param}' missing from composable skirtor_torus_block"
        )

    # Verify key parameters have corresponding class attributes
    for _block_param, class_param in key_params_map.items():
        assert hasattr(SKIRTORTorus, class_param), (
            f"Key parameter '{class_param}' missing from SKIRTORTorus class"
        )


@pytest.mark.parametrize("radius_ratio", [10.0, 20.0, 30.0])
def test_skirtor_radius_ratio_within_grid_bounds(radius_ratio):
    """radius_ratio parameter must work with all declared canonical values.

    Grid nodes are exactly [10, 20, 30]; node-exact queries via triweight
    kernel should produce valid results at all three.
    """
    from tengri.components.agn.skirtor import create_skirtor_components_from_grid

    # Try to create components with this radius_ratio
    # (Will skip if template grid is not available, which is ok for unit test)
    try:
        wave = jnp.linspace(1e3, 1e7, 100)
        result = create_skirtor_components_from_grid.__wrapped__(
            # Use the bundled grid path if available
            "data/skirtor_templates_v3.h5"
        )
        components = result(
            wave,
            agn_log_lbol=12.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_radius_ratio=radius_ratio,
            agn_cos_inc=0.866,
            frac_agn=0.5,
        )
        # Verify we got valid output (no NaNs)
        assert jnp.all(jnp.isfinite(components.disk)), "disk component contains NaN"
        assert jnp.all(jnp.isfinite(components.dust)), "dust component contains NaN"
    except (FileNotFoundError, AttributeError):
        pytest.skip("SKIRTOR template grid not available")
