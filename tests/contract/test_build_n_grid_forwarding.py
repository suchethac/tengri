# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``SEDModel.build()`` forwards top-level settings such as ``n_grid``.

Protects the build() public-API surface: top-level parameter settings
(``_TOP_LEVEL_SETTINGS`` — ``n_grid``, ``redshift``, ``apply_igm``) must reach
``parse_groups`` / the ``Parameters`` spec, not ``SEDModel.__init__``.

Regression for the bug (fixed in PR #518) where ``build()`` forwarded unknown
keywords straight to ``__init__``, so ``SEDModel.build(..., n_grid=128)`` raised
``TypeError: __init__() got an unexpected keyword argument 'n_grid'`` and users
had to fall back to ``parse_groups(...) + SEDModel(...)``.
"""

import pytest

pytestmark = pytest.mark.contract

from tengri.parameters import DEFAULT, Fixed


def _build(ssp, obs, **kw):
    from tengri import SEDModel

    # A composed "field" SFH is what makes n_grid meaningful (the GP latent).
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": ["const", "field"], "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        **kw,
    )


def test_build_forwards_n_grid(synthetic_ssp, simple_observation):
    """``n_grid`` reaches the spec through build() (previously a TypeError)."""
    model = _build(synthetic_ssp, simple_observation, n_grid=32)
    assert model.spec.n_grid == 32


def test_build_n_grid_does_not_raise_typeerror(synthetic_ssp, simple_observation):
    """The exact failure mode the fix removes: no TypeError from __init__."""
    try:
        _build(synthetic_ssp, simple_observation, n_grid=64)
    except TypeError as exc:  # pragma: no cover - regression guard
        pytest.fail(f"build(n_grid=...) leaked to __init__ and raised: {exc}")


def test_build_n_grid_distinct_values(synthetic_ssp, simple_observation):
    """Each n_grid value is honored independently (not silently clamped)."""
    for n in (16, 64, 128):
        assert _build(synthetic_ssp, simple_observation, n_grid=n).spec.n_grid == n
