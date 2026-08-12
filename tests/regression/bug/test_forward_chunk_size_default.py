# SPDX-License-Identifier: BSD-3-Clause
r"""One galaxy per dispatch is a bug, not a slow path (#1189).

Five entry points shipped ``forward_chunk_size=1`` — both ``PopulationFitter``
VI paths, ``CatalogFitter.run``, ``catalog.py``, and (on the other axis)
``n_chains=1``. The machinery to batch existed at every one of them
(``lax.map(..., batch_size=K)``); only the *default* was sequential, so the
accelerator was asked to do one galaxy's work per launch.

These tests pin the **policy**, not a particular K: an explicit request always
wins (reproducibility), the budget divides across ``K * n_chains`` (chains are a
second vmap axis and the NUTS warmup OOM is real), and the cases where widening
would break something resolve to 1 rather than to an error.
"""

import pytest

from tengri.inference._batching import AUTO, DEFAULT_MEMORY_BUDGET_GB, resolve_forward_chunk_size

pytestmark = pytest.mark.regression_bug

_COMMON = {"n_gal": 1000, "n_data_per_gal": 50}


def test_auto_batches_more_than_one_galaxy():
    """The whole point: the default must not be one galaxy per dispatch."""
    k = resolve_forward_chunk_size(AUTO, **_COMMON)
    assert k > 1, (
        f"auto resolved to K={k} — a dispatch still carries one galaxy, which is "
        "the anti-pattern this module exists to remove"
    )


def test_none_is_treated_as_auto():
    """``None`` must not silently mean 1."""
    assert resolve_forward_chunk_size(None, **_COMMON) == resolve_forward_chunk_size(
        AUTO, **_COMMON
    )


def test_an_explicit_request_always_wins():
    """A caller that measured its own machine must stay reproducible."""
    assert resolve_forward_chunk_size(4, **_COMMON) == 4
    assert resolve_forward_chunk_size(1, **_COMMON) == 1, (
        "an explicit K=1 must remain honored — it is how a user opts back into "
        "O(1) peak memory, and how a heterogeneous catalog is handled today"
    )


def test_k_never_exceeds_the_number_of_galaxies():
    assert resolve_forward_chunk_size(AUTO, n_gal=3, n_data_per_gal=50) == 3
    assert resolve_forward_chunk_size(999, n_gal=3, n_data_per_gal=50) == 3


def test_chains_divide_the_same_budget():
    """Chains are a second vmap axis; ignoring them is how the OOM comes back.

    ``n_chains`` multiplies the live activation set exactly as K does. A helper
    that sized K against the full budget would hand back a K that OOMs the
    moment somebody asks for four chains — the documented NUTS warmup failure
    (20+ GB at D ~ 8 with a dense mass matrix).
    """
    one = resolve_forward_chunk_size(AUTO, **_COMMON, n_chains=1)
    four = resolve_forward_chunk_size(AUTO, **_COMMON, n_chains=4)
    assert four < one, f"K did not shrink with n_chains ({one} -> {four})"
    # 1/n_chains up to integer flooring — the axes multiply, so the budget a
    # single chain gets is the whole budget divided by the number of chains.
    assert abs(four - one / 4) <= 1, f"K should scale as 1/n_chains ({one} -> {four})"


def test_heterogeneous_catalogs_fall_back_to_one():
    """``K > 1`` requires equal ``n_data``; the callers raise otherwise.

    So auto must resolve to 1 here rather than to a width that turns a catalog
    which fits today into a ``ValueError``.
    """
    assert resolve_forward_chunk_size(AUTO, **_COMMON, homogeneous=False) == 1


def test_unknown_shape_falls_back_to_one():
    """Guessing a width from an unknown shape is how an OOM ships."""
    assert resolve_forward_chunk_size(AUTO, n_gal=1000, n_data_per_gal=None) == 1
    assert resolve_forward_chunk_size(AUTO, n_gal=1000, n_data_per_gal=0) == 1


def test_budget_is_pinnable_and_monotone():
    """A bigger pinned budget must buy a bigger K — that is the knob's contract."""
    small = resolve_forward_chunk_size(AUTO, **_COMMON, memory_budget_gb=0.5)
    large = resolve_forward_chunk_size(AUTO, **_COMMON, memory_budget_gb=8.0)
    assert large > small, f"budget is not monotone in K ({small} -> {large})"


def test_budget_env_var_is_honored(monkeypatch):
    """Pinning without touching call sites."""
    monkeypatch.setenv("TENGRI_FORWARD_MEMORY_BUDGET_GB", "8.0")
    env = resolve_forward_chunk_size(AUTO, **_COMMON)
    monkeypatch.delenv("TENGRI_FORWARD_MEMORY_BUDGET_GB")
    default = resolve_forward_chunk_size(AUTO, **_COMMON)
    assert env > default, "the env budget did not raise K"
    assert default == resolve_forward_chunk_size(
        AUTO, **_COMMON, memory_budget_gb=DEFAULT_MEMORY_BUDGET_GB
    )


def test_the_result_is_a_static_python_int():
    """K is ``lax.map``'s batch_size, which must be a compile-time constant."""
    k = resolve_forward_chunk_size(AUTO, **_COMMON)
    assert type(k) is int


@pytest.mark.parametrize("bad", [0, -1, "widest"])
def test_invalid_requests_raise_rather_than_silently_clamp(bad):
    with pytest.raises(ValueError):
        resolve_forward_chunk_size(bad, **_COMMON)
