# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for #496/#497/#498: helper functions reachable on tengri.*

The BAGPIPES reproduction notebook (PR #493) exposed four gaps where examples
and downstream code had to reach into private namespaces. After this fix,
every name below must be importable as ``tengri.<name>`` and appear in
``tengri.__all__``.
"""

import pytest

import tengri

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "name",
    [
        # #496
        "load_ssp",
        "load_ssp_data",
        "SSPData",
        # #497
        "igm_transmission",
        # #498
        "velocity_broaden",
        "apply_lsf",
    ],
)
def test_public_helper_exposed(name):
    assert hasattr(tengri, name), f"tengri.{name} missing — see #496/#497/#498"
    assert name in tengri.__all__, (
        f"{name!r} present on tengri but missing from __all__ — "
        "tab-completion / `from tengri import *` will skip it."
    )


def test_load_ssp_accepts_explicit_path(tmp_path):
    """``load_ssp`` accepts an absolute/relative .h5 path (closes #496).

    Reproduction notebooks ship SSP files under
    ``reproduction/<code>/_drivers/data/<name>.h5`` rather than the
    canonical ``<root>/data/``. ``load_ssp(path)`` must resolve such a
    path directly instead of only walking ancestor ``data/`` directories.
    """
    # Pure existence/dispatch check — we want load_ssp to attempt the
    # explicit path (and fail downstream if the file is bogus), not to
    # silently fall back to the ancestor-walk on a path string.
    bogus = tmp_path / "not_a_real_grid.h5"
    bogus.write_bytes(b"")  # zero-byte file, will fail on h5 read

    with pytest.raises((OSError, ValueError, RuntimeError)):
        tengri.load_ssp(str(bogus))


def test_igm_transmission_callable_at_module_top():
    """``tengri.igm_transmission(wave_obs, z)`` returns a transmission curve."""
    import jax.numpy as jnp

    T = tengri.igm_transmission(jnp.array([5000.0, 6000.0, 7000.0]), 3.0)
    assert T.shape == (3,)
    assert float(T.min()) >= 0.0
    assert float(T.max()) <= 1.0
