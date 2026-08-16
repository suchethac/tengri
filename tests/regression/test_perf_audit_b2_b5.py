# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the 2026-05-23 audit follow-ups B2, B4, B5.

* **B2** — ``float(lnu_to_fnu(...))`` at ``utils/grid_interp.py:285`` and
  ``components/stellar/sps/precompute.py:311`` raised
  ``ConcretizationTypeError`` whenever the dust-IR / spectrum precompute
  paths ran inside a ``jax.jit`` trace. This broke every model using
  ``ForwardModel.predict(...)`` under jit when an IR-emission model was
  configured. The fix replaces ``float(lnu_to_fnu(...))`` with an
  inline ``(1+z)/(4π d_L²)`` that returns a Python float when inputs are
  concrete and a JAX scalar when traced.
* **B5** — the user-facing SSP fetch helpers in ``tengri.data`` exist and
  expose the canonical hosted catalog URL.

B3 is covered by PR #291 (multi-chain via vmap across all 7 MCMC
backends); B4 is the rewritten ``bench/scripts/benchmark_forward_model.py``
itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import tengri
from tengri import (
    FIXED,
    FREE,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    builders,
)
from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.regression_bug


# ---------------------------------------------------------------------------
# B2: lnu_to_fnu tracer leak through dust IR precompute under jit
# ---------------------------------------------------------------------------


def _ssp_path_or_skip():
    """Return a usable SSP path or skip the test."""
    import os

    for p in (
        "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
        "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    ):
        if os.path.exists(p):
            return p
    pytest.skip("No SSP data file available under data/")


@pytest.mark.parametrize(
    "emission_label",
    ["dale2014", "modified_blackbody"],
)
def test_forward_predict_under_jit_with_dust_ir(emission_label):
    """``jit(forward.predict)`` must succeed when an IR-emission model is
    configured. Before the fix this raised ``ConcretizationTypeError`` at
    ``utils/grid_interp.py:285`` (``float(lnu_to_fnu(...))``)."""
    from tengri.sps.dsps_wrapper import load_ssp_data

    ssp = load_ssp_data(_ssp_path_or_skip())
    obs = Observation(photometry=Photometry.from_names(["sdss_r", "wise_w4"]))
    if emission_label == "dale2014":
        emission = builders.dust.emission.dale2014(defaults=FIXED)
    else:
        emission = builders.dust.emission.modified_blackbody(defaults=FIXED)

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(defaults=FREE),
        dust=builders.dust.two_component(
            defaults=FIXED, law_bc="calzetti", tau_bc=Uniform(0, 1), emission=emission
        ),
        neb=builders.neb.none(),
        redshift=Fixed(0.05),
    )
    forward = ForwardModel.build(sed=model, observation=obs)
    p = {**model.spec.get_fixed_values(), **model.spec.sample(jax.random.PRNGKey(0))}

    out = assert_jit_matches_eager(lambda pp: forward.predict(pp).photometry(), p)
    jax.block_until_ready(out)

    assert out.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(out))), f"non-finite output: {out}"


# ---------------------------------------------------------------------------
# B5: tengri.data fetch helpers exist and surface the catalog URL
# ---------------------------------------------------------------------------


def test_tengri_data_module_exposes_helpers():
    """``tengri.data`` must expose ``download_ssp`` / ``list_remote_ssps`` /
    ``SSP_CATALOG_URL`` / ``local_ssp_path``."""
    from tengri import data

    for name in ("SSP_CATALOG_URL", "download_ssp", "list_remote_ssps", "local_ssp_path"):
        assert hasattr(data, name), f"tengri.data missing {name}"


def test_ssp_catalog_url_looks_sane():
    """The catalog URL must be the canonical https hostname and end in
    a directory path. Guards against typos in future edits."""
    from tengri.data import SSP_CATALOG_URL

    assert SSP_CATALOG_URL.startswith("https://")
    assert SSP_CATALOG_URL.endswith("/")
    assert "halos.as.arizona.edu" in SSP_CATALOG_URL


def test_download_ssp_rejects_path_traversal(tmp_path):
    """``download_ssp`` must reject names with separators so it cannot
    accidentally write outside the destination directory."""
    from tengri.data import download_ssp

    for bad in ("../escape.h5", "subdir/file.h5", ".hidden.h5", "a\\b.h5"):
        with pytest.raises(ValueError, match="must be a bare filename"):
            download_ssp(bad, dest_dir=tmp_path)


def test_download_ssp_returns_existing_file_without_network(tmp_path):
    """If the target file already exists and ``overwrite=False``, the
    function must return its path without making a network request."""
    from tengri.data import download_ssp

    name = "already_here.h5"
    target = tmp_path / name
    target.write_bytes(b"fake")
    # No network access at all — would raise URLError if it tried.
    path = download_ssp(name, dest_dir=tmp_path, overwrite=False, progress=False)
    assert path == target.resolve()
    assert path.read_bytes() == b"fake"


def test_cue_error_message_points_to_download_ssp():
    """The CueWNESSPError message must guide users to the new helper.
    Otherwise B5's value is invisible — the helper exists but users don't
    discover it from the error they actually hit."""
    from tengri.components.nebular.cue import CueWNESSPError

    # The class docstring also still mentions the catalog URL; the
    # check below is on the runtime message, which is the surface a user
    # facing the error sees.
    src = tengri.components.nebular.cue
    # We can't easily trigger the error without a wNE SSP; instead assert
    # the source contains the new guidance string.
    import inspect

    text = inspect.getsource(src)
    assert "from tengri.data import download_ssp" in text, (
        "CueWNESSPError message should reference tengri.data.download_ssp"
    )
    assert CueWNESSPError.__doc__ is not None
