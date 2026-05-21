"""Phase 4-D Category B tests: dust IR template data threading as JIT runtime inputs.

Extends Phase 4-C (nebular threading) to dust IR emission templates (PAHspec,
Astrodust, etc.). The contract:

* ``SEDModel._template_data_for_jit()`` returns a dict with ``"dust_ir"``
  key carrying the template arrays when the dust emission component uses
  a template-based model (PAHspec, Astrodust), ``None`` otherwise.
* ``predict_observables_jit(params)`` threads it so dust template arrays
  become JIT ``Parameter`` ops rather than baked Constants.
* JIT and non-JIT paths agree to floating-point precision.
"""

from __future__ import annotations

import pathlib
import warnings

import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed

_SSP_WNEREF = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def ssp_wneref():
    if not _SSP_WNEREF.exists():
        pytest.skip(f"wNE SSP not available at {_SSP_WNEREF}")
    return load_ssp_data(str(_SSP_WNEREF))


@pytest.fixture(scope="module")
def obs():
    return Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )


def _base_spec():
    """Base spec without dust emission."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_peak_sfr=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.1),
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Fixed(0.5),
        apply_igm=False,
    )


def _silent_build(spec, ssp, obs, dust_config=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dust_config is not None:
            kwargs["dust"] = dust_config
        return SEDModel(spec, ssp, observation=obs, **kwargs)


# ── _template_data_for_jit() contract ────────────────────────────────────────


def test_no_dust_emission_returns_no_dust_ir_template_data(ssp_wneref, obs):
    """No dust emission configured — ``_template_data_for_jit()`` has no dust_ir key."""
    model = _silent_build(_base_spec(), ssp_wneref, obs)
    td = model._template_data_for_jit()
    # Either None or an empty/minimal mapping is acceptable.
    if td is not None:
        # If non-None, it might have other keys (nebular, etc.) but not dust_ir
        assert "dust_ir" not in td or td.get("dust_ir") is None, (
            f"No dust emission should not publish dust_ir template data; got {td!r}"
        )


def test_mbb_dust_returns_no_dust_ir_template_data(ssp_wneref, obs):
    """Analytic dust (MBB) has no templates to thread."""
    from tengri.config import DustConfig

    dust_cfg = DustConfig(emission="modified_blackbody")
    model = _silent_build(_base_spec(), ssp_wneref, obs, dust_config=dust_cfg)
    td = model._template_data_for_jit()
    # MBB has no templates to thread.
    if td is not None:
        assert "dust_ir" not in td or td.get("dust_ir") is None, (
            f"MBB dust should not publish dust_ir template data; got {td!r}"
        )


def test_pahspec_dust_publishes_template_data_for_jit(ssp_wneref, obs):
    """PAHspec dust backend publishes template_data with dust_ir arrays."""
    from tengri.config import DustConfig

    try:
        dust_cfg = DustConfig(emission="draine2021_pah")
        model = _silent_build(_base_spec(), ssp_wneref, obs, dust_config=dust_cfg)
    except (ImportError, FileNotFoundError, KeyError):
        pytest.skip("PAHspec template files not available")

    td = model._template_data_for_jit()
    assert td is not None, "PAHspec should publish non-None template_data"
    assert "dust_ir" in td, (
        f"dust_ir key missing from template_data; got {td.keys() if td else 'None'}"
    )

    # Verify the required PAHspec arrays are present.
    dust_ir = td["dust_ir"]
    assert "pahspec_lgU_grid" in dust_ir, "Missing pahspec_lgU_grid"
    assert "pahspec_lnu_template" in dust_ir, "Missing pahspec_lnu_template"
    assert "pahspec_norm_per_lgU" in dust_ir, "Missing pahspec_norm_per_lgU"

    # Check they're non-empty arrays.
    assert jnp.prod(jnp.asarray(dust_ir["pahspec_lgU_grid"]).shape) > 0
    assert jnp.prod(jnp.asarray(dust_ir["pahspec_lnu_template"]).shape) > 0
    assert jnp.prod(jnp.asarray(dust_ir["pahspec_norm_per_lgU"]).shape) > 0


def test_astrodust_dust_publishes_template_data_for_jit(ssp_wneref, obs):
    """Astrodust backend publishes template_data with dust_ir arrays."""
    from tengri.config import DustConfig

    try:
        dust_cfg = DustConfig(emission="astrodust")
        model = _silent_build(_base_spec(), ssp_wneref, obs, dust_config=dust_cfg)
    except (ImportError, FileNotFoundError, KeyError):
        pytest.skip("Astrodust template files not available")

    td = model._template_data_for_jit()
    assert td is not None, "Astrodust should publish non-None template_data"
    assert "dust_ir" in td, (
        f"dust_ir key missing from template_data; got {td.keys() if td else 'None'}"
    )

    # Verify the required Astrodust arrays are present.
    dust_ir = td["dust_ir"]
    assert "astrodust_lgU_grid" in dust_ir, "Missing astrodust_lgU_grid"
    assert "astrodust_lnu_template" in dust_ir, "Missing astrodust_lnu_template"
    assert "astrodust_norm_per_lgU" in dust_ir, "Missing astrodust_norm_per_lgU"
    assert "astrodust_lnu_spinning" in dust_ir, "Missing astrodust_lnu_spinning"

    # Check they're non-empty arrays.
    assert jnp.prod(jnp.asarray(dust_ir["astrodust_lgU_grid"]).shape) > 0
    assert jnp.prod(jnp.asarray(dust_ir["astrodust_lnu_template"]).shape) > 0
    assert jnp.prod(jnp.asarray(dust_ir["astrodust_norm_per_lgU"]).shape) > 0


# ── JIT-path bit-exactness ──────────────────────────────────────────────────


def test_jit_and_non_jit_paths_agree_with_mbb_dust(ssp_wneref, obs):
    """Sanity: MBB dust has no templates; JIT and non-JIT paths must agree."""
    from tengri.config import DustConfig

    dust_cfg = DustConfig(emission="modified_blackbody")
    model = _silent_build(_base_spec(), ssp_wneref, obs, dust_config=dust_cfg)
    params = {}

    via_jit = model.predict_observables_jit(params).phot_fnu
    via_nojit = model.predict_observables(params).phot_fnu

    assert jnp.allclose(via_jit, via_nojit, rtol=1e-12, atol=0), (
        "JIT and non-JIT paths diverged for MBB dust after Phase 4-D-B wiring"
    )


def test_jit_and_non_jit_paths_agree_with_pahspec(ssp_wneref, obs):
    """PAHspec dust: JIT and non-JIT paths agree after threading."""
    from tengri.config import DustConfig

    try:
        dust_cfg = DustConfig(emission="draine2021_pah")
        model = _silent_build(_base_spec(), ssp_wneref, obs, dust_config=dust_cfg)
    except (ImportError, FileNotFoundError, KeyError):
        pytest.skip("PAHspec template files not available")

    params = {}
    via_jit = model.predict_observables_jit(params).phot_fnu
    via_nojit = model.predict_observables(params).phot_fnu

    assert jnp.allclose(via_jit, via_nojit, rtol=1e-12, atol=0), (
        "JIT and non-JIT paths diverged for PAHspec dust after Phase 4-D-B threading"
    )


def test_jit_and_non_jit_paths_agree_with_astrodust(ssp_wneref, obs):
    """Astrodust dust: JIT and non-JIT paths agree after threading."""
    from tengri.config import DustConfig

    try:
        dust_cfg = DustConfig(emission="astrodust")
        model = _silent_build(_base_spec(), ssp_wneref, obs, dust_config=dust_cfg)
    except (ImportError, FileNotFoundError, KeyError):
        pytest.skip("Astrodust template files not available")

    params = {}
    via_jit = model.predict_observables_jit(params).phot_fnu
    via_nojit = model.predict_observables(params).phot_fnu

    assert jnp.allclose(via_jit, via_nojit, rtol=1e-12, atol=0), (
        "JIT and non-JIT paths diverged for Astrodust dust after Phase 4-D-B threading"
    )
