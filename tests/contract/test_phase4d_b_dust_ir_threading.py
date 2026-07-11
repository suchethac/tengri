# SPDX-License-Identifier: BSD-3-Clause
"""Dust IR emission under JIT: components self-load, JIT and non-JIT paths agree.

Since #871 the dust IR emission templates (PAHspec, Astrodust, Dale, …) are
self-loading ``SEDModelComponent`` components — they load their HDF5 grids in
``EmissionComponent.load``/``predict`` rather than being threaded through
``SEDModel._template_data_for_jit()`` (the Phase 4-D-B adapter threading was
removed with ``DustEmissionSEDComponent``). The contract these tests pin:

* ``_template_data_for_jit()`` threads **no** dust-IR template arrays for the
  emission components (they self-load); only the build-time energy-balance LUT /
  band response (MBB WavePrecomp) and nebular/AGN templates are threaded.
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

pytestmark = pytest.mark.contract

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


def _base_spec(dust_emission=None):
    """Base spec; ``dust_emission`` (a string) bakes the IR emission into the spec.

    Dust emission is configured on the Parameters spec (``spec.dust_emission``),
    not via a separate kwarg to ``SEDModel`` — the latter signature was removed.
    """
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.1),
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Fixed(0.5),
        apply_igm=False,
        dust_emission=dust_emission,
    )


def _silent_build(spec, ssp, obs, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

    dust_cfg = "modified_blackbody"
    model = _silent_build(_base_spec(dust_emission=dust_cfg), ssp_wneref, obs)
    td = model._template_data_for_jit()
    # MBB has no templates to thread.
    if td is not None:
        assert "dust_ir" not in td or td.get("dust_ir") is None, (
            f"MBB dust should not publish dust_ir template data; got {td!r}"
        )


def test_pahspec_dust_does_not_thread_template_data(ssp_wneref, obs):
    """PAHspec emission self-loads; no dust-IR template threading (#871).

    The Draine+2021 PAHspec model is a self-loading ``SEDModelComponent``
    component: its HDF5 grid loads in ``EmissionComponent.load``/``predict``, so
    ``_template_data_for_jit()`` threads no ``pahspec_*`` template arrays (the
    Phase 4-D-B adapter threading was removed with ``DustEmissionSEDComponent``).
    """

    try:
        model = _silent_build(_base_spec(dust_emission="draine2021_pah"), ssp_wneref, obs)
    except (ImportError, FileNotFoundError, KeyError):
        pytest.skip("PAHspec build unavailable")

    td = model._template_data_for_jit()
    dust_ir = (td or {}).get("dust_ir", {}) or {}
    assert not any(k.startswith("pahspec_") for k in dust_ir), (
        f"PAHspec component must self-load, not thread template arrays; got {list(dust_ir)}"
    )


def test_astrodust_dust_does_not_thread_template_data(ssp_wneref, obs):
    """Astrodust emission self-loads; no dust-IR template threading (#871).

    See the PAHspec sibling: the native HD23 astrodust implementation loads its grid in
    ``EmissionComponent.load``/``predict``, so ``_template_data_for_jit()`` threads no
    ``astrodust_*`` template arrays.
    """

    try:
        model = _silent_build(_base_spec(dust_emission="astrodust"), ssp_wneref, obs)
    except (ImportError, FileNotFoundError, KeyError):
        pytest.skip("Astrodust build unavailable")

    td = model._template_data_for_jit()
    dust_ir = (td or {}).get("dust_ir", {}) or {}
    assert not any(k.startswith("astrodust_") for k in dust_ir), (
        f"Astrodust component must self-load, not thread template arrays; got {list(dust_ir)}"
    )


# ── JIT-path bit-exactness ──────────────────────────────────────────────────


def test_jit_and_non_jit_paths_agree_with_mbb_dust(ssp_wneref, obs):
    """Sanity: MBB dust has no templates; JIT and non-JIT paths must agree."""

    dust_cfg = "modified_blackbody"
    model = _silent_build(_base_spec(dust_emission=dust_cfg), ssp_wneref, obs)
    params = {}

    via_jit = model.predict_observables_jit(params).phot_fnu
    via_nojit = model.predict_observables(params).phot_fnu

    assert jnp.allclose(via_jit, via_nojit, rtol=1e-12, atol=0), (
        "JIT and non-JIT paths diverged for MBB dust after Phase 4-D-B wiring"
    )


def test_astrodust_component_re_radiates_far_ir(ssp_wneref, obs):
    """The native astrodust implementation actually re-emits (#871).

    JIT/non-JIT agreement alone would pass even if the self-loading component
    were a silent no-op (both zero). This pins the physics end-to-end: with dust
    attenuation on (L_ir > 0), the astrodust model must add far-IR flux the
    no-emission model lacks — the guard against the silent-no-op class.
    """
    import numpy as np

    wave_obs = jnp.asarray(np.geomspace(1.0e3, 1.0e7, 2000))  # observed Å into far-IR
    m = _silent_build(_base_spec(dust_emission="astrodust"), ssp_wneref, obs)
    m0 = _silent_build(_base_spec(dust_emission=None), ssp_wneref, obs)
    lnu = np.asarray(m.predict_spectrum({}, wave_obs=wave_obs))
    lnu0 = np.asarray(m0.predict_spectrum({}, wave_obs=wave_obs))
    wo = np.asarray(wave_obs)
    fir = (wo > 1.0e5) & (wo < 1.0e7)  # ~10 μm – 1 mm (observed)
    assert np.all(np.isfinite(lnu))
    assert np.nansum(lnu[fir]) > 10.0 * np.nansum(lnu0[fir]), (
        "astrodust component added no far-IR re-emission (silent no-op)"
    )


def test_jit_and_non_jit_paths_agree_with_pahspec(ssp_wneref, obs):
    """PAHspec dust: JIT and non-JIT paths agree after threading."""

    try:
        dust_cfg = "draine2021_pah"
        model = _silent_build(_base_spec(dust_emission=dust_cfg), ssp_wneref, obs)
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

    try:
        dust_cfg = "astrodust"
        model = _silent_build(_base_spec(dust_emission=dust_cfg), ssp_wneref, obs)
    except (ImportError, FileNotFoundError, KeyError):
        pytest.skip("Astrodust template files not available")

    params = {}
    via_jit = model.predict_observables_jit(params).phot_fnu
    via_nojit = model.predict_observables(params).phot_fnu

    assert jnp.allclose(via_jit, via_nojit, rtol=1e-12, atol=0), (
        "JIT and non-JIT paths diverged for Astrodust dust after Phase 4-D-B threading"
    )
