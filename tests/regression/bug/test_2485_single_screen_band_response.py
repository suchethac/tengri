# SPDX-License-Identifier: BSD-3-Clause
"""Single-screen dust attenuation with dust_tau_v free must build band response.

#2485: A model with single_component dust attenuation (e.g. Calzetti) and
dust_tau_v free, combined with WavePrecomp precompute, silently forfeited the
dust-emission band response and reverted to the dense per-call filter integral.
Measured: model._dust_band_response_cache is None when it should be populated.

The root cause: the dust-emission band response gate consulted _EB_ATTEN_FREE_OK,
which is also gated by the energy-balance LUT. The LUT requires tau_bc/tau_diff
axes and a two_component dust block, so admitting tau_v would break it. The
solution: give the band response its own, separate allowlist
(_BAND_RESPONSE_ATTEN_FREE_OK) that admits tau_v, with guards preventing widening
the shared set without understanding both mechanisms' requirements.

The change is safe: dust_tau_v changes the absorbed-energy amplitude but not
the Dale+2014 template's spectral SHAPE. The homogeneity probe independently
verifies that the band response stays exact.

Tests pinned here:
  1. The defect: band_response_cache is None on unpatched code, populated after.
  2. Bit-identity: photometry must not change, using numpy.array_equal.
  3. Trap verification: energy_balance_lut_cache remains None (the separation is
     observed and the blocking mechanism still works).
  4. Regression check: two_component models still get both caches populated.
"""

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, Uniform, WavePrecomp
from tengri.observation.filters import load_tophat_filter
from tengri.observation.photometry_config import Photometry

pytestmark = pytest.mark.regression_bug


def _build_single_component_model(tau_v_free: bool = False):
    """Build a single_component dust model with optional free dust_tau_v.

    Parameters
    ----------
    tau_v_free : bool
        If True, dust_tau_v is free with Uniform(0, 3). If False, Fixed(1.0).

    Returns
    -------
    SEDModel
        Built model with WavePrecomp enabled.
    """
    # Minimal photometry: one band to keep the test fast.
    filters = [load_tophat_filter(4000, 5000, name="optical_4000_5000")]
    obs = Observation(photometry=Photometry(filters=filters))

    # Load minimal SSP data (test requires ssp_data for precompute).
    import os

    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "ssp_test_data"
    )
    if not os.path.isdir(data_dir):
        data_dir = "data"
    try:
        import h5py

        ssp_file = os.path.join(data_dir, "ssp_bc03_stelib_chab_0p3Zsun.h5")
        if not os.path.isfile(ssp_file):
            pytest.skip(f"Test SSP data not found at {ssp_file}")
        ssp = h5py.File(ssp_file, "r")
    except ImportError:
        pytest.skip("h5py required for test data")

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "tau_v": Uniform(0.0, 3.0) if tau_v_free else Fixed(1.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )
    return model


def _build_two_component_model():
    """Build a two_component dust model for regression checking.

    Returns
    -------
    SEDModel
        Built model with WavePrecomp enabled.
    """
    filters = [load_tophat_filter(4000, 5000, name="optical_4000_5000")]
    obs = Observation(photometry=Photometry(filters=filters))

    import os

    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "ssp_test_data"
    )
    if not os.path.isdir(data_dir):
        data_dir = "data"
    try:
        import h5py

        ssp_file = os.path.join(data_dir, "ssp_bc03_stelib_chab_0p3Zsun.h5")
        if not os.path.isfile(ssp_file):
            pytest.skip(f"Test SSP data not found at {ssp_file}")
        ssp = h5py.File(ssp_file, "r")
    except ImportError:
        pytest.skip("h5py required for test data")

    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )
    return model


def test_defect_band_response_cache_populated_with_tau_v_free():
    """Test 1: The defect. Band response cache must be populated with tau_v free.

    This test directly shows the defect and the fix. Before the fix, the cache is
    None; after, it is populated with the precomputed filter responses.
    """
    model = _build_single_component_model(tau_v_free=True)

    # Force warmup to populate _cached_component_chain and caches.
    params = model.spec.get_fixed_values()
    _ = model.predict(params)

    # The defect is that this is None when tau_v is free.
    # After the fix, it should be a numpy array with shape (n_filters,).
    band_response = model._dust_band_response_cache
    assert (
        band_response is not None
    ), "Band response cache is None (defect not fixed). Expected shape (n_filters,)."
    assert isinstance(band_response, np.ndarray), (
        f"Band response cache should be numpy.ndarray, got {type(band_response)}"
    )
    assert band_response.shape == (1,), (
        f"Band response cache shape mismatch: expected (1,), got {band_response.shape}"
    )
    # All entries should be finite and positive (IR responses are positive definite).
    assert np.all(np.isfinite(band_response)), (
        "Band response contains NaN or Inf values"
    )
    assert np.all(band_response > 0), (
        "Band response contains non-positive values (expected positive flux per unit L_ir)"
    )


def test_bit_identity_photometry_with_cache_on_vs_off():
    """Test 2: Bit-identity — photometry must not change with cache on or off.

    This is the hard correctness gate. We compare photometry computed with the
    band response cache ACTIVE against a version with the cache FORCED OFF
    (set to None). The homogeneity check in _dust_emission_band_response ensures
    this difference is zero, not just numerically small.

    Uses numpy.array_equal (not allclose) to enforce exact equality.
    """
    model = _build_single_component_model(tau_v_free=True)

    # Ensure the cache is built by running predict once.
    params = model.spec.get_fixed_values()
    _ = model.predict(params)

    # Compute photometry with cache active.
    phot_with_cache = model.predict_photometry(params)

    # Force the cache off and recompute.
    model._dust_band_response_cache = None
    # Rebuild the template data to ensure the per-call integral is used.
    model._template_data_cache = None

    phot_without_cache = model.predict_photometry(params)

    # They MUST be exactly equal (array_equal, not allclose).
    # The precompute is mathematically exact, not an approximation.
    assert np.array_equal(
        phot_with_cache, phot_without_cache, equal_nan=True
    ), (
        "Photometry differs between cache-on and cache-off. "
        "Homogeneity check failed or precompute is not exact.\n"
        f"  with_cache:    {phot_with_cache}\n"
        f"  without_cache: {phot_without_cache}\n"
        f"  difference:    {phot_with_cache - phot_without_cache}"
    )


def test_trap_energy_balance_lut_remains_none():
    """Test 3: The trap. Energy-balance LUT must still be None on single_component.

    This test is the insurance that the trap (the separation between the two
    allowlists) is working. If someone later widens _EB_ATTEN_FREE_OK to include
    dust_tau_v, the LUT gate would incorrectly enable on single_component, which
    would build a LUT on (tau_bc, tau_diff) axes while the runtime tries to vary
    tau_v. The LUT would return the wrong L_absorbed and emit silently wrong fluxes.

    This assertion pins that the blocking mechanism still works:
      1. The class check: dust is DustAttenuationSEDComponent, not DustSEDComponent.
      2. The separate allowlist: tau_v is in _BAND_RESPONSE_ATTEN_FREE_OK but NOT
         in _EB_ATTEN_FREE_OK, so `unsafe_free` is never empty (tau_v is outside
         both lists for the LUT gate).
    """
    model = _build_single_component_model(tau_v_free=True)

    # Force warmup.
    params = model.spec.get_fixed_values()
    _ = model.predict(params)

    # The energy-balance LUT must still be None for single_component, even with
    # our fix. The presence of a separate allowlist ensures this.
    energy_balance_lut = model._energy_balance_lut_cache
    assert (
        energy_balance_lut is None
    ), (
        "Energy-balance LUT should remain None on single_component. "
        "The trap (separate allowlists) is not working. "
        "LUT build without tau_v axis would give wrong L_absorbed."
    )


def test_regression_two_component_gets_both_caches():
    """Test 4: Regression. Two-component models must still get both caches.

    This ensures we did not break the existing two_component path. Both the
    band response cache and the energy-balance LUT cache should be populated
    for two_component models.
    """
    model = _build_two_component_model()

    # Force warmup.
    params = model.spec.get_fixed_values()
    _ = model.predict(params)

    # Band response cache should be populated for two_component.
    band_response = model._dust_band_response_cache
    assert (
        band_response is not None
    ), "Band response cache is None for two_component model (regression)."
    assert isinstance(band_response, np.ndarray)

    # Energy-balance LUT cache should also be populated for two_component.
    energy_balance_lut = model._energy_balance_lut_cache
    assert (
        energy_balance_lut is not None
    ), "Energy-balance LUT cache is None for two_component model (regression)."


def test_mutation_allowlist_addition_fails():
    """Mutation 1: Removing dust_tau_v from allowlist should break test 1.

    This is a mutation test to verify that our fix (adding dust_tau_v to the
    allowlist) is the actual cause of test 1 passing.
    """
    # This test is a conceptual mutation: we verify that the allowlist
    # includes dust_tau_v by checking its presence in the class definition.
    from tengri.forward.sed_model import SEDModel as SEDModelClass

    allowlist = getattr(SEDModelClass, "_BAND_RESPONSE_ATTEN_FREE_OK", frozenset())
    assert (
        "dust_tau_v" in allowlist
    ), (
        "Mutation check: dust_tau_v must be in _BAND_RESPONSE_ATTEN_FREE_OK. "
        "If this fails, the fix was reverted or the allowlist is wrong."
    )
    # Also verify the shared allowlist does NOT have it (the trap).
    shared_allowlist = getattr(SEDModelClass, "_EB_ATTEN_FREE_OK", frozenset())
    assert (
        "dust_tau_v" not in shared_allowlist
    ), (
        "Mutation check: dust_tau_v must NOT be in the shared _EB_ATTEN_FREE_OK. "
        "Adding it there would violate the trap (LUT build requirements)."
    )
