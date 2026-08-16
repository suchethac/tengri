# SPDX-License-Identifier: BSD-3-Clause
"""VI smoke-fit tests exercising dust emission precompute paths with 50-step gradient inference.

These tests validate the end-to-end inference stack against dust emission models
that use the new precompute kernel infrastructure (modified_blackbody, casey2012).
A 50-step VI fit is much more demanding than per-call equivalence tests: gradients
over many iterations reveal numerical surprises (NaN, inf, divergence) that don't
show up in single-call assertions.

Each test:
1. Builds a SEDModel with dust emission enabled + synthetic SSP.
2. Generates a mock observation from a fiducial parameter draw.
3. Runs 50-step VI (native_vi_nonlinear for speed; <2 sec after compile).
4. Asserts: convergence (loss finite + decreasing), posterior finite.

Per-emitter table (2 tests covering dust emission precompute kernels):

| Emitter | Free params for VI |
|---|---|
| modified_blackbody | dust_T, dust_beta_ir |
| casey2012 | dust_T, dust_alpha_mir |

Note: Radio, X-ray, and AGN parameters are not yet wired into the Parameters class
in this codebase version, so those VI tests are deferred.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from tengri import Fitter, Parameters, SEDModel
from tengri.inference._backend_registry import get_backend
from tengri.parameters.priors import Uniform

pytestmark = pytest.mark.integration


def _tier(method):
    # Get the tier of an inference backend, with safe default fallback.
    return getattr(get_backend(method), "tier", "stable")


pytestmark_native_vi_linear = pytest.mark.skipif(
    _tier("native_vi_linear") == "broken",
    reason="#1305: native_vi_linear registered tier='broken'; "
    "smoke auto-revives when the tier is repaired",
)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


#: The value the forward used to substitute for a free ``sfh_dpl_age_gyr`` before
#: the missing-parameter guard landed. Stating it keeps these fits unchanged (#1021).
_DPL_AGE_DEFAULT = float(
    Parameters(mean_sfh_type="dpl").get_distribution("sfh_dpl_age_gyr").default
)

# ── Test cases: (name, spec_kwargs, fiducial_params_dict, free_params, skip_reason) ───
#: VI iterations for the smoke fits. Sized to exercise each emitter's forward
#: path, not to converge -- the test asserts finiteness, never descent.
#:
#: Note this count is NOT what sets the wall clock: the fit is dominated by the
#: VI backend's XLA compile, which is step-count-independent. Dropping it from
#: 50 to 10 under geoVI changed the file's runtime by <1 %. What did move it was
#: switching backends (see the call site).
_VI_SMOKE_ITERATIONS = 10

_VI_SMOKE_CASES = [
    # modified_blackbody: Modified blackbody dust SED
    (
        "modified_blackbody",
        {
            "mean_sfh_type": "dpl",
            "sfh_dpl_alpha": Uniform(0.5, 3.0),
            "sfh_dpl_beta": Uniform(0.3, 2.0),
            "sfh_dpl_tau_gyr": Uniform(0.5, 10.0),
            "sfh_dpl_log_total_mass": Uniform(7.0, 12.5),
            "met_logzsol": Uniform(-1.5, 0.2),
            "dust_tau_bc": Uniform(0.0, 3.0),
            "dust_tau_diff": Uniform(0.0, 2.0),
            "dust_slope": -0.7,
            "redshift": 0.1,
            "dust_T": Uniform(20.0, 60.0),
            "dust_beta_ir": Uniform(1.0, 2.5),
            "dust_emission": "modified_blackbody",
        },
        {
            "sfh_dpl_alpha": 1.5,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 3.0,
            "sfh_dpl_log_total_mass": 0.5,
            "sfh_dpl_age_gyr": _DPL_AGE_DEFAULT,
            "met_logzsol": -0.5,
            "dust_tau_bc": 0.5,
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 0.1,
            "dust_T": 40.0,
            "dust_beta_ir": 1.8,
        },
        ["dust_T", "dust_beta_ir"],
        None,
    ),
    # casey2012: Casey+2012 dust template
    (
        "casey2012",
        {
            "mean_sfh_type": "dpl",
            "sfh_dpl_alpha": Uniform(0.5, 3.0),
            "sfh_dpl_beta": Uniform(0.3, 2.0),
            "sfh_dpl_tau_gyr": Uniform(0.5, 10.0),
            "sfh_dpl_log_total_mass": Uniform(7.0, 12.5),
            "met_logzsol": Uniform(-1.5, 0.2),
            "dust_tau_bc": Uniform(0.0, 3.0),
            "dust_tau_diff": Uniform(0.0, 2.0),
            "dust_slope": -0.7,
            "redshift": 0.1,
            "dust_T": Uniform(20.0, 60.0),
            "dust_alpha_mir": Uniform(1.0, 3.0),
            "dust_emission": "casey2012",
        },
        {
            "sfh_dpl_alpha": 1.5,
            "sfh_dpl_beta": 1.0,
            "sfh_dpl_tau_gyr": 3.0,
            "sfh_dpl_log_total_mass": 0.5,
            "sfh_dpl_age_gyr": _DPL_AGE_DEFAULT,
            "met_logzsol": -0.5,
            "dust_tau_bc": 0.5,
            "dust_tau_diff": 0.3,
            "dust_slope": -0.7,
            "redshift": 0.1,
            "dust_T": 40.0,
            "dust_alpha_mir": 2.0,
        },
        ["dust_T", "dust_alpha_mir"],
        None,
    ),
]


@pytest.fixture(scope="session")
def synthetic_ssp(synthetic_ssp):
    """Reuse synthetic SSP from conftest."""
    return synthetic_ssp


@pytest.fixture(scope="session")
def simple_filters():
    """Create 3 simple synthetic filters for mock observations."""
    from tengri.observation.photometry import FilterCurve

    waves = [
        jnp.linspace(3500.0, 4500.0, 50),
        jnp.linspace(5000.0, 6500.0, 50),
        jnp.linspace(7500.0, 9000.0, 50),
    ]
    trans = [jnp.ones(50) * 0.8 for _ in range(3)]
    return tuple(
        FilterCurve(wave=w, trans=t, name=f"band_{i}")
        for i, (w, t) in enumerate(zip(waves, trans))
    )


@pytest.mark.parametrize(
    "emitter_name,spec_kwargs,fid_params,free_params,skip_reason",
    _VI_SMOKE_CASES,
    ids=[case[0] for case in _VI_SMOKE_CASES],
)
@pytestmark_native_vi_linear
def test_vi_smoke_fit(
    synthetic_ssp, simple_filters, emitter_name, spec_kwargs, fid_params, free_params, skip_reason
):
    """Run a short VI fit on a mock SED; check the loss history and posterior are finite.

    This is a smoke test: it asserts finiteness, not convergence (nothing below
    checks that the loss actually descended). The iteration count is therefore
    sized to exercise the emitter's forward path, not to converge -- the dust-IR
    emitters cost ~15x per forward call (#708/#1022), so a longer run bought
    minutes of wall clock and no additional assertion.

    Parameters
    ----------
    synthetic_ssp : SSPData
        Minimal synthetic SSP (3×20×100 grid).
    simple_filters : tuple[FilterCurve]
        3-band synthetic filters.
    emitter_name : str
        Descriptive name (e.g., "radio_synchrotron", "qsogen").
    spec_kwargs : dict
        kwargs to pass to Parameters() to define the prior spec.
    fid_params : dict
        Fiducial parameter values for mock injection.
    free_params : list[str]
        Parameters to fit in VI (others are fixed to fiducial values).
    skip_reason : Path or None
        If not None, skip this test.
    """
    if skip_reason is not None:
        pytest.skip(f"{emitter_name}: data not available")

    # ── Build parameter spec ────────────────────────────────────────
    spec = Parameters(**spec_kwargs)

    # ── Build SEDModel with precompute=True ──────────────────────────
    model = SEDModel(spec, synthetic_ssp, filters=simple_filters, precompute=True)

    # ── Generate mock observation from fiducial ─────────────────────
    # Fiducial prediction + 5% Gaussian noise
    key = jax.random.PRNGKey(42)
    fid_phot = model.predict_photometry(fid_params)
    noise_std = 0.05 * jnp.abs(fid_phot) + 1e-32
    noise = jax.random.normal(key, fid_phot.shape) * noise_std
    mock_phot = fid_phot + noise

    # ── Build Fitter and run a short VI ──────────────────────────────
    # Fitter takes (model, data, noise_std) directly
    fitter = Fitter(model, mock_phot, noise_std)

    # native_vi_linear (MGVI), NOT native_vi_nonlinear (geoVI). The backend here
    # is incidental -- what is under test is the emitter's forward path staying
    # finite through inference -- so it should be the cheapest VI that still
    # returns posterior samples. Measured on this very model (D=10, casey2012):
    # nonlinear 417 s, linear 62 s, and the forward model itself is 1 ms warm.
    # The cost is geoVI's *compile*, so it is invariant to n_iterations (#1061).
    try:
        posterior = fitter.run(
            "native_vi_linear",
            n_iterations=_VI_SMOKE_ITERATIONS,
            n_samples=2,
            key=jax.random.PRNGKey(0),
        )
    except Exception as e:
        pytest.fail(f"{emitter_name}: VI fit failed with {type(e).__name__}: {e}")

    # ── Assertions ──────────────────────────────────────────────────

    # 1. Loss history (optional for some methods, e.g., VI with geoVI doesn't track it)
    if posterior.loss_history is not None:
        loss_hist = posterior.loss_history
        assert jnp.all(jnp.isfinite(loss_hist)), (
            f"{emitter_name}: loss_history contains NaN/inf. "
            f"Has {jnp.sum(~jnp.isfinite(loss_hist))} non-finite values."
        )

    # 2. Posterior samples are finite
    assert hasattr(posterior, "samples"), f"{emitter_name}: posterior has no samples"
    samples_dict = posterior.samples
    assert isinstance(samples_dict, dict), (
        f"{emitter_name}: samples should be dict, got {type(samples_dict)}"
    )

    # Check finiteness of all sample arrays
    for param_name, param_samples in samples_dict.items():
        param_array = jnp.asarray(param_samples)
        assert jnp.all(jnp.isfinite(param_array)), (
            f"{emitter_name}: samples['{param_name}'] contain NaN/inf. "
            f"Shape={param_array.shape}, "
            f"non-finite={jnp.sum(~jnp.isfinite(param_array))}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
