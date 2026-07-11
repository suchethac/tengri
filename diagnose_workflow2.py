import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
import jax
import numpy as np
import tengri

warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

jax.config.update("jax_enable_x64", True)

ssp = tengri.load_ssp()
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "log_total_mass": 10.0,
        "peak_lbt_gyr": tengri.Uniform(0.5, 12.0),
        "width_gyr": tengri.Uniform(0.3, 5.0),
        "skew": tengri.Uniform(-1.0, 1.5),
        "trunc": tengri.Uniform(1.0, 10.0),
        "logzsol": tengri.Uniform(-2.0, 0.2),
    },
    dust={
        "type": "two_component",
        "tau_bc": tengri.Uniform(0.0, 2.0),
        "tau_diff": tengri.Uniform(0.0, 1.5),
        "slope": tengri.Fixed(-0.7),
    },
    redshift=tengri.Fixed(0.1),
)

key = jax.random.PRNGKey(42)
truth_params = {
    "sfh_tsnorm_peak_lbt_gyr": 2.5,
    "sfh_tsnorm_width_gyr": 1.5,
    "sfh_tsnorm_log_total_mass": 0.9,
    "sfh_tsnorm_skew": 0.2,
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": -0.1,
    "dust_tau_bc": 0.6,
    "dust_tau_diff": 0.4,
    "dust_slope": -0.7,
    "redshift": 0.1,
}

print("Truth params keys:", list(truth_params.keys()))

# Check the truth SED
truth_result = model.predict_panchromatic(truth_params)
print(f"\nTruth photometry shape: {np.array(truth_result.photometry).shape}")
print(f"Truth photometry values: {np.array(truth_result.photometry)}")

# Generate mock data
mock = model.mock(truth_params, snr=20.0, key=key)
print(f"\nMock flux shape: {mock.flux_obs.shape}")
print(f"Mock flux: {mock.flux_obs}")
print(f"Mock noise: {mock.noise}")

# Try to fit
forward = tengri.ForwardModel.build(sed=model, observation=obs)
print(f"\nForward model type: {type(forward)}")

# Evaluate at truth
truth_preds = forward.predict(truth_params)
print(f"\nTruth pred photometry: {truth_preds}")

# Try a quick fit with more output
posterior_map = forward.fit(
    mock.flux_obs,
    mock.noise,
    method="map",
    optimizer="adam",
    n_steps=100,
    verbose=True,
)

print(f"\nMAP params keys: {list(posterior_map.params.keys())}")
print(f"MAP params sample: {list(posterior_map.params.items())[:5]}")
