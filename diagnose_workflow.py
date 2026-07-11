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
mock = model.mock(truth_params, snr=20.0, key=key)

print("Running MAP fit...")
forward = tengri.ForwardModel.build(sed=model, observation=obs)
posterior_map = forward.fit(
    mock.flux_obs,
    mock.noise,
    method="map",
    optimizer="adam",
    n_steps=300,
    verbose=True,
)

sfh_truth = model.predict_sfh(truth_params)
sfh_map = model.predict_sfh(posterior_map.params)

t_gyr_truth = np.array(sfh_truth["t_gyr"])
sfr_truth = np.array(sfh_truth["sfr_mean"])

t_gyr_map = np.array(sfh_map["t_gyr"])
sfr_map = np.array(sfh_map["sfr_mean"])

print(f"\nTruth SFR min/max: {sfr_truth.min():.4e} / {sfr_truth.max():.4e}")
print(f"MAP SFR min/max: {sfr_map.min():.4e} / {sfr_map.max():.4e}")

mask = t_gyr_truth < 5.0
print(f"\nTruth SFR (t<5 Gyr) min/max: {sfr_truth[mask].min():.4e} / {sfr_truth[mask].max():.4e}")
print(f"MAP SFR (t<5 Gyr) min/max: {sfr_map[mask].min():.4e} / {sfr_map[mask].max():.4e}")

print(f"\nTruth SFR (t<5 Gyr) values: {sfr_truth[mask][:10]}")
print(f"MAP SFR (t<5 Gyr) values: {sfr_map[mask][:10]}")
