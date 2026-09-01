import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax
import jax.numpy as jnp
import numpy as np

import tengri

warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*SFHBurst.*")

# Age and metallicity grids
LOG10_AGES_GYR = np.linspace(-2.0, 1.1, 13)
AGES_GYR = 10.0**LOG10_AGES_GYR
MET_LOGZSOL = np.linspace(-2.0, 0.4, 10)

BANDS = ["galex_nuv", "sdss_u", "sdss_g", "sdss_r"]

obs = tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))
ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "peak_lbt_gyr": 1.0,
        "width_gyr": 0.05,
        "log_total_mass": 10.0,
        "skew": 0.0,
        "trunc": 13.0,
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "tau_diff": 0.0,
        "tau_bc": 0.0,
    },
    redshift=tengri.Fixed(0.05),
)

baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Just test a few points
print("Testing color grid computation:")
for i, age_gyr in enumerate(AGES_GYR[:3]):
    age_clamped = np.clip(age_gyr, 0.03, 13.0)
    for j, met_logzsol in enumerate(MET_LOGZSOL[:3]):
        p = {
            **baseline,
            "sfh_tsnorm_peak_lbt_gyr": jnp.float64(age_clamped),
            "met_logzsol": jnp.float64(met_logzsol),
        }
        flux = np.asarray(model.predict_photometry(p))

        # Compute u-r color
        flux_u = flux[1]
        flux_r = flux[3]
        mag_u = -2.5 * np.log10(np.maximum(flux_u, 1e-20))
        mag_r = -2.5 * np.log10(np.maximum(flux_r, 1e-20))
        ur_color = mag_u - mag_r

        print(
            f"  age={age_clamped:.4f} Gyr, Z={met_logzsol:.4f}: flux_u={flux_u:.4e}, flux_r={flux_r:.4e}, u-r={ur_color:.4f}"
        )
