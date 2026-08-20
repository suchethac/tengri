import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
import jax
import jax.numpy as jnp
import numpy as np
import tengri

warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 2.0,
        "beta": 2.5,
        "tau_gyr": 1.0,
        "log_total_mass": 10.0,
    },
    dust_attenuation={"law": "power_law", 
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_bc": 0.3,
        "tau_diff": 0.2,
    }, dust_emission={"type": "dale2014", "*": tengri.FIXED},
    agn={
        "type": "composable",
        "disc": {"type": "qsogen", "*": tengri.FIXED},
    },
    redshift=tengri.Fixed(0.05),
)

baseline = dict(model.spec.sample(jax.random.PRNGKey(42)))
print("Baseline params keys:", list(baseline.keys()))
print("agn_log_lbol value:", baseline.get("agn_log_lbol", "NOT FOUND"))

for agn_log_lbol in [0.0, 10.0, 11.0, 11.5]:
    params = dict(baseline)
    params["agn_log_lbol"] = agn_log_lbol

    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    sed = np.asarray(out.sed)

    print(f"\nagn_log_lbol={agn_log_lbol}:")
    print(f"  SED min/max: {sed.min():.4e} / {sed.max():.4e}")
    print(f"  NaN count: {np.isnan(sed).sum()}")
    print(f"  Negative count: {(sed < 0).sum()}")
    print(f"  Zero count: {(sed == 0).sum()}")

    # Check if everything is NaN
    if np.all(np.isnan(sed)):
        print("  WARNING: ALL VALUES ARE NAN!")
