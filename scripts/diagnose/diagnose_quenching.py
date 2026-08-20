import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
import jax
import numpy as np
import tengri

warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*SFH.*Before.*Big.*Bang.*")

ssp = tengri.load_ssp()

models = []
baselines = []
outputs = []

# Scenario 1: Nearly constant star formation
model1 = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 0.1,
        "beta": 0.1,
        "tau_gyr": 3.0,
        "log_total_mass": 10.0,
    },
    dust={"law": "power_law", "type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline1 = dict(model1.spec.sample(jax.random.PRNGKey(0)))
out1 = model1.predict_rest_sed(baseline1)
models.append(("Nearly constant", model1))
baselines.append(baseline1)
outputs.append(out1)

# Scenario 2: Exponential decline
model2 = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 1.0,
        "beta": 2.0,
        "tau_gyr": 2.0,
        "log_total_mass": 10.0,
    },
    dust={"law": "power_law", "type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline2 = dict(model2.spec.sample(jax.random.PRNGKey(1)))
out2 = model2.predict_rest_sed(baseline2)
models.append(("Exponential decline", model2))
baselines.append(baseline2)
outputs.append(out2)

# Scenario 3: Sharp truncation
model3 = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 3.0,
        "beta": 3.0,
        "tau_gyr": 1.5,
        "log_total_mass": 10.0,
    },
    dust={"law": "power_law", "type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline3 = dict(model3.spec.sample(jax.random.PRNGKey(2)))
out3 = model3.predict_rest_sed(baseline3)
models.append(("Sharp truncation", model3))
baselines.append(baseline3)
outputs.append(out3)

# Scenario 4: Recent burst
model4 = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "tsnorm",
        "*": tengri.FIXED,
        "log_total_mass": 10.0,
        "peak_lbt_gyr": 0.2,
        "width_gyr": 0.5,
        "skew": 0.3,
        "trunc": 2.0,
    },
    dust={"law": "power_law", "type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline4 = dict(model4.spec.sample(jax.random.PRNGKey(3)))
out4 = model4.predict_rest_sed(baseline4)
models.append(("Recent burst", model4))
baselines.append(baseline4)
outputs.append(out4)

for name, out in zip([m[0] for m in models], outputs):
    wave = np.asarray(out.wavelength)
    sed = np.asarray(out.sed)
    print(f"\n{name}:")
    print(f"  SED min/max: {sed.min():.4e} / {sed.max():.4e}")
    print(f"  NaN count: {np.isnan(sed).sum()}")
    print(f"  Negative count: {(sed < 0).sum()}")
