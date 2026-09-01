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
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "alpha": 2.0,
        "beta": 2.5,
        "tau_gyr": 1.0,
        "log_total_mass": 10.0,
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "tau_bc": 0.5,
        "tau_diff": tengri.Uniform(0.0, 3.0),
    },
    dust_emission={"type": "dale2014", "all_params": tengri.Fixed(tengri.DEFAULT)},
    redshift=tengri.Fixed(0.05),
)

baseline = dict(model.spec.sample(jax.random.PRNGKey(42)))

# Dust optical depths to sweep
tau_diffs = np.array([0.0, 0.3, 0.7, 1.5, 3.0])

for tau_diff in tau_diffs:
    params = {**baseline, "dust_tau_diff": jnp.float64(tau_diff)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    sed = np.asarray(out.sed)
    nu = 2.998e18 / wave
    nu_l_nu = nu * sed

    print(f"\ntau_diff={tau_diff}:")
    print(f"  SED min/max: {sed.min():.4e} / {sed.max():.4e}")
    print(f"  nu*L_nu min/max: {nu_l_nu.min():.4e} / {nu_l_nu.max():.4e}")
    print(f"  NaN count: {np.isnan(nu_l_nu).sum()}")
    print(f"  Infinite count: {np.isinf(nu_l_nu).sum()}")
