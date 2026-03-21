# Quick Start

A minimal example: five SDSS bands, from observation to inference.

```python
from tengri import Model, ParamSpec, Fitter, Uniform, Gaussian
from tengri import Observation, Photometry, load_ssp_data

# Load SSP grid
ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")

# Declare observation
obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]))

# Define parameters
spec = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
    sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
    sfh_tsnorm_width_gyr=Uniform(0.5, 5),
    sfh_field_psd_sigma=Uniform(0.01, 1.0),
    sfh_field_psd_tau_myr=Uniform(10, 500),
    met_logzsol=Gaussian(-0.3, 0.2),
    dust_tau_bc=Uniform(0, 4),
    redshift=0.1,
)

# Build model --- observation drives filter precomputation and data_type
model = Model(spec, ssp, observation=obs)

# Fit --- data_type is inferred from obs, no need to specify it
fitter = Fitter(model, flux_obs, noise_obs)
result = fitter.run("geovi")
print(result.summary_table())
```

The key line is:

```python
obs = Observation(photometry=Photometry.from_names([...]))
```

This single object tells `Model` which filters to precompute and tells `Fitter`
that the data is photometric. Everything else follows automatically.

:::{tip}
Call `obs.summary()` at any time to see a readable overview of the observation
configuration, including auto-generated parameters.
:::
