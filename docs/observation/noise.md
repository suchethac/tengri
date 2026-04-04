# Noise Model

The `NoiseModel` class (formerly `NoiseConfig`) configures the noise model for the likelihood. By default,
tengri uses a standard Gaussian likelihood --- `NoiseModel` is only needed when you
want a calibration floor or a heavy-tailed likelihood.

```python
from tengri import NoiseModel, Uniform
```

## Default: Gaussian likelihood

If you do not pass a `noise` argument to `Observation`, the likelihood is a simple
Gaussian with the observational uncertainties you provide to `Fitter`:

```python
obs = Observation(photometry=Photometry.from_names(["sdss_r", "sdss_i"]))
# No noise= argument -> standard Gaussian likelihood
```

## Calibration floor

Photometric calibration is never perfect. A fractional calibration floor is added
in quadrature with the observational noise:

$$
\sigma_{\text{eff}} = \sqrt{\sigma_{\text{obs}}^2 + (f_{\text{cal}} \cdot \text{model})^2}
$$

### Fixed calibration floor

Pass a float to fix the floor to a known value:

```python
noise = NoiseModel(calibration_floor=0.05)  # 5% floor, fixed
```

This creates a `Fixed(0.05)` parameter internally.

### Free calibration floor

Pass a `Distribution` to make it a free parameter during inference:

```python
noise = NoiseModel(calibration_floor=Uniform(0.01, 0.15))
```

This adds `noise_frac_cal` as a free parameter in `Parameters` with the specified
prior. The sampler will explore values between 1% and 15%.

## Student-t likelihood

For data with outliers or poorly characterized uncertainties, a Student-t likelihood
provides heavier tails than a Gaussian. Set `student_t_dof` to the desired degrees
of freedom:

```python
noise = NoiseModel(student_t_dof=5.0)
```

Lower degrees of freedom mean heavier tails (more outlier tolerance). As
`student_t_dof` approaches infinity, the Student-t converges to a Gaussian.

:::{tip}
Common choices: `dof=5` for moderately heavy tails, `dof=2` for very heavy tails.
Values below 2 give undefined variance, which can cause inference issues.
:::

## Combining options

Calibration floor and Student-t likelihood can be used together:

```python
noise = NoiseModel(
    calibration_floor=Uniform(0.01, 0.10),
    student_t_dof=5.0,
)
obs = Observation(
    photometry=Photometry.from_names(["jwst_f200w", "jwst_f356w"]),
    noise=noise,
)
```

## How noise parameters flow into Parameters

When an `Observation` with a `NoiseModel` is passed to `SEDModel`, the parameters
are automatically generated and merged into `Parameters`:

| `NoiseModel` setting | Generated parameter | Type |
|--------------------|--------------------|------|
| `calibration_floor=0.05` | `noise_frac_cal = Fixed(0.05)` | Fixed |
| `calibration_floor=Uniform(0.01, 0.15)` | `noise_frac_cal = Uniform(0.01, 0.15)` | Free |
| `student_t_dof=5.0` | `noise_dof = Fixed(5.0)` | Fixed |

You can verify this with:

```python
obs.get_all_params()
# {'noise_frac_cal': Uniform(0.01, 0.15)}
```

:::{note}
You do not need to add `noise_frac_cal` or `noise_dof` to `Parameters` manually.
The `Observation` handles this automatically via `obs.get_all_params()`, which is
called during `SEDModel.__init__`.
:::
