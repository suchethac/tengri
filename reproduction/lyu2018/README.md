# Live FSPS and public-template comparison

`compare_fsps.py` compares one fixed tengri model with an independently calculated reference. The stellar spectrum comes from a live `python-fsps` call; NumPy independently interpolates and normalizes the public Lyu 2018 NORMAL AGN and Haro 11 templates. Tengri is evaluated with its public forward model. This is a forward-model comparison, not a Prospector fit. The component luminosities are illustrative values, not measurements of a galaxy.

## Fixed example

Both calculations use a delayed-τ star-formation history with age 5 Gyr, τ=1 Gyr, solar metallicity, Chabrier IMF, and formed stellar mass $10^{10}\,M_\odot$. Attenuation, nebular emission, FSPS dust emission, and the FSPS AGN option are off. The added components are the Lyu 2018 `NORMAL` family at τV=5 and the Haro 11 template; each is scaled to $10^{10}\,L_\odot$. Redshift is zero.

The live FSPS build uses MIST isochrones and MILES spectra at $Z_\odot=0.0142$. The small SSP grid contains 3 metallicities, 107 ages, and 5,994 wavelength points. Tengri's solar-metallicity interpolation uses 0.001 dex scatter and node weights `[0, 0, 1]`, so it uses the solar FSPS spectrum without mixing neighboring metallicity nodes. This grid is only for the fixed-solar comparison.

FSPS labels its native luminosity unit with $L_\odot=3.839\times10^{33}$ erg s⁻¹, while tengri uses $3.828\times10^{33}$ erg s⁻¹. The SSP loader converts the FSPS values by $3.839/3.828$, and tengri then applies its own solar-luminosity constant. This preserves the FSPS stellar spectrum in cgs units. The AGN and Haro 11 amplitudes use tengri's $3.828\times10^{33}$ erg s⁻¹ convention. Spectra are reported as $L_\nu$ in erg s⁻¹ Hz⁻¹ at rest-frame wavelengths.

The AGN luminosity sets the integral of the unreddened τV=0 reference spectrum; the τV=5 spectrum uses the same $C_0$. Haro 11 is sampled from its public source rows above 5 μm and normalized over that valid part of the tengri wavelength grid. Neither normalization defines an 8–1000 μm luminosity.

## Recorded comparison

The comparison covers 23 GALEX-to-SCUBA-2 bands, integrating the independent reference and tengri spectra through the same filter curves. All 23 bands fall within illustrative 5% errors; 5% provides scale for the residuals and is not a pass/fail threshold.

| Comparison | Median absolute fractional residual | 95th percentile | Maximum |
| --- | ---: | ---: | ---: |
| 23 broadband measurements | 0.000278% | 0.003798% | 0.003970% |
| 4,312 optical samples, 3,500–9,000 Å | 0.002145% | 0.003064% | 0.003612% |

The AGN normalization $C_0$, recomputed from the public τV=0 spectrum, matches the value stored with that template. Haro 11 normalization used 2,301 source rows above 5 μm. The reference photometry is synthetic, not observed galaxy data. This comparison covers one solar-metallicity, zero-redshift example and does not establish agreement for other SFHs, metallicities, attenuation settings, or parameter fits.

## Running the comparison

Use an environment with an editable tengri install and project dependencies. The comparison also requires `python-fsps` and the FSPS data files. Set `SPS_HOME` to the FSPS data directory. Run from the tengri package root:

```bash
SPS_HOME=/path/to/fsps \
JAX_PLATFORMS=cpu \
python reproduction/lyu2018/compare_fsps.py \
  --output-dir /path/to/comparison
```

The required output directory receives `comparison_results.json`, `broadband_sed.png`, and `optical_spectrum.png`. The script builds a small SSP HDF5 grid from live FSPS unless an existing file is passed with `--ssp-grid`. Generated files are not included in the repository.

## Sampler execution check

`validate_fitting.py` builds a separate synthetic 12-band example with 5% errors. It samples formed stellar mass, AGN reference luminosity, polar-dust optical depth, and Haro 11 luminosity while redshift and SFH shape stay fixed. Each run uses one chain, 100 warmup steps, and 50 retained draws. HMC uses 10 leapfrog steps; NUTS allows up to six tree doublings. The script records diagnostics and checks that all draws are finite and each parameter varies. These checks show that the fit ran; they do not establish convergence or reliable posterior uncertainties.

The recorded HMC and NUTS runs each produced finite, varying draws for all four parameters and reported zero warmup and sampling divergences. NUTS used at most four tree doublings. Convergence and posterior accuracy were not assessed.

Run each method in an environment where JAX can see a GPU; output paths are supplied by the caller:

```bash
python reproduction/lyu2018/validate_fitting.py --method hmc --outdir /path/to/hmc
python reproduction/lyu2018/validate_fitting.py --method nuts --outdir /path/to/nuts
```

The script writes a summary JSON and compressed samples file to each output directory. `--preflight` builds the model and checks synthetic data without running a sampler; it can be used on CPU.
