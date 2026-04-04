# PAH Features and PAHFIT Decomposition

## Citation

Smith, J. D. T., Draine, B. T., Dale, D. A., et al. 2007, ApJ, 656, 770.
"The Mid-Infrared Spectrum of Star-forming Galaxies: Global Properties of
Polycyclic Aromatic Hydrocarbon Emission."

## Overview

PAHFIT is the standard method for decomposing mid-infrared galaxy spectra into
PAH emission features, atomic/molecular emission lines, starlight continuum, and
thermal dust continuum. The key innovation is the use of **Drude profiles** to
represent PAH features, which correctly account for the broad wings of these
features and recover 2--4 times more flux than spline-based continuum
subtraction methods.

Smith et al. (2007) applied PAHFIT to low-resolution 5--38 $\mu$m Spitzer IRS
spectra of 59 SINGS galaxies, establishing the canonical PAH feature
decomposition used throughout the literature.

## Drude Profile Formula

Each PAH emission feature is modeled as a Drude profile:

$$
I_\nu(\lambda) = \frac{b\, \gamma^2}
{\left(\lambda/\lambda_0 - \lambda_0/\lambda\right)^2 + \gamma^2}
$$

where:

| Symbol | Definition |
|--------|-----------|
| $b$ | Peak central intensity (amplitude) |
| $\lambda_0$ | Central wavelength of the feature |
| $\gamma$ | Fractional FWHM = FWHM / $\lambda_0$ |

The FWHM in wavelength units is:

$$
\mathrm{FWHM} = \gamma \times \lambda_0
$$

The **integrated power** of a Drude profile is:

$$
P = \frac{\pi}{2}\, b\, \gamma\, c / \lambda_0
$$

### Relation to Lorentzian

The Drude profile is equivalent to a Lorentzian in frequency space. In
wavelength space, the $(\lambda/\lambda_0 - \lambda_0/\lambda)$ term ensures the
correct asymmetric line shape.

## PAH Feature Parameters (Smith+2007 / PAHFIT Classic)

The following table lists the complete set of PAH dust features as defined in
PAHFIT classic (pahfit.pro), with center wavelengths and fractional FWHM values:

| Feature | $\lambda_0$ ($\mu$m) | $\gamma$ (frac. FWHM) | FWHM ($\mu$m) | Complex |
|---------|---------------------|----------------------|---------------|---------|
| PAH 5.3 | 5.27 | 0.034 | 0.179 | 5.25 $\mu$m |
| PAH 5.7 | 5.70 | 0.035 | 0.200 | 5.7 $\mu$m |
| PAH 6.2 | 6.22 | 0.030 | 0.187 | 6.2 $\mu$m |
| PAH 6.7 | 6.69 | 0.070 | 0.468 | Underlying plateau |
| PAH 7.4 | 7.42 | 0.126 | 0.935 | 7.7 $\mu$m complex |
| PAH 7.6 | 7.60 | 0.044 | 0.334 | 7.7 $\mu$m complex |
| PAH 7.8 | 7.85 | 0.053 | 0.416 | 7.7 $\mu$m complex |
| PAH 8.3 | 8.33 | 0.050 | 0.417 | 8.3 $\mu$m |
| PAH 8.6 | 8.61 | 0.039 | 0.336 | 8.6 $\mu$m |
| PAH 10.7 | 10.68 | 0.020 | 0.214 | 10.7 $\mu$m |
| PAH 11.2a | 11.23 | 0.012 | 0.135 | 11.3 $\mu$m complex |
| PAH 11.3b | 11.33 | 0.032 | 0.363 | 11.3 $\mu$m complex |
| PAH 12.0 | 11.99 | 0.045 | 0.540 | 12.0 $\mu$m |
| PAH 12.6a | 12.62 | 0.042 | 0.530 | 12.7 $\mu$m complex |
| PAH 12.7b | 12.69 | 0.013 | 0.165 | 12.7 $\mu$m complex |
| PAH 13.5 | 13.48 | 0.040 | 0.539 | 13.5 $\mu$m |
| PAH 14.0 | 14.04 | 0.016 | 0.225 | 14.0 $\mu$m |
| PAH 14.2 | 14.19 | 0.025 | 0.355 | 14.2 $\mu$m |
| PAH 15.9 | 15.90 | 0.020 | 0.318 | 15.9 $\mu$m |
| PAH 16.4 | 16.45 | 0.014 | 0.230 | 16.4 $\mu$m |
| PAH 17.0 | 17.04 | 0.065 | 1.108 | 17 $\mu$m complex |
| PAH 17.4 | 17.375 | 0.012 | 0.209 | 17 $\mu$m complex |
| PAH 17.9 | 17.87 | 0.016 | 0.286 | 17 $\mu$m complex |
| PAH 18.9 | 18.92 | 0.019 | 0.359 | 18.9 $\mu$m |
| PAH 33.1 | 33.10 | 0.050 | 1.655 | 33 $\mu$m |

**Total: 25 dust features.**

### Feature complexes

Several nominal "PAH bands" are decomposed into multiple Drude components:

| Complex | Components | Total nominal wavelength |
|---------|------------|--------------------------|
| 7.7 $\mu$m | 7.42 + 7.60 + 7.85 | Strongest PAH feature |
| 11.3 $\mu$m | 11.23 + 11.33 | Second strongest |
| 12.7 $\mu$m | 12.62 + 12.69 | Blended |
| 17 $\mu$m | 17.04 + 17.375 + 17.87 | Broad complex |

## Additional Model Components

PAHFIT includes three types of components beyond PAH features:

### Starlight continuum

A modified blackbody at $T_\star = 5000$ K representing evolved stellar
photospheric emission that contributes at $\lambda < 10\;\mu$m.

### Thermal dust continuum

Multiple modified blackbodies at temperatures:

$$
T_{\rm dust} = 35, 40, 50, 65, 90, 135, 200, 300\;\text{K}
$$

with emissivity $\propto \lambda^{-2}$ (standard $\beta = 2$).

### Emission lines

Atomic fine-structure lines ([Ne II] 12.81 $\mu$m, [Ne III] 15.56 $\mu$m,
[S III] 18.71 $\mu$m, [S III] 33.48 $\mu$m, H$_2$ lines, etc.) modeled as
unresolved Gaussians at the instrument resolution.

## Key Results from Smith+2007

### PAH luminosity fractions

For the SINGS galaxy sample, the dominant PAH features contribute to total PAH
luminosity as:

| Feature | Fraction of total PAH luminosity |
|---------|----------------------------------|
| 7.7 $\mu$m complex | $\sim$25--30% |
| 11.3 $\mu$m complex | $\sim$10--15% |
| 6.2 $\mu$m | $\sim$8--12% |
| 8.6 $\mu$m | $\sim$5--8% |
| 12.7 $\mu$m complex | $\sim$5--8% |
| 17 $\mu$m complex | $\sim$8--15% |

### PAH band ratios as diagnostics

| Ratio | Diagnostic for |
|-------|---------------|
| 6.2/7.7 | PAH ionization state |
| 7.7/11.3 | PAH ionization fraction (ionized/neutral) |
| 11.3/7.7 | PAH size distribution |
| 3.3/11.3 | PAH size (small vs large) |
| 6.2/11.3 | Combined ionization + size |

Ionized PAHs produce stronger 6.2, 7.7, and 8.6 $\mu$m bands relative to 11.3
$\mu$m. The 7.7/11.3 ratio is the most widely used ionization diagnostic.

## PAHFIT v2 (JWST era)

The modern Python implementation (PAHFIT v2, pahfit.readthedocs.io) extends the
original with:

- **Science packs**: Configurable feature sets for different source types
  (PDR pack with 66 features, classic pack with 25)
- **JWST wavelength coverage**: Extended to $\sim$1--28 $\mu$m
- **Flexible feature definitions**: Drude profile parameterized by total power
  and wavelength-based FWHM rather than amplitude and fractional FWHM
- **Multi-segment spectra**: Support for combining multiple instrument modes

## Comparison to Related Approaches

| Method | Feature representation | Features modeled | Continuum |
|--------|-----------------------|-----------------|-----------|
| **PAHFIT** (Smith+2007) | Drude profiles | 25 dust + lines | Multi-$T$ MBB |
| Spline continuum subtraction | Spline interpolation | Integrated bands | Spline |
| CAFE (Marshall+2007) | Drude profiles | Similar to PAHFIT | Physical dust |
| DL07 templates | Precomputed models | Implicit | Full dust model |

The Drude profile approach recovers 2--4 times more PAH flux than spline-based
methods because it accounts for the extended wings of the broad features.

## Implementation Notes for tengri

1. **MIR spectroscopy**: PAH feature modeling is primarily relevant when fitting
   mid-infrared spectroscopy (e.g., Spitzer IRS, JWST MIRI/MRS). For broadband
   photometry, PAH features are implicitly included in the dust emission
   templates (DL07, THEMIS, BOSA).

2. **Feature templates**: Pre-compute Drude profiles for all 25 features on the
   wavelength grid. The total PAH emission is then:

   $$
   F_{\rm PAH}(\lambda) = \sum_{r=1}^{25} \frac{b_r\, \gamma_r^2}
   {(\lambda/\lambda_r - \lambda_r/\lambda)^2 + \gamma_r^2}
   $$

   where $b_r$ are free amplitudes and $(\lambda_r, \gamma_r)$ are fixed.

3. **Amplitude parameterization**: For SED fitting, do not fit all 25 amplitudes
   independently. Instead:
   - Fit the total PAH luminosity as one parameter
   - Use fixed relative strengths (from Smith+2007 medians) or
   - Fit a few band ratios (e.g., 7.7/11.3 for ionization)

4. **Differentiability**: Drude profiles are smooth, analytic functions,
   trivially compatible with JAX autodiff.

5. **Wavelength coverage**: The features span 5--33 $\mu$m. Ensure the
   wavelength grid extends to cover the 33.1 $\mu$m feature if needed.

6. **Silicate absorption**: At 9.7 and 18 $\mu$m, silicate absorption features
   overlap with PAH emission. PAHFIT handles this by fitting the extinction
   separately, but in tengri, this interplay would need to be modeled
   consistently with the dust attenuation module.
