# BOSA Dust Emission Templates

## Citation

Boquien, M. & Salim, S. 2021, A&A, 653, A149.
"New-generation dust emission templates for star-forming galaxies."

## Overview

BOSA (Boquien & Salim 2021 IR templates) provides empirically calibrated
infrared dust emission templates that, for the first time, include a dependence
on both **total infrared luminosity** ($L_{\rm TIR}$) and **specific star
formation rate** (sSFR). Previous templates (Dale & Helou 2002, Chary & Elbaz
2001) were parameterized on luminosity alone, missing the independent effect of
sSFR on dust SED shape.

## Data Foundation

Templates are constructed from:
- **2584 normal star-forming galaxies** from H-ATLAS (Herschel)
- Combined with SDSS, GALEX, WISE, and 2MASS photometry
- GALEX-SDSS-WISE Legacy Catalog (GSWLC) for stellar masses and SFRs
- Redshift range: $z \approx 0.01$--0.3 (local sample)
- SED fitting with CIGALE using Draine & Li (2007) dust models

### Sample properties

| Property | Range |
|----------|-------|
| $L_{\rm TIR}$ | $\sim 10^9$--$10^{12}\, L_\odot$ |
| sSFR | Main-sequence and above, including high-$z$ analogs |
| Stellar mass | Wide range |
| Wavelength coverage | 1 $\mu$m -- 10 mm |

## Template Parameterization

### Single-property templates

Templates parameterized on $L_{\rm TIR}$ alone:

$$
\log \lambda L_\lambda(\lambda) = \alpha(\lambda) \times \log L_{\rm TIR} + \beta(\lambda)
$$

where $\alpha(\lambda)$ and $\beta(\lambda)$ are wavelength-dependent
coefficients fit to the data.

### Two-property templates

Templates parameterized on both $L_{\rm TIR}$ and sSFR:

$$
\log \lambda L_\lambda(\lambda) = \alpha(\lambda) \times \log L_{\rm TIR}
+ \gamma(\lambda) \times \log \mathrm{sSFR} + \beta(\lambda)
$$

This two-parameter form captures the independent effects of luminosity and
star-formation activity on the dust SED shape.

### Physical origin of the sSFR dependence

At fixed $L_{\rm TIR}$, galaxies with higher sSFR have:
- Warmer average dust temperatures (more intense radiation fields)
- Stronger MIR features (more compact star-forming regions)
- Different PAH-to-continuum ratios

This effect is real and not captured by luminosity-only templates.

## Single-Band $L_{\rm TIR}$ Estimation

A key practical application: estimate $L_{\rm TIR}$ from a single photometric
band:

$$
\log L_{\rm TIR} = m(b) \times \log \lambda L_\lambda(b) + n(b)
$$

With sSFR correction:

$$
\log L_{\rm TIR} = m(b) \times \log \lambda L_\lambda(b) + n(b) + s(b) \times \log \mathrm{sSFR}
$$

### Optimal wavelength ranges

| Rest-frame range | Scatter ($\sigma$) | Notes |
|------------------|--------------------|-------|
| 90--100 $\mu$m | $\sim$0.05 dex ($\sim$13%) | Best single-band estimator |
| 55--130 $\mu$m | $\sim$0.05--0.10 dex | Excellent range |
| 12--17 $\mu$m | $\sim$0.15--0.16 dex | Good MIR estimator |
| 20--25 $\mu$m | $\sim$0.16--0.20 dex | Moderate scatter |

## Template Grid Structure

### Dust model parameters (from CIGALE fitting)

| Parameter | Range | Description |
|-----------|-------|-------------|
| $U_{\min}$ | 0.1 -- 50 | Minimum starlight intensity |
| $\alpha$ | 1 -- 3 | Power-law slope of $dM_d/dU$ |
| $q_{\rm PAH}$ | 0.47% -- 7.32% | PAH mass fraction |
| $\gamma$ | $10^{-2.5}$ -- $10^{-0.3}$ | Fraction exposed to $U > U_{\min}$ |

Total grid: 124,740 models per redshift.

### Template output

- 1001 flux densities at constant separation in log space (0.004 dex)
- Wavelength range: 1 $\mu$m -- 10 mm
- Four template types:
  1. Parameterized on $L_{\rm TIR}$ alone
  2. Parameterized on SFR alone
  3. Parameterized on $L_{\rm TIR}$ + sSFR
  4. Parameterized on SFR + sSFR

## Comparison to Prior Templates

| Template set | Sample size | Parameterization | sSFR dependence | Wavelength range |
|-------------|-------------|------------------|-----------------|------------------|
| **BOSA (this work)** | 2584 | $L_{\rm TIR}$ + sSFR | Yes | 1 $\mu$m -- 10 mm |
| Dale & Helou 2002 | 69 | Single $\alpha$ parameter | No | 3--1100 $\mu$m |
| Chary & Elbaz 2001 | 105 | $L_{\rm TIR}$ | No | 0.44--1000 $\mu$m |
| Draine & Li 2007 | Model-based | $q_{\rm PAH}$, $U_{\min}$, $\gamma$ | Indirectly via $U$ | Full |
| Rieke+2009 | 11 | $L_{\rm TIR}$ | No | 5--30 $\mu$m |

Key improvements of BOSA over prior work:
- Order of magnitude larger sample than most previous efforts
- Previous templates based on IRAS/ISO selections favor high-luminosity galaxies
  with warmer SEDs than typical star-forming galaxies
- The sSFR dependence is a genuinely new dimension

## BOSA Software

The templates are distributed as a Python 3 package:

```bash
pip install bosa
```

```python
import bosa

# Generate templates parameterized on L_TIR
templates = bosa.templates(param='ltir')

# Generate templates with sSFR dependence
templates = bosa.templates(param='ltir_ssfr')

# Estimate L_TIR from single-band photometry
ltir = bosa.estimate_ltir(band='WISE4', flux=flux_value)
```

Available at: https://github.com/mboquien/bosa and
https://salims.pages.iu.edu/bosa/

## Implementation Notes for tengri

1. **Template integration**: BOSA templates can serve as an alternative to the
   existing DL07-based dust emission in tengri. The main advantage is the
   built-in sSFR dependence.

2. **Parameter mapping**: BOSA templates need only 1--2 parameters
   ($L_{\rm TIR}$ and optionally sSFR) compared to DL07's 3 parameters. The
   total dust luminosity is already determined by energy balance.

3. **sSFR computation**: tengri already computes SFR from the SFH and stellar
   mass from the SPS. The sSFR = SFR/$M_\star$ can be computed on the fly for
   template selection.

4. **Wavelength coverage**: BOSA covers 1--$10^4$ $\mu$m, which is sufficient
   for all photometric applications.

5. **Differentiability**: If using BOSA as a lookup table, interpolation between
   templates must be differentiable. Use linear interpolation in log $L_{\rm TIR}$
   and log sSFR space.

6. **High-redshift applicability**: The relations are provided up to $z = 4$,
   with the local sample including sSFR values typical of high-$z$ main-sequence
   galaxies. However, extrapolation beyond the calibration range should be
   flagged.

7. **Comparison mode**: A useful validation is to fit galaxies with both DL07
   and BOSA dust emission and compare the recovered physical parameters.
