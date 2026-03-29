# MAPPINGS III/V Shock Emission Models

## Citation

Allen, M. G., Groves, B. A., Dopita, M. A., Sutherland, R. S., & Kewley, L. J.
2008, ApJS, 178, 20.
"The MAPPINGS III Library of Fast Radiative Shock Models."

## Overview

The MAPPINGS III library provides a comprehensive grid of fully radiative shock
models for fast shocks ($v_s = 100$--1000 km/s) propagating through the
interstellar medium. Each model includes both the **radiative shock** itself and
its **photoionized precursor** (the region ahead of the shock ionized by EUV and
soft X-ray photons from the cooling post-shock gas). The library provides
emission line predictions separately for the shock, precursor, and
shock+precursor composite.

## Model Grid Parameters

### Velocity grid

| Parameter | Range | Spacing |
|-----------|-------|---------|
| Shock velocity $v_s$ | 100 -- 1000 km/s | 25 km/s steps (100--500), 100 km/s steps (500--1000) |

### Magnetic parameter

The magnetic field is parameterized as:

$$
\frac{B}{\sqrt{n}} \quad [\mu\text{G}\; \text{cm}^{3/2}]
$$

| Parameter | Range |
|-----------|-------|
| $B/\sqrt{n}$ | $10^{-4}$, 0.5, 1.0, 2.0, 3.23, 4.0, 5.0, 10.0 $\mu$G cm$^{3/2}$ |

The value $B/\sqrt{n} = 3.23\;\mu$G cm$^{3/2}$ corresponds to the typical
Galactic ISM value.

### Pre-shock density

| Abundance set | Densities $n_0$ (cm$^{-3}$) |
|---------------|----------------------------|
| Solar | 0.01, 0.1, 1.0, 10, 100, 1000 |
| Other (SMC, LMC, 2$\times$solar, 0.25$\times$solar) | 1.0 only |

### Abundance sets

Five atomic abundance sets are provided:

| Set | Description | Key metallicities |
|-----|-------------|-------------------|
| Solar | Allen (1973) + Grevesse & Sauval (1998) | $Z \approx Z_\odot$ |
| 2$\times$ Solar | Twice solar | $Z \approx 2 Z_\odot$ |
| LMC | Large Magellanic Cloud | $Z \approx 0.5 Z_\odot$ |
| SMC | Small Magellanic Cloud | $Z \approx 0.25 Z_\odot$ |
| 0.25$\times$ Solar | Quarter solar | $Z \approx 0.25 Z_\odot$ |

## Physical Structure

### Shock component

The radiative shock heats gas to temperatures:

$$
T_{\rm shock} \approx \frac{3}{16} \frac{\mu m_H}{k_B}\, v_s^2 \approx 1.4 \times 10^5
\left(\frac{v_s}{100\;\text{km/s}}\right)^2\;\text{K}
$$

The post-shock gas cools radiatively, emitting UV, optical, and infrared lines
as it passes through a sequence of ionization/temperature states.

### Precursor component

EUV and soft X-ray photons from the shock propagate upstream, creating a
photoionized precursor region. The precursor can contribute significantly to the
total emission, especially for strong optical lines like H$\alpha$ and
[O III]$\lambda$5007.

### Composite (shock + precursor)

For unresolved observations, the composite emission is the relevant quantity.
The relative contribution of the precursor increases with shock velocity because
faster shocks produce harder ionizing spectra.

## Emission Line Predictions

The library provides fluxes for hundreds of emission lines from UV to IR. Key
diagnostic lines include:

### Optical diagnostic lines

| Line | Wavelength (A) | Primary diagnostic use |
|------|---------------|----------------------|
| H$\beta$ | 4861 | Reference flux |
| [O III] | 5007 | Excitation indicator |
| H$\alpha$ | 6563 | SFR, flux reference |
| [N II] | 6584 | Metallicity, shock indicator |
| [S II] | 6717, 6731 | Density, shock indicator |
| [O I] | 6300 | Shock indicator (strong in shocks) |

### Representative line ratios (solar abundance, $n_0 = 1$ cm$^{-3}$, shock+precursor)

The following are representative values for the composite shock+precursor at
$B/\sqrt{n} = 3.23\;\mu$G cm$^{3/2}$:

| $v_s$ (km/s) | log([N II]/H$\alpha$) | log([O III]/H$\beta$) | log([S II]/H$\alpha$) | log([O I]/H$\alpha$) |
|---------------|----------------------|----------------------|----------------------|---------------------|
| 100 | $\sim -0.8$ | $\sim -0.5$ | $\sim -0.5$ | $\sim -2.0$ |
| 200 | $\sim -0.3$ | $\sim 0.5$ | $\sim -0.2$ | $\sim -1.0$ |
| 300 | $\sim -0.1$ | $\sim 0.8$ | $\sim -0.1$ | $\sim -0.7$ |
| 500 | $\sim 0.0$ | $\sim 0.3$ | $\sim 0.0$ | $\sim -0.5$ |
| 1000 | $\sim -0.1$ | $\sim -0.2$ | $\sim -0.1$ | $\sim -0.5$ |

**Note**: These are approximate values read from the published diagnostic
diagrams. For exact values at specific grid points, consult the online model
library at https://mappings.anu.edu.au/ or the CDS tables.

### Key trends with velocity

- **[O III]/H$\beta$**: Peaks at $v_s \sim 300$--400 km/s (highest excitation),
  declines at higher velocities as cooling becomes less efficient
- **[N II]/H$\alpha$**: Increases monotonically to $v_s \sim 500$ km/s, then
  plateaus
- **[O I]/H$\alpha$**: Strong enhancement relative to photoionization models;
  this is a key shock diagnostic
- **[S II]/H$\alpha$**: Enhanced in shocks compared to HII regions; standard
  shock diagnostic

## BPT Diagram Location

Shock models occupy a characteristic region of the BPT diagram
([O III]/H$\beta$ vs [N II]/H$\alpha$):

- **Low velocity** ($v_s < 200$ km/s): Overlaps with HII region/AGN mixing
  sequences
- **Intermediate velocity** (200--500 km/s): Extends into the LINER/Seyfert
  region
- **High velocity** ($v_s > 500$ km/s): Moves toward low [O III]/H$\beta$ and
  moderate [N II]/H$\alpha$ (LINER-like)

The shock+precursor composite typically traces a curved sequence on the BPT
diagram as velocity increases.

## Comparison to Related Models

| Model | Version | Velocity range | Physics |
|-------|---------|---------------|---------|
| **MAPPINGS III** (this work) | III | 100--1000 km/s | Radiative shock + precursor |
| MAPPINGS V | V | 10--1000 km/s | Updated atomic data, dust |
| MAPPINGS Ie | Ie | Slow shocks | Molecular cooling included |
| Hartigan+1987 | -- | Slow ($<$100 km/s) | Simpler atomic physics |

## Implementation Notes for tengri

1. **Template grid**: Pre-tabulate shock emission line luminosities on the
   $(v_s, B/\sqrt{n}, n_0, Z)$ grid from the MAPPINGS library. For most SED
   fitting applications, the composite (shock+precursor) values are appropriate.

2. **Line emission only**: MAPPINGS provides emission line fluxes, not continuum
   SEDs. For SED fitting, shock emission enters as **additional emission lines**
   superimposed on the stellar+nebular continuum.

3. **Parameterization**: For practical SED fitting, reduce to 2--3 parameters:
   - $v_s$: shock velocity (primary parameter)
   - $f_{\rm shock}$: fraction of total H$\alpha$ luminosity from shocks (vs.
     star formation)
   - $n_0$: pre-shock density (secondary; can be fixed to 1 cm$^{-3}$)

4. **Shock contribution**: In many galaxies, shocks contribute only a few
   percent of the total line emission. However, they can dominate in:
   - Radio galaxies and AGN-driven outflows
   - Post-starburst galaxies with galactic-scale winds
   - Merging systems with high-velocity gas

5. **Differentiability**: Interpolation on the shock model grid must be
   differentiable. Use linear interpolation in log $v_s$ space.

6. **Online library**: The full model library with machine-readable tables is
   available at https://mappings.anu.edu.au/ (Strasbourg mirror:
   https://cds.unistra.fr/~allen/mappings_page1).
