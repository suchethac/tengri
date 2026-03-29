# Chemical Evolution Z(t) Models

## Citations

Robotham, A. S. G., et al. 2020, MNRAS, 495, 905.
"ProSpect: generating spectral energy distributions with complex star formation
and metallicity histories."

Bellstedt, S., et al. 2020, MNRAS, 498, 5581.
"Galaxy And Mass Assembly (GAMA): a forensic SED reconstruction of the cosmic
star formation history and metallicity evolution by galaxy type."

Bellstedt, S., et al. 2021, MNRAS, 503, 3309.
"Galaxy And Mass Assembly (GAMA): The inferred mass-metallicity relation from
z = 0 to 3.5 via forensic SED fitting."

## Overview

Realistic galaxy SEDs require metallicity to **evolve with time** rather than
remain fixed. As stars form, they enrich the ISM with metals, so $Z(t)$ tracks
the star formation history. The simplest physically motivated prescriptions are
the **closed-box** and **leaky-box** (gas regulator) models from classical
chemical evolution theory, as implemented in the ProSpect SED fitting code.

## Closed-Box Model

### Assumptions

1. Galaxy is a closed system (no inflows or outflows)
2. Stars form from gas with the current ISM metallicity
3. **Instantaneous recycling**: metals from dying stars are immediately mixed
   into the ISM
4. Yield $y$ is constant in time

### Gas fraction

$$
\mu(t) = \frac{M_{\rm gas}(t)}{M_{\rm tot}} = \frac{M_{\rm gas}(t)}{M_{\rm gas}(t) + M_\star(t)}
$$

The gas mass decreases as stars form:

$$
M_{\rm gas}(t) = M_{\rm gas,0} - \int_0^t (1 - R)\, \psi(t')\, dt'
$$

where $\psi(t)$ is the star formation rate and $R$ is the **return fraction**
(mass fraction returned to the ISM by dying stars, typically $R \approx 0.4$ for
a Chabrier IMF).

### Metallicity solution

The classical closed-box solution is:

$$
\boxed{Z(t) = -y\, \ln\!\left[\mu(t)\right]}
$$

Or equivalently:

$$
Z(t) = y\, \ln\!\left[\frac{1}{\mu(t)}\right]
$$

where:

| Symbol | Definition | Typical value |
|--------|-----------|---------------|
| $Z$ | Mass fraction of metals in the gas | 0--0.05 |
| $y$ | **Yield**: mass of metals produced per unit mass locked in stars | 0.02--0.04 |
| $\mu$ | Gas fraction $M_{\rm gas}/M_{\rm tot}$ | 0--1 |

**Key property**: This solution is independent of the star formation history
$\psi(t)$. The metallicity depends **only** on the gas fraction, regardless of
how fast or slow the gas was consumed.

### Yield values

| IMF | Yield $y$ | Reference |
|-----|----------|-----------|
| Chabrier (2003) | $\sim$0.03 | Vincenzo+2016 |
| Kroupa (2001) | $\sim$0.025 | Vincenzo+2016 |
| Salpeter (1955) | $\sim$0.015 | Matteucci 2012 |

The effective yield can also be calibrated empirically from the present-day
mass-metallicity relation.

## Leaky-Box Model (Outflows)

### Outflow modification

If gas is expelled from the galaxy at a rate proportional to the SFR:

$$
\dot{M}_{\rm out} = \eta\, \psi(t)
$$

where $\eta$ is the **mass-loading factor**, the metallicity solution becomes:

$$
Z(t) = \frac{y}{1 + \eta}\, \ln\!\left[\frac{1}{\mu(t)}\right]
$$

This is equivalent to the closed-box solution with an **effective yield**:

$$
y_{\rm eff} = \frac{y}{1 + \eta}
$$

Outflows reduce the effective yield because metals are expelled along with the
gas.

## Gas Regulator Model

A more complete model includes both inflows and outflows:

$$
\dot{M}_{\rm gas} = \dot{M}_{\rm in} - (1 - R + \eta)\, \psi
$$

$$
\frac{d(Z M_{\rm gas})}{dt} = -Z(1 - R + \eta)\, \psi + y(1 - R)\, \psi + Z_{\rm in}\, \dot{M}_{\rm in}
$$

where:
- $\dot{M}_{\rm in}$ is the gas inflow rate
- $Z_{\rm in}$ is the metallicity of infalling gas (often assumed pristine,
  $Z_{\rm in} = 0$)
- $\eta$ is the outflow mass-loading factor
- $R$ is the instantaneous return fraction
- $y$ is the nucleosynthetic yield

### Equilibrium metallicity

In the equilibrium (bathtub) limit where $\dot{M}_{\rm gas} \approx 0$:

$$
Z_{\rm eq} = \frac{y}{1 + \eta - (1-R)^{-1}\, Z_{\rm in}\, \lambda_{\rm in}}
$$

where $\lambda_{\rm in} = \dot{M}_{\rm in}/\psi$ is the inflow rate normalized
to the SFR.

## ProSpect Implementation (Robotham+2020)

### Coupling Z(t) to SFH

ProSpect couples metallicity to the star formation history through the
closed-box model:

1. Define a parametric SFH: $\psi(t)$
2. Compute cumulative stellar mass: $M_\star(t) = (1-R) \int_0^t \psi(t') dt'$
3. Compute gas fraction: $\mu(t) = 1 - M_\star(t)/M_{\rm tot}$
4. Apply closed-box: $Z(t) = y_{\rm eff} \ln[1/\mu(t)]$
5. Optionally: cap $Z(t)$ at the present-day gas-phase metallicity $Z_{\rm gas,0}$

### Free parameters in ProSpect

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Present-day gas metallicity | $Z_{\rm gas,0}$ | Sets the effective yield |
| Initial gas fraction | $\mu_0$ | Usually 1 (pure gas initially) |
| Return fraction | $R$ | Fixed by IMF choice |

The effective yield is **derived** from the present-day constraint:

$$
y_{\rm eff} = \frac{Z_{\rm gas,0}}{-\ln(\mu_0^{\rm final})}
$$

This means ProSpect has **one free parameter** for the metallicity history: the
present-day gas-phase metallicity $Z_{\rm gas,0}$.

### Bellstedt+2020/2021 results

Applying ProSpect's evolving metallicity to $\sim$7000 GAMA galaxies:

- The recovered mass-metallicity relation (MZR) at $z = 0$ matches
  direct gas-phase measurements
- The MZR evolution from $z = 0$ to $z \sim 3.5$ is consistent with
  independent observations at each epoch
- Massive galaxies ($M_\star > 10^{10.5} M_\odot$) reach high metallicity early
  and evolve little, while low-mass galaxies show significant metallicity
  growth over cosmic time

## Comparison to Related Approaches

| Approach | $Z(t)$ model | Free params | SED codes |
|----------|-------------|-------------|-----------|
| **Closed-box** (ProSpect) | $Z = y \ln(1/\mu)$ | 1 ($Z_{\rm gas,0}$) | ProSpect |
| **Leaky-box** | $Z = y_{\rm eff} \ln(1/\mu)$ | 2 ($y_{\rm eff}$, $\eta$) | -- |
| **Gas regulator** | Differential equation | 3+ ($y$, $\eta$, $\lambda_{\rm in}$) | Simulations |
| Fixed metallicity | $Z(t) = Z_0$ | 1 ($Z_0$) | MAGPHYS, most codes |
| Linear ramp | $Z(t) = Z_0 + \alpha t$ | 2 ($Z_0$, $\alpha$) | Bagpipes (option) |
| Piecewise | $Z(t_i)$ at nodes | $N_{\rm nodes}$ | Prospector (option) |

## Implementation Notes for tengri

1. **Current state**: tengri uses a single metallicity parameter `met_logzsol`
   (log solar). Implementing $Z(t)$ would be a significant extension.

2. **Closed-box integration**: The closed-box model couples naturally to
   tengri's SFH:
   - Compute $M_\star(t)$ from the GP-based SFH
   - Derive $\mu(t)$ from the total baryonic mass
   - Apply $Z(t) = y_{\rm eff} \ln[1/\mu(t)]$

3. **SSP metallicity interpolation**: The DSPS SSP grid has a metallicity axis
   (`log10(Z)` absolute). With $Z(t)$, each time step may require interpolation
   to a different metallicity, replacing the current single-metallicity
   assumption.

4. **Differentiability**: The closed-box formula is smooth and differentiable.
   The main challenge is the SSP interpolation along the metallicity axis, which
   must be done with a differentiable interpolation scheme (e.g., linear or
   cubic in log Z).

5. **Free parameters**: Add one parameter: `met_z_gas_present` (present-day
   gas-phase metallicity). The yield is derived from this and the final gas
   fraction.

6. **Computational cost**: Computing $Z(t)$ adds negligible cost. The main cost
   increase is from metallicity-dependent SSP interpolation at each time step
   rather than using a single metallicity for all ages.

7. **Return fraction**: For Chabrier IMF, $R \approx 0.4$. This should be a
   fixed constant derived from the IMF choice, not a free parameter.
