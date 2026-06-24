# CSP Integration: Metallicity and Age Interpolation in SED Fitting Codes

## The CSP integral

The composite stellar population (CSP) SED is a weighted sum over SSP
templates, discretized on a grid of ages and metallicities:

$$L_\text{CSP}(\lambda) = \sum_{i=1}^{n_\text{age}} \sum_{m=1}^{n_\text{met}} w_i(t) \; \phi_m(Z) \; \text{SSP}(\lambda,\, Z_m,\, t_i)$$

Two choices determine how accurately this sum approximates the true integral:

1. **Age weights** $w_i$ — how much stellar mass formed in each SSP age bin
2. **Metallicity weights** $\phi_m$ — how to distribute weight across Z grid points

This document compares the approaches used by FSPS, Prospector, Bagpipes,
CIGALE, DSPS, and tengri, quantifies the impact on photometry, colors,
and spectra, and explains when each choice matters.

---

## Metallicity convention

Two conventions exist:

| Convention | Symbol | Definition | Solar value |
|---|---|---|---|
| Absolute | log₁₀(Z) | log₁₀ of mass fraction in metals | -1.8477 |
| Solar-relative | [Z/H] or log₁₀(Z/Z☉) | offset from solar | 0.0 |

$$\log_{10}(Z)_\text{abs} = \log_{10}(Z/Z_\odot) + \log_{10}(Z_\odot)$$

where log₁₀(Z☉) = -1.8477 (Asplund et al. 2009).

**tengri convention:** The user-facing parameter `met_logzsol` is solar-relative.
Internally, the model adds the solar offset and works in absolute log₁₀(Z),
matching the SSP grid. This is identical to FSPS and DSPS. The ~10%
difference historically reported between tengri and DSPS was **not** a
convention mismatch — it was an algorithmic difference in interpolation
method.

---

## Age weight methods

The age axis typically dominates the CSP accuracy because SSP spectral
shapes change by factors of ~100 across the age grid (young blue →
old red), compared to factors of ~2--3 across the metallicity grid.

### Midpoint rule

Evaluate the SFR at each SSP age and multiply by the bin width:

$$w_i = \text{SFR}(t_i) \times \Delta t_i$$

where $\Delta t_i$ is the trapezoidal half-width. On log-spaced SSP grids,
this **underweights old populations** because the bin widths in linear time
grow exponentially. Typical mass loss: 5--15% depending on SFH shape.

Used by: tengri (parametric SFH path)

### Cumulative mass interpolation (DSPS)

Compute cumulative stellar mass $M_*(t)$ from the SFH, then difference
at SSP age bin edges:

$$w_i = M_*(t_{i+1/2}) - M_*(t_{i-1/2})$$

This is exact for any SFH shape — the total mass is conserved by
construction (Hearin et al. 2023, Eq. 9).

Used by: DSPS, tengri (table SFH path)

### Analytic piecewise integration (FSPS)

For piecewise-linear or piecewise-exponential SFH parametrizations, FSPS
integrates analytically within each age bin. Exact for the assumed
functional form.

Used by: FSPS, Prospector (via FSPS)

### Fine-grid histogram (Bagpipes)

Evaluate the SFH on a fine time grid (Δt ~ 1 Myr), bin into SSP age
intervals, and sum. Accuracy depends on the fine-grid resolution.

Used by: Bagpipes

---

## Metallicity interpolation methods

### 1. Two-point linear in log(Z) — FSPS, Prospector

Find the two grid points bracketing the target Z and linearly interpolate:

$$\text{SSP}_\text{interp}(\lambda) = (1 - f) \cdot \text{SSP}_{k}(\lambda) + f \cdot \text{SSP}_{k+1}(\lambda)$$

where

$$f = \frac{\log Z - \log Z_k}{\log Z_{k+1} - \log Z_k}, \quad k = \text{argmax}\{j : Z_j \le Z\}$$

**Properties:**
- Exact at grid points
- Piecewise-linear between grid points
- First derivative is **piecewise-constant** — discontinuous at every grid boundary
- Uses exactly 2 SSP templates per evaluation

### 2. Linear in linear Z — Bagpipes

Same as above but in linear Z:

$$f = \frac{Z - Z_k}{Z_{k+1} - Z_k}$$

Since SSP spectra vary more linearly in log(Z) than linear Z, this
introduces a small systematic bias. Typically <5% for ΔlogZ ≤ 0.25.

### 3. Nearest-neighbor — CIGALE

$$\text{SSP}_\text{interp}(\lambda) = \text{SSP}_{k^*}(\lambda), \quad k^* = \text{argmin}_k |\log Z - \log Z_k|$$

Zero gradient everywhere; step-function discontinuities at bin midpoints.
Acceptable for CIGALE's grid-based Bayesian analysis where Z is itself
discretized.

### 4. Triweight kernel — DSPS, tengri (default)

Distributes weight across **multiple** grid points using a smooth,
compact-support kernel (Hearin et al. 2023). The triweight kernel CDF:

$$F(z) = -\frac{5z^7}{69984} + \frac{7z^5}{2592} - \frac{35z^3}{864} + \frac{35z}{96} + \frac{1}{2}$$

where z = (x − μ)/σ, with support |z| < 3. Each bin weight is:

$$w_m = F\!\left(\frac{Z_\text{target} - Z_{\text{lo},m}}{\sigma}\right) - F\!\left(\frac{Z_\text{target} - Z_{\text{hi},m}}{\sigma}\right)$$

Normalized: $\hat{w}_m = w_m / \sum_j w_j$. The interpolated SSP:

$$\text{SSP}_\text{interp}(\lambda) = \sum_{m=1}^{n_\text{met}} \hat{w}_m \cdot \text{SSP}_m(\lambda)$$

**Properties:**
- C² continuous — smooth second derivative, no kinks
- Physically motivated: σ represents intrinsic metallicity scatter
- At σ → 0, converges to nearest-neighbor
- At σ = 0.1 dex (default), 3--4 bins contribute

### 5. Lognormal MDF — DSPS (full 2D mode)

Computes a full 2D weight matrix (n_met × n_age) by applying the triweight
kernel with per-age metallicity histories. Handles time-evolving Z(t) with
lognormal scatter at each epoch. tengri supports this via
`met_interp="smooth"` + `evolving_metallicity=True`.

---

## Comparison table

| Code | Age weights | Z interpolation | Z space | Z bins | Gradient |
|---|---|---|---|---|---|
| **FSPS** | analytic piecewise | 2-point linear | log(Z) | 2 | Kinks |
| **Prospector** | → FSPS | → FSPS | log(Z) | 2 | Kinks |
| **Bagpipes** | fine-grid histogram | 2-point linear | linear Z | 2 | Kinks |
| **CIGALE** | grid-based | nearest-neighbor | — | 1 | Step function |
| **DSPS** | cumulative mass | triweight CDF | log(Z) | 3--5 | C² smooth |
| **tengri** | cumulative (table) / midpoint (parametric) | triweight CDF (default) or 2-point linear | log(Z) | 3--5 or 2 | C² smooth or kinks |

---

## Impact on observables

### Broadband photometry (fluxes)

The maximum flux difference between 2-point linear and triweight smooth
across all SDSS bands and a wide metallicity range:

| log₁₀(Z/Z☉) | Max flux difference |
|---|---|
| -1.5 to -0.5 | < 0.3% |
| -0.5 to 0.0 | 0.1--0.6% |
| 0.0 to +0.2 | 0.5--1.3% |

The largest differences occur at super-solar metallicity near the SSP grid
edge, where the triweight distributes weight to more distant bins. In all
cases, the difference is **well below typical photometric uncertainties**
(2--5%).

### Broadband colors

Colors are magnitude differences between bands: color = m₁ - m₂. If the
interpolation method shifts all bands by a constant factor, colors cancel
exactly. In practice the shift is mildly wavelength-dependent, but:

| Metric | Value |
|---|---|
| Max |Δcolor| across all bands and Z | 0.009 mag |
| Typical |Δcolor| | 0.001--0.003 mag |
| SDSS photometric uncertainty | ~0.02 mag |

**Colors are unaffected.** The maximum color shift (0.009 mag in u−g at
[Z/H] = +0.2) is 2× below SDSS photometric errors and undetectable
with any current survey.

### Spectra

At spectral resolution, the differences are also small:

| Spectral region | Median Δ | Max |Δ| | RMS Δ |
|---|---|---|---|
| UV (1200--2500 Å) | 0.05% | 1.4% | 0.18% |
| Optical (3500--7000 Å) | 0.02--0.24% | 0.6% | 0.05--0.22% |
| NIR (8000--25000 Å) | 0.06--0.23% | 0.5% | 0.10--0.22% |
| D4000 break | 0.06--0.21% | 0.6% | 0.07--0.22% |
| Mg b (5175 Å) | 0.03--0.24% | 0.3% | 0.04--0.24% |
| H-alpha region | 0.05--0.23% | 0.6% | 0.08--0.24% |
| Ca II triplet | 0.04--0.27% | 0.3% | 0.05--0.26% |
| **Full 1000--25000 Å** | **0.05%** | **0.3%** | **0.06%** |

Context for spectroscopic S/N:

| S/N per pixel | Uncertainty | Interpolation detectable? |
|---|---|---|
| 20 (typical survey) | ~5% | No |
| 50 (good spectrum) | ~2% | No |
| 100 (excellent) | ~1% | No |
| 500 (stacked) | ~0.2% | Marginal |

The interpolation method difference is below noise for any individual
galaxy spectrum. It could in principle matter for very high-S/N stacked
spectra, but at that level other systematics (continuum normalization,
flux calibration, sky subtraction residuals) dominate.

### Where the difference IS detectable

The largest spectral differences (~7--8%) occur in the **extreme UV below
~110 Å** — a regime that is:
- Completely unobservable (absorbed by the ISM/IGM)
- Physically dominated by the ionizing spectrum (relevant only for
  nebular emission modeling via Q_H)

For the Q_H computation, the 2-point vs triweight difference propagates
to a ~1--2% change in the ionizing photon rate, which translates to
<1% change in emission line luminosities. Negligible.

---

## Why smooth is the default: gradient quality

The primary advantage of the triweight kernel is **gradient smoothness**,
not SED accuracy. Both methods produce indistinguishable SEDs, but the
optimization landscape differs:

| Metric | 2-point linear | Triweight smooth |
|---|---|---|
| Gradient max jump | 4.23 | 0.49 |
| Gradient RMS jump | 0.76 | 0.27 |
| **Smoothness improvement** | — | **8.5× (max), 2.8× (RMS)** |

This matters for:
- **MAP:** Avoids oscillation at Z grid boundaries
- **NUTS:** Fewer divergent transitions, better step-size adaptation
- **geoVI/EVI:** Smoother variational loss surface, faster convergence

It does **not** matter for:
- **Ray Tracing Sampler:** Uses gradients but the refraction analogy is
  robust to kinks
- **Nested sampling / Metropolis:** No gradients used
- **Grid search:** No gradients used

---

## Speed

| Path | Array shape | 2-point | Triweight | Overhead |
|---|---|---|---|---|
| Exact (full SED) | (12, 107, 7000) | ~240 μs | ~650 μs | +170% |
| Fused kernel | (12, 107, 5) | ~0.1 μs | ~0.3 μs | negligible |

**In context of the full forward model:**

| Component | Time (μs) | Share |
|---|---|---|
| SFH computation | 113 | 3% |
| SFR interpolation | 75 | 2% |
| CSP weights | 4 | 0.1% |
| **Z interpolation (smooth)** | **650** | **14%** |
| Dust attenuation | 2400 | **57%** |
| CSP SED einsum | 440 | 10% |
| Photometric integration | 280 | 6% |
| **Total (exact)** | **~4600** | |

The fused kernel (used during inference with fixed redshift) is 8--12×
faster and the Z interpolation overhead vanishes entirely:

| Path | Forward | Gradient |
|---|---|---|
| Fused (linear) | 293 μs | 68 μs |
| Fused (smooth) | 295 μs | 68 μs |
| Full model loss+grad | 75.8 ms (linear) / 76.4 ms (smooth) | +0.8% |

---

## The broader CSP integration picture

The metallicity interpolation is just one piece of the CSP integral.
The age integration typically matters **more** because SSP spectra vary
much more steeply with age (~100× across the grid) than with
metallicity (~2--3×). The hierarchy of importance:

1. **SFH shape** — the dominant source of SED variation (by far)
2. **Age integration method** — midpoint vs cumulative can differ by 5--15%
   in total mass, directly scaling the SED normalization
3. **Dust model** — changes SED shape by factors of 2--10 in the UV
4. **Metallicity interpolation** — changes SED by <1% (colors by <0.01 mag)
5. **Other physics** (nebular, IGM, AGN) — wavelength-dependent,
   can dominate in specific regimes

For accurate CSP integration, the priority order is:
- Get the SFH right (stochastic PSD model, fine time grid)
- Use exact age weights (DSPS cumulative for table SFH)
- Use a physically appropriate dust model
- The Z interpolation method is a **refinement**, not a correction

---

## tengri configuration

```python
from tengri import ParamSpec, Uniform

# Default: smooth triweight (DSPS-compatible)
spec = ParamSpec(
    met_logzsol=Uniform(-2, 0.2),
    # met_interp="smooth",    # default
    # lgmet_scatter=0.1,      # DSPS default bandwidth
)

# FSPS/Prospector-compatible
spec = ParamSpec(
    met_interp="linear",
    met_logzsol=Uniform(-2, 0.2),
)
```

The `lgmet_scatter` parameter controls the triweight bandwidth:

| Value | Bins active | Use case |
|---|---|---|
| 0.001 | 1 | Effectively nearest-neighbor |
| 0.05 | 2--3 | Mild smoothing, conservative |
| **0.1** | **3--4** | **DSPS default, recommended** |
| 0.2 | 4--6 | Broad smoothing |
| 0.3 | 5--8 | Maximum smoothing |

---

## Cross-validation with DSPS

tengri's triweight matches DSPS to machine precision:

| Scatter | max|w_tengri − w_DSPS| |
|---|---|
| 0.05 | < 10⁻¹⁶ |
| 0.1 | < 10⁻¹⁶ |
| 0.2 | < 10⁻¹⁶ |
| 0.3 | 2.7 × 10⁻⁴ |

---

## References

- Asplund, M., Grevesse, N., Sauval, A.J., & Scott, P. 2009, ARA&A, 47, 481
- Boquien, M. et al. 2019, A&A, 622, A103 (CIGALE)
- Carnall, A.C. et al. 2018, MNRAS, 480, 4379 (Bagpipes)
- Conroy, C. & Gunn, J.E. 2010, ApJ, 712, 833 (FSPS)
- Hearin, A.P. et al. 2023, MNRAS, 521, 1741 (DSPS)
- Johnson, B.D. et al. 2021, ApJS, 254, 22 (Prospector)
