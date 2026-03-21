# Examples

Standalone example scripts demonstrating tengri's capabilities. Each script is
self-contained, produces one figure, and can be run directly:

```bash
python examples/quickstart/plot_first_fit.py
```

## Directory structure

| Directory | Topics |
|-----------|--------|
| `quickstart/` | First fit, SED components |
| `sfh/` | Parametric SFH models, stochastic GP SFH, PSD burstiness |
| `dust/` | Attenuation laws, two-component model, IR emission |
| `photometry/` | Filter curves, photometric fitting |
| `spectroscopy/` | Spectrum fitting, spectral features |
| `inference/` | Method comparison, convergence diagnostics, corner plots |
| `agn/` | AGN disc + torus templates |
| `nebular/` | Nebular emission backends |
| `advanced/` | Hierarchical inference, gradient sensitivity, batch fitting |

## Requirements

Most physics examples (dust curves, SFH shapes) need only tengri's core
dependencies. Fitting examples additionally require SSP data at `data/ssp_fsps_v3.2.h5`.
