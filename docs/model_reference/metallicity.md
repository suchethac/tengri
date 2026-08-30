(app-met-details)=

# Metallicity Parameterization

tengri models stellar metallicity through three interpolation modes of increasing flexibility, and optionally couples metallicity to the SFH via a self-consistent chemical evolution model. The extension to $\alpha$-enhanced abundance patterns is described separately in Appendix {ref}`app-alpha-details`.

(app-sps-metallicity)=

## Metallicity Interpolation Modes

Three metallicity interpolation modes are provided:

#### Fixed (linear).

Linear interpolation in $\log_{10}(Z/Z_\odot)$ between the two nearest grid points: $$F_{\nu}(Z) = (1 - f) \, F_{\nu}(Z_i) + f \, F_{\nu}(Z_{i+1}),$$ where $f = (\log Z - \log Z_i) / (\log Z_{i+1} - \log Z_i)$ and $i$ is the index of the lower bracketing metallicity.

#### Linear ramp.

A time-dependent metallicity that increases linearly from an initial value $\log Z_0$ at the oldest age to a final value $\log Z_f$ at the present. The SSP flux at each age $t_j$ is interpolated to the metallicity $\log Z(t_j)$ using the fixed linear method above.

#### Triweight kernel (DSPS-compatible).

The metallicity is distributed across multiple grid points using the triweight kernel CDF from DSPS(Hearin et al. 2023, Eq. 10): $$\begin{aligned}
w_k = {} & \Phi_{\rm tw}\!\left(\frac{\log Z_{\rm edge,k+1} - \log Z}{\sigma_Z}\right) \nonumber \\
         & - \Phi_{\rm tw}\!\left(\frac{\log Z_{\rm edge,k} - \log Z}{\sigma_Z}\right),
\end{aligned}$$ where $\Phi_{\rm tw}$ is the cumulative triweight kernel with compact support $|x| < 3$, and $\sigma_Z$ (default 0.1 dex) controls the scatter. The kernel density is the $C^2$-smooth polynomial $$K_{\rm tw}(x) = \frac{35}{96}\left(1 - \frac{x^2}{9}\right)^{\!3}, \quad |x| < 3,$$ with $x = (\log Z - \log Z_{\rm grid})/\sigma_Z$. Compact support limits each query to a finite number of contributing grid points. The $C^2$ smoothness guarantees that second derivatives $\partial^2
f/\partial Z^2$ are continuous across grid boundaries, linear interpolation, by contrast, has discontinuous first derivatives that create kinks in the log-likelihood and slow gradient-based samplers. tengri uses the same triweight kernel for mass-remaining fraction interpolation ({ref}`app-sps-mass-remaining`) and free-redshift photometry tables ({ref}`app-sps-precompute`).

#### Grid interpolation for tabulated models.

Nebular, AGN torus, and dust emission grids use multilinear (bilinear or trilinear) interpolation in log-parameter space. All interpolation operations are implemented as JAX array operations compatible with `jax.jit` and `jax.grad`, so gradients propagate through the grid lookups. Multilinear interpolation has discontinuous second derivatives at grid boundaries; for models where this matters (e.g., the SKIRTOR torus grid), the grid spacing is chosen fine enough that the discontinuities do not affect sampler convergence.

(app-sps-chemical)=

## Chemical Evolution Models

Beyond the fixed and linearly ramped metallicity modes ({ref}`app-sps-metallicity`), tengri provides a self-consistent chemical evolution model (Bellstedt et al. 2021; Robotham et al. 2020). In the closed-box limit, the gas-phase metallicity evolves as $Z(t) = -y\,\ln\mu(t)$, where $y$ is the nucleosynthetic yield and $\mu(t) = M_{\rm gas}(t)/M_{\rm tot}$ is the gas fraction derived from the cumulative SFH. A leaky-box extension replaces $y$ with an effective yield $y_{\rm eff} = y/(1+\eta)$, where $\eta$ is the mass-loading factor. The single free parameter is the present-day gas metallicity $\log(Z_{\rm gas}/Z_\odot)$; the yield is solved to match this anchor. The resulting $Z(t)$ is interpolated onto the SSP metallicity grid at each age bin, adding negligible computational cost.

(app-alpha-details)=

# Alpha-Element Enhancement

Real galaxies do not have solar-scaled abundance patterns. Alpha elements (O, Mg, Si, Ca, Ti) are produced primarily by core-collapse supernovae on short timescales (${\sim}3$--$30$ Myr), while iron-peak elements are produced predominantly by Type Ia supernovae whose delay-time distribution extends to Gyr timescales. Stars formed before significant Ia enrichment therefore exhibit elevated $[\alpha/\mathrm{Fe}]$ ratios of $+0.3$ to $+0.5$ dex (Thomas et al. 2003; Conroy et al. 2014). This $\alpha$-enhancement is ubiquitous in massive ellipticals, the Milky Way thick disc, and high-redshift quiescent galaxies at $z > 2$ (Beverage et al. 2024). Ignoring $\alpha$-enhancement biases stellar population parameters: at fixed total metallicity, an $\alpha$-enhanced population has weaker Fe lines and stronger Mg/Ca/Ti features, which can be misinterpreted as an age or metallicity shift when only solar-scaled templates are available.

(app-sps-alpha)=

## The 4D SSP Grid

tengri supports $\alpha$-enhanced stellar population synthesis through a four-dimensional SSP grid with axes $(\mathrm{[Fe/H]},\, [\alpha/\mathrm{Fe}],\, t_{\rm age},\, \lambda)$. This replaces the effective-metallicity approximation with bilinear interpolation across pre-computed $\alpha$-enhanced templates from sMILES (Knowles et al. 2023), BPASS v2.3 (Byrne et al. 2022), or the $\alpha$-MC library (Park et al. 2024). All three libraries provide SSPs at $[\alpha/\mathrm{Fe}] = \{-0.2, 0.0, +0.2, +0.4, +0.6\}$ dex, with self-consistent treatment of $\alpha$-enhancement at both the isochrone and stellar atmosphere levels. The interpolation is bilinear in $(\log Z, [\alpha/\mathrm{Fe}])$ space, four multiplications per wavelength pixel, fully JIT-compiled and differentiable with respect to both $[\mathrm{M/H}]$ and $[\alpha/\mathrm{Fe}]$.

#### Metallicity convention.

The internal grid axis uses $[\mathrm{Fe/H}]$ (iron abundance) rather than total metallicity $[\mathrm{M/H}]$. This convention is preferred because $[\mathrm{Fe/H}]$ is directly measured by spectroscopic surveys (SDSS, GALAH, APOGEE), is the native variable of MESA/MIST stellar evolution models used by $\alpha$-MC (Park et al. 2024), and interpolating at fixed $[\mathrm{Fe/H}]$ cleanly isolates the spectral effect of varying $[\alpha/\mathrm{Fe}]$ (since the iron-line blanketing is held constant). Libraries natively parameterized in $[\mathrm{M/H}]$ (e.g., sMILES) are converted at load time using the Salaris et al. (1993) relation: $$[\mathrm{M/H}] = [\mathrm{Fe/H}] + 0.66\,[\alpha/\mathrm{Fe}]
+ 0.20\,[\alpha/\mathrm{Fe}]^2.

$$ (eq-salaris)
 At $[\alpha/\mathrm{Fe}] = +0.4$ (typical for massive ellipticals), $[\mathrm{M/H}]$ exceeds $[\mathrm{Fe/H}]$ by ${\sim}0.30$ dex.

(app-sps-alpha-evolving)=

## Time-Evolving $[\alpha/\mathrm{Fe}]$

In the simplest mode, a single $[\alpha/\mathrm{Fe}]$ is applied uniformly to all stellar populations. However, tengri also supports a time-evolving $[\alpha/\mathrm{Fe}](t)$ that captures the physical expectation from chemical evolution: old populations formed from $\alpha$-enriched gas, while young populations formed from gas with approximately solar abundance ratios. We parameterize this as a linear ramp in lookback time: $$[\alpha/\mathrm{Fe}](t_{\rm lb}) = [\alpha/\mathrm{Fe}]_{\rm young}
+ \bigl([\alpha/\mathrm{Fe}]_{\rm old}
- [\alpha/\mathrm{Fe}]_{\rm young}\bigr)
\times \frac{t_{\rm lb}}{t_{\rm universe}},

$$ (eq-alpha-evolving)
 adding one free parameter ($[\alpha/\mathrm{Fe}]_{\rm old}$, with $[\alpha/\mathrm{Fe}]_{\rm young}$ typically fixed at 0.0). Each age bin receives its own bilinear interpolation in the 4D grid.

(app-sps-alpha-fallback)=

## Effective-Metallicity Fallback

When 4D $\alpha$-enhanced grids are not available, tengri*optionally* applies the effective-metallicity approximation of Thomas et al. (2003; Vazdekis et al. 2015): $$[\mathrm{Z/H}]_{\rm eff} = [\mathrm{Fe/H}] + 0.75 \times
[\alpha/\mathrm{Fe}],

$$ (eq-alpha-correction)
 which shifts the total metallicity used for standard 3D SSP interpolation. This correction is *opt-in*: it is only applied when `met_alpha_fe` is explicitly declared as a free parameter in `Parameters` (i.e. not `Fixed`). When $[\alpha/\mathrm{Fe}]$ is fixed at zero (the default), tengri uses plain metallicity interpolation on the 3D grid with no correction, avoiding any implicit coupling between the metallicity and $\alpha$-element abundances. The approximation is adequate for broadband photometry but breaks down for spectroscopic fitting, where the spectral signatures of $\alpha$-enhancement (e.g., Mg $b$ at 5177 Å, Ca triplet at 8498--8662 Å, UV iron-line blanketing at 1500--1900 Å) are qualitatively different from a simple metallicity shift.

## References

Bellstedt, Sabine, Aaron S. G. Robotham, Simon P. Driver, et al. 2021. "Galaxy and mass assembly (GAMA): the inferred mass-metallicity relation from z = 0 to 3.5 via forensic SED fitting." 503 (3): 3309--25. <https://doi.org/10.1093/mnras/stab550>.

Beverage, Andrew G., Mariska Kriek, Charlie Conroy, Katherine A. Suess, David J. Setton, and Rachel Bezanson. 2024. "The Heavy Metal Survey. I. The Evolution of Stellar Metallicities, Abundance Ratios, and Ages of Massive Quiescent Galaxies since z $\sim$ 2." 966 (1): 1. <https://doi.org/10.3847/1538-4357/ad372d>.

Byrne, C. M., E. R. Stanway, J. J. Eldridge, L. McSwiney, and O. T. Townsend. 2022. "The dependence of theoretical synthetic spectra on $\alpha$-enhancement in young, binary stellar populations." 512 (4): 5329--38. <https://doi.org/10.1093/mnras/stac807>.

Conroy, Charlie, Genevieve J. Graves, and Pieter G. van Dokkum. 2014. "Early-type Galaxy Archeology: Ages, Chemical Abundances, and Star Formation Histories." 780 (1): 33. <https://doi.org/10.1088/0004-637X/780/1/33>.

Hearin, Andrew P., Jonás Chaves-Montero, Alex Alarcon, Matthew R. Becker, and Andrew Benson. 2023. "DSPS: Differentiable stellar population synthesis." 521 (2): 1741--56. <https://doi.org/10.1093/mnras/stad456>.

Knowles, Adam T., A. E. Sansom, C. Allende Prieto, and A. Vazdekis. 2023. "[sMILES: a library of semi-empirical MILES stellar spectra with variable \[$\alpha$/Fe\] abundances]{.nocase}." 523 (3): 3450--70. <https://doi.org/10.1093/mnras/stad1647>.

Park, Minjung, Charlie Conroy, Benjamin D. Johnson, Joel Leja, Aaron Dotter, and Phillip A. Cargile. 2024. "$\alpha$-MC: Self-consistent $\alpha$-enhanced stellar population models covering a wide range of age, metallicity, and wavelength." *arXiv e-Prints*, arXiv:2410.21375. <https://doi.org/10.48550/arXiv.2410.21375>.

Robotham, A. S. G., S. Bellstedt, C. del P. Lagos, et al. 2020. "ProSpect: generating spectral energy distributions with complex star formation and metallicity histories." 495 (1): 905--31. <https://doi.org/10.1093/mnras/staa1116>.

Salaris, M., A. Chieffi, and O. Straniero. 1993. "The alpha -enhanced isochrones and their impact on the FITS to the Galactic globular cluster system." 414: 580. <https://doi.org/10.1086/173105>.

Thomas, Daniel, Claudia Maraston, and Ralf Bender. 2003. "Stellar population models of Lick indices with variable element abundance ratios." 339: 897--911. <https://doi.org/10.1046/j.1365-8711.2003.06248.x>.

Vazdekis, A., P. Coelho, S. Cassisi, et al. 2015. "Evolutionary stellar population synthesis with MILES - II. Scaled-solar and $\alpha$-enhanced models." 449 (2): 1177--214. <https://doi.org/10.1093/mnras/stv151>.
