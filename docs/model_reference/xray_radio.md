(app-multiwavelength)=

# Multi-wavelength Extensions

X-ray and radio emission each have contributions from both the AGN and the host galaxy's stellar populations. We describe them as a separate section because their galaxy components (X-ray binaries, star-forming synchrotron) are independent of AGN activity.

(app-xray-module)=

## X-ray Emission

### AGN Corona

#### The $\alpha_{\rm ox}$ bridge.

For empirical disc backends that do not model a corona explicitly, the X-ray luminosity is anchored to the disc UV output via the Just et al. (2007) relation: $$\alpha_{\rm ox} = -0.137\,\log L_{\nu}(2500\,\text{\AA}) + 2.638,

$$ (eq-alpha-ox)
 where $L_\nu(2500\,\text{\AA})$ is in $\mathrm{erg\,s^{-1}\,Hz^{-1}}$. The definition $\alpha_{\rm ox} \equiv 0.384\,\log(L_{2\,\rm keV}/L_{2500})$ then gives the 2 keV monochromatic luminosity. An additive offset $\Delta\alpha_{\rm ox}$ (prior $|\Delta\alpha_{\rm ox}| < 0.2$) absorbs scatter around the mean relation.

#### X-ray spectral shape.

The corona spectrum is a power law with exponential cutoff: $$L_\nu \propto E^{1 - \Gamma}\,\exp\!\left(-\frac{E}{E_{\rm cut}}\right),

$$ (eq-xray-powerlaw)
 with photon index $\Gamma = 1.8$ and cutoff energy $E_{\rm cut} = 300$ keV as defaults (free when fitting X-ray data).

#### X-ray anisotropy.

Following Yang et al. (2022), the observed X-ray luminosity depends on viewing angle: $$L_X(\theta) = L_X(0)\,\bigl[a_1\cos\theta + a_2\cos^2\!\theta + (1 - a_1 - a_2)\bigr],

$$ (eq-xray-aniso)
 with default coefficients $a_1 = 0.5$, $a_2 = 0$. This produces a factor ${\sim}\,2$ suppression for edge-on (Type 2) sightlines relative to face-on, consistent with X-ray surveys.

#### Physical corona mode.

When a physical disc backend (qsosed or RELAGN) is selected, the corona photon index is derived self-consistently from the Beloborodov (1999) Comptonization formula: $$\Gamma_{\rm hot} = \frac{7}{3}\left(\frac{L_{\rm diss}}{L_{\rm seed}}\right)^{\!-0.1}\!,

$$ (eq-beloborodov)
 where $L_{\rm diss}$ is the power dissipated in the corona and $L_{\rm seed}$ is the seed photon luminosity intercepted from the disc. This bypasses the $\alpha_{\rm ox}$ bridge entirely, since $L_X$ and $\Gamma$ emerge from the same accretion physics as the disc SED.

### Galaxy X-ray Binaries

X-ray binary (XRB) emission from the host galaxy is independent of AGN activity and arises from accretion onto compact remnants in binary systems. Two populations contribute:

#### High-mass X-ray binaries (HMXBs)

scale with star-formation rate because the massive donor stars that power them are short-lived ($\lesssim 30\,$Myr).

#### Low-mass X-ray binaries (LMXBs)

scale with stellar mass because their low-mass donors evolve on Gyr timescales.

The total XRB luminosity (Grimm et al. 2003; Gilfanov 2004) is: $$L_X^{\rm XRB} = 2.6 \times 10^{39}\,\frac{\mathrm{SFR}}{M_\odot\,\mathrm{yr}^{-1}}
  \;+\; 8.3 \times 10^{28}\,\frac{M_\star}{M_\odot}
  \quad [\mathrm{erg\,s^{-1}}],

$$ (eq-xrb)
 with logarithmic offset parameters $\delta_{\rm HMXB}$ and $\delta_{\rm LMXB}$ (in dex) to absorb population-level scatter (Fragos et al. 2013). Future work will replace these empirical scaling relations with self-consistent XRB population synthesis from X-BPASS (Briel et al. 2025), which models accreting compact-object spectra alongside their stellar progenitors.

(app-radio-module)=

## Radio Emission

The radio module decomposes the total luminosity into three physically distinct contributions: $$L_\nu^{\rm radio} = L_\nu^{\rm synch} + L_\nu^{\rm ff} + L_\nu^{\rm AGN},

$$ (eq-radio-total)
 where $L_\nu^{\rm synch}$ is the star-forming synchrotron component (anchored to the FIRRC), $L_\nu^{\rm ff}$ is the thermal free-free contribution from HII regions, and $L_\nu^{\rm AGN}$ is the AGN jet component. Each term can be inspected independently via `radio_components()`.

### Star-Forming Synchrotron and FIRRC Modes

Synchrotron radiation from supernova remnants and cosmic-ray electrons is the dominant radio continuum at $\lesssim 10\,$GHz. Its normalization is set by the FIR--radio correlation (FIRRC): $$q_{\rm IR} = \log\!\left(\frac{L_{\rm IR}}{3.75 \times 10^{12}\,{\rm Hz}\cdot L_{\nu}(1.4\,{\rm GHz})}\right).

$$ (eq-fir-radio)
 The spectral shape is a power law $L_\nu \propto \nu^{-\alpha_{\rm SF}}$ (default $\alpha_{\rm SF} = 0.8$). Three FIRRC calibrations are available via `sfr_mode`:

- `bell2003` (Bell 2003): constant $q_{\rm IR} = 2.64$, the calibration of the total 1.4 GHz luminosity (see the free-free section below).

- `bell2003_split` (AGNfitter-rX): the same Bell total split 90% / 10% into a non-thermal ($\alpha = 0.75$) and a thermal ($\alpha = 0.1$) power law; it takes no separate free-free term. Beside a nebular backend that carries free-free (`cue`, `cloudy_grid`) the split keeps its 90 % synchrotron and drops its own 10 % thermal law, so the nebular continuum is the only thermal term; the 1.4 GHz total is then the synchrotron share of Bell's calibration plus that backend's free-free, as for `bell2003` above.

- `delvecchio2021` (Delvecchio et al. 2021): mass- and redshift-dependent correlation, $q_{\rm IR} = q_0(1+z)^{z_s} - m_s(\log M_\star - 10)$, with defaults $q_0 = 2.743$, $m_s = 0.234$, $z_s = -0.025$.

- `mccheyne2022` (McCheyne et al. 2022): same functional form with defaults $q_0 = 1.98$, $m_s = -0.22$, $z_s = 0.02$.

The three FIRRC parameters ($q_0$, $m_s$, $z_s$) are exposed as free parameters in all mass/redshift-dependent modes, enabling hierarchical inference over galaxy populations.

#### Normalization relative to CIGALE.

CIGALE's `radio` module normalizes the star-forming synchrotron at the 21 cm wavelength ($\nu = 1.42758\,$GHz) with a default $q_{\rm IR} = 2.58$. tengri anchors at $1.4\,$GHz, the frequency at which Bell (2003) defines $q_{\rm IR}$, and its default is $q_{\rm IR} = 2.64$. At equal $L_{\rm IR}$ and $\alpha_{\rm SF}$ the tengri luminosity is the CIGALE one times the frequency-independent factor $10^{\,q_{\rm CIGALE} - q_{\rm IR}}\,(1.4/1.42758)^{\alpha_{\rm SF}}$, which is 0.857 at the defaults ($\alpha_{\rm SF} = 0.8$). Setting `radio_q_ir = 2.58` leaves the anchor factor alone: 0.985 at $\alpha_{\rm SF} = 0.8$, 0.988 at 0.6 and 0.981 at 1.0. The anchor is a model-grammar key: `radio={'sf': {'type': 'bell2003', 'nu_ref': '21cm', 'freefree': False}}` normalizes at CIGALE's 1.42758 GHz (a number in Hz is also accepted; `'1.4GHz'`, the default, is Bell's). With that spelling and `radio_q_ir` set to CIGALE's value, tengri and the installed pcigale 2025.1 radio module agree to 2e-5 at 0.15, 1.4, 5 and 30 GHz (`tests/regression/radio/test_radio_21cm_anchor_2762.py`); without it the ratio is the constant 0.98451 at $\alpha_{\rm SF} = 0.8$. In $q_{\rm IR}$ the frequency is a property of the calibration: Bell (2003, Eq. 1) and Yun et al. (2001) measure $L_\nu$ at 1.4 GHz, Helou et al. (1985) at 1.49 GHz, so the default keeps Bell's. The key applies to `bell2003` only.

#### Synchrotron suppression at low SFR.

At low star-formation rates, cosmic-ray electrons lose energy through inverse Compton scattering off the CMB and infrared radiation field before producing significant synchrotron emission (Klein et al. 1984; Price and Duric 1992; Bell 2003). Bell (2003) parameterizes this as a luminosity-dependent non-thermal fraction (Eq. 3): $$n = \begin{cases}
0.9 & L > L_\star, \\
0.9\,(L/L_\star)^{0.3} & L \leq L_\star,
\end{cases}

$$ (eq-synch-suppression)
 where $L_\star$ corresponds to $M_V = -21$ and the total radio luminosity is $L_\nu = L_\nu^{\rm thermal} + n\,L_\nu^{\rm non\text{-}thermal}$. This correction is significant for dwarf galaxies with $\mathrm{SFR} \lesssim 0.1\,M_\odot\,{\rm yr}^{-1}$.

### Thermal Free-Free Emission

Thermal bremsstrahlung (free-free) from HII regions contributes at the $\sim$5--15% level at GHz frequencies and dominates above $\sim$30 GHz, tracing the instantaneous SFR on $\lesssim 10\,$Myr timescales. Following Murphy et al. (2011) (Eq. 11): $$\begin{aligned}
 
L_\nu^{\rm ff} = {} &
  \frac{1}{4.6\times10^{-28}}
  \left(\frac{T_e}{10^4\,{\rm K}}\right)^{\!0.45} \nonumber \\
  & \times \left(\frac{\nu}{{\rm GHz}}\right)^{\!\alpha_{\rm ff}}
  \,{\rm SFR},
  \qquad {\rm SFR} = 3.88\times10^{-44}\,L_{\rm IR}\ [{\rm erg\,s^{-1}}],
\end{aligned}

$$ (eq-radio-freefree)
 where the SFR [$M_\odot\,{\rm yr}^{-1}$] is derived from $L_{\rm IR}$ via Eq. 4 of Murphy et al. (2011) and $L_\nu^{\rm ff}$ is in erg s$^{-1}$ Hz$^{-1}$. At $1.4\,$GHz and $T_e = 10^4\,$K this gives $2.10 \times 10^{27}$ erg s$^{-1}$ Hz$^{-1}$ (5.5$\times10^{-7}\,L_\odot\,{\rm Hz}^{-1}$) per ${\rm M_\odot\,yr^{-1}}$.

The free-free spectral index $\alpha_{\rm ff} \approx -0.1$ is nearly flat (default), in contrast to the steep synchrotron slope $\alpha_{\rm SF} \approx -0.8$. Both $T_e$ (default $10^4\,$K) and $\alpha_{\rm ff}$ are exposed as free parameters for hierarchical inference.

#### Controlling free-free inclusion

The free-free component's default depends on where the model already carries thermal emission. It is **off** when the nebular backend carries a free-free continuum (`cue` or `cloudy_grid`): those backends extend their continuum beyond 1 cm as optically thin free-free with slope $\nu^{-0.1}$, so a second term via Murphy et al. (2011) would double-count it. It is **on from the SSP grid edge upward only** when the SSP includes nebular emission (a wNE grid, whatever `neb` type is declared on it): the SSP flux already carries the thermal continuum up to its last wavelength (1 cm for `ssp_prsc_miles_chabrier_wNE`), so the radio term starts there and the model has one thermal term at every wavelength. It is **on over its whole range** (from 1 mm) otherwise, for example a bare-stellar SSP with `none` or `cb19` (which publishes no continuum). The first rule is keyed on the declared nebular backend name (`nebular_backend_carries_freefree` in `tengri.components.nebular._models`), the second on the SSP's nebular stamp (`ssp_data.nebular == "included"`, set by the `nebular_included` file attribute or the `wNE` filename). An SSP file with neither keeps the term over its whole range; stamp it with `tools/stamp_ssp_nebular_attrs.py --included` if it includes nebular emission.

To override the default, pass `freefree: True` or `freefree: False` explicitly in the `radio.sf` sub-dict:

```python
radio={
    "sf": {"type": "bell2003", "freefree": False},
    "agn": {"type": "powerlaw"},
}
```

Omitting the `freefree` key applies the auto rule above. Exception: `bell2003_split` forces `freefree: False` (that mode already holds a thermal share of its own total), and passing `freefree: True` with it raises an error.

#### What `radio_q_ir` calibrates

`radio_q_ir` is the calibration of Bell (2003): the ratio of the 8--1000 $\mu$m luminosity to the **total** 1.4 GHz radio luminosity, about 10% of which is thermal. The $L_{\rm IR}$ used is the total absorbed dust power $L_{\rm absorbed}\,\eta$; the 8--1000 $\mu$m band holds 0.944 of it for the default dust emission model (DL14) and 0.958 for Casey (2012), for the delayed-$\tau$ galaxy used in the checks (integral of the dust-emission SED over the model grid). The three spellings:

- **`freefree` omitted (default).** $q_{\rm IR}$ is the total. The synchrotron term is $(1 - f_{\rm th})$ of $L_{\rm IR}/(3.75\times10^{12}\,{\rm Hz}\times10^{q_{\rm IR}})$, with $f_{\rm th}$ the free-free share at 1.4 GHz from the Murphy et al. (2011) term (0.1335 at $q_{\rm IR} = 2.64$, $T_e = 10^4\,$K), and the thermal term is added by this block, or by the nebular continuum when the nebular backend carries free-free. Without a nebular backend the 1.4 GHz total equals the calibration. With one it equals the calibration to within the difference between that backend's free-free share and $f_{\rm th}$: with Cue the total is 1.7% below the calibration for a constant star formation history and 7.3% below for a delayed-$\tau = 1$ Gyr galaxy at 5 Gyr, and the difference follows the ionizing photon rate per unit $L_{\rm IR}$.
- **`freefree: False`.** $q_{\rm IR}$ calibrates the non-thermal term alone, as in CIGALE: the synchrotron term carries the whole $L_{\rm IR}/(3.75\times10^{12}\,{\rm Hz}\times10^{q_{\rm IR}})$, this block adds no thermal term, and a nebular free-free continuum adds to it.
- **`freefree: True`.** As the default, with the thermal term from this block even beside a nebular backend that carries free-free (two thermal terms).

For the default spelling the synchrotron term would turn negative for $q_{\rm IR}$ above $q_\ast = -\log_{10}[3.75\times10^{12}\,(3.88\times10^{-44}/4.6\times10^{-28})(T_e/10^4\,{\rm K})^{0.45}\,1.4^{\alpha_{\rm ff}}]$ (3.5145 at $10^4\,$K, 3.379 at $2\times10^4\,$K). `SEDModel.build` raises if the support of `radio_q_ir` (a fixed value or the upper end of its prior) reaches that limit for the largest $T_e$ and $\alpha_{\rm ff}$ the model allows; the declared free prior of `radio_q_ir` ends at 3.37.

Passing `freefree: True` explicitly keeps the term over its whole range on a nebular-included SSP (it then overlaps the SSP's own continuum between 1 mm and the SSP edge); `freefree: False` removes it, leaving no thermal emission beyond the SSP edge.

### AGN Radio Jets

For radio-quiet AGN, a simple power law $L_\nu \propto \nu^{-\alpha_{\rm AGN}}$ ($\alpha_{\rm AGN} = 0.7$) is sufficient. It is normalized through the radio loudness `radio_loudness` $= \log_{10}[L_\nu(5\,{\rm GHz})/L_\nu(4400\,\text{\AA})]$, the B-band ratio. $L_\nu(4400\,\text{\AA})$ is the intrinsic accretion-disc luminosity that a composable AGN model publishes at its 30° reference inclination; the other AGN models publish no disc luminosity, and the radio block then uses $L_{\rm bol}/(5.15\,\nu_B)$, a bolometric correction of 5.15 at 4400 Å. The power law is multiplied by $\exp(-\nu/\nu_{\rm cut})$ with `radio_log_nu_cut` $= \log_{10}(\nu_{\rm cut}/{\rm Hz}) = 13$ by default, the synchrotron-aging cutoff of AGNfitter-rX; `radio_log_nu_cut = 40` removes it. At the default the cutoff lowers the jet by the factors 0.997, 0.990 and 0.970 at 30, 100 and 300 GHz, and by 0.9995 at the 5 GHz anchor of the loudness.

CIGALE defines the loudness on the 2500 Å anchor, linearly, as $R = L_\nu(5\,{\rm GHz})/L_\nu(2500\,\text{\AA})$ with the intrinsic disc luminosity at the same 30° inclination, and applies no cutoff. A CIGALE value $R$ corresponds to `radio_loudness` $= \log_{10}R + \log_{10}[L_\nu(2500\,\text{\AA})/L_\nu(4400\,\text{\AA})]$; the last term is the logarithm of the disc color and depends on the disc model.

For radio-loud AGN with spectral curvature, the double power-law (DPL) model from AGNfitter-rx (Martı́nez-Ramı́rez et al. 2024) captures the optically thick/thin transition: $$L_\nu = L_{5{\rm GHz}} \left(\frac{\nu}{\nu_t}\right)^{\!\alpha_1}\!
  \left[1 - \exp\!\left(-\!\left(\frac{\nu_t}{\nu}\right)^{\!\alpha_1-\alpha_2}\right)\right]
  \exp\!\left(-\frac{\nu}{\nu_{\rm cut}}\right),

$$ (eq-dpl-radio)
 where $\alpha_1$ is the steep (optically thin) slope (default $-0.75$), $\alpha_2$ is the flat (optically thick) slope (default $-0.1$), $\nu_t = 10^{\log\nu_t}$ is the transition frequency, and $\nu_{\rm cut} = 10^{13}\,$Hz is the synchrotron aging cutoff. The cutoff frequency is `radio_log_nu_cut`, shared with the power-law model.

## References

Bell, E. F. 2003. "Estimating Star Formation Rates from Infrared and Radio Luminosities: The Origin of the Radio-Infrared Correlation." 586 (April): 794--813. <https://doi.org/10.1086/367829>.

Beloborodov, Andrei M. 1999. "On the Number of Active Galactic Nuclei at High Accretion Rates." 510: L123--26. <https://doi.org/10.1086/311810>.

Briel, M. M., J. J. Eldridge, E. R. Stanway, and H. F. Stevance. 2025. "X-BPASS: self-consistent X-ray binary evolution in stellar population synthesis." *arXiv e-Prints*. <https://arxiv.org/abs/2508.18628>.

Delvecchio, I., E. Daddi, M. T. Sargent, et al. 2021. "The infrared-radio correlation of star-forming galaxies is strongly $M_{\star}$-dependent but nearly redshift-invariant since $z \sim 4$." 647 (March): A123. <https://doi.org/10.1051/0004-6361/202039647>.

Fragos, T., B. D. Lehmer, S. Naoz, A. Zezas, and A. Basu-Zych. 2013. "Energy Feedback from X-Ray Binaries in the Early Universe." 776: L31. <https://doi.org/10.1088/2041-8205/776/2/L31>.

Gilfanov, M. 2004. "Low-mass X-ray binaries as a stellar mass indicator for the host galaxy." 349: 146--68. <https://doi.org/10.1111/j.1365-2966.2004.07473.x>.

Grimm, H.-J., M. Gilfanov, and R. Sunyaev. 2003. "High-mass X-ray binaries as a star formation rate indicator in distant galaxies." 339: 793--809. <https://doi.org/10.1046/j.1365-8711.2003.06224.x>.

Just, Darren W., W. N. Brandt, Ohad Shemmer, et al. 2007. "The X-Ray Properties of the Most Luminous Quasars from the Sloan Digital Sky Survey." 665: 1004--22. <https://doi.org/10.1086/519990>.

Kennicutt, Jr., Robert C. 1998. "Star Formation in Galaxies Along the Hubble Sequence." 36 (January): 189--232. <https://doi.org/10.1146/annurev.astro.36.1.189>.

Klein, U., R. Wielebinski, and R. Beck. 1984. "A radio continuum study of the Magellanic Clouds. III. The radio continuum spectra of the bright H II regions in the Large Magellanic Cloud." 133: 19--26.

Martı́nez-Ramı́rez, Gabriela C., Gabriela Calistro Rivera, Elisabeta Lusso, and Francesco Shankar. 2024. "AGNfitter-rx: modelling AGN and galaxy SEDs from radio to X-rays." 535: 2961--85. <https://doi.org/10.1093/mnras/stae2437>.

McCheyne, I., S. Oliver, M. Sargent, et al. 2022. "The LOFAR Two-metre Sky Survey Deep fields. The mass dependence of the far-infrared radio correlation at 150 MHz using deblended Herschel fluxes." 662 (June): A100. <https://doi.org/10.1051/0004-6361/202141307>.

Murphy, E. J., J. J. Condon, E. Schinnerer, et al. 2011. "Calibrating Extinction-free Star Formation Rate Diagnostics with 33 GHz Free-free Emission in NGC 6946." 737 (2): 67. <https://doi.org/10.1088/0004-637X/737/2/67>.

Price, Richard, and Nebojsa Duric. 1992. "New Results on the Radio--Far-Infrared Relation for Galaxies." 401: 81. <https://doi.org/10.1086/172041>.

Yang, Guang, Médéric Boquien, W. N. Brandt, et al. 2022. "Fitting AGN/galaxy X-ray-to-radio SEDs with CIGALE and improvement of the code." 927: 192. <https://doi.org/10.3847/1538-4357/ac4971>.
