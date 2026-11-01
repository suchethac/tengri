| Name | Units | Group | Component | Description |
|------|-------|-------|-----------|-------------|
| `L_agn_disc_los_lsun` | Lsun | agn | agn | log_L_agn_disc_los in solar luminosities |
| `L_agn_disc_lsun` | Lsun | agn | agn | log_L_agn_disc in solar luminosities |
| `L_agn_disc_unemitted_lsun` | Lsun | agn | agn | log_L_agn_disc_unemitted in solar luminosities |
| `L_agn_lines_los_lsun` | Lsun | agn | agn | log_L_agn_lines_los in solar luminosities |
| `L_agn_lines_lsun` | Lsun | agn | agn | log_L_agn_lines in solar luminosities |
| `L_agn_lines_unemitted_lsun` | Lsun | agn | agn | log_L_agn_lines_unemitted in solar luminosities |
| `L_agn_los_lsun` | Lsun | agn | agn | log_L_agn_los in solar luminosities |
| `L_agn_polar_dust_los_lsun` | Lsun | agn | agn | log_L_agn_polar_dust_los in solar luminosities |
| `L_agn_polar_dust_lsun` | Lsun | agn | agn | log_L_agn_polar_dust in solar luminosities |
| `L_agn_power_lsun` | Lsun | agn | agn | log_L_agn_power in solar luminosities |
| `L_agn_torus_los_lsun` | Lsun | agn | agn | log_L_agn_torus_los in solar luminosities |
| `L_agn_torus_lsun` | Lsun | agn | agn | log_L_agn_torus in solar luminosities |
| `L_agn_torus_unemitted_lsun` | Lsun | agn | agn | log_L_agn_torus_unemitted in solar luminosities |
| `agn_ir_frac_realized` | — | agn | agn | AGN fraction agn_ir_frac measures, from what the model emitted: the AGN dust power over AGN dust plus the host's absorbed power (total dust), or the AGN share of the light in ir_frac_window. Equals agn_ir_frac where agn_ir_frac_saturated is 0 and agn_ledger_closed is 1; composable AGN only |
| `agn_ledger_closed` | — | agn | agn | 1.0 when the angle-integrated pieces plus the unemitted ones add up to the accretion power agn_lum_ratio * 10**agn_log_lbol L_sun, else 0.0 (self-contained quasar templates, agn_norm='independent', 'cigale_joint' with lines or an active agn_ir_frac tie, polar dust beside a library torus with anisotropy); composable AGN only |
| `balmer_break` | — | sed | stellar | Balmer break diagnostic |
| `balmer_decrement` | — | lines | nebular | Balmer decrement: Hα/Hβ |
| `bpt_nii` | dex | lines | nebular | BPT-NII diagnostic: log10([NII]6584 / Hα) |
| `bpt_sii` | dex | lines | nebular | BPT-SII diagnostic: log10(([SII]6717+6731) / Hα) |
| `civ_1549` | Lsun | lines | nebular | CIV 1549 line luminosity, NaN with a warning on cue's legacy 128-line subset (full_catalog=false), where no catalog line falls within tolerance of this wavelength (#2239) |
| `dn4000` | — | sed | stellar | D n4000 break diagnostic |
| `fuv_flux` | erg/s/Hz | sed | stellar | FUV flux (1000–1700 Å) |
| `fuv_flux_intrinsic` | erg/s/Hz | sed | stellar | Intrinsic FUV flux before dust attenuation |
| `halpha` | Lsun | lines | nebular | Hα line luminosity |
| `hbeta` | Lsun | lines | nebular | Hβ line luminosity |
| `irx` | dex | sed | stellar | Infrared excess, log10(L_TIR / nu*L_nu at 1600 A): the Meurer+99 IRX-beta anchor |
| `irx_fuv` | dex | sed | stellar | Infrared excess against the band-averaged FUV (1000-1700 A), pivoted at 1500 A |
| `l_1p4ghz` | erg/s/Hz | radio | radio | 1.4 GHz radio flux |
| `l_bol` | Lsun | sed | stellar | Bolometric luminosity |
| `l_dust_absorbed` | Lsun | sed | stellar | Dust-absorbed luminosity |
| `l_nonthermal` | erg/s/Hz | radio | radio | Radio non-thermal synchrotron luminosity |
| `l_thermal` | erg/s/Hz | radio | radio | Radio thermal (free-free) luminosity |
| `l_tir` | Lsun | sed | stellar | Infrared luminosity (8–1000 µm) |
| `l_x_agn` | Lsun | xray | xray | X-ray luminosity from AGN. `l_x_agn = 10**(log_l_x_agn - log10(L_sun))` |
| `l_x_total` | Lsun | xray | xray | Total X-ray luminosity (XRB + AGN). `l_x_total = 10**(log_l_x_total - log10(L_sun))` |
| `l_x_xrb` | Lsun | xray | xray | X-ray luminosity from X-ray binaries. `l_x_xrb = 10**(log_l_x_xrb - log10(L_sun))` |
| `log_L_agn_disc` | dex | agn | agn | log10 power [dex re erg/s] of the disc (with its corona) as emitted, integrated over all directions (sphere mean over cos i); composable AGN only |
| `log_L_agn_disc_los` | dex | agn | agn | log10 power [dex re erg/s] of the disc (with its corona) as emitted along agn_cos_inc, the integral of that piece of the SED; composable AGN only |
| `log_L_agn_disc_unemitted` | dex | agn | agn | log10 power [dex re erg/s] carried by the disc (with its corona) but removed before it leaves the AGN (torus screen, polar dust, attenuation block, disc reddening), integrated over all directions; composable AGN only |
| `log_L_agn_lines` | dex | agn | agn | log10 power [dex re erg/s] of the NLR + BLR + FeII emission lines as emitted, integrated over all directions (sphere mean over cos i); composable AGN only |
| `log_L_agn_lines_los` | dex | agn | agn | log10 power [dex re erg/s] of the NLR + BLR + FeII emission lines as emitted along agn_cos_inc, the integral of that piece of the SED; composable AGN only |
| `log_L_agn_lines_unemitted` | dex | agn | agn | log10 power [dex re erg/s] carried by the NLR + BLR + FeII emission lines but removed before it leaves the AGN (torus screen, polar dust, attenuation block, disc reddening), integrated over all directions; composable AGN only |
| `log_L_agn_los` | dex | agn | agn | log10 of the emitted AGN SED integrated over frequency along agn_cos_inc [dex re erg/s], the line-of-sight sum of the ledger pieces; composable AGN only |
| `log_L_agn_polar_dust` | dex | agn | agn | log10 power [dex re erg/s] of the polar-dust re-emission as emitted, integrated over all directions (sphere mean over cos i); composable AGN only |
| `log_L_agn_polar_dust_los` | dex | agn | agn | log10 power [dex re erg/s] of the polar-dust re-emission as emitted along agn_cos_inc, the integral of that piece of the SED; composable AGN only |
| `log_L_agn_power` | dex | agn | agn | log10 of the AGN dust power the runner normalizes the torus and the polar dust on [dex re erg/s], the torus block's power along agn_cos_inc (CIGALE's agn_power, the anchor of the agn_ir_frac tie); composable AGN with a torus only |
| `log_L_agn_torus` | dex | agn | agn | log10 power [dex re erg/s] of the torus dust as emitted, integrated over all directions (sphere mean over cos i); composable AGN only |
| `log_L_agn_torus_los` | dex | agn | agn | log10 power [dex re erg/s] of the torus dust as emitted along agn_cos_inc, the integral of that piece of the SED; composable AGN only |
| `log_L_agn_torus_unemitted` | dex | agn | agn | log10 power [dex re erg/s] carried by the torus dust but removed before it leaves the AGN (torus screen, polar dust, attenuation block, disc reddening), integrated over all directions; composable AGN only |
| `log_civ_1549` | dex | lines | nebular | log10 of civ 1549 line luminosity, nan with a warning on cue's legacy 128-line subset (full_catalog=false), where no catalog line falls within tolerance of this wavelength (#2239) [dex re erg/s]; float32-safe form of `civ_1549`. `log_civ_1549 = log10(civ_1549) + log10(L_sun)`. |
| `log_halpha` | dex | lines | nebular | log10 of hα line luminosity [dex re erg/s]; float32-safe form of `halpha`. `log_halpha = log10(halpha) + log10(L_sun)`. |
| `log_hbeta` | dex | lines | nebular | log10 of hβ line luminosity [dex re erg/s]; float32-safe form of `hbeta`. `log_hbeta = log10(hbeta) + log10(L_sun)`. |
| `log_l_x_agn` | dex | xray | xray | log10 X-ray luminosity from AGN [dex re erg/s]; float32-safe form of `l_x_agn`. -inf when no AGN is present, where the linear form is 0.0 |
| `log_l_x_agn_total` | dex | xray | xray | log10 luminosity of the unabsorbed AGN corona integrated over 0.25-1240 keV, the pcigale yang20 grid [dex re erg/s]; -inf without an AGN |
| `log_l_x_hmxb_2_10` | dex | xray | xray | log10 2-10 keV luminosity of the HMXB term [dex re erg/s] |
| `log_l_x_hotgas_0p5_2` | dex | xray | xray | log10 0.5-2 keV luminosity of the hot-gas term [dex re erg/s] |
| `log_l_x_lmxb_2_10` | dex | xray | xray | log10 2-10 keV luminosity of the LMXB term [dex re erg/s] |
| `log_l_x_total` | dex | xray | xray | log10 total X-ray luminosity (XRB + AGN) [dex re erg/s]; float32-safe form of `l_x_total` |
| `log_l_x_xrb` | dex | xray | xray | log10 X-ray luminosity from X-ray binaries [dex re erg/s]; float32-safe form of `l_x_xrb` |
| `log_lnu_2kev_30deg` | dex | xray | xray | log10 intrinsic 2 keV monochromatic L_nu at an inclination of 30 deg (cos i = 0.866), the alpha_ox anchor [dex re erg/s/Hz]; -inf without an AGN and under lopez24 |
| `log_lya` | dex | lines | nebular | log10 of lyman alpha line luminosity [dex re erg/s]; float32-safe form of `lya`. `log_lya = log10(lya) + log10(L_sun)`. |
| `log_nii_6548` | dex | lines | nebular | log10 of nii 6548 line luminosity [dex re erg/s]; float32-safe form of `nii_6548`. `log_nii_6548 = log10(nii_6548) + log10(L_sun)`. |
| `log_nii_6584` | dex | lines | nebular | log10 of nii 6584 line luminosity [dex re erg/s]; float32-safe form of `nii_6584`. `log_nii_6584 = log10(nii_6584) + log10(L_sun)`. |
| `log_no` | dex | lines | nebular | Absolute [N/O] = log10((N/O)/(N/O)_sun) the Cue backend is fed: gas_logno under nitrogen='absolute', else the named N/O-O/H relation at the gas metallicity plus gas_logno. NaN for other backends. |
| `log_oii` | dex | lines | nebular | log10 of oii line luminosity [dex re erg/s]; float32-safe form of `oii`. `log_oii = log10(oii) + log10(L_sun)`. |
| `log_oiii_4959` | dex | lines | nebular | log10 of oiii 4959 line luminosity [dex re erg/s]; float32-safe form of `oiii_4959`. `log_oiii_4959 = log10(oiii_4959) + log10(L_sun)`. |
| `log_oiii_5007` | dex | lines | nebular | log10 of oiii 5007 line luminosity [dex re erg/s]; float32-safe form of `oiii_5007`. `log_oiii_5007 = log10(oiii_5007) + log10(L_sun)`. |
| `log_q_h` | dex | ionizing | stellar | log10(ionizing photon production rate / (photons/s)). `q_h [photons/s] = 10**log_q_h`. |
| `log_sii_6717` | dex | lines | nebular | log10 of sii 6717 line luminosity [dex re erg/s]; float32-safe form of `sii_6717`. `log_sii_6717 = log10(sii_6717) + log10(L_sun)`. |
| `log_sii_6731` | dex | lines | nebular | log10 of sii 6731 line luminosity [dex re erg/s]; float32-safe form of `sii_6731`. `log_sii_6731 = log10(sii_6731) + log10(L_sun)`. |
| `luminosity_weighted_age_gyr` | Gyr | sfh | stellar | Luminosity-weighted mean age of stellar population |
| `luminosity_weighted_metallicity` | dex | sfh | stellar | Luminosity-weighted mean metallicity (log10 Z/Zsun) |
| `lya` | Lsun | lines | nebular | Lyman alpha line luminosity |
| `m_uv` | AB mag | sed | stellar | UV absolute magnitude (1600 Å) |
| `mass_weighted_age_gyr` | Gyr | sfh | stellar | Mass-weighted mean age of stellar population |
| `mass_weighted_metallicity` | dex | sfh | stellar | Mass-weighted mean metallicity (log10 Z/Zsun) |
| `nii_6548` | Lsun | lines | nebular | NII 6548 line luminosity |
| `nii_6584` | Lsun | lines | nebular | NII 6584 line luminosity |
| `nuv_flux` | erg/s/Hz | sed | stellar | NUV flux (1700–3000 Å) |
| `nuv_flux_intrinsic` | erg/s/Hz | sed | stellar | Intrinsic NUV flux before dust attenuation |
| `o32` | dex | lines | nebular | O32 ionization parameter: log10([OIII]5007 / [OII]) |
| `o3hb` | dex | lines | nebular | [OIII]5007/Hβ diagnostic: log10([OIII]5007 / Hβ) |
| `oii` | Lsun | lines | nebular | OII line luminosity |
| `oiii_4959` | Lsun | lines | nebular | OIII 4959 line luminosity |
| `oiii_5007` | Lsun | lines | nebular | OIII 5007 line luminosity |
| `q_ir` | — | radio | radio | Radio-infrared correlation parameter |
| `r23` | dex | lines | nebular | R23 metallicity indicator: log10(([OII]+[OIII]4959+5007)/Hβ) |
| `rest_uv_color` | AB mag | sed | stellar | Rest-frame U-V color [AB mag]: top-hats 3200–3900 Å minus 5000–5800 Å |
| `sfh_cexp_age_gyr` | Gyr | sfh | stellar | Resolved sfh_cexp_age_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_const_start_gyr` | Gyr | sfh | stellar | Resolved sfh_const_start_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_declining_exp_age_gyr` | Gyr | sfh | stellar | Resolved sfh_declining_exp_age_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_delayed_age_gyr` | Gyr | sfh | stellar | Resolved sfh_delayed_age_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_delayed_bq_age_main_gyr` | Gyr | sfh | stellar | Resolved sfh_delayed_bq_age_main_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_dexp_start_gyr` | Gyr | sfh | stellar | Resolved sfh_dexp_start_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_dpl_age_gyr` | Gyr | sfh | stellar | Resolved sfh_dpl_age_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_dpl_lookback_age_gyr` | Gyr | sfh | stellar | Resolved sfh_dpl_lookback_age_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_exp_start_gyr` | Gyr | sfh | stellar | Resolved sfh_exp_start_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_lnorm_age_gyr` | Gyr | sfh | stellar | Resolved sfh_lnorm_age_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_norm_peak_lbt_gyr` | Gyr | sfh | stellar | Resolved sfh_norm_peak_lbt_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_periodic_age_gyr` | Gyr | sfh | stellar | Resolved sfh_periodic_age_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_psb2022_tflex_gyr` | Gyr | sfh | stellar | Resolved sfh_psb2022_tflex_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_psb2022_tlast_gyr` | Gyr | sfh | stellar | Resolved sfh_psb2022_tlast_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_psb_age_gyr` | Gyr | sfh | stellar | Resolved sfh_psb_age_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_psb_flex_tflex_gyr` | Gyr | sfh | stellar | Resolved sfh_psb_flex_tflex_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_psb_flex_tlast_gyr` | Gyr | sfh | stellar | Resolved sfh_psb_flex_tlast_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_sfh2exp_age_gyr` | Gyr | sfh | stellar | Resolved sfh_sfh2exp_age_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_snorm_burst_peak_lbt_gyr` | Gyr | sfh | stellar | Resolved sfh_snorm_burst_peak_lbt_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_snorm_peak_lbt_gyr` | Gyr | sfh | stellar | Resolved sfh_snorm_peak_lbt_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_trunc_exp_age_gyr` | Gyr | sfh | stellar | Resolved sfh_trunc_exp_age_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_tsnorm_burst_peak_lbt_gyr` | Gyr | sfh | stellar | Resolved sfh_tsnorm_burst_peak_lbt_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfh_tsnorm_peak_lbt_gyr` | Gyr | sfh | stellar | Resolved sfh_tsnorm_peak_lbt_gyr of the active SFH: frac * age(z) when its fraction is sampled, the Fixed Gyr otherwise; NaN when the SFH has no such timing |
| `sfr_100myr` | Msun/yr | sfh | stellar | Star formation rate averaged over past 100 Myr |
| `sfr_10myr` | Msun/yr | sfh | stellar | Star formation rate averaged over past 10 Myr |
| `sii_6717` | Lsun | lines | nebular | SII 6717 line luminosity |
| `sii_6731` | Lsun | lines | nebular | SII 6731 line luminosity |
| `ssfr` | 1/yr | sfh | stellar | Specific star formation rate (sfr_100myr / stellar_mass_surviving; falls back to the formed mass when the SSP has no mass-remaining table) |
| `stellar_mass` | Msun | sfh | stellar | Total stellar mass formed by the SFH: its time-integral, 1.5-1.9x above stellar_mass_surviving |
| `stellar_mass_surviving` | Msun | sfh | stellar | Total surviving stellar mass |
| `uv_slope_beta` | — | sed | stellar | UV slope (β in L_ν ∝ ν^β) |
| `xi_ion` | Hz/erg | ionizing | stellar | Ionizing photon efficiency |
