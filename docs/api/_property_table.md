| Name | Units | Group | Component | Description |
|------|-------|-------|-----------|-------------|
| `balmer_break` | — | sed | stellar | Balmer break diagnostic |
| `balmer_decrement` | — | lines | nebular | Balmer decrement: Hα/Hβ |
| `bpt_nii` | dex | lines | nebular | BPT-NII diagnostic: log10([NII]6584 / Hα) |
| `bpt_sii` | dex | lines | nebular | BPT-SII diagnostic: log10(([SII]6717+6731) / Hα) |
| `civ_1549` | erg/s | lines | nebular | CIV 1549 line luminosity |
| `dn4000` | — | sed | stellar | D n4000 break diagnostic |
| `fuv_flux` | erg/s/Hz | sed | stellar | FUV flux (1000–1700 Å) |
| `fuv_flux_intrinsic` | erg/s/Hz | sed | stellar | Intrinsic FUV flux before dust attenuation |
| `halpha` | erg/s | lines | nebular | Hα line luminosity |
| `hbeta` | erg/s | lines | nebular | Hβ line luminosity |
| `irx` | dex | sed | stellar | Infrared excess, log10(L_TIR / nu*L_nu at 1600 A) — the Meurer+99 IRX-beta anchor |
| `irx_fuv` | dex | sed | stellar | Infrared excess against the band-averaged FUV (1000-1700 A), pivoted at 1500 A |
| `l_1p4ghz` | erg/s/Hz | radio | radio | 1.4 GHz radio flux |
| `l_bol` | Lsun | sed | stellar | Bolometric luminosity |
| `l_dust_absorbed` | Lsun | sed | stellar | Dust-absorbed luminosity |
| `l_nonthermal` | erg/s/Hz | radio | radio | Radio non-thermal synchrotron luminosity |
| `l_thermal` | erg/s/Hz | radio | radio | Radio thermal (free-free) luminosity |
| `l_tir` | Lsun | sed | stellar | Infrared luminosity (8–1000 µm) |
| `l_x_agn` | erg/s | xray | xray | X-ray luminosity from AGN |
| `l_x_total` | erg/s | xray | xray | Total X-ray luminosity (XRB + AGN) |
| `l_x_xrb` | erg/s | xray | xray | X-ray luminosity from X-ray binaries |
| `log_civ_1549` | dex | lines | nebular | log10 of civ 1549 line luminosity [dex re erg/s]; float32-safe form of `civ_1549` |
| `log_halpha` | dex | lines | nebular | log10 of hα line luminosity [dex re erg/s]; float32-safe form of `halpha` |
| `log_hbeta` | dex | lines | nebular | log10 of hβ line luminosity [dex re erg/s]; float32-safe form of `hbeta` |
| `log_lya` | dex | lines | nebular | log10 of lyman alpha line luminosity [dex re erg/s]; float32-safe form of `lya` |
| `log_nii_6548` | dex | lines | nebular | log10 of nii 6548 line luminosity [dex re erg/s]; float32-safe form of `nii_6548` |
| `log_nii_6584` | dex | lines | nebular | log10 of nii 6584 line luminosity [dex re erg/s]; float32-safe form of `nii_6584` |
| `log_oii` | dex | lines | nebular | log10 of oii line luminosity [dex re erg/s]; float32-safe form of `oii` |
| `log_oiii_4959` | dex | lines | nebular | log10 of oiii 4959 line luminosity [dex re erg/s]; float32-safe form of `oiii_4959` |
| `log_oiii_5007` | dex | lines | nebular | log10 of oiii 5007 line luminosity [dex re erg/s]; float32-safe form of `oiii_5007` |
| `log_q_h` | dex | ionizing | stellar | log10(ionizing photon production rate / (photons/s)) — float32-safe form of q_h |
| `log_sii_6717` | dex | lines | nebular | log10 of sii 6717 line luminosity [dex re erg/s]; float32-safe form of `sii_6717` |
| `log_sii_6731` | dex | lines | nebular | log10 of sii 6731 line luminosity [dex re erg/s]; float32-safe form of `sii_6731` |
| `luminosity_weighted_age_gyr` | Gyr | sfh | stellar | Luminosity-weighted mean age of stellar population |
| `luminosity_weighted_metallicity` | dex | sfh | stellar | Luminosity-weighted mean metallicity (log10 Z/Zsun) |
| `lya` | erg/s | lines | nebular | Lyman alpha line luminosity |
| `m_uv` | AB mag | sed | stellar | UV absolute magnitude (1600 Å) |
| `mass_weighted_age_gyr` | Gyr | sfh | stellar | Mass-weighted mean age of stellar population |
| `mass_weighted_metallicity` | dex | sfh | stellar | Mass-weighted mean metallicity (log10 Z/Zsun) |
| `nii_6548` | erg/s | lines | nebular | NII 6548 line luminosity |
| `nii_6584` | erg/s | lines | nebular | NII 6584 line luminosity |
| `nuv_flux` | erg/s/Hz | sed | stellar | NUV flux (1700–3000 Å) |
| `nuv_flux_intrinsic` | erg/s/Hz | sed | stellar | Intrinsic NUV flux before dust attenuation |
| `o32` | dex | lines | nebular | O32 ionization parameter: log10([OIII]5007 / [OII]) |
| `o3hb` | dex | lines | nebular | [OIII]5007/Hβ diagnostic: log10([OIII]5007 / Hβ) |
| `oii` | erg/s | lines | nebular | OII line luminosity |
| `oiii_4959` | erg/s | lines | nebular | OIII 4959 line luminosity |
| `oiii_5007` | erg/s | lines | nebular | OIII 5007 line luminosity |
| `q_h` | photons/s | ionizing | stellar | Ionizing photon production rate |
| `q_ir` | — | radio | radio | Radio-infrared correlation parameter |
| `r23` | dex | lines | nebular | R23 metallicity indicator: log10(([OII]+[OIII]4959+5007)/Hβ) |
| `rest_uv_color` | AB mag | sed | stellar | Rest-frame UV color (FUV–NUV) |
| `sfr_100myr` | Msun/yr | sfh | stellar | Star formation rate averaged over past 100 Myr |
| `sfr_10myr` | Msun/yr | sfh | stellar | Star formation rate averaged over past 10 Myr |
| `sii_6717` | erg/s | lines | nebular | SII 6717 line luminosity |
| `sii_6731` | erg/s | lines | nebular | SII 6731 line luminosity |
| `ssfr` | 1/yr | sfh | stellar | Specific star formation rate (sfr_100myr / stellar_mass_surviving; falls back to the formed mass when the SSP has no mass-remaining table) |
| `stellar_mass` | Msun | sfh | stellar | Total stellar mass formed by the SFH — its time-integral, 1.5-1.9x above stellar_mass_surviving |
| `stellar_mass_surviving` | Msun | sfh | stellar | Total surviving stellar mass |
| `uv_slope_beta` | — | sed | stellar | UV slope (β in L_ν ∝ ν^β) |
| `xi_ion` | Hz/erg | ionizing | stellar | Ionizing photon efficiency |
