(app-parameters)=

# Parameter Reference

Table {ref}`1 <tab-params>` lists all configurable parameters in tengri, grouped by physical module. Parameters are specified via the `Parameters` interface using the naming convention `module_submodule_parameter`.

(tab-params)=

| Parameter | Symbol | Prior | Range | Units | Description |
|:---|:---|:---|:---|:---|:---|
| *Table {ref}`1 <tab-params>` continued* |  |  |  |  |  |
| Parameter | Symbol | Prior | Range | Units | Description |
| *Continued on next page* |  |  |  |  |  |
|  |  |  |  |  |  |
| `sfh_tsnorm_log_peak_sfr` | $\log \dot{M}_\star^{\rm peak}$ | Uniform | $[-1, 3]$ | $\log(M_\odot\,{\rm yr}^{-1})$ | Peak SFR |
| `sfh_tsnorm_peak_lbt_gyr` | $t_{\rm peak}$ | Uniform | $[0.5, 13]$ | Gyr | Lookback time of peak |
| `sfh_tsnorm_width_gyr` | $w$ | Uniform | $[0.1, 8]$ | Gyr | Width of SFH kernel |
| `sfh_tsnorm_skew` | $\gamma$ | Uniform | $[-3, 10]$ | --- | Skewness parameter |
| `sfh_tsnorm_trunc` | $\alpha_t$ | Uniform | $[0, 3]$ | --- | Truncation steepness |
| *Star Formation History (stochastic field)* |  |  |  |  |  |
| `sfh_field_psd_sigma` | $\sigma_{\rm PS}$ | Uniform | $[0.01, 2.0]$ | dex | PSD amplitude |
| `sfh_field_psd_tau_myr` | $\tau_{\rm PS}$ | Uniform | $[1, 1000]$ | Myr | PSD timescale |
| `sfh_field_xi_*` | $\xi_n$ | Fixed(0) | $[-4, 4]$ | --- | Latent field (128--256 dims) |
| *Burst component (optional)* |  |  |  |  |  |
| `sfh_burst_frac` | $f_b$ | Uniform | $[0, 0.5]$ | --- | Burst mass fraction |
| `sfh_burst_age_myr` | $t_b$ | Uniform | $[1, 100]$ | Myr | Burst age |
| `sfh_burst_width_dex` | $\Delta_b$ | Uniform | $[0.1, 1.0]$ | dex | Burst width in log-age |
| *Metallicity* |  |  |  |  |  |
| `met_logzsol` | $\log(Z/Z_\odot)$ | Gaussian | $[-2.0, 0.2]$ | dex | Log solar metallicity |
| *Alpha-Element Enhancement (opt-in; requires 4D SSP grid or fallback mode)* |  |  |  |  |  |
| `met_alpha_fe` | $[\alpha/\mathrm{Fe}]$ | Uniform | $[-0.2, 0.6]$ | dex | Alpha-to-iron ratio (uniform mode) |
| `met_alpha_fe_old` | $[\alpha/\mathrm{Fe}]_{\rm old}$ | Uniform | $[0.0, 0.6]$ | dex | Alpha enhancement at old ages (time-evolving mode) |
| `met_alpha_fe_young` | $[\alpha/\mathrm{Fe}]_{\rm young}$ | Fixed(0) | $[-0.2, 0.4]$ | dex | Alpha enhancement at young ages (typically fixed at 0) |
| *Dust Attenuation* |  |  |  |  |  |
| `dust_tau_bc` | $\tau_{\rm BC}$ | Uniform | $[0, 4]$ | --- | Birth cloud optical depth |
| `dust_tau_ism` | $\tau_{\rm ISM}$ | Uniform | $[0, 4]$ | --- | Diffuse ISM optical depth |
| `dust_n_bc` | $n_{\rm BC}$ | Uniform | $[-2, 0.5]$ | --- | Birth cloud slope |
| `dust_n_ism` | $n_{\rm ISM}$ | Uniform | $[-2, 0.5]$ | --- | ISM slope |
| `dust_delta` | $\delta$ | Uniform | $[-1, 0.4]$ | --- | Power-law modification |
| `dust_Eb` | $E_b$ | Uniform | $[0, 6]$ | --- | UV bump strength |
| `dust_f_obscuration` | $f_{\rm obs}$ | Uniform | $[0, 1]$ | --- | Clumpy geometry fraction |
| *AGN Core Parameters* |  |  |  |  |  |
| `agn_frac` | $f_{\rm AGN}$ | Fixed(0) | $[0, 1]$ | --- | AGN bolometric luminosity fraction |
| `agn_log_lbol` | $\log L_{\rm bol}$ | Fixed(10) | $[40, 48]$ | $\log({\rm erg\,s}^{-1})$ | AGN bolometric luminosity (direct) |
| `agn_alpha` | $\alpha$ | Fixed($-1$) | $[-2, 0]$ | --- | Disc power-law slope |
| `agn_log_mbh` | $\log M_{\rm BH}$ | Fixed(7) | $[5, 10]$ | $\log(M_\odot)$ | Black hole mass |
| `agn_log_ledd` | $\log(\dot{m}/\dot{m}_{\rm Edd})$ | Fixed($-1$) | $[-3, 0]$ | --- | Eddington ratio |
| *AGN Torus* |  |  |  |  |  |
| `agn_T_torus` | $T_{\rm torus}$ | Fixed(1000) | $[100, 2000]$ | K | Torus temperature |
| `agn_tau_torus` | $\tau_{9.7}$ | Fixed(5) | $[0, 20]$ | --- | Torus optical depth at 9.7 $\mu$m |
| `agn_torus_frac` | $f_{\rm cov}$ | Fixed(0.5) | $[0, 1]$ | --- | Torus covering factor |
| `agn_tau_skirtor` | $\tau_{9.7}$ | Fixed(7) | $[3, 11]$ | --- | SKIRTOR optical depth |
| `agn_p_skirtor` | $p$ | Fixed(1) | $[0, 1.5]$ | --- | SKIRTOR radial density gradient |
| `agn_q_skirtor` | $q$ | Fixed(1) | $[0, 1.5]$ | --- | SKIRTOR polar density gradient |
| `agn_oa_skirtor` | $\theta_{\rm oa}$ | Fixed(40) | $[20, 60]$ | deg | SKIRTOR half-opening angle |
| `agn_cos_inc` | $\cos i$ | Fixed(0.5) | $[0, 1]$ | --- | Cosine of inclination |
| *Nebular Emission (CloudyGrid / Cue)* |  |  |  |  |  |
| `neb_gas_logz` | $\log Z_{\rm gas}$ | Uniform | $[-2, 0.5]$ | --- | Gas metallicity |
| `neb_gas_logu` | $\log U$ | Uniform | $[-4, -1]$ | --- | Ionization parameter |
| `neb_dig_frac` | $f_{\rm DIG}$ | Uniform | $[0, 0.6]$ | --- | DIG mixing fraction |
| *Shock Emission (MAPPINGS V / 3MdBs)* |  |  |  |  |  |
| `shock_frac` | $f_{\rm shock}$ | Fixed(0) | $[0, 1]$ | --- | Shock fraction of nebular H$\alpha$ |
| `shock_velocity` | $v_s$ | Fixed(300) | $[100, 1000]$ | km s$^{-1}$ | Shock velocity |
| `shock_log_density` | $\log n$ | Fixed(0) | $[-2, 3]$ | cm$^{-3}$ | Log pre-shock density (snapped) |
| `shock_b_over_sqrt_n` | $B$ | Fixed(1) | --- | $\mu$G | Magnetic field strength (snapped) |
| `shock_abundance` | --- | Fixed(`solar`) | --- | --- | Abundance set (categorical) |
| `shock_component` | --- | Fixed(`combined`) | --- | --- | Emission component (categorical) |
| *Dust Emission* |  |  |  |  |  |
| `dust_T` | $T_{\rm dust}$ | Uniform | $[15, 60]$ | K | Dust temperature (MBB) |
| `dust_beta_ir` | $\beta_{\rm IR}$ | Uniform | $[1.0, 2.5]$ | --- | MBB emissivity index |
| `dust_alpha_dale` | $\alpha$ | Uniform | $[1.0, 4.0]$ | --- | Dale template slope |
| `dust_eta_balance` | $\eta$ | Fixed(1) | $[0.5, 2.0]$ | --- | Energy balance relaxation |
| `dust_xi_pah` | $\xi_{\rm PAH}$ | Uniform | $[0, 0.15]$ | --- | PAH fraction (MAGPHYS) |
| `dust_xi_mir` | $\xi_{\rm MIR}$ | Uniform | $[0, 0.15]$ | --- | Hot MIR fraction (MAGPHYS) |
| `dust_xi_warm` | $\xi_W$ | Uniform | $[0.1, 0.5]$ | --- | Warm fraction (MAGPHYS) |
| `dust_T_cold` | $T_C$ | Uniform | $[15, 25]$ | K | Cold ISM temperature (MAGPHYS) |
| `dust_qhac` | $q_{\rm hac}$ | Uniform | $[0, 0.4]$ | --- | Small a-C(:H) fraction (THEMIS) |
| `dust_log_ssfr` | $\log\,\mathrm{sSFR}$ | Fixed | --- | $\log\,{\rm yr}^{-1}$ | sSFR for BOSA templates |
| *IGM (optional extensions)* |  |  |  |  |  |
| `igm_x_hi` | $\bar{x}_{\rm HI}$ | Uniform | $[0, 1]$ | --- | Neutral fraction (patchy) |
| `igm_r_bubble` | $R_b$ | Uniform | $[0.1, 50]$ | pMpc | Ionized bubble radius |
| *Chemical Evolution (optional)* |  |  |  |  |  |
| `met_logzsol_final` | $\log(Z_f/Z_\odot)$ | Gaussian | $[-2, 0.5]$ | dex | Present-day gas metallicity |
| `met_eta_outflow` | $\eta_{\rm out}$ | Uniform | $[0, 10]$ | --- | Mass-loading factor |
| *Observation* |  |  |  |  |  |
| `redshift` | $z$ | Fixed | --- | --- | Redshift |
| `velocity_dispersion` | $\sigma_v$ | Fixed | $[50, 400]$ | km s$^{-1}$ | Velocity dispersion |
| `noise_calibration` | $f_{\rm cal}$ | Uniform | $[0, 0.1]$ | --- | Calibration noise floor |
| `noise_dof` | $\nu$ | Fixed($\infty$) | $2, \infty)$ | --- | Student-$t$ degrees of freedom |

: Complete parameter reference for [tengri. "Prior" indicates the recommended distribution type. Ranges are indicative defaults; users may adjust them via `Parameters`.
