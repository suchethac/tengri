(app-igm-details)=

# IGM Absorption

tengri implements the Inoue et al. (2014) mean IGM transmission model, which accounts for four optical depth components: $$\tau_{\rm IGM}(\lambda_{\rm obs}, z_s) = \tau_{\rm LS}^{\rm LAF} + \tau_{\rm LS}^{\rm DLA} + \tau_{\rm LC}^{\rm LAF} + \tau_{\rm LC}^{\rm DLA},

$$ (eq-igm-total)
 where LS denotes Lyman-series line absorption and LC denotes Lyman-continuum absorption, from the Ly-$\alpha$ forest (LAF) and damped Ly-$\alpha$ systems (DLA) respectively.

The transmission is $T_{\rm IGM} = \exp(-\tau_{\rm IGM})$. Each component is a sum over the 39 Lyman-series lines ($j = 2$ to $40$), evaluated using piecewise power-law coefficients from Inoue et al. (2014) (Tables 2--3).

The LAF line optical depth for line $j$ has three regimes: $$\tau_{j}^{\rm LAF}(\lambda_{\rm obs}) = \begin{cases}
A_{j,1}^{\rm LAF} \,(\lambda_{\rm obs}/\lambda_j)^{1.2} & \lambda_{\rm obs} < 2.2\,\lambda_j \\
A_{j,2}^{\rm LAF} \,(\lambda_{\rm obs}/\lambda_j)^{3.7} & 2.2\,\lambda_j \leq \lambda_{\rm obs} < 5.7\,\lambda_j \\
A_{j,3}^{\rm LAF} \,(\lambda_{\rm obs}/\lambda_j)^{5.5} & \lambda_{\rm obs} \geq 5.7\,\lambda_j
\end{cases}$$ for $\lambda_j < \lambda_{\rm obs} < \lambda_j(1 + z_s)$.

### CGM Damping Wing Extension

At $z > 5$, neutral hydrogen in the circumgalactic medium produces Ly-$\alpha$ damping wing absorption not captured by the Inoue et al. (2014) model. Following Asada et al. (2025), tengri adds: $$\tau_{\rm DW}(\lambda_{\rm obs}) = N_{\rm HI}(z) \cdot \sigma_{\rm DW}(\Delta\nu),$$ where the column density follows a sigmoid evolution $N_{\rm HI}(z) = N_{\rm HI,0} / [1 + \exp(-(z - z_{\rm mid})/\Delta z)]$ with default $\log_{10}(N_{\rm HI,0}/{\rm cm}^{-2}) = 20.0$, $z_{\rm mid} = 7.0$, $\Delta z = 0.5$, and the damping wing cross-section is the Lorentzian far-wing of Ly-$\alpha$: $$\sigma_{\rm DW}(\Delta\nu) = \sigma_0 \cdot \frac{\Gamma_{\rm Ly\alpha}/(4\pi)}{(\Delta\nu)^2 + [\Gamma_{\rm Ly\alpha}/(4\pi)]^2},$$ with $\sigma_0 = 5.9 \times 10^{-14}\,{\rm cm}^2\,{\rm Hz}$ and $\Gamma_{\rm Ly\alpha} = 6.265 \times 10^8\,{\rm s}^{-1}$. This term is applied only redward of Ly-$\alpha$ at the source redshift and only for $z_s > 5$.

### Patchy Reionization

At $z \gtrsim 6$ the IGM is not uniformly neutral: galaxies reside in ionized bubbles of radius $R_b$ embedded in a neutral intergalactic medium with volume-averaged neutral fraction $\bar{x}_{\rm HI}$ (Miralda-Escudé 1998; Mason et al. 2018). The Gunn--Peterson optical depth for a fully neutral IGM is $\tau_{\rm GP} \approx 7.16 \times 10^5\,[(1+z_s)/10]^{3/2}$; the damping wing redward of Ly-$\alpha$ is computed by integrating the Lorentzian cross-section over the neutral path length outside the bubble. The two free parameters $\bar{x}_{\rm HI}$ and $R_b$ are directly constrained by Ly-$\alpha$ visibility statistics at $z > 6$.

## References

Asada, Yoshihisa, Guillaume Desprez, Chris J. Willott, et al. 2025. "Improving Photometric Redshifts of Epoch of Reionization Galaxies: A New Empirical Transmission Curve with Neutral Hydrogen Damping Wing Ly$\alpha$ Absorption." 983 (1): L2. <https://doi.org/10.3847/2041-8213/adc388>.

Inoue, Akio K., Ikkoh Shimizu, Ikuru Iwata, and Masayuki Tanaka. 2014. "An updated analytic model for attenuation by the intergalactic medium." 442 (2): 1805--20. <https://doi.org/10.1093/mnras/stu936>.

Mason, Charlotte A., Tommaso Treu, Mark Dijkstra, et al. 2018. "The Universe Is Reionizing at z $\sim$ 7: Bayesian Inference of the IGM Neutral Fraction Using Ly$\alpha$ Emission from Galaxies." 856: 2. <https://doi.org/10.3847/1538-4357/aab0a7>.

Miralda-Escudé, Jordi. 1998. "Reionization of the Intergalactic Medium and the Damping Wing of the Gunn-Peterson Trough." 501 (1): 15--22. <https://doi.org/10.1086/305799>.
