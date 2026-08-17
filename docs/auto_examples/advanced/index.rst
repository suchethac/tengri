:orphan:

.. _sphx_glr_auto_examples_advanced:

Advanced Topics
===============

Hierarchical population inference, gradient diagnostics, batch fitting, panchromatic multi-component SEDs, joint photometry + spectroscopy.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The on-ramp for adding a custom physics block to tengri. Subclass SEDModelComponent, declare name, parameter_prefix, priors as class attributes, and implement predict(p, sed_in, wave). __init_subclass__ registers the new variant and auto-fills the inputs() / outputs() contracts.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_custom_attenuation_component_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_custom_attenuation_component`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Authoring a new physics block with SEDModelComponent</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Validates that AB magnitude zero-point definitions are consistent across filters. Compares photometry converted to magnitude via the formula m_AB = -2.5 log10(F_ν) - 48.6 against tengri&#x27;s built-in magnitude conversion. The AB magnitude system requires this relationship to hold across all filters—any deviation signals a zero-point calibration issue.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_ab_mag_zero_point_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_ab_mag_zero_point`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AB Magnitude Zero-point Consistency Check</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Verify tengri&#x27;s Calzetti implementation against Eq. 1 in Calzetti et al. 2000 (ApJ 533, 682). The canonical k(V=5500 Å) = 4.05 must be reproduced exactly.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_calzetti_kv_norm_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_calzetti_kv_norm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diagnostic: Calzetti 2000 attenuation law vs. published formula</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In the dust-free limit with Case B recombination (T_e=10,000 K, n_e=100 cm^-3), the intrinsic Hα/Hβ ratio is 2.86, nearly independent of ionization parameter and metallicity below ~0.5 Z☉ (Storey &amp; Hummer 1995, MNRAS 272, 41). This diagnostic checks that tengri&#x27;s Cue nebular emulator reproduces the canonical value across its (logU, logZ_gas) grid, identifying any library drift or implementation errors.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_case_b_balmer_ratio_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_case_b_balmer_ratio`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Case B Hα/Hβ ratio across ionization and metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="External ground truth: Chabrier 2003 PASP 115 763, Eq. 16–17.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_chabrier_imf_norm_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_chabrier_imf_norm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Chabrier 2003 IMF — analytic normalization and SSP mean stellar mass</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compares tengri&#x27;s Planck18 cosmology implementation (DSPS-backed, Ω_m = 0.315, h = 0.674) against astropy.cosmology.Planck18 (which uses slightly different parameter values) across z = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]. Validates luminosity distance d_L(z), comoving distance d_C(z), age(z), and comoving volume element consistency. Residuals should be stable across z and &lt;1% due to underlying parameter differences rather than numerical bugs. Tengri&#x27;s PLANCK18 parameters (Om0=0.315, h=0.674) match Planck 2018 published values.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_cosmology_vs_astropy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_cosmology_vs_astropy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cosmological Distance Validation: tengri vs Astropy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Validates that the νF_ν peak position of Draine &amp; Li (2007) dust emission templates follows Wien&#x27;s displacement law, an effective dust temperature diagnostic. The DL07 templates encode different dust temperatures for different U_min values; the Wien law applied to the νF_ν peak recovers this temperature.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_dl07_temperature_proxy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_dl07_temperature_proxy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine & Li 2007: dust temperature from SED peak position</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Diagnostic figure comparing tengri&#x27;s Calzetti and Cardelli/CCM89 attenuation laws against the reference implementations in the dust_extinction package (Barbary et al., widely used by astropy workflows). Residuals reveal systematic offsets and validity ranges. If k(λ) residuals exceed 5% outside known singularities, the implementation may need verification against the original papers.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_dust_extinction_vs_pypi_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_dust_extinction_vs_pypi`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation law validation: tengri vs dust_extinction PyPI package</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tests dust attenuation–emission consistency via energy conservation. Sweeps diffuse optical depth τ_diff while measuring agreement between independent attenuation and emission modules. Ratio = L_emitted / L_absorbed should equal 1.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_energy_balance_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_energy_balance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Energy balance: dust absorption vs. emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When ionizing photons escape (f_esc &gt; 0), fewer LyC photons ionize the ISM within the galaxy, suppressing all nebular line emission proportionally: L(Hα) ∝ (1 − f_esc) × Q_H, where Q_H is the intrinsic ionizing photon rate.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_fesc_lyc_conservation_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_fesc_lyc_conservation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman continuum escape fraction conservation in Cue nebular model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Validates the AB magnitude photometric filter convolution formula by computing the effective F_ν through a photometric filter manually and comparing against predict_photometry(). The AB convention defines the filter-weighted flux as">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_filter_integral_manual_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_filter_integral_manual`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Manual Filter Integral vs predict_photometry Consistency Check</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="tengri is a differentiable JAX package. Every model gradient ∂L/∂θ computed via jax.grad() should numerically match a central finite-difference approximation. This diagnostic builds a star-forming model with several free parameters, defines a chi-squared loss, and compares autodiff vs FD gradients for each parameter. A mismatch (&gt;1e-3) indicates a non-differentiable operation.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_gradient_finite_difference_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_gradient_finite_difference`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Autodiff gradients vs. finite-difference derivatives: diagnostic verification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Verifies that JIT-compiled predictions are bit-identical to eager-mode evaluations. For predict_photometry and predict(params).lines, we sample random parameter sets and compare max relative difference between eager and JIT outputs. A value &lt; 1e-10 confirms no spurious numerical divergence; &gt; 1e-10 suggests platform-dependent floating-point behavior.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_jit_concrete_identity_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_jit_concrete_identity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">JAX JIT Compilation: Eager vs Compiled Numerical Equivalence</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Diagnostic: the Hα luminosity traces the ionizing photon rate from young stars, which correlates with the instantaneous SFR. Kennicutt (1998, ApJ 498 541, Eq. 2) calibrated this relationship for Salpeter IMF; for Chabrier IMF (used by tengri), the coefficient is 4.97e-42: SFR / (M☉/yr) = 4.97e-42 × L(Hα) / (erg/s).">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_kennicutt_halpha_sfr_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_kennicutt_halpha_sfr`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hα-to-SFR calibration against Kennicutt (1998)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Diagnostic: Lyman-series line optical depth τ_LS vs observed wavelength in the Lyman-alpha forest, comparing tengri&#x27;s Madau+1995 model to manual calculation from published coefficients (Madau 1995 Table 1, Eq. 15).">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_madau_published_table_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_madau_published_table`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Validating IGM transmission against Madau 1995 published table</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Internal consistency check: the cumulative SFR integral ∫₀ᵗ SFR(t) dt should equal the stellar mass returned by predict_properties(). This diagnostic varies the DPL SFH parameters and verifies that the two pathways (manual trapz of the trajectory vs library integration) agree to ~0.1%. Discrepancies &gt; 5% trigger a warning and would indicate a bug in either the SFH trajectory or the mass integration kernel.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_mass_conservation_sfh_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_mass_conservation_sfh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mass conservation in SFH: manual integration vs predict_properties</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When constructing a model with priors like Uniform(0, 2), the sampling method model.spec.sample(key) should actually draw from that declared distribution. This example verifies the sampling implementation empirically: we draw 10000 samples from a model with mixed prior types (Uniform, LogUniform) and compare each empirical histogram against its theoretical PDF.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_prior_sample_distributions_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_prior_sample_distributions`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diagnosing prior sampling distributions with empirical histograms</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The rest-frame SED depends only on intrinsic galaxy properties (SFH, dust, metallicity, nebular, AGN) and is independent of redshift. Redshift only enters via the observation (wavelength shift, distance dimming, IGM attenuation). This diagnostic verifies that Prediction.rest_sed returns bit-identical SEDs across a range of redshifts for identical intrinsic parameters. Age-of-the-Universe constraints at high-z may truncate the SFH legitimately, producing smooth variation; any non-smooth jump signals a coupling bug.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_redshift_rest_invariance_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_redshift_rest_invariance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rest-frame SED Redshift Invariance</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Verifies the SED chain is additive by comparing the full pred.rest_sed() output against a manual sum of per-component SEDs. The forward model chains stellar continuum through dust attenuation, dust emission, and nebular processing; if modular, the sum should reconstruct the total.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_sed_additivity_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_sed_additivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SED additivity: stellar, dust attenuation, emission, and nebular components</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Stellar population synthesis grids cover finite (age, metallicity) ranges. This diagnostic probes what happens at boundaries: clip, extrapolate, or error? We fix the SFH and vary stellar metallicity across the SSP grid boundary—inside, at the edge, and beyond. The resulting SEDs reveal the interpolation behavior; any NaN or error surfaces immediately in the plot.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_ssp_grid_edge_behavior_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_ssp_grid_edge_behavior`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP grid edge behavior: clipping, extrapolation, NaN</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The WavePrecomp approximation pre-integrates SSP × filter LUTs and interpolates photometry through a redshift table, trading exact calculations for speed. This diagnostic compares exact-wave-grid photometry against WavePrecomp variants at different ztable densities n_z, showing how fractional errors decrease with finer redshift grids.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_waveprecomp_accuracy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_waveprecomp_accuracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">WavePrecomp photometric accuracy across redshift grids</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cramér-Rao bound from the Fisher Information Matrix shows that SDSS 5-band photometry alone cannot separately constrain age, dust, and metallicity. Adding NIR or MIR bands breaks the degeneracy by factors of 2–5×, quantifying the information gain from multiwavelength coverage.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_fisher_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_fisher_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-Dust-Metallicity Degeneracy: Fisher Analysis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Computes the Jacobian d(flux)/d(theta) of the forward model and displays it as a heatmap showing which photometric bands are sensitive to which physical parameters. Each column shows normalized sensitivity to one parameter; dark blue/red indicates strong dependence.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_gradient_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_gradient_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient Sensitivity Heatmap</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates tengri&#x27;s Observation API for joint fitting across two data streams. Creates a mock galaxy with SDSS photometry and low-resolution spectroscopy, then recovers parameters via MAP. Shows how spectroscopy breaks photometric degeneracies.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_joint_fit_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_joint_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Joint Photometry + Spectroscopy Fit</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Low-level component orchestration using build_components and run_components. For production use, see plot_joint_fit.py and plot_radio_xray.py which use the SEDModel.build() nested-dict grammar.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_orchestrator_demo_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_orchestrator_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Component Orchestrator End-to-End</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a full galaxy SED spanning X-ray to radio wavelengths. Shows stellar emission, dust attenuation, dust IR emission, radio synchrotron, and X-ray binary contributions. Demonstrates tengri&#x27;s multiwavelength physics modules for radio and X-ray—no SSP data required for these components.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_radio_xray_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_radio_xray`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Panchromatic SED: UV to Radio</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/advanced/plot_custom_attenuation_component
   /auto_examples/advanced/plot_diag_ab_mag_zero_point
   /auto_examples/advanced/plot_diag_calzetti_kv_norm
   /auto_examples/advanced/plot_diag_case_b_balmer_ratio
   /auto_examples/advanced/plot_diag_chabrier_imf_norm
   /auto_examples/advanced/plot_diag_cosmology_vs_astropy
   /auto_examples/advanced/plot_diag_dl07_temperature_proxy
   /auto_examples/advanced/plot_diag_dust_extinction_vs_pypi
   /auto_examples/advanced/plot_diag_energy_balance
   /auto_examples/advanced/plot_diag_fesc_lyc_conservation
   /auto_examples/advanced/plot_diag_filter_integral_manual
   /auto_examples/advanced/plot_diag_gradient_finite_difference
   /auto_examples/advanced/plot_diag_jit_concrete_identity
   /auto_examples/advanced/plot_diag_kennicutt_halpha_sfr
   /auto_examples/advanced/plot_diag_madau_published_table
   /auto_examples/advanced/plot_diag_mass_conservation_sfh
   /auto_examples/advanced/plot_diag_prior_sample_distributions
   /auto_examples/advanced/plot_diag_redshift_rest_invariance
   /auto_examples/advanced/plot_diag_sed_additivity
   /auto_examples/advanced/plot_diag_ssp_grid_edge_behavior
   /auto_examples/advanced/plot_diag_waveprecomp_accuracy
   /auto_examples/advanced/plot_fisher_degeneracy
   /auto_examples/advanced/plot_gradient_sensitivity
   /auto_examples/advanced/plot_joint_fit
   /auto_examples/advanced/plot_orchestrator_demo
   /auto_examples/advanced/plot_radio_xray

