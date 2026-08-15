:orphan:

.. _sphx_glr_auto_examples_usecases:

Use Cases
=========

Paper-style diagnostics: UVJ, JWST color-color, SFR indicators, mass completeness, age–dust degeneracy, emission-line correlations.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Left panel: The age–dust degeneracy as seen in optical g−r color. A 5 Gyr stellar population with no dust is nearly indistinguishable from a 1 Gyr population reddened by τ_diff = 0.4 when observed in optical broadband colors alone. A 2-D grid in (age, τ_diff) with iso-color contours reveals the orientation of the degeneracy—lines of constant color show why optical colors alone cannot break this ambiguity.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_age_dust_2d_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_age_dust_2d`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-dust degeneracy: optical colors vs. UV constraining power</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Optical photometry alone cannot uniquely break the degeneracy between stellar age, dust attenuation, and redshift — a fundamental limitation in photo-z and SED fitting. Three physically distinct galaxy populations can produce nearly identical SDSS ugriz photometry:">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_age_dust_redshift_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_age_dust_redshift_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The age-dust-redshift degeneracy in photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Balmer decrement measures dust attenuation via hydrogen recombination line ratios: Hα / H-beta is sensitive to extinction (Calzetti et al. 2000). Without dust, the intrinsic ratio is ~2.78–2.86 (Case B). Here we sweep dust optical depth (τ_diff ∈ [0, 2]) and measure how the predicted Hα and H-beta change. We derive A_V = 1.086 × τ_diff and compare against the Calzetti+2000 expectation.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_balmer_decrement_av_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_balmer_decrement_av`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Balmer Decrement Tests Dust Attenuation on Emission Lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Planck 2018: H₀ = 67.4 km/s/Mpc, Ω_M = 0.315. Riess et al. 2022: H₀ = 73.04 ± 1.04 km/s/Mpc, Ω_M = 0.30. Apparent magnitude shift is ~0.15 mag across z = 0.05–3.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_cosmology_distance_modulus_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_cosmology_distance_modulus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hubble Tension: Cosmology-dependent distance modulus</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="tengri.cosmology exposes the standard FRW distances — comoving, luminosity, angular-diameter — and lookback time as pure-JAX functions over a Planck-18 default. They are differentiable, JIT-able, and interchangeable with astropy&#x27;s API for tengri&#x27;s own forward model.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_cosmology_ladder_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_cosmology_ladder`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cosmological distance ladder and the K-correction for a flat SED</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Kauffmann+2003 separation of star-forming and quiescent SDSS galaxies plotted as a sample track: stellar-burst age varied from 30 Myr to 11 Gyr (single-burst SSP), with each model giving a (``D_n(4000)``, sSFR) pair.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_d4000_vs_ssfr_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_d4000_vs_ssfr`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">D_n(4000) – specific SFR: the Kauffmann+2003 sequence</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Steidel+1996 U-dropout box is calibrated for a specific filter set and does not transfer to arbitrary filters: (U − G) &gt; 1.0, (G − R) &lt; 1.5, (U − G) &gt; 1.5(G − R) + 0.3. True z~3 galaxies cluster inside; lower-redshift galaxies fall outside.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_dropout_selection_z3_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_dropout_selection_z3`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">z~3 Lyman-break galaxy U-dropout selection: color-color diagnosis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Mannucci+2010 fundamental metallicity relation (FMR) describes how a galaxy&#x27;s gas-phase metallicity (Z) depends not only on its stellar mass (M) but also on its star formation rate (SFR). This three-parameter relation is a schematic* demonstration of the physical interplay between assembly, star formation, and chemical enrichment.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_fundamental_metallicity_relation_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_fundamental_metallicity_relation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The Fundamental Metallicity Relation: M*-Z-SFR three-body interaction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Coleman, Wu &amp; Weedman 1980 spectral templates remain the textbook illustration of how the integrated SED morphs along the Hubble sequence — from quiescent ellipticals with deep 4000 Å breaks to gas-rich irregulars dominated by ongoing star formation and nebular emission.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_hubble_sequence_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_hubble_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A morphological atlas: E, Sa, Sb, Sc, Im galaxy SEDs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="JWST NIRCam color-color diagnostics for high-z galaxy classification">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_jwst_color_color_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_jwst_color_color`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">JWST NIRCam color-color diagnostics for high-z galaxy classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Kennicutt+1998 baseline calibrations under constant-SFR assumption: L_FUV(1500 Å): SFR/L_FUV = 1.4 × 10⁻²⁸; L_Hα: SFR/L_Hα = 7.9 × 10⁻⁴²; L_IR(8–1000 μm): SFR/L_IR = 4.5 × 10⁻⁴⁴. Stochastic SFH introduces variance in each indicator; Hα most sensitive to recent star formation.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_kennicutt_sfr_calibrations_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_kennicutt_sfr_calibrations`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kennicutt+1998 SFR calibrations: baseline + stochastic variance</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates how strong gravitational lensing elevates intrinsically-faint high-redshift (z=7) galaxies above the JWST NIRCam 5σ detection threshold.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_lensed_galaxy_magnification_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_lensed_galaxy_magnification`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Strong-lensing magnification: EoR galaxy SEDs at μ = 1, 5, 20, 100</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The star-forming main sequence (MS) defines a tight relation between stellar mass (M*) and star formation rate (SFR) for actively forming galaxies. This example demonstrates how the MS shifts upward by ~0.7 dex from z=0 to z=2, reflecting the Universe&#x27;s peak epoch of star formation. The left panel shows recovery of the z~0 MS from mock SEDModel photometry; the right panel reveals MS evolution to high-z.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_main_sequence_cosmic_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_main_sequence_cosmic_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Star-forming main sequence: z = 0 → 2 cosmic evolution + recovery</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a population of N=200 quiescent galaxies replicating the SDSS Luminous Red Galaxy (LRG) sample selection (Eisenstein et al. 2001, SDSS-I): old, massive systems at z~0.3 with log M* ≈ 11 and ages sampling the red-sequence range Uniform(6, 11) Gyr (Thomas et al. 2005).">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sdss_lrg_stack_template_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sdss_lrg_stack_template`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SDSS Luminous Red Galaxy Stacked Template Spectrum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The cosmic star formation rate density (SFRD) — the total stellar mass created per unit time per unit comoving volume — rises from z~0 to peak at z~2, then declines toward higher redshift. Madau &amp; Dickinson 2014 assembled multi-wavelength observational data and fit a smooth analytic form:">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sfh_to_madau_dickinson_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sfh_to_madau_dickinson`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Madau-Dickinson 2014 cosmic SFRD(z) from a population of mock galaxies</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Star formation rate calibrations depend on which wavelengths we observe. At high dust optical depth, UV-only SFR estimators severely underestimate the true SFR because dusty starbursts radiate most energy in the infrared. The hybrid SFR(UV+IR) recipe recovers the true SFR by combining both tracers.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sfr_uv_ir_consistency_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sfr_uv_ir_consistency`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SFR calibrations: UV only vs UV+IR hybrid estimators vs dust optical depth</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The stellar mass function (SMF) describes the number density of galaxies as a function of stellar mass, a fundamental probe of galaxy assembly. The Schechter function provides an excellent fit to observed SMF across cosmic time:">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_stellar_mass_luminosity_function_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_stellar_mass_luminosity_function`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Mass and Luminosity Functions from Mock Survey</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Tully-Fisher relation is a tight empirical correlation between the baryonic mass of disc galaxies and their observed rotation velocity, parametrized as M_baryon ∝ V_rot^4 (slope 4.0 on the log-log plane).">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_tully_fisher_relation_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_tully_fisher_relation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tully-Fisher Relation: Baryonic Mass — Rotation Velocity Scaling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The infrared excess (IRX = L_IR / L_FUV) versus UV-continuum slope β diagram is the standard tool for inferring attenuation in star-forming galaxies. However, β is degenerate between dust and stellar age: young dusty and old dust-free populations both exhibit red UV continua.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uv_slope_beta_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uv_slope_beta`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV slope β degeneracy: dust optical depth and stellar age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="For any self-consistent dust model: L_IR ≈ L_UV_absorbed. Using tabulated Dale14 templates, agreement reaches ~5% across τ_V ∈ {0, 0.1, …, 4}. Non-conservation flags calibration issues in the dust emission routing.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uv_to_ir_bolometric_balance_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uv_to_ir_bolometric_balance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust energy balance: L_IR = L_UV_absorbed across opacity variations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Rest-frame U−V vs V−J colors separate star-forming from quiescent galaxies. The Williams+2009 quiescent wedge marks the boundary between dusty star-forming and passive systems.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uvj_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uvj_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UVJ diagram: rest-frame colors separate star-forming from quiescent</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/usecases/plot_usecase_age_dust_2d
   /auto_examples/usecases/plot_usecase_age_dust_redshift_degeneracy
   /auto_examples/usecases/plot_usecase_balmer_decrement_av
   /auto_examples/usecases/plot_usecase_cosmology_distance_modulus
   /auto_examples/usecases/plot_usecase_cosmology_ladder
   /auto_examples/usecases/plot_usecase_d4000_vs_ssfr
   /auto_examples/usecases/plot_usecase_dropout_selection_z3
   /auto_examples/usecases/plot_usecase_fundamental_metallicity_relation
   /auto_examples/usecases/plot_usecase_hubble_sequence
   /auto_examples/usecases/plot_usecase_jwst_color_color
   /auto_examples/usecases/plot_usecase_kennicutt_sfr_calibrations
   /auto_examples/usecases/plot_usecase_lensed_galaxy_magnification
   /auto_examples/usecases/plot_usecase_main_sequence_cosmic_evolution
   /auto_examples/usecases/plot_usecase_sdss_lrg_stack_template
   /auto_examples/usecases/plot_usecase_sfh_to_madau_dickinson
   /auto_examples/usecases/plot_usecase_sfr_uv_ir_consistency
   /auto_examples/usecases/plot_usecase_stellar_mass_luminosity_function
   /auto_examples/usecases/plot_usecase_tully_fisher_relation
   /auto_examples/usecases/plot_usecase_uv_slope_beta
   /auto_examples/usecases/plot_usecase_uv_to_ir_bolometric_balance
   /auto_examples/usecases/plot_usecase_uvj_diagram

