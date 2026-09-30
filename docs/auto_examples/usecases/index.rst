:orphan:

.. _sphx_glr_auto_examples_usecases:

Use Cases
=========

Paper-style diagnostics: UVJ, JWST color-color, SFR indicators, age–dust degeneracy, main sequence evolution, dropout selection, spectral indices. Simulated-population Catalog examples.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

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

    <div class="sphx-glr-thumbcontainer" tooltip="Star formation rate calibrations depend on which wavelengths we observe. At high dust optical depth, UV-only SFR estimators severely underestimate the true SFR because dusty starbursts radiate most energy in the infrared. The hybrid SFR(UV+IR) recipe recovers the true SFR by combining both tracers.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sfr_uv_ir_consistency_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sfr_uv_ir_consistency`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SFR calibrations: UV only vs UV+IR hybrid estimators vs dust optical depth</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Replacing metallicity history Z(t) with its mass-weighted mean introduces 10–23% flux errors in u and 1–6% in z. The SED is a nonlinear mass-weighted sum of SSP templates; young metal-rich stars (dominant in UV) and old metal-poor stars do not average.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_simulation_seds_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_simulation_seds`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Predicting SEDs for a simulated population: what collapsing Z(t) costs</div>
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

   /auto_examples/usecases/plot_usecase_age_dust_redshift_degeneracy
   /auto_examples/usecases/plot_usecase_balmer_decrement_av
   /auto_examples/usecases/plot_usecase_d4000_vs_ssfr
   /auto_examples/usecases/plot_usecase_dropout_selection_z3
   /auto_examples/usecases/plot_usecase_hubble_sequence
   /auto_examples/usecases/plot_usecase_jwst_color_color
   /auto_examples/usecases/plot_usecase_main_sequence_cosmic_evolution
   /auto_examples/usecases/plot_usecase_sdss_lrg_stack_template
   /auto_examples/usecases/plot_usecase_sfr_uv_ir_consistency
   /auto_examples/usecases/plot_usecase_simulation_seds
   /auto_examples/usecases/plot_usecase_uv_slope_beta
   /auto_examples/usecases/plot_usecase_uvj_diagram

