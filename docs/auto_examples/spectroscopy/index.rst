:orphan:

.. _sphx_glr_auto_examples_spectroscopy:

Spectroscopy
============

Absorption-line indices (D4000, Hδ) from stellar age and metallicity. Velocity dispersion and line broadening. Instrumental resolution effects (prism vs grating). High-redshift examples: JWST/NIRSpec out to z ≈ 6 Lyα emitters.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Stellar absorption features, especially the Mg b and Fe5270 line strengths, encode both age and metallicity in a classical anti-correlation pattern: at fixed metallicity, both features strengthen with age (population becomes older, cooler); at fixed age, increasing metallicity also strengthens the features (enhanced α-element abundances + stronger metal absorption).">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_bandheads_age_metallicity_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_bandheads_age_metallicity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar absorption bandheads: age and metallicity anti-correlation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Population diagnostics: single-burst SSP populations (3 SFH shapes × 5 ages × 3 metallicities = 45 points) colored by SFH shape and marked by metallicity. The Hδ_A vs D_n(4000) diagram discriminates starburst (high Hδ_A, low D_n(4000)) from quiescent (low Hδ_A, high D_n(4000)) populations and is sensitive to recent star formation and metal enrichment.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_d4000_hdelta_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_d4000_hdelta_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kauffmann+2003 D_n(4000) vs Hδ_A Diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="High-redshift star-forming galaxy with strong rest-frame UV and optical emission lines redshifted into the JWST NIRSpec G395M window (2.9–5.1 μm). Lines include Lyα, CIV, HeII, CIII], [OII], Hβ, [OIII], and Hα, each annotated with rest wavelength and vacuum-frame position.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_jwst_nirspec_high_z_spectrum_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_jwst_nirspec_high_z_spectrum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mock JWST NIRSpec G395M spectrum of a z=7 star-forming galaxy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="High-redshift Lyα emitter at z=6 with young age (~10 Myr), low metallicity (Z~0.1 Z☉), and minimal dust. The observed-frame spectrum (7000–13000 Å) reveals the redshifted Lyα emission line at 8512 Å, the Lyman break at 6384 Å, characteristic IGM blue-wing absorption, and the rest-UV continuum. Demonstrates Lyα radiative transfer and reionization-era observability.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_lae_spectrum_z6_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_lae_spectrum_z6`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-alpha emitter spectrum at z=6: IGM absorption and Lyα escape</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A z=5 JADES-like star-forming galaxy observed with JWST NIRSpec in two modes: PRISM (R~100, low-resolution) and G395M grating (R~1000, medium-resolution). The Hα line at rest 6564.61 Å appears as a single blob in PRISM but resolves into three peaks in the grating: Hα + [NII] λλ6549,6585 Å doublet.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_nirspec_prism_vs_grating_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_nirspec_prism_vs_grating`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">JWST NIRSpec PRISM vs G395M grating: Hα + [NII] resolution comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hδ_A peaks ~100 Myr post-quench, then decays as A-type stars die out. This absorption index traces the lifetime of A-type stars responsible for Balmer absorption, a key driver of galaxy quenching (Worthey &amp; Ottaviani 1997).">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_post_starburst_diagnostic_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_post_starburst_diagnostic`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Post-Starburst K+A Diagnostic: Hδ_A vs Time Since Quench</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spectral resolution R determines whether the Hα + [N II] emission-line complex appears as a single blended feature (low R) or resolves into three distinct lines (high R). Varying R from 100 to 10000 reveals the transition from kinematically degenerate at R~100 (SDSS/DESI-like) to fully resolved at R~5000 (JWST-like).">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_resolution_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_resolution_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Instrumental resolution controls Hα + [N II] line blending</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Mg b 5170 Å region of an old stellar population observed at spectral resolution R = 3000, convolved with increasing stellar velocity dispersion σ_v from 50 to 400 km/s. The classic kinematic diagnostic — line core depth tracks σ_v, asymmetric wings appear with rotational broadening (not modeled here, sigma only).">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_sigma_v_absorption_broadening_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_sigma_v_absorption_broadening`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Velocity-dispersion broadening of stellar absorption features</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="D4000, Hδ absorption, and the Mg b feature respond differently to stellar age and metallicity, providing complementary constraints when used together. D4000 rises with age; Hδ peaks at intermediate ages (A-star dominated); Mg b traces metallicity on the RGB/AGB branch.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_spectral_features_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_spectral_features`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Key Spectral Features as Age and Metallicity Probes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three of the most-used optical absorption / emission diagnostics evaluated on a single-burst stellar population from 30 Myr to 13 Gyr, at solar metallicity, no dust. The figure makes obvious which diagnostic responds on which timescale:">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_spectral_indices_vs_age_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_spectral_indices_vs_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Classic spectral indices vs single-burst age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Rest-frame spectrum with stellar population ages">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_spectrum_fit_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_spectrum_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rest-frame spectrum with stellar population ages</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep stellar velocity dispersion σ_v ∈ {50, 100, 150, 250, 400} km/s to show how line broadening increases with dynamical heating. The Mg b absorption feature (~5170 Å) widens progressively, demonstrating the kinematic signature of higher-velocity stellar populations.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_velocity_dispersion_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_velocity_dispersion_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Velocity Dispersion Sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Narrow-line regions sit at σ_v ≈ 50–300 km/s; a broad Hα component from the AGN accretion disk reaches thousands of km/s. The [NII] doublet is separated by 35.4 Å (6549.86 and 6585.28 Å vacuum), which corresponds to σ_v ≈ 1600 km/s — above that the two lines merge into the wing of Hα.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_velocity_offset_lines_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_velocity_offset_lines`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Emission-line velocity dispersion: narrow [NII] to broad Hα</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/spectroscopy/plot_bandheads_age_metallicity
   /auto_examples/spectroscopy/plot_d4000_hdelta_diagram
   /auto_examples/spectroscopy/plot_jwst_nirspec_high_z_spectrum
   /auto_examples/spectroscopy/plot_lae_spectrum_z6
   /auto_examples/spectroscopy/plot_nirspec_prism_vs_grating
   /auto_examples/spectroscopy/plot_post_starburst_diagnostic
   /auto_examples/spectroscopy/plot_resolution_sweep
   /auto_examples/spectroscopy/plot_sigma_v_absorption_broadening
   /auto_examples/spectroscopy/plot_spectral_features
   /auto_examples/spectroscopy/plot_spectral_indices_vs_age
   /auto_examples/spectroscopy/plot_spectrum_fit
   /auto_examples/spectroscopy/plot_velocity_dispersion_sweep
   /auto_examples/spectroscopy/plot_velocity_offset_lines

