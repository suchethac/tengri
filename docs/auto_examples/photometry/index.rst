

.. _sphx_glr_auto_examples_photometry:

Photometry
==========

Filter curves and photometric fitting.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 4000 Å break is a sharp discontinuity in the stellar continuum at the boundary between the Balmer and Paschen series, caused by hydrogen Lyman absorption blanketing in the overlying atmosphere. In the rest frame it sits at 4000 Å for all galaxies. In the observer frame, the break shifts to longer wavelengths with increasing redshift: z × 4000 Å. This is why different photometric bands probe the break at different redshifts — the fundamental principle behind photo-z estimation and dust/age degeneracies.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_balmer_break_redshift_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_balmer_break_redshift_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Balmer break (4000 Å) position in observed-frame filters vs redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does a galaxy&#x27;s location in colour–colour space evolve with redshift? We compute SDSS g − r and r − z colours for two galaxy populations — a young star-forming and an old quiescent — across z = 0 to 3, with arrows marking the integer redshift stops. This is the reference picture for photometric redshift classifiers and for stellar-template grids.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_color_tracks_redshift_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_color_tracks_redshift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photometric colour tracks vs redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does the observed photometric flux of a FIXED-luminosity galaxy decline with redshift? We track a star-forming galaxy (log M* = 10.5, SFR = 10 Msun/yr) across z = 0.1 to 6 in three optical/infrared bands (SDSS r, JWST J, JWST H), visualizing the three physical effects:">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_cosmic_dimming_observed_flux_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_cosmic_dimming_observed_flux`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cosmic dimming and K-correction with redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Plot the ugriz filter transmission curves from the SDSS photometric system. Filters are loaded from the SVO Filter Profile Service via tengri&#x27;s filter registry.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_filter_curves_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_filter_curves`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SDSS Filter Transmission Curves</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare filter coverage from three different photometric surveys on the same mock galaxy SED — SDSS (optical ugriz), 2MASS (NIR JHKs), and HST (UV/optical ACS). Demonstrates how filter placement controls which spectral features are captured. Each panel overlays the filter throughputs (orange) on the same underlying SED (blue).">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_filter_set_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_filter_set_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Filter Set Comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Show a typical star-forming galaxy SED at z=1 with observed-frame filter throughputs overlaid as semi-transparent fills from 0.3 to 25 μm. This helps visualize which rest-frame stellar and dust features each photometric system samples across the spectrum.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_filter_throughput_overlay_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_filter_throughput_overlay`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">HST+JWST+LSST+Spitzer Filter Overlay on Star-Forming SED at z=1</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="What each photometric system measures depends on where its filters sit relative to the rest-frame spectral features. We overlay six common filter sets (GALEX NUV, SDSS ugriz, 2MASS JHK, WISE W1/W2/W3, Euclid YJH, JWST NIRCam wide bands) on top of an observed-frame star-forming galaxy SED at z = 0.5.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_galaxy_with_filters_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_galaxy_with_filters`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Galaxy SED with photometric filter coverage</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do K-corrections vary with redshift for different galaxy populations? K-corrections quantify the shift in filter response as galaxies move to higher redshifts: K(z) = −2.5 log₁₀[(1+z) × F_ν(z) / F_ν(0)] for a fixed rest-frame filter. We compute K(z) for the SDSS r-band across four galaxy types — young star-forming, old star-forming, red-sequence elliptical, and post-starburst — from z = 0.01 to z = 2.0. This illustrates why stellar mass measurements require careful K-corrections at high redshift and why colour-matched template sets dominate photometric redshift algorithms.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_k_correction_grid_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_k_correction_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">K-corrections as a function of redshift for different SED types</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The NUV−r colour is a sensitive probe of stellar age in galaxies. We show how a single-burst star formation history (tsnorm, truncated-skew-normal) evolves across the GALEX green valley (NUV−r ≈ 4–5 mag) as the stellar population ages from 0.05 to 5.5 Gyr. The colour exhibits a sharp discontinuity as the stellar population cools through the transition between young, UV-bright stars and older, redder populations.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_nuv_r_age_track_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_nuv_r_age_track`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">NUV−r colour vs stellar age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Generate a mock galaxy with SDSS ugriz photometry and fit it using tengri&#x27;s variational inference. Shows observed vs model photometry with error bars and residuals.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_photometric_fit_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_photometric_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photometric SED Fit</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Galaxy colour–magnitude diagram showing the distinct red and blue populations. We model two populations — 25 quiescent old galaxies (peak SFH ~8 Gyr) and 25 star-forming galaxies (continuous SFR) — varying stellar mass via log_peak_sfr. Each sample is placed at z = 0.05, computing u − r colour and rest-frame M_r magnitude. The colour bimodality and green valley are key signatures of galaxy assembly across cosmic time (Strateva et al. 2001 SDSS, Baldry et al. 2004).">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_red_sequence_blue_cloud_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_red_sequence_blue_cloud`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Red Sequence vs Blue Cloud Bimodality</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Rest-frame stellar continuum overlaid with redshifted SDSS ugriz transmission curves at z ∈ {0.1, 0.5, 1.0, 2.0}. The plot shows which features each band actually samples as a galaxy moves out — the textbook source of k-correction sign.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_redshift_filter_grid_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_redshift_filter_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Filter Sampling Across Redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep signal-to-noise ratio (SNR) from {3, 5, 10, 30, 100} on a fixed mock photometric galaxy in SDSS ugriz. Demonstrates how measurement uncertainty affects photometric precision. Higher SNR = tighter error bars.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_snr_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_snr_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">S/N Ratio Parameter Sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Color-color diagram in Spitzer IRAC bands (3.6, 4.5, 5.8, 8.0 μm) showing the Lacy+2007 / Donley+2012 AGN selection wedge. Population of 50 star-forming galaxies (z=0–2) are plotted as blue cloud; 10 AGN with varying bolometric luminosity cluster inside the wedge (red region) demonstrating the diagnostic power of mid-infrared colors for AGN identification.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_spitzer_irac_agn_wedge_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_spitzer_irac_agn_wedge`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Spitzer IRAC AGN Wedge Diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The UVJ (U−V vs V−J) diagram is a classic method for separating star-forming from quiescent galaxies. We populate it with four model tracks: (1) constant star-forming galaxies with varying dust optical depth, (2) an old quiescent population, (3) a post-starburst galaxy, and (4) a dusty starburst. The grey box marks the &quot;quiescent region&quot; from Williams+2009, a visual guide for identifying passive galaxies.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_uvj_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_uvj_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The UVJ colour–colour diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The WISE color-color diagram (Stern et al. 2012) is a powerful tool for separating AGN from star-forming galaxies using mid-infrared colors. The diagnostic exploits the fact that AGN emit power-law SEDs (flat in νLν) while star-forming galaxies have cooler dust emission (Rayleigh-Jeans slope at long wavelengths).">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_wise_agn_color_color_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_wise_agn_color_color`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">WISE W1–W2 vs W2–W3 Color-Color Diagram with Stern+2012 AGN Wedge</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/photometry/plot_balmer_break_redshift_evolution
   /auto_examples/photometry/plot_color_tracks_redshift
   /auto_examples/photometry/plot_cosmic_dimming_observed_flux
   /auto_examples/photometry/plot_filter_curves
   /auto_examples/photometry/plot_filter_set_comparison
   /auto_examples/photometry/plot_filter_throughput_overlay
   /auto_examples/photometry/plot_galaxy_with_filters
   /auto_examples/photometry/plot_k_correction_grid
   /auto_examples/photometry/plot_nuv_r_age_track
   /auto_examples/photometry/plot_photometric_fit
   /auto_examples/photometry/plot_red_sequence_blue_cloud
   /auto_examples/photometry/plot_redshift_filter_grid
   /auto_examples/photometry/plot_snr_sweep
   /auto_examples/photometry/plot_spitzer_irac_agn_wedge
   /auto_examples/photometry/plot_uvj_diagram
   /auto_examples/photometry/plot_wise_agn_color_color

