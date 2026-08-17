:orphan:

.. _sphx_glr_auto_examples_photometry:

Photometry
==========

Broadband filter selection, k-corrections, cosmological dimming. Diagnostic planes: UVJ, NUV–r, WISE/IRAC AGN wedges. Photometric-redshift color degeneracies.


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

    <div class="sphx-glr-thumbcontainer" tooltip="How many photometric bands are needed to recover stellar mass accurately? We mock a single galaxy with fixed parameters at different signal-to-noise levels using progressively larger filter sets, then MAP-fit to measure the recovered mass uncertainty. The figure shows that stellar mass constraints improve dramatically with filter count: a 2-band measurement is degenerate (wide posterior), while a 10-band panchromatic set (optical + NIR + mid-IR) tightens the mass estimate by an order of magnitude or more.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_band_count_mass_recovery_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_band_count_mass_recovery`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar mass recovery with increasing photometric band count</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does a galaxy&#x27;s location in color–color space evolve with redshift? We compute SDSS g − r and r − z colors for two galaxy populations — a young star-forming and an old quiescent — across z = 0 to 3, with arrows marking the integer redshift stops. This is the reference picture for photometric redshift classifiers and for stellar-template grids.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_color_tracks_redshift_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_color_tracks_redshift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photometric color tracks vs redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does the observed photometric flux of a FIXED-luminosity galaxy decline with redshift? We track a star-forming galaxy (log M* = 10.5, SFR = 10 M☉/yr) across z = 0.1 to 6 in three optical/infrared bands (SDSS r, JWST J, JWST H), visualizing the three physical effects:">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_cosmic_dimming_observed_flux_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_cosmic_dimming_observed_flux`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cosmic dimming and K-correction with redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The same star-forming galaxy SED is intercepted by three different filter sets — SDSS ugriz (optical), 2MASS JHKs (near-infrared), and HST ACS F435W/F606W/F814W (UV-optical). Each panel overlays the survey&#x27;s throughputs on the shared SED so the reader sees, at a glance, which spectral features (the 4000-Å break, Hα + [N II], the 1.6-μm stellar bump) fall inside each band.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_filter_set_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_filter_set_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Filter placement decides which spectral features a survey can see</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The Subaru HSC and Blanco DECam i-bands have different red-edge cutoffs (HSC i-2 at ~850 nm, DECam i at ~870 nm). This 20 nm difference produces measurable color offsets when a sharp spectral feature sweeps through the i-band — particularly the Lyman break at z~3.5–4.5. We show (r − i) colors for an LBG template across both filter sets to highlight the divergence in the high-redshift regime.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_hsc_vs_des_color_high_z_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_hsc_vs_des_color_high_z`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">HSC vs DES filter i-band differences at high redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do K-corrections vary with redshift for different galaxy populations? K-corrections quantify the shift in filter response as galaxies move to higher redshifts: K(z) = −2.5 log₁₀[(1+z) × F_ν(z) / F_ν(0)] for a fixed rest-frame filter. We compute K(z) for the SDSS r-band across four galaxy types — young star-forming, old star-forming, red-sequence elliptical, and post-starburst — from z = 0.01 to z = 2.0. This illustrates why stellar mass measurements require careful K-corrections at high redshift and why color-matched template sets dominate photometric redshift algorithms.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_k_correction_grid_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_k_correction_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">K-corrections as a function of redshift for different SED types</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The NUV−r color is a sensitive probe of stellar age in galaxies. We show how a single-burst star formation history (tsnorm, truncated-skew-normal) evolves across the GALEX green valley (NUV−r ≈ 4–5 mag) as the stellar population ages from 0.05 to 5.5 Gyr. The color exhibits a sharp discontinuity as the stellar population cools through the transition between young, UV-bright stars and older, redder populations.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_nuv_r_age_track_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_nuv_r_age_track`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">NUV−r color vs stellar age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A higher-redshift counterpart to the SDSS quickstart fit. We mock JWST NIRCam wide-band photometry of a star-forming galaxy at z=3 (S/N=15), run a MAP fit, and show the recovered SED + per-band residuals. NIRCam samples the rest-frame UV-optical at this redshift, so the SFH and dust attenuation are the dominant levers.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_photometric_fit_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_photometric_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recovering a z=3 galaxy from JWST NIRCam photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two galaxies with very different star formation histories and dust can collide in color–color space, making photo-z ambiguous. Here, a young dusty star-forming galaxy at z≈0.5 and an old quiescent galaxy at z≈2 follow nearly identical (u-g, g-r) tracks and intersect at a single point. This shows why intermediate-wavelength photometry is essential for robust photo-z classification.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_photoz_color_degeneracy_grid_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_photoz_color_degeneracy_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photo-z degeneracy in color–color space: low-z dusty vs high-z quiescent</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Galaxy color–magnitude diagram showing the distinct red and blue populations. We model two populations — 25 quiescent old galaxies (peak SFH ~8 Gyr) and 25 star-forming galaxies (continuous SFR) — varying stellar mass via log_total_mass. Each sample is placed at z = 0.05, computing u − r color and rest-frame M_r magnitude. The color bimodality and green valley are key signatures of galaxy assembly across cosmic time (Strateva et al. 2001 SDSS, Baldry et al. 2004).">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_red_sequence_blue_cloud_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_red_sequence_blue_cloud`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Red Sequence vs Blue Cloud Bimodality</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Same rest-frame star-forming SED at four observed redshifts, with the SDSS ugriz throughputs plotted in their observed position so the reader sees which rest-frame features each band samples. The Balmer break enters the u band by z=1; by z=2 the bands fall longward of the 4000-A break entirely. This is the geometric source of the k-correction&#x27;s sign.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_redshift_filter_grid_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_redshift_filter_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SDSS ugriz sweep through a galaxy SED as z grows from 0.1 to 2</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Same star-forming galaxy, same SDSS ugriz set, three signal-to-noise levels (5, 20, 100). For each S/N we mock the photometry, run a MAP fit, and overlay the recovered SED on the truth. The figure surfaces the expected scaling — posterior offset and band-by-band residuals shrink as 1/S/N — and makes the inference cost concrete: even at S/N=5 the dust amplitude is degenerate enough that a single MAP run misses it by ~0.3 mag in the u band.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_snr_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_snr_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Posterior width tracks 1/S/N for fixed-truth SDSS photometry</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The UVJ (U−V vs V−J) diagram is a classic method for separating star-forming from quiescent galaxies. We populate it with four model tracks: (1) constant star-forming galaxies with varying dust optical depth, (2) an old quiescent population, (3) a post-starburst galaxy, and (4) a dusty starburst. The gray box marks the &quot;quiescent region&quot; from Williams+2009, a visual guide for identifying passive galaxies.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_uvj_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_uvj_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The UVJ color–color diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The WISE color-color diagram (Stern et al. 2012) is a tool for separating AGN from star-forming galaxies using mid-infrared colors. The diagnostic exploits the fact that AGN emit power-law SEDs (flat in νLν) while star-forming galaxies have cooler dust emission (Rayleigh-Jeans slope at long wavelengths).">

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
   /auto_examples/photometry/plot_band_count_mass_recovery
   /auto_examples/photometry/plot_color_tracks_redshift
   /auto_examples/photometry/plot_cosmic_dimming_observed_flux
   /auto_examples/photometry/plot_filter_set_comparison
   /auto_examples/photometry/plot_filter_throughput_overlay
   /auto_examples/photometry/plot_galaxy_with_filters
   /auto_examples/photometry/plot_hsc_vs_des_color_high_z
   /auto_examples/photometry/plot_k_correction_grid
   /auto_examples/photometry/plot_nuv_r_age_track
   /auto_examples/photometry/plot_photometric_fit
   /auto_examples/photometry/plot_photoz_color_degeneracy_grid
   /auto_examples/photometry/plot_red_sequence_blue_cloud
   /auto_examples/photometry/plot_redshift_filter_grid
   /auto_examples/photometry/plot_snr_sweep
   /auto_examples/photometry/plot_spitzer_irac_agn_wedge
   /auto_examples/photometry/plot_uvj_diagram
   /auto_examples/photometry/plot_wise_agn_color_color

