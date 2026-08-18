:orphan:

.. _sphx_glr_auto_examples_photometry:

Photometry
==========

Broadband filter selection, cosmological dimming, color tracks and redshift evolution. Diagnostic planes: WISE/IRAC AGN wedges, red sequence/blue cloud. Photometric-redshift color degeneracies.


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

    <div class="sphx-glr-thumbcontainer" tooltip="Show a typical star-forming galaxy SED at z=1 with observed-frame filter throughputs overlaid as semi-transparent fills from 0.3 to 25 μm. This helps visualize which rest-frame stellar and dust features each photometric system samples across the spectrum.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_filter_throughput_overlay_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_filter_throughput_overlay`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">HST+JWST+LSST+Spitzer Filter Overlay on Star-Forming SED at z=1</div>
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
   /auto_examples/photometry/plot_color_tracks_redshift
   /auto_examples/photometry/plot_cosmic_dimming_observed_flux
   /auto_examples/photometry/plot_filter_throughput_overlay
   /auto_examples/photometry/plot_photoz_color_degeneracy_grid
   /auto_examples/photometry/plot_red_sequence_blue_cloud
   /auto_examples/photometry/plot_wise_agn_color_color

