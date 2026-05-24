

.. _sphx_glr_auto_examples_photometry:

Photometry
==========

Filter curves and photometric fitting.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

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

    <div class="sphx-glr-thumbcontainer" tooltip="What each photometric system measures depends on where its filters sit relative to the rest-frame spectral features. We overlay six common filter sets (GALEX NUV, SDSS ugriz, 2MASS JHK, WISE W1/W2/W3, Euclid YJH, JWST NIRCam wide bands) on top of an observed-frame star-forming galaxy SED at z = 0.5.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_galaxy_with_filters_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_galaxy_with_filters`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Galaxy SED with photometric filter coverage</div>
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


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/photometry/plot_color_tracks_redshift
   /auto_examples/photometry/plot_filter_curves
   /auto_examples/photometry/plot_filter_set_comparison
   /auto_examples/photometry/plot_galaxy_with_filters
   /auto_examples/photometry/plot_photometric_fit
   /auto_examples/photometry/plot_redshift_filter_grid
   /auto_examples/photometry/plot_snr_sweep

