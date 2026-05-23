

.. _sphx_glr_auto_examples_sps:

Stellar Population Synthesis
=============================

DSPS-based SSP grids: age, metallicity, and spectral properties.

- ``plot_ssp_grid.py`` — SSP grid visualization (age, metallicity, spectrum)



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The hydrogen-ionising photon production rate Q_H of a simple stellar population drops by ~5 dex from 1 Myr to 100 Myr as O stars die. Different SSP libraries predict different Q_H(t) because they differ in upper-IMF treatment, stellar rotation, and (most dramatically) whether massive binaries are included — BPASS extends the Q_H-producing phase to ~30 Myr.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ionising_lum_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ionising_lum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionising-photon production rate vs SSP age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="M_★/L_band rises with stellar age in every band; the rise is steepest in g, where massive young stars dominate, and shallowest in K_s, where red giants contribute at every age past the first ~100 Myr.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_mass_to_light_ratios_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_mass_to_light_ratios`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar mass-to-light ratios vs SSP age, per band</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Different stellar population synthesis codes use different stellar spectral libraries, isochrone families, and binary treatments. The SED of a ~1 Gyr-old, solar-metallicity simple stellar population already shows visible differences in the UV (BPASS binaries add a hot continuum) and in the NIR (treatment of TP-AGB).">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_sps_library_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_sps_library_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP library comparison at a fixed age and metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A single stellar population transitions from UV-dominated (young, hot) to NIR-dominated (old, red) with age. Peak-normalized λF_λ on log-log axes makes the temperature inversion visible across five representative ages at solar metallicity.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_age_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_age_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Population Aging: SSP at Solar Metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The choice of SSP library propagates into the colors a photometric fitter recovers — a single fixed galaxy SFH and dust law, rebuilt with FSPS-MIST, FSPS-Padova/MILES, BPASS, BC03, and CB19 in turn, produces a noticeable spread in NUV − r, u − g, g − r, and r − K.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_color_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_color_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Galaxy broadband colors depend on the SSP library</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four-panel overview of the DSPS SSP grid: age evolution at fixed metallicity, metallicity evolution at fixed age, monochromatic flux vs age, and color-color diagram across the full grid. Shows how stellar populations age from UV-hot to IR-red as they cool.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP Grid: Age and Metallicity Evolution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Different Initial Mass Functions produce different M/L ratios at fixed age and metallicity. We rescale a Chabrier SSP by literature M/L values for Chabrier, Kroupa, and Salpeter at 1 Gyr, solar metallicity. The NIR (where massive stars dominate the mass budget) is most diagnostic of IMF choice.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_imf_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_imf_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">IMF Comparison: Mass-to-Light Ratio</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Metallicity reddens the optical continuum and shifts iron-peak absorption features in the near-IR. We show five metallicity points spanning the SSP grid at fixed age (1 Gyr). Peak-normalized λF_λ makes spectral shape variations visible without large luminosity differences obscuring them.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_metallicity_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_metallicity_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Metallicity Effects on SED</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/sps/plot_ionising_lum
   /auto_examples/sps/plot_mass_to_light_ratios
   /auto_examples/sps/plot_sps_library_compare
   /auto_examples/sps/plot_ssp_age_sweep
   /auto_examples/sps/plot_ssp_color_compare
   /auto_examples/sps/plot_ssp_grid
   /auto_examples/sps/plot_ssp_imf_compare
   /auto_examples/sps/plot_ssp_metallicity_sweep

