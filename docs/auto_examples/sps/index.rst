:orphan:

.. _sphx_glr_auto_examples_sps:

Stellar Population Synthesis
============================

SSP grid and age/metallicity sweeps, IMF choice, mass-to-light band comparison, and library shootout.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Initial Mass Function (IMF) parameterizes the fraction of massive versus low-mass stars born during star formation. Chabrier, Kroupa, and Salpeter IMFs differ most in the high-mass end: Salpeter has more massive stars, producing a higher M/L ratio (more mass per unit light) and harder UV continua. We vary IMF while fixing SFH, age, and metallicity, overlaying rest-frame νL_ν to reveal the IMF signature in the SED continuum shape and M/L.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_imf_choice_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_imf_choice_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Initial Mass Function choice and stellar mass-to-light ratio</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The M_★/L_band ratio depends on population age, but the sensitivity varies dramatically by band. At short wavelengths (u, V), M/L is very age-sensitive: young starbursts are bright, so M/L is small; old populations are faint in the UV, so M/L grows rapidly (factor ~100 over 10 Gyr).">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_mass_to_light_band_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_mass_to_light_band_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar mass-to-light ratios across bands: age sensitivity</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Per-SSP-library solar metallicity differs: MIST Z☉ = 0.0142, BC03/Padova Z☉ = 0.0190, PARSEC Z☉ = 0.0152, BASTI Z☉ = 0.0200.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP Grid: Age and Metallicity Evolution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Stellar population synthesis templates differ across stellar evolution codes and isochrone libraries, producing measurable offsets in predicted spectra even at fixed age and metallicity. This gallery script loads four representative SSP libraries shipped with tengri (BC03, FSPS MILES, FSPS C3K, BPASS, ProGeny), constructs minimal SEDModel instances at age = 5 Gyr and Z = 0 (solar), and overlays rest-frame SED predictions (νL_ν) on log-log axes to reveal template-dependent uncertainties and continuum shape differences.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_library_shootout_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_library_shootout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP Library Shootout: Comparing Spectral Predictions at 5 Gyr, Z=0</div>
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

   /auto_examples/sps/plot_imf_choice_sweep
   /auto_examples/sps/plot_mass_to_light_band_comparison
   /auto_examples/sps/plot_ssp_age_sweep
   /auto_examples/sps/plot_ssp_grid
   /auto_examples/sps/plot_ssp_library_shootout
   /auto_examples/sps/plot_ssp_metallicity_sweep

