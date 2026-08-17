:orphan:

.. _sphx_glr_auto_examples_sps:

Stellar Population Synthesis
============================

DSPS-based SSP grids: age, metallicity, and spectral properties.

- ``plot_ssp_grid.py`` — SSP grid visualization (age, metallicity, spectrum)


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Per-SSP-library solar metallicity differs: MIST Z☉ = 0.0142, BC03/Padova Z☉ = 0.0190, PARSEC Z☉ = 0.0152, BASTI Z☉ = 0.0200. A given logzsol is only meaningful against its library.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_age_metallicity_color_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_age_metallicity_color_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-metallicity color degeneracy in SDSS colors</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The stellar populations in massive elliptical galaxies are typically α-enhanced ([α/Fe] &gt; 0) due to rapid star formation timescales that terminate before iron-peak elements fully enrich the gas. Increasing [α/Fe] shifts absorption-feature strengths — particularly Mg b and Fe5270 — which serve as diagnostics of the galaxy&#x27;s star-formation history timescale. We sweep [α/Fe] from 0.0 to 0.6 at fixed age (5 Gyr) and solar metallicity, showing the full rest-frame SED and a zoom on the optical feature region.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_alpha_enhanced_population_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_alpha_enhanced_population`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Alpha-element enhancement shifts absorption features in old stellar populations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The bolometric correction in band X is BC_X = M_bol − M_X (equivalently 2.5 log10(L_X / L_bol) up to a sign). For a single-burst SSP it traces which part of the spectrum carries the bolometric luminosity at each age: at young ages the UV dominates so BC_UV is small and BC_K is large (positive); as the population ages the SED reddens and the correction inverts — BC_K shrinks while BC_UV blows up.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_bolometric_correction_vs_age_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_bolometric_correction_vs_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bolometric correction per band as a single burst ages</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Energy is conserved: dust attenuation removes UV/optical flux, which is re-radiated in the far-infrared (Dale 2014 templates restore the balance). IGM absorption (Inoue 2014) sculpts the rest-frame continuum below the Lyman break (912 Å).">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_component_buildup_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_component_buildup`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Building the panchromatic SED component by component</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The same stellar population SED looks different depending on the units chosen for visualization. a single galaxy SED in three complementary representations on a 3-panel grid:">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_fnu_vs_flambda_units_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_fnu_vs_flambda_units`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SED Conventions: F_λ vs F_ν vs νF_ν</div>
    </div>


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

    <div class="sphx-glr-thumbcontainer" tooltip="The hydrogen-ionizing photon production rate Q_H of a simple stellar population drops by ~5 dex from 1 Myr to 100 Myr as O stars die. Different SSP libraries predict different Q_H(t) because they differ in upper-IMF treatment, stellar rotation, and (most dramatically) whether massive binaries are included — BPASS extends the Q_H-producing phase to ~30 Myr.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ionizing_lum_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ionizing_lum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionizing-photon production rate vs SSP age</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Per-SSP-library solar metallicity differs: MIST Z☉ = 0.0142, BC03/Padova Z☉ = 0.0190, PARSEC Z☉ = 0.0152, BASTI Z☉ = 0.0200.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP Grid: Age and Metallicity Evolution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Different Initial Mass Functions produce different continuum shapes at fixed age and metallicity. Salpeter (top-heavy) produces harder UV and near-IR continua. We compare the rest-frame νL_ν at 1 Gyr solar metallicity, peak-normalized at 5500 Å to reveal chromatic differences. The NIR is most diagnostic of IMF choice because massive stars dominate the red-giant branch.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_imf_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_imf_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">IMF choice revealed in SED continuum shape: Chabrier vs Kroupa vs Salpeter</div>
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


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust-free UV continuum slope of an SSP swept from 10 Myr to 1 Gyr. β is fit in the Calzetti+1994 windows (1268–2580 Å) to F_λ ∝ λ^β.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_uv_slope_age_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_uv_slope_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Intrinsic UV continuum slope β vs single-burst age</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/sps/plot_age_metallicity_color_grid
   /auto_examples/sps/plot_alpha_enhanced_population
   /auto_examples/sps/plot_bolometric_correction_vs_age
   /auto_examples/sps/plot_component_buildup
   /auto_examples/sps/plot_fnu_vs_flambda_units
   /auto_examples/sps/plot_imf_choice_sweep
   /auto_examples/sps/plot_ionizing_lum
   /auto_examples/sps/plot_mass_to_light_band_comparison
   /auto_examples/sps/plot_sps_library_compare
   /auto_examples/sps/plot_ssp_age_sweep
   /auto_examples/sps/plot_ssp_color_compare
   /auto_examples/sps/plot_ssp_grid
   /auto_examples/sps/plot_ssp_imf_compare
   /auto_examples/sps/plot_ssp_library_shootout
   /auto_examples/sps/plot_ssp_metallicity_sweep
   /auto_examples/sps/plot_uv_slope_age

