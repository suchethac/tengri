

.. _sphx_glr_auto_examples_dust_attenuation:

Dust Attenuation
================

How starlight is extincted on its way out of the galaxy — Calzetti vs
power-law slopes, the 2175 Å UV bump, birth-cloud and diffuse-ISM optical
depths, two-component geometry, and law comparisons.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All major attenuation laws implemented in tengri evaluated at fixed τ_V = 1.0. Shows wavelength dependence (k(λ)) from UV through near-infrared, highlighting the UV bump (2175 Å) and the steepness differences between Milky Way, SMC, and starburst models.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_attenuation_law_compare_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_attenuation_law_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Attenuation Law Comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Plot all available attenuation laws in tengri. Each curve k(lambda) describes the wavelength dependence of dust attenuation, normalized at 5500 A. No SSP data required.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_curves_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_curves`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust Attenuation Curves</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare transmission curves for three dust geometries (Witt &amp; Gordon 2000): screen (foreground), mixed (slab), and clumpy (two-phase medium). At fixed optical depth τ_V = 1.0, geometry controls the spectral shape: screens are reddest, mixed intermediate, clumpy greyest.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_geometry_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_geometry_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust Geometry: Screen vs Mixed vs Clumpy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The power-law slope δ modifies the Calzetti attenuation curve shape. Negative δ steepens UV attenuation; positive δ flattens it. This controls whether dust absorbs more or less light at short wavelengths relative to optical.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_slope_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_slope_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Attenuation Curve Slope (δ)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Birth-cloud dust optical depth τ_BC controls how much of the youngest stellar light escapes the cocoon. Higher τ_BC reddens the UV and suppresses nebular emission from embedded HII regions.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_tau_bc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_tau_bc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Birth Cloud Optical Depth (τ_BC)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The diffuse ISM attenuation affects all stellar light (not just young stars). Higher τ_diff reddens the optical continuum and weakens the 4000 Å break, a signature of aging stellar populations.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_tau_diff_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_tau_diff_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diffuse ISM Optical Depth (τ_diff)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Visualize the Charlot &amp; Fall (2000) two-component dust model: birth cloud attenuation affects only young stars (age &lt; ~10 Myr), while diffuse ISM attenuation affects all stars. The smooth sigmoid transition between components is shown as a function of stellar age.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_two_component_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_two_component`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Two-Component Dust SEDModel</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 2175 Å UV bump in dust attenuation curves is usually attributed to PAHs or small graphite grains. Sweeping the bump amplitude from zero to MW-like takes the attenuation curve from a smooth power law to the characteristic MW shape.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_uv_bump_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_uv_bump_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV Bump Strength (dust_bump_strength)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/dust_attenuation/plot_attenuation_law_compare
   /auto_examples/dust_attenuation/plot_dust_curves
   /auto_examples/dust_attenuation/plot_dust_geometry_sweep
   /auto_examples/dust_attenuation/plot_dust_slope_sweep
   /auto_examples/dust_attenuation/plot_tau_bc_sweep
   /auto_examples/dust_attenuation/plot_tau_diff_sweep
   /auto_examples/dust_attenuation/plot_two_component
   /auto_examples/dust_attenuation/plot_uv_bump_sweep

