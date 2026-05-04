

.. _sphx_glr_auto_examples_dust:

Dust Models
===========

Attenuation laws, two-component dust, and IR emission.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The characteristic temperature T of the modified blackbody dust emission controls the peak wavelength of far-infrared emission. Wien&#x27;s law: λ_peak ≈ 2900 μm·K / T. Hotter dust (higher T) peaks at shorter wavelengths (more mid-IR), cooler dust peaks further into the far-IR/submm.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_dust_T_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_dust_T_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Modified Blackbody Dust Temperature</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Plot all available attenuation laws in tengri. Each curve k(lambda) describes the wavelength dependence of dust attenuation, normalized at 5500 A. No SSP data required.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_dust_curves_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_dust_curves`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust Attenuation Curves</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All tengri dust emission models evaluated at L_absorbed = 1e10 L☉ and T = 35 K. Template-based models (DL07, DL14, Dale+2014) are shown only when data files are present; they are skipped gracefully otherwise.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_dust_emission_models_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_dust_emission_models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust Emission Models: Overview</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The power-law slope δ modifies the Calzetti attenuation curve shape. Negative δ steepens UV attenuation; positive δ flattens it. This controls whether dust absorbs more or less light at short wavelengths relative to optical.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_dust_slope_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_dust_slope_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Attenuation Curve Slope (δ)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The PAH mass fraction q_PAH controls the strength of polycyclic aromatic hydrocarbon (PAH) mid-infrared emission features at 3.3, 6.2, 7.7, 8.6, and 11.3 μm. Higher q_PAH → stronger PAH features. Range: 0.47–4.58 % for DL07, 0.47–7.32 % for DL14.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_qpah_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_qpah_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PAH Mass Fraction (q_PAH)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does birth cloud dust optical depth reshape the SED of a young star-forming galaxy? Higher τ_BC reddens the UV and suppresses nebular emission from very young stars embedded in dust cocoons.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_tau_bc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_tau_bc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Birth Cloud Optical Depth (τ_BC)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The diffuse ISM attenuation affects all stellar light (not just young stars). Higher τ_diff reddens the optical continuum and weakens the 4000 Å break, a signature of aging stellar populations.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_tau_diff_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_tau_diff_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diffuse ISM Optical Depth (τ_diff)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Visualize the Charlot &amp; Fall (2000) two-component dust model: birth cloud attenuation affects only young stars (age &lt; ~10 Myr), while diffuse ISM attenuation affects all stars. The smooth sigmoid transition between components is shown as a function of stellar age.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_two_component_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_two_component`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Two-Component Dust SEDModel</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="U_min sets the intensity of the diffuse radiation field heating the dust. Higher U_min implies a hotter, more intense radiation field → hotter overall dust → shifted far-infrared peak to shorter wavelengths. Range: 0.1–25 for DL07, 0.1–50 for DL14.">

.. only:: html

  .. image:: /auto_examples/dust/images/thumb/sphx_glr_plot_umin_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust/plot_umin_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine & Li Minimum Radiation Field (U_min)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/dust/plot_dust_T_sweep
   /auto_examples/dust/plot_dust_curves
   /auto_examples/dust/plot_dust_emission_models
   /auto_examples/dust/plot_dust_slope_sweep
   /auto_examples/dust/plot_qpah_sweep
   /auto_examples/dust/plot_tau_bc_sweep
   /auto_examples/dust/plot_tau_diff_sweep
   /auto_examples/dust/plot_two_component
   /auto_examples/dust/plot_umin_sweep

