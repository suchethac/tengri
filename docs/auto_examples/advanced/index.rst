:orphan:

.. _sphx_glr_auto_examples_advanced:

Advanced Topics
===============

Hierarchical inference, gradient sensitivity, batch fitting, panchromatic SED
with radio and X-ray components, and joint photometry + spectroscopy fitting.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The on-ramp for adding a custom physics block to tengri. Subclass SEDModelComponent, declare name, parameter_prefix, priors as class attributes, and implement predict(p, sed_in, wave). __init_subclass__ registers the new variant and auto-fills the inputs() / outputs() contracts.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_custom_attenuation_component_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_custom_attenuation_component`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Authoring a new physics block with SEDModelComponent</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cramér-Rao bound from the Fisher Information Matrix shows that SDSS 5-band photometry alone cannot separately constrain age, dust, and metallicity. Adding NIR or MIR bands breaks the degeneracy by factors of 2–5×, quantifying the information gain from multiwavelength coverage.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_fisher_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_fisher_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-Dust-Metallicity Degeneracy: Fisher Analysis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Computes the Jacobian d(flux)/d(theta) of the forward model and displays it as a heatmap showing which photometric bands are sensitive to which physical parameters. Each column shows normalized sensitivity to one parameter; dark blue/red indicates strong dependence.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_gradient_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_gradient_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient Sensitivity Heatmap</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Every import tengri enables an on-disk JAX compile cache at ~/.cache/tengri_jax_cache (override with TENGRI_JAX_CACHE_DIR, opt out with TENGRI_DISABLE_JAX_CACHE=1). The cache survives Python restarts: notebook re-runs, slurm tasks, and benchmark workers all skip the expensive first compile.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_jit_cache_speedup_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_jit_cache_speedup`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The persistent JIT cache turns the second-run cold start into a no-op</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates tengri&#x27;s Observation API for joint fitting across two data streams. Creates a mock galaxy with SDSS photometry and low-resolution spectroscopy, then recovers parameters via MAP. Shows how spectroscopy breaks photometric degeneracies.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_joint_fit_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_joint_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Joint Photometry + Spectroscopy Fit</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="TODO[examples-sweep]: This script uses low-level component orchestration (build_components, run_components) which is experimental Phase II-2.6 API intended for infrastructure use, not recommended for user-facing examples.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_orchestrator_demo_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_orchestrator_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Component Orchestrator End-to-End</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a full galaxy SED spanning X-ray to radio wavelengths. Shows stellar emission, dust attenuation, dust IR emission, radio synchrotron, and X-ray binary contributions. Demonstrates tengri&#x27;s multiwavelength physics modules for radio and X-ray—no SSP data required for these components.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_radio_xray_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_radio_xray`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Panchromatic SED: UV to Radio</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The build-time switch approx=WavePrecomp(...) publishes the SSP × filter look-up table at construction time, then routes predict_photometry through a tabulated path. The exact wave-grid path recomputes the rest-frame SED on a ~3000-point grid and integrates against each filter response on every call; the LUT path reduces the per-call work to filter-count-sized array ops on a pre-cached grid.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_waveprecomp_scaling_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_waveprecomp_scaling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">WavePrecomp turns photometry into a near-constant-cost call</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/advanced/plot_custom_attenuation_component
   /auto_examples/advanced/plot_fisher_degeneracy
   /auto_examples/advanced/plot_gradient_sensitivity
   /auto_examples/advanced/plot_jit_cache_speedup
   /auto_examples/advanced/plot_joint_fit
   /auto_examples/advanced/plot_orchestrator_demo
   /auto_examples/advanced/plot_radio_xray
   /auto_examples/advanced/plot_waveprecomp_scaling

