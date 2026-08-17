:orphan:

.. _sphx_glr_auto_examples_recipes:

Recipes
=======

Common workflows: prior comparison, photometry I/O, redshift fixing, filter set swapping, posterior persistence.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six curated recipes span galaxy populations: star-forming at 0–6 (bare-stellar SSP), quiescent at z ≈ 0.05 (bare-stellar, τ_diff-free to trace dust), AGN panchromatic (bare-stellar, full AGN composite with disc+torus+radio+xray), stochastic JWST high-z with burstiness (bare-stellar, DPL+field at 0.5–12), mock-recovery minimal (any SSP, 4–5 free params for benchmarking), and dust-demo (wNE only — baked nebular emission visualized). All use WavePrecomp() except photoz (ztable does not cover z &gt; 12). Use load_ssp(&quot;*.wNE&quot;) only for dust_demo; others silently under-predict if fed wNE.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_compare_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">What each shipped tengri recipe produces</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a FilterCurve from a Gaussian transmission profile and combine it with standard filters. The Photometry object merges them, then SEDModel predicts photometry on all bands at once — custom filters compose naturally with the standard library.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_custom_filter_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_custom_filter`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Register and use a custom photometric filter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Call tengri.list_recipes() to see the shipped menu with SSP requirements (bare-stellar, wNE, or any) and tengri.describe_recipe(name) to fetch a recipe&#x27;s docstring. Three models showcase the morphological diversity: star-forming (DPL+Cue nebular, free z to 6), quiescent at z=0.05 (dexp, lower dust ceiling), and AGN-panchromatic (full composite, z to 6). All require bare-stellar SSP (Cue backend).">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_introspection_tour_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_introspection_tour`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recipe introspection and SED morphology comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Mock 3 galaxies, fit each independently with MAP. The workflow is: sample true parameters → generate mock fluxes + noise → fit with free SFH/dust and fixed redshift. Demonstrates vectorizing catalog-scale fits when redshift is already known (e.g., spectroscopy).">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_load_real_csv_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_load_real_csv`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Load and fit photometry from CSV</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="MAP-fit a model, serialize the Posterior to HDF5 with .save(), reload in a new session with Posterior.load(), and recover the fit parameters and diagnostics. Enables checkpoint-driven analysis pipelines and multi-stage fits.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_save_load_posterior_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_save_load_posterior`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Save and load a posterior to disk</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two fits on the same mock data: one with redshift fixed (spectroscopic known, free SFH/dust/met), one with redshift free (photometric only). The fixed-z fit converges to truth; free-z is degenerate with dust and SFH, showing why spectroscopy breaks the age-dust-redshift degeneracies that plague photometry-only fitting.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_specific_redshift_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_specific_redshift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Redshift constraint: spectroscopy vs photometry alone</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/recipes/plot_recipe_compare
   /auto_examples/recipes/plot_recipe_custom_filter
   /auto_examples/recipes/plot_recipe_introspection_tour
   /auto_examples/recipes/plot_recipe_load_real_csv
   /auto_examples/recipes/plot_recipe_save_load_posterior
   /auto_examples/recipes/plot_recipe_specific_redshift

