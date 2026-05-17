

.. _sphx_glr_auto_examples_recipes:

Recipes
=======

Short, focused snippets for common how-to questions — comparing priors,
loading photometry from CSV, fixing redshift, swapping filter sets, and
saving/loading a posterior to disk.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does prior choice affect the posterior? This recipe compares fitting with a Uniform prior vs Gaussian prior on metallicity, showing how prior assumptions constrain the posterior.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_compare_priors_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_compare_priors`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Prior Sensitivity: Gaussian vs Uniform</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do I register a custom photometric filter and use it in SED modeling? This recipe generates a synthetic filter response and uses it to compute photometry through a model SED.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_custom_filter_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_custom_filter`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Register and Use Custom Filters</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do I load photometric data from a CSV file and fit it? This recipe demonstrates loading a table of measured fluxes and uncertainties, building observations per galaxy, and running a MAP fit on each.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_load_real_csv_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_load_real_csv`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Load and Fit Real CSV Photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do I save a posterior to disk and load it later? This recipe demonstrates running a NUTS fit, saving the Posterior to an HDF5 file, reloading it, and analyzing the saved results.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_save_load_posterior_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_save_load_posterior`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Save and Load a Posterior</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do I fit a spectrum when redshift is known from spectroscopy? This recipe shows how fixing redshift with Fixed() constrains other parameters more tightly compared to letting it vary.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_specific_redshift_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_specific_redshift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fix Redshift to a Known Value</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/recipes/plot_recipe_compare_priors
   /auto_examples/recipes/plot_recipe_custom_filter
   /auto_examples/recipes/plot_recipe_load_real_csv
   /auto_examples/recipes/plot_recipe_save_load_posterior
   /auto_examples/recipes/plot_recipe_specific_redshift

