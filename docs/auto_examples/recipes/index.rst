

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

    <div class="sphx-glr-thumbcontainer" tooltip="The astronomer&#x27;s-eye-view of the tengri ingest path. Starting from a single CSV row of SDSS ugriz fluxes and per-band errors (the same shape pandas would hand you from a survey catalogue), we parse the row, build the photometric Observation from the column names, fit with MAP, and overlay the recovered SED on the observed bands with normalised residuals.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_real_data_fit_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_real_data_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">From a CSV row to a MAP SED fit, end to end</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="tengri.recipes ships several curated starting-point model configs that map common astronomer use-cases onto the nested-dict SEDModel.build grammar. This card overlays the rest-frame SED of every shipped recipe so users can pick by eye:">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_compare_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">What each shipped tengri recipe produces</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Optical broadband photometry constrains metallicity weakly, so the prior carries real information. We mock a star-forming galaxy at log Z/Zsun = -0.5 in five SDSS bands at S/N=20, then fit it twice under the same model — once with a uniform Z prior, once with a Gaussian prior centred on 0 with sigma=0.3. The posteriors shift by ~0.3 dex toward each prior&#x27;s preferred region, illustrating how informative external priors propagate through a tengri inference.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_compare_priors_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_compare_priors`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Metallicity prior choice moves the photometric posterior by ~0.3 dex</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do I combine a custom photometric filter with standard filters? This recipe generates a synthetic Gaussian filter at 2 μm and pairs it with SDSS optical bands, then predicts the full SED and photometry.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_custom_filter_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_custom_filter`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Register and use a custom photometric filter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do I load measured photometry from a table and fit it? This recipe generates mock photometry for 3 galaxies and fits each one independently with a MAP fit, demonstrating the workflow for catalogue-scale SED fitting.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_load_real_csv_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_load_real_csv`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Load and fit photometry from CSV</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do I persist a posterior between sessions? This recipe runs a MAP fit, saves the result to HDF5, reloads it, and demonstrates basic analysis. Posterior objects can be checkpointed for long-running fits or multi-stage analysis pipelines.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_save_load_posterior_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_save_load_posterior`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Save and load a posterior to disk</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When redshift is known from spectroscopy, the SED fit is more precise than when inferring redshift from photometry alone. This recipe generates mock photometry at a known redshift, then fits it with redshift fixed (spectroscopic) and redshift free (photometric only), showing how redshift degeneracies affect parameter recovery.">

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

   /auto_examples/recipes/plot_real_data_fit
   /auto_examples/recipes/plot_recipe_compare
   /auto_examples/recipes/plot_recipe_compare_priors
   /auto_examples/recipes/plot_recipe_custom_filter
   /auto_examples/recipes/plot_recipe_load_real_csv
   /auto_examples/recipes/plot_recipe_save_load_posterior
   /auto_examples/recipes/plot_recipe_specific_redshift

