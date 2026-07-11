:orphan:

.. _sphx_glr_auto_examples_showcase:

Showcase
========

Headline demonstrations that exercise the full stack: population-scale
forward modeling, JAX gradient diagnostics, and end-to-end recipe tours.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Fisher Information Matrix quantifies which linear combinations of parameters are constrained by data — and which are degenerate. Tengri&#x27;s fully differentiable forward model makes it trivial to compute the Fisher matrix at any point in parameter space.">

.. only:: html

  .. image:: /auto_examples/showcase/images/thumb/sphx_glr_plot_gradient_degeneracy_direction_thumb.png
    :alt:

  :doc:`/auto_examples/showcase/plot_gradient_degeneracy_direction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fisher Information Ellipses from the Hessian</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Every photometric band&#x27;s flux is a differentiable function of model parameters. This example computes sensitivities ∂(log F) / ∂(log θ) — the logarithmic elasticity of each photometric band to changes in four key stellar population parameters: peak star formation rate, metallicity, dust optical depth, and age. Astronomers fitting galaxy SEDs in other codes use finite differences (∂F/∂θ ≈ [F(θ+δ) − F(θ−δ)] / (2δ), slow and numerically fragile); tengri exposes JAX&#x27;s autodiff to compute these sensitivities exactly in one forward and one reverse pass per parameter. This heatmap demonstrates the approach.">

.. only:: html

  .. image:: /auto_examples/showcase/images/thumb/sphx_glr_plot_jax_gradient_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/showcase/plot_jax_gradient_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Automatic differentiation: parameter sensitivities via jax.grad</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each recipe is a nested-dict configuration — drop-in templates for common galaxy fitting scenarios. This gallery overlays the rest-frame SED of all five shipped recipes, highlighting how model complexity scales from minimal mock-recovery to panchromatic AGN:">

.. only:: html

  .. image:: /auto_examples/showcase/images/thumb/sphx_glr_plot_recipes_gallery_thumb.png
    :alt:

  :doc:`/auto_examples/showcase/plot_recipes_gallery`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">recipes for common science cases</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/showcase/plot_gradient_degeneracy_direction
   /auto_examples/showcase/plot_jax_gradient_sensitivity
   /auto_examples/showcase/plot_recipes_gallery

