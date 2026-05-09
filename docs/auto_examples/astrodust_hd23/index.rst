

.. _sphx_glr_auto_examples_astrodust_hd23:

Hensley & Draine 2023 Astrodust+PAH
====================================

Reproductions of every figure from the upstream
``brandonshensley/Astrodust/notebooks/model_file_tutorial.ipynb`` notebook,
using tengri's ``DustEmissionSEDComponent`` (template ``"astrodust"``)
and the canonical FITS file at ``doi:10.7910/DVN/3B6E6S``.

Build the data with::

    python scripts/build_astrodust_hdf5.py --output data/astrodust_templates.h5 --download

Reference: Hensley & Draine 2023, ApJ 948, 55 (arXiv:2208.12365).



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Per-H grain volume distribution (4\pi/3)\,a^3\,dn/d\ln a / n_{\rm H} versus grain radius for the Hensley &amp; Draine 2023 fiducial size distribution (MW high-latitude R_V=3.1 sightline), reading from the HDU 1 metadata embedded in tengri&#x27;s HDF5.">

.. only:: html

  .. image:: /auto_examples/astrodust_hd23/images/thumb/sphx_glr_plot_01_size_distribution_thumb.png
    :alt:

  :doc:`/auto_examples/astrodust_hd23/plot_01_size_distribution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH size distribution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Plots \lambda I_\lambda / N_{\rm H} / U for several \log_{10} U values from the Hensley &amp; Draine 2023 grid. Dividing by U makes the U-dependence of the PAH-vs-FIR ratio visible: low-U curves stack atop each other in the FIR while the MIR features rise steeply with U.">

.. only:: html

  .. image:: /auto_examples/astrodust_hd23/images/thumb/sphx_glr_plot_02_emission_vs_lgU_thumb.png
    :alt:

  :doc:`/auto_examples/astrodust_hd23/plot_02_emission_vs_lgU`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH emission vs log U</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Astrodust vs PAH components at the fiducial U=1.6 (lgU=0.2).">

.. only:: html

  .. image:: /auto_examples/astrodust_hd23/images/thumb/sphx_glr_plot_03_components_at_fiducial_U_thumb.png
    :alt:

  :doc:`/auto_examples/astrodust_hd23/plot_03_components_at_fiducial_U`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH per-component decomposition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Switch between dust IR templates with one config-field change.">

.. only:: html

  .. image:: /auto_examples/astrodust_hd23/images/thumb/sphx_glr_plot_04_sedmodel_dust_emission_swap_thumb.png
    :alt:

  :doc:`/auto_examples/astrodust_hd23/plot_04_sedmodel_dust_emission_swap`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DustEmissionSEDComponent — swap MBB / PAHspec / Astrodust</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="f_ion(a) and f_align(a) versus grain size — H&amp;D 2023 fiducials.">

.. only:: html

  .. image:: /auto_examples/astrodust_hd23/images/thumb/sphx_glr_plot_05_ionization_alignment_thumb.png
    :alt:

  :doc:`/auto_examples/astrodust_hd23/plot_05_ionization_alignment`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH ionization fraction and alignment</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Extinction, polarized extinction, and albedo — H&amp;D 2023 fiducial.">

.. only:: html

  .. image:: /auto_examples/astrodust_hd23/images/thumb/sphx_glr_plot_06_extinction_and_scattering_thumb.png
    :alt:

  :doc:`/auto_examples/astrodust_hd23/plot_06_extinction_and_scattering`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH extinction, scattering, and albedo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spinning dust microwave emission — H&amp;D 2023 fiducial.">

.. only:: html

  .. image:: /auto_examples/astrodust_hd23/images/thumb/sphx_glr_plot_07_spinning_dust_thumb.png
    :alt:

  :doc:`/auto_examples/astrodust_hd23/plot_07_spinning_dust`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH spinning-dust microwave emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Polarized emission and polarization fraction — H&amp;D 2023.">

.. only:: html

  .. image:: /auto_examples/astrodust_hd23/images/thumb/sphx_glr_plot_08_polarized_emission_thumb.png
    :alt:

  :doc:`/auto_examples/astrodust_hd23/plot_08_polarized_emission`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH polarized emission and polarization fraction</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/astrodust_hd23/plot_01_size_distribution
   /auto_examples/astrodust_hd23/plot_02_emission_vs_lgU
   /auto_examples/astrodust_hd23/plot_03_components_at_fiducial_U
   /auto_examples/astrodust_hd23/plot_04_sedmodel_dust_emission_swap
   /auto_examples/astrodust_hd23/plot_05_ionization_alignment
   /auto_examples/astrodust_hd23/plot_06_extinction_and_scattering
   /auto_examples/astrodust_hd23/plot_07_spinning_dust
   /auto_examples/astrodust_hd23/plot_08_polarized_emission

