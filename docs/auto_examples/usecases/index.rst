

.. _sphx_glr_auto_examples_usecases:

Use Cases
=========

Paper-style figures and diagnostic plots — UVJ diagram, JWST color-color,
SFR-indicator comparison, mass completeness, age–dust degeneracy, and
emission-line Pearson coefficients.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates the famous age-dust-metallicity degeneracy in broadband photometry. Two galaxies with identical SDSS ugriz photometry — one old and dust-poor, one young and dust-rich — reveal dramatically different stellar ages, dust content, and metallicities. Adding GALEX FUV/NUV photometry breaks this degeneracy, illustrating why UV coverage is critical for accurate stellar population age dating.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_age_dust_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_age_dust_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-Dust-Metallicity Degeneracy with UV Break</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Computes a Pearson correlation matrix between five classical emission-line ratios ([OIII]/Hβ, Hα/Hβ, [NII]/Hα, [SII]/Hα, [SII]/[OIII]) across a grid of 1000 mock galaxies varying ionization parameter (log U ∈ [-4, -1]), gas metallicity (log Z/Zsun ∈ [-1, +0.2]), and age (1–5 Gyr). Shows which line ratios are independent diagnostics (low correlation) vs degenerate (high correlation), directly applicable to BPT-like classification schemes.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_emission_line_pcc_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_emission_line_pcc`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Emission Line Ratio Correlation Matrix for Nebular Diagnostics</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Generates 200 mock high-redshift galaxies spanning star-forming (z=1–7), quiescent/passive (z=1–3), and AGN/dusty-starburst (z=2–4) classes. Computes JWST NIRCam F150W–F277W vs F277W–F444W colors and plots the diagnostic diagram. Demonstrates how JWST color-color plots separate UV-to-IR spectral types for high-redshift source classification.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_jwst_color_color_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_jwst_color_color`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">JWST NIRCam Color-Color Diagram for High-z Classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates mass completeness limits for SDSS-like photometric surveys (ugriz depths comparable to SDSS). Mocks a population of 150 galaxies spanning log M* ∈ [7, 12] at z=0.1, injects realistic photometric noise, and measures the 95% completeness threshold: the stellar mass below which ≥5% of sources are undetected due to noise. Critical for survey design and sample construction.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_mass_completeness_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_mass_completeness`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Mass Completeness in Photometric Surveys</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compares four classical SFR indicators on the same set of 30 mock galaxies spanning burstiness amplitudes from σ=0.1 (smooth) to σ=3.0 (bursty). Indicators: UV continuum (1500 Å), Hα emission, FIR bolometric (8–1000 µm), and total bolometric SFR. Demonstrates how stochastic SFHs bias different indicators, with Hα showing highest variance and bolometric most stable.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sfr_indicator_compare_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sfr_indicator_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SFR Indicator Comparison: UV, Hα, FIR, Bolometric</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Generates a mock galaxy population (star-forming + passive) and plots each galaxy on the rest-frame UVJ color-color plane (U-V vs V-J). The UVJ diagram (Wuyts+2007, Williams+2009) is the workhorse diagnostic for separating quiescent galaxies from dusty star-forming galaxies, which are degenerate in single-color cuts.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uvj_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uvj_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UVJ Diagram: Star-Forming vs Passive Galaxy Population</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/usecases/plot_usecase_age_dust_degeneracy
   /auto_examples/usecases/plot_usecase_emission_line_pcc
   /auto_examples/usecases/plot_usecase_jwst_color_color
   /auto_examples/usecases/plot_usecase_mass_completeness
   /auto_examples/usecases/plot_usecase_sfr_indicator_compare
   /auto_examples/usecases/plot_usecase_uvj_diagram

