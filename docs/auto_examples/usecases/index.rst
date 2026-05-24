

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

    <div class="sphx-glr-thumbcontainer" tooltip="A 5 Gyr stellar population with no dust is nearly indistinguishable from a 1 Gyr population reddened by τ_diff = 0.4 when observed in optical broadband colors alone. This is the central degeneracy that limits SED-fitting accuracy from optical-only photometry, and the reason FUV/NUV (sensitive to recent star formation) or rest-frame IR (sensitive to dust mass) bands break the ambiguity.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_age_dust_2d_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_age_dust_2d`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The age–dust degeneracy on the optical g − r color</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two synthetic galaxies with identical SDSS ugriz photometry — one old and dust-poor, one young and dust-rich — produce wildly different SED fits. Adding GALEX FUV/NUV observation breaks the degeneracy by constraining the UV slope. Demonstrates the critical importance of short-wavelength coverage for stellar age and dust determination.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_age_dust_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_age_dust_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-dust-metallicity degeneracy: why UV photometry is critical</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 4000 Å break D_n(4000) — Bruzual 1983, Balogh+1999 — measures the discontinuity around 4000 Å produced by the line-blanketing of ionised metals in the atmospheres of old stars. It rises monotonically with the mass-weighted age of the stellar population and is one of the most widely used age indicators in SDSS-style optical-only data (Kauffmann+2003).">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_d4000_age_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_d4000_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The 4000 Å break as a stellar age proxy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates emission-line diagnostics using [OIII]/Hbeta, [NII]/Halpha ratios across a population mock sample. At different redshifts, nebular lines shift into different broadband filters creating photometric signatures useful for photo-z and ionization state estimation.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_emission_line_pcc_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_emission_line_pcc`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Emission-line pseudo-color-color diagram for redshift classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Coleman, Wu &amp; Weedman 1980 spectral templates remain the textbook illustration of how the integrated SED morphs along the Hubble sequence — from quiescent ellipticals with deep 4000 Å breaks to gas-rich irregulars dominated by ongoing star formation and nebular emission.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_hubble_sequence_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_hubble_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A morphological atlas: E, Sa, Sb, Sc, Im galaxy SEDs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Generates 150 mock galaxies spanning star-forming (z=1-7), passive (z=1-3), and dusty/AGN (z=2-4) populations. Computes JWST NIRCam F150W-F277W vs F277W-F444W colors and plots the diagnostic plane. Shows how JWST color-color diagnostics separate spectral types and enable redshift estimation in the rest-frame UV-to-IR with minimal prior knowledge.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_jwst_color_color_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_jwst_color_color`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">JWST NIRCam color-color diagnostics for high-z galaxy classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three of the most-used star-formation-rate indicators agree only for specific assumed SFHs. We mock a constant-SFR galaxy across SFR = 0.01 to 100 M☉/yr and read each indicator out:">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_kennicutt_sfr_calibrations_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_kennicutt_sfr_calibrations`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kennicutt+1998 SFR calibrations: UV, Hα, and L_IR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Measures the 95% stellar mass completeness threshold for SDSS-like photometry. Mocks a population of 150 star-forming and passive galaxies spanning log M* [7-12] at z=0.1, injects realistic photometric noise, and measures below which stellar mass more than 5% of sources drop below detection limit. Critical for constructing mass-limited galaxy samples and understanding survey selection effects.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_mass_completeness_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_mass_completeness`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mass completeness limit in SDSS-like photometric surveys</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compares four classical SFR indicators (UV continuum, Hα emission, FIR, bolometric) on a population of mock galaxies spanning burstiness amplitudes. Stochastic SFHs introduce variance that differs between indicators. Hα shows highest scatter while bolometric is most stable — a key consideration for survey design.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sfr_indicator_compare_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sfr_indicator_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SFR indicators: comparing UV, Hα, FIR under stochastic star formation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The infrared excess (IRX = L_IR / L_FUV) versus UV-continuum slope β diagram (Meurer+1999) is the standard tool for inferring attenuation in unresolved star-forming galaxies. We mock a population of star-forming galaxies with a fixed SFH and a range of diffuse dust optical depths, measure each galaxy&#x27;s β by fitting a power-law to its rest-frame UV continuum (1268–2580 Å, Calzetti+1994 windows), and overplot the empirical Meurer+1999 starburst relation.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uv_slope_beta_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uv_slope_beta`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The IRX–β relation emerges from the dust model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Generates a mock star-forming and quiescent galaxy population and plots each on the rest-frame UVJ color-color plane (U-V vs V-J). The Williams+2009 quiescent wedge (z &lt; 1) marks the boundary between dusty star-forming and passive galaxies — a key degeneracy-breaking diagnostic.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uvj_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uvj_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UVJ diagram: rest-frame colors separate star-forming from quiescent</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/usecases/plot_usecase_age_dust_2d
   /auto_examples/usecases/plot_usecase_age_dust_degeneracy
   /auto_examples/usecases/plot_usecase_d4000_age
   /auto_examples/usecases/plot_usecase_emission_line_pcc
   /auto_examples/usecases/plot_usecase_hubble_sequence
   /auto_examples/usecases/plot_usecase_jwst_color_color
   /auto_examples/usecases/plot_usecase_kennicutt_sfr_calibrations
   /auto_examples/usecases/plot_usecase_mass_completeness
   /auto_examples/usecases/plot_usecase_sfr_indicator_compare
   /auto_examples/usecases/plot_usecase_uv_slope_beta
   /auto_examples/usecases/plot_usecase_uvj_diagram

