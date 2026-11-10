# SPDX-License-Identifier: BSD-3-Clause
"""Band tables of the additive emitters: the rank-1 term response and the shape tables.

An additive emitter (radio, X-ray, the composable AGN) is a sum of rank-1 terms. Its band
response is built once per model from the filter curves, and read back through the band-table
memo of :class:`~tengri.forward.sed_model.SEDModel`. The adapters name these slots in
``BAND_SLOTS`` and call :func:`term_band_table` for them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri.forward.precompute.band_table import BandTable, Decline

#: Relative tolerance for the rank-1 check in
#: :func:`additive_term_band_response`. Two probe draws must reproduce
#: each term's spectral shape to this precision for the term to earn a constant
#: band response. Loose enough not to trip on fp re-association across a ~6000-node
#: grid, and far tighter than any shape error worth exploiting.
_RANK1_RTOL = 1e-10


def _term_response_refusal(comp, chain, free, fixed) -> str | None:
    """Why the term band response must stay dense for emitter ``comp``, or ``None``.

    An emitter that declares ``term_response_refusal`` (the AGN) decides for itself
    whether its shape is fixed; the galaxy dust screen on an AGN is checked here because
    it lives on the dust component. Other emitters (radio, X-ray) are refused when any of
    their own parameters is free, other than the amplitudes listed in ``AMPLITUDE_PARAMS``.

    Parameters
    ----------
    comp : SEDComponent
        The additive emitter.
    chain : sequence
        The component chain, for the dust-side ``agn_screen`` check.
    free : set of str
        Free parameter names of the model.
    fixed : mapping
        Build-time fixed values, keyed by parameter name.

    Returns
    -------
    str or None
        The reason, or ``None`` when the emitter passes the gate.
    """
    hook = getattr(comp, "term_response_refusal", None)
    if hook is not None:
        reason = hook(free, fixed)
        if reason is None and any(
            getattr(getattr(c, "config", None), "agn_screen", "none") != "none" for c in chain
        ):
            return "the galaxy dust screens the AGN (agn_screen): its photometry needs the dense"
            " SED"
        return reason
    amplitudes = getattr(comp, "AMPLITUDE_PARAMS", frozenset())
    prefix = f"{comp.name}_"
    free_emitter = sorted(p for p in free if p.startswith(prefix) and p not in amplitudes)
    if free_emitter:
        return f"free {comp.name} parameters: {free_emitter}"
    return None


def additive_term_band_response(ctx, name):
    r"""Build-time per-filter response of each rank-1 term of an additive emitter.

    Generalizes the dust band response from one term to
    many. An
    additive emitter is a *sum of rank-1 terms*, each a scalar amplitude times a
    spectral shape that depends only on the emitter's own (fixed) shape parameters:

    .. math::

        L_\nu(\lambda) = \sum_k A_k(\text{inputs}) \, S_k(\lambda; \text{shape})

    The filter integral is linear, so each term's band flux factorizes and

    .. math::

        \int \Big[\sum_k A_k S_k(\lambda)\Big] R_f(\lambda)\, d\lambda
            = \sum_k A_k \underbrace{\int S_k(\lambda) R_f(\lambda)\, d\lambda}_{R_{kf}}

    with :math:`R_{kf}` a build-time constant. This is **exact**, not an
    approximation: the true filter transmission is still integrated on the full
    wavelength grid with the identical quadrature the dense path uses
    (:func:`~tengri.observation.photometry.lnu_filter_integral_batch`), just once
    instead of on every call. Contrast ``fast_emission``, which samples the emitter
    at one effective wavelength, and the stellar WavePrecomp Taylor projection (#617).

    Dust IR happens to be the :math:`k=1` case (``sed = L_ir * S_unit``). X-ray is
    :math:`k=4` (HMXB, LMXB, hot gas, corona) and radio :math:`k=3` (SF synchrotron,
    free-free, AGN jet). Their *summed* SEDs are **not** rank-1, HMXB and LMXB carry
    different photon indices, so the HMXB/LMXB mix shifts with SFR and stellar mass,
    which is why the terms must be integrated separately rather than as a total.

    At runtime the amplitudes come back from a single-wavelength evaluation,
    :math:`A_k = \text{term}_k(\lambda^{\rm ref}_k) / S_k(\lambda^{\rm ref}_k)`. That
    needs no knowledge of *which* input carries the amplitude, nor that it enters
    linearly, :math:`\alpha_{\rm ox}` is famously nonlinear in :math:`L_{2500}`, and
    it does not matter, because it is still a scalar.

    Parameters
    ----------
    ctx : BandTableContext
        The model state; ``ctx.chain`` is the cached component chain.
    name : str
        Component name to look for (``"xray"``, ``"radio"``).

    Returns
    -------
    BandTable or Decline
        ``{"ln1pz": (n_z,), "R": (n_z, n_terms, n_filters), "S_ref": (n_z, n_terms),
        "lam_ref": (n_terms,)}`` [dimensionless, erg/s/Hz per unit amplitude,
        erg/s/Hz, Å], tabulated over the model's redshift range (one node for a
        single ``Fixed`` redshift), or a ``Decline`` with the reason when the fast path
        is refused, in
        which case the caller keeps the exact per-call dense filter integral.
        Term order is the emitter's ``emission_terms`` dict order. The consumer
        reads it at the evaluation redshift via
        ``term_band_response`` in ``components/_term_response.py``.

    Notes
    -----
    **Gate.** The response is a function of the evaluation redshift alone only while
    every one of the emitter's own parameters is fixed: a free *shape* parameter (a
    photon index, a spectral index, a turnover frequency) would move :math:`S_k` under
    the LUT. A free redshift, or a per-galaxy runtime redshift under
    ``catalog_z_range``, moves :math:`R_f` and any redshift-dependent :math:`S_k`,
    so both are tabulated over the model's redshift range (uniform in
    :math:`\ln(1+z)`) and read at the evaluation redshift. Gating on *all* the emitter's
    parameters
    rather than on a known set of shape parameters fails **safe**, an unrecognized
    free parameter simply disables the optimization. Denylisting known shape knobs
    would fail *dangerous* the day a new one is added and forgotten (#1107 shipped
    exactly that hole with ``dust_log_ssfr``).

    **Rank-1 probe.** Being a sum of rank-1 terms is a *property*, not a promise, so
    it is verified rather than declared: each term is built twice from deliberately
    distant input draws (``EMITTER_PROBE_INPUTS``) and must come back proportional.
    A term whose *shape* (not just amplitude) responds to a runtime input fails,
    and the whole emitter drops to the dense path. This is the BOSA lesson from
    #1107: an emitter whose template shape tracked its luminosity sailed through a
    band response built at ``L_ir = 1`` and returned fluxes 13 % wrong, silently.

    **Zero terms.** A term that is identically zero under both probes is zero for a
    *structural* reason, a Python-level switch (``include_freefree=False``) or a
    fixed zero parameter (``radio_loudness = 0``, the default), never because a
    probe happened to zero it: every probe input is nonzero by construction. Under
    the all-parameters-fixed gate it is therefore zero at runtime too, so
    :math:`R_k = 0` is correct rather than a silent drop.

    **JIT.** Runs at build time on concrete values; the returned arrays are threaded
    into the JIT as ``template_data``.
    """
    comp = _emitter(ctx.chain, name)
    if comp is None:
        return Decline("no such emitter in the component chain")
    free = set(ctx.spec.free_params)
    # ``redshift`` is deliberately not part of this gate: the response is
    # tabulated over it. Only the emitter's own shape parameters gate.
    refusal = _term_response_refusal(comp, ctx.chain, free, ctx.spec.get_fixed_values())
    if refusal is not None:
        return Decline(refusal)
    fw_pad, ft_pad = ctx.padded_filters
    if not (
        ctx.wave_precomp
        and fw_pad is not None
        and ft_pad is not None
        and hasattr(comp, "emission_terms")
        and hasattr(comp, "EMITTER_PROBE_INPUTS")
    ):
        return Decline(None)
    amplitudes_fixed = bool(getattr(comp, "FIXED_TERM_AMPLITUDES", False)) and not any(
        p in getattr(comp, "AMPLITUDE_PARAMS", ()) for p in free
    )
    response = _build_term_response_table(ctx, comp, fw_pad, ft_pad, amplitudes=amplitudes_fixed)
    if response is None:
        return Decline(
            "the rank-1 probe rejected a term: its spectral shape moves with its "
            "amplitude, or it is zero at some redshift node"
        )
    return BandTable(response)


def _build_term_response_table(ctx, comp, fw_pad, ft_pad, *, amplitudes=False):
    """Tabulate an additive emitter's per-term response over redshift.

    Each node evaluates the emitter's terms with ``redshift`` set to the
    node value, so a term whose amplitude or shape reads the redshift (the
    radio FIRRC evolution, the CMB suppression) is tabulated as it is
    evaluated. ``lam_ref`` is one wavelength per term (the peak of the
    model-range midpoint node), ``S_ref(z)`` the shape at that wavelength on
    every node, so ``A_k = term_k(lam_ref) / S_ref_k(z)`` stays the amplitude
    at every redshift.

    With ``amplitudes=True`` (every amplitude parameter fixed) the table also
    carries ``amp`` = ``A_k`` at the fixed values, so the emitter need not
    re-evaluate its terms at ``lam_ref`` on every call (#2987).

    Returns ``None`` (dense path) when a term is not rank-1 at any node, or is
    zero at some nodes only (its reference amplitude would be 0/0 there).
    """
    from tengri.components._z_response import build_z_table, single_node
    from tengri.observation.photometry import lnu_filter_integral_batch

    fixed = {k: jnp.asarray(v) for k, v in ctx.spec.get_fixed_values().items()}
    wave = ctx.wave
    nodes = ctx.z_nodes
    single = nodes is None
    if single:
        nodes = single_node(fixed["redshift"], wave.dtype)
    z_mid = fixed["redshift"] if single else jnp.expm1(nodes[nodes.shape[0] // 2])

    probes = tuple(comp.EMITTER_PROBE_INPUTS)

    def _terms(z, probe):
        return comp.emission_terms({**fixed, "redshift": z}, wave, **probe)

    # Reference wavelengths: the peak of each term, fixed once so S_ref(z) is
    # the same wavelength's shape on every node.
    # One traced program for the midpoint terms (#2914), then host arithmetic for
    # the peaks: evaluated eagerly, the X-ray and radio terms dispatched ~40 XLA
    # compiles per build, each below the persistent cache's floor. The term ORDER
    # is captured inside the trace and the values returned as a tuple: a jitted
    # function hands back a dict with its keys sorted, while the table's rows must
    # follow the emitter's own ``emission_terms`` order, which every consumer pairs
    # by position (``term_band_response``; tests/contract/test_term_response_order.py).
    order: list[str] = []

    def _mid_terms(z):
        terms = _terms(z, probes[0])
        order.extend(terms)
        return tuple(terms.values())

    mid_values = jax.jit(_mid_terms)(z_mid)
    mid_terms = {key: np.asarray(v) for key, v in zip(order, mid_values, strict=True)}
    peaks = {key: int(np.argmax(np.abs(s_a))) for key, s_a in mid_terms.items()}
    # A term that is zero on the whole grid has no peak: ``argmax`` of zeros is node 0,
    # the shortest wavelength of the grid (0.0413 A once an X-ray wing is attached), and
    # the emitter is then evaluated there on every call. Its amplitude is 0 at any
    # wavelength, so reference it at a live term's peak: the same evaluation point the
    # live term already pays for, and one the emitter supports.
    live_peaks = [peaks[k] for k, s_a in mid_terms.items() if float(np.max(np.abs(s_a))) > 0]
    if live_peaks:
        peaks = {
            k: (peaks[k] if float(np.max(np.abs(mid_terms[k]))) > 0 else live_peaks[0])
            for k in peaks
        }
    keys = tuple(peaks)
    # Wavelengths at which the emitter publishes point diagnostics (the AGN's 6 and 12 um
    # luminosities). Each term is its amplitude times a fixed shape, so the shape there is
    # one more build-time constant and no per-call evaluation is needed for it.
    diag_wave = jnp.asarray(getattr(comp, "EMITTER_DIAGNOSTIC_NODES", ()), dtype=wave.dtype)
    # The reference wavelengths as one array, evaluated exactly as the per-call path does.
    lam_ref = jnp.asarray(np.asarray(wave)[[peaks[key] for key in keys]], dtype=wave.dtype)

    def _node(z):
        terms_all = [_terms(z, probe) for probe in probes]
        terms_a = terms_all[0]
        rows, s_ref, ok, s_diag = [], [], [], []
        diag_terms = (
            comp.emission_terms({**fixed, "redshift": z}, diag_wave, **probes[0])
            if diag_wave.shape[0]
            else None
        )
        at_ref = comp.emission_terms({**fixed, "redshift": z}, lam_ref) if amplitudes else None
        for key in keys:
            s_a, peak = terms_a[key], peaks[key]
            ref = s_a[peak]
            live = jnp.abs(ref) > 0.0
            # A term identically zero here is zero for a STRUCTURAL reason (a
            # Python-level switch, or a fixed zero parameter): every probe
            # input is nonzero by construction. It must be zero everywhere
            # on the grid, or its reference amplitude is 0/0 at some nodes.
            zero_ok = live | (jnp.max(jnp.abs(s_a)) == 0.0)
            term_ok = zero_ok & jnp.all(jnp.isfinite(s_a))
            # RANK-1 CHECK: every later draw must be proportional to the first. A
            # draw that switches a live term off, or makes it non-finite, fails
            # the check: an all-zero draw would otherwise pass as 0 * s_a.
            for terms_b in terms_all[1:]:
                s_b = terms_b[key]
                scale = s_b[peak] / jnp.where(live, ref, 1.0)
                rank1_ok = ~live | jnp.allclose(s_b, scale * s_a, rtol=_RANK1_RTOL, atol=0.0)
                alive_ok = ~live | (jnp.all(jnp.isfinite(s_b)) & (jnp.max(jnp.abs(s_b)) > 0.0))
                term_ok = term_ok & rank1_ok & alive_ok
            row = lnu_filter_integral_batch(s_a, wave, fw_pad, ft_pad, z)
            rows.append(jnp.where(live, row, jnp.zeros_like(row)))
            # S_ref = 1 keeps A = term/1 = 0 finite instead of 0/0.
            s_ref.append(jnp.where(live, ref, jnp.ones_like(ref)))
            ok.append(term_ok)
            if diag_terms is not None:
                s_diag.append(jnp.where(live, diag_terms[key], jnp.zeros_like(diag_terms[key])))
        out = {"R": jnp.stack(rows), "S_ref": jnp.stack(s_ref), "ok": jnp.stack(ok)}
        if diag_terms is not None:
            out["S_diag"] = jnp.stack(s_diag)
        if at_ref is not None:
            # A_k = term_k(lam_ref) / S_ref_k, the per-call amplitude at the fixed values.
            at_lam = jnp.stack([at_ref[key][i] for i, key in enumerate(keys)])
            out["amp"] = at_lam / jnp.stack(s_ref)
        return out

    built = build_z_table(_node, nodes, fixed["redshift"] if single else None)
    values = built["values"]
    if not bool(jnp.all(values["ok"])):
        return None
    table = {
        "ln1pz": built["ln1pz"],
        "R": values["R"],
        "S_ref": values["S_ref"],
        "lam_ref": lam_ref,
    }
    for key in ("S_diag", "amp"):
        if key in values:
            table[key] = values[key]
    return table


def additive_term_shape_table(ctx, name) -> BandTable | Decline:
    r"""Shape-only band tables of an additive emitter, built through its registry adapters.

    :func:`additive_term_band_response` needs every emitter parameter fixed. With a shape
    parameter free the component integrates the filters on every call. Each term of the
    emitter that has a registry adapter (``TERM_SHAPE_KEYS`` of the component, via
    ``term_shape_keys()``) gets a table of its shape-only band flux over that shape, built by
    the adapter at build time (see ``tengri.components._term_shape_table``). The amplitude
    is not tabulated: the component multiplies each table by the term at its own reference
    wavelength, so FIRRC, redshift evolution and suppression stay exact.

    Parameters
    ----------
    ctx : BandTableContext
        The model state the tables are built from.
    name : str
        Component name, for example ``"radio"`` or ``"xray"``.

    Returns
    -------
    BandTable or Decline
        ``{term: table}`` for every term, or a ``Decline`` with no reason (``None``) when the
        emitter declares no such table, no parameter of it is free, the all-fixed exact
        response applies, the redshift is not ``Fixed``, or any term's adapter declines. The
        caller falls back to the dense per-call integral.

    Notes
    -----
    **JIT-compatible**: no, build-time; the returned arrays are threaded into the JIT.
    """
    comp = _emitter(ctx.chain, name)
    keys = comp.term_shape_keys() if hasattr(comp, "term_shape_keys") else {}
    fw_pad, ft_pad = ctx.padded_filters
    prefix = f"{name}_"
    if not (
        keys
        and any(p.startswith(prefix) for p in ctx.spec.free_params)
        and ctx.wave_precomp
        and fw_pad is not None
        and ft_pad is not None
        and ctx.z_nodes is None
        and not isinstance(ctx.ask(name, "term_band_response"), BandTable)
    ):
        return Decline(None)
    from tengri.forward.precompute import registry

    redshift = float(ctx.spec.get_fixed_values()["redshift"])
    tables = {}
    for term, key in keys.items():
        module = registry.resolve(key)
        term_table = module.term_shape_table(
            wave=ctx.wave,
            filter_waves_padded=fw_pad,
            filter_trans_padded=ft_pad,
            redshift=redshift,
            parameters=ctx.spec,
            model=key,
            term=term,
        )
        if term_table is None:
            return Decline(None)
        tables[term] = term_table
    return BandTable(tables)


def term_band_table(ctx, *, component: str, slot: str) -> BandTable | Decline:
    """Return one term slot of ``component``: its rank-1 band response or its shape tables.

    Parameters
    ----------
    ctx : BandTableContext
        The model state.
    component : str
        The additive emitter, for example ``"radio"``, ``"xray"`` or ``"agn"``.
    slot : str
        ``"term_band_response"`` or ``"term_shape_table"``.

    Returns
    -------
    BandTable or Decline
        The table, or the refusal with the reason it gives.

    Raises
    ------
    ValueError
        If ``slot`` is not a term slot.
    """
    if slot == "term_band_response":
        return additive_term_band_response(ctx, component)
    if slot == "term_shape_table":
        return additive_term_shape_table(ctx, component)
    raise ValueError(f"no term band-table slot {slot!r} of component {component!r}")


def _emitter(chain, name: str):
    """Return the component of ``chain`` named ``name``, or None when it is absent."""
    return next((c for c in chain if getattr(c, "name", "") == name), None)
