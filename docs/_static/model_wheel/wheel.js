// Model catalog wheel: physics module -> sub-physics group -> implemented model.
// Ported from the approved reference figure. Data is the JSON that docs/conf.py
// (_write_model_wheel) resolves from docs/model_wheel.yaml and the live registries
// and inlines as <script id="tg-wheel-data">. Everything is drawn inside the
// .tg-wheel container; keys act only while focus is inside it.
(function () {
  "use strict";
  const SCRIPT_SRC = document.currentScript && document.currentScript.src;
  const STATIC_BASE = SCRIPT_SRC ? new URL("../", SCRIPT_SRC).href : "_static/";

  function init() {
    const host = document.getElementById("tg-wheel");
    const dataEl = document.getElementById("tg-wheel-data");
    if (!host || !dataEl || typeof d3 === "undefined") return;
    let DATA;
    try { DATA = JSON.parse(dataEl.textContent); }
    catch (e) { host.textContent = "The model catalog data could not be read."; console.error(e); return; }
    build(host, DATA);
  }

  // ---- theme ---------------------------------------------------------------
  function themeIsLight() {
    const t = document.body.dataset.theme;
    if (t === "light") return true;
    if (t === "dark") return false;
    return !matchMedia("(prefers-color-scheme: dark)").matches;
  }
  const lum = c => {
    const f = v => { v /= 255; return v <= .03928 ? v / 12.92 : Math.pow((v + .055) / 1.055, 2.4); };
    const r = d3.rgb(c); return .2126 * f(r.r) + .7152 * f(r.g) + .0722 * f(r.b);
  };
  const contrast = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + .05) / (Math.min(x, y) + .05); };
  // Darken a hue until it reads at >= 4.5:1 on the background (light theme text).
  function readableOn(c, bg) {
    const h = d3.hsl(c);
    for (let i = 0; i < 60 && contrast(h + "", bg) < 4.5; i++) h.l = Math.max(0, h.l - .02);
    return h + "";
  }
  function palette() {
    const light = themeIsLight(), bg = light ? "#FAFBFC" : "#0F1419";
    const shade = (c, k) => { const h = d3.hsl(c); h.l = Math.min(.92, h.l + k); return h + ""; };
    return {
      bg,
      module: c => c,
      group: c => shade(c, .12),
      model: c => d3.interpolateRgb(c, bg)(.45),
      text: (c, k) => light ? readableOn(c, bg) : shade(c, k),
    };
  }

  // ---- build ---------------------------------------------------------------
  function build(host, DATA) {
    host.setAttribute("tabindex", "0");
    host.setAttribute("aria-label", "Tengri model catalog wheel. Click or press Enter on a wedge to zoom; Escape zooms out.");
    host.insertAdjacentHTML("beforeend",
      '<div class="fig"><svg role="img" aria-label="Tengri model catalog: physics modules, sub-physics and implemented models"></svg></div>' +
      '<ul class="legend"></ul>' +
      '<aside class="card" aria-live="polite"><button class="x" type="button" aria-label="Close">×</button>' +
      '<p class="path"></p><p class="name"></p><p class="lab">Reference</p><p class="cite"></p></aside>' +
      '<div class="key"><b>Click</b> a wedge to zoom, a model for its reference · <b>hub</b> or <b>Esc</b> to go out · ' +
      '<b>← →</b> step through modules · <b>L</b> rim labels · <b>H</b> hide key</div>');
    const fig = host.querySelector(".fig"), leg = host.querySelector(".legend"), card = host.querySelector(".card");
    const keyEl = host.querySelector(".key");

    const SHORT = {};
    const tree = {
      name: "tengri",
      children: DATA.modules.map(m => {
        if (m.short) SHORT[m.label] = m.short;
        return {
          name: m.label, color: m.color, children: m.groups.map(g => {
            if (g.short) SHORT[g.label] = g.short;
            return {
              name: g.label, children: g.models.map(md => {
                if (md.short) SHORT[md.label] = md.short;
                return { name: md.label, cite: md.cite || "", flag: md.cite_status === "unverified" ? "x" : "", model: md };
              })
            };
          })
        };
      })
    };
    const root = d3.hierarchy(tree).count();
    d3.partition().size([1, 3])(root);
    const nodes = root.descendants().slice(1);
    nodes.forEach((d, i) => { d.id = "tgw" + i; const m = d.depth === 1 ? d : d.ancestors().find(a => a.depth === 1); d.mod = m; d.col = m.data.color; });
    const mods = root.children;
    let P = palette();

    leg.innerHTML = mods.map((m, i) => `<li data-i="${i}" tabindex="0"><i style="background:${m.data.color}"></i>${m.data.name}</li>`).join("");
    leg.addEventListener("click", e => { const li = e.target.closest("li"); if (li) zoomTo(mods[+li.dataset.i]); });
    leg.addEventListener("keydown", e => { const li = e.target.closest("li"); if (li && e.key === "Enter") { e.stopPropagation(); zoomTo(mods[+li.dataset.i]); } });

    const svg = d3.select(fig.querySelector("svg"));
    const g = svg.append("g");
    const arcs = g.append("g").selectAll("path").data(nodes).join("path")
      .attr("class", "arc").attr("tabindex", 0)
      .on("click", (e, d) => { if (d.depth === 3) { showCard(d); e.stopPropagation(); } else zoomTo(d); })
      .on("keydown", (e, d) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); d.depth === 3 ? showCard(d) : zoomTo(d); } });
    arcs.append("title").text(d => d.ancestors().reverse().slice(1).map(a => a.data.name).join(" › "));

    const labG = g.append("g").attr("class", "labs");
    const defs = svg.append("defs");
    const ringNodes = nodes.filter(d => d.depth < 3);
    const guides = defs.selectAll("path.g1").data(ringNodes).join("path").attr("class", "g1").attr("id", d => "p" + d.id);
    const guides2 = defs.selectAll("path.g2").data(ringNodes).join("path").attr("class", "g2").attr("id", d => "q" + d.id);
    const ringT = labG.append("g").selectAll("text").data(ringNodes).join("text").attr("class", "ring-t")
      .append("textPath").attr("href", d => "#p" + d.id).attr("startOffset", "50%").attr("text-anchor", "middle");
    // Second line, used when a long name is wrapped onto two curved lines in a narrow wedge.
    const ringT2 = labG.append("g").selectAll("text").data(ringNodes).join("text").attr("class", "ring-t")
      .append("textPath").attr("href", d => "#q" + d.id).attr("startOffset", "50%").attr("text-anchor", "middle");
    const leafNodes = nodes.filter(d => d.depth === 3);
    const leafT = labG.append("g").selectAll("text").data(leafNodes).join("text").attr("class", "leaf-t").text(d => d.data.name);
    const subNodes = nodes.filter(d => d.depth === 2);
    const subLead = labG.append("g").selectAll("path").data(subNodes).join("path").attr("fill", "none").attr("stroke-width", 1);
    const subT = labG.append("g").selectAll("text").data(subNodes).join("text").attr("class", "sub-t").text(d => d.data.name);

    const hub = g.append("g").attr("class", "hub").on("click", () => zoomTo(root)); // the hub always returns to the full wheel
    const hubC = hub.append("circle").attr("fill", "#E6EDF3").attr("stroke", "#2C4E74").attr("stroke-width", 2);
    const hubL = hub.append("image").attr("href", STATIC_BASE + "tengri-logo.png").attr("preserveAspectRatio", "xMidYMid meet");
    const hubTxt = hub.append("g").attr("class", "hubtxt");
    const hubT = hubTxt.append("text").attr("text-anchor", "middle").attr("fill", "#2b2b2b").style("font-weight", 400).style("letter-spacing", "0");
    const hubB = hubTxt.append("text").attr("text-anchor", "middle").attr("fill", "#2C4E74").style("font-weight", 500);

    let hubMix = 1, grow = 1, moving = false, labelsOn = true, W = 0, H = 0, R = 0, focus = root, view = { x0: 0, x1: 1 };
    const T = Math.PI * 2;
    const ang = x => Math.max(0, Math.min(1, (x - view.x0) / (view.x1 - view.x0))) * T;
    const radii = () => { const k = grow; return [R * .155 * k, R * .275 * k, R * .405 * k, R * .53 * k]; };
    const measure = (() => { const c = document.createElement("canvas").getContext("2d"); return (t, px, w) => { c.font = `${w || 600} ${px}px "IBM Plex Sans"`; return c.measureText(t).width; }; })();

    function colorArcs() {
      arcs.attr("fill", d => d.depth === 1 ? P.module(d.col) : d.depth === 2 ? P.group(d.col) : P.model(d.col)).attr("stroke", P.bg);
      // Ring labels carry a halo in their ring's fill: a label that runs past its wedge edge
      // then masks the border line under it instead of being cut by it.
      labG.selectAll("text.ring-t")
        .attr("stroke", d => d.depth === 1 ? P.module(d.col) : P.group(d.col))
        .attr("stroke-width", 3.5).attr("stroke-linejoin", "round").style("paint-order", "stroke");
    }

    function render() {
      const rs = radii(), r3 = rs[3];
      const arc = d3.arc().startAngle(d => ang(d.x0)).endAngle(d => ang(d.x1))
        .padAngle(d => Math.min((ang(d.x1) - ang(d.x0)) / 3, d.depth === 3 ? .0015 : .004)).padRadius(R * .5)
        .innerRadius(d => rs[d.depth - 1] + (d.depth === 1 ? 2 : 1)).outerRadius(d => rs[d.depth] - (d.depth === 3 ? 0 : 1));
      arcs.attr("d", arc).each(function (d) {
        const wpx = (ang(d.x1) - ang(d.x0)) * rs[d.depth], base = d.depth === 3 ? .6 : 1.6;
        this.setAttribute("visibility", wpx > 1 ? "visible" : "hidden");
        this.setAttribute("stroke-width", Math.min(base, wpx * .12).toFixed(2));
      });
      renderHub(rs[0]);
      if (moving) return;
      renderRingLabels(rs);
      renderLeafLabels(r3);
      renderRimLabels(r3);
    }

    // Arc geometry of a ring wedge's label, and a guide path at radius rr along it.
    function arcGeom(d, rs) {
      const a0 = ang(d.x0), a1 = ang(d.x1), mid = (a0 + a1) / 2;
      const full = a1 - a0 > T - 1e-3;
      return {
        a0, a1, full,
        flip: !full && mid > Math.PI / 2 && mid < Math.PI * 1.5,
        rm: (rs[d.depth - 1] + rs[d.depth]) / 2,
        band: rs[d.depth] - rs[d.depth - 1],
        span: Math.min(a1 - a0, Math.PI * .98), // usable label length
      };
    }
    function guideAt(g, rr) {
      const s = g.full ? -Math.PI * .49 : g.a0, e = g.full ? Math.PI * .49 : g.a1;
      const p = (a, r) => [r * Math.sin(a), -r * Math.cos(a)];
      const [x0, y0] = p(g.flip ? e : s, rr), [x1, y1] = p(g.flip ? s : e, rr);
      // The large-arc flag follows the true drawn angle, not the capped label span:
      // a wedge wider than 180° otherwise gets its guide on the wrong circle.
      const large = (g.full ? Math.PI * .98 : e - s) > Math.PI ? 1 : 0;
      return `M${x0},${y0}A${rr},${rr} 0 ${large} ${g.flip ? 0 : 1} ${x1},${y1}`;
    }
    // Split a name into two lines of near-equal width at a word boundary.
    function splitTwo(txt) {
      const w = txt.split(" "); if (w.length < 2) return null;
      let best = null;
      for (let i = 1; i < w.length; i++) {
        const a = w.slice(0, i).join(" "), b = w.slice(i).join(" ");
        const cost = Math.max(measure(a, 10), measure(b, 10));
        if (!best || cost < best.cost) best = { a, b, cost };
      }
      return [best.a, best.b];
    }
    // Largest font for a one-line or two-line label on this wedge's arc.
    function fitOne(txt, px0, g) {
      const len = g.span * g.rm * .88, w = measure(txt, px0);
      return w > len ? px0 * len / w : px0;
    }
    function fitTwo(lines, px0, g) {
      // Two lines need about 2.25 line heights of band; each line fits its own arc length.
      let px = Math.min(px0, g.band * .82 / 2.25);
      lines.forEach((t, i) => {
        const r = g.rm + (i === 0 ? 1 : -1) * .55 * px * (g.flip ? -1 : 1);
        const len = g.span * r * .88, w = measure(t, px);
        if (w > len) px *= len / w;
      });
      return px;
    }

    // Angular half-width a label occupies on its ring.
    const labelHalf = d => {
      const lab = d._lab, w = Math.max(...lab.lines.map(t => measure(t, lab.px)));
      return w / (2 * d._g.rm * .97);
    };
    // A label still under `target` px in a narrow wedge may run past its wedge edges
    // along the arc, at `target` px, when the angle is free of its neighbors' labels.
    function overflowNarrow(vis, target) {
      [1, 2].forEach(depth => {
        const ring = vis.filter(d => d.depth === depth).sort((a, b) => a._g.a0 - b._g.a0);
        if (ring.length < 2) return;
        ring.forEach((d, i) => {
          if (d._lab.px >= target && !d._lab.short) return;  // shortened labels try their full name too
          const prev = ring[(i - 1 + ring.length) % ring.length], next = ring[(i + 1) % ring.length];
          const mid = (d._g.a0 + d._g.a1) / 2, pad = 12 / d._g.rm;
          const wrap = a => ((a % T) + T) % T;
          const midOf = o => (o._g.a0 + o._g.a1) / 2;
          // Free angle on each side: up to where the neighbor's own label starts.
          const freeL = wrap(mid - midOf(prev)) - labelHalf(prev) - pad;
          const freeR = wrap(midOf(next) - mid) - labelHalf(next) - pad;
          // Free arc runs from mid-freeL to mid+freeR. The label may shift toward the roomier
          // side, but its center stays inside its own wedge so it still reads as that wedge's.
          const lo = mid - freeL, hi = mid + freeR;
          const g = d._g;
          const cands = [{ lines: [d.data.name] }];
          const two = splitTwo(d.data.name);
          if (two && 2.25 * target <= g.band * .82) cands.push({ lines: two });
          const keepShort = d._lab.short && d._lab.px >= target;
          if (SHORT[d.data.name] && !keepShort) cands.push({ lines: [SHORT[d.data.name]] });
          for (const c of cands) {
            const r = c.lines.length === 2 ? g.rm - .55 * target : g.rm;
            const need = Math.max(...c.lines.map(t => measure(t, target))) / (r * .97);
            const cMin = Math.max(lo + need / 2, g.a0), cMax = Math.min(hi - need / 2, g.a1);
            if (cMin <= cMax) {
              const cen = Math.max(cMin, Math.min(cMax, mid)), half = need / 2 / .88;
              d._g = { ...g, a0: cen - half, a1: cen + half, span: 2 * half };
              d._lab = { lines: c.lines, px: target };
              return;
            }
          }
        });
      });
    }

    function renderRingLabels(rs) {
      // Full view: all-or-nothing per ring (one shared size, one line). Zoomed: every
      // label shows, sized on its own. A name too long for its wedge is first wrapped onto
      // two curved lines; the short name is used only when that is still the smaller text.
      const MIN = 13, ZMIN = 12, FULLMIN = 10.5, OVER = 11, FLOOR = 9, top = focus === root;
      const fit = { 1: [], 2: [] };
      ringT.each(function (d) {
        const g = arcGeom(d, rs);
        d._g = g;
        d._vis = g.a1 - g.a0 > 1e-3 && !(top && d.depth === 2);
        if (!d._vis) return;
        // Zoomed, a ring label may start at 13-15 px even on a small wheel, and may use
        // more of its band; the full view keeps the reference sizing.
        const base = (d.depth === 1 ? R * .05 : R * .034) * grow;
        const px0 = top
          ? Math.min(base, g.band * .36, d.depth === 1 ? 32 : 22)
          : Math.min(Math.max(base, d.depth === 1 ? 15 : 13), g.band * .55, d.depth === 1 ? 32 : 22);
        let best = { lines: [d.data.name], px: fitOne(d.data.name, px0, g) };
        if (!top && best.px < Math.max(ZMIN, px0 * .8)) {
          const two = splitTwo(d.data.name);
          const cands = [];
          if (two) cands.push({ lines: two, px: fitTwo(two, px0, g) });
          const sh = SHORT[d.data.name];
          if (sh) cands.push({ lines: [sh], px: fitOne(sh, px0, g), short: true });
          cands.forEach(c => {
            // Prefer the full name (one or two lines) whenever it is readable.
            if (c.short && best.px >= FULLMIN) return;
            if (!c.short && c.px >= FULLMIN && best.px < FULLMIN) { best = c; return; }
            if (c.px > best.px) best = c;
          });
        }
        d._lab = best; fit[d.depth].push(best.px);
      });
      if (!top) { overflowNarrow(ringT.data().filter(d => d._vis), OVER); overflowNarrow(ringT.data().filter(d => d._vis), OVER - 1); }
      const lvl = {}; [1, 2].forEach(k => { const m = fit[k].length ? Math.min(...fit[k]) : 0; lvl[k] = m >= MIN ? m : 0; });

      const baseOff = d => (d.depth === 1 ? R * .018 : R * .016);
      guides.attr("d", d => {
        if (!d._vis) return "M0,0";
        const g = d._g, two = !top && d._lab.lines.length === 2, px = d._lab.px;
        // One line: baseline offset so the text sits centered in the band (as before).
        // Two lines: line 1 centered above the middle radius, line 2 below, in reading order.
        if (!two) return guideAt(g, g.flip ? g.rm + baseOff(d) : g.rm - baseOff(d));
        return guideAt(g, g.flip ? g.rm - .2 * px : g.rm + .2 * px);
      });
      guides2.attr("d", d => {
        if (!d._vis || top || d._lab.lines.length !== 2) return "M0,0";
        const g = d._g, px = d._lab.px;
        return guideAt(g, g.flip ? g.rm + .9 * px : g.rm - .9 * px);
      });
      ringT.each(function (d) {
        const sel = d3.select(this);
        if (!d._vis) { sel.text(""); return; }
        if (top) { if (!lvl[d.depth]) { sel.text(""); return; } sel.text(d.data.name).attr("font-size", lvl[d.depth].toFixed(1)); return; }
        if (d._lab.px < FLOOR) { sel.text(""); return; }
        sel.text(d._lab.lines[0]).attr("font-size", d._lab.px.toFixed(1));
      });
      ringT2.each(function (d) {
        const sel = d3.select(this);
        if (!d._vis || top || d._lab.lines.length !== 2 || d._lab.px < FLOOR) { sel.text(""); return; }
        sel.text(d._lab.lines[1]).attr("font-size", d._lab.px.toFixed(1));
      });
    }

    function renderLeafLabels(r3) {
      const ZMIN = 12, FLOOR = 9, top = focus === root;
      // Room for a radial label: out to the container edge in its own direction, so
      // labels at the top and bottom of a tall layout (or the sides of a wide one) use it.
      const edge = a => Math.min(W / 2 / Math.max(Math.abs(Math.sin(a)), 1e-3), H / 2 / Math.max(Math.abs(Math.cos(a)), 1e-3));
      leafT.each(function (d) {
        const sel = d3.select(this), w = ang(d.x1) - ang(d.x0);
        if (w < 1e-4 || top) { sel.attr("visibility", "hidden"); return; }
        const maxLen = Math.min(edge((ang(d.x0) + ang(d.x1)) / 2) * .98, R * 1.6) - r3 - 10;
        const cap = Math.min(w * (r3 + 6) * .72, Math.max(R * .034, 12), 22);
        let txt = d.data.name, px = cap; const tw = measure(txt, px, 400); if (tw > maxLen) px *= maxLen / tw;
        if (px < ZMIN && SHORT[txt]) { const t2 = SHORT[txt]; let p2 = cap; const tw2 = measure(t2, p2, 400); if (tw2 > maxLen) p2 *= maxLen / tw2; if (p2 > px) { txt = t2; px = p2; } }
        if (px < FLOOR) { sel.attr("visibility", "hidden"); return; }
        const mid = (ang(d.x0) + ang(d.x1)) / 2, deg = mid * 180 / Math.PI, flip = deg > 180;
        sel.text(txt).attr("visibility", "visible").attr("font-size", px.toFixed(1)).attr("fill", P.text(d.col, .18))
          .attr("dy", "0.35em").attr("text-anchor", flip ? "end" : "start")
          .attr("transform", `rotate(${deg - 90}) translate(${r3 + 7},0) ${flip ? "rotate(180)" : ""}`);
      });
    }

    // Sub-physics rim labels (full view only), relaxed apart so none overlap.
    function renderRimLabels(r3) {
      const show = focus === root && labelsOn;
      subT.attr("visibility", show ? "visible" : "hidden"); subLead.attr("visibility", show ? "visible" : "hidden");
      if (!show) return;
      const rL = r3 + R * .035, px = Math.min(R * .042, 24), maxW = R * .99 - rL - 6;
      const items = subNodes.map(d => ({ d, want: (ang(d.x0) + ang(d.x1)) / 2 }));
      const gap = (px * 1.22) / rL;
      items.forEach(o => { o.a = o.want; });
      for (let it = 0; it < 60; it++) {
        for (let i = 1; i < items.length; i++) { if (items[i].a - items[i - 1].a < gap) { const m = (items[i].a + items[i - 1].a) / 2; items[i - 1].a = m - gap / 2; items[i].a = m + gap / 2; } }
        const n = items.length, wrap = items[0].a + T - items[n - 1].a; if (wrap < gap) { items[n - 1].a -= (gap - wrap) / 2; items[0].a += (gap - wrap) / 2; }
      }
      const Pt = (a, r) => [r * Math.sin(a), -r * Math.cos(a)];
      items.forEach(o => {
        const d = o.d, deg = o.a * 180 / Math.PI, flip = deg > 180 && deg < 360;
        const t = subT.filter(x => x === d);
        let f = px; t.attr("font-size", f); const w = t.node().getComputedTextLength() || measure(d.data.name, f, 500); if (w > maxW) f *= maxW / w; o.f = f;
        t.attr("fill", P.text(d.col, .2)).attr("dy", "0.35em").attr("text-anchor", flip ? "end" : "start")
          .attr("transform", `rotate(${deg - 90}) translate(${rL + 6},0) ${flip ? "rotate(180)" : ""}`);
        const [x0, y0] = Pt(o.want, r3 + 3), [x1, y1] = Pt(o.want, r3 + R * .022), [x2, y2] = Pt(o.a, rL + 2);
        o.t = t;
        subLead.filter(x => x === d).attr("stroke", P.text(d.col, .05)).attr("stroke-opacity", .75).attr("d", `M${x0},${y0}L${x1},${y1}L${x2},${y2}`);
      });
      const fmin = Math.min(...items.map(o => o.f)), ok = fmin >= 13;
      items.forEach(o => o.t.attr("font-size", fmin.toFixed(1)).attr("visibility", ok ? "visible" : "hidden"));
      subLead.attr("visibility", ok ? "visible" : "hidden");
    }

    function renderHub(r0) {
      // hubMix: 1 = full-view layout, 0 = zoomed layout; tweened with the zoom so the logo glides
      hubC.attr("r", Math.max(0, r0 - 3));
      const m = hubMix, L = r0 * (.86 + (1.02 - .86) * m), y = -r0 * (.9 - (.9 - .78) * m);
      hubL.attr("width", L).attr("height", L).attr("x", -L / 2).attr("y", y);
      hubT.attr("y", r0 * (.3 + .2 * m)); hubB.attr("y", r0 * .58);
      if (moving) return;
      const isRoot = focus === root;
      let nm = isRoot ? "tengri" : focus.data.name;
      hubT.style("font-family", isRoot ? '"Segoe UI","Segoe UI Variable",Selawik,"Noto Sans",sans-serif' : '"IBM Plex Sans",sans-serif')
        .style("font-weight", isRoot ? 400 : 600).attr("fill", isRoot ? "#2b2b2b" : "#0F1419");
      const fs0 = isRoot ? r0 * .42 : r0 * .24;
      const hubFit = t => { hubT.text(t).attr("font-size", fs0); const tw = hubT.node().getComputedTextLength() || measure(t, fs0, 700); return tw > r0 * 1.62 ? fs0 * r0 * 1.62 / tw : fs0; };
      let fs = hubFit(nm);
      // A long focus name shrinks to unreadable in the hub; use its short name instead.
      if (!isRoot && fs < fs0 * .75 && SHORT[nm]) { const f2 = hubFit(SHORT[nm]); if (f2 > fs) { nm = SHORT[nm]; fs = f2; } }
      hubT.text(nm);
      hubT.attr("font-size", fs.toFixed(1));
      hubB.text(isRoot ? "" : "‹ tengri").attr("font-size", (r0 * .15).toFixed(1));
    }

    // ---- layout ---------------------------------------------------------------
    // Small screens: 1.8x in the full view to fill the width; 0.9x zoomed, which keeps the
    // ring bands tall enough for readable labels while model names use the edge room.
    // Applies only when the wheel is width-limited (a phone), never to a short window.
    const growFor = d => (W < 600 && W <= H) ? (d === root ? 1.8 : .9) : 1;
    // The legend is a row across the top; the wheel area starts below it.
    function legH() {
      host.style.setProperty("--legh", (leg.style.opacity !== "0" ? leg.offsetHeight + 6 : 0) + "px");
      // The key line sits below the wheel area, never over it.
      const keyOn = !keyEl.classList.contains("off") && getComputedStyle(keyEl).display !== "none";
      host.style.setProperty("--keyh", (keyOn ? keyEl.offsetHeight + 14 : 0) + "px");
    }
    function size() {
      W = fig.clientWidth; H = fig.clientHeight; if (!W || !H) return;
      R = Math.min(W, H) / 2 * .99; if (!moving) grow = growFor(focus); svg.attr("viewBox", `${-W / 2} ${-H / 2} ${W} ${H}`);
      render();
    }
    new ResizeObserver(() => { legH(); size(); }).observe(host);
    new ResizeObserver(legH).observe(leg);

    // ---- zoom -----------------------------------------------------------------
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const ZOOM_MS = 750;
    let zoomSeq = 0, zoomGuard = 0;
    const slug = d => d.ancestors().reverse().slice(1).map(a => a.data.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")).join("--");
    const bySlug = h => nodes.find(d => d.depth < 3 && slug(d) === h);
    function zoomTo(d, quiet) {
      if (!d) return; focus = d;
      const fm = d === root ? null : (d.depth === 1 ? d : d.parent);
      leg.querySelectorAll("li").forEach((li, i) => li.classList.toggle("on", mods[i] === fm));
      const target = { x0: d === root ? 0 : d.x0, x1: d === root ? 1 : d.x1 };
      const iv = d3.interpolate({ ...view }, target), ig = d3.interpolate(grow, growFor(d)), ih = d3.interpolate(hubMix, d === root ? 1 : 0);
      try { if (!quiet) history.replaceState(null, "", d === root ? location.pathname + location.search : "#" + slug(d)); } catch (e) { /* file:// or sandboxed */ }
      hideCard();
      // Only the latest zoom owns the wheel's state. Each zoom finishes exactly once: on its
      // transition's end, or from a guard timer if the transition is slow, stalls or is lost
      // (a busy or throttled tab). An older zoom pre-empted by a newer one does nothing, so
      // the newer zoom keeps the cheap per-frame path and restores the labels itself.
      // Labels fade out and back in by CSS on the "moving" class, not by JS animation, so
      // they always reach full opacity even when the browser throttles animation frames.
      const seq = ++zoomSeq;
      clearTimeout(zoomGuard);
      const settle = { view: target, grow: growFor(d), hubMix: d === root ? 1 : 0 };
      const finish = () => {
        if (seq !== zoomSeq || !moving) return;
        clearTimeout(zoomGuard);
        svg.interrupt("z");
        view = settle.view; grow = settle.grow; hubMix = settle.hubMix;
        moving = false; host.classList.remove("moving");
        render();
      };
      if (reduce) { view = settle.view; grow = settle.grow; hubMix = settle.hubMix; moving = false; host.classList.remove("moving"); render(); return; }
      moving = true; host.classList.add("moving");
      svg.transition("z").duration(ZOOM_MS).ease(d3.easeCubicInOut)
        .tween("z", () => t => { if (seq === zoomSeq) { view = iv(t); grow = ig(t); hubMix = ih(t); render(); } })
        .on("end", finish);
      zoomGuard = setTimeout(finish, ZOOM_MS + 350);
    }

    // ---- model card -------------------------------------------------------------
    function showCard(d) {
      arcs.classed("sel", x => x === d);
      const path = card.querySelector(".path");
      path.innerHTML = "";
      const dot = document.createElement("i"); dot.style.background = d.col; path.appendChild(dot);
      path.appendChild(document.createTextNode(`${d.mod.data.name} › ${d.parent.data.name}`));
      card.querySelector(".name").textContent = d.data.name;
      const c = card.querySelector(".cite");
      c.textContent = d.data.cite || "No reference recorded in the tengri registry.";
      if (d.data.flag === "x") { const n = document.createElement("span"); n.className = "note"; n.textContent = "Not yet in tengri's citation list."; c.appendChild(n); }
      card.classList.add("on");
    }
    function hideCard() { card.classList.remove("on"); arcs.classed("sel", false); }
    card.querySelector(".x").addEventListener("click", hideCard);

    // ---- keys (only while focus is inside the wheel) ---------------------------
    const seq = [root, ...mods];
    host.addEventListener("keydown", e => {
      const k = e.key;
      if (k === "Escape") { if (card.classList.contains("on")) hideCard(); else zoomTo(focus.parent || root); }
      else if (k === "l" || k === "L") { labelsOn = !labelsOn; render(); }
      else if (k === "ArrowRight" || k === "ArrowLeft" || k === "PageDown" || k === "PageUp") {
        const dir = (k === "ArrowRight" || k === "PageDown") ? 1 : -1;
        if (focus.depth === 2) { const sib = focus.parent.children, i = sib.indexOf(focus) + dir; zoomTo(i < 0 || i >= sib.length ? focus.parent : sib[i]); }
        else { const cur = seq.indexOf(focus); zoomTo(seq[(cur + dir + seq.length) % seq.length]); }
      }
      else if (k === "ArrowDown" && focus.children && focus.depth < 2) zoomTo(focus.children[0]);
      else if (k === "ArrowUp") zoomTo(focus.parent || root);
      else if (k === "h" || k === "H") { keyEl.classList.toggle("off"); leg.style.opacity = leg.style.opacity === "0" ? "1" : "0"; legH(); }
      else return;
      e.preventDefault();
    });

    // ---- theme follow -----------------------------------------------------------
    const retheme = () => { P = palette(); colorArcs(); if (!moving) render(); };
    new MutationObserver(retheme).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);

    function fromHash() { const h = location.hash.slice(1); const d = h ? bySlug(h) : root; if (d || !h) zoomTo(d || root, true); }
    addEventListener("hashchange", fromHash);
    colorArcs(); legH(); size();
    if (location.hash && bySlug(location.hash.slice(1))) fromHash();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
