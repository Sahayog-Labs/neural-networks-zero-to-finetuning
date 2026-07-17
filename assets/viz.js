/* ==========================================================================
   ANN Course — visualization primitives beyond x/y curves.
   Loads AFTER course.js (reuses Plot/cssVar) and, where a demo needs live
   math, after nn.js (reuses NN.*). Exposes: Heatmap, NetGraph, Surface3D,
   TensorViz, Timeline, hl — all as plain globals, matching Plot's house
   style. Wrapped in an IIFE (like nn.js) purely so a page can safely write
   `const {Value} = NN;` elsewhere without this file's internal `class`
   declarations fighting over the global lexical scope — see nn.js's own
   note on that exact bug. The public names are still assigned to `window`,
   so call sites read identically to `new Plot(...)`.

   Every primitive:
   - reads colors via getComputedStyle(document.documentElement) so the
     dark/light theme toggle retints it immediately, never a hardcoded hex.
   - uses the same devicePixelRatio fit as Plot (canvas.dataset.baseH cache,
     only reallocate the backing store when the CSS size actually changes).
   - sanitizes degenerate input (empty/NaN/all-equal/huge) so a demo can
     never hang or blank out no matter what a slider drags into it.
   ========================================================================== */
"use strict";

(function () {

  /* ============================== shared utils ============================ */

  function cssVar(name, fallback) {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fallback;
    } catch (e) { return fallback; }
  }

  function clamp01(t) { return t < 0 ? 0 : t > 1 ? 1 : t; }
  function clampNum(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  function hexToRgb(hex, fallback) {
    if (typeof hex !== "string") return fallback;
    let m = /^#([0-9a-f]{3})$/i.exec(hex.trim());
    if (m) { const h = m[1]; return [parseInt(h[0] + h[0], 16), parseInt(h[1] + h[1], 16), parseInt(h[2] + h[2], 16)]; }
    m = /^#([0-9a-f]{6})$/i.exec(hex.trim());
    if (m) { const h = m[1]; return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }
    m = /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i.exec(hex.trim());
    if (m) return [+m[1], +m[2], +m[3]];
    return fallback;
  }
  function parseCssRgb(css) {
    const m = /rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/.exec(css || "");
    return m ? [+m[1], +m[2], +m[3]] : [128, 128, 128];
  }
  function rgbToCss(rgb, alpha) {
    const r = rgb[0] | 0, g = rgb[1] | 0, b = rgb[2] | 0;
    return alpha == null ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${alpha})`;
  }
  function lerpRgb(a, b, t) {
    t = clamp01(t);
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  }
  function contrastText(bgCss) {
    const [r, g, b] = parseCssRgb(bgCss);
    const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return lum > 0.55 ? "#0b0f18" : "#f2f5fb";
  }
  function fmtNum(v) {
    if (!isFinite(v)) return v === Infinity ? "+inf" : v === -Infinity ? "-inf" : "NaN";
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a < 1e-3 || a >= 1e4) return v.toExponential(1);
    // fixed "round to 2 decimal places" loses ALL resolution below ~0.005
    // (0.0014 -> "0.00" -> displayed as "0", which is a silent lie on a
    // log-scale loss axis). Round to ~3 significant figures instead.
    const digits = Math.max(0, 2 - Math.floor(Math.log10(a)));
    const scale = Math.pow(10, digits);
    const r = Math.round(v * scale) / scale;
    return Object.is(r, -0) ? "0" : String(r);
  }

  /* fresh read every call (not cached) so a live theme toggle retints instantly */
  function themeColors() {
    return {
      text: cssVar("--text", "#e4e0f2"),
      textDim: cssVar("--text-dim", "#9a90b5"),
      border: cssVar("--border", "#2c2640"),
      accent: cssVar("--accent", "#b48cff"),
      accent2: cssVar("--accent2", "#45e0b8"),
      good: cssVar("--good", "#5ce6a0"),
      bad: cssVar("--bad", "#ff6b6b"),
      warn: cssVar("--warn", "#ffd166"),
      deep: cssVar("--deep", "#ff8fc4"),
      try_: cssVar("--try", "#ffab5e"),
      bgPanel: cssVar("--bg-panel", "#16121f"),
      bgPanel2: cssVar("--bg-panel2", "#1d1829"),
      bgCode: cssVar("--bg-code", "#0a0810"),
      bgCanvas: cssVar("--bg-canvas", "#0a0810"),
    };
  }

  /* sequential: low -> high, t in [0,1].  diverging: neg -> mid -> pos, t in [-1,1]. */
  function sequentialCmap(lowHex, highHex) {
    const lo = hexToRgb(lowHex, [10, 8, 16]), hi = hexToRgb(highHex, [180, 140, 255]);
    return (t) => isFinite(t) ? rgbToCss(lerpRgb(lo, hi, t)) : "rgb(120,120,120)";
  }
  function divergingCmap(negHex, midHex, posHex) {
    const neg = hexToRgb(negHex, [255, 107, 107]), mid = hexToRgb(midHex, [40, 35, 55]), pos = hexToRgb(posHex, [180, 140, 255]);
    return (t) => {
      if (!isFinite(t)) return "rgb(120,120,120)";
      t = clampNum(t, -1, 1);
      return t < 0 ? rgbToCss(lerpRgb(mid, neg, -t)) : rgbToCss(lerpRgb(mid, pos, t));
    };
  }

  /* devicePixelRatio-correct canvas fit — identical contract/bug-fix to Plot's
     constructor (see course.js comments): read the CSS height ONCE via a
     cached dataset attribute, only touch canvas.width/height when the CSS
     size actually changed (reallocating the backing store on every redraw
     is what kills slider-drag framerate at high DPR). */
  function fitCanvas(canvas, defaultH) {
    const dpr = window.devicePixelRatio || 1;
    if (!canvas.dataset.baseH) canvas.dataset.baseH = canvas.getAttribute("height") || defaultH || 320;
    const w = canvas.clientWidth || canvas.width / dpr || 600;
    const h = +canvas.dataset.baseH;
    const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw; canvas.height = bh;
      canvas.style.height = h + "px";
    }
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { g, W: w, H: h, dpr };
  }

  /* Coerce arbitrary/jagged/NaN-laced input into a rectangular matrix that is
     always safe to index. Missing cells (short rows) become 0; a value that
     was actually NaN/Infinity/non-numeric is PRESERVED as NaN so Heatmap can
     render it as an honest "broken cell" instead of silently lying with 0. */
  function sanitizeMatrix(data) {
    if (!Array.isArray(data) || data.length === 0) return [[0]];
    let cols = 0;
    for (const row of data) if (Array.isArray(row)) cols = Math.max(cols, row.length);
    if (!(cols > 0)) return [[0]];
    return data.map(row => {
      const out = new Array(cols);
      for (let j = 0; j < cols; j++) {
        const v = Array.isArray(row) ? row[j] : undefined;
        out[j] = v === undefined ? 0 : (typeof v === "number" && isFinite(v) ? v : NaN);
      }
      return out;
    });
  }
  function sanitizeShape(s) {
    if (!Array.isArray(s) || s.length === 0) return [1, 1];
    return s.map(v => (typeof v === "number" && isFinite(v) && v > 0) ? Math.round(v) : 1);
  }
  function range(n) { return Array.from({ length: Math.max(0, n | 0) }, (_, i) => i); }
  function avg4(a, b, c, d) {
    let s = 0, n = 0;
    for (const v of [a, b, c, d]) if (isFinite(v)) { s += v; n++; }
    return n ? s / n : NaN;
  }

  /* Standard marching-squares (bit order TL=1,TR=2,BR=4,BL=8) over a
     (res+1)x(res+1) scalar grid, in CELL-INDEX space (caller scales). Used
     for the Surface3D contour's isolines; skips any cell touching a NaN
     sample instead of drawing a line through a hole. */
  function marchingSquares(grid, res, thresh, emit) {
    const lerp = (a, b) => { const d = b - a; return Math.abs(d) < 1e-12 ? 0.5 : clampNum((thresh - a) / d, 0, 1); };
    const SEGS = {
      1: [["W", "N"]], 2: [["N", "E"]], 3: [["W", "E"]], 4: [["E", "S"]],
      5: [["W", "N"], ["E", "S"]], 6: [["N", "S"]], 7: [["W", "S"]], 8: [["W", "S"]],
      9: [["N", "S"]], 10: [["N", "E"], ["S", "W"]], 11: [["E", "S"]], 12: [["W", "E"]],
      13: [["N", "E"]], 14: [["W", "N"]]
    };
    for (let i = 0; i < res; i++) {
      for (let j = 0; j < res; j++) {
        const TL = grid[i][j], TR = grid[i][j + 1], BR = grid[i + 1][j + 1], BL = grid[i + 1][j];
        if (!isFinite(TL) || !isFinite(TR) || !isFinite(BR) || !isFinite(BL)) continue;
        let c = 0;
        if (TL > thresh) c |= 1; if (TR > thresh) c |= 2; if (BR > thresh) c |= 4; if (BL > thresh) c |= 8;
        const pairs = SEGS[c];
        if (!pairs) continue;
        const P = {
          N: [j + lerp(TL, TR), i], E: [j + 1, i + lerp(TR, BR)],
          S: [j + lerp(BL, BR), i + 1], W: [j, i + lerp(TL, BL)]
        };
        for (const [a, b] of pairs) emit(P[a][0], P[a][1], P[b][0], P[b][1]);
      }
    }
  }

  /* ================================ Heatmap ================================
     new Heatmap(canvas, {
       data: number[][],                 // rows x cols, jagged/NaN-safe
       rowLabels, colLabels: string[],
       cellText: false | true | (v,i,j)=>string,
       colormap: 'sequential' | 'diverging' | (t)=>cssColor,
       domain: [min,max] | null,         // auto if omitted
       colorbar: true,
       title: string,
       hover: true,                      // click/hover-to-inspect a cell
       onHover: (i,j,v) => {}  | (null,null,null) on leave,
     })
     .setData(data)  .draw()  .destroy()
     ========================================================================== */
  class Heatmap {
    constructor(canvas, opts) {
      this.c = canvas;
      this.o = Object.assign({
        data: [[0]], rowLabels: null, colLabels: null, cellText: false,
        colormap: "sequential", domain: null, colorbar: true, title: null,
        pad: null, minCellPx: 22, hover: true, onHover: null, height: 320,
      }, opts);
      this.data = sanitizeMatrix(this.o.data);
      this._hover = null;
      this._geom = null;
      if (this.o.hover) this._bindHover();
      this.draw();
    }
    setData(data) { this.data = sanitizeMatrix(data); this.draw(); }
    destroy() {
      if (this._onMove) this.c.removeEventListener("mousemove", this._onMove);
      if (this._onLeave) this.c.removeEventListener("mouseleave", this._onLeave);
    }
    _bindHover() {
      this._onMove = (e) => {
        const rect = this.c.getBoundingClientRect();
        const cell = this._cellAt(e.clientX - rect.left, e.clientY - rect.top);
        this._hover = cell; this.draw();
        if (this.o.onHover) this.o.onHover(cell ? cell.i : null, cell ? cell.j : null, cell ? cell.v : null);
      };
      this._onLeave = () => { this._hover = null; this.draw(); if (this.o.onHover) this.o.onHover(null, null, null); };
      this.c.addEventListener("mousemove", this._onMove);
      this.c.addEventListener("mouseleave", this._onLeave);
    }
    _cellAt(px, py) {
      const g = this._geom; if (!g) return null;
      if (px < g.x0 || px > g.x1 || py < g.y0 || py > g.y1) return null;
      const j = Math.min(this.data[0].length - 1, Math.floor((px - g.x0) / g.cw));
      const i = Math.min(this.data.length - 1, Math.floor((py - g.y0) / g.rh));
      if (i < 0 || j < 0) return null;
      return { i, j, v: this.data[i][j] };
    }
    draw() {
      const th = themeColors();
      const rows = this.data.length, cols = this.data[0].length;
      const rowLabels = this.o.rowLabels, colLabels = this.o.colLabels;
      const hasRowLabels = !!rowLabels, hasColLabels = !!colLabels;
      const pad = this.o.pad || {
        l: hasRowLabels ? 84 : 20,
        r: this.o.colorbar ? 58 : 16,
        t: (this.o.title ? 24 : 6) + (hasColLabels ? 34 : 0),
        b: 16
      };
      const { g, W, H } = fitCanvas(this.c, this.o.height);
      g.clearRect(0, 0, W, H);
      const x0 = pad.l, y0 = pad.t, x1 = W - pad.r, y1 = H - pad.b;
      const gw = Math.max(1, x1 - x0), gh = Math.max(1, y1 - y0);
      const cw = gw / cols, rh = gh / rows;
      this._geom = { x0, y0, x1, y1, cw, rh };

      let vals = [];
      for (const row of this.data) for (const v of row) if (isFinite(v)) vals.push(v);
      let dom = this.o.domain;
      const isDiverging = this.o.colormap === "diverging";
      if (!dom) {
        if (vals.length === 0) dom = [0, 1];
        else {
          const mn = Math.min(...vals), mx = Math.max(...vals);
          dom = isDiverging ? [-Math.max(Math.abs(mn), Math.abs(mx), 1e-9), Math.max(Math.abs(mn), Math.abs(mx), 1e-9)]
                             : (mx > mn ? [mn, mx] : [mn - 0.5, mx + 0.5]);
        }
      }
      const [dmin, dmax] = dom;
      const cmapFn = typeof this.o.colormap === "function" ? this.o.colormap
        : isDiverging ? divergingCmap(th.bad, th.bgPanel2, th.accent)
        : sequentialCmap(th.bgCode, th.accent);
      const halfSpan = Math.max(Math.abs(dmin), Math.abs(dmax), 1e-9);
      const colorFor = (v) => {
        if (!isFinite(v)) return null; // NaN -> hatched cell
        if (isDiverging) return cmapFn(clampNum(v / halfSpan, -1, 1));
        return cmapFn(clamp01(dmax > dmin ? (v - dmin) / (dmax - dmin) : 0.5));
      };

      const showText = this.o.cellText && cw >= this.o.minCellPx && rh >= this.o.minCellPx;
      g.textAlign = "center"; g.textBaseline = "middle";
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const v = this.data[i][j];
          const x = x0 + j * cw, y = y0 + i * rh;
          const col = colorFor(v);
          if (col == null) {
            g.save(); g.fillStyle = th.bgPanel; g.fillRect(x, y, cw, rh);
            g.strokeStyle = th.bad; g.lineWidth = 1;
            g.beginPath(); g.moveTo(x + 2, y + 2); g.lineTo(x + cw - 2, y + rh - 2);
            g.moveTo(x + cw - 2, y + 2); g.lineTo(x + 2, y + rh - 2); g.stroke();
            g.restore();
          } else {
            g.fillStyle = col; g.fillRect(x, y, cw, rh);
          }
          if (this._hover && this._hover.i === i && this._hover.j === j) {
            g.save(); g.strokeStyle = th.text; g.lineWidth = 2; g.strokeRect(x + 1, y + 1, cw - 2, rh - 2); g.restore();
          }
          if (showText) {
            const txt = typeof this.o.cellText === "function" ? this.o.cellText(v, i, j) : (isFinite(v) ? v.toFixed(2) : "NaN");
            g.fillStyle = isFinite(v) ? contrastText(col) : th.bad;
            g.font = "10px " + cssVar("--mono", "monospace");
            g.fillText(txt, x + cw / 2, y + rh / 2);
          }
        }
      }
      g.strokeStyle = th.border; g.lineWidth = 1;
      for (let i = 0; i <= rows; i++) { const y = y0 + i * rh; g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke(); }
      for (let j = 0; j <= cols; j++) { const x = x0 + j * cw; g.beginPath(); g.moveTo(x, y0); g.lineTo(x, y1); g.stroke(); }

      g.fillStyle = th.textDim; g.font = "11px system-ui";
      if (hasColLabels) {
        g.save(); g.textAlign = "left"; g.textBaseline = "middle";
        for (let j = 0; j < cols; j++) {
          g.save(); g.translate(x0 + j * cw + cw / 2, y0 - 6); g.rotate(-Math.PI / 4);
          g.fillText(String(colLabels[j] ?? ""), 0, 0);
          g.restore();
        }
        g.restore();
      }
      if (hasRowLabels) {
        g.save(); g.textAlign = "right"; g.textBaseline = "middle";
        for (let i = 0; i < rows; i++) g.fillText(String(rowLabels[i] ?? ""), x0 - 8, y0 + i * rh + rh / 2);
        g.restore();
      }
      if (this.o.title) { g.save(); g.fillStyle = th.text; g.font = "bold 12px system-ui"; g.textAlign = "left"; g.fillText(this.o.title, 4, 14); g.restore(); }

      if (this.o.colorbar) {
        const bx = W - pad.r + 16, by = y0, bh = y1 - y0, bw = 12, steps = 40;
        for (let k = 0; k < steps; k++) {
          const t = k / (steps - 1);
          g.fillStyle = isDiverging ? cmapFn(t * 2 - 1) : cmapFn(t);
          g.fillRect(bx, by + bh * (1 - t), bw, bh / steps + 1);
        }
        g.strokeStyle = th.border; g.strokeRect(bx, by, bw, bh);
        g.fillStyle = th.textDim; g.font = "10px monospace"; g.textAlign = "left"; g.textBaseline = "middle";
        g.fillText(fmtNum(dmax), bx + bw + 4, by + 4);
        g.fillText(fmtNum(dmin), bx + bw + 4, by + bh - 4);
        if (isDiverging) g.fillText("0", bx + bw + 4, by + bh / 2);
      }

      if (this._hover) {
        const { i, j, v } = this._hover;
        const label = `[${i},${j}] = ${isFinite(v) ? v.toPrecision(4) : "NaN"}`;
        g.save(); g.font = "11px monospace";
        const tw = g.measureText(label).width + 12;
        let tx = clampNum(x0 + j * cw + cw / 2 - tw / 2, 2, W - tw - 2);
        let ty = y0 + i * rh - 8; if (ty < 14) ty = y0 + i * rh + rh + 16;
        g.fillStyle = th.bgPanel; g.strokeStyle = th.border;
        g.fillRect(tx, ty - 14, tw, 18); g.strokeRect(tx, ty - 14, tw, 18);
        g.fillStyle = th.text; g.textAlign = "left"; g.textBaseline = "middle";
        g.fillText(label, tx + 6, ty - 5);
        g.restore();
      }
    }
  }

  /* ================================ NetGraph ================================
     new NetGraph(canvas, {
       layers: number[],                 // e.g. [2,2,1]
       weights: Mat[]|number[][][],      // weights[l] shape (layers[l+1] x layers[l])
       activations: number[][],          // activations[l] length layers[l]
       labels: string[][],               // optional per-node labels
       maxNodes: 24,                     // degrade beyond this per layer
     })
     .setActivations(a) .setWeights(w) .setLayers(sizes) .draw()
     .animateForward(activations, {duration,onDone})   // signal flows left->right
     .animateBackward(grads, {duration,onDone})         // gradients flow right->left
     .stop()
     ========================================================================== */
  class NetGraph {
    constructor(canvas, opts) {
      this.c = canvas;
      this.o = Object.assign({
        layers: [2, 2, 1], weights: null, activations: null, labels: null,
        maxNodes: 24, nodeR: null, height: 300, edgeAlpha: 1,
      }, opts);
      this._animState = null; this._raf = null;
      this.draw();
    }
    setActivations(a) { this.o.activations = a; this.draw(); }
    setWeights(w) { this.o.weights = w; this.draw(); }
    setLayers(sizes) { this.o.layers = sizes; this.draw(); }
    stop() { if (this._raf != null) { cancelAnimationFrame(this._raf); this._raf = null; } this._animState = null; }

    _layout(W, H) {
      const L = this.o.layers.length, padX = 50, padY = 26;
      const xs = []; for (let l = 0; l < L; l++) xs.push(L === 1 ? W / 2 : padX + (W - 2 * padX) * l / (L - 1));
      const pos = [];
      for (let l = 0; l < L; l++) {
        const n = Math.max(0, this.o.layers[l] | 0);
        const show = Math.min(n, this.o.maxNodes);
        const arr = []; const usableH = H - 2 * padY;
        for (let i = 0; i < show; i++) arr.push({ x: xs[l], y: show === 1 ? H / 2 : padY + usableH * i / (show - 1), idx: i });
        arr._n = n; arr._show = show; arr._compressed = n > show;
        pos.push(arr);
      }
      return { pos, xs };
    }

    _mat(M, l) { return this.o.weights && this.o.weights[l] ? this.o.weights[l] : null; }
    _row(M, i) {
      if (!M) return null;
      if (typeof M.row === "function") return M.row(i).toArray ? M.row(i).toArray()[0] : M.row(i); // Mat support
      return M[i];
    }

    _scheduleLayers(dir, L, duration) {
      const nSeg = Math.max(1, L - 1), seg = duration / nSeg;
      const order = dir === "fwd" ? range(L) : range(L).reverse();
      const sched = {}; sched[order[0]] = { start: -1, end: 0 };
      for (let k = 1; k < order.length; k++) sched[order[k]] = { start: (k - 1) * seg, end: k * seg };
      return { sched, order, seg };
    }
    _reveal(l) {
      const st = this._animState; if (!st) return 1;
      const sc = st.sched[l]; if (!sc) return 1;
      if (sc.start < 0) return 1;
      return clamp01((st.elapsed - sc.start) / (sc.end - sc.start));
    }
    _wavePos(xs) {
      const st = this._animState; if (!st) return null;
      const nSeg = st.order.length - 1; if (nSeg <= 0) return null;
      let segIdx = clampNum(Math.floor(st.elapsed / st.seg), 0, nSeg - 1);
      const p = clamp01((st.elapsed - segIdx * st.seg) / st.seg);
      const lFrom = st.order[segIdx], lTo = st.order[segIdx + 1];
      return { x: xs[lFrom] + (xs[lTo] - xs[lFrom]) * p, dir: st.dir };
    }

    _runAnim(dir, values, opts) {
      this.stop();
      const L = this.o.layers.length;
      if (values) this.o.activations = values;
      if (L < 2) { this.draw(); if (opts.onDone) opts.onDone(); return; }
      const duration = opts.duration ?? 1100;
      const { sched, order, seg } = this._scheduleLayers(dir, L, duration);
      this._animState = { dir, sched, order, seg, duration, start: performance.now(), elapsed: 0 };
      const step = () => {
        const elapsed = Math.min(performance.now() - this._animState.start, duration);
        this._animState.elapsed = elapsed;
        this.draw();
        if (elapsed < duration) this._raf = requestAnimationFrame(step);
        else { this._raf = null; this._animState = null; this.draw(); if (opts.onDone) opts.onDone(); }
      };
      this._raf = requestAnimationFrame(step);
    }
    animateForward(activations, opts = {}) { this._runAnim("fwd", activations, opts); }
    animateBackward(grads, opts = {}) { this._runAnim("bwd", grads, opts); }

    draw() {
      const th = themeColors();
      const { g, W, H } = fitCanvas(this.c, this.o.height);
      g.clearRect(0, 0, W, H);
      const layers = this.o.layers;
      const { pos, xs } = this._layout(W, H);
      const maxShown = Math.max(1, ...layers.map(n => Math.min(n, this.o.maxNodes)));
      const nodeR = this.o.nodeR || clampNum((H / maxShown) * 0.3, 5, 15);

      let maxAbsW = 1e-9;
      if (this.o.weights) for (let l = 0; l < layers.length - 1; l++) {
        const M = this._mat(null, l); if (!M) continue;
        for (let bj = 0; bj < pos[l + 1]._show; bj++) { const row = this._row(M, bj); if (!row) continue; for (let ai = 0; ai < pos[l]._show; ai++) if (isFinite(row[ai])) maxAbsW = Math.max(maxAbsW, Math.abs(row[ai])); }
      }
      const edgeCmap = divergingCmap(th.bad, th.textDim, th.accent);

      for (let l = 0; l < pos.length - 1; l++) {
        const A = pos[l], B = pos[l + 1], M = this._mat(null, l);
        const nA = A._show, nB = B._show, dense = nA * nB;
        const skipDetail = dense > 900;
        const dir = this._animState && this._animState.dir;
        const ep = this._reveal(dir === "bwd" ? l : l + 1);
        for (let bj = 0; bj < nB; bj++) {
          const row = M ? this._row(M, bj) : null;
          for (let ai = 0; ai < nA; ai++) {
            const w = row && isFinite(row[ai]) ? row[ai] : 0;
            const alpha = (skipDetail ? 0.06 : Math.min(0.9, 0.12 + 0.75 * Math.min(1, Math.abs(w) / maxAbsW))) * (this.o.edgeAlpha ?? 1) * ep;
            g.save();
            g.strokeStyle = edgeCmap(maxAbsW > 0 ? w / maxAbsW : 0);
            g.globalAlpha = alpha;
            g.lineWidth = skipDetail ? 0.6 : clampNum(0.5 + 3.5 * Math.abs(w) / maxAbsW, 0.5, 4);
            g.beginPath(); g.moveTo(A[ai].x, A[ai].y); g.lineTo(B[bj].x, B[bj].y); g.stroke();
            g.restore();
          }
        }
        if (A._compressed || B._compressed || dense > 900) {
          g.save(); g.fillStyle = th.textDim; g.font = "10px monospace"; g.textAlign = "center";
          g.fillText(`${A._n}×${B._n} = ${A._n * B._n} connections`, (xs[l] + xs[l + 1]) / 2, H - 4);
          g.restore();
        }
      }

      const wave = this._wavePos(xs);
      if (wave) {
        g.save();
        const grad = g.createLinearGradient(wave.x - 10, 0, wave.x + 10, 0);
        const c = wave.dir === "fwd" ? th.accent : th.bad;
        const rgb = hexToRgb(c, [180, 140, 255]);
        grad.addColorStop(0, rgbToCss(rgb, 0)); grad.addColorStop(0.5, rgbToCss(rgb, 0.5)); grad.addColorStop(1, rgbToCss(rgb, 0));
        g.fillStyle = grad; g.fillRect(wave.x - 10, 0, 20, H);
        g.restore();
      }

      const acts = this.o.activations;
      let actAbs = 1e-9;
      if (acts) for (const layer of acts) for (const v of (layer || [])) if (isFinite(v)) actAbs = Math.max(actAbs, Math.abs(v));
      const nodeCmap = divergingCmap(th.bad, th.bgPanel2, th.accent);
      for (let l = 0; l < pos.length; l++) {
        const arr = pos[l], layerActs = acts && acts[l];
        const revealT = this._reveal(l);
        for (let i = 0; i < arr.length; i++) {
          const p = arr[i];
          const v = layerActs && isFinite(layerActs[i]) ? layerActs[i] : null;
          g.beginPath(); g.arc(p.x, p.y, nodeR, 0, 2 * Math.PI);
          const fill = v == null ? th.bgPanel2 : nodeCmap(clamp01(revealT) * (actAbs > 0 ? v / actAbs : 0));
          g.fillStyle = fill; g.fill();
          g.lineWidth = 1.4; g.strokeStyle = th.border; g.stroke();
          if (nodeR >= 9 && v != null && revealT > 0.35) {
            g.fillStyle = contrastText(fill);
            g.font = "9px monospace"; g.textAlign = "center"; g.textBaseline = "middle";
            g.fillText(v.toFixed(2), p.x, p.y);
          }
          const lbl = this.o.labels && this.o.labels[l] && this.o.labels[l][i];
          if (lbl) { g.save(); g.fillStyle = th.textDim; g.font = "10px system-ui"; g.textAlign = "center"; g.fillText(lbl, p.x, p.y - nodeR - 6); g.restore(); }
        }
        if (arr._compressed) {
          g.save(); g.fillStyle = th.textDim; g.font = "10px system-ui"; g.textAlign = "center";
          g.fillText(`(${arr._n} nodes, showing ${arr._show})`, arr[0] ? arr[0].x : xs[l], H - 4);
          g.restore();
        }
      }
    }
  }

  /* ================================ Surface3D ===============================
     Default mode is a 2-D filled contour (brief-training.md §3/DEMO1 argues
     for this explicitly: a 3-D loss surface is a shadow of a shadow — a
     filled contour + trajectory teaches the same gradient-descent geometry
     without pretending to show more dimensions than it does). A lightweight
     rotatable wireframe mode is included for the rare spot where the "this
     is a literal surface" picture earns its keep, but contour is the
     documented, recommended default.

     new Surface3D(canvas, {
       fn: (x,y) => z, xmin,xmax,ymin,ymax,
       mode: 'contour' | 'wireframe',    // default 'contour'
       resolution: 48, levels: 10,       // contour mode
       wireRes: 22, az, el,              // wireframe mode (drag to rotate)
       colormap: (t in [0,1]) => cssColor,
       onClick: (x,y) => {},             // contour mode: click to set a point
     })
     .setFn(fn) .setDomain(xmin,xmax,ymin,ymax)
     .setTrajectory(pts) .addPoint(x,y) .clearTrajectory()
     .setViewpoint(az,el) .draw()
     ========================================================================== */
  class Surface3D {
    constructor(canvas, opts) {
      this.c = canvas;
      this.o = Object.assign({
        fn: () => 0, xmin: -3, xmax: 3, ymin: -3, ymax: 3,
        resolution: 48, mode: "contour", levels: 10, colormap: null,
        height: 340, onClick: null, wireRes: 22, az: -0.9, el: 0.55,
      }, opts);
      const s = this.o;
      if (!isFinite(s.xmin)) s.xmin = -3;
      if (!isFinite(s.xmax) || s.xmax <= s.xmin) s.xmax = s.xmin + 6;
      if (!isFinite(s.ymin)) s.ymin = -3;
      if (!isFinite(s.ymax) || s.ymax <= s.ymin) s.ymax = s.ymin + 6;
      this.trajectory = [];
      this._bg = null; this._geom = null;
      this._bindEvents();
      this.recompute();
    }
    setFn(fn) { this.o.fn = fn; this.recompute(); }
    setDomain(xmin, xmax, ymin, ymax) { Object.assign(this.o, { xmin, xmax, ymin, ymax }); this.recompute(); }
    setTrajectory(pts) { this.trajectory = (pts || []).filter(p => isFinite(p[0]) && isFinite(p[1])); this.draw(); }
    addPoint(x, y) { if (isFinite(x) && isFinite(y)) { this.trajectory.push([x, y]); this.draw(); } }
    clearTrajectory() { this.trajectory = []; this.draw(); }
    setViewpoint(az, el) { this.o.az = az; this.o.el = clampNum(el, -1.4, 1.4); this.draw(); }

    recompute() {
      const res = this.o.mode === "wireframe" ? this.o.wireRes : this.o.resolution;
      const { xmin, xmax, ymin, ymax, fn } = this.o;
      const grid = []; let zmin = Infinity, zmax = -Infinity;
      for (let i = 0; i <= res; i++) {
        const row = []; const y = ymax - (ymax - ymin) * i / res; // row 0 = ymax (screen-up = math-up)
        for (let j = 0; j <= res; j++) {
          const x = xmin + (xmax - xmin) * j / res;
          let z; try { z = fn(x, y); } catch (e) { z = NaN; }
          if (typeof z !== "number" || !isFinite(z)) z = NaN;
          row.push(z);
          if (isFinite(z)) { if (z < zmin) zmin = z; if (z > zmax) zmax = z; }
        }
        grid.push(row);
      }
      if (!isFinite(zmin) || !isFinite(zmax)) { zmin = 0; zmax = 1; }
      if (zmax <= zmin) zmax = zmin + 1e-6;
      this._grid = grid; this._res = res; this._zmin = zmin; this._zmax = zmax;
      this._bg = null;
      this.draw();
    }

    _bindEvents() {
      let dragging = false, lastX = 0, lastY = 0;
      this.c.addEventListener("mousedown", (e) => { if (this.o.mode !== "wireframe") return; dragging = true; lastX = e.clientX; lastY = e.clientY; });
      window.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const dx = e.clientX - lastX, dy = e.clientY - lastY; lastX = e.clientX; lastY = e.clientY;
        this.o.az += dx * 0.008; this.o.el = clampNum(this.o.el - dy * 0.008, -1.4, 1.4);
        this.draw();
      });
      window.addEventListener("mouseup", () => { dragging = false; });
      this.c.addEventListener("click", (e) => {
        if (this.o.mode === "wireframe" || !this._geom) return;
        const rect = this.c.getBoundingClientRect();
        const px = e.clientX - rect.left, py = e.clientY - rect.top;
        const { x0, y0, x1, y1 } = this._geom;
        if (px < x0 || px > x1 || py < y0 || py > y1) return;
        const x = this.o.xmin + (this.o.xmax - this.o.xmin) * (px - x0) / (x1 - x0);
        const y = this.o.ymax - (this.o.ymax - this.o.ymin) * (py - y0) / (y1 - y0);
        if (this.o.onClick) this.o.onClick(x, y);
      });
    }

    draw() {
      const th = themeColors();
      const { g, W, H } = fitCanvas(this.c, this.o.height);
      g.clearRect(0, 0, W, H);
      if (this.o.mode === "wireframe") this._drawWire(g, W, H, th);
      else this._drawContour(g, W, H, th);
    }

    _drawContour(g, W, H, th) {
      const pad = { l: 44, r: 20, t: 14, b: 34 };
      const x0 = pad.l, y0 = pad.t, x1 = W - pad.r, y1 = H - pad.b;
      this._geom = { x0, y0, x1, y1 };
      const res = this._res, grid = this._grid, zmin = this._zmin, zmax = this._zmax;
      const cmap = this.o.colormap || sequentialCmap(th.bgCode, th.accent);
      const gw = Math.max(1, x1 - x0), gh = Math.max(1, y1 - y0);
      const bw = Math.max(1, Math.round(gw)), bh = Math.max(1, Math.round(gh));
      if (!this._bg || this._bg.w !== bw || this._bg.h !== bh || this._bg.grid !== grid) {
        const off = document.createElement("canvas"); off.width = bw; off.height = bh;
        const og = off.getContext("2d");
        const cw = bw / res, ch = bh / res;
        for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) {
          const z = avg4(grid[i][j], grid[i][j + 1], grid[i + 1][j], grid[i + 1][j + 1]);
          og.fillStyle = isFinite(z) ? cmap(clamp01((z - zmin) / (zmax - zmin))) : th.bgPanel;
          og.fillRect(j * cw, i * ch, cw + 1, ch + 1);
        }
        og.strokeStyle = "rgba(0,0,0,0.28)"; og.lineWidth = 1;
        for (let lv = 1; lv < this.o.levels; lv++) {
          const thresh = zmin + (zmax - zmin) * lv / this.o.levels;
          marchingSquares(grid, res, thresh, (ax, ay, bx, by) => { og.beginPath(); og.moveTo(ax * cw, ay * ch); og.lineTo(bx * cw, by * ch); og.stroke(); });
        }
        this._bg = { canvas: off, w: bw, h: bh, grid };
      }
      g.drawImage(this._bg.canvas, x0, y0, gw, gh);
      g.strokeStyle = th.border; g.strokeRect(x0, y0, gw, gh);

      g.fillStyle = th.textDim; g.font = "10px monospace"; g.textAlign = "center"; g.textBaseline = "top";
      g.fillText(fmtNum(this.o.xmin), x0, y1 + 4); g.fillText(fmtNum(this.o.xmax), x1, y1 + 4);
      g.save(); g.textAlign = "right"; g.textBaseline = "middle";
      g.fillText(fmtNum(this.o.ymax), x0 - 6, y0); g.fillText(fmtNum(this.o.ymin), x0 - 6, y1);
      g.restore();

      const tx = (x) => x0 + (x - this.o.xmin) / (this.o.xmax - this.o.xmin) * gw;
      const ty = (y) => y1 - (y - this.o.ymin) / (this.o.ymax - this.o.ymin) * gh;
      if (this.trajectory.length) {
        g.save(); g.strokeStyle = th.text; g.lineWidth = 1.6;
        g.beginPath();
        this.trajectory.forEach((p, idx) => { const px = tx(p[0]), py = ty(p[1]); if (idx === 0) g.moveTo(px, py); else g.lineTo(px, py); });
        g.stroke();
        this.trajectory.forEach((p, idx) => {
          const px = tx(p[0]), py = ty(p[1]), last = idx === this.trajectory.length - 1;
          g.beginPath(); g.arc(px, py, last ? 4.5 : 2, 0, 2 * Math.PI);
          g.fillStyle = last ? th.accent2 : th.text; g.fill();
        });
        g.restore();
      }
    }

    _drawWire(g, W, H, th) {
      const pad = 10, grid = this._grid, res = this._res, zmin = this._zmin, zmax = this._zmax;
      const { xmin, xmax, ymin, ymax, az, el } = this.o;
      const cmap = this.o.colormap || sequentialCmap(th.bgCode, th.accent2);
      const proj = (x, y, z) => {
        const cx = (x - (xmin + xmax) / 2) / ((xmax - xmin) / 2 || 1);
        const cy = (y - (ymin + ymax) / 2) / ((ymax - ymin) / 2 || 1);
        const cz = zmax > zmin ? (z - (zmin + zmax) / 2) / ((zmax - zmin) / 2) : 0;
        const rx = cx * Math.cos(az) - cy * Math.sin(az);
        const ry0 = cx * Math.sin(az) + cy * Math.cos(az);
        const sy = ry0 * Math.cos(el) - cz * Math.sin(el);
        const depth = ry0 * Math.sin(el) + cz * Math.cos(el);
        return { x: rx, y: sy - depth * 0.15 };
      };
      const pts = []; let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (let i = 0; i <= res; i++) {
        const row = [];
        for (let j = 0; j <= res; j++) {
          const x = xmin + (xmax - xmin) * j / res, y = ymax - (ymax - ymin) * i / res, z = grid[i][j];
          const P = proj(x, y, isFinite(z) ? z : zmin);
          row.push(P);
          if (P.x < minX) minX = P.x; if (P.x > maxX) maxX = P.x; if (P.y < minY) minY = P.y; if (P.y > maxY) maxY = P.y;
        }
        pts.push(row);
      }
      const scale = Math.min((W - 2 * pad) / Math.max(1e-6, maxX - minX), (H - 2 * pad) / Math.max(1e-6, maxY - minY));
      const ox = W / 2 - scale * (minX + maxX) / 2, oy = H / 2 - scale * (minY + maxY) / 2;
      const sx = (P) => ox + scale * P.x, sy = (P) => oy + scale * P.y;
      g.lineWidth = 1;
      for (let i = 0; i <= res; i++) {
        g.beginPath();
        for (let j = 0; j <= res; j++) {
          const P = pts[i][j], z = grid[i][j];
          g.strokeStyle = isFinite(z) ? cmap(clamp01((z - zmin) / (zmax - zmin))) : th.bgPanel;
          if (j === 0) g.moveTo(sx(P), sy(P)); else { g.lineTo(sx(P), sy(P)); g.stroke(); g.beginPath(); g.moveTo(sx(P), sy(P)); }
        }
      }
      for (let j = 0; j <= res; j++) {
        g.beginPath();
        for (let i = 0; i <= res; i++) {
          const P = pts[i][j], z = grid[i][j];
          g.strokeStyle = isFinite(z) ? cmap(clamp01((z - zmin) / (zmax - zmin))) : th.bgPanel;
          if (i === 0) g.moveTo(sx(P), sy(P)); else { g.lineTo(sx(P), sy(P)); g.stroke(); g.beginPath(); g.moveTo(sx(P), sy(P)); }
        }
      }
      if (this.trajectory.length) {
        g.save(); g.strokeStyle = th.text; g.lineWidth = 2; g.beginPath();
        this.trajectory.forEach((p, idx) => {
          let z; try { z = this.o.fn(p[0], p[1]); } catch (e) { z = zmin; }
          const P = proj(p[0], p[1], isFinite(z) ? z : zmin), X = sx(P), Y = sy(P);
          if (idx === 0) g.moveTo(X, Y); else g.lineTo(X, Y);
        });
        g.stroke(); g.restore();
      }
      g.fillStyle = th.textDim; g.font = "10px system-ui"; g.textAlign = "left";
      g.fillText("drag to rotate", 8, H - 8);
    }
  }

  /* ================================ TensorViz ===============================
     new TensorViz(canvas, {
       mode: 'matmul' | 'shape',
       A: {shape:[r,k], label:'A'}, B: {shape:[k,c], label:'B'}, labels:{C:'Y'},
       shapes: [{shape:[...], label, note}],   // 'shape' mode: chain of tensors
     })
     .setMatmul(A,B) .setShapes(shapes) .draw()
     .animateMatmul({A,B,msPerStep,onDone})   // sweeps row(A) x col(B) -> C
     .stop()
     ========================================================================== */
  class TensorViz {
    constructor(canvas, opts) {
      this.c = canvas;
      this.o = Object.assign({
        mode: "matmul",
        A: { shape: [3, 4], label: "A" }, B: { shape: [4, 2], label: "B" },
        labels: null, shapes: null, height: 220, cellMax: 34,
      }, opts);
      this._anim = null; this._raf = null; this._timer = null;
      this.draw();
    }
    setMatmul(A, B) { this.o.mode = "matmul"; this.o.A = A; this.o.B = B; this.stop(); this.draw(); }
    setShapes(shapes) { this.o.mode = "shape"; this.o.shapes = shapes; this.stop(); this.draw(); }
    stop() {
      if (this._timer != null) { clearTimeout(this._timer); this._timer = null; }
      if (this._raf != null) { cancelAnimationFrame(this._raf); this._raf = null; }
      this._anim = null;
    }
    draw() {
      const th = themeColors();
      const { g, W, H } = fitCanvas(this.c, this.o.height);
      g.clearRect(0, 0, W, H);
      if (this.o.mode === "shape") this._drawShapes(g, W, H, th);
      else this._drawMatmul(g, W, H, th);
    }
    _boxSize(rc, maxCell, capPx) {
      const rr = Math.max(1, rc[0] | 0), cc = Math.max(1, rc[1] | 0);
      return { rr, cc, cell: clampNum(capPx / Math.max(rr, cc), 4, maxCell) };
    }
    _drawShapes(g, W, H, th) {
      const shapes = (this.o.shapes && this.o.shapes.length) ? this.o.shapes : [{ shape: [1, 1], label: "(empty)" }];
      let x = 16; const midY = H / 2;
      shapes.forEach((s, idx) => {
        const shp = sanitizeShape(s.shape);
        const rc = shp.length >= 2 ? [shp[shp.length - 2], shp[shp.length - 1]] : [1, shp[0] || 1];
        const { rr, cc, cell } = this._boxSize(rc, this.o.cellMax, Math.min(W * 0.5, H - 70));
        const w = cc * cell, h = rr * cell, y = midY - h / 2;
        const extra = shp.length > 2 ? shp.slice(0, shp.length - 2) : [];
        const off = Math.min(10, 24 / Math.max(1, Math.min(extra.length, 3) || 1));
        for (let e = Math.min(extra.length, 3) - 1; e >= 0; e--) {
          g.save(); g.strokeStyle = th.border; g.globalAlpha = 0.5; g.lineWidth = 1;
          g.strokeRect(x + (e + 1) * off, y - (e + 1) * off, w, h);
          g.restore();
        }
        g.fillStyle = th.bgPanel2; g.strokeStyle = th.accent; g.lineWidth = 1.5;
        g.fillRect(x, y, w, h); g.strokeRect(x, y, w, h);
        if (cell >= 6) {
          g.strokeStyle = th.border; g.lineWidth = 0.5;
          for (let i = 1; i < rr; i++) { g.beginPath(); g.moveTo(x, y + i * cell); g.lineTo(x + w, y + i * cell); g.stroke(); }
          for (let j = 1; j < cc; j++) { g.beginPath(); g.moveTo(x + j * cell, y); g.lineTo(x + j * cell, y + h); g.stroke(); }
        }
        const stackH = extra.length ? off * Math.min(extra.length, 3) : 0;
        g.fillStyle = th.text; g.textAlign = "center"; g.font = "bold 12px system-ui";
        g.fillText(s.label || `T${idx}`, x + w / 2, y - 16 - stackH);
        g.fillStyle = th.textDim; g.font = "10px monospace";
        g.fillText("(" + shp.join(" × ") + ")", x + w / 2, y + h + 14);
        if (s.note) { g.fillStyle = th.try_; g.font = "10px system-ui"; g.fillText(s.note, x + w / 2, y + h + 28); }
        const adv = Math.max(w + stackH, 40) + 46;
        x += adv;
        if (idx < shapes.length - 1) { g.fillStyle = th.textDim; g.font = "16px system-ui"; g.textAlign = "center"; g.fillText("→", x - 30, midY); }
      });
    }
    _drawMatmul(g, W, H, th) {
      const A = this.o.A || {}, B = this.o.B || {};
      const shA = sanitizeShape(A.shape), shB = sanitizeShape(B.shape);
      const [r, k] = shA, [k2, c] = shB;
      const ok = k === k2;
      const areaH = H - 60;
      const cell = clampNum(Math.min(areaH / Math.max(r, k, 1), areaH / Math.max(k2, c, 1)), 4, this.o.cellMax);
      const wA = k * cell, hA = r * cell, wB = c * cell, hB = k2 * cell, wC = c * cell, hC = r * cell;
      const gap = 46;
      const totalW = wA + gap + wB + gap + wC;
      const x0 = Math.max(10, (W - totalW) / 2);
      const topY = 20;
      const Ax = x0, Ay = topY + (Math.max(hA, hB) - hA) / 2;
      const Bx = Ax + wA + gap, By = topY;
      const Cx = Bx + wB + gap, Cy = Ay;

      const drawGrid = (x, y, rows, cols, label, shapeTxt, hlRow, hlCol, fillFn) => {
        for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
          const cx = x + j * cell, cy = y + i * cell;
          let fill = th.bgPanel2;
          const extra = fillFn && fillFn(i, j); if (extra) fill = extra;
          if (hlRow === i) fill = th.accent;
          if (hlCol === j) fill = th.accent2;
          if (hlRow === i && hlCol === j) fill = th.good;
          g.fillStyle = fill; g.fillRect(cx, cy, cell, cell);
        }
        g.strokeStyle = th.border; g.lineWidth = 0.6;
        for (let i = 0; i <= rows; i++) { g.beginPath(); g.moveTo(x, y + i * cell); g.lineTo(x + cols * cell, y + i * cell); g.stroke(); }
        for (let j = 0; j <= cols; j++) { g.beginPath(); g.moveTo(x + j * cell, y); g.lineTo(x + j * cell, y + rows * cell); g.stroke(); }
        g.strokeStyle = th.text; g.lineWidth = 1.4; g.strokeRect(x, y, cols * cell, rows * cell);
        g.fillStyle = th.text; g.font = "bold 12px system-ui"; g.textAlign = "center";
        g.fillText(label, x + cols * cell / 2, y - 14);
        g.fillStyle = th.textDim; g.font = "10px monospace";
        g.fillText(shapeTxt, x + cols * cell / 2, y + rows * cell + 14);
      };

      const anim = this._anim;
      const hlAi = anim ? anim.i : -1, hlBj = anim ? anim.j : -1;
      drawGrid(Ax, Ay, r, k, A.label || "A", `(${r} × ${k})`, hlAi, -1);
      drawGrid(Bx, By, k2, c, B.label || "B", `(${k2} × ${c})`, -1, hlBj);
      drawGrid(Cx, Cy, r, c, (this.o.labels && this.o.labels.C) || "C", `(${r} × ${c})`,
        anim ? anim.i : -1, anim ? anim.j : -1,
        (i, j) => (anim && anim.filled && anim.filled.has(i + "," + j)) ? th.good : null);

      g.font = "16px system-ui"; g.fillStyle = th.textDim; g.textAlign = "center";
      g.fillText("@", Ax + wA + gap / 2, Ay + hA / 2);
      g.fillText("=", Bx + wB + gap / 2, By + hB / 2);

      g.font = "12px system-ui"; g.textAlign = "center";
      g.fillStyle = ok ? th.good : th.bad;
      g.fillText(ok ? `inner dims match: k = ${k} ✓` : `shape mismatch: A's cols (${k}) ≠ B's rows (${k2}) ✗`, (Ax + Cx + wC) / 2, H - 8);
    }
    animateMatmul(opts = {}) {
      this.stop();
      const A = opts.A || this.o.A, B = opts.B || this.o.B;
      this.o.A = A; this.o.B = B;
      const shA = sanitizeShape(A.shape), shB = sanitizeShape(B.shape);
      const [r, k] = shA, [k2, c] = shB;
      if (k !== k2 || r <= 0 || c <= 0) { this.draw(); if (opts.onDone) opts.onDone(); return; }
      const cells = []; for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) cells.push([i, j]);
      const stride = Math.max(1, Math.ceil(cells.length / (opts.maxSteps || 60)));
      const perStep = opts.msPerStep ?? clampNum(2400 / cells.length, 20, 140);
      const filled = new Set();
      let idx = 0;
      const step = () => {
        const batch = cells.slice(idx, idx + stride);
        if (!batch.length) { this._anim = null; this.draw(); this._timer = null; if (opts.onDone) opts.onDone(); return; }
        batch.forEach(([bi, bj]) => filled.add(bi + "," + bj));
        const [i, j] = batch[batch.length - 1];
        this._anim = { i, j, filled: new Set(filled) };
        this.draw();
        idx += stride;
        this._timer = setTimeout(() => { this._raf = requestAnimationFrame(step); }, perStep);
      };
      step();
    }
  }

  /* ================================ Timeline ================================
     new Timeline(canvas, {ymin,ymax,ylog,xlabel,ylabel,maxPoints,color,markers})
     .push(step, value)        // append one training step, auto-scales, redraws
     .addMarker(step,label,color) .setCurrent(step) .reset() .draw()
     Thin wrapper around Plot — reuses it rather than reimplementing axes/grid,
     per this course's house style. ========================================== */
  class Timeline {
    constructor(canvas, opts) {
      this.c = canvas;
      this.o = Object.assign({ ymin: null, ymax: null, ylog: false, xlabel: "step", ylabel: "loss", maxPoints: 4000, color: null, markers: [] }, opts);
      this.steps = []; this.values = []; this.current = null;
      this.draw();
    }
    reset() { this.steps = []; this.values = []; this.current = null; this.o.markers = []; this.draw(); }
    push(step, value) {
      if (!isFinite(step)) step = this.steps.length;
      this.steps.push(step); this.values.push(typeof value === "number" ? value : NaN);
      if (this.steps.length > this.o.maxPoints) { const drop = Math.ceil(this.o.maxPoints * 0.1); this.steps.splice(0, drop); this.values.splice(0, drop); }
      this.current = step;
      this.draw();
    }
    addMarker(step, label, color) { this.o.markers.push({ step, label, color }); this.draw(); }
    setCurrent(step) { this.current = step; this.draw(); }
    draw() {
      // NOTE: course.js's `class Plot` is a top-level `class` declaration in a
      // classic script — that makes `Plot` a global LEXICAL binding, not a
      // `window` property (`window.Plot` is undefined even though bare `Plot`
      // resolves fine — this is the exact footgun nn.js's own write-up flags
      // for top-level `class`). `typeof Plot` is safe even if course.js
      // somehow isn't loaded (typeof never throws on an unbound identifier).
      if (typeof Plot !== "function") return;
      const th = themeColors();
      const finite = this.values.filter(isFinite);
      let ymin = this.o.ymin, ymax = this.o.ymax;
      if (ymin == null || ymax == null) {
        if (finite.length) {
          const mn = Math.min(...finite), mx = Math.max(...finite), span = (mx - mn) || Math.abs(mx) || 1;
          if (ymin == null) ymin = this.o.ylog ? Math.max(mn * 0.5, 1e-9) : mn - span * 0.1;
          if (ymax == null) ymax = mx + span * 0.1;
        } else { ymin = ymin ?? 0; ymax = ymax ?? 1; }
      }
      const xs = this.steps.length ? this.steps : [0, 1];
      const xmin = xs[0], xmax0 = xs.length > 1 ? xs[xs.length - 1] : xs[0] + 1;
      const p = new Plot(this.c, { xmin, xmax: xmax0 > xmin ? xmax0 : xmin + 1, ymin, ymax, ylog: this.o.ylog, xlabel: this.o.xlabel, ylabel: this.o.ylabel });
      p.clear();
      // Plot._ticks()/_eng() format tick text in SI engineering notation
      // (100m, 10µ, ...) — right for the PI/SI course's volts-and-seconds
      // axes, wrong for a dimensionless training metric (a loss of 0.1 is
      // "0.1", not "100m"). Reuse Plot's tick POSITIONS (still correctly
      // spaced/log-aware) but relabel with plain decimal text.
      const relabel = (ticks) => ticks.map(t => ({ v: t.v, s: fmtNum(t.v) }));
      p.grid(relabel(p._ticks(p.o.xmin, p.o.xmax, p.o.xlog)), relabel(p._ticks(p.o.ymin, p.o.ymax, p.o.ylog)));
      if (this.steps.length) p.trace(this.steps, this.values, this.o.color || th.accent, 2);
      for (const m of this.o.markers) p.vline(m.step, m.color || th.accent2, [4, 3]);
      if (this.current != null) p.vline(this.current, th.text, [2, 2]);
    }
  }

  /* ================================== hl() ==================================
     Tiny regex syntax highlighter for Python / JS. Not a parser — "good
     enough to read". Emits ONLY what course.css's .codeblock rules style:
     <span class="line">...<span class="tok-kw|tok-str|tok-num|tok-fn|tok-com">
     matching the exact contract documented at course.css's .codeblock block
     (identifiers/operators/punctuation are left as plain escaped text — the
     CSS doesn't style them, so wrapping them would just be inert markup).

     hl(code, lang, {highlight:[1,3]})  -> inner HTML for <code>...</code>,
       one <span class="line"[ hl]"> per physical line (multi-line strings/
       comments are re-wrapped per line so `.line{display:block}` still works).
     hl.block(code, lang, {label, highlight}) -> a full, ready-to-insert
       .codeblock element (head + lang tag + copy button + <pre><code>).
       NOTE: course.js's wireCodeblocks() wires the copy button once at
       DOMContentLoaded. If a demo inserts a block via hl.block() AFTER that
       (the common case — most demos build their DOM inside their own
       DOMContentLoaded handler, which runs after course.js's), call the
       global wireCodeblocks() again after inserting it.
     ========================================================================== */
  const JS_KW = new Set(["const", "let", "var", "function", "return", "if", "else", "for", "while", "do", "class",
    "new", "this", "extends", "super", "import", "export", "default", "from", "as", "of", "in", "typeof",
    "instanceof", "true", "false", "null", "undefined", "break", "continue", "switch", "case", "try", "catch",
    "finally", "throw", "async", "await", "yield", "static", "get", "set", "void", "delete"]);
  const PY_KW = new Set(["def", "return", "if", "elif", "else", "for", "while", "class", "import", "from", "as",
    "in", "is", "not", "and", "or", "True", "False", "None", "break", "continue", "try", "except", "finally",
    "raise", "with", "lambda", "yield", "pass", "global", "nonlocal", "assert", "del", "async", "await", "self"]);

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function tokenize(code, lang) {
    const isPy = String(lang || "js").toLowerCase().startsWith("py");
    const KW = isPy ? PY_KW : JS_KW;
    const re = isPy
      ? /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(\b0[xX][0-9a-fA-F]+\b|\b\d+\.?\d*(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*)/g
      : /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`)|(\b0[xX][0-9a-fA-F]+\b|\b\d+\.?\d*(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*)/g;
    const tokens = []; let last = 0, m;
    while ((m = re.exec(code))) {
      if (m.index > last) tokens.push({ type: "plain", text: code.slice(last, m.index) });
      last = re.lastIndex;
      if (m[1] != null) tokens.push({ type: "com", text: m[1] });
      else if (m[2] != null) tokens.push({ type: "str", text: m[2] });
      else if (m[3] != null) tokens.push({ type: "num", text: m[3] });
      else if (m[4] != null) {
        const word = m[4];
        if (KW.has(word)) tokens.push({ type: "kw", text: word });
        else if (/^\s*\(/.test(code.slice(re.lastIndex, re.lastIndex + 40))) tokens.push({ type: "fn", text: word });
        else tokens.push({ type: "plain", text: word });
      }
    }
    if (last < code.length) tokens.push({ type: "plain", text: code.slice(last) });
    return tokens;
  }

  function hl(code, lang, opts) {
    opts = opts || {};
    const hlSet = new Set(opts.highlight || []);
    code = String(code == null ? "" : code).replace(/\r\n?/g, "\n");
    const tokens = tokenize(code, lang);
    const lines = [[]];
    for (const tok of tokens) {
      const parts = tok.text.split("\n");
      parts.forEach((part, i) => {
        if (i > 0) lines.push([]);
        if (part.length) lines[lines.length - 1].push({ type: tok.type, text: part });
      });
    }
    return lines.map((segs, idx) => {
      const body = segs.map(s => s.type === "plain" ? escapeHtml(s.text) : `<span class="tok-${s.type}">${escapeHtml(s.text)}</span>`).join("");
      const cls = "line" + (hlSet.has(idx + 1) ? " hl" : "");
      return `<span class="${cls}">${body}</span>`;
    }).join("\n");
  }
  hl.block = function (code, lang, opts) {
    opts = opts || {};
    const langLabel = opts.label || String(lang || "js").toUpperCase();
    return `<div class="codeblock" data-lang="${escapeHtml(String(lang || "js"))}">
  <div class="codeblock-head"><span class="lang-tag">${escapeHtml(langLabel)}</span><button class="copy-btn" type="button">Copy</button></div>
  <pre><code>${hl(code, lang, opts)}</code></pre>
</div>`;
  };

  /* ---------------------------------- exports ------------------------------ */
  window.Heatmap = Heatmap;
  window.NetGraph = NetGraph;
  window.Surface3D = Surface3D;
  window.TensorViz = TensorViz;
  window.Timeline = Timeline;
  window.hl = hl;
  // exposed for demos/tests that want the exact same colormaps/theme reader
  window.vizUtils = { cssVar, themeColors, sequentialCmap, divergingCmap, fitCanvas, sanitizeMatrix, sanitizeShape, fmtNum };

})();
