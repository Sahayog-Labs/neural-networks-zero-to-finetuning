/* ==========================================================================
   ANN Course — shared runtime: nav, KaTeX, quiz engine, demo helpers
   Every page: <script src="assets/course.js"></script> at end of <body>,
   with <body data-page="NN-filename.html">.
   ========================================================================== */
"use strict";

/* ---------------- course map (single source of truth for nav) ------------ */
const COURSE_PARTS = [];   // populated by COURSE_PAGES groupings below
const COURSE_PAGES = [];   // [{file,num,title,part}] — filled in course-map.js
/* The actual map lives in assets/course-map.js so it can be regenerated
   without touching this engine file. Entries may optionally carry
   track: "llm" | "diffusion" to tag a page as belonging to one of the two
   parallel tracks the trunk forks into; pages without it render as trunk. */

/* ---------------- theme (dark default / cream light), persisted ----------- */
function getTheme() {
  try { return localStorage.getItem("ann-theme") || "dark"; } catch (e) { return "dark"; }
}
function applyTheme(t) {
  if (t === "light") document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem("ann-theme", t); } catch (e) { /* file:// or private mode */ }
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = t === "light" ? "🌙 Dark mode" : "☀️ Light mode";
}
applyTheme(getTheme()); // apply immediately (course.js loads before first paint of most content)

/* ---------------- favicon (data URL, avoids 404) -------------------------- */
(function () {
  const l = document.createElement("link");
  l.rel = "icon";
  l.href = "data:image/svg+xml," + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y="80" font-size="80">🧠</text></svg>');
  document.head.appendChild(l);
})();

/* ---------------- KaTeX (local bundle, auto-render) ---------------------- */
(function loadKaTeX() {
  const base = "assets/katex/";
  const css = document.createElement("link");
  css.rel = "stylesheet"; css.href = base + "katex.min.css";
  document.head.appendChild(css);
  const s1 = document.createElement("script");
  s1.src = base + "katex.min.js"; s1.defer = true;
  s1.onload = () => {
    const s2 = document.createElement("script");
    s2.src = base + "contrib/auto-render.min.js"; s2.defer = true;
    s2.onload = () => {
      renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false
      });
    };
    document.head.appendChild(s2);
  };
  document.head.appendChild(s1);
})();

/* ---------------- navigation ------------------------------------------- */
/* Sidebar renders each COURSE_MAP "part" as a collapsible <details> group
   (only the group containing the current page starts open) so a ~50-entry,
   trunk + two-track map stays a scannable outline instead of one long scroll.
   A group where every page shares the same `track` gets that track's rail
   color and a small track chip next to its heading; individual links inside
   a mixed group can still be tagged per-page via p.track. */
function buildNav() {
  if (!window.COURSE_MAP) return;
  const current = document.body.dataset.page || location.pathname.split("/").pop();
  const layout = document.getElementById("layout");
  if (!layout) return;
  const pages = window.COURSE_MAP;

  const sb = document.createElement("nav");
  sb.id = "sidebar";
  const idx0 = pages.findIndex(p => p.file === current);
  const pct = pages.length > 1 && idx0 >= 0 ? Math.round((idx0 / (pages.length - 1)) * 100) : 0;
  let html = `<div class="course-title">🧠 Neural Networks<br>Zero to Fine-Tuning</div>
    <div class="sidebar-progress"><div class="sidebar-progress-fill" style="width:${pct}%"></div></div>
    <div class="sidebar-progress-label">${idx0 >= 0 ? `Page ${idx0 + 1} of ${pages.length} · ${pct}%` : `${pages.length} pages`}</div>
    <button id="theme-toggle" class="btn" style="width:calc(100% - 12px);margin:0 6px 10px;font-size:.8rem;padding:6px 10px;"></button>`;

  // group consecutive pages by `part`, preserving map order
  const groups = [];
  for (const p of pages) {
    if (!groups.length || groups[groups.length - 1].part !== p.part) groups.push({ part: p.part, pages: [] });
    groups[groups.length - 1].pages.push(p);
  }
  for (const g of groups) {
    const hasCurrent = g.pages.some(p => p.file === current);
    const tracks = new Set(g.pages.map(p => p.track).filter(Boolean));
    const groupTrack = tracks.size === 1 ? [...tracks][0] : null;
    const groupCls = "nav-group" + (groupTrack ? " track-" + groupTrack : "");
    html += `<details class="${groupCls}"${hasCurrent ? " open" : ""}><summary class="nav-part">${g.part}</summary>`;
    for (const p of g.pages) {
      const cls = ["nav-link"];
      if (p.file === current) cls.push("current");
      if (p.track) cls.push("track-" + p.track);
      if (p.milestone) cls.push("milestone-link");
      html += `<a class="${cls.join(" ")}" href="${p.file}">${p.num ? p.num + ". " : ""}${p.title}</a>`;
    }
    html += `</details>`;
  }
  sb.innerHTML = html;
  layout.insertBefore(sb, layout.firstChild);
  const tbtn = sb.querySelector("#theme-toggle");
  tbtn.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
  });
  applyTheme(getTheme()); // sets the button label now that it exists

  // prev/next pager — devOnly entries (e.g. the template gallery) are listed
  // in the sidebar but never appear as a pager neighbor, so index→01→…→65
  // stays an unbroken content-only chain.
  const pagerPages = pages.filter(p => !p.devOnly);
  const idx = pagerPages.findIndex(p => p.file === current);
  if (idx >= 0) {
    const inner = document.querySelector("#content .inner") || document.getElementById("content");
    const pager = document.createElement("div");
    pager.className = "pager";
    const prev = pagerPages[idx - 1], next = pagerPages[idx + 1];
    pager.innerHTML =
      (prev ? `<a class="prev" href="${prev.file}"><span class="dir">← Previous</span><span class="ttl">${prev.title}</span></a>` : `<span style="flex:1"></span>`) +
      (next ? `<a class="next" href="${next.file}"><span class="dir">Next →</span><span class="ttl">${next.title}</span></a>` : `<span style="flex:1"></span>`);
    inner.appendChild(pager);
  }
  const cur = sb.querySelector(".nav-link.current");
  if (cur) cur.scrollIntoView({ block: "center" });
}

/* ---------------- codeblocks: line-number CSS counters need no JS, but the
   copy button does. Wires every .codeblock on the page once. -------------- */
function wireCodeblocks() {
  document.querySelectorAll(".codeblock").forEach(block => {
    const btn = block.querySelector(".copy-btn");
    const code = block.querySelector("code");
    if (!btn || !code) return;
    btn.addEventListener("click", () => {
      const text = code.innerText;
      const done = () => {
        const old = btn.textContent;
        btn.textContent = "Copied!"; btn.classList.add("copied");
        setTimeout(() => { btn.textContent = old; btn.classList.remove("copied"); }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
      } else {
        fallbackCopy(text, done);
      }
    });
  });
}
function fallbackCopy(text, done) {
  try {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    done();
  } catch (e) { /* clipboard unavailable — silently no-op */ }
}

/* ---------------- milestones: data-progress sets the fill width ---------- */
function wireMilestones() {
  document.querySelectorAll(".milestone[data-progress]").forEach(m => {
    const v = clampPct(parseFloat(m.dataset.progress));
    m.style.setProperty("--milestone-fill", v + "%");
  });
}
function clampPct(v) { return isFinite(v) ? Math.max(0, Math.min(100, v)) : 0; }

/* ---------------- quiz engine ------------------------------------------- */
/* renderQuiz("containerId", [{q, opts:[..], a:idx, why} | {q, num:val, tol, unit, why}], "Title") */
function renderQuiz(containerId, questions, title) {
  const host = document.getElementById(containerId);
  if (!host) return;
  host.classList.add("quiz");
  const qid = containerId;
  let html = `<div class="quiz-head">📝 ${title || "Check your understanding"}</div>`;
  questions.forEach((item, i) => {
    html += `<div class="q" id="${qid}-q${i}"><div class="q-text"><span class="q-num">Q${i + 1}.</span>${item.q}</div>`;
    if (item.opts) {
      item.opts.forEach((opt, j) => {
        html += `<label class="opt"><input type="radio" name="${qid}-q${i}" value="${j}">${opt}</label>`;
      });
    } else {
      html += `<input class="numin" type="text" placeholder="answer${item.unit ? " (" + item.unit + ")" : ""}" id="${qid}-q${i}-in"> ${item.unit ? `<span style="color:var(--text-dim)">${item.unit}</span>` : ""}`;
    }
    html += `<div class="explain" id="${qid}-q${i}-ex"></div></div>`;
  });
  html += `<div class="quiz-foot"><button id="${qid}-grade">Grade quiz</button><span class="score" id="${qid}-score"></span></div>`;
  host.innerHTML = html;

  document.getElementById(qid + "-grade").addEventListener("click", () => {
    let score = 0;
    questions.forEach((item, i) => {
      const ex = document.getElementById(`${qid}-q${i}-ex`);
      let good = false;
      if (item.opts) {
        const chosen = host.querySelector(`input[name="${qid}-q${i}"]:checked`);
        const labels = host.querySelectorAll(`#${qid}-q${i} .opt`);
        labels.forEach(l => l.classList.remove("correct", "incorrect"));
        labels[item.a].classList.add("correct");
        if (chosen) {
          good = +chosen.value === item.a;
          if (!good) labels[+chosen.value].classList.add("incorrect");
        }
      } else {
        const inEl = document.getElementById(`${qid}-q${i}-in`);
        const v = parseFloat((inEl.value || "").replace(/,/g, ""));
        const tol = item.tol != null ? item.tol : Math.abs(item.num) * 0.05;
        good = isFinite(v) && Math.abs(v - item.num) <= tol;
        inEl.classList.remove("correct", "incorrect");
        inEl.classList.add(good ? "correct" : "incorrect");
      }
      if (good) score++;
      ex.innerHTML = (good ? "✅ " : "❌ ") + (item.why || (item.num != null ? `Answer: ${item.num}${item.unit ? " " + item.unit : ""}` : ""));
      ex.classList.add("shown");
    });
    const sc = document.getElementById(qid + "-score");
    sc.textContent = `Score: ${score} / ${questions.length}`;
    sc.className = "score " + (score >= Math.ceil(questions.length * 0.7) ? "pass" : "fail");
    if (window.renderMathInElement) renderMathInElement(host, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }], throwOnError: false });
  });
  if (window.renderMathInElement) renderMathInElement(host, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }], throwOnError: false });
}

/* ---------------- demo helpers ------------------------------------------ */
/* Slider with live value display.
   makeCtrl(parentEl, {label, min, max, step, value, unit, fmt}) -> {input, get(), onInput(fn)} */
function makeCtrl(parent, cfg) {
  const wrap = document.createElement("div");
  wrap.className = "ctrl";
  const fmt = cfg.fmt || (v => v);
  wrap.innerHTML = `<label>${cfg.label}: <span class="val"></span></label>
    <input type="range" min="${cfg.min}" max="${cfg.max}" step="${cfg.step}" value="${cfg.value}">`;
  parent.appendChild(wrap);
  const input = wrap.querySelector("input");
  const val = wrap.querySelector(".val");
  const upd = () => { val.textContent = fmt(parseFloat(input.value)) + (cfg.unit ? " " + cfg.unit : ""); };
  input.addEventListener("input", upd); upd();
  return {
    input,
    get: () => parseFloat(input.value),
    onInput: fn => input.addEventListener("input", () => { upd(); fn(parseFloat(input.value)); })
  };
}

/* Read a CSS custom property (theme-aware — re-reads live so the theme toggle
   retints anything using it without a page reload). Falls back to a literal
   if the variable is unset or getComputedStyle throws (e.g. detached canvas). */
function cssVar(name, fallback) {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  } catch (e) { return fallback; }
}

/* Canvas plot helper — linear or log axes, grid, multiple traces.
   const p = new Plot(canvas, {xmin,xmax,ymin,ymax, xlog, ylog, xlabel, ylabel});
   p.clear(); p.grid(); p.trace(xs, ys, color); p.text(...); */
class Plot {
  constructor(canvas, o) {
    this.c = canvas;
    this.o = Object.assign({ xlog: false, ylog: false, pad: { l: 62, r: 16, t: 14, b: 42 } }, o);
    // ---- sanitize axis ranges: a degenerate or non-finite range must never
    // hang the tick generator or produce division by zero. ----
    const s = this.o;
    if (!isFinite(s.xmin)) s.xmin = 0;
    if (!isFinite(s.xmax)) s.xmax = s.xmin + 1;
    if (!isFinite(s.ymin)) s.ymin = 0;
    if (!isFinite(s.ymax)) s.ymax = s.ymin + 1;
    if (s.xlog) {
      if (s.xmin <= 0) s.xmin = 1e-12;
      if (s.xmax <= s.xmin) s.xmax = s.xmin * 10;
    } else if (s.xmax <= s.xmin) { const c0 = s.xmin; s.xmin = c0 - 0.5; s.xmax = c0 + 0.5; }
    if (s.ylog) {
      if (s.ymin <= 0) s.ymin = 1e-12;
      if (s.ymax <= s.ymin) s.ymax = s.ymin * 10;
    } else if (s.ymax <= s.ymin) { const c0 = s.ymin; s.ymin = c0 - 0.5; s.ymax = c0 + 0.5; }
    const dpr = window.devicePixelRatio || 1;
    // The design height must be read from the ORIGINAL height attribute only
    // once: assigning canvas.height later reflects into that same attribute,
    // so re-reading it each draw would compound dpr into the size every
    // redraw (exponential growth → broken/blank canvas on scaled displays).
    if (!canvas.dataset.baseH) canvas.dataset.baseH = canvas.getAttribute("height") || 360;
    const w = canvas.clientWidth || canvas.width / dpr || 600,
          h = +canvas.dataset.baseH;
    // Only touch the backing store when the size actually changed — assigning
    // canvas.width discards and reallocates the whole buffer (very expensive at
    // high devicePixelRatio, and doing it on every slider tick starves the main
    // thread on long drags).
    const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw; canvas.height = bh;
      canvas.style.height = h + "px";
    }
    this.g = canvas.getContext("2d");
    this.g.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = w; this.H = h;
  }
  _tx(x) { const o = this.o, p = o.pad; const v = o.xlog ? Math.log10(x) : x; const a = o.xlog ? Math.log10(o.xmin) : o.xmin, b = o.xlog ? Math.log10(o.xmax) : o.xmax; return p.l + (v - a) / (b - a) * (this.W - p.l - p.r); }
  _ty(y) { const o = this.o, p = o.pad; const v = o.ylog ? Math.log10(y) : y; const a = o.ylog ? Math.log10(o.ymin) : o.ymin, b = o.ylog ? Math.log10(o.ymax) : o.ymax; return this.H - p.b - (v - a) / (b - a) * (this.H - p.t - p.b); }
  clear() { this.g.clearRect(0, 0, this.W, this.H); }
  grid(xticks, yticks) {
    const g = this.g, o = this.o, p = o.pad;
    const gridC = cssVar("--plot-grid", "#243049"), tickC = cssVar("--plot-tick", "#8391ab"),
          borderC = cssVar("--plot-border", "#3b4a6b"), labelC = cssVar("--plot-label", "#a8b4cc");
    g.save(); g.strokeStyle = gridC; g.fillStyle = tickC; g.lineWidth = 1; g.font = "11px Consolas, monospace";
    const xt = xticks || this._ticks(o.xmin, o.xmax, o.xlog);
    const yt = yticks || this._ticks(o.ymin, o.ymax, o.ylog);
    g.textAlign = "center"; g.textBaseline = "top";
    for (const t of xt) { const x = this._tx(t.v); g.beginPath(); g.moveTo(x, p.t); g.lineTo(x, this.H - p.b); g.stroke(); g.fillText(t.s, x, this.H - p.b + 6); }
    g.textAlign = "right"; g.textBaseline = "middle";
    for (const t of yt) { const y = this._ty(t.v); g.beginPath(); g.moveTo(p.l, y); g.lineTo(this.W - p.r, y); g.stroke(); g.fillText(t.s, p.l - 8, y); }
    g.strokeStyle = borderC;
    g.strokeRect(p.l, p.t, this.W - p.l - p.r, this.H - p.t - p.b);
    if (o.xlabel) { g.textAlign = "center"; g.textBaseline = "bottom"; g.fillStyle = labelC; g.font = "12px system-ui"; g.fillText(o.xlabel, p.l + (this.W - p.l - p.r) / 2, this.H - 4); }
    if (o.ylabel) { g.save(); g.translate(13, p.t + (this.H - p.t - p.b) / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.textBaseline = "top"; g.fillStyle = labelC; g.font = "12px system-ui"; g.fillText(o.ylabel, 0, 0); g.restore(); }
    g.restore();
  }
  _ticks(a, b, isLog) {
    const out = [];
    if (isLog) {
      const lo = Math.ceil(Math.log10(a)), hi = Math.floor(Math.log10(b));
      if (!isFinite(lo) || !isFinite(hi)) return out;
      for (let e = lo; e <= hi && out.length < 60; e++) { const v = Math.pow(10, e); out.push({ v, s: this._eng(v) }); }
    } else {
      const span = b - a;
      if (!(span > 0) || !isFinite(span)) return out;
      const step = Math.pow(10, Math.floor(Math.log10(span / 5)));
      const mult = span / step > 10 ? (span / step > 20 ? 5 : 2) : 1;
      const st = step * mult;
      if (!isFinite(st) || st <= 0) return out;
      for (let v = Math.ceil(a / st) * st; v <= b + 1e-12 && out.length < 200; v += st) out.push({ v, s: this._eng(v) });
    }
    return out;
  }
  _eng(v) {
    if (v === 0) return "0";
    const av = Math.abs(v);
    const units = [[1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""], [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"], [1e-12, "p"]];
    for (const [m, s] of units) if (av >= m * 0.999) { const x = v / m; return (Math.round(x * 100) / 100) + s; }
    return v.toExponential(0);
  }
  trace(xs, ys, color, width) {
    const g = this.g; g.save(); g.strokeStyle = color || "#4fc3f7"; g.lineWidth = width || 2; g.lineJoin = "round";
    g.beginPath();
    let started = false;
    const o = this.o, p = o.pad;
    g.rect(p.l, p.t, this.W - p.l - p.r, this.H - p.t - p.b); g.clip(); g.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const y = ys[i];
      if (!isFinite(y)) { started = false; continue; }
      const px = this._tx(xs[i]), py = this._ty(y);
      if (!started) { g.moveTo(px, py); started = true; } else g.lineTo(px, py);
    }
    g.stroke(); g.restore();
  }
  hline(y, color, dash) { const g = this.g, p = this.o.pad; g.save(); g.strokeStyle = color; if (dash) g.setLineDash(dash); g.beginPath(); const py = this._ty(y); g.moveTo(p.l, py); g.lineTo(this.W - p.r, py); g.stroke(); g.restore(); }
  vline(x, color, dash) { const g = this.g, p = this.o.pad; g.save(); g.strokeStyle = color; if (dash) g.setLineDash(dash); g.beginPath(); const px = this._tx(x); g.moveTo(px, p.t); g.lineTo(px, this.H - p.b); g.stroke(); g.restore(); }
  label(x, y, txt, color) { const g = this.g; g.save(); g.fillStyle = color || "#dce3f0"; g.font = "12px system-ui"; g.fillText(txt, this._tx(x), this._ty(y)); g.restore(); }
}

/* engineering-notation formatter for readouts */
function eng(v, unit, digits) {
  if (!isFinite(v)) return "—";
  if (v === 0) return "0 " + (unit || "");
  const av = Math.abs(v);
  const units = [[1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""], [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"], [1e-12, "p"], [1e-15, "f"]];
  for (const [m, s] of units) if (av >= m * 0.9995) return (v / m).toPrecision(digits || 3) * 1 + " " + s + (unit || "");
  return v.toExponential(2) + " " + (unit || "");
}

/* ---------------- boot ---------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  buildNav();
  wireCodeblocks();
  wireMilestones();
});