// Second Reader public site — progressive enhancement only.
// For research use only. Not for diagnostic use.
//
// 1. Sub-navigation: highlight the section in view (details page).
// 2. Cover hero: a looping, low-cost illustration of one second read —
//    a scanning sweep, a region of attention fading in, and three findings
//    ticking to calibrated probabilities against their operating thresholds,
//    with an "unable to assess" cycle every third loop. The values shown are
//    an illustration, not a model output. Pauses when the tab is hidden or
//    the hero is scrolled out of view; renders one static frame when the
//    visitor prefers reduced motion.
(function () {
  "use strict";

  /* ---------- 1. Current-section highlight ---------- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".subnav a[href^='#']"));
  var sections = links
    .map(function (a) { return document.getElementById(a.getAttribute("href").slice(1)); })
    .filter(Boolean);
  if ("IntersectionObserver" in window && sections.length) {
    var current = null;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) { current = entry.target.id; }
      });
      links.forEach(function (a) {
        if (a.getAttribute("href") === "#" + current) {
          a.setAttribute("aria-current", "true");
          if (a.scrollIntoView && a.parentNode && a.parentNode.parentNode) {
            var ul = a.parentNode.parentNode;
            var left = a.offsetLeft - ul.clientWidth / 2 + a.offsetWidth / 2;
            if (ul.scrollTo) { ul.scrollTo({ left: left, behavior: "auto" }); }
          }
        } else {
          a.removeAttribute("aria-current");
        }
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    sections.forEach(function (s) { io.observe(s); });
  }

  /* ---------- 2. Cover hero animation ---------- */
  var canvas = document.getElementById("scan");
  if (!canvas || !canvas.getContext) { return; }
  var ctx = canvas.getContext("2d");
  var stage = canvas.closest(".stage") || canvas.parentNode;
  var readout = document.getElementById("readout");
  var gateText = document.getElementById("gate-text");
  var stateText = document.getElementById("scan-state");
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var SIZE = 512;
  // Radiologist's box for pleural effusion on this NIH ChestX-ray14 image
  // (BBox_List_2017.csv, 00014015_003.png, 1024-px frame), scaled to 512 px.
  var region = { x: 52.6, y: 204.4, w: 137.7, h: 89.3 };
  var findings = [
    { id: "eff", p: 0.81, thr: 0.099 },
    { id: "ptx", p: 0.02, thr: 0.040 },
    { id: "nod", p: 0.06, thr: 0.097 }
  ];
  findings.forEach(function (f) {
    f.pEl = document.getElementById("p-" + f.id);
    f.fillEl = document.getElementById("fill-" + f.id);
    f.flagEl = document.getElementById("flag-" + f.id);
    f.thrEl = document.getElementById("thr-" + f.id);
    if (f.thrEl) { f.thrEl.style.left = (f.thr * 100).toFixed(1) + "%"; }
  });

  var img = new Image();
  img.decoding = "async";
  img.src = canvas.getAttribute("data-src");

  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = SIZE * dpr;
  canvas.height = SIZE * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function ease(v) { v = clamp01(v); return v < 0.5 ? 4 * v * v * v : 1 - Math.pow(-2 * v + 2, 3) / 2; }
  function easeOut(v) { v = clamp01(v); return 1 - Math.pow(1 - v, 3); }
  function ramp(t, a, b) { return clamp01((t - a) / (b - a)); }

  // Timeline for one loop. Every third loop is an "unable to assess" cycle.
  var NORMAL_T = 12, UNABLE_T = 8;
  function frameState(t, unable) {
    var s = { sweep: -1, sweepAlpha: 0, heat: 0, box: 0, prob: 0, dim: 0, unable: unable, phase: "read" };
    var sweepEnd = unable ? 3.3 : 4.3;
    if (t >= 0.3 && t <= sweepEnd) {
      s.sweep = ease(ramp(t, 0.3, sweepEnd)) * SIZE;
      s.sweepAlpha = 1 - ramp(t, sweepEnd - 0.4, sweepEnd);
    }
    if (unable) {
      var fadeOut = 1 - ease(ramp(t, 7.2, 8));
      s.dim = ease(ramp(t, 3.3, 3.9)) * fadeOut;
      s.phase = t < 3.3 ? "read" : t < 7.2 ? "unable" : "reset";
    } else {
      var out = 1 - ease(ramp(t, 11.1, 12));
      s.heat = ease(ramp(t, 2.4, 4.6)) * out;
      s.box = ease(ramp(t, 4.4, 5.1)) * out;
      s.prob = easeOut(ramp(t, 3.0, 5.6)) * out;
      s.phase = t < 3.0 ? "read" : t < 11.1 ? "result" : "reset";
    }
    return s;
  }

  function drawHeat(alpha) {
    if (alpha <= 0) { return; }
    var cx = region.x + region.w / 2, cy = region.y + region.h / 2;
    ctx.save();
    ctx.globalAlpha = alpha * 0.65;
    ctx.globalCompositeOperation = "source-over";
    var blobs = [
      [cx, cy, region.w * 0.62, region.h * 0.72],
      [cx - region.w * 0.22, cy + region.h * 0.18, region.w * 0.42, region.h * 0.5],
      [cx + region.w * 0.2, cy - region.h * 0.1, region.w * 0.4, region.h * 0.45]
    ];
    blobs.forEach(function (b) {
      ctx.save();
      ctx.translate(b[0], b[1]);
      ctx.scale(b[2] / b[3], 1);
      var g = ctx.createRadialGradient(0, 0, 0, 0, 0, b[3]);
      g.addColorStop(0, "rgba(255,61,0,0.95)");
      g.addColorStop(0.45, "rgba(255,176,0,0.6)");
      g.addColorStop(1, "rgba(255,176,0,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(0, 0, b[3], 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
    ctx.restore();
  }

  function drawBox(alpha) {
    if (alpha <= 0) { return; }
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = "#3fd6c4";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(region.x, region.y, region.w, region.h);
    ctx.setLineDash([]);
    ctx.fillStyle = "#3fd6c4";
    ctx.font = "600 11px 'IBM Plex Mono', ui-monospace, Menlo, monospace";
    ctx.fillText("effusion · region", region.x, region.y - 6);
    ctx.restore();
  }

  function drawSweep(y, alpha) {
    if (y < 0 || alpha <= 0) { return; }
    ctx.save();
    ctx.globalAlpha = alpha;
    var g = ctx.createLinearGradient(0, y - 70, 0, y);
    g.addColorStop(0, "rgba(63,214,196,0)");
    g.addColorStop(1, "rgba(63,214,196,0.28)");
    ctx.fillStyle = g;
    ctx.fillRect(0, y - 70, SIZE, 70);
    ctx.fillStyle = "rgba(63,214,196,0.95)";
    ctx.fillRect(0, y - 1, SIZE, 2);
    ctx.restore();
  }

  function drawFrame(s) {
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, SIZE, SIZE);
    if (img.complete && img.naturalWidth) {
      ctx.drawImage(img, 0, 0, SIZE, SIZE);
    }
    drawHeat(s.heat);
    drawBox(s.box);
    if (s.dim > 0) {
      ctx.save();
      ctx.globalAlpha = s.dim * 0.6;
      ctx.fillStyle = "#0a0e11";
      ctx.fillRect(0, 0, SIZE, SIZE);
      ctx.globalAlpha = s.dim * 0.35;
      ctx.strokeStyle = "#f2b134";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (var k = -SIZE; k < SIZE * 2; k += 22) { ctx.moveTo(k, 0); ctx.lineTo(k + SIZE, SIZE); }
      ctx.stroke();
      ctx.restore();
    }
    drawSweep(s.sweep, s.sweepAlpha);
    // corner reticle marks
    ctx.save();
    ctx.strokeStyle = "rgba(183,195,196,0.45)";
    ctx.lineWidth = 1;
    var m = 14, L = 18;
    [[m, m, 1, 1], [SIZE - m, m, -1, 1], [m, SIZE - m, 1, -1], [SIZE - m, SIZE - m, -1, -1]].forEach(function (c) {
      ctx.beginPath();
      ctx.moveTo(c[0], c[1] + c[3] * L); ctx.lineTo(c[0], c[1]); ctx.lineTo(c[0] + c[2] * L, c[1]);
      ctx.stroke();
    });
    ctx.restore();
  }

  var lastPhase = null;
  function updateReadout(s) {
    if (readout) { readout.classList.toggle("unable", s.phase === "unable"); }
    findings.forEach(function (f) {
      var v = s.unable ? 0 : f.p * s.prob;
      var shown = s.unable || s.prob <= 0 ? "—" : "p = " + v.toFixed(2);
      if (f.pEl && f.pEl.textContent !== shown) { f.pEl.textContent = shown; }
      if (f.fillEl) { f.fillEl.style.width = (v * 100).toFixed(1) + "%"; }
      var flag = "";
      if (s.phase === "unable") { flag = "not assessed"; }
      else if (s.phase === "result") { flag = v >= f.thr ? "▲ above threshold " + f.thr.toFixed(3) : "below threshold " + f.thr.toFixed(3); }
      if (f.flagEl && f.flagEl.textContent !== flag) { f.flagEl.textContent = flag; }
    });
    if (s.phase !== lastPhase) {
      lastPhase = s.phase;
      var msg;
      if (s.phase === "unable") {
        msg = "unable to assess — learned OOD distance above threshold. No probabilities returned.";
      } else if (s.phase === "read") {
        msg = "gate: frontal view · single frame · pixels ok · checking distribution…";
      } else if (s.phase === "result") {
        msg = "gate passed: PA view, in distribution. Model cc2f56aa · Platt-calibrated.";
      } else {
        msg = "next study…";
      }
      if (gateText) { gateText.textContent = msg; }
      if (stateText) {
        stateText.textContent = s.phase === "unable"
          ? "Illustration: the quality gate returned unable to assess; no probabilities are shown."
          : s.phase === "result"
            ? "Illustration: pleural effusion p = 0.81 above its 0.099 threshold with a heat-map over the right lung base; pneumothorax p = 0.02 and nodule/mass p = 0.06 below their thresholds."
            : "Illustration: the radiograph is being read.";
      }
    }
  }

  function render(s) { drawFrame(s); updateReadout(s); }

  if (reduced) {
    // One static frame with the finished result; no animation.
    var still = frameState(6.5, false);
    var paint = function () { render(still); };
    if (img.complete) { paint(); } else { img.onload = paint; }
    return;
  }

  var loop = 0, loopStart = null, paused = true, rafId = null, pausedAt = 0;
  function loopIsUnable(n) { return n % 3 === 2; }
  function loopLength(n) { return loopIsUnable(n) ? UNABLE_T : NORMAL_T; }

  function tick(now) {
    if (paused) { return; }
    if (loopStart === null) { loopStart = now; }
    var t = (now - loopStart) / 1000;
    var T = loopLength(loop);
    if (t >= T) { loop += 1; loopStart = now; t = 0; }
    render(frameState(t, loopIsUnable(loop)));
    rafId = window.requestAnimationFrame(tick);
  }

  function start() {
    if (!paused) { return; }
    paused = false;
    if (pausedAt && loopStart !== null) { loopStart = performance.now() - pausedAt; }
    rafId = window.requestAnimationFrame(tick);
  }
  function stop() {
    if (paused) { return; }
    paused = true;
    if (loopStart !== null) { pausedAt = performance.now() - loopStart; }
    if (rafId) { window.cancelAnimationFrame(rafId); rafId = null; }
  }

  var visible = !document.hidden, inView = true;
  function reconcile() { if (visible && inView) { start(); } else { stop(); } }
  document.addEventListener("visibilitychange", function () { visible = !document.hidden; reconcile(); });
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { inView = e.isIntersecting; });
      reconcile();
    }, { threshold: 0.05 }).observe(stage);
  }

  var begin = function () { render(frameState(0, false)); reconcile(); };
  if (img.complete && img.naturalWidth) { begin(); } else { img.onload = begin; img.onerror = begin; }

  // Expose a tiny hook for manual verification (not used by the page).
  window.__scanHero = {
    isPaused: function () { return paused; },
    loop: function () { return loop; },
    renderAt: function (t, unable) { stop(); render(frameState(t, !!unable)); }
  };
})();
