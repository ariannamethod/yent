(function (root) {
  'use strict';

  function finiteNumber(value, fallback) {
    const n = Number(typeof value === 'function' ? value() : value);
    return Number.isFinite(n) ? n : fallback;
  }

  function pixelRatio(windowRef, maxDpr) {
    const max = finiteNumber(maxDpr, 2);
    const raw = finiteNumber(windowRef && windowRef.devicePixelRatio, 1);
    return Math.max(1, Math.min(max > 0 ? max : 2, raw > 0 ? raw : 1));
  }

  function viewport(windowRef, maxDpr) {
    const win = windowRef || root;
    return {
      width: Math.max(0, finiteNumber(win && win.innerWidth, 0)),
      height: Math.max(0, finiteNumber(win && win.innerHeight, 0)),
      dpr: pixelRatio(win, maxDpr)
    };
  }

  function resizeSurface(surface, base) {
    surface = surface || {};
    const canvas = surface.canvas;
    const context = surface.context;
    if (!canvas || !context || typeof context.setTransform !== 'function') {
      throw new Error('interface canvas surface unavailable');
    }

    const cssW = Math.max(0, finiteNumber(surface.width, base.width));
    const cssH = Math.max(0, finiteNumber(surface.height, base.height));
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width = Math.max(1, Math.floor(cssW * base.dpr));
    canvas.height = Math.max(1, Math.floor(cssH * base.dpr));
    context.setTransform(base.dpr, 0, 0, base.dpr, 0, 0);
    return { canvas, context, width: cssW, height: cssH };
  }

  function resize(options) {
    options = options || {};
    const base = viewport(options.window, options.maxDpr);
    const surfaces = Array.isArray(options.surfaces)
      ? options.surfaces
      : [{ canvas: options.canvas, context: options.context }];
    return {
      width: base.width,
      height: base.height,
      dpr: base.dpr,
      surfaces: surfaces.map(surface => resizeSurface(surface, base))
    };
  }

  const api = { pixelRatio, viewport, resize };
  root.YentInterfaceCanvas = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
