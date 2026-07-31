(function (root) {
  'use strict';

  function finiteNumber(value, fallback) {
    const n = Number(typeof value === 'function' ? value() : value);
    return Number.isFinite(n) ? n : fallback;
  }

  function hasOwn(value, key) {
    return Object.prototype.hasOwnProperty.call(Object(value), key);
  }

  function looksLikeDocument(value) {
    return !!value && (
      typeof value.getElementById === 'function' ||
      typeof value.createElement === 'function'
    );
  }

  function looksLikeViewport(value) {
    return !!value && (
      hasOwn(value, 'innerWidth') ||
      hasOwn(value, 'innerHeight') ||
      hasOwn(value, 'devicePixelRatio')
    );
  }

  function rejectBareViewport(value) {
    if (looksLikeViewport(value)) {
      throw new Error('interface canvas viewport must be passed as { viewport }');
    }
  }

  function pixelRatio(options) {
    rejectBareViewport(options);
    options = options || {};
    const max = finiteNumber(options.maxDpr, 2);
    const viewportRef = hasOwn(options, 'viewport') ? options.viewport : root;
    const raw = finiteNumber(viewportRef && viewportRef.devicePixelRatio, 1);
    return Math.max(1, Math.min(max > 0 ? max : 2, raw > 0 ? raw : 1));
  }

  function viewport(options) {
    rejectBareViewport(options);
    options = options || {};
    const win = hasOwn(options, 'viewport') ? options.viewport : root;
    return {
      width: Math.max(0, finiteNumber(win && win.innerWidth, 0)),
      height: Math.max(0, finiteNumber(win && win.innerHeight, 0)),
      dpr: pixelRatio({ viewport: win, maxDpr: options.maxDpr })
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
    if (hasOwn(options, 'canvas') || hasOwn(options, 'context')) {
      throw new Error('interface canvas resize surface must be passed as { surface } or { surfaces }');
    }
    const viewportOptions = { maxDpr: options.maxDpr };
    if (hasOwn(options, 'viewport')) viewportOptions.viewport = options.viewport;
    const base = viewport(viewportOptions);
    const surfaces = Array.isArray(options.surfaces)
      ? options.surfaces
      : (hasOwn(options, 'surface') ? [options.surface] : []);
    if (surfaces.length === 0) {
      throw new Error('interface canvas resize surface must be passed as { surface } or { surfaces }');
    }
    return {
      width: base.width,
      height: base.height,
      dpr: base.dpr,
      surfaces: surfaces.map(surface => resizeSurface(surface, base))
    };
  }

  function bind(options) {
    if (typeof options === 'string') {
      throw new Error('interface canvas id must be passed as { id }');
    }
    if (looksLikeDocument(options)) {
      throw new Error('interface canvas document must be passed as { document }');
    }
    options = options || {};
    const documentRef = hasOwn(options, 'document') ? options.document : root.document;
    if (!documentRef || typeof documentRef.getElementById !== 'function') {
      throw new Error('interface canvas document unavailable');
    }
    const id = options.id;
    if (!id) throw new Error('interface canvas id missing');
    const canvas = documentRef.getElementById(id);
    if (!canvas || typeof canvas.getContext !== 'function') {
      throw new Error(`interface canvas element unavailable: ${id}`);
    }
    const contextType = options.contextType || '2d';
    const context = canvas.getContext(contextType, options.contextOptions);
    if (!context) throw new Error(`interface canvas context unavailable: ${id}`);
    return { canvas, context };
  }

  function createScratch(options) {
    if (looksLikeDocument(options)) {
      throw new Error('interface canvas document must be passed as { document }');
    }
    options = options || {};
    const documentRef = hasOwn(options, 'document') ? options.document : root.document;
    if (!documentRef || typeof documentRef.createElement !== 'function') {
      throw new Error('interface canvas document unavailable');
    }
    const canvas = documentRef.createElement('canvas');
    if (!canvas || typeof canvas.getContext !== 'function') {
      throw new Error('interface canvas scratch surface unavailable');
    }
    const contextType = options.contextType || '2d';
    const context = canvas.getContext(contextType, options.contextOptions);
    if (!context) throw new Error('interface canvas scratch context unavailable');
    return { canvas, context };
  }

  const api = { pixelRatio, viewport, resize, bind, createScratch };
  root.YentInterfaceCanvas = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
