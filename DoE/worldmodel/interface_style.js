(function (root) {
  'use strict';

  const FALLBACKS = {
    mono: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace',
    serif: 'ui-serif, Georgia, "Times New Roman", serif',
    sans: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
  };

  function normalizeName(name) {
    return String(name || '').replace(/^--/, '').trim();
  }

  function hasOwn(value, key) {
    return !!value && Object.prototype.hasOwnProperty.call(value, key);
  }

  function create(options) {
    options = options || {};
    const documentRef = hasOwn(options, 'document') ? options.document : root.document;
    const getComputedStyleRef = hasOwn(options, 'getComputedStyle') ? options.getComputedStyle : root.getComputedStyle;
    const host = hasOwn(options, 'host') ? options.host : (documentRef && documentRef.documentElement) || null;
    const cache = Object.create(null);

    function family(name, fallback) {
      const key = normalizeName(name);
      if (!key) return fallback || '';
      if (Object.prototype.hasOwnProperty.call(cache, key)) return cache[key];

      let value = '';
      if (host && typeof getComputedStyleRef === 'function') {
        const style = getComputedStyleRef(host);
        if (style && typeof style.getPropertyValue === 'function') {
          value = String(style.getPropertyValue(`--${key}`) || '').trim();
        }
      }
      cache[key] = value || fallback || FALLBACKS[key] || FALLBACKS.mono;
      return cache[key];
    }

    function reset() {
      for (const key of Object.keys(cache)) delete cache[key];
    }

    return {
      family,
      mono: () => family('mono', FALLBACKS.mono),
      serif: () => family('serif', FALLBACKS.serif),
      sans: () => family('sans', FALLBACKS.sans),
      reset
    };
  }

  const api = { create, FALLBACKS };
  root.YentInterfaceStyle = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
