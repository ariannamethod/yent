(function (root) {
  'use strict';

  function cleanWords(text) {
    return (typeof text === 'string' ? text : '')
      .replace(/[^\p{L}\p{N}_'\- ]/gu, ' ')
      .split(/\s+/)
      .filter(Boolean);
  }

  function tokenTapeText(text) {
    return (typeof text === 'string' ? text : '')
      .replace(/\s+/g, '_')
      .replace(/[^\p{L}\p{N}_./=+\-*#@%&_]/gu, '');
  }

  function appendTape(options) {
    if (typeof options === 'string') {
      throw new Error('interface text appendTape inputs must be passed as { tape, text, limit }');
    }
    options = options || {};
    const tape = options.tape;
    const text = options.text;
    const limit = options.limit;
    const base = typeof tape === 'string' ? tape : '';
    const max = Number.isFinite(limit) && limit > 0 ? Math.floor(limit) : 900;
    const next = tokenTapeText(text);
    if (!next) return base.slice(-max);
    return (base + next + ' ').slice(-max);
  }

  const api = { cleanWords, tokenTapeText, appendTape };
  root.YentInterfaceText = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
