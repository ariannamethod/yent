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

  function appendTape(tape, text, limit) {
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
