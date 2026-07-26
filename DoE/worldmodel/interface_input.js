(function (root) {
  'use strict';

  function clampNumber(value, fallback, min, max) {
    const n = Number.isFinite(value) ? value : fallback;
    return Math.max(min, Math.min(max, n));
  }

  function clampInteger(value, fallback, min, max) {
    return Math.floor(clampNumber(value, fallback, min, max));
  }

  function elementValue(documentRef, id) {
    if (!documentRef || typeof documentRef.getElementById !== 'function') return '';
    const el = documentRef.getElementById(id);
    if (!el || typeof el.value !== 'string') return '';
    return el.value;
  }

  function elementFor(documentRef, id, label) {
    if (!documentRef || typeof documentRef.getElementById !== 'function') {
      throw new Error('YentInterfaceInput document unavailable');
    }
    const el = documentRef.getElementById(id);
    if (!el) throw new Error(`YentInterfaceInput ${label} control unavailable: ${id}`);
    return el;
  }

  function bindControls(documentRef, ids) {
    ids = ids || {};
    const composer = elementFor(documentRef, ids.composer || 'composer', 'composer');
    const promptInput = elementFor(documentRef, ids.prompt || 'prompt', 'prompt');
    const sendButton = elementFor(documentRef, ids.send || 'send', 'send');
    if (typeof composer.addEventListener !== 'function') {
      throw new Error('YentInterfaceInput composer control cannot receive submit events');
    }
    if (typeof promptInput.value !== 'string') {
      throw new Error('YentInterfaceInput prompt control must expose a string value');
    }
    return { composer, promptInput, sendButton };
  }

  function readParams(documentRef) {
    const temperature = clampNumber(parseFloat(elementValue(documentRef, 'temp')), 0.8, 0, 2);
    const maxTokens = clampInteger(parseInt(elementValue(documentRef, 'max-tokens'), 10), 512, 1, 512);
    return { temperature, maxTokens };
  }

  function isFocused(documentRef, control) {
    return !!documentRef && !!control && documentRef.activeElement === control;
  }

  function streamFor(options) {
    options = options || {};
    const replayMode = !!options.replayMode;
    const replayRequest = options.replayRequest || {};
    const replay = options.interfaceReplay || root.YentInterfaceReplay;
    const chat = options.chatStream || root.YentChatStream;

    if (replayMode) {
      if (!replay || typeof replay.play !== 'function') {
        throw new Error('YentInterfaceReplay helper missing');
      }
      return streamOptions => replay.play(Object.assign({}, streamOptions, {
        scenario: replayRequest.name,
        delayMs: replayRequest.delayMs
      }));
    }

    if (!chat || typeof chat.stream !== 'function') {
      throw new Error('YentChatStream helper missing');
    }
    return streamOptions => chat.stream(streamOptions);
  }

  const api = { bindControls, readParams, isFocused, streamFor };
  root.YentInterfaceInput = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
