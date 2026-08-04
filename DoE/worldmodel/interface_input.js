(function (root) {
  'use strict';

  function clampNumber(value, fallback, min, max) {
    const n = Number.isFinite(value) ? value : fallback;
    return Math.max(min, Math.min(max, n));
  }

  function clampInteger(value, fallback, min, max) {
    return Math.floor(clampNumber(value, fallback, min, max));
  }

  function hasDocument(value) {
    return !!value && typeof value.getElementById === 'function';
  }

  function defaultDocument() {
    return root && root.document;
  }

  function hasOwn(value, key) {
    return !!value && Object.prototype.hasOwnProperty.call(value, key);
  }

  function requireNamedDocument(value) {
    if (hasDocument(value)) {
      throw new Error('YentInterfaceInput document must be passed as { document }');
    }
  }

  function optionDocument(options) {
    requireNamedDocument(options);
    if (hasOwn(options, 'document')) return options.document;
    return defaultDocument();
  }

  function bindIds(options) {
    if (!options || typeof options !== 'object') return {};
    if (hasOwn(options, 'composer') || hasOwn(options, 'prompt') || hasOwn(options, 'send')) {
      throw new Error('YentInterfaceInput control ids must be passed as { ids }');
    }
    if (!hasOwn(options, 'ids')) return {};
    const ids = options.ids;
    if (!ids || typeof ids !== 'object') {
      throw new Error('YentInterfaceInput control ids must be passed as an object');
    }
    return ids;
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

  function bindControls(options) {
    options = options || {};
    const doc = optionDocument(options);
    const names = bindIds(options);
    const composer = elementFor(doc, names.composer || 'composer', 'composer');
    const promptInput = elementFor(doc, names.prompt || 'prompt', 'prompt');
    const sendButton = elementFor(doc, names.send || 'send', 'send');
    if (typeof composer.addEventListener !== 'function') {
      throw new Error('YentInterfaceInput composer control cannot receive submit events');
    }
    if (typeof promptInput.value !== 'string') {
      throw new Error('YentInterfaceInput prompt control must expose a string value');
    }
    return { composer, promptInput, sendButton };
  }

  function readParams(options) {
    options = options || {};
    const doc = optionDocument(options);
    const temperature = clampNumber(parseFloat(elementValue(doc, 'temp')), 0.8, 0, 2);
    const maxTokens = clampInteger(parseInt(elementValue(doc, 'max-tokens'), 10), 512, 1, 512);
    return { temperature, maxTokens };
  }

  function isFocused(options) {
    requireNamedDocument(options);
    if (options && typeof options === 'object' && !hasOwn(options, 'control') && !hasOwn(options, 'document')) {
      throw new Error('YentInterfaceInput focus control must be passed as { control }');
    }
    const doc = optionDocument(options || {});
    const target = options && typeof options === 'object' ? options.control : null;
    return !!doc && !!target && doc.activeElement === target;
  }

  function streamFor(options) {
    options = options || {};
    const replayMode = !!options.replayMode;
    const replayRequest = options.replayRequest || {};
    const replay = hasOwn(options, 'interfaceReplay') ? options.interfaceReplay : root.YentInterfaceReplay;
    const chat = hasOwn(options, 'chatStream') ? options.chatStream : root.YentChatStream;

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
