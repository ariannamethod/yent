(function (root) {
  'use strict';

  const DEFAULT_KEY = 'yent.interface.session.v1';
  const DEFAULT_LIMIT = 12;
  const DEFAULT_CONTENT_LIMIT = 12000;

  function optionNumber(options, key, fallback) {
    const value = options && options[key];
    return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback;
  }

  function hasOwn(value, key) {
    return Object.prototype.hasOwnProperty.call(Object(value), key);
  }

  function looksLikeStorage(value) {
    return !!value && !hasOwn(value, 'storage') && (
      typeof value.getItem === 'function' ||
      typeof value.setItem === 'function'
    );
  }

  function normalize(options) {
    if (Array.isArray(options)) {
      throw new Error('session normalize inputs must be passed as { messages }');
    }
    options = options || {};
    const source = options.messages;
    if (!Array.isArray(source)) return [];
    const limit = optionNumber(options, 'limit', DEFAULT_LIMIT);
    const contentLimit = optionNumber(options, 'contentLimit', DEFAULT_CONTENT_LIMIT);
    const out = [];
    for (const msg of source) {
      if (!msg || (msg.role !== 'user' && msg.role !== 'assistant')) continue;
      if (typeof msg.content !== 'string' || !msg.content.trim()) continue;
      out.push({ role: msg.role, content: msg.content.slice(0, contentLimit) });
    }
    return out.slice(-limit);
  }

  function defaultStorage() {
    try {
      return root.sessionStorage || null;
    } catch (_) {
      return null;
    }
  }

  function storageOrDefault(storage) {
    return storage || defaultStorage();
  }

  function load(options) {
    if (looksLikeStorage(options)) {
      throw new Error('session load inputs must be passed as { storage }');
    }
    options = options || {};
    const target = storageOrDefault(options.storage);
    if (!target) return [];
    const key = (options && options.key) || DEFAULT_KEY;
    try {
      const raw = target.getItem(key);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return normalize(Object.assign({}, options, { messages: parsed && parsed.messages }));
    } catch (_) {
      return [];
    }
  }

  function save(options) {
    if (looksLikeStorage(options) || Array.isArray(options)) {
      throw new Error('session save inputs must be passed as { storage, messages }');
    }
    options = options || {};
    const target = storageOrDefault(options.storage);
    if (!target) return false;
    const key = (options && options.key) || DEFAULT_KEY;
    try {
      target.setItem(key, JSON.stringify({
        savedAt: Date.now(),
        messages: normalize(options)
      }));
      return true;
    } catch (_) {
      return false;
    }
  }

  function createAdapter(options) {
    options = options || {};
    const storage = storageOrDefault(options.storage);
    const replayMode = !!(options.replayMode || options.replay);
    const now = typeof options.now === 'function' ? options.now : () => Date.now();
    const saveIntervalMs = Number.isFinite(options.saveIntervalMs)
      ? Math.max(0, Math.floor(options.saveIntervalMs))
      : 250;
    let lastSaveAt = 0;

    return {
      normalize: source => normalize(Object.assign({}, options, { messages: source })),
      load() {
        if (replayMode) return [];
        return load(Object.assign({}, options, { storage }));
      },
      save(nextMessages, force = false) {
        if (replayMode) return false;
        const at = now();
        if (!force && at - lastSaveAt < saveIntervalMs) return false;
        if (!save(Object.assign({}, options, { storage, messages: nextMessages }))) return false;
        lastSaveAt = at;
        return true;
      },
      commitUser(modelMessages, visibleMessages, text) {
        const message = { role: 'user', content: typeof text === 'string' ? text : '' };
        const nextModel = Array.isArray(modelMessages) ? modelMessages.concat(message) : [message];
        const nextVisible = normalize(Object.assign({}, options, {
          messages: (Array.isArray(visibleMessages) ? visibleMessages : []).concat(message)
        }));
        this.save(nextVisible, true);
        return { messages: nextModel, visibleMessages: nextVisible, message };
      },
      previewAssistant(visibleMessages, text) {
        const message = { role: 'assistant', content: typeof text === 'string' ? text : '' };
        return this.save((Array.isArray(visibleMessages) ? visibleMessages : []).concat(message));
      },
      commitAssistant(modelMessages, visibleMessages, text) {
        if (typeof text !== 'string' || !text.trim()) {
          return {
            messages: Array.isArray(modelMessages) ? modelMessages : [],
            visibleMessages: Array.isArray(visibleMessages) ? visibleMessages : [],
            committed: false
          };
        }
        const message = { role: 'assistant', content: text };
        const nextModel = Array.isArray(modelMessages) ? modelMessages.concat(message) : [message];
        const nextVisible = normalize(Object.assign({}, options, {
          messages: (Array.isArray(visibleMessages) ? visibleMessages : []).concat(message)
        }));
        this.save(nextVisible, true);
        return { messages: nextModel, visibleMessages: nextVisible, message, committed: true };
      }
    };
  }

  const api = {
    KEY: DEFAULT_KEY,
    LIMIT: DEFAULT_LIMIT,
    CONTENT_LIMIT: DEFAULT_CONTENT_LIMIT,
    normalize,
    load,
    save,
    createAdapter
  };

  root.YentInterfaceSession = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
