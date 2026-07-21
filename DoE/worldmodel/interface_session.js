(function (root) {
  'use strict';

  const DEFAULT_KEY = 'yent.interface.session.v1';
  const DEFAULT_LIMIT = 12;
  const DEFAULT_CONTENT_LIMIT = 12000;

  function optionNumber(options, key, fallback) {
    const value = options && options[key];
    return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback;
  }

  function normalize(source, options) {
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

  function load(storage, options) {
    const target = storageOrDefault(storage);
    if (!target) return [];
    const key = (options && options.key) || DEFAULT_KEY;
    try {
      const raw = target.getItem(key);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return normalize(parsed && parsed.messages, options);
    } catch (_) {
      return [];
    }
  }

  function save(storage, nextMessages, options) {
    const target = storageOrDefault(storage);
    if (!target) return false;
    const key = (options && options.key) || DEFAULT_KEY;
    try {
      target.setItem(key, JSON.stringify({
        savedAt: Date.now(),
        messages: normalize(nextMessages, options)
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
      normalize: source => normalize(source, options),
      load() {
        if (replayMode) return [];
        return load(storage, options);
      },
      save(nextMessages, force = false) {
        if (replayMode) return false;
        const at = now();
        if (!force && at - lastSaveAt < saveIntervalMs) return false;
        if (!save(storage, nextMessages, options)) return false;
        lastSaveAt = at;
        return true;
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
