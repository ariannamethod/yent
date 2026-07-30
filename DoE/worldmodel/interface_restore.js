(function (root) {
  'use strict';

  function arrayOrEmpty(source) {
    return Array.isArray(source) ? source : [];
  }

  function messageText(message) {
    return message && typeof message.content === 'string' ? message.content : '';
  }

  function lastAssistant(messages) {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg && msg.role === 'assistant') return msg;
    }
    return null;
  }

  function hasOwn(value, key) {
    return value && Object.prototype.hasOwnProperty.call(value, key);
  }

  function load(options) {
    options = options || {};
    if (hasOwn(options, 'session')) {
      throw new Error('session adapter must be passed as { sessionReceipt }');
    }
    if (options.replayMode) return null;
    const session = options.sessionReceipt;
    if (!session || typeof session.load !== 'function') {
      throw new Error('YentInterfaceSession adapter missing');
    }

    const visibleMessages = arrayOrEmpty(session.load());
    if (!visibleMessages.length) return null;

    const restored = {
      visibleMessages,
      combinedText: visibleMessages.map(messageText).join(' '),
      lastAssistant: lastAssistant(visibleMessages)
    };

    if (typeof options.onRestore === 'function') options.onRestore(restored);
    return restored;
  }

  const api = { load };
  root.YentInterfaceRestore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
