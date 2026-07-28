(function (root) {
  'use strict';

  function defaultDocument() {
    return root && root.document;
  }

  function hasDocument(value) {
    return !!value && typeof value.getElementById === 'function';
  }

  function bind(options) {
    if (hasDocument(options)) {
      throw new Error('YentInterfaceOutput document must be passed as { document }');
    }
    if (typeof options === 'string') {
      throw new Error('YentInterfaceOutput target id must be passed as { id }');
    }
    const id = options && options.id;
    const documentRef = (options && options.document) || defaultDocument();
    if (!documentRef || typeof documentRef.getElementById !== 'function') {
      throw new Error('YentInterfaceOutput document unavailable');
    }
    const target = documentRef.getElementById(id);
    if (!target) throw new Error(`YentInterfaceOutput target unavailable: ${id}`);
    if (typeof target.textContent !== 'string') {
      throw new Error(`YentInterfaceOutput target must expose textContent: ${id}`);
    }
    return target;
  }

  function setText(target, text) {
    if (!target) return;
    target.textContent = text == null ? '' : String(text);
  }

  function scrollBottom(target) {
    if (!target) return;
    const height = target.scrollHeight;
    if (typeof height === 'number' && Number.isFinite(height)) {
      target.scrollTop = height;
    }
  }

  function setTextAndScroll(target, text, scrollTarget) {
    setText(target, text);
    scrollBottom(scrollTarget || target);
  }

  const api = { bind, setText, scrollBottom, setTextAndScroll };
  root.YentInterfaceOutput = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
