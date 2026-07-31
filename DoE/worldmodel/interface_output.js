(function (root) {
  'use strict';

  function defaultDocument() {
    return root && root.document;
  }

  function hasDocument(value) {
    return !!value && typeof value.getElementById === 'function';
  }

  function hasOwn(value, key) {
    return !!value && Object.prototype.hasOwnProperty.call(value, key);
  }

  function bind(options) {
    if (hasDocument(options)) {
      throw new Error('YentInterfaceOutput document must be passed as { document }');
    }
    if (typeof options === 'string') {
      throw new Error('YentInterfaceOutput target id must be passed as { id }');
    }
    const id = options && options.id;
    const documentRef = hasOwn(options, 'document') ? options.document : defaultDocument();
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

  function looksLikeOutputTarget(value) {
    return !!value && (
      typeof value.textContent === 'string' ||
      Object.prototype.hasOwnProperty.call(value, 'scrollTop') ||
      Object.prototype.hasOwnProperty.call(value, 'scrollHeight')
    );
  }

  function setText(options) {
    if (looksLikeOutputTarget(options)) {
      throw new Error('YentInterfaceOutput text inputs must be passed as { target, text }');
    }
    options = options || {};
    const target = options.target;
    const text = options.text;
    if (!target) return;
    target.textContent = text == null ? '' : String(text);
  }

  function scrollBottom(options) {
    if (looksLikeOutputTarget(options)) {
      throw new Error('YentInterfaceOutput scroll target must be passed as { target }');
    }
    options = options || {};
    const target = options.target;
    if (!target) return;
    const height = target.scrollHeight;
    if (typeof height === 'number' && Number.isFinite(height)) {
      target.scrollTop = height;
    }
  }

  function setTextAndScroll(options) {
    if (looksLikeOutputTarget(options)) {
      throw new Error('YentInterfaceOutput text/scroll inputs must be passed as { target, text, scrollTarget }');
    }
    options = options || {};
    const target = options.target;
    setText({ target, text: options.text });
    scrollBottom({ target: options.scrollTarget || target });
  }

  const api = { bind, setText, scrollBottom, setTextAndScroll };
  root.YentInterfaceOutput = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
