(function (root) {
  'use strict';

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

  const api = { setText, scrollBottom, setTextAndScroll };
  root.YentInterfaceOutput = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
