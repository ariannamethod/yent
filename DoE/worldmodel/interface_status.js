(function (root) {
  'use strict';

  function hasDocument(value) {
    return !!value && typeof value.getElementById === 'function';
  }

  function defaultDocument() {
    return root && root.document;
  }

  function element(documentRef, id) {
    if (!id || !documentRef || typeof documentRef.getElementById !== 'function') return null;
    return documentRef.getElementById(id);
  }

  function hasOwn(value, key) {
    return Object.prototype.hasOwnProperty.call(Object(value), key);
  }

  function rejectsDirectIds(options) {
    return ['run', 'note', 'manifest', 'shell'].some(key => hasOwn(options, key));
  }

  function bind(options) {
    if (hasDocument(options)) {
      throw new Error('YentInterfaceStatus document must be passed as { document }');
    }
    options = options || {};
    const documentRef = hasOwn(options, 'document') ? options.document : defaultDocument();
    if (rejectsDirectIds(options)) {
      throw new Error('YentInterfaceStatus ids must be passed as { ids }');
    }
    const ids = options.ids || {};
    return {
      run: element(documentRef, ids.run),
      note: element(documentRef, ids.note),
      manifest: element(documentRef, ids.manifest),
      shell: element(documentRef, ids.shell)
    };
  }

  function looksLikeStatusTarget(value) {
    return !!value && (
      typeof value.textContent === 'string' ||
      !!value.dataset
    );
  }

  function looksLikeStatusLabels(value) {
    return !!value && (
      Object.prototype.hasOwnProperty.call(value, 'manifest') ||
      Object.prototype.hasOwnProperty.call(value, 'shell')
    ) && !Object.prototype.hasOwnProperty.call(value, 'labels');
  }

  function setText(options) {
    if (looksLikeStatusTarget(options)) {
      throw new Error('YentInterfaceStatus text inputs must be passed as { target, text }');
    }
    options = options || {};
    const target = options.target;
    const text = options.text;
    if (target) target.textContent = text == null ? '' : String(text);
  }

  function setActive(options) {
    if (looksLikeStatusTarget(options)) {
      throw new Error('YentInterfaceStatus active inputs must be passed as { target, active }');
    }
    options = options || {};
    const target = options.target;
    const active = options.active;
    if (!target || !target.dataset || typeof active !== 'boolean') return;
    target.dataset.active = active ? 'true' : 'false';
  }

  function setManifest(options) {
    if (looksLikeStatusLabels(options)) {
      throw new Error('YentInterfaceStatus manifest inputs must be passed as { labels, text, active }');
    }
    options = options || {};
    const labels = options.labels || {};
    setText({ target: labels.manifest, text: options.text });
    setActive({ target: labels.shell, active: options.active });
  }

  const api = { bind, setText, setActive, setManifest };
  root.YentInterfaceStatus = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
