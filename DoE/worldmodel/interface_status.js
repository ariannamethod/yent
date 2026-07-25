(function (root) {
  'use strict';

  function element(documentRef, id) {
    if (!id || !documentRef || typeof documentRef.getElementById !== 'function') return null;
    return documentRef.getElementById(id);
  }

  function bind(documentRef, ids) {
    ids = ids || {};
    return {
      run: element(documentRef, ids.run),
      note: element(documentRef, ids.note),
      manifest: element(documentRef, ids.manifest),
      shell: element(documentRef, ids.shell)
    };
  }

  function setText(target, text) {
    if (target) target.textContent = text == null ? '' : String(text);
  }

  function setActive(target, active) {
    if (!target || !target.dataset || typeof active !== 'boolean') return;
    target.dataset.active = active ? 'true' : 'false';
  }

  function setManifest(labels, text, active) {
    labels = labels || {};
    setText(labels.manifest, text);
    setActive(labels.shell, active);
  }

  const api = { bind, setText, setActive, setManifest };
  root.YentInterfaceStatus = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
