(function (root) {
  'use strict';

  function controllerCtor(options) {
    const Controller = (options && options.AbortController) || root.AbortController;
    if (typeof Controller !== 'function') throw new Error('AbortController unavailable');
    return Controller;
  }

  function setButton(button, text) {
    if (!button) return;
    button.textContent = text;
    if ('disabled' in button) button.disabled = false;
  }

  function hasOwn(value, key) {
    return !!value && Object.prototype.hasOwnProperty.call(value, key);
  }

  function create(options) {
    options = options || {};
    const button = options.button || null;
    const idleText = typeof options.idleText === 'string' ? options.idleText : 'SEND';
    const busyText = typeof options.busyText === 'string' ? options.busyText : 'STOP';
    const Controller = controllerCtor(options);
    let running = false;
    let aborter = null;
    let runId = 0;

    function begin() {
      if (running) throw new Error('generation already running');
      aborter = new Controller();
      running = true;
      runId++;
      setButton(button, busyText);
      return {
        id: runId,
        controller: aborter,
        signal: aborter.signal
      };
    }

    function finish(run) {
      if (!running || !run || !Number.isFinite(run.id) || run.id !== runId) return false;
      running = false;
      aborter = null;
      setButton(button, idleText);
      return true;
    }

    function abortRunning() {
      if (!running) return false;
      if (aborter && typeof aborter.abort === 'function') aborter.abort();
      return true;
    }

    function looksLikeForm(value) {
      return !!value && typeof value.addEventListener === 'function';
    }

    function bindComposer(options) {
      if (looksLikeForm(options)) {
        throw new Error('composer binding inputs must be passed as { form, promptInput, onSubmit }');
      }
      options = options || {};
      if (hasOwn(options, 'input')) {
        throw new Error('composer prompt input must be passed as { promptInput }');
      }
      const form = options.form;
      const input = options.promptInput;
      const onSubmit = options.onSubmit;
      if (!form || typeof form.addEventListener !== 'function') {
        throw new Error('composer form unavailable');
      }
      if (!input || typeof input.value !== 'string') {
        throw new Error('composer input unavailable');
      }
      if (typeof onSubmit !== 'function') {
        throw new Error('composer submit handler unavailable');
      }
      form.addEventListener('submit', event => {
        if (event && typeof event.preventDefault === 'function') event.preventDefault();
        if (abortRunning()) return;
        const text = input.value.trim();
        if (!text) return;
        input.value = '';
        onSubmit(text);
      });
    }

    return {
      begin,
      finish,
      abortRunning,
      bindComposer,
      isRunning: () => running
    };
  }

  const api = { create };
  root.YentInterfaceRun = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
