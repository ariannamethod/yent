(function (root) {
  'use strict';

  function requireFunction(value, name) {
    if (typeof value !== 'function') throw new Error(`${name} unavailable`);
    return value;
  }

  function hasOwn(value, key) {
    return Object.prototype.hasOwnProperty.call(Object(value), key);
  }

  function requireReplay(options) {
    const replay = (options && options.interfaceReplay) || root.YentInterfaceReplay;
    if (!replay || typeof replay.startIfRequested !== 'function') {
      throw new Error('YentInterfaceReplay helper missing');
    }
    return replay;
  }

  function resolveResizeTarget(options) {
    return hasOwn(options, 'resizeTarget') ? options.resizeTarget : root;
  }

  function bindResize(options) {
    options = options || {};
    const target = resolveResizeTarget(options);
    const resize = options.resize;
    const listenerOptions = options.listenerOptions;
    if (target && typeof target.addEventListener === 'function') {
      target.addEventListener('resize', resize, listenerOptions);
    }
  }

  function bindComposer(options) {
    if (!options || !options.composer) return;
    const generationRun = options.generationRun;
    if (!generationRun || typeof generationRun.bindComposer !== 'function') {
      throw new Error('YentInterfaceRun composer binding unavailable');
    }
    generationRun.bindComposer({
      form: options.composer,
      promptInput: options.promptInput,
      onSubmit: requireFunction(options.generate, 'interface generate')
    });
  }

  function start(options) {
    options = options || {};
    requireFunction(options.restore, 'interface restore')();
    const resize = requireFunction(options.resize, 'interface resize');
    resize();
    const resizeOptions = {
      resize,
      listenerOptions: options.resizeListenerOptions
    };
    if (hasOwn(options, 'resizeTarget')) resizeOptions.resizeTarget = options.resizeTarget;
    bindResize(resizeOptions);
    bindComposer(options);
    requireFunction(options.startAnimation, 'interface animation start')();

    return requireReplay(options).startIfRequested({
      replayMode: !!options.replayMode,
      replayRequest: options.replayRequest,
      promptInput: options.promptInput,
      generationRun: options.generationRun,
      generate: options.generate,
      startDelayMs: options.startDelayMs,
      setTimeout: options.setTimeout
    });
  }

  const api = { start };
  root.YentInterfaceBoot = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
