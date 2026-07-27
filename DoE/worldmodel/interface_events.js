(function (root) {
  'use strict';

  function resolveTarget(options) {
    return (options && options.target) || root;
  }

  function requireTarget(target) {
    if (!target || typeof target.addEventListener !== 'function') {
      throw new Error('interface event target unavailable');
    }
    return target;
  }

  function cleanupFor(target, type, handler, options) {
    return function cleanup() {
      if (target && typeof target.removeEventListener === 'function') {
        target.removeEventListener(type, handler, options);
      }
    };
  }

  function bind(target, type, handler, options, cleanups) {
    target.addEventListener(type, handler, options);
    cleanups.push(cleanupFor(target, type, handler, options));
  }

  function makeCleanup(cleanups) {
    let active = true;
    return function cleanup() {
      if (!active) return;
      active = false;
      for (let i = cleanups.length - 1; i >= 0; i--) cleanups[i]();
    };
  }

  function keyName(event) {
    return event && typeof event.key === 'string' ? event.key.toLowerCase() : '';
  }

  function bindKeyState(options) {
    options = options || {};
    const keys = options.keys;
    if (!keys) throw new Error('interface key state target missing');
    const target = requireTarget(resolveTarget(options));
    const ignore = typeof options.ignore === 'function' ? options.ignore : () => false;
    const cleanups = [];

    bind(target, 'keydown', event => {
      if (ignore(event)) return;
      const key = keyName(event);
      if (key) keys[key] = true;
    }, undefined, cleanups);

    bind(target, 'keyup', event => {
      const key = keyName(event);
      if (key) keys[key] = false;
    }, undefined, cleanups);

    return makeCleanup(cleanups);
  }

  function pointerPoint(event) {
    const x = event && Number.isFinite(event.clientX) ? event.clientX : 0;
    const y = event && Number.isFinite(event.clientY) ? event.clientY : 0;
    return { x, y };
  }

  function bindPointer(options) {
    options = options || {};
    const target = requireTarget(resolveTarget(options));
    const cleanups = [];

    if (typeof options.onMove === 'function') {
      bind(target, 'mousemove', event => options.onMove(pointerPoint(event), event), undefined, cleanups);
    }
    if (typeof options.onLeave === 'function') {
      bind(target, 'mouseout', event => options.onLeave(event), undefined, cleanups);
    }
    if (typeof options.onDown === 'function') {
      bind(target, 'mousedown', event => options.onDown(pointerPoint(event), event), undefined, cleanups);
    }

    return makeCleanup(cleanups);
  }

  const api = { bindKeyState, bindPointer };
  root.YentInterfaceEvents = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
