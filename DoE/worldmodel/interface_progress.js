(function (root) {
  'use strict';

  function hasOwn(value, key) {
    return !!value && Object.prototype.hasOwnProperty.call(value, key);
  }

  function optionTable(options) {
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
      throw new Error('interface progress options must be passed as an object');
    }
    return options;
  }

  function formatElapsed(milliseconds) {
    const total = Math.max(0, Math.floor((Number.isFinite(milliseconds) ? milliseconds : 0) / 1000));
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }

  function start(options) {
    options = optionTable(options);
    const now = hasOwn(options, 'now') ? options.now : () => Date.now();
    const schedule = hasOwn(options, 'setInterval') ? options.setInterval : root.setInterval;
    const cancel = hasOwn(options, 'clearInterval') ? options.clearInterval : root.clearInterval;
    const onTick = options.onTick;
    if (typeof now !== 'function' || typeof schedule !== 'function' || typeof cancel !== 'function') {
      throw new Error('interface progress clock unavailable');
    }
    if (typeof onTick !== 'function') throw new Error('interface progress onTick callback missing');
    const intervalMs = Number.isFinite(options.intervalMs)
      ? Math.max(250, Math.floor(options.intervalMs))
      : 1000;
    const beganAt = now();
    let stopped = false;
    let timer = null;

    function tick() {
      if (stopped) return;
      const elapsedMs = Math.max(0, now() - beganAt);
      onTick({ elapsedMs, label: formatElapsed(elapsedMs) });
    }

    tick();
    timer = schedule(tick, intervalMs);
    return {
      stop() {
        if (stopped) return false;
        stopped = true;
        cancel(timer);
        return true;
      }
    };
  }

  const api = { formatElapsed, start };
  root.YentInterfaceProgress = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
