(function (root) {
  'use strict';

  function hasOwn(value, key) {
    return Object.prototype.hasOwnProperty.call(Object(value), key);
  }

  function nowFrom(options) {
    const perf = hasOwn(options, 'performance') ? options.performance : root.performance;
    if (perf && typeof perf.now === 'function') return perf.now();
    return Date.now();
  }

  function create(options) {
    options = options || {};
    const minElapsedSeconds =
      Number.isFinite(options.minElapsedSeconds) && options.minElapsedSeconds > 0
        ? options.minElapsedSeconds
        : 0.01;
    let startedAt = 0;
    let tokens = 0;

    function reset(at) {
      startedAt = Number.isFinite(at) ? at : nowFrom(options);
      tokens = 0;
    }

    function tick(at) {
      tokens += 1;
      const current = Number.isFinite(at) ? at : nowFrom(options);
      const elapsed = Math.max((current - startedAt) / 1000, minElapsedSeconds);
      return tokens / elapsed;
    }

    function now(at) {
      return Number.isFinite(at) ? at : nowFrom(options);
    }

    function count() {
      return tokens;
    }

    function started() {
      return startedAt;
    }

    reset(options.startedAt);
    return { reset, tick, count, started, now };
  }

  const api = { create };
  root.YentInterfaceClock = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
