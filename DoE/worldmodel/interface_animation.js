(function (root) {
  'use strict';

  function frameRequestFor(options) {
    const requestFrame = (options && options.requestAnimationFrame) || root.requestAnimationFrame;
    if (typeof requestFrame !== 'function') throw new Error('requestAnimationFrame unavailable');
    return requestFrame;
  }

  function create(options) {
    const frameRequest = frameRequestFor(options);

    function requestFrame(callback) {
      if (typeof callback !== 'function') throw new Error('animation frame callback unavailable');
      return frameRequest(callback);
    }

    return {
      requestFrame,
      start: requestFrame
    };
  }

  const api = { create };
  root.YentInterfaceAnimation = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
