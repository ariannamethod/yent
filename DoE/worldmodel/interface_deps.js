(function (root) {
  'use strict';

  const COMMON = [
    ['interfaceSession', 'YentInterfaceSession'],
    ['eventStream', 'YentEventStream'],
    ['chatStream', 'YentChatStream'],
    ['tokenTelemetry', 'YentTokenTelemetry'],
    ['interfaceHud', 'YentInterfaceHud'],
    ['interfaceReplay', 'YentInterfaceReplay'],
    ['interfaceInput', 'YentInterfaceInput'],
    ['interfaceTurn', 'YentInterfaceTurn'],
    ['interfaceRun', 'YentInterfaceRun'],
    ['interfaceBoot', 'YentInterfaceBoot'],
    ['interfaceMath', 'YentInterfaceMath']
  ];

  function requireHelper(host, globalName) {
    const value = host && host[globalName];
    if (!value) throw new Error(`${globalName} helper missing`);
    return value;
  }

  function load(options) {
    options = options || {};
    const host = options.root || root;
    const deps = {};
    for (const [key, globalName] of COMMON) {
      deps[key] = requireHelper(host, globalName);
    }
    if (options.worldGeometry) {
      deps.worldGeometry = requireHelper(host, 'YentWorldmodelGeometry');
    }
    return deps;
  }

  const api = { load };
  root.YentInterfaceDeps = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
