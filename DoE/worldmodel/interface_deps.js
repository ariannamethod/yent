(function (root) {
  'use strict';

  const COMMON = [
    ['interfaceSession', 'YentInterfaceSession'],
    ['interfaceRestore', 'YentInterfaceRestore'],
    ['eventStream', 'YentEventStream'],
    ['chatStream', 'YentChatStream'],
    ['interfaceText', 'YentInterfaceText'],
    ['tokenTelemetry', 'YentTokenTelemetry'],
    ['interfaceState', 'YentInterfaceState'],
    ['interfaceClock', 'YentInterfaceClock'],
    ['interfaceHud', 'YentInterfaceHud'],
    ['interfaceReplay', 'YentInterfaceReplay'],
    ['interfaceInput', 'YentInterfaceInput'],
    ['interfaceEvents', 'YentInterfaceEvents'],
    ['interfaceTurn', 'YentInterfaceTurn'],
    ['interfaceSubmit', 'YentInterfaceSubmit'],
    ['interfaceOutcome', 'YentInterfaceOutcome'],
    ['interfaceRun', 'YentInterfaceRun'],
    ['interfaceBoot', 'YentInterfaceBoot'],
    ['interfaceMath', 'YentInterfaceMath'],
    ['interfaceCanvas', 'YentInterfaceCanvas'],
    ['interfaceStyle', 'YentInterfaceStyle']
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
