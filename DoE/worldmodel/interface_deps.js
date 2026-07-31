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
    ['interfaceStatus', 'YentInterfaceStatus'],
    ['interfaceOutput', 'YentInterfaceOutput'],
    ['interfaceHud', 'YentInterfaceHud'],
    ['interfaceReplay', 'YentInterfaceReplay'],
    ['interfaceInput', 'YentInterfaceInput'],
    ['interfaceEvents', 'YentInterfaceEvents'],
    ['interfaceTurn', 'YentInterfaceTurn'],
    ['interfaceSubmit', 'YentInterfaceSubmit'],
    ['interfaceOutcome', 'YentInterfaceOutcome'],
    ['interfaceRun', 'YentInterfaceRun'],
    ['interfaceBoot', 'YentInterfaceBoot'],
    ['interfaceAnimation', 'YentInterfaceAnimation'],
    ['interfaceMath', 'YentInterfaceMath'],
    ['interfaceCanvas', 'YentInterfaceCanvas'],
    ['interfaceStyle', 'YentInterfaceStyle']
  ];

  function requireHelper(host, globalName) {
    const value = host && host[globalName];
    if (!value) throw new Error(`${globalName} helper missing`);
    return value;
  }

  function hasOwn(value, key) {
    return !!value && Object.prototype.hasOwnProperty.call(value, key);
  }

  function looksLikeDependencyHost(value) {
    if (!value || typeof value !== 'object') return false;
    if (hasOwn(value, 'YentWorldmodelGeometry') ||
        hasOwn(value, 'YentInterfaceTranscript')) return true;
    return COMMON.some(([, globalName]) => hasOwn(value, globalName));
  }

  function load(options) {
    if (looksLikeDependencyHost(options)) {
      throw new Error('interface dependency root must be passed as { root }');
    }
    options = options || {};
    const host = hasOwn(options, 'root') ? options.root : root;
    const deps = {};
    for (const [key, globalName] of COMMON) {
      deps[key] = requireHelper(host, globalName);
    }
    if (options.worldGeometry) {
      deps.worldGeometry = requireHelper(host, 'YentWorldmodelGeometry');
    }
    if (options.transcript) {
      deps.interfaceTranscript = requireHelper(host, 'YentInterfaceTranscript');
    }
    return deps;
  }

  const api = { load };
  root.YentInterfaceDeps = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
