(function (root) {
  'use strict';

  const BASELINE = Object.freeze({
    debt: 0.0,
    consensus: 0.62,
    field: 1.0,
    experts: 0,
    tokps: 0.0,
    step: 0,
    entropy: 0.0,
    selectedProb: 0.0,
    selectedRank: 0,
    candidateTail: 0.0,
    hasCandidateTelemetry: false
  });

  function overrideTable(overrides) {
    if (overrides === undefined) return {};
    if (!overrides || typeof overrides !== 'object') {
      throw new Error('state overrides must be passed as an object');
    }
    return overrides;
  }

  function create(overrides) {
    return Object.assign({}, BASELINE, overrideTable(overrides));
  }

  const api = { BASELINE, create };
  root.YentInterfaceState = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
