(function (root) {
  'use strict';

  function hasDocument(value) {
    return !!value && typeof value.getElementById === 'function';
  }

  function defaultDocument() {
    return root && root.document;
  }

  function element(documentRef, id) {
    if (!documentRef || typeof documentRef.getElementById !== 'function') return null;
    return documentRef.getElementById(id);
  }

  function bind(documentRef, ids) {
    if (arguments.length === 0 || (arguments.length === 1 && !hasDocument(documentRef))) {
      ids = documentRef || {};
      documentRef = defaultDocument();
    }
    ids = ids || {};
    return {
      tok: element(documentRef, ids.tok || 'hud-tok'),
      exp: element(documentRef, ids.exp || 'hud-exp'),
      step: element(documentRef, ids.step || 'hud-step'),
      ent: element(documentRef, ids.ent || 'hud-ent'),
      debt: element(documentRef, ids.debt || 'hud-debt'),
      cons: element(documentRef, ids.cons || 'hud-cons'),
      field: element(documentRef, ids.field || 'hud-field'),
      prob: element(documentRef, ids.prob || 'hud-prob'),
      rank: element(documentRef, ids.rank || 'hud-rank'),
      tail: element(documentRef, ids.tail || 'hud-tail')
    };
  }

  function write(target, value) {
    if (target) target.textContent = value;
  }

  function finiteNumber(value, fallback) {
    return Number.isFinite(value) ? value : fallback;
  }

  function fixed(value, digits, fallback) {
    return finiteNumber(value, fallback).toFixed(digits);
  }

  function positiveIntegerText(value) {
    return Number.isFinite(value) && value > 0 ? String(Math.floor(value)) : '-';
  }

  function probabilityText(options, value) {
    const telemetry = (options && options.tokenTelemetry) || root.YentTokenTelemetry;
    if (!telemetry || typeof telemetry.metricProb !== 'function') {
      throw new Error('YentTokenTelemetry helper missing');
    }
    return telemetry.metricProb(value);
  }

  function render(hud, state, options) {
    hud = hud || {};
    state = state || {};
    const hasCandidates = !!state.hasCandidateTelemetry;
    write(hud.tok, fixed(state.tokps, 1, 0));
    write(hud.exp, positiveIntegerText(state.experts));
    write(hud.step, String(Math.max(0, Math.floor(finiteNumber(state.step, 0)))));
    write(hud.ent, fixed(state.entropy, 2, 0));
    write(hud.debt, fixed(state.debt, 2, 0));
    write(hud.cons, fixed(state.consensus, 2, 0));
    write(hud.field, fixed(state.field, 2, 0));
    write(hud.prob, hasCandidates ? probabilityText(options, state.selectedProb) : '-');
    write(hud.rank, hasCandidates ? positiveIntegerText(state.selectedRank) : '-');
    write(hud.tail, hasCandidates ? fixed(state.candidateTail, 2, 0) : '-');
  }

  const api = { bind, render };
  root.YentInterfaceHud = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
