(function (root) {
  'use strict';

  const DEFAULT_TOP_LIMIT = 32;
  const DEFAULT_WORD_LIMIT = 18;
  const DEFAULT_WORDS_PER_TOKEN = 2;
  const textHelper = root.YentInterfaceText ||
    (typeof require === 'function' ? require('./interface_text.js') : null);

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function isFiniteNumber(value) {
    return typeof value === 'number' && Number.isFinite(value);
  }

  function integerField(source, key) {
    const value = source && source[key];
    return isFiniteNumber(value) ? Math.floor(value) : null;
  }

  function floatField(source, key, min, max) {
    const value = source && source[key];
    if (!isFiniteNumber(value)) return null;
    return Number.isFinite(min) && Number.isFinite(max) ? clamp(value, min, max) : value;
  }

  function optionInteger(options, key, fallback, min, max) {
    const value = options && options[key];
    if (!isFiniteNumber(value)) return fallback;
    return Math.max(min, Math.min(max, Math.floor(value)));
  }

  function cleanWords(text) {
    if (!textHelper || typeof textHelper.cleanWords !== 'function') {
      throw new Error('YentInterfaceText helper missing');
    }
    return textHelper.cleanWords(text);
  }

  function normalizeTopTokens(source, options) {
    const limit = optionInteger(options, 'topLimit', DEFAULT_TOP_LIMIT, 1, 256);
    if (!Array.isArray(source)) return [];

    const out = [];
    for (let i = 0; i < source.length && out.length < limit; i++) {
      const item = source[i];
      if (!item || typeof item.token !== 'string') continue;
      const prob = floatField(item, 'prob', 0, 1);
      const rank = integerField(item, 'rank');
      out.push({
        token: item.token,
        prob: prob === null ? 0 : prob,
        logprob: floatField(item, 'logprob', -Infinity, Infinity),
        rank: Math.max(0, rank === null ? i + 1 : rank),
        selected: item.selected === true
      });
    }
    return out;
  }

  function normalize(data, options) {
    data = data || {};
    const topTokens = normalizeTopTokens(data.top_tokens, options);
    const tokenId = integerField(data, 'token_id');
    const step = integerField(data, 'step');
    const experts = integerField(data, 'experts');
    const selectedRank = integerField(data, 'selected_rank');
    const selectedProb = floatField(data, 'selected_prob', 0, 1);
    const candidateTailMass = floatField(data, 'candidate_tail_mass', 0, 1);
    const selectedLogprob = floatField(data, 'selected_logprob', -Infinity, Infinity);
    const debt = floatField(data, 'debt', 0, 1);
    const prophecyDebt = floatField(data, 'prophecy_debt', 0, 1);
    const consensus = floatField(data, 'consensus', 0, 1);
    const fieldHealth = floatField(data, 'field_health', 0, 1);
    const entropy = floatField(data, 'entropy', -Infinity, Infinity);
    const resonance = floatField(data, 'resonance', -Infinity, Infinity);
    const emergence = floatField(data, 'emergence', -Infinity, Infinity);
    const temperature = floatField(data, 'temperature', 0, 2);

    return {
      token: typeof data.token === 'string' ? data.token : '',
      hasToken: typeof data.token === 'string',
      tokenId: tokenId === null ? 0 : tokenId,
      hasTokenId: tokenId !== null,
      step: step === null ? 0 : Math.max(0, step),
      hasStep: step !== null,
      experts: experts === null ? 0 : Math.max(0, experts),
      hasExperts: experts !== null,
      debt: debt === null ? 0 : debt,
      hasDebt: debt !== null,
      prophecyDebt: prophecyDebt === null ? 0 : prophecyDebt,
      hasProphecyDebt: prophecyDebt !== null,
      consensus: consensus === null ? 0 : consensus,
      hasConsensus: consensus !== null,
      fieldHealth: fieldHealth === null ? 1 : fieldHealth,
      hasFieldHealth: fieldHealth !== null,
      entropy: entropy === null ? 0 : Math.max(0, entropy),
      hasEntropy: entropy !== null,
      resonance: resonance === null ? 0 : resonance,
      hasResonance: resonance !== null,
      emergence: emergence === null ? 0 : emergence,
      hasEmergence: emergence !== null,
      temperature: temperature === null ? 0.8 : temperature,
      hasTemperature: temperature !== null,
      selectedProb: selectedProb === null ? 0 : selectedProb,
      hasSelectedProb: selectedProb !== null,
      selectedLogprob: selectedLogprob === null ? 0 : selectedLogprob,
      hasSelectedLogprob: selectedLogprob !== null,
      selectedRank: selectedRank === null ? 0 : Math.max(0, selectedRank),
      hasSelectedRank: selectedRank !== null,
      candidateTailMass: candidateTailMass === null ? 0 : candidateTailMass,
      hasCandidateTailMass: candidateTailMass !== null,
      topTokens,
      hasTopTokens: topTokens.length > 0,
      hasCandidateTelemetry: selectedProb !== null || selectedRank !== null ||
        candidateTailMass !== null || topTokens.length > 0
    };
  }

  function telemetryOrNormalize(data, options) {
    if (data && Array.isArray(data.topTokens)) return data;
    return normalize(data, options);
  }

  function candidateWords(data, options) {
    const telemetry = telemetryOrNormalize(data, options);
    const limit = optionInteger(options, 'limit', DEFAULT_WORD_LIMIT, 1, 256);
    const wordsPerToken = optionInteger(options, 'wordsPerToken', DEFAULT_WORDS_PER_TOKEN, 1, 8);
    const includeSelected = options && options.includeSelected === true;
    const out = [];

    for (const item of telemetry.topTokens) {
      if (!includeSelected && item.selected) continue;
      const words = cleanWords(item.token).slice(0, wordsPerToken);
      for (const word of words) {
        out.push({
          word,
          token: item.token,
          prob: item.prob,
          logprob: item.logprob === null ? Math.log(Math.max(item.prob, 1e-9)) : item.logprob,
          rank: item.rank,
          selected: item.selected
        });
        if (out.length >= limit) return out;
      }
    }
    return out;
  }

  function candidateText(data, options) {
    const sanitizer = options && typeof options.sanitizer === 'function' ? options.sanitizer : null;
    const parts = [];
    for (const entry of candidateWords(data, options)) {
      const text = sanitizer ? sanitizer(entry.word) : entry.word;
      if (text) parts.push(text);
    }
    return parts.join(' ');
  }

  function metricProb(value) {
    if (!isFiniteNumber(value)) return '-';
    const v = clamp(value, 0, 1);
    return v.toFixed(v < 0.01 ? 4 : 3);
  }

  function resetCandidateState(state) {
    if (!state) return state;
    state.selectedProb = 0;
    state.selectedRank = 0;
    state.candidateTail = 0;
    state.hasCandidateTelemetry = false;
    return state;
  }

  function applyCandidateState(state, data, options) {
    if (!state) throw new Error('candidate state missing');
    options = options || {};
    const telemetry = telemetryOrNormalize(data, options);
    const selectedProb = telemetry.hasSelectedProb ? telemetry.selectedProb :
      (isFiniteNumber(state.selectedProb) ? state.selectedProb : 0);
    const selectedRank = telemetry.hasSelectedRank ? telemetry.selectedRank :
      (isFiniteNumber(state.selectedRank) ? state.selectedRank : 0);
    const tailMass = telemetry.candidateTailMass;

    state.candidateTail = tailMass;
    state.selectedProb = selectedProb;
    state.selectedRank = selectedRank;
    state.hasCandidateTelemetry = !!state.hasCandidateTelemetry ||
      telemetry.hasCandidateTelemetry ||
      options.hasCandidates === true ||
      (isFiniteNumber(options.candidateCount) && options.candidateCount > 0);

    return {
      candidateTail: tailMass,
      selectedProb,
      selectedRank,
      hasSelectedProb: telemetry.hasSelectedProb,
      hasSelectedRank: telemetry.hasSelectedRank,
      hasCandidateTelemetry: state.hasCandidateTelemetry
    };
  }

  const api = {
    DEFAULT_TOP_LIMIT,
    DEFAULT_WORD_LIMIT,
    DEFAULT_WORDS_PER_TOKEN,
    cleanWords,
    normalize,
    normalizeTopTokens,
    candidateWords,
    candidateText,
    metricProb,
    resetCandidateState,
    applyCandidateState
  };

  root.YentTokenTelemetry = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
