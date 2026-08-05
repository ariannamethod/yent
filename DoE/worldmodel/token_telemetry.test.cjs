const assert = require('node:assert/strict');
const telemetry = require('./token_telemetry.js');

{
  const t = telemetry.normalize();
  assert.equal(t.hasCandidateTelemetry, false);
  assert.equal(t.token, '');
  assert.deepEqual(t.topTokens, []);
}

{
  assert.throws(() => telemetry.normalize(null), /telemetry data must be passed as an object/);
  assert.throws(() => telemetry.normalize('token'), /telemetry data must be passed as an object/);
  assert.throws(() => telemetry.normalize([]), /telemetry data must be passed as an object/);
  assert.throws(() => telemetry.candidateWords(null), /telemetry data must be passed as an object/);
  assert.throws(
    () => telemetry.applyCandidateState({ hasCandidateTelemetry: false }, null),
    /telemetry data must be passed as an object/
  );
}

{
  const t = telemetry.normalize({
    token: 'A',
    token_id: 7.8,
    step: 4.9,
    experts: 3,
    debt: 1.7,
    consensus: -0.4,
    field_health: 0.75,
    entropy: 2.25,
    selected_prob: 0.0042,
    selected_rank: 2.9,
    candidate_tail_mass: -2,
    top_tokens: [
      { token: ' chosen', prob: 0.6, selected: true },
      { token: ' alternate path', prob: 1.4, logprob: -0.2 },
      { token: 42, prob: 0.1 }
    ]
  });

  assert.equal(t.token, 'A');
  assert.equal(t.tokenId, 7);
  assert.equal(t.step, 4);
  assert.equal(t.experts, 3);
  assert.equal(t.debt, 1);
  assert.equal(t.consensus, 0);
  assert.equal(t.fieldHealth, 0.75);
  assert.equal(t.entropy, 2.25);
  assert.equal(t.selectedProb, 0.0042);
  assert.equal(t.selectedRank, 2);
  assert.equal(t.candidateTailMass, 0);
  assert.equal(t.hasCandidateTelemetry, true);
  assert.deepEqual(t.topTokens, [
    { token: ' chosen', prob: 0.6, logprob: null, rank: 1, selected: true },
    { token: ' alternate path', prob: 1, logprob: -0.2, rank: 2, selected: false }
  ]);
}

{
  const t = telemetry.normalize({ token: 'old-stream' });
  assert.equal(t.hasCandidateTelemetry, false);
  assert.equal(t.hasDebt, false);
  assert.equal(t.hasFieldHealth, false);
  assert.deepEqual(t.topTokens, []);
}

{
  const words = telemetry.candidateWords({
    top_tokens: [
      { token: ' selected', prob: 0.9, selected: true },
      { token: ' wall rejected', prob: 0.25 },
      { token: 'scar', prob: 0.01, rank: 6 }
    ]
  }, { limit: 3, wordsPerToken: 2 });

  assert.deepEqual(words.map(w => [w.word, w.prob, w.rank, w.selected]), [
    ['wall', 0.25, 2, false],
    ['rejected', 0.25, 2, false],
    ['scar', 0.01, 6, false]
  ]);
}

{
  assert.equal(telemetry.candidateText({
    top_tokens: [
      { token: ' first', prob: 0.2 },
      { token: ' second', prob: 0.1 }
    ]
  }, { limit: 2, wordsPerToken: 1, sanitizer: text => text.toUpperCase() }), 'FIRST SECOND');
}

{
  assert.equal(telemetry.metricProb(Number.NaN), '-');
  assert.equal(telemetry.metricProb(0.004), '0.0040');
  assert.equal(telemetry.metricProb(0.4), '0.400');
  assert.equal(telemetry.metricProb(4), '1.000');
}

{
  const state = {
    selectedProb: 0.8,
    selectedRank: 4,
    candidateTail: 0.3,
    hasCandidateTelemetry: true
  };
  assert.equal(telemetry.resetCandidateState(state), state);
  assert.deepEqual(state, {
    selectedProb: 0,
    selectedRank: 0,
    candidateTail: 0,
    hasCandidateTelemetry: false
  });
}

{
  const state = {
    selectedProb: 0.2,
    selectedRank: 5,
    candidateTail: 0,
    hasCandidateTelemetry: false
  };
  const result = telemetry.applyCandidateState(state, {
    selected_prob: 0.42,
    selected_rank: 3,
    candidate_tail_mass: 0.16
  });
  assert.deepEqual(result, {
    selectedProb: 0.42,
    selectedRank: 3,
    candidateTail: 0.16,
    hasSelectedProb: true,
    hasSelectedRank: true,
    hasCandidateTelemetry: true
  });
  assert.deepEqual(state, {
    selectedProb: 0.42,
    selectedRank: 3,
    candidateTail: 0.16,
    hasCandidateTelemetry: true
  });
}

{
  const state = {
    selectedProb: 0.73,
    selectedRank: 8,
    candidateTail: 0.45,
    hasCandidateTelemetry: false
  };
  const result = telemetry.applyCandidateState(state, { token: 'legacy' }, { candidateCount: 2 });
  assert.deepEqual(result, {
    selectedProb: 0.73,
    selectedRank: 8,
    candidateTail: 0,
    hasSelectedProb: false,
    hasSelectedRank: false,
    hasCandidateTelemetry: true
  });
  assert.deepEqual(state, {
    selectedProb: 0.73,
    selectedRank: 8,
    candidateTail: 0,
    hasCandidateTelemetry: true
  });
}
