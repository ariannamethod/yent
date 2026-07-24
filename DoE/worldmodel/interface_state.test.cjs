const assert = require('node:assert/strict');
const state = require('./interface_state.js');

function main() {
  {
    const first = state.create();
    const second = state.create();

    assert.equal(first.debt, 0.0);
    assert.equal(first.consensus, 0.62);
    assert.equal(first.field, 1.0);
    assert.equal(first.tokps, 0.0);
    assert.equal(first.step, 0);
    assert.equal(first.entropy, 0.0);
    assert.equal(first.selectedProb, 0.0);
    assert.equal(first.selectedRank, 0);
    assert.equal(first.candidateTail, 0.0);
    assert.equal(first.hasCandidateTelemetry, false);

    first.debt = 0.9;
    assert.equal(second.debt, 0.0);
    assert.equal(state.BASELINE.debt, 0.0);
  }

  {
    const custom = state.create({
      debt: 0.42,
      cameraZ: 12,
      velocity: 1.2,
      hasCandidateTelemetry: true
    });

    assert.equal(custom.debt, 0.42);
    assert.equal(custom.consensus, 0.62);
    assert.equal(custom.cameraZ, 12);
    assert.equal(custom.velocity, 1.2);
    assert.equal(custom.hasCandidateTelemetry, true);
  }
}

main();
