const assert = require('node:assert/strict');
const clock = require('./interface_clock.js');

function main() {
  {
    const c = clock.create({ startedAt: 1000, minElapsedSeconds: 0.01 });
    assert.equal(c.started(), 1000);
    assert.equal(c.count(), 0);
    assert.equal(c.now(1016), 1016);
    assert.equal(c.tick(1005), 100);
    assert.equal(c.count(), 1);
    assert.equal(c.tick(3000), 1);
    c.reset(5000);
    assert.equal(c.started(), 5000);
    assert.equal(c.count(), 0);
    assert.equal(c.tick(6000), 1);
  }

  {
    let now = 2000;
    const perf = { now: () => now };
    const c = clock.create({ performance: perf, minElapsedSeconds: 0.1 });
    now = 2200;
    assert.equal(c.now(), 2200);
    assert.equal(c.tick(), 5);
    now = 2400;
    assert.equal(c.tick(), 5);
  }

  {
    const hadPerformance = Object.prototype.hasOwnProperty.call(globalThis, 'performance');
    const previousPerformance = globalThis.performance;
    let now = 7000;
    globalThis.performance = { now: () => now };
    try {
      const c = clock.create({ minElapsedSeconds: 0.1 });
      assert.equal(c.started(), 7000);
      now = 7500;
      assert.equal(c.now(), 7500);
      assert.equal(c.tick(), 2);
    } finally {
      if (hadPerformance) globalThis.performance = previousPerformance;
      else delete globalThis.performance;
    }
  }

  {
    const hadPerformance = Object.prototype.hasOwnProperty.call(globalThis, 'performance');
    const previousPerformance = globalThis.performance;
    const previousDateNow = Date.now;
    let perfTouched = false;
    globalThis.performance = {
      now() {
        perfTouched = true;
        return 9000;
      }
    };
    Date.now = () => 1234;
    try {
      const c = clock.create({ performance: null, minElapsedSeconds: 0.1 });
      assert.equal(c.started(), 1234);
      assert.equal(c.now(), 1234);
      assert.equal(perfTouched, false);
    } finally {
      Date.now = previousDateNow;
      if (hadPerformance) globalThis.performance = previousPerformance;
      else delete globalThis.performance;
    }
  }
}

main();
