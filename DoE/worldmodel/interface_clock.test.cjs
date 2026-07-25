const assert = require('node:assert/strict');
const clock = require('./interface_clock.js');

function main() {
  {
    const c = clock.create({ startedAt: 1000, minElapsedSeconds: 0.01 });
    assert.equal(c.started(), 1000);
    assert.equal(c.count(), 0);
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
    assert.equal(c.tick(), 5);
    now = 2400;
    assert.equal(c.tick(), 5);
  }
}

main();
