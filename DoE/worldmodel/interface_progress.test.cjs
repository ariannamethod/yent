const assert = require('node:assert/strict');
const progress = require('./interface_progress.js');

assert.equal(progress.formatElapsed(0), '00:00');
assert.equal(progress.formatElapsed(65_999), '01:05');
assert.equal(progress.formatElapsed(-4), '00:00');

{
  let now = 1_000;
  let scheduled = null;
  let cancelled = null;
  const ticks = [];
  const run = progress.start({
    now: () => now,
    setInterval(callback, interval) {
      scheduled = { callback, interval };
      return 17;
    },
    clearInterval(id) {
      cancelled = id;
    },
    intervalMs: 500,
    onTick: tick => ticks.push(tick)
  });
  assert.deepEqual(ticks, [{ elapsedMs: 0, label: '00:00' }]);
  assert.equal(scheduled.interval, 500);
  now += 62_000;
  scheduled.callback();
  assert.deepEqual(ticks.at(-1), { elapsedMs: 62_000, label: '01:02' });
  assert.equal(run.stop(), true);
  assert.equal(run.stop(), false);
  assert.equal(cancelled, 17);
}

assert.throws(() => progress.start(), /options must be passed as an object/);
assert.throws(() => progress.start([]), /options must be passed as an object/);
assert.throws(() => progress.start({ onTick: null }), /onTick callback missing/);
