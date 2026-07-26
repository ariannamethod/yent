const assert = require('node:assert/strict');
const animation = require('./interface_animation.js');

function main() {
  {
    const calls = [];
    const helper = animation.create({
      requestAnimationFrame(callback) {
        calls.push(callback);
        return 17;
      }
    });
    function frame() {}
    assert.equal(helper.requestFrame(frame), 17);
    assert.equal(helper.start(frame), 17);
    assert.deepEqual(calls, [frame, frame]);
  }

  {
    const prior = globalThis.requestAnimationFrame;
    globalThis.requestAnimationFrame = undefined;
    try {
      assert.throws(() => animation.create({}), /requestAnimationFrame unavailable/);
    } finally {
      globalThis.requestAnimationFrame = prior;
    }
  }

  {
    const helper = animation.create({ requestAnimationFrame() {} });
    assert.throws(() => helper.requestFrame(null), /animation frame callback unavailable/);
  }
}

main();
