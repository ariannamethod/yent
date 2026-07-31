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
    assert.equal(helper.requestFrame({ callback: frame }), 17);
    assert.equal(helper.start({ callback: frame }), 17);
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
    const prior = globalThis.requestAnimationFrame;
    let touched = false;
    globalThis.requestAnimationFrame = () => {
      touched = true;
      return 99;
    };
    try {
      assert.throws(
        () => animation.create({ requestAnimationFrame: null }),
        /requestAnimationFrame unavailable/
      );
      assert.equal(touched, false);
    } finally {
      globalThis.requestAnimationFrame = prior;
    }
  }

  {
    const helper = animation.create({ requestAnimationFrame() {} });
    assert.throws(() => helper.requestFrame(null), /animation frame callback unavailable/);
    assert.throws(() => helper.requestFrame(() => {}), /callback must be passed as \{ callback \}/);
    assert.throws(() => helper.start(() => {}), /callback must be passed as \{ callback \}/);
  }
}

main();
