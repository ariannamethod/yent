const assert = require('node:assert/strict');
const boot = require('./interface_boot.js');

function main() {
  {
    const calls = [];
    const replayRequest = { enabled: true, prompt: 'fixture' };
    const composer = { id: 'composer' };
    const promptInput = { value: '' };
    const generate = () => {};
    const generationRun = {
      isRunning: () => false,
      bindComposer(form, input, handler) {
        calls.push('composer');
        assert.equal(form, composer);
        assert.equal(input, promptInput);
        assert.equal(handler, generate);
      }
    };
    const timer = () => {};
    const hadAdd = Object.prototype.hasOwnProperty.call(globalThis, 'addEventListener');
    const previousAdd = globalThis.addEventListener;
    globalThis.addEventListener = (type, handler) => {
      calls.push(`listen:${type}`);
      assert.equal(type, 'resize');
      assert.equal(handler, resize);
    };
    function resize() {
      calls.push('resize');
    }
    const replay = {
      startIfRequested(options) {
        calls.push('replay');
        assert.deepEqual(options.request, replayRequest);
        assert.equal(options.replayMode, true);
        assert.equal(options.promptInput, promptInput);
        assert.equal(options.generationRun, generationRun);
        assert.equal(options.generate, generate);
        assert.equal(options.startDelayMs, 5);
        assert.equal(options.setTimeout, timer);
        return true;
      }
    };

    let started;
    try {
      started = boot.start({
        restore: () => calls.push('restore'),
        resize,
        composer,
        startAnimation: () => calls.push('animation'),
        interfaceReplay: replay,
        replayMode: true,
        replayRequest,
        promptInput,
        generationRun,
        generate,
        startDelayMs: 5,
        setTimeout: timer
      });
    } finally {
      if (hadAdd) globalThis.addEventListener = previousAdd;
      else delete globalThis.addEventListener;
    }
    assert.equal(started, true);
    assert.deepEqual(calls, ['restore', 'resize', 'listen:resize', 'composer', 'animation', 'replay']);
  }

  {
    assert.throws(() => boot.start({}), /interface restore unavailable/);
    assert.throws(
      () => boot.start({ restore() {}, resize() {}, startAnimation() {} }),
      /YentInterfaceReplay helper missing/
    );
    assert.throws(
      () => boot.start({ restore() {}, resize() {}, composer: {}, startAnimation() {} }),
      /YentInterfaceRun composer binding unavailable/
    );
  }
}

main();
