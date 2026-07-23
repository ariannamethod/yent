const assert = require('node:assert/strict');
const boot = require('./interface_boot.js');

function main() {
  {
    const calls = [];
    const replayRequest = { enabled: true, prompt: 'fixture' };
    const promptInput = { value: '' };
    const generationRun = { isRunning: () => false };
    const generate = () => {};
    const timer = () => {};
    const windowRef = {
      addEventListener(type, handler) {
        calls.push(`listen:${type}`);
        assert.equal(type, 'resize');
        assert.equal(handler, resize);
      }
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

    const started = boot.start({
      restore: () => calls.push('restore'),
      resize,
      window: windowRef,
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
    assert.equal(started, true);
    assert.deepEqual(calls, ['restore', 'resize', 'listen:resize', 'animation', 'replay']);
  }

  {
    assert.throws(() => boot.start({}), /interface restore unavailable/);
    assert.throws(
      () => boot.start({ restore() {}, resize() {}, startAnimation() {} }),
      /YentInterfaceReplay helper missing/
    );
  }
}

main();
