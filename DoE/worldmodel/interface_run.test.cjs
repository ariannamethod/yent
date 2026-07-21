const assert = require('assert');
const runHelper = require('./interface_run.js');

function button() {
  return { textContent: 'SEND', disabled: true };
}

class FakeAbortController {
  constructor() {
    this.signal = { owner: this };
    this.aborted = 0;
  }

  abort() {
    this.aborted++;
  }
}

function form() {
  return {
    handler: null,
    addEventListener(type, handler) {
      assert.strictEqual(type, 'submit');
      this.handler = handler;
    }
  };
}

{
  const send = button();
  const run = runHelper.create({ button: send, AbortController: FakeAbortController });
  assert.strictEqual(run.isRunning(), false);
  assert.strictEqual(run.abortRunning(), false);

  const current = run.begin();
  assert.strictEqual(run.isRunning(), true);
  assert.strictEqual(send.textContent, 'STOP');
  assert.strictEqual(send.disabled, false);
  assert.ok(current.signal);
  assert.throws(() => run.begin(), /already running/);

  assert.strictEqual(run.finish(), false);
  assert.strictEqual(run.isRunning(), true);
  assert.strictEqual(send.textContent, 'STOP');

  assert.strictEqual(run.finish({ id: 'not-a-run-id' }), false);
  assert.strictEqual(run.isRunning(), true);
  assert.strictEqual(send.textContent, 'STOP');

  assert.strictEqual(run.finish({ id: current.id + 1 }), false);
  assert.strictEqual(run.isRunning(), true);
  assert.strictEqual(send.textContent, 'STOP');

  assert.strictEqual(run.finish(current), true);
  assert.strictEqual(run.isRunning(), false);
  assert.strictEqual(send.textContent, 'SEND');

  assert.strictEqual(run.finish(current), false);
  assert.strictEqual(run.isRunning(), false);
}

{
  const send = button();
  const run = runHelper.create({ button: send, AbortController: FakeAbortController });
  const current = run.begin();
  assert.strictEqual(run.abortRunning(), true);
  assert.strictEqual(current.controller.aborted, 1);
  run.finish(current);
}

{
  const send = button();
  const run = runHelper.create({ button: send, AbortController: FakeAbortController });
  const f = form();
  const input = { value: '  hello Yent  ' };
  const submitted = [];
  let prevented = 0;

  run.bindComposer(f, input, text => submitted.push(text));
  f.handler({ preventDefault: () => { prevented++; } });
  assert.strictEqual(prevented, 1);
  assert.deepStrictEqual(submitted, ['hello Yent']);
  assert.strictEqual(input.value, '');

  input.value = '   ';
  f.handler({ preventDefault: () => { prevented++; } });
  assert.strictEqual(prevented, 2);
  assert.deepStrictEqual(submitted, ['hello Yent']);
}

{
  const send = button();
  const run = runHelper.create({ button: send, AbortController: FakeAbortController });
  const f = form();
  const input = { value: 'do not submit' };
  const submitted = [];
  let prevented = 0;

  run.bindComposer(f, input, text => submitted.push(text));
  const current = run.begin();
  f.handler({ preventDefault: () => { prevented++; } });
  assert.strictEqual(prevented, 1);
  assert.deepStrictEqual(submitted, []);
  assert.strictEqual(input.value, 'do not submit');
  assert.strictEqual(current.controller.aborted, 1);
  run.finish(current);
}
