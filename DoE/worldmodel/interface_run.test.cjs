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

  run.bindComposer({ form: f, promptInput: input, onSubmit: text => submitted.push(text) });
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

  run.bindComposer({ form: f, promptInput: input, onSubmit: text => submitted.push(text) });
  const current = run.begin();
  f.handler({ preventDefault: () => { prevented++; } });
  assert.strictEqual(prevented, 1);
  assert.deepStrictEqual(submitted, []);
  assert.strictEqual(input.value, 'do not submit');
  assert.strictEqual(current.controller.aborted, 1);
  run.finish(current);
}

{
  const run = runHelper.create({ button: button(), AbortController: FakeAbortController });
  assert.throws(() => run.bindComposer(form(), { value: '' }, () => {}), /must be passed as \{ form, promptInput, onSubmit \}/);
  assert.throws(() => run.bindComposer({ input: { value: '' }, onSubmit() {} }), /prompt input must be passed as \{ promptInput \}/);
  assert.throws(() => run.bindComposer({ promptInput: { value: '' }, onSubmit() {} }), /composer form unavailable/);
  assert.throws(() => run.bindComposer({ form: form(), onSubmit() {} }), /composer input unavailable/);
  assert.throws(() => run.bindComposer({ form: form(), promptInput: { value: '' } }), /composer submit handler unavailable/);
}
