const assert = require('node:assert/strict');
const replay = require('./interface_replay.js');

async function main() {
{
  assert.throws(() => replay.request('/worldmodel'), /location must be passed as \{ location \}/);
  assert.deepEqual(replay.request({ location: '/worldmodel' }), {
    enabled: false,
    name: 'boundary',
    prompt: '',
    delayMs: 0
  });
  const req = replay.request({ location: '/worldmodel?replay=1&delay=0' });
  assert.equal(req.enabled, true);
  assert.equal(req.name, 'boundary');
  assert.equal(req.delayMs, 0);
  assert.match(req.prompt, /boundary/);
}

{
  const req = replay.request({ location: { search: '?demo=1&scenario=unknown&delay=9999' } });
  assert.equal(req.enabled, true);
  assert.equal(req.name, replay.DEFAULT_SCENARIO);
  assert.equal(req.delayMs, 2000);
}

{
  assert.throws(
    () => replay.request({ search: '?demo=1&delay=0' }),
    /replay request search must be passed as \{ location \}/
  );
  const req = replay.request({ location: { search: '?demo=1&delay=0' } });
  assert.equal(req.enabled, true);
  assert.equal(req.delayMs, 0);
}

{
  const prior = globalThis.location;
  globalThis.location = { search: '?replay=1&delay=0' };
  try {
    const req = replay.request();
    assert.equal(req.enabled, true);
    assert.equal(req.name, 'boundary');
    assert.equal(req.delayMs, 0);
  } finally {
    if (prior === undefined) delete globalThis.location;
    else globalThis.location = prior;
  }
}

{
  assert.throws(
    () => replay.scenario('boundary'),
    /replay fixture scenario must be passed as \{ scenario \}/
  );
  assert.throws(
    () => replay.scenario({ name: 'boundary' }),
    /replay fixture scenario name must be passed as \{ scenario \}/
  );
  const one = replay.scenario({ scenario: 'boundary' });
  const two = replay.scenario({ scenario: 'boundary' });
  assert.notEqual(one.events, two.events);
  assert.notEqual(one.events[0], two.events[0]);
  assert.notEqual(one.events[0].top_tokens, two.events[0].top_tokens);
  one.events[0].top_tokens[0].token = 'mutated';
  assert.notEqual(two.events[0].top_tokens[0].token, 'mutated');
}

{
  await assert.rejects(
    () => replay.play({ name: 'boundary', delayMs: 0 }),
    /replay scenario name must be passed as \{ scenario \}/
  );
  const seen = [];
  let done = false;
  const result = await replay.play({
    scenario: 'boundary',
    delayMs: 0,
    onToken: (token, data) => seen.push({ token, step: data.step, prob: data.selected_prob }),
    onDone: event => { done = event.done === true; }
  });
  assert.equal(done, true);
  assert.equal(seen.length, replay.scenario({ scenario: 'boundary' }).events.length);
  assert.equal(seen[0].token, 'The');
  assert.equal(seen.at(-1).token, '.');
  assert.equal(seen[0].step, 1);
  assert.equal(result.done, true);
  assert.equal(result.tokens, seen.length);
  assert.equal(result.events, seen.length + 1);
  assert.equal(result.pending, '');
}

{
  const ac = new AbortController();
  ac.abort();
  await assert.rejects(
    () => replay.play({ signal: ac.signal }),
    err => err && err.name === 'AbortError'
  );
}

{
  const seen = [];
  const custom = {
    events: [
      { token: 'x', step: 1 },
      { error: 'fixture failed' },
      { token: 'never', step: 2 }
    ]
  };
  await assert.rejects(
    () => replay.play({ scenario: custom, delayMs: 0, onToken: token => seen.push(token) }),
    /fixture failed/
  );
  assert.deepEqual(seen, ['x']);
}

{
  const priorSetTimeout = globalThis.setTimeout;
  let touched = false;
  globalThis.setTimeout = () => {
    touched = true;
    return 1;
  };
  try {
    assert.throws(
      () => replay.startIfRequested({
        replayMode: true,
        replayRequest: { prompt: 'do not borrow timer' },
        promptInput: { value: '' },
        generationRun: { isRunning: () => false },
        generate() {},
        setTimeout: null
      }),
      /setTimeout unavailable/
    );
    assert.equal(touched, false);
  } finally {
    globalThis.setTimeout = priorSetTimeout;
  }
}

{
  const input = { value: '' };
  const timers = [];
  const generated = [];
  const run = { isRunning: () => false };
  const started = replay.startIfRequested({
    replayMode: true,
    replayRequest: { prompt: 'manifest boundary' },
    promptInput: input,
    generationRun: run,
    generate: text => generated.push(text),
    startDelayMs: 9,
    setTimeout: (fn, delay) => {
      timers.push({ fn, delay });
      return timers.length;
    }
  });
  assert.equal(started, true);
  assert.equal(input.value, 'manifest boundary');
  assert.equal(timers.length, 1);
  assert.equal(timers[0].delay, 9);
  timers[0].fn();
  assert.deepEqual(generated, ['manifest boundary']);
}

{
  const input = { value: '' };
  const timers = [];
  const generated = [];
  const run = { isRunning: () => true };
  replay.startIfRequested({
    replayMode: true,
    replayRequest: { prompt: 'do not start yet' },
    promptInput: input,
    generationRun: run,
    generate: text => generated.push(text),
    setTimeout: fn => {
      timers.push(fn);
      return timers.length;
    }
  });
  timers[0]();
  assert.deepEqual(generated, []);
}

{
  assert.equal(replay.startIfRequested({ replayMode: false }), false);
  assert.throws(
    () => replay.startIfRequested({
      replayMode: true,
      request: { prompt: 'old generic alias' },
      promptInput: { value: '' },
      generationRun: { isRunning: () => false },
      generate() {}
    }),
    /replay request must be passed as \{ replayRequest \}/
  );
  assert.throws(
    () => replay.startIfRequested({
      replayMode: true,
      replayRequest: { prompt: 'old input alias' },
      input: { value: '' },
      generationRun: { isRunning: () => false },
      generate() {}
    }),
    /replay prompt input must be passed as \{ promptInput \}/
  );
  assert.throws(
    () => replay.startIfRequested({
      replayMode: true,
      replayRequest: { prompt: 'old run alias' },
      promptInput: { value: '' },
      run: { isRunning: () => false },
      generate() {}
    }),
    /generation run must be passed as \{ generationRun \}/
  );
  assert.throws(() => replay.startIfRequested({ replayMode: true }), /replay prompt input unavailable/);
  assert.throws(
    () => replay.startIfRequested({ replayMode: true, promptInput: { value: '' } }),
    /generation run helper unavailable/
  );
  assert.throws(
    () => replay.startIfRequested({
      replayMode: true,
      promptInput: { value: '' },
      generationRun: { isRunning: () => false }
    }),
    /replay generate handler unavailable/
  );
}
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
