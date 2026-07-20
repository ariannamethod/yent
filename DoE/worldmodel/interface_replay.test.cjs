const assert = require('node:assert/strict');
const replay = require('./interface_replay.js');

async function main() {
{
  assert.deepEqual(replay.request('/worldmodel'), {
    enabled: false,
    name: 'boundary',
    prompt: '',
    delayMs: 0
  });
  const req = replay.request('/worldmodel?replay=1&delay=0');
  assert.equal(req.enabled, true);
  assert.equal(req.name, 'boundary');
  assert.equal(req.delayMs, 0);
  assert.match(req.prompt, /boundary/);
}

{
  const req = replay.request({ search: '?demo=1&scenario=unknown&delay=9999' });
  assert.equal(req.enabled, true);
  assert.equal(req.name, replay.DEFAULT_SCENARIO);
  assert.equal(req.delayMs, 2000);
}

{
  const one = replay.scenario('boundary');
  const two = replay.scenario('boundary');
  assert.notEqual(one.events, two.events);
  assert.notEqual(one.events[0], two.events[0]);
  assert.notEqual(one.events[0].top_tokens, two.events[0].top_tokens);
  one.events[0].top_tokens[0].token = 'mutated';
  assert.notEqual(two.events[0].top_tokens[0].token, 'mutated');
}

{
  const seen = [];
  let done = false;
  const result = await replay.play({
    scenario: 'boundary',
    delayMs: 0,
    onToken: (token, data) => seen.push({ token, step: data.step, prob: data.selected_prob }),
    onDone: event => { done = event.done === true; }
  });
  assert.equal(done, true);
  assert.equal(seen.length, replay.scenario('boundary').events.length);
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
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
