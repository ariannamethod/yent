const assert = require('node:assert/strict');
const outcome = require('./interface_outcome.js');

function submit(result) {
  return {
    turn: {
      text: result.text || '',
      outcome: result
    },
    outcome: result
  };
}

{
  const seen = [];
  const handled = outcome.handle(submit({ stopped: true, hasText: true, kind: 'stopped' }), {
    stopped: (turn, result) => seen.push(['stopped', turn.text, result.hasText]),
    fault: () => seen.push(['fault']),
    complete: () => seen.push(['complete'])
  });
  assert.equal(handled.kind, 'stopped');
  assert.deepEqual(seen, [['stopped', '', true]]);
}

{
  const seen = [];
  const handled = outcome.handle(submit({ fault: true, message: 'boom', text: 'partial' }), {
    stopped: () => seen.push(['stopped']),
    fault: (turn, result) => seen.push(['fault', turn.text, result.message]),
    complete: () => seen.push(['complete'])
  });
  assert.equal(handled.kind, 'fault');
  assert.deepEqual(seen, [['fault', 'partial', 'boom']]);
}

{
  const seen = [];
  const handled = outcome.handle(submit({ kind: 'empty', hasText: false }), {
    complete: (_turn, result) => seen.push(['complete', result.kind])
  });
  assert.equal(handled.kind, 'empty');
  assert.deepEqual(seen, [['complete', 'empty']]);
}

{
  const handled = outcome.handle(submit({ kind: 'complete', hasText: true }));
  assert.equal(handled.kind, 'complete');
}

assert.throws(() => outcome.handle({}), /YentInterfaceOutcome outcome missing/);
