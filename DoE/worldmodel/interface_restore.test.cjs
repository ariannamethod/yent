const assert = require('node:assert/strict');
const restore = require('./interface_restore.js');

{
  let calls = 0;
  const result = restore.load({
    replayMode: true,
    sessionReceipt: { load: () => { calls++; return []; } }
  });
  assert.equal(result, null);
  assert.equal(calls, 0);
}

{
  const result = restore.load({ sessionReceipt: { load: () => [] } });
  assert.equal(result, null);
}

{
  const messages = [
    { role: 'user', content: 'hello' },
    { role: 'assistant', content: 'first answer' },
    { role: 'user', content: 'again' },
    { role: 'assistant', content: 'final answer' }
  ];
  let callback = null;
  const result = restore.load({
    sessionReceipt: { load: () => messages },
    onRestore: restored => { callback = restored; }
  });
  assert.equal(result.visibleMessages, messages);
  assert.equal(result.combinedText, 'hello first answer again final answer');
  assert.equal(result.lastAssistant.content, 'final answer');
  assert.equal(callback, result);
}

{
  const result = restore.load({
    sessionReceipt: { load: () => [{ role: 'user', content: 'only user' }] }
  });
  assert.equal(result.lastAssistant, null);
  assert.equal(result.combinedText, 'only user');
}

assert.throws(() => restore.load({}), /YentInterfaceSession adapter missing/);
