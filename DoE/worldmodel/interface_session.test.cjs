const assert = require('node:assert/strict');
const session = require('./interface_session.js');

function storage() {
  const data = new Map();
  return {
    getItem(key) {
      return data.has(key) ? data.get(key) : null;
    },
    setItem(key, value) {
      data.set(key, value);
    }
  };
}

{
  const source = [
    { role: 'system', content: 'ignore' },
    { role: 'user', content: 'first' },
    { role: 'assistant', content: '' },
    { role: 'assistant', content: 'second' },
    { role: 'user', content: 'third' }
  ];
  assert.deepEqual(session.normalize(source, { limit: 2 }), [
    { role: 'assistant', content: 'second' },
    { role: 'user', content: 'third' }
  ]);
}

{
  const long = 'x'.repeat(session.CONTENT_LIMIT + 9);
  const normalized = session.normalize([{ role: 'user', content: long }]);
  assert.equal(normalized.length, 1);
  assert.equal(normalized[0].content.length, session.CONTENT_LIMIT);
}

{
  const s = storage();
  assert.equal(session.save(s, [
    { role: 'user', content: 'visible prompt' },
    { role: 'assistant', content: 'visible answer' },
    { role: 'tool', content: 'not visible' }
  ]), true);
  assert.deepEqual(session.load(s), [
    { role: 'user', content: 'visible prompt' },
    { role: 'assistant', content: 'visible answer' }
  ]);
}

{
  const s = storage();
  s.setItem(session.KEY, '{not valid json');
  assert.deepEqual(session.load(s), []);
}

{
  const broken = {
    getItem() {
      throw new Error('read denied');
    },
    setItem() {
      throw new Error('write denied');
    }
  };
  assert.equal(session.save(broken, [{ role: 'user', content: 'x' }]), false);
  assert.deepEqual(session.load(broken), []);
}

{
  let writes = 0;
  const s = storage();
  const originalSetItem = s.setItem;
  s.setItem = function setItem(key, value) {
    writes++;
    return originalSetItem.call(this, key, value);
  };
  session.save(s, [{ role: 'assistant', content: 'real receipt' }]);

  const receipt = session.createAdapter({ storage: s, replayMode: true });
  assert.deepEqual(receipt.load(), []);
  assert.equal(receipt.save([{ role: 'user', content: 'replay prompt' }], true), false);
  assert.equal(writes, 1);
}

{
  const s = storage();
  let now = 1000;
  const receipt = session.createAdapter({
    storage: s,
    now: () => now,
    saveIntervalMs: 100
  });

  assert.deepEqual(receipt.normalize([{ role: 'user', content: 'hello' }]), [
    { role: 'user', content: 'hello' }
  ]);
  assert.equal(receipt.save([{ role: 'user', content: 'first' }]), true);
  now += 40;
  assert.equal(receipt.save([{ role: 'assistant', content: 'too soon' }]), false);
  assert.deepEqual(receipt.load(), [{ role: 'user', content: 'first' }]);
  now += 60;
  assert.equal(receipt.save([{ role: 'assistant', content: 'later' }]), true);
  assert.deepEqual(receipt.load(), [{ role: 'assistant', content: 'later' }]);
  assert.equal(receipt.save([{ role: 'user', content: 'forced' }], true), true);
  assert.deepEqual(receipt.load(), [{ role: 'user', content: 'forced' }]);
}
