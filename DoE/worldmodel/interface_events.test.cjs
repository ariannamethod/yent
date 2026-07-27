const assert = require('node:assert/strict');
const events = require('./interface_events.js');

function target() {
  const listeners = {};
  return {
    listeners,
    addEventListener(type, handler, options) {
      if (!listeners[type]) listeners[type] = [];
      listeners[type].push({ handler, options });
    },
    removeEventListener(type, handler) {
      listeners[type] = (listeners[type] || []).filter(entry => entry.handler !== handler);
    }
  };
}

function fire(t, type, event) {
  for (const entry of (t.listeners[type] || []).slice()) entry.handler(event || {});
}

function main() {
  {
    const t = target();
    const keys = {};
    const cleanup = events.bindKeyState({
      target: t,
      keys,
      ignore: event => event && event.ignore
    });

    fire(t, 'keydown', { key: 'W' });
    assert.equal(keys.w, true);
    fire(t, 'keydown', { key: 'A', ignore: true });
    assert.equal(keys.a, undefined);
    fire(t, 'keyup', { key: 'W' });
    assert.equal(keys.w, false);

    cleanup();
    fire(t, 'keydown', { key: 'S' });
    assert.equal(keys.s, undefined);
  }

  {
    const t = target();
    const seen = [];
    const cleanup = events.bindPointer({
      target: t,
      onMove: point => seen.push(['move', point.x, point.y]),
      onLeave: () => seen.push(['leave']),
      onDown: point => seen.push(['down', point.x, point.y])
    });

    fire(t, 'mousemove', { clientX: 12, clientY: 34 });
    fire(t, 'mousedown', { clientX: 21, clientY: 43 });
    fire(t, 'mouseout', {});
    assert.deepEqual(seen, [
      ['move', 12, 34],
      ['down', 21, 43],
      ['leave']
    ]);

    cleanup();
    fire(t, 'mousemove', { clientX: 99, clientY: 88 });
    assert.equal(seen.length, 3);
  }

  {
    const hadAdd = Object.prototype.hasOwnProperty.call(globalThis, 'addEventListener');
    const hadRemove = Object.prototype.hasOwnProperty.call(globalThis, 'removeEventListener');
    const previousAdd = globalThis.addEventListener;
    const previousRemove = globalThis.removeEventListener;
    const listeners = {};
    globalThis.addEventListener = (type, handler, options) => {
      if (!listeners[type]) listeners[type] = [];
      listeners[type].push({ handler, options });
    };
    globalThis.removeEventListener = (type, handler) => {
      listeners[type] = (listeners[type] || []).filter(entry => entry.handler !== handler);
    };
    try {
      const keys = {};
      const cleanup = events.bindKeyState({ keys });
      assert.equal((listeners.keydown || []).length, 1);
      listeners.keydown[0].handler({ key: 'Q' });
      assert.equal(keys.q, true);
      cleanup();
      assert.equal((listeners.keydown || []).length, 0);
    } finally {
      if (hadAdd) globalThis.addEventListener = previousAdd;
      else delete globalThis.addEventListener;
      if (hadRemove) globalThis.removeEventListener = previousRemove;
      else delete globalThis.removeEventListener;
    }
  }

  assert.throws(() => events.bindKeyState({ target: target() }), /interface key state target missing/);
  assert.throws(() => events.bindPointer({ target: {} }), /interface event target unavailable/);
}

main();
