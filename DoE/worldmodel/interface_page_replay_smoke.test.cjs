const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ROOT = path.resolve(__dirname, '..', '..');
const WORLD = path.join(ROOT, 'DoE', 'worldmodel');

function read(rel) {
  return fs.readFileSync(path.join(WORLD, rel), 'utf8');
}

function makeContext(page) {
  const elements = new Map();
  const timers = [];
  const frames = [];
  const storageWrites = [];
  const fetchCalls = [];

  function gradient() {
    return { addColorStop() {} };
  }

  function context2d() {
    return new Proxy({}, {
      get(_target, prop) {
        if (prop === 'createRadialGradient' || prop === 'createLinearGradient') return gradient;
        if (prop === 'measureText') return text => ({ width: String(text || '').length * 8 });
        if (prop === 'getImageData') {
          return (_x, _y, w, h) => ({ data: new Uint8ClampedArray(Math.max(0, w * h * 4)) });
        }
        return () => {};
      },
      set() {
        return true;
      }
    });
  }

  function element(tagName, id) {
    const el = {
      id: id || '',
      tagName,
      style: {},
      dataset: {},
      children: [],
      className: '',
      textContent: '',
      value: '',
      disabled: false,
      width: 0,
      height: 0,
      scrollTop: 0,
      get scrollHeight() { return this.children.length * 20 + this.textContent.length; },
      appendChild(child) {
        this.children.push(child);
        return child;
      },
      addEventListener(type, handler) {
        this[`on${type}`] = handler;
      },
      getContext() {
        return context2d();
      }
    };
    if (id === 'temp') el.value = '0.8';
    if (id === 'max-tokens') el.value = '512';
    return el;
  }

  const document = {
    documentElement: element('html', 'html'),
    activeElement: null,
    createElement(tagName) {
      return element(String(tagName || '').toLowerCase(), '');
    },
    getElementById(id) {
      if (!elements.has(id)) {
        const tagName = id === 'field' || id === 'trace' ? 'canvas' : 'div';
        elements.set(id, element(tagName, id));
      }
      return elements.get(id);
    }
  };

  const context = {
    console,
    Math,
    JSON,
    Date,
    Number,
    String,
    Array,
    Object,
    RegExp,
    Error,
    Uint8ClampedArray,
    URLSearchParams,
    AbortController,
    setTimeout(fn) {
      timers.push(fn);
      return timers.length;
    },
    clearTimeout() {},
    requestAnimationFrame(fn) {
      if (typeof fn === 'function') frames.push(fn);
      return 1;
    },
    performance: {
      now: (() => {
        let now = 1000;
        return () => {
          now += 16;
          return now;
        };
      })()
    },
    getComputedStyle() {
      return { getPropertyValue: () => 'ui-monospace, monospace' };
    },
    fetch() {
      fetchCalls.push(true);
      return Promise.reject(new Error('fetch must not run during replay smoke'));
    },
    sessionStorage: {
      getItem() {
        return null;
      },
      setItem(key, value) {
        storageWrites.push({ key, value });
      }
    },
    document,
    innerWidth: page === 'worldmodel' ? 1280 : 960,
    innerHeight: 720,
    devicePixelRatio: 1,
    location: {
      href: `http://127.0.0.1/${page}?replay=1&delay=0`,
      search: '?replay=1&delay=0'
    },
    addEventListener(type, handler) {
      this[`on${type}`] = handler;
    },
    __elements: elements,
    __timers: timers,
    __frames: frames,
    __storageWrites: storageWrites,
    __fetchCalls: fetchCalls
  };
  context.window = context;
  context.globalThis = context;
  return vm.createContext(context);
}

function runScript(context, rel) {
  vm.runInContext(read(rel), context, { filename: rel });
}

async function drainReplay(context) {
  while (context.__timers.length) {
    const fn = context.__timers.shift();
    fn();
    for (let i = 0; i < 40; i++) await Promise.resolve();
  }
  for (let i = 0; i < 80; i++) await Promise.resolve();
  if (context.__frames.length) {
    const fn = context.__frames.shift();
    fn();
    for (let i = 0; i < 20; i++) await Promise.resolve();
  }
}

async function runPage(page) {
  const context = makeContext(page);
  for (const rel of [
    'interface_session.js',
    'event_stream.js',
    'chat_stream.js',
    'token_telemetry.js',
    'interface_replay.js',
    'interface_input.js',
    'interface_turn.js',
    'interface_run.js'
  ]) {
    runScript(context, rel);
  }
  if (page === 'worldmodel') runScript(context, 'worldmodel_geometry.js');
  runScript(context, page === 'worldmodel' ? 'worldmodel.js' : 'yent.js');
  await drainReplay(context);
  return context;
}

async function main() {
{
  const ctx = await runPage('yent');
  const transcript = ctx.document.getElementById('transcript');
  const runState = ctx.document.getElementById('run-state');
  const send = ctx.document.getElementById('send');

  assert.equal(ctx.__fetchCalls.length, 0, 'yent replay must not call fetch');
  assert.equal(ctx.__storageWrites.length, 0, 'yent replay must not write sessionStorage');
  assert.equal(runState.textContent, 'COMPLETE');
  assert.equal(send.textContent, 'SEND');
  assert.equal(transcript.children.length, 2);
  assert.match(transcript.children[1].children[1].textContent, /chosen answer becomes visible/);
  assert.notEqual(ctx.document.getElementById('hud-prob').textContent, '-');
  assert.notEqual(ctx.document.getElementById('hud-tail').textContent, '-');
}

{
  const ctx = await runPage('worldmodel');
  const status = ctx.document.getElementById('status-note');
  const manifestState = ctx.document.getElementById('manifest-state');
  const manifestText = ctx.document.getElementById('manifest-text');
  const send = ctx.document.getElementById('send');

  assert.equal(ctx.__fetchCalls.length, 0, 'worldmodel replay must not call fetch');
  assert.equal(ctx.__storageWrites.length, 0, 'worldmodel replay must not write sessionStorage');
  assert.equal(status.textContent, 'FIELD SETTLED.');
  assert.equal(manifestState.textContent, 'COMPLETE');
  assert.equal(send.textContent, 'SEND');
  assert.match(manifestText.textContent, /chosen answer becomes visible/);
  assert.notEqual(ctx.document.getElementById('hud-prob').textContent, '-');
  assert.notEqual(ctx.document.getElementById('hud-tail').textContent, '-');
}
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
