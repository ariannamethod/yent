const assert = require('node:assert/strict');
const canvas = require('./interface_canvas.js');

function makeSurface() {
  const calls = [];
  return {
    calls,
    canvas: { style: {}, width: 0, height: 0 },
    context: {
      setTransform(...args) {
        calls.push(args);
      }
    }
  };
}

{
  assert.equal(canvas.pixelRatio({ devicePixelRatio: 3 }, 2), 2);
  assert.equal(canvas.pixelRatio({ devicePixelRatio: 0 }, 2), 1);
  assert.equal(canvas.pixelRatio({}, 2), 1);
}

{
  const vp = canvas.viewport({ innerWidth: 1200, innerHeight: 800, devicePixelRatio: 1.5 });
  assert.deepEqual(vp, { width: 1200, height: 800, dpr: 1.5 });
}

{
  const main = makeSurface();
  const trace = makeSurface();
  const result = canvas.resize({
    window: { innerWidth: 960, innerHeight: 720, devicePixelRatio: 2.5 },
    surfaces: [
      { canvas: main.canvas, context: main.context },
      { canvas: trace.canvas, context: trace.context, height: 22 }
    ]
  });

  assert.equal(result.width, 960);
  assert.equal(result.height, 720);
  assert.equal(result.dpr, 2);
  assert.equal(main.canvas.style.width, '960px');
  assert.equal(main.canvas.style.height, '720px');
  assert.equal(main.canvas.width, 1920);
  assert.equal(main.canvas.height, 1440);
  assert.deepEqual(main.calls[0], [2, 0, 0, 2, 0, 0]);
  assert.equal(trace.canvas.style.width, '960px');
  assert.equal(trace.canvas.style.height, '22px');
  assert.equal(trace.canvas.width, 1920);
  assert.equal(trace.canvas.height, 44);
}

{
  const calls = [];
  const documentRef = {
    createElement(tag) {
      assert.equal(tag, 'canvas');
      return {
        getContext(type, options) {
          calls.push([type, options]);
          return { type, options };
        }
      };
    }
  };
  const scratch = canvas.createScratch({
    document: documentRef,
    contextOptions: { willReadFrequently: true }
  });
  assert.deepEqual(calls, [['2d', { willReadFrequently: true }]]);
  assert.equal(scratch.context.type, '2d');
  assert.equal(scratch.context.options.willReadFrequently, true);
}

{
  const calls = [];
  const documentRef = {
    getElementById(id) {
      assert.equal(id, 'field');
      return {
        getContext(type, options) {
          calls.push([type, options]);
          return { type, options };
        }
      };
    }
  };
  const bound = canvas.bind({
    document: documentRef,
    id: 'field',
    contextOptions: { alpha: false }
  });
  assert.deepEqual(calls, [['2d', { alpha: false }]]);
  assert.equal(bound.context.type, '2d');
  assert.equal(bound.context.options.alpha, false);
}

{
  assert.throws(() => canvas.resize({ canvas: {}, context: {} }), /interface canvas surface unavailable/);
  assert.throws(() => canvas.bind({ document: {} }), /interface canvas document unavailable/);
  assert.throws(() => canvas.bind({ document: { getElementById: () => null }, id: 'missing' }), /interface canvas element unavailable: missing/);
  assert.throws(() => canvas.bind({
    document: { getElementById: () => ({ getContext: () => null }) },
    id: 'field'
  }), /interface canvas context unavailable: field/);
  assert.throws(() => canvas.createScratch({ document: {} }), /interface canvas document unavailable/);
  assert.throws(() => canvas.createScratch({ document: { createElement: () => ({}) } }), /interface canvas scratch surface unavailable/);
  assert.throws(() => canvas.createScratch({
    document: { createElement: () => ({ getContext: () => null }) }
  }), /interface canvas scratch context unavailable/);
}
