const assert = require('node:assert/strict');
const style = require('./interface_style.js');

function main() {
  {
    let reads = 0;
    const fonts = style.create({
      document: { documentElement: {} },
      getComputedStyle() {
        reads += 1;
        return {
          getPropertyValue(name) {
            if (name === '--mono') return ' Test Mono ';
            if (name === '--sans') return ' Test Sans ';
            return '';
          }
        };
      }
    });

    assert.equal(fonts.mono(), 'Test Mono');
    assert.equal(fonts.mono(), 'Test Mono');
    assert.equal(reads, 1);
    assert.equal(fonts.sans(), 'Test Sans');
    assert.match(fonts.serif(), /serif/);
    fonts.reset();
    assert.equal(fonts.mono(), 'Test Mono');
    assert.equal(reads, 4);
  }

  {
    const fonts = style.create({ document: null, getComputedStyle: null });
    assert.match(fonts.mono(), /monospace/);
    assert.match(fonts.serif(), /serif/);
    assert.match(fonts.sans(), /sans-serif/);
    assert.equal(fonts.family('--unknown', 'Custom Family'), 'Custom Family');
    assert.equal(fonts.family('', 'Empty Name'), 'Empty Name');
  }

  {
    const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document');
    const hadGetComputedStyle = Object.prototype.hasOwnProperty.call(globalThis, 'getComputedStyle');
    const previousDocument = globalThis.document;
    const previousGetComputedStyle = globalThis.getComputedStyle;
    let hostSeen = null;
    globalThis.document = { documentElement: { id: 'root' } };
    globalThis.getComputedStyle = host => {
      hostSeen = host;
      return {
        getPropertyValue(name) {
          return name === '--serif' ? ' Root Serif ' : '';
        }
      };
    };
    try {
      const fonts = style.create();
      assert.equal(fonts.serif(), 'Root Serif');
      assert.equal(hostSeen, globalThis.document.documentElement);
      assert.match(fonts.mono(), /monospace/);
    } finally {
      if (hadDocument) globalThis.document = previousDocument;
      else delete globalThis.document;
      if (hadGetComputedStyle) globalThis.getComputedStyle = previousGetComputedStyle;
      else delete globalThis.getComputedStyle;
    }
  }
}

main();
