const assert = require('node:assert/strict');
const math = require('./interface_math.js');

function main() {
  assert.equal(math.clamp(0.5, 0, 1), 0.5);
  assert.equal(math.clamp(-2, 0, 1), 0);
  assert.equal(math.clamp(3, 0, 1), 1);

  assert.equal(math.mix(2, 10, 0), 2);
  assert.equal(math.mix(2, 10, 1), 10);
  assert.equal(math.mix(2, 10, 0.25), 4);
}

main();
