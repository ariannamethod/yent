const assert = require('node:assert/strict');
const output = require('./interface_output.js');

function element(scrollHeight) {
  return {
    textContent: 'old',
    scrollTop: 0,
    scrollHeight
  };
}

function main() {
  {
    const target = element(42);
    output.setText(target, 'new');
    assert.equal(target.textContent, 'new');
    output.setText(target, null);
    assert.equal(target.textContent, '');
    output.setText(target, 17);
    assert.equal(target.textContent, '17');
  }

  {
    const target = element(91);
    output.scrollBottom(target);
    assert.equal(target.scrollTop, 91);
  }

  {
    const body = element(14);
    const transcript = element(188);
    output.setTextAndScroll(body, 'answer', transcript);
    assert.equal(body.textContent, 'answer');
    assert.equal(transcript.scrollTop, 188);
    assert.equal(body.scrollTop, 0);
  }

  {
    const target = element(77);
    output.setTextAndScroll(target, 'manifest');
    assert.equal(target.textContent, 'manifest');
    assert.equal(target.scrollTop, 77);
  }

  assert.doesNotThrow(() => output.setText(null, 'x'));
  assert.doesNotThrow(() => output.scrollBottom(null));
  assert.doesNotThrow(() => output.setTextAndScroll(null, 'x'));
}

main();
