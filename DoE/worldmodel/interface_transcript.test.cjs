const assert = require('node:assert/strict');
const transcript = require('./interface_transcript.js');
const output = require('./interface_output.js');

function documentMock() {
  return {
    createElement(tagName) {
      return {
        tagName,
        className: '',
        textContent: '',
        children: [],
        appendChild(child) {
          this.children.push(child);
          return child;
        }
      };
    }
  };
}

function containerMock() {
  return {
    textContent: 'old',
    scrollTop: 0,
    scrollHeight: 144,
    children: [],
    appendChild(child) {
      this.children.push(child);
      return child;
    }
  };
}

function main() {
  assert.equal(transcript.labelFor('user', { user: 'OLEG' }), 'OLEG');
  assert.equal(transcript.labelFor('assistant', { assistant: 'YENT' }), 'YENT');
  assert.equal(transcript.labelFor('observer'), 'OBSERVER');

  {
    const container = containerMock();
    const body = transcript.appendTurn({
      container,
      document: documentMock(),
      interfaceOutput: output,
      role: 'assistant',
      text: 'I am here.',
      labels: { assistant: 'YENT' }
    });
    assert.equal(container.children.length, 1);
    assert.equal(container.scrollTop, 144);
    const node = container.children[0];
    assert.equal(node.tagName, 'article');
    assert.equal(node.className, 'turn assistant');
    assert.equal(node.children[0].className, 'role');
    assert.equal(node.children[0].textContent, 'YENT');
    assert.equal(node.children[1], body);
    assert.equal(body.className, 'text');
    assert.equal(body.textContent, 'I am here.');
  }

  {
    const container = containerMock();
    transcript.clear({ container, interfaceOutput: output });
    assert.equal(container.textContent, '');
  }

  {
    const container = containerMock();
    const body = transcript.appendTurn({
      container,
      document: documentMock(),
      role: 'user',
      text: 'global fallback',
      labels: { user: 'OLEG' }
    });
    assert.equal(body.textContent, 'global fallback');
    assert.equal(container.children[0].children[0].textContent, 'OLEG');
  }

  assert.throws(() => transcript.appendTurn(containerMock()), /container must be passed as \{ container \}/);
  assert.throws(() => transcript.clear(containerMock()), /container must be passed as \{ container \}/);
  assert.throws(() => transcript.appendTurn({ document: documentMock(), interfaceOutput: output }), /transcript container missing/);
  assert.throws(() => transcript.appendTurn({ container: containerMock(), interfaceOutput: output }), /document helper missing/);
  {
    const saved = globalThis.YentInterfaceOutput;
    delete globalThis.YentInterfaceOutput;
    assert.throws(() => transcript.appendTurn({ container: containerMock(), document: documentMock() }), /YentInterfaceOutput helper missing/);
    globalThis.YentInterfaceOutput = saved;
  }
}

main();
