const assert = require('node:assert/strict');
const input = require('./interface_input.js');

function doc(values) {
  return {
    getElementById(id) {
      if (!Object.prototype.hasOwnProperty.call(values, id)) return null;
      return { value: values[id] };
    }
  };
}

function docElements(elements) {
  return {
    getElementById(id) {
      return Object.prototype.hasOwnProperty.call(elements, id) ? elements[id] : null;
    }
  };
}

{
  const composer = { addEventListener() {} };
  const promptInput = { value: 'speak' };
  const sendButton = { textContent: 'SEND' };
  assert.deepEqual(input.bindControls(docElements({
    composer,
    prompt: promptInput,
    send: sendButton
  })), {
    composer,
    promptInput,
    sendButton
  });
}

{
  const composer = { addEventListener() {} };
  const promptInput = { value: '' };
  const sendButton = {};
  assert.deepEqual(input.bindControls(docElements({
    ask: composer,
    words: promptInput,
    go: sendButton
  }), {
    composer: 'ask',
    prompt: 'words',
    send: 'go'
  }), {
    composer,
    promptInput,
    sendButton
  });
}

{
  assert.throws(() => input.bindControls(null), /document unavailable/);
  assert.throws(() => input.bindControls(docElements({})), /composer control unavailable: composer/);
  assert.throws(() => input.bindControls(docElements({
    composer: {},
    prompt: { value: '' },
    send: {}
  })), /composer control cannot receive submit events/);
  assert.throws(() => input.bindControls(docElements({
    composer: { addEventListener() {} },
    prompt: {},
    send: {}
  })), /prompt control must expose a string value/);
  assert.throws(() => input.bindControls(docElements({
    composer: { addEventListener() {} },
    prompt: { value: '' }
  })), /send control unavailable: send/);
}

{
  assert.deepEqual(input.readParams(doc({
    temp: '0.35',
    'max-tokens': '33'
  })), {
    temperature: 0.35,
    maxTokens: 33
  });
}

{
  assert.deepEqual(input.readParams(doc({
    temp: '9',
    'max-tokens': '-4'
  })), {
    temperature: 2,
    maxTokens: 1
  });
}

{
  assert.deepEqual(input.readParams(doc({
    temp: '0',
    'max-tokens': '0'
  })), {
    temperature: 0,
    maxTokens: 1
  });
}

{
  assert.deepEqual(input.readParams(doc({
    temp: 'not-a-number',
    'max-tokens': 'not-a-number'
  })), {
    temperature: 0.8,
    maxTokens: 512
  });
}

{
  assert.deepEqual(input.readParams(null), {
    temperature: 0.8,
    maxTokens: 512
  });
}

{
  const promptInput = { value: '' };
  const other = { value: '' };
  const d = docElements({ prompt: promptInput });
  d.activeElement = promptInput;
  assert.equal(input.isFocused(d, promptInput), true);
  assert.equal(input.isFocused(d, other), false);
  assert.equal(input.isFocused(null, promptInput), false);
  assert.equal(input.isFocused(d, null), false);
}

async function main() {
  {
    let seen = null;
    const stream = input.streamFor({
      replayMode: false,
      chatStream: {
        stream: async options => {
          seen = options;
          return { done: true, live: true };
        }
      }
    });
    const result = await stream({ messages: [{ role: 'user', content: 'hi' }] });
    assert.deepEqual(seen, { messages: [{ role: 'user', content: 'hi' }] });
    assert.deepEqual(result, { done: true, live: true });
  }

  {
    let seen = null;
    const stream = input.streamFor({
      replayMode: true,
      replayRequest: { name: 'boundary', delayMs: 7 },
      interfaceReplay: {
        play: async options => {
          seen = options;
          return { done: true, replay: true };
        }
      }
    });
    const result = await stream({ signal: 'signal', maxTokens: 12 });
    assert.deepEqual(seen, {
      signal: 'signal',
      maxTokens: 12,
      scenario: 'boundary',
      delayMs: 7
    });
    assert.deepEqual(result, { done: true, replay: true });
  }

  assert.throws(() => input.streamFor({ replayMode: false }), /YentChatStream helper missing/);
  assert.throws(() => input.streamFor({ replayMode: true }), /YentInterfaceReplay helper missing/);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
