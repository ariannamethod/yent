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
  assert.deepEqual(input.bindControls({
    document: docElements({
      composer,
      prompt: promptInput,
      send: sendButton
    })
  }), {
    composer,
    promptInput,
    sendButton
  });
}

{
  const composer = { addEventListener() {} };
  const promptInput = { value: '' };
  const sendButton = {};
  assert.deepEqual(input.bindControls({
    document: docElements({
      ask: composer,
      words: promptInput,
      go: sendButton
    }),
    ids: {
      composer: 'ask',
      prompt: 'words',
      send: 'go'
    }
  }), {
    composer,
    promptInput,
    sendButton
  });
}

{
  const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document');
  const previousDocument = globalThis.document;
  const composer = { addEventListener() {} };
  const promptInput = { value: 'default doc' };
  const sendButton = { textContent: 'SEND' };
  globalThis.document = docElements({ composer, prompt: promptInput, send: sendButton });
  try {
    assert.deepEqual(input.bindControls(), { composer, promptInput, sendButton });
  } finally {
    if (hadDocument) globalThis.document = previousDocument;
    else delete globalThis.document;
  }
}

{
  const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document');
  const previousDocument = globalThis.document;
  const composer = { addEventListener() {} };
  const promptInput = { value: 'default ids' };
  const sendButton = {};
  globalThis.document = docElements({ ask: composer, words: promptInput, go: sendButton });
  try {
    assert.deepEqual(input.bindControls({
      ids: {
        composer: 'ask',
        prompt: 'words',
        send: 'go'
      }
    }), { composer, promptInput, sendButton });
  } finally {
    if (hadDocument) globalThis.document = previousDocument;
    else delete globalThis.document;
  }
}

{
  assert.throws(() => input.bindControls(null), /document unavailable/);
  assert.throws(() => input.bindControls(docElements({})), /document must be passed as \{ document \}/);
  assert.throws(() => input.bindControls({ composer: 'ask' }), /control ids must be passed as \{ ids \}/);
  assert.throws(() => input.bindControls({ prompt: 'words' }), /control ids must be passed as \{ ids \}/);
  assert.throws(() => input.bindControls({ send: 'go' }), /control ids must be passed as \{ ids \}/);
  assert.throws(() => input.bindControls({ document: docElements({}), ids: null }), /control ids must be passed as an object/);
  assert.throws(() => input.bindControls({ document: docElements({}), ids: 'ask' }), /control ids must be passed as an object/);
  assert.throws(() => input.bindControls({ document: docElements({}) }), /composer control unavailable: composer/);
  assert.throws(() => input.bindControls({ document: docElements({
    composer: {},
    prompt: { value: '' },
    send: {}
  }) }), /composer control cannot receive submit events/);
  assert.throws(() => input.bindControls({ document: docElements({
    composer: { addEventListener() {} },
    prompt: {},
    send: {}
  }) }), /prompt control must expose a string value/);
  assert.throws(() => input.bindControls({ document: docElements({
    composer: { addEventListener() {} },
    prompt: { value: '' }
  }) }), /send control unavailable: send/);
}

{
  assert.throws(() => input.readParams(doc({ temp: '0.35' })), /document must be passed as \{ document \}/);
  assert.deepEqual(input.readParams({
    document: doc({
      temp: '0.35',
      'max-tokens': '33'
    })
  }), {
    temperature: 0.35,
    maxTokens: 33
  });
}

{
  assert.deepEqual(input.readParams({
    document: doc({
      temp: '9',
      'max-tokens': '-4'
    })
  }), {
    temperature: 2,
    maxTokens: 1
  });
}

{
  assert.deepEqual(input.readParams({
    document: doc({
      temp: '0',
      'max-tokens': '0'
    })
  }), {
    temperature: 0,
    maxTokens: 1
  });
}

{
  assert.deepEqual(input.readParams({
    document: doc({
      temp: 'not-a-number',
      'max-tokens': 'not-a-number'
    })
  }), {
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
  const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document');
  const previousDocument = globalThis.document;
  globalThis.document = doc({
    temp: '1.25',
    'max-tokens': '99'
  });
  try {
    assert.deepEqual(input.readParams(), {
      temperature: 1.25,
      maxTokens: 99
    });
  } finally {
    if (hadDocument) globalThis.document = previousDocument;
    else delete globalThis.document;
  }
}

{
  const promptInput = { value: '' };
  const other = { value: '' };
  const d = docElements({ prompt: promptInput });
  d.activeElement = promptInput;
  assert.throws(() => input.isFocused(d), /document must be passed as \{ document \}/);
  assert.equal(input.isFocused({ document: d, control: promptInput }), true);
  assert.equal(input.isFocused({ document: d, control: other }), false);
  assert.equal(input.isFocused({ document: null, control: promptInput }), false);
  assert.equal(input.isFocused({ document: d, control: null }), false);
}

{
  const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document');
  const previousDocument = globalThis.document;
  const promptInput = { value: '' };
  globalThis.document = { activeElement: promptInput };
  try {
    assert.equal(input.isFocused({ control: promptInput }), true);
    assert.equal(input.isFocused({ control: { value: '' } }), false);
    assert.throws(() => input.isFocused(promptInput), /focus control must be passed as \{ control \}/);
  } finally {
    if (hadDocument) globalThis.document = previousDocument;
    else delete globalThis.document;
  }
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

  {
    let touched = false;
    const hadReplay = Object.prototype.hasOwnProperty.call(globalThis, 'YentInterfaceReplay');
    const previousReplay = globalThis.YentInterfaceReplay;
    globalThis.YentInterfaceReplay = {
      play() {
        touched = true;
        return { done: true };
      }
    };
    try {
      assert.throws(
        () => input.streamFor({ replayMode: true, interfaceReplay: null }),
        /YentInterfaceReplay helper missing/
      );
      assert.equal(touched, false);
    } finally {
      if (hadReplay) globalThis.YentInterfaceReplay = previousReplay;
      else delete globalThis.YentInterfaceReplay;
    }
  }

  {
    let touched = false;
    const hadChat = Object.prototype.hasOwnProperty.call(globalThis, 'YentChatStream');
    const previousChat = globalThis.YentChatStream;
    globalThis.YentChatStream = {
      stream() {
        touched = true;
        return { done: true };
      }
    };
    try {
      assert.throws(
        () => input.streamFor({ replayMode: false, chatStream: null }),
        /YentChatStream helper missing/
      );
      assert.equal(touched, false);
    } finally {
      if (hadChat) globalThis.YentChatStream = previousChat;
      else delete globalThis.YentChatStream;
    }
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
