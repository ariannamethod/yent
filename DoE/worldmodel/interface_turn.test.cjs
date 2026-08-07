const assert = require('node:assert/strict');
const turn = require('./interface_turn.js');
const chatStream = require('./chat_stream.js');

function session() {
  const previews = [];
  const commits = [];
  return {
    previews,
    commits,
    previewAssistant(visibleMessages, text) {
      previews.push({ visibleMessages, text });
      return true;
    },
    commitAssistant(messages, visibleMessages, text) {
      commits.push({ messages, visibleMessages, text });
      return {
        messages: messages.concat({ role: 'assistant', content: text }),
        visibleMessages: visibleMessages.concat({ role: 'assistant', content: text }),
        committed: true
      };
    }
  };
}

function inputFor(stream) {
  return {
    readParams(options) {
      assert.equal(options.document.name, 'doc');
      return { temperature: 0.33, maxTokens: 17 };
    },
    streamFor(options) {
      assert.equal(options.replayMode, false);
      assert.equal(options.replayRequest.name, 'boundary');
      return stream;
    }
  };
}

async function main() {
  {
    const sess = session();
    const seen = [];
    const result = await turn.streamAssistant({
      paramsDocument: { name: 'doc' },
      interfaceInput: inputFor(async options => {
        assert.equal(options.temperature, 0.33);
        assert.equal(options.maxTokens, 17);
        assert.deepEqual(options.messages, [{ role: 'user', content: 'hi' }]);
        options.onToken('he', { step: 1 });
        options.onToken('llo', { step: 2 });
        return { done: true };
      }),
      chatStream,
      sessionReceipt: sess,
      replayMode: false,
      replayRequest: { name: 'boundary' },
      messages: [{ role: 'user', content: 'hi' }],
      visibleMessages: [{ role: 'user', content: 'hi' }],
      signal: 'signal',
      onToken: (token, data, text) => seen.push({ token, step: data.step, text })
    });
    assert.equal(result.text, 'hello');
    assert.equal(result.outcome.kind, 'complete');
    assert.equal(result.committed, true);
    assert.deepEqual(result.messages.at(-1), { role: 'assistant', content: 'hello' });
    assert.deepEqual(seen, [
      { token: 'he', step: 1, text: 'he' },
      { token: 'llo', step: 2, text: 'hello' }
    ]);
    assert.deepEqual(sess.previews.map(p => p.text), ['he', 'hello']);
    assert.equal(sess.commits.length, 1);
  }

  {
    const sess = session();
    const boom = new Error('stream broke');
    const result = await turn.streamAssistant({
      paramsDocument: { name: 'doc' },
      interfaceInput: inputFor(async options => {
        options.onToken('partial', { step: 1 });
        throw boom;
      }),
      chatStream,
      sessionReceipt: sess,
      replayRequest: { name: 'boundary' },
      messages: [{ role: 'user', content: 'hi' }],
      visibleMessages: []
    });
    assert.equal(result.text, 'partial');
    assert.equal(result.outcome.kind, 'fault');
    assert.equal(result.outcome.commitAssistant, false);
    assert.equal(result.error, boom);
    assert.equal(result.committed, false);
    assert.equal(sess.commits.length, 0);
  }

  {
    const sess = session();
    const abort = new Error('user stopped');
    abort.name = 'AbortError';
    const result = await turn.streamAssistant({
      paramsDocument: { name: 'doc' },
      interfaceInput: inputFor(async options => {
        options.onToken('partial', { step: 1 });
        throw abort;
      }),
      chatStream,
      sessionReceipt: sess,
      replayRequest: { name: 'boundary' },
      messages: [{ role: 'user', content: 'hi' }],
      visibleMessages: []
    });
    assert.equal(result.outcome.kind, 'stopped');
    assert.equal(result.outcome.commitAssistant, true);
    assert.equal(result.committed, true);
    assert.deepEqual(result.messages.at(-1), { role: 'assistant', content: 'partial' });
  }

  {
    await assert.rejects(
      () => turn.streamAssistant(),
      /YentInterfaceInput helper missing/
    );
    await assert.rejects(
      () => turn.streamAssistant({}),
      /YentInterfaceInput helper missing/
    );
    await assert.rejects(
      () => turn.streamAssistant(null),
      /interface turn options must be passed as an object/
    );
    await assert.rejects(
      () => turn.streamAssistant([]),
      /interface turn options must be passed as an object/
    );
    await assert.rejects(
      () => turn.streamAssistant('legacy'),
      /interface turn options must be passed as an object/
    );
    await assert.rejects(
      () => turn.streamAssistant({
        interfaceInput: { readParams() { return {}; }, streamFor() {} },
        chatStream,
        session: session()
      }),
      /session adapter must be passed as \{ sessionReceipt \}/
    );
    await assert.rejects(
      () => turn.streamAssistant({
        interfaceInput: { readParams() { return {}; }, streamFor() {} },
        chatStream,
        sessionReceipt: session(),
        request: { name: 'old request' }
      }),
      /replay request must be passed as \{ replayRequest \}/
    );
    await assert.rejects(
      () => turn.streamAssistant({
        interfaceInput: { readParams() { return {}; }, streamFor() {} },
        chatStream,
        sessionReceipt: {}
      }),
      /YentInterfaceSession adapter helper missing/
    );
  }

  {
    let touched = false;
    const hadInput = Object.prototype.hasOwnProperty.call(globalThis, 'YentInterfaceInput');
    const previousInput = globalThis.YentInterfaceInput;
    globalThis.YentInterfaceInput = {
      readParams() {
        touched = true;
        return {};
      },
      streamFor() {
        touched = true;
        return async () => {};
      }
    };
    try {
      await assert.rejects(
        () => turn.streamAssistant({
          interfaceInput: null,
          chatStream,
          sessionReceipt: session()
        }),
        /YentInterfaceInput helper missing/
      );
      assert.equal(touched, false);
    } finally {
      if (hadInput) globalThis.YentInterfaceInput = previousInput;
      else delete globalThis.YentInterfaceInput;
    }
  }

  {
    let touched = false;
    const hadChat = Object.prototype.hasOwnProperty.call(globalThis, 'YentChatStream');
    const previousChat = globalThis.YentChatStream;
    globalThis.YentChatStream = {
      outcome() {
        touched = true;
        return { kind: 'complete' };
      }
    };
    try {
      await assert.rejects(
        () => turn.streamAssistant({
          interfaceInput: { readParams() { return {}; }, streamFor() { return async () => {}; } },
          chatStream: null,
          sessionReceipt: session()
        }),
        /YentChatStream helper missing/
      );
      assert.equal(touched, false);
    } finally {
      if (hadChat) globalThis.YentChatStream = previousChat;
      else delete globalThis.YentChatStream;
    }
  }

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
      const interfaceInput = require('./interface_input.js');
      const sess = session();
      const result = await turn.streamAssistant({
        paramsDocument: {
          getElementById(id) {
            return { value: id === 'temp' ? '0.8' : '8' };
          }
        },
        interfaceInput,
        interfaceReplay: null,
        chatStream,
        sessionReceipt: sess,
        replayMode: true,
        replayRequest: { name: 'boundary' },
        messages: [{ role: 'user', content: 'hi' }],
        visibleMessages: []
      });
      assert.match(result.error && result.error.message, /YentInterfaceReplay helper missing/);
      assert.equal(result.outcome.kind, 'fault');
      assert.equal(result.committed, false);
      assert.equal(touched, false);
    } finally {
      if (hadReplay) globalThis.YentInterfaceReplay = previousReplay;
      else delete globalThis.YentInterfaceReplay;
    }
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
