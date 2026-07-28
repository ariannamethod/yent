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
      () => turn.streamAssistant({}),
      /YentInterfaceInput helper missing/
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
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
