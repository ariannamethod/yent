const assert = require('node:assert/strict');
const chat = require('./chat_stream.js');
const eventStream = require('./event_stream.js');

function makeResponse(chunks, options = {}) {
  let index = 0;
  const encoder = new TextEncoder();
  return {
    ok: options.ok !== false,
    status: options.status || 200,
    body: options.noBody ? null : {
      getReader() {
        return {
          async read() {
            if (index >= chunks.length) return { done: true };
            return { done: false, value: encoder.encode(chunks[index++]) };
          }
        };
      }
    }
  };
}

async function main() {
{
  const body = JSON.parse(chat.requestBody({
    messages: [{ role: 'user', content: 'hello' }],
    temperature: 0.4,
    maxTokens: 33.8
  }));
  assert.deepEqual(body, {
    messages: [{ role: 'user', content: 'hello' }],
    temperature: 0.4,
    max_tokens: 33
  });
}

{
  const body = JSON.parse(chat.requestBody({
    messages: 'not-array',
    temperature: 9,
    maxTokens: -3
  }));
  assert.deepEqual(body, {
    messages: [],
    temperature: 2,
    max_tokens: 1
  });
}

{
  assert.deepEqual(chat.outcome({ error: null, responseText: 'answer' }), {
    kind: 'complete',
    hasText: true,
    commitAssistant: true,
    fault: false,
    stopped: false,
    message: ''
  });
  assert.deepEqual(chat.outcome({ error: null, responseText: '   ' }), {
    kind: 'empty',
    hasText: false,
    commitAssistant: false,
    fault: false,
    stopped: false,
    message: ''
  });
  const abort = new Error('aborted');
  abort.name = 'AbortError';
  assert.deepEqual(chat.outcome({ error: abort, responseText: 'partial' }), {
    kind: 'stopped',
    hasText: true,
    commitAssistant: true,
    fault: false,
    stopped: true,
    message: 'aborted'
  });
  assert.deepEqual(chat.outcome({ error: new Error('stream ended before done'), responseText: 'partial' }), {
    kind: 'fault',
    hasText: true,
    commitAssistant: false,
    fault: true,
    stopped: false,
    message: 'stream ended before done'
  });
  assert.throws(
    () => chat.outcome(new Error('old'), 'partial'),
    /outcome inputs must be passed as \{ error, responseText \}/
  );
}

{
  let captured = null;
  const seen = [];
  const result = await chat.stream({
    eventStream,
    messages: [{ role: 'user', content: 'hi' }],
    temperature: 0.7,
    maxTokens: 12,
    fetch: async (url, init) => {
      captured = { url, init };
      return makeResponse([
        'data: {"token":"he',
        'llo","selected_prob":0.5}\n\n',
        'data: {"done":true}\n\n'
      ]);
    },
    onToken: (token, data) => seen.push({ token, prob: data.selected_prob })
  });
  assert.equal(captured.url, '/chat/completions');
  assert.equal(captured.init.method, 'POST');
  assert.equal(captured.init.headers['Content-Type'], 'application/json');
  assert.deepEqual(JSON.parse(captured.init.body), {
    messages: [{ role: 'user', content: 'hi' }],
    temperature: 0.7,
    max_tokens: 12
  });
  assert.deepEqual(seen, [{ token: 'hello', prob: 0.5 }]);
  assert.deepEqual(result, { done: true, events: 2, tokens: 1, pending: '' });
}

{
  const prior = globalThis.YentEventStream;
  let touched = false;
  globalThis.YentEventStream = {
    createParser() {
      touched = true;
      return eventStream.createParser(() => {});
    }
  };
  try {
    await assert.rejects(
      () => chat.stream({
        eventStream: null,
        fetch: async () => makeResponse(['data: {"done":true}\n\n'])
      }),
      /YentEventStream helper missing/
    );
    assert.equal(touched, false);
  } finally {
    if (prior === undefined) delete globalThis.YentEventStream;
    else globalThis.YentEventStream = prior;
  }
}

{
  const prior = globalThis.fetch;
  let touched = false;
  globalThis.fetch = async () => {
    touched = true;
    return makeResponse(['data: {"done":true}\n\n']);
  };
  try {
    await assert.rejects(
      () => chat.stream({
        eventStream,
        fetch: null
      }),
      /fetch unavailable/
    );
    assert.equal(touched, false);
  } finally {
    if (prior === undefined) delete globalThis.fetch;
    else globalThis.fetch = prior;
  }
}

{
  const prior = globalThis.TextDecoder;
  let touched = false;
  globalThis.TextDecoder = function TouchingDecoder() {
    touched = true;
    return new prior();
  };
  try {
    await assert.rejects(
      () => chat.stream({
        eventStream,
        TextDecoder: null,
        fetch: async () => makeResponse(['data: {"done":true}\n\n'])
      }),
      /TextDecoder unavailable/
    );
    assert.equal(touched, false);
  } finally {
    if (prior === undefined) delete globalThis.TextDecoder;
    else globalThis.TextDecoder = prior;
  }
}

{
  let touched = false;
  await assert.rejects(
    () => chat.stream({
      eventStream,
      endpoint: null,
      fetch: async () => {
        touched = true;
        return makeResponse(['data: {"done":true}\n\n']);
      }
    }),
    /chat endpoint unavailable/
  );
  assert.equal(touched, false);
}

{
  const seen = [];
  await assert.rejects(
    () => chat.stream({
      eventStream,
      fetch: async () => makeResponse([
        'data: {"token":"before"}\n\n',
        'data: {"error":"tokenization failed"}\n\ndata: {"token":"after"}\n\n'
      ]),
      onToken: token => seen.push(token)
    }),
    /tokenization failed/
  );
  assert.deepEqual(seen, ['before']);
}

{
  await assert.rejects(
    () => chat.stream({
      eventStream,
      fetch: async () => makeResponse(['data: {"token":"cut"}\n\n'])
    }),
    /stream ended before done/
  );
}

{
  const result = await chat.stream({
    eventStream,
    allowEof: true,
    fetch: async () => makeResponse(['data: {"token":"partial"}\n\n'])
  });
  assert.deepEqual(result, { done: false, events: 1, tokens: 1, pending: '' });
}

{
  await assert.rejects(
    () => chat.stream({
      eventStream,
      fetch: async () => makeResponse([], { ok: false, status: 503 })
    }),
    /HTTP 503/
  );
}

{
  await assert.rejects(
    () => chat.stream({
      eventStream,
      fetch: async () => makeResponse([], { noBody: true })
    }),
    /response body unavailable/
  );
}

}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
