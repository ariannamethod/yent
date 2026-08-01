const assert = require('node:assert/strict');
const submit = require('./interface_submit.js');

function generationRun() {
  const calls = [];
  return {
    calls,
    begin() {
      calls.push('begin');
      return { id: 1, signal: 'signal' };
    },
    finish(run) {
      calls.push(`finish:${run && run.id}`);
      return true;
    }
  };
}

function session() {
  return {
    commitUser(messages, visibleMessages, text) {
      return {
        messages: messages.concat({ role: 'user', content: text }),
        visibleMessages: visibleMessages.concat({ role: 'user', content: text }),
        committed: true
      };
    }
  };
}

async function main() {
  {
    const run = generationRun();
    const events = [];
    const result = await submit.run({
      generationRun: run,
      sessionReceipt: session(),
      interfaceTurn: {
        async streamAssistant(options) {
          events.push({
            signal: options.signal,
            messages: options.messages,
            visibleMessages: options.visibleMessages,
            paramsDocument: options.paramsDocument
          });
          options.onToken('ok', { step: 1 }, 'ok');
          return {
            text: 'ok',
            outcome: { kind: 'complete', hasText: true },
            messages: options.messages.concat({ role: 'assistant', content: 'ok' }),
            visibleMessages: options.visibleMessages.concat({ role: 'assistant', content: 'ok' })
          };
        }
      },
      paramsDocument: { name: 'doc' },
      interfaceInput: { name: 'input' },
      chatStream: { name: 'chat' },
      interfaceReplay: { name: 'replay' },
      replayRequest: { name: 'request' },
      messages: [],
      visibleMessages: [],
      text: 'hello',
      beforeUser: current => events.push({ before: current.signal }),
      onUser: userTurn => events.push({ user: userTurn.messages.at(-1) }),
      onToken: (token, data, text) => events.push({ token, step: data.step, text })
    });

    assert.deepEqual(run.calls, ['begin', 'finish:1']);
    assert.deepEqual(events[0], { before: 'signal' });
    assert.deepEqual(events[1], { user: { role: 'user', content: 'hello' } });
    assert.equal(events[2].signal, 'signal');
    assert.deepEqual(events[2].messages, [{ role: 'user', content: 'hello' }]);
    assert.deepEqual(events[2].paramsDocument, { name: 'doc' });
    assert.deepEqual(events[3], { token: 'ok', step: 1, text: 'ok' });
    assert.equal(result.text, 'ok');
    assert.equal(result.outcome.kind, 'complete');
    assert.deepEqual(result.messages.at(-1), { role: 'assistant', content: 'ok' });
  }

  {
    const run = generationRun();
    await assert.rejects(
      submit.run({
        generationRun: run,
        sessionReceipt: session(),
        interfaceTurn: {
          async streamAssistant() {
            throw new Error('boom');
          }
        },
        text: 'hello'
      }),
      /boom/
    );
    assert.deepEqual(run.calls, ['begin', 'finish:1']);
  }

  {
    await assert.rejects(() => submit.run({}), /YentInterfaceRun controller helper missing/);
    await assert.rejects(
      () => submit.run({ run: generationRun(), sessionReceipt: session(), interfaceTurn: { streamAssistant() {} } }),
      /generation run must be passed as \{ generationRun \}/
    );
    await assert.rejects(
      () => submit.run({ generationRun: generationRun(), session: session(), interfaceTurn: { streamAssistant() {} } }),
      /session adapter must be passed as \{ sessionReceipt \}/
    );
    await assert.rejects(
      () => submit.run({
        generationRun: generationRun(),
        sessionReceipt: session(),
        request: { name: 'old request' },
        interfaceTurn: { streamAssistant() {} }
      }),
      /replay request must be passed as \{ replayRequest \}/
    );
    await assert.rejects(
      () => submit.run({ generationRun: generationRun(), sessionReceipt: {}, interfaceTurn: { streamAssistant() {} } }),
      /YentInterfaceSession adapter helper missing/
    );
    await assert.rejects(
      () => submit.run({ generationRun: generationRun(), sessionReceipt: session(), interfaceTurn: {} }),
      /YentInterfaceTurn helper missing/
    );
  }

  {
    let touched = false;
    const hadTurn = Object.prototype.hasOwnProperty.call(globalThis, 'YentInterfaceTurn');
    const previousTurn = globalThis.YentInterfaceTurn;
    globalThis.YentInterfaceTurn = {
      streamAssistant() {
        touched = true;
        return {};
      }
    };
    try {
      await assert.rejects(
        () => submit.run({
          generationRun: generationRun(),
          sessionReceipt: session(),
          interfaceTurn: null,
          text: 'hello'
        }),
        /YentInterfaceTurn helper missing/
      );
      assert.equal(touched, false);
    } finally {
      if (hadTurn) globalThis.YentInterfaceTurn = previousTurn;
      else delete globalThis.YentInterfaceTurn;
    }
  }

  {
    const run = generationRun();
    const seen = {};
    const result = await submit.run({
      generationRun: run,
      sessionReceipt: session(),
      interfaceTurn: {
        async streamAssistant(options) {
          seen.interfaceInput = options.interfaceInput;
          seen.chatStream = options.chatStream;
          seen.interfaceReplay = options.interfaceReplay;
          return {
            text: '',
            outcome: { kind: 'complete' },
            messages: options.messages,
            visibleMessages: options.visibleMessages
          };
        }
      },
      interfaceInput: null,
      chatStream: null,
      interfaceReplay: null,
      text: 'hello'
    });
    assert.equal(seen.interfaceInput, null);
    assert.equal(seen.chatStream, null);
    assert.equal(seen.interfaceReplay, null);
    assert.equal(result.outcome.kind, 'complete');
    assert.deepEqual(run.calls, ['begin', 'finish:1']);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
