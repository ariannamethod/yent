(function (root) {
  'use strict';

  function requireMethod(source, name, method) {
    if (!source || typeof source[method] !== 'function') {
      throw new Error(`${name} helper missing`);
    }
    return source;
  }

  function arrayOrEmpty(source) {
    return Array.isArray(source) ? source : [];
  }

  function hasOwn(value, key) {
    return value && Object.prototype.hasOwnProperty.call(value, key);
  }

  async function streamAssistant(options) {
    options = options || {};
    if (hasOwn(options, 'session')) {
      throw new Error('session adapter must be passed as { sessionReceipt }');
    }
    if (hasOwn(options, 'request')) {
      throw new Error('replay request must be passed as { replayRequest }');
    }
    const input = requireMethod(
      hasOwn(options, 'interfaceInput') ? options.interfaceInput : root.YentInterfaceInput,
      'YentInterfaceInput',
      'readParams'
    );
    requireMethod(input, 'YentInterfaceInput', 'streamFor');
    const chat = requireMethod(
      hasOwn(options, 'chatStream') ? options.chatStream : root.YentChatStream,
      'YentChatStream',
      'outcome'
    );
    const session = options.sessionReceipt;
    requireMethod(session, 'YentInterfaceSession adapter', 'previewAssistant');
    requireMethod(session, 'YentInterfaceSession adapter', 'commitAssistant');

    let messages = arrayOrEmpty(options.messages);
    let visibleMessages = arrayOrEmpty(options.visibleMessages);
    let text = '';
    let streamError = null;
    let result = null;

    try {
      const requestParams = options.paramsDocument ? input.readParams({ document: options.paramsDocument }) : input.readParams();
      const stream = input.streamFor({
        replayMode: !!options.replayMode,
        replayRequest: options.replayRequest,
        interfaceReplay: hasOwn(options, 'interfaceReplay') ? options.interfaceReplay : root.YentInterfaceReplay,
        chatStream: chat
      });

      await stream({
        messages,
        temperature: requestParams.temperature,
        maxTokens: requestParams.maxTokens,
        signal: options.signal,
        onEvent: options.onEvent,
        onDone: options.onDone,
        onError: options.onError,
        onToken: (token, data) => {
          const chunk = typeof token === 'string' ? token : '';
          text += chunk;
          session.previewAssistant(visibleMessages, text);
          if (typeof options.onToken === 'function') options.onToken(chunk, data, text);
        }
      });
      result = chat.outcome({ error: null, responseText: text });
    } catch (err) {
      streamError = err;
      result = chat.outcome({ error: err, responseText: text });
    }

    let committed = false;
    if (result.commitAssistant) {
      const next = session.commitAssistant(messages, visibleMessages, text);
      messages = arrayOrEmpty(next.messages);
      visibleMessages = arrayOrEmpty(next.visibleMessages);
      committed = !!next.committed;
    }

    return {
      messages,
      visibleMessages,
      text,
      outcome: result,
      error: streamError,
      committed
    };
  }

  const api = { streamAssistant };
  root.YentInterfaceTurn = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
