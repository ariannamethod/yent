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

  function call(callback, ...args) {
    if (typeof callback === 'function') callback(...args);
  }

  async function run(options) {
    options = options || {};
    const generationRun = options.generationRun || options.run;
    requireMethod(generationRun, 'YentInterfaceRun controller', 'begin');
    requireMethod(generationRun, 'YentInterfaceRun controller', 'finish');

    const session = options.sessionReceipt || options.session;
    requireMethod(session, 'YentInterfaceSession adapter', 'commitUser');

    const turnHelper = options.interfaceTurn || root.YentInterfaceTurn;
    requireMethod(turnHelper, 'YentInterfaceTurn', 'streamAssistant');

    let messages = arrayOrEmpty(options.messages);
    let visibleMessages = arrayOrEmpty(options.visibleMessages);
    const text = typeof options.text === 'string' ? options.text : '';
    const currentRun = generationRun.begin();

    try {
      call(options.beforeUser, currentRun);

      const userTurn = session.commitUser(messages, visibleMessages, text) || {};
      messages = arrayOrEmpty(userTurn.messages);
      visibleMessages = arrayOrEmpty(userTurn.visibleMessages);
      call(options.onUser, userTurn, currentRun);

      const turn = await turnHelper.streamAssistant({
        document: options.document || root.document,
        interfaceInput: options.interfaceInput || root.YentInterfaceInput,
        chatStream: options.chatStream || root.YentChatStream,
        interfaceReplay: options.interfaceReplay || root.YentInterfaceReplay,
        replayMode: !!options.replayMode,
        replayRequest: options.replayRequest || options.request,
        sessionReceipt: session,
        messages,
        visibleMessages,
        signal: currentRun.signal,
        onEvent: options.onEvent,
        onDone: options.onDone,
        onError: options.onError,
        onToken: options.onToken
      });

      return {
        currentRun,
        userTurn,
        turn,
        messages: arrayOrEmpty(turn && turn.messages),
        visibleMessages: arrayOrEmpty(turn && turn.visibleMessages),
        text: turn && typeof turn.text === 'string' ? turn.text : '',
        outcome: turn ? turn.outcome : null
      };
    } finally {
      generationRun.finish(currentRun);
    }
  }

  const api = { run };
  root.YentInterfaceSubmit = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
