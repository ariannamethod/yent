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

  function hasOwn(value, key) {
    return value && Object.prototype.hasOwnProperty.call(value, key);
  }

  function optionTable(options) {
    if (options === undefined) return {};
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
      throw new Error('interface submit options must be passed as an object');
    }
    return options;
  }

  async function run(options) {
    options = optionTable(options);
    if (hasOwn(options, 'run')) {
      throw new Error('generation run must be passed as { generationRun }');
    }
    if (hasOwn(options, 'session')) {
      throw new Error('session adapter must be passed as { sessionReceipt }');
    }
    if (hasOwn(options, 'request')) {
      throw new Error('replay request must be passed as { replayRequest }');
    }
    const generationRun = options.generationRun;
    requireMethod(generationRun, 'YentInterfaceRun controller', 'begin');
    requireMethod(generationRun, 'YentInterfaceRun controller', 'finish');

    const session = options.sessionReceipt;
    requireMethod(session, 'YentInterfaceSession adapter', 'commitUser');

    const turnHelper = hasOwn(options, 'interfaceTurn') ? options.interfaceTurn : root.YentInterfaceTurn;
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
        paramsDocument: options.paramsDocument,
        interfaceInput: hasOwn(options, 'interfaceInput') ? options.interfaceInput : root.YentInterfaceInput,
        chatStream: hasOwn(options, 'chatStream') ? options.chatStream : root.YentChatStream,
        interfaceReplay: hasOwn(options, 'interfaceReplay') ? options.interfaceReplay : root.YentInterfaceReplay,
        replayMode: !!options.replayMode,
        replayRequest: options.replayRequest,
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
