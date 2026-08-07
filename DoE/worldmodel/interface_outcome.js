(function (root) {
  'use strict';

  function optionTable(options, message) {
    if (options === undefined) return {};
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
      throw new Error(message);
    }
    return options;
  }

  function call(options) {
    options = optionTable(options, 'interface outcome callback options must be passed as an object');
    const callback = options.callback;
    const turn = options.turn;
    const outcome = options.outcome;
    if (typeof callback === 'function') callback(turn, outcome);
  }

  function hasOwn(value, key) {
    return !!value && Object.prototype.hasOwnProperty.call(value, key);
  }

  function resolve(submit) {
    const turn = submit && submit.turn;
    const outcome = (submit && submit.outcome) || (turn && turn.outcome);
    if (!turn || !outcome) throw new Error('YentInterfaceOutcome outcome missing');
    return { turn, outcome };
  }

  function resolveHandlers(options) {
    if (!hasOwn(options, 'handlers')) return {};
    const handlers = options.handlers;
    if (!handlers || typeof handlers !== 'object') {
      throw new Error('YentInterfaceOutcome handlers must be passed as an object');
    }
    return handlers;
  }

  function handle(options) {
    options = optionTable(options, 'interface outcome options must be passed as an object');
    if (options && options.turn && !options.submit) {
      throw new Error('YentInterfaceOutcome handle inputs must be passed as { submit, handlers }');
    }
    const handlers = resolveHandlers(options);
    const submit = options.submit;
    const resolved = resolve(submit);
    const turn = resolved.turn;
    const outcome = resolved.outcome;

    if (outcome.stopped) {
      call({ callback: handlers.stopped, turn, outcome });
      return { kind: 'stopped', turn, outcome };
    }
    if (outcome.fault) {
      call({ callback: handlers.fault, turn, outcome });
      return { kind: 'fault', turn, outcome };
    }
    call({ callback: handlers.complete, turn, outcome });
    return { kind: outcome.kind === 'empty' ? 'empty' : 'complete', turn, outcome };
  }

  const api = { handle };
  root.YentInterfaceOutcome = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
