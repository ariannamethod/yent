(function (root) {
  'use strict';

  function call(options) {
    options = options || {};
    const callback = options.callback;
    const turn = options.turn;
    const outcome = options.outcome;
    if (typeof callback === 'function') callback(turn, outcome);
  }

  function resolve(submit) {
    const turn = submit && submit.turn;
    const outcome = (submit && submit.outcome) || (turn && turn.outcome);
    if (!turn || !outcome) throw new Error('YentInterfaceOutcome outcome missing');
    return { turn, outcome };
  }

  function handle(options) {
    options = options || {};
    if (options && options.turn && !options.submit) {
      throw new Error('YentInterfaceOutcome handle inputs must be passed as { submit, handlers }');
    }
    const handlers = options.handlers || {};
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
