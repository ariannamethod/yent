(function (root) {
  'use strict';

  function call(callback, turn, outcome) {
    if (typeof callback === 'function') callback(turn, outcome);
  }

  function resolve(submit) {
    const turn = submit && submit.turn;
    const outcome = (submit && submit.outcome) || (turn && turn.outcome);
    if (!turn || !outcome) throw new Error('YentInterfaceOutcome outcome missing');
    return { turn, outcome };
  }

  function handle(submit, handlers) {
    handlers = handlers || {};
    const resolved = resolve(submit);
    const turn = resolved.turn;
    const outcome = resolved.outcome;

    if (outcome.stopped) {
      call(handlers.stopped, turn, outcome);
      return { kind: 'stopped', turn, outcome };
    }
    if (outcome.fault) {
      call(handlers.fault, turn, outcome);
      return { kind: 'fault', turn, outcome };
    }
    call(handlers.complete, turn, outcome);
    return { kind: outcome.kind === 'empty' ? 'empty' : 'complete', turn, outcome };
  }

  const api = { handle };
  root.YentInterfaceOutcome = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
