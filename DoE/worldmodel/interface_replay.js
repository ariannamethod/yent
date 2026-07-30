(function (root) {
  'use strict';

  const DEFAULT_SCENARIO = 'boundary';
  const DEFAULT_DELAY_MS = 42;

  const SCENARIOS = {
    boundary: {
      prompt: 'show the boundary between chosen answer and rejected candidates',
      events: [
        tokenEvent('The', 1, 0.32, 1, 0.47, [' The', ' A', ' This', ' There']),
        tokenEvent(' chosen', 2, 0.28, 1, 0.41, [' chosen', ' rejected', ' possible', ' hidden']),
        tokenEvent(' answer', 3, 0.24, 1, 0.39, [' answer', ' path', ' token', ' field']),
        tokenEvent(' becomes', 4, 0.19, 2, 0.44, [' is', ' becomes', ' stays', ' turns']),
        tokenEvent(' visible', 5, 0.31, 1, 0.36, [' visible', ' partial', ' silent', ' unstable']),
        tokenEvent(' while', 6, 0.22, 1, 0.33, [' while', ' because', ' before', ' through']),
        tokenEvent(' rejected', 7, 0.17, 3, 0.52, [' candidate', ' latent', ' rejected', ' alternate']),
        tokenEvent(' candidates', 8, 0.26, 1, 0.43, [' candidates', ' thoughts', ' routes', ' shadows']),
        tokenEvent(' remain', 9, 0.34, 1, 0.29, [' remain', ' dissolve', ' return', ' pulse']),
        tokenEvent(' in', 10, 0.38, 1, 0.22, [' in', ' as', ' near', ' under']),
        tokenEvent(' motion', 11, 0.29, 1, 0.35, [' motion', ' memory', ' pressure', ' silence']),
        tokenEvent('.', 12, 0.45, 1, 0.18, ['.', ';', ',', ' --'])
      ]
    }
  };

  function tokenEvent(token, step, prob, rank, tail, topTokens) {
    const selected = Math.max(0, Math.min(topTokens.length - 1, rank - 1));
    return {
      token,
      token_id: 30000 + step,
      step,
      experts: 2 + (step % 5),
      debt: clamp(0.44 - step * 0.018 + tail * 0.07, 0, 1),
      consensus: clamp(0.16 + step * 0.038, 0, 1),
      field_health: clamp(0.9 + step * 0.004 - tail * 0.06, 0, 1),
      entropy: clamp(4.1 - step * 0.13 + tail * 0.4, 0, 12),
      selected_prob: prob,
      selected_rank: rank,
      selected_logprob: Math.log(Math.max(prob, 1e-9)),
      candidate_tail_mass: tail,
      top_tokens: topTokens.map((text, i) => {
        const p = i === selected ? prob : Math.max(0.015, tail / Math.max(2, topTokens.length + i));
        return {
          token: text,
          prob: p,
          logprob: Math.log(Math.max(p, 1e-9)),
          rank: i + 1,
          selected: i === selected
        };
      })
    };
  }

  function clamp(value, min, max) {
    const n = Number.isFinite(value) ? value : min;
    return Math.max(min, Math.min(max, n));
  }

  function clampInteger(value, fallback, min, max) {
    const n = Number.isFinite(value) ? Math.floor(value) : fallback;
    return Math.max(min, Math.min(max, n));
  }

  function searchText(location) {
    if (!location) return '';
    let search = '';
    if (typeof location === 'string') {
      search = location;
    } else if (typeof location.search === 'string') {
      search = location.search;
    } else if (typeof location.href === 'string') {
      search = location.href;
    }
    const hashAt = search.indexOf('#');
    if (hashAt >= 0) search = search.slice(0, hashAt);
    const queryAt = search.indexOf('?');
    if (queryAt >= 0) search = search.slice(queryAt + 1);
    if (search.charAt(0) === '?') search = search.slice(1);
    return search;
  }

  function paramsFor(location) {
    return new URLSearchParams(searchText(location));
  }

  function hasOwn(value, key) {
    return !!value && Object.prototype.hasOwnProperty.call(value, key);
  }

  function enabledFlag(params, key) {
    if (!params.has(key)) return false;
    const value = String(params.get(key) || '').toLowerCase();
    return value === '' || value === '1' || value === 'true' || value === 'yes' ||
      value === 'on' || value === 'demo';
  }

  function scenarioName(value) {
    const name = typeof value === 'string' && value ? value : DEFAULT_SCENARIO;
    return Object.prototype.hasOwnProperty.call(SCENARIOS, name) ? name : DEFAULT_SCENARIO;
  }

  function cloneEvent(event) {
    const next = Object.assign({}, event);
    if (Array.isArray(event.top_tokens)) {
      next.top_tokens = event.top_tokens.map(item => Object.assign({}, item));
    }
    return next;
  }

  function scenario(name) {
    const selected = SCENARIOS[scenarioName(name)];
    return {
      prompt: selected.prompt,
      events: selected.events.map(cloneEvent)
    };
  }

  function request(options) {
    if (typeof options === 'string') {
      throw new Error('replay request location must be passed as { location }');
    }
    options = options || {};
    if (hasOwn(options, 'search')) {
      throw new Error('replay request search must be passed as { location }');
    }
    const location = hasOwn(options, 'location') ? options.location : root.location;
    const params = paramsFor(location);
    const enabled = enabledFlag(params, 'replay') || enabledFlag(params, 'demo');
    const name = scenarioName(params.get('fixture') || params.get('scenario'));
    const selected = scenario(name);
    return {
      enabled,
      name,
      prompt: enabled ? selected.prompt : '',
      delayMs: enabled ? clampInteger(Number(params.get('delay')), DEFAULT_DELAY_MS, 0, 2000) : 0
    };
  }

  function abortError() {
    const err = new Error('replay aborted');
    err.name = 'AbortError';
    return err;
  }

  function wait(delayMs, signal) {
    if (signal && signal.aborted) return Promise.reject(abortError());
    if (!delayMs) return Promise.resolve();
    return new Promise((resolve, reject) => {
      let settled = false;
      let timer = null;
      const done = fn => {
        if (settled) return;
        settled = true;
        if (timer !== null) clearTimeout(timer);
        if (signal && typeof signal.removeEventListener === 'function') {
          signal.removeEventListener('abort', onAbort);
        }
        fn();
      };
      const onAbort = () => done(() => reject(abortError()));
      timer = setTimeout(() => done(resolve), delayMs);
      if (signal && typeof signal.addEventListener === 'function') {
        signal.addEventListener('abort', onAbort, { once: true });
      }
    });
  }

  async function play(options) {
    options = options || {};
    if (hasOwn(options, 'name')) {
      throw new Error('replay scenario name must be passed as { scenario }');
    }
    const selected = typeof options.scenario === 'object' && options.scenario
      ? { prompt: options.scenario.prompt || '', events: (options.scenario.events || []).map(cloneEvent) }
      : scenario(options.scenario);
    const delayMs = clampInteger(Number(options.delayMs), DEFAULT_DELAY_MS, 0, 2000);
    let events = 0;
    let tokens = 0;

    for (let i = 0; i < selected.events.length; i++) {
      await wait(i === 0 ? 0 : delayMs, options.signal);
      const event = cloneEvent(selected.events[i]);
      events++;
      if (typeof options.onEvent === 'function') options.onEvent(event);
      if (event && typeof event.error === 'string' && event.error) {
        throw new Error(event.error);
      }
      if (event && typeof event.token === 'string') {
        tokens++;
        if (typeof options.onToken === 'function') options.onToken(event.token, event);
      }
    }

    await wait(delayMs, options.signal);
    const doneEvent = { done: true };
    events++;
    if (typeof options.onEvent === 'function') options.onEvent(doneEvent);
    if (typeof options.onDone === 'function') options.onDone(doneEvent);
    return { done: true, events, tokens, pending: '' };
  }

  function startIfRequested(options) {
    options = options || {};
    if (hasOwn(options, 'request')) {
      throw new Error('replay request must be passed as { replayRequest }');
    }
    if (hasOwn(options, 'input')) {
      throw new Error('replay prompt input must be passed as { promptInput }');
    }
    if (hasOwn(options, 'run')) {
      throw new Error('generation run must be passed as { generationRun }');
    }
    const req = options.replayRequest || {};
    const active = typeof options.replayMode === 'boolean' ? options.replayMode : !!req.enabled;
    if (!active) return false;

    const input = options.promptInput;
    const run = options.generationRun;
    const generate = options.generate;
    const timer = options.setTimeout || root.setTimeout;
    if (!input || typeof input.value !== 'string') {
      throw new Error('replay prompt input unavailable');
    }
    if (!run || typeof run.isRunning !== 'function') {
      throw new Error('generation run helper unavailable');
    }
    if (typeof generate !== 'function') {
      throw new Error('replay generate handler unavailable');
    }
    if (typeof timer !== 'function') {
      throw new Error('setTimeout unavailable');
    }

    const prompt = typeof req.prompt === 'string' ? req.prompt : '';
    const startDelayMs = clampInteger(Number(options.startDelayMs), 120, 0, 5000);
    input.value = prompt;
    timer(() => {
      if (!run.isRunning()) generate(prompt);
    }, startDelayMs);
    return true;
  }

  const api = {
    DEFAULT_SCENARIO,
    DEFAULT_DELAY_MS,
    request,
    scenario,
    play,
    startIfRequested
  };
  root.YentInterfaceReplay = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
