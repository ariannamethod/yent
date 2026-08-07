(function (root) {
  'use strict';

  const DEFAULT_ENDPOINT = '/chat/completions';

  function hasOwn(value, key) {
    return !!value && Object.prototype.hasOwnProperty.call(value, key);
  }

  function optionTable(options) {
    if (options === undefined) return {};
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
      throw new Error('chat stream options must be passed as an object');
    }
    return options;
  }

  function dependency(options) {
    const eventStream = hasOwn(options, 'eventStream') ? options.eventStream : root.YentEventStream;
    if (!eventStream || typeof eventStream.createParser !== 'function') {
      throw new Error('YentEventStream helper missing');
    }
    return eventStream;
  }

  function fetchImpl(options) {
    if (hasOwn(options, 'fetch')) {
      if (typeof options.fetch === 'function') return options.fetch;
      throw new Error('fetch unavailable');
    }
    if (typeof root.fetch === 'function') return root.fetch.bind(root);
    throw new Error('fetch unavailable');
  }

  function decoderImpl(options) {
    const Decoder = hasOwn(options, 'TextDecoder') ? options.TextDecoder : root.TextDecoder;
    if (typeof Decoder !== 'function') throw new Error('TextDecoder unavailable');
    return new Decoder();
  }

  function endpointFor(options) {
    const endpoint = hasOwn(options, 'endpoint') ? options.endpoint : DEFAULT_ENDPOINT;
    if (typeof endpoint !== 'string' || endpoint.length === 0) {
      throw new Error('chat endpoint unavailable');
    }
    return endpoint;
  }

  function clampNumber(value, fallback, min, max) {
    const n = Number.isFinite(value) ? value : fallback;
    return Math.max(min, Math.min(max, n));
  }

  function clampInteger(value, fallback, min, max) {
    return Math.floor(clampNumber(value, fallback, min, max));
  }

  function requestBody(options) {
    options = optionTable(options);
    const messages = Array.isArray(options.messages) ? options.messages : [];
    const temperature = clampNumber(options.temperature, 0.8, 0, 2);
    const maxTokens = clampInteger(options.maxTokens, 512, 1, 512);
    return JSON.stringify({
      messages,
      temperature,
      max_tokens: maxTokens
    });
  }

  function outcome(options) {
    if (options instanceof Error || typeof options === 'string') {
      throw new Error('chat outcome inputs must be passed as { error, responseText }');
    }
    options = optionTable(options);
    const error = options.error || null;
    const responseText = options.responseText;
    const text = typeof responseText === 'string' ? responseText : '';
    const hasText = text.trim().length > 0;
    if (!error) {
      return {
        kind: hasText ? 'complete' : 'empty',
        hasText,
        commitAssistant: hasText,
        fault: false,
        stopped: false,
        message: ''
      };
    }
    const stopped = error && error.name === 'AbortError';
    const message = error && error.message ? error.message : 'stream failed';
    return {
      kind: stopped ? 'stopped' : 'fault',
      hasText,
      commitAssistant: stopped && hasText,
      fault: !stopped,
      stopped,
      message
    };
  }

  async function stream(options) {
    options = optionTable(options);
    const eventStream = dependency(options);
    const endpoint = endpointFor(options);
    const response = await fetchImpl(options)(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: options.signal,
      body: requestBody(options)
    });

    if (!response || !response.ok) {
      const status = response && Number.isFinite(response.status) ? response.status : 0;
      throw new Error(`HTTP ${status}`);
    }
    if (!response.body || typeof response.body.getReader !== 'function') {
      throw new Error('response body unavailable');
    }

    const reader = response.body.getReader();
    const decoder = decoderImpl(options);
    let doneSeen = false;
    let streamError = null;
    let events = 0;
    let tokens = 0;
    const parser = eventStream.createParser(data => {
      if (doneSeen || streamError) return;
      events++;
      if (typeof options.onEvent === 'function') options.onEvent(data);
      if (data && typeof data.error === 'string' && data.error) {
        streamError = new Error(data.error);
        if (typeof options.onError === 'function') options.onError(streamError, data);
        doneSeen = true;
        return;
      }
      if (data && data.done) {
        doneSeen = true;
        if (typeof options.onDone === 'function') options.onDone(data);
        return;
      }
      if (data && typeof data.token === 'string') {
        tokens++;
        if (typeof options.onToken === 'function') options.onToken(data.token, data);
      }
    });

    while (!doneSeen) {
      const next = await reader.read();
      if (!next || next.done) break;
      parser.push(decoder.decode(next.value, { stream: true }));
    }
    parser.push(decoder.decode());

    if (streamError) throw streamError;
    if (!doneSeen && !options.allowEof) throw new Error('stream ended before done');

    return {
      done: doneSeen,
      events,
      tokens,
      pending: parser.pending()
    };
  }

  const api = { DEFAULT_ENDPOINT, outcome, requestBody, stream };
  root.YentChatStream = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
