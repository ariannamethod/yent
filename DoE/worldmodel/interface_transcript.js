(function (root) {
  'use strict';

  function requireDocument(documentRef) {
    if (!documentRef || typeof documentRef.createElement !== 'function') {
      throw new Error('document helper missing');
    }
    return documentRef;
  }

  function requireOutput(output) {
    if (!output || typeof output.setText !== 'function' || typeof output.scrollBottom !== 'function') {
      throw new Error('YentInterfaceOutput helper missing');
    }
    return output;
  }

  function hasContainer(value) {
    return !!value && typeof value.appendChild === 'function';
  }

  function rejectBareContainer(value) {
    if (hasContainer(value)) {
      throw new Error('transcript container must be passed as { container }');
    }
  }

  function labelFor(role, labels) {
    const table = labels || {};
    if (Object.prototype.hasOwnProperty.call(table, role)) return String(table[role]);
    const text = role == null ? 'turn' : String(role);
    return text.toUpperCase();
  }

  function appendTurn(options) {
    rejectBareContainer(options);
    options = options || {};
    const container = options.container;
    const documentRef = requireDocument(options.document || root.document);
    const output = requireOutput(options.interfaceOutput || root.YentInterfaceOutput);
    if (!hasContainer(container)) {
      throw new Error('transcript container missing');
    }

    const role = options.role == null ? 'assistant' : String(options.role);
    const node = documentRef.createElement('article');
    node.className = `turn ${role}`;

    const label = documentRef.createElement('div');
    label.className = 'role';
    output.setText(label, labelFor(role, options.labels));

    const body = documentRef.createElement('div');
    body.className = 'text';
    output.setText(body, options.text || '');

    node.appendChild(label);
    node.appendChild(body);
    container.appendChild(node);
    output.scrollBottom(container);
    return body;
  }

  function clear(options) {
    rejectBareContainer(options);
    options = options || {};
    const container = options.container;
    if (!container) return;
    const output = requireOutput((options && options.interfaceOutput) || root.YentInterfaceOutput);
    output.setText(container, '');
  }

  const api = { labelFor, appendTurn, clear };
  root.YentInterfaceTranscript = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
