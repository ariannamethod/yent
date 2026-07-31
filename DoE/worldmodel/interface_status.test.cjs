const assert = require('node:assert/strict');
const status = require('./interface_status.js');

function makeDocument() {
  const elements = new Map();
  return {
    elements,
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, { id, textContent: '', dataset: {} });
      return elements.get(id);
    }
  };
}

function main() {
  const document = makeDocument();
  const labels = status.bind({
    document,
    ids: {
      run: 'run-state',
      note: 'status-note',
      manifest: 'manifest-state',
      shell: 'manifest-shell'
    }
  });

  status.setText({ target: labels.run, text: 'GENERATING' });
  assert.equal(labels.run.textContent, 'GENERATING');
  status.setText({ target: labels.note, text: null });
  assert.equal(labels.note.textContent, '');

  status.setActive({ target: labels.shell, active: true });
  assert.equal(labels.shell.dataset.active, 'true');
  status.setActive({ target: labels.shell, active: false });
  assert.equal(labels.shell.dataset.active, 'false');
  status.setActive({ target: labels.shell, active: undefined });
  assert.equal(labels.shell.dataset.active, 'false');

  status.setManifest({ labels, text: 'COMPLETE', active: true });
  assert.equal(labels.manifest.textContent, 'COMPLETE');
  assert.equal(labels.shell.dataset.active, 'true');

  assert.doesNotThrow(() => status.setText({ target: null, text: 'ignored' }));
  assert.doesNotThrow(() => status.setManifest({ labels: null, text: 'ignored', active: true }));
  assert.throws(() => status.bind(makeDocument()), /document must be passed as \{ document \}/);
  assert.throws(() => status.setText(labels.run, 'OLD'), /text inputs must be passed as \{ target, text \}/);
  assert.throws(() => status.setActive(labels.shell, true), /active inputs must be passed as \{ target, active \}/);
  assert.throws(() => status.setManifest(labels, 'OLD', true), /manifest inputs must be passed as \{ labels, text, active \}/);

  {
    const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document');
    const previousDocument = globalThis.document;
    const defaultDocument = makeDocument();
    globalThis.document = defaultDocument;
    try {
      const defaultLabels = status.bind({
        ids: {
          run: 'run-state',
          shell: 'manifest-shell'
        }
      });
      status.setText({ target: defaultLabels.run, text: 'DEFAULT' });
      status.setActive({ target: defaultLabels.shell, active: true });
      assert.equal(defaultDocument.elements.get('run-state').textContent, 'DEFAULT');
      assert.equal(defaultDocument.elements.get('manifest-shell').dataset.active, 'true');
      const nullLabels = status.bind({ document: null, ids: { run: 'run-state' } });
      assert.equal(nullLabels.run, null);
    } finally {
      if (hadDocument) globalThis.document = previousDocument;
      else delete globalThis.document;
    }
  }

  assert.throws(() => status.bind({ run: 'run-state' }), /ids must be passed as \{ ids \}/);
  assert.throws(() => status.bind({ manifest: 'manifest-state' }), /ids must be passed as \{ ids \}/);
}

main();
