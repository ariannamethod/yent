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
  const labels = status.bind(document, {
    run: 'run-state',
    note: 'status-note',
    manifest: 'manifest-state',
    shell: 'manifest-shell'
  });

  status.setText(labels.run, 'GENERATING');
  assert.equal(labels.run.textContent, 'GENERATING');
  status.setText(labels.note, null);
  assert.equal(labels.note.textContent, '');

  status.setActive(labels.shell, true);
  assert.equal(labels.shell.dataset.active, 'true');
  status.setActive(labels.shell, false);
  assert.equal(labels.shell.dataset.active, 'false');
  status.setActive(labels.shell, undefined);
  assert.equal(labels.shell.dataset.active, 'false');

  status.setManifest(labels, 'COMPLETE', true);
  assert.equal(labels.manifest.textContent, 'COMPLETE');
  assert.equal(labels.shell.dataset.active, 'true');

  assert.doesNotThrow(() => status.setText(null, 'ignored'));
  assert.doesNotThrow(() => status.setManifest(null, 'ignored', true));
}

main();
