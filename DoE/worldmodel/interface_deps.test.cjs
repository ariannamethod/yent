const assert = require('node:assert/strict');
const deps = require('./interface_deps.js');

const commonGlobals = [
  'YentInterfaceSession',
  'YentEventStream',
  'YentChatStream',
  'YentInterfaceText',
  'YentTokenTelemetry',
  'YentInterfaceHud',
  'YentInterfaceReplay',
  'YentInterfaceInput',
  'YentInterfaceTurn',
  'YentInterfaceSubmit',
  'YentInterfaceOutcome',
  'YentInterfaceRun',
  'YentInterfaceBoot',
  'YentInterfaceMath'
];

function makeRoot() {
  const root = {};
  for (const name of commonGlobals) root[name] = { name };
  root.YentWorldmodelGeometry = { name: 'YentWorldmodelGeometry' };
  return root;
}

function main() {
  {
    const root = makeRoot();
    const loaded = deps.load({ root, worldGeometry: true });
    assert.equal(loaded.interfaceSession, root.YentInterfaceSession);
    assert.equal(loaded.eventStream, root.YentEventStream);
    assert.equal(loaded.chatStream, root.YentChatStream);
    assert.equal(loaded.interfaceText, root.YentInterfaceText);
    assert.equal(loaded.tokenTelemetry, root.YentTokenTelemetry);
    assert.equal(loaded.interfaceHud, root.YentInterfaceHud);
    assert.equal(loaded.interfaceReplay, root.YentInterfaceReplay);
    assert.equal(loaded.interfaceInput, root.YentInterfaceInput);
    assert.equal(loaded.interfaceTurn, root.YentInterfaceTurn);
    assert.equal(loaded.interfaceSubmit, root.YentInterfaceSubmit);
    assert.equal(loaded.interfaceOutcome, root.YentInterfaceOutcome);
    assert.equal(loaded.interfaceRun, root.YentInterfaceRun);
    assert.equal(loaded.interfaceBoot, root.YentInterfaceBoot);
    assert.equal(loaded.interfaceMath, root.YentInterfaceMath);
    assert.equal(loaded.worldGeometry, root.YentWorldmodelGeometry);
  }

  assert.throws(() => deps.load({ root: {} }), /YentInterfaceSession helper missing/);

  {
    const root = makeRoot();
    delete root.YentWorldmodelGeometry;
    assert.throws(() => deps.load({ root, worldGeometry: true }), /YentWorldmodelGeometry helper missing/);
  }
}

main();
