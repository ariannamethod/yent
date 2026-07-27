const assert = require('node:assert/strict');
const hud = require('./interface_hud.js');

function documentWith(ids) {
  const nodes = Object.fromEntries(ids.map(id => [id, { id, textContent: '' }]));
  return {
    nodes,
    getElementById(id) {
      return nodes[id] || null;
    }
  };
}

function main() {
  {
    const doc = documentWith([
      'hud-tok', 'hud-exp', 'hud-step', 'hud-ent', 'hud-debt', 'hud-cons',
      'hud-field', 'hud-prob', 'hud-rank', 'hud-tail'
    ]);
    const cells = hud.bind(doc);
    hud.render(cells, {
      tokps: 12.34,
      experts: 7,
      step: 42.9,
      entropy: 3.456,
      debt: 0.27,
      consensus: 0.62,
      field: 0.98,
      hasCandidateTelemetry: true,
      selectedProb: 0.12345,
      selectedRank: 3.7,
      candidateTail: 0.4321
    }, {
      tokenTelemetry: { metricProb: value => `p=${value.toFixed(3)}` }
    });
    assert.equal(doc.nodes['hud-tok'].textContent, '12.3');
    assert.equal(doc.nodes['hud-exp'].textContent, '7');
    assert.equal(doc.nodes['hud-step'].textContent, '42');
    assert.equal(doc.nodes['hud-ent'].textContent, '3.46');
    assert.equal(doc.nodes['hud-debt'].textContent, '0.27');
    assert.equal(doc.nodes['hud-cons'].textContent, '0.62');
    assert.equal(doc.nodes['hud-field'].textContent, '0.98');
    assert.equal(doc.nodes['hud-prob'].textContent, 'p=0.123');
    assert.equal(doc.nodes['hud-rank'].textContent, '3');
    assert.equal(doc.nodes['hud-tail'].textContent, '0.43');
  }

  {
    const doc = documentWith(['hud-tok', 'hud-prob', 'hud-rank', 'hud-tail']);
    const cells = hud.bind(doc);
    hud.render(cells, {
      tokps: Number.NaN,
      hasCandidateTelemetry: false,
      selectedProb: 0.9,
      selectedRank: 1,
      candidateTail: 0.2
    });
    assert.equal(doc.nodes['hud-tok'].textContent, '0.0');
    assert.equal(doc.nodes['hud-prob'].textContent, '-');
    assert.equal(doc.nodes['hud-rank'].textContent, '-');
    assert.equal(doc.nodes['hud-tail'].textContent, '-');
  }

  {
    const doc = documentWith(['custom-tok']);
    const cells = hud.bind(doc, { tok: 'custom-tok' });
    hud.render(cells, { tokps: 1.25 });
    assert.equal(doc.nodes['custom-tok'].textContent, '1.3');
  }

  {
    const hadDocument = Object.prototype.hasOwnProperty.call(globalThis, 'document');
    const previousDocument = globalThis.document;
    const doc = documentWith(['custom-tok']);
    globalThis.document = doc;
    try {
      const cells = hud.bind({ tok: 'custom-tok' });
      hud.render(cells, { tokps: 2.25 });
      assert.equal(doc.nodes['custom-tok'].textContent, '2.3');
    } finally {
      if (hadDocument) globalThis.document = previousDocument;
      else delete globalThis.document;
    }
  }

  {
    assert.doesNotThrow(() => hud.render({}, { hasCandidateTelemetry: false }));
    assert.throws(
      () => hud.render(hud.bind(documentWith(['hud-prob'])), { hasCandidateTelemetry: true }),
      /YentTokenTelemetry helper missing/
    );
  }
}

main();
