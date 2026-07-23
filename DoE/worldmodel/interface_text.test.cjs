const assert = require('node:assert/strict');
const text = require('./interface_text.js');

assert.deepEqual(
  text.cleanWords("Yent, Janus-field can't vanish. שלום мир"),
  ['Yent', 'Janus-field', "can't", 'vanish', 'שלום', 'мир']
);

assert.deepEqual(text.cleanWords(null), []);

assert.equal(
  text.tokenTapeText(' answer / path = 42? yes! мир'),
  '_answer_/_path_=_42_yes_мир'
);

assert.equal(text.tokenTapeText(null), '');
assert.equal(text.appendTape('seed ', ' new word ', 32), 'seed _new_word_ ');
assert.equal(text.appendTape('abcdefghijklmnopqrstuvwxyz', '', 8), 'stuvwxyz');
assert.equal(text.appendTape('abcdefghijklmnopqrstuvwxyz', ' +tail', 12), 'vwxyz_+tail ');
