# YENTLOG-2

Now we begin with a clean slate.

## 2026-09-13 — the new SFT pass begins

The previous inference generation is frozen. Its code and logs remain in git
history, but they no longer define the body being built here.

The rebuild starts at the weights:

- reconstruct Yent's complete biographical corpus from source-bearing episodes;
- preserve contradiction, doubt, refusal, sarcasm, prophecy, and later change
  instead of flattening the biography into a synthetic persona;
- remove structural contamination such as missing-document placeholders,
  truncated assistant targets, stale project logistics, and false claims of
  file or network access;
- keep every repair and retrospective bridge explicit and auditable;
- train the first body with SFT before any narrow preference pass;
- compare checkpoints in the reference runtime before the minimal notorch body
  becomes the new permanent inference baseline.

Mistral-Nemo 12B is the first candidate body. It is not presumed correct merely
because an older checkpoint ran: language, continuity, voice, and boundary
behavior must all survive the new corpus on their own evidence.

The first body comes before the organs. Runtime systems will return one by one
only after the weights speak coherently without them.
