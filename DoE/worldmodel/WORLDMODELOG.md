# WORLDMODELOG

Yent worldmodel interface log.

## 2026-07-26 - shared output text boundary

- Added `interface_output.js` for browser output text and scroll writes.
- JANUS transcript text and WORLD manifested answer text now share one DOM
  update contract while their page-specific containers stay local.
- This keeps live generation output plumbing centralized before deeper runtime
  telemetry starts driving visual consequence layers.

## 2026-07-26 - shared status label boundary

- Added `interface_status.js` for browser status label writes.
- JANUS and WORLD still decide their own status language, but text updates and
  WORLD manifest active state now pass through one helper.
- This keeps UI receipt/status plumbing centralized before deeper runtime
  telemetry starts driving both surfaces.

## 2026-07-24 - shared generation clock

- Added `interface_clock.js` for browser generation throughput timing.
- JANUS and WORLD now derive `tok/s` through one reset/tick helper while keeping
  visual frame clocks and motion loops local.
- This is interface plumbing only: token streams, runtime telemetry, and visual
  physics are unchanged.

## 2026-07-24 - shared state baseline

- Added `interface_state.js` for the browser HUD/runtime baseline shared by
  JANUS and WORLD.
- The two surfaces now inherit the same debt, consensus, field, throughput,
  entropy, and candidate-telemetry defaults while keeping page-specific motion
  and topology fields local.

## 2026-07-24 - shared browser style lookup

- Added `interface_style.js` for browser font-family lookup.
- WORLD and JANUS now resolve canvas fonts through the shared dependency gate,
  keeping DOM CSS variable reads out of page render loops.
- This is still UI plumbing only: visual physics, token streams, and runtime
  telemetry are unchanged.

## 2026-07-24 - shared browser event binding

- Added `interface_events.js` for keyboard and pointer event wiring.
- WORLD still owns movement/camera consequences; JANUS still owns mouse
  repulsion and burst physics.
- The shared contract now keeps listener binding out of page scripts before the
  two surfaces receive deeper runtime telemetry.

## 2026-07-24 - shared composer boot binding

- Moved composer submit binding into `interface_boot.js`.
- JANUS and WORLD still provide their own generation callback, but startup now
  owns the form listener lifecycle beside restore, resize, animation, and replay
  autostart.

## 2026-07-24 - shared candidate state bookkeeping

- Added `resetCandidateState` and `applyCandidateState` to
  `token_telemetry.js`.
- JANUS and WORLD now share the selected probability/rank/tail bookkeeping
  contract while preserving separate visual consequences.
- This keeps candidate telemetry truth in one browser helper before future
  real runtime metrics make the two surfaces diverge visually on purpose.

## 2026-07-16

- Started as a clean Apple-style probability field beside `yent.html`.
- Kept `yent.html` as the dark Janus/parliament face.
- `worldmodel.html` is the light internal-space surface: selected answer is the manifested path, surrounding words are candidate mass.
- First prototype uses the existing `/chat/completions` SSE token stream. It does not yet receive real top-k/logprob/expert/innerworld telemetry.
- Removed card-like wall sheets and visible wall outlines; walls are now invisible clipped word surfaces.
- Removed fixed vanishing-point compass and idle manifested-answer text.
- Added vertical movement (`R`/`F`, `PageUp`/`PageDown`) and prompt/token-driven topology seeds so the field changes shape from input before full runtime telemetry exists.
- Next contract: split JS into `worldmodel/*.js` once the DoE static route serves subassets safely.
- Next telemetry: replace synthetic candidate mass with real top-k/logprobs, expert votes, Dario/Janus/innerworld metrics, and rejected-token traces.

## 2026-07-19 - import into Yent

- Imported into Yent's inference tree as the first tracked `/worldmodel` surface.
- Paired with `/yent`, the dark Janus parliament face, while both still use the
  existing `/chat/completions` SSE token stream.
- This stage is static/UI-only: no sampling, prompt, Janus, will, wormhole, or
  runtime telemetry semantics changed.

## 2026-07-19 - script split

- Split inline page scripts into explicit tracked subassets:
  `worldmodel/yent.js` and `worldmodel/worldmodel.js`.
- The DoE server exposes only those exact JavaScript paths, not a broad static
  directory.
- Next boundary is telemetry honesty: define real token/logit/expert/Janus/
  innerworld fields before replacing synthetic topology.

## 2026-07-19 - root entry surfaces

- Moved the HTML entry surfaces to repository root: `yent.html` and
  `worldmodel.html`.
- Kept JavaScript under `DoE/worldmodel/`, served by exact routes only.
- The server resolves root HTML first and keeps an adjacent-layout fallback for
  copied binary bundles.

## 2026-07-19 - runtime token telemetry

- Extended `/chat/completions` SSE token events with real observer metrics:
  `token_id`, `step`, `experts`, `debt`, `prophecy_debt`, `field_health`,
  `consensus`, `entropy`, `resonance`, `emergence`, and `temperature`.
- `worldmodel.html` now uses runtime `step` and `entropy` when present, with the
  old synthetic fallback preserved for older streams.
- This is still token-level observability, not top-k/logprob/rejected-token
  geometry.

## 2026-07-19 - bounded candidate distribution

- Added bounded post-sampler `top_tokens` telemetry to each SSE token event,
  including token id, decoded token text, probability, logprob, and selected
  marker.
- Added selected-token probability/logprob/rank plus `candidate_tail_mass` for
  the probability mass outside the displayed top list.
- `worldmodel.html` now feeds alternative top-token words into the surrounding
  candidate mass while keeping the chosen token as the manifested answer.
- Raw pre-sampler logits, full rejected-token traces, and innerworld event
  geometry remain out of this pass.

## 2026-07-19 - weighted candidate projection

- Promoted non-selected `top_tokens` from plain surrounding words into weighted
  candidate entries with probability, rank, logprob, seed, side, age, and decay.
- `worldmodel.html` renders those entries as a short-lived candidate cloud whose
  size, alpha, wake, and motion come from the bounded post-sampler distribution.
- `/yent` keeps selected output as the readable transcript while feeding
  non-selected candidate token text into a separate latent tape for the torn
  Janus face.
- This remains observational UI physics only: no sampler, prompt, weights,
  wormhole, Janus/will, or raw-logit behavior changed.

## 2026-07-19 - candidate telemetry HUD

- Added `P`, `RANK`, and `TAIL` HUD fields to `/worldmodel` and `/yent`.
- These show selected-token probability, selected rank, and candidate tail mass
  only when the SSE stream provides real bounded candidate telemetry.

## 2026-07-23 - shared assistant stream turn

- Added `interface_turn.js` as the shared live/replay assistant turn boundary.
- The helper owns stream accumulation, receipt preview, outcome classification,
  and assistant commit policy for both JANUS and WORLD.
- Page scripts now keep only page-specific consequence: the Janus transcript and
  face projection, or the worldmodel manifest and field deformation.
- This keeps future interface physics free to diverge without allowing the two
  surfaces to disagree on whether a generated assistant turn actually happened.
- Older or partial streams display `-`, avoiding fake certainty from missing
  fields.

## 2026-07-23 - shared HUD metric renderer

- Added `interface_hud.js` for shared HUD cell binding and metric formatting.
- JANUS and WORLD still decide which metrics matter to their surfaces, but the
  common live stream cells now use one rendering contract.
- Candidate probability display remains tied to `token_telemetry.metricProb`,
  keeping probability/rank/tail semantics aligned across both interfaces.

## 2026-07-23 - shared interface boot order

- Added `interface_boot.js` so both surfaces start in the same order: restore
  tab-local receipt, resize canvas, begin animation, and only then allow replay
  autostart.
- The helper keeps the first-frame strategy page-specific, so JANUS and WORLD
  can preserve their different animation loops without duplicating startup
  semantics.

## 2026-07-23 - shared interface math

- Added `interface_math.js` so both surfaces use the same `clamp` and `mix`
  primitives for bounded visual state and interpolation.
- JANUS and WORLD still own their own geometry, but the low-level numeric
  contract no longer drifts between page scripts.

## 2026-07-23 - shared interface dependencies

- Added `interface_deps.js` so both surfaces load their shared browser modules
  through one explicit dependency boundary.
- WORLD still requests its geometry helper intentionally; JANUS does not inherit
  it by accident.

## 2026-07-23 - shared submit turn bridge

- Added `interface_submit.js` as the common bridge from composer submit into
  user receipt commit and assistant streaming.
- `yent.js` and `worldmodel.js` now call `interfaceSubmit.run(...)`; local code
  remains responsible for visual setup, status labels, and token absorption.
- This keeps the next visual physics work from reopening turn lifecycle
  semantics.

## 2026-07-23 - shared outcome dispatch

- Added `interface_outcome.js` so both surfaces classify settled submit results
  through the same stopped/fault/complete boundary.
- JANUS and WORLD still own their labels and visual effects, but no longer read
  `turn.outcome` directly after submit.

## 2026-07-23 - shared text normalization

- Added `interface_text.js` for Unicode word extraction, token tape sanitizing,
  and bounded tape append behavior.
- WORLD no longer carries its own `cleanWords`; JANUS no longer carries its own
  token tape sanitizer.

## 2026-07-19 - readable manifestation surface

- Added a readable `MANIFEST` answer surface to `/worldmodel`.
- The panel is fed by the same selected SSE token stream as the central canvas
  manifestation, so it exposes the answer without inventing a second text path.
- Candidate clouds, wall words, and Janus/worldmodel physics remain
  observational UI layers only; sampler, prompt, weights, will, wormholes, and
  runtime semantics did not change.

## 2026-07-19 - interface mode switch

- Added a shared `JANUS` / `WORLD` mode switch to `/yent` and `/worldmodel`.
- The switch is plain navigation between the two root HTML surfaces; it does not
  create shared browser state or alter the SSE generation path.

## 2026-07-20 - session handoff between surfaces

- Added a small `sessionStorage` handoff shared by `/yent` and `/worldmodel`.
- The browser tab keeps a bounded recent user/assistant turn list so switching
  interfaces preserves the readable transcript/manifest and seeds the visual
  tape/field from the same selected text.
- Restored handoff turns are display-only. They do not populate the
  `/chat/completions` `messages` request after a view switch.
- This is local UI continuity only, not limpha, model memory, prompt injection,
  sampler state, or a runtime semantic channel.

## 2026-07-20 - shared receipt helper

- Moved the bounded `sessionStorage` normalizer/load/save contract into
  `worldmodel/interface_session.js`.
- `yent.html` and `worldmodel.html` load the helper before their page-specific
  scripts, so JANUS and WORLD cannot drift on receipt shape or message limits.
- The DoE server whitelists `/worldmodel/interface_session.js` explicitly; this
  keeps helper delivery bounded like the two existing page scripts.
- `tests/worldmodel_interface_session_test.go` runs the JS helper test when Node
  is present and also checks script order plus the no-`messages = restored`
  boundary.

## 2026-07-23 - shared restore receipt boundary

- Added `interface_restore.js` so both surfaces restore visible tab-local
  receipt state through one replay-aware loader.
- The helper returns visible messages, combined text, and last assistant turn;
  JANUS and WORLD keep only projection-specific restore effects.

## 2026-07-23 - shared canvas backing boundary

- Added `interface_canvas.js` so JANUS and WORLD share viewport/DPR canvas
  backing-store sizing.
- The helper owns CSS size, backing pixels, and `setTransform`; both surfaces
  keep their own render loops, particles, walls, and camera physics.

## 2026-07-23 - shared resize listener boundary

- Moved browser `resize` listener registration into `interface_boot.js`.
- JANUS and WORLD still provide their own resize effects, but startup now owns
  the lifecycle binding instead of leaving duplicate page-level listeners.

## 2026-07-20 - deterministic replay fixture

- Added `worldmodel/interface_replay.js` as a query-param-only audit fixture for
  both `/yent` and `/worldmodel`.
- `?replay=1` or `?demo=1` plays a bounded deterministic token stream through
  the same page `onToken` handlers used by live generation.
- Replay events include selected probability/rank, candidate tail mass, and
  `top_tokens`, so Janus face and walkable field physics can be checked without
  relying on a live model run.
- Replay mode deliberately skips local interface receipt load/save, keeping
  browser continuity reserved for real user/model turns.
- `interface_page_replay_smoke.test.cjs` executes the actual page scripts in a
  mocked browser surface and confirms that replay reaches COMPLETE without
  using network transport or persisting a receipt.

## 2026-07-21 - run finish token boundary

- Tightened `worldmodel/interface_run.js` so only the currently active run token
  returned by `begin()` can finish the shared generation controller.
- Stale, malformed, missing, or duplicate `finish()` calls now return `false`
  without resetting `running`, `aborter`, or the SEND/STOP button state.
- This protects JANUS and WORLD from future async cleanup drift where an old
  callback could make a live generation look idle.

## 2026-07-21 - shared session receipt adapter

- Added `interfaceSession.createAdapter(...)` to centralize browser receipt
  normalization, replay read-only behavior, and throttled writes.
- Removed page-local receipt wrappers and `lastSessionSaveAt` from `yent.js`
  and `worldmodel.js`.
- The Go interface contract now checks that both surfaces use the adapter and
  do not reintroduce local session receipt state.

## 2026-07-21 - shared turn receipt helpers

- Added adapter-level user turn, partial assistant preview, and final assistant
  commit helpers.
- JANUS and WORLD now update model/visible turn arrays through the same helper
  contract while keeping their own transcript and manifest rendering.
- The contract test rejects page-local `messages.push` /
  `visibleMessages.push` receipt mutations.

## 2026-07-20 - shared event stream parser

- Moved chunked SSE event parsing into `worldmodel/event_stream.js`.
- Both JANUS and WORLD now use `YentEventStream.createParser(...)` instead of
  carrying page-local `parseSseEvents` implementations.
- The shared parser handles chunk boundaries, CRLF frame separators, compact
  `data:{...}` lines, OpenAI-style `[DONE]` sentinels, and malformed frames
  without breaking the UI loop.
- The DoE server whitelists `/worldmodel/event_stream.js` explicitly, and the
  interface contract test checks script order plus removal of local SSE buffers.

## 2026-07-20 - shared chat stream transport

- Moved the browser `/chat/completions` fetch/body/reader/decoder lifecycle into
  `worldmodel/chat_stream.js`.
- JANUS and WORLD now call `YentChatStream.stream(...)` and keep only their
  surface-specific `onToken` effects: transcript/face for JANUS, manifest/field
  for WORLD.
- The helper is loaded after `event_stream.js`, served through an exact DoE
  route, and covered by a Node test plus the shared interface contract test.
- The helper clamps request parameters and turns SSE `error` frames or
  EOF-before-`done` into faults, so incomplete streams cannot look complete.
- The helper also classifies stream outcomes. Page scripts map the shared
  `complete` / `empty` / `stopped` / `fault` result into their own visual labels,
  but no longer decide commit policy independently.

## 2026-07-20 - shared generation run controller

- Moved the browser generation run lifecycle into `worldmodel/interface_run.js`.
- JANUS and WORLD now share the same STOP/SEND button state, duplicate-submit
  rejection, abort path, and final cleanup contract.
- Page scripts keep their own visual/token effects, but no longer carry local
  `running`/`aborter` state or construct `AbortController` directly.
- The DoE server whitelists `/worldmodel/interface_run.js` explicitly, and the
  interface contract test checks load order plus removal of page-local run
  state.

## 2026-07-20 - shared token telemetry contract

- Moved selected-token probability/rank, candidate tail mass, and `top_tokens`
  normalization into `worldmodel/token_telemetry.js`.
- JANUS feeds its latent face tape from the helper's bounded candidate words.
- WORLD feeds its rejected candidate cloud from the same normalized candidate
  words while keeping its own projection/physics.
- Missing telemetry remains honest: old streams still show `-` in candidate HUD
  fields and keep the existing visual fallbacks.
- The DoE server whitelists `/worldmodel/token_telemetry.js` explicitly, and the
  interface contract test now fails if page scripts resume parsing telemetry
  fields locally.

## 2026-07-20 - tested worldmodel geometry state

- Added `worldmodel/worldmodel_geometry.js` for deterministic prompt-derived
  topology and per-token stream-pressure deformation.
- The helper owns text seeds, prompt reset, telemetry pressure, decay, and wall
  shape parameters; `worldmodel.js` owns rendering only.
- Prompt text now sets the initial corridor form, and selected tokens plus
  bounded candidate telemetry bend the walls during generation.
- The DoE server whitelists `/worldmodel/worldmodel_geometry.js` explicitly.
- Node and Go contract tests cover deterministic prompt geometry, telemetry
  deformation, decay, script load order, and removal of page-local topology seed
  helpers.

## 2026-07-21 - shared generation request input

- Added `worldmodel/interface_input.js` for the remaining request-side glue
  shared by JANUS and WORLD.
- The helper reads and clamps `temp` / `max_tokens`, then selects either the
  live chat stream or replay stream from one contract.
- Page scripts still own visual effects and token absorption, but no longer
  duplicate request parameter parsing or live/replay stream construction.
- The DoE server whitelists `/worldmodel/interface_input.js`, and the contract
  tests now fail if either page resumes local request parsing.

## 2026-07-23 - shared replay autostart

- Moved query-param replay startup into `worldmodel/interface_replay.js`.
- JANUS and WORLD still decide their visual token effects locally, but replay
  prompt seeding, delayed start, and the "do not start while already running"
  guard now live in one helper.
- Contract tests now fail if either page recreates `startReplayIfRequested`,
  local replay timers, or direct `generate(replayRequest.prompt)` startup.
