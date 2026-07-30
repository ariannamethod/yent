# WORLDMODELOG

Yent worldmodel interface log.

## 2026-07-31 - replay fixture boundary

- Fixture selection now names the requested replay case as `{ scenario }`.
- The exported `scenario` helper rejects both bare strings and the generic
  `name` alias.
- Replay URL parsing and playback keep the same behavior after the selected
  fixture name is normalized.

## 2026-07-31 - run prompt input boundary

- Composer submit binding now names the prompt control as `promptInput`.
- The old generic `input` field is rejected at the run helper boundary.
- Page submit behavior stays unchanged.

## 2026-07-31 - canvas resize surface boundary

- Resize no longer accepts split top-level `canvas/context` inputs.
- WORLD now passes its bound field surface through explicit `surface`.
- JANUS keeps the existing explicit `surfaces` path for field plus trace.

## 2026-07-30 - replay scenario boundary

- Replay playback now rejects the old generic `name` alias.
- Scenario selection passes only through explicit `scenario`.
- Demo playback keeps the same selected fixture because the page bridge already
  performs that naming step.

## 2026-07-30 - replay location boundary

- Replay request parsing now rejects the old top-level `search` shortcut.
- Explicit `location` is the only supported injected URL/search source.
- Browser default replay detection still comes from `root.location`.

## 2026-07-30 - visual ids boundary

- HUD and status binding helpers now accept custom element ids only through
  `ids`.
- Direct id aliases on helper options are rejected.
- Default HUD binding and page behavior stay unchanged; status labels now use
  the same named id contract as input controls.

## 2026-07-30 - input control boundary

- Input control binding now accepts custom element ids only through `ids`.
- Direct `composer`, `prompt`, and `send` aliases are rejected.
- Focus checks now name the observed `control`; WORLD keeps the same key
  behavior without a positional helper call.

## 2026-07-30 - session replay mode boundary

- Session adapter replay behavior now uses only explicit `replayMode`.
- The old generic `replay` alias is rejected.
- Restore and replay smoke behavior are unchanged; this only tightens the
  persistence-mode input name.

## 2026-07-30 - restore session boundary

- Restore startup now rejects the old generic `session` alias.
- Receipt restore uses only explicit `sessionReceipt`.
- Replay mode still skips restore work; invalid helper input names no longer
  hide behind that no-op path.

## 2026-07-30 - submit turn boundary

- Submit startup now rejects the old generic `run`, `session`, and `request`
  aliases.
- Turn streaming now rejects the old generic `session` and `request` aliases.
- JANUS and WORLD keep the same stream lifecycle; only the cross-helper input
  names are tightened.

## 2026-07-30 - replay autostart boundary

- Replay autostart now uses explicit `replayRequest`, `promptInput`, and
  `generationRun` inputs.
- Generic `request`, `input`, and `run` aliases are no longer accepted by the
  replay helper.
- JANUS and WORLD keep the same boot order; this only tightens the startup
  handoff surface.

## 2026-07-30 - boot resize boundary

- Boot resize listener binding now uses named
  `{ resizeTarget, resize, listenerOptions }` inputs.
- Optional resize listener options are carried into `addEventListener` without
  reopening page-local window plumbing.
- Boot startup order and replay behavior are unchanged.

## 2026-07-29 - event listener boundary

- Shared browser listener binding now uses named internal inputs instead of a
  positional target/type/handler/options tuple.
- Key and pointer bindings preserve `listenerOptions` through both add and
  remove, keeping cleanup symmetric.
- Page surfaces keep the same `bindKeyState` and `bindPointer` API.

## 2026-07-29 - text tape boundary

- Token tape append now takes named `{ tape, text, limit }` inputs.
- JANUS no longer sends the tape state, token text, and cap as a positional
  triple.
- Pure text cleanup helpers stay stable; this closes only the mutable tape
  append surface.

## 2026-07-29 - chat stream outcome boundary

- Chat stream outcome classification now takes named `{ error, responseText }`
  inputs.
- Turn streaming no longer passes error and accumulated assistant text as a
  positional pair.
- This keeps transport behavior unchanged while making completion/fault
  classification explicit.

## 2026-07-29 - outcome dispatch boundary

- Stream outcome dispatch now takes named `{ submit, handlers }` inputs.
- JANUS and WORLD no longer pass submit receipts and handler tables as
  positional arguments.
- Handler callback signatures stay stable; this closes only the exported
  outcome-helper call boundary.

## 2026-07-29 - session persistence boundary

- Top-level session normalize/load/save now take named persistence objects.
- Storage and message arrays no longer share the same positional call surface.
- Adapter turn methods remain stable for submit/restore; this closes only the
  exported persistence helper boundary.

## 2026-07-29 - animation callback boundary

- Animation scheduling now accepts named `{ callback }` inputs.
- JANUS and WORLD no longer pass bare loop functions into
  `interfaceAnimation`.
- This keeps frame ownership explicit before deeper telemetry-driven rendering
  changes resume.

## 2026-07-29 - status writer boundary

- Status text, activity, and manifest writes now take named objects.
- JANUS run-state and WORLD note/manifest-state updates no longer pass raw
  positional DOM targets into `interfaceStatus`.
- This keeps status mutation aligned with the named output writer contract.

## 2026-07-29 - output writer boundary

- Output text and scroll writes now take named
  `{ target, text, scrollTarget }` inputs.
- JANUS assistant output, WORLD manifest text, and transcript turn rendering
  all use the same explicit writer contract.
- This keeps output binding and output mutation aligned before the interface
  starts carrying richer runtime telemetry.

## 2026-07-29 - composer binding boundary

- The generation-run composer hook now accepts named
  `{ form, input, onSubmit }` inputs.
- Boot remains the single shared place that binds submit events; the page
  surfaces do not regain local generation-run state.
- This closes another positional UI resource path before deeper worldmodel
  wiring resumes.

## 2026-07-29 - HUD render boundary

- HUD rendering no longer accepts positional cells/state arguments.
- JANUS and WORLD pass `{ hud, state, tokenTelemetry }`, keeping selected-token
  metrics explicit without changing their displayed values.
- This matches the named boundary style now used by input, output, transcript,
  replay, and canvas helpers.

## 2026-07-29 - canvas viewport boundary

- Canvas viewport and DPR helpers no longer accept bare viewport/window
  objects.
- Tests name viewport injection as `{ viewport, maxDpr }`; page resize still
  uses the shared helper's default browser state.
- This removes another old positional browser-target seam from the interface
  helpers.

## 2026-07-29 - transcript container boundary

- Transcript rendering no longer accepts a bare container argument.
- JANUS injects the transcript target as `{ container }`, while `clear` follows
  the same named surface.
- This closes the remaining positional DOM target in the transcript helper.

## 2026-07-28 - replay location boundary

- Replay request parsing no longer accepts a bare location string.
- Tests inject replay URLs through named `{ location }` or `{ search }`;
  JANUS and WORLD keep the default browser-location call.
- The replay helper now matches the named interface surface used by output,
  input, and visual binding helpers.

## 2026-07-28 - output target boundary

- Output helper binding no longer accepts a bare string target id.
- JANUS and WORLD bind transcript/manifest output through named `{ id }`
  objects, matching the rest of the interface helper surface.
- Explicit DOM injection for tests remains named as `{ document, id }`.

## 2026-07-28 - input document boundary

- Request-side input helpers no longer accept positional document arguments.
- `bindControls`, `readParams`, and focus checks use named document objects in
  tests while JANUS and WORLD keep default browser calls.
- Turn orchestration now asks `interfaceInput` for request params through the
  named `{ document: paramsDocument }` seam.

## 2026-07-28 - visual binding document boundary

- Output, HUD, and status helpers no longer accept positional document
  arguments.
- Explicit document injection is named (`{ document, id }` or
  `{ document, ids }`), while JANUS and WORLD still use helper-owned browser
  defaults.

## 2026-07-28 - page global boundary

- JANUS and WORLD now enter the shared dependency loader through
  `globalThis.YentInterfaceDeps`.
- Page scripts no longer touch the browser-specific `window` global directly;
  browser state remains behind interface helpers.

## 2026-07-28 - request params document boundary

- Submit and turn orchestration no longer expose generic `options.document`.
- Request-control DOM reads use `interfaceInput.readParams()` by default, with
  a named `paramsDocument` seam only for tests.
- JANUS and WORLD keep page startup free of browser document plumbing while the
  shared input helper owns the live default.

## 2026-07-28 - document default boundary

- Shared interface helpers now own browser `document` defaults for page-level
  startup lookup.
- JANUS and WORLD no longer pass browser `document` into shared helpers or the
  submit/focus path.
- Test-only document injection remains available inside helpers, but page
  scripts no longer carry browser document plumbing.

## 2026-07-27 - window alias boundary

- Removed helper-level `options.window` aliases from boot, event, and canvas
  APIs.
- Test-only injection now names the semantic resource: `resizeTarget` or
  `viewport`.
- Browser defaults stay helper-owned without keeping page-window aliases alive.

## 2026-07-27 - boot target boundary

- `interfaceBoot.start()` now owns default browser resize target lookup.
- JANUS and WORLD boot without passing `window` through page code.
- Page scripts still own resize effects, generation callbacks, and visual
  startup consequences.

## 2026-07-27 - event target boundary

- `interfaceEvents.bindKeyState()` and `bindPointer()` now own default browser
  event target lookup.
- JANUS and WORLD bind input events without passing `window` through page code.
- Page scripts no longer own event target injection.

## 2026-07-27 - canvas viewport boundary

- `interfaceCanvas.resize()` now owns default browser `window` viewport lookup.
- JANUS and WORLD resize canvases without passing `window` through page code.
- Page scripts no longer own resize viewport injection.

## 2026-07-27 - clock default boundary

- `interfaceClock.create()` now owns default browser `performance` lookup.
- JANUS and WORLD create clocks without passing browser performance through page
  code.
- Page scripts no longer own clock source injection.

## 2026-07-27 - browser defaults boundary

- `interfaceReplay.request()` now owns default browser location lookup.
- JANUS and WORLD call replay request parsing without passing `window.location`
  through page code.
- `interfaceStyle.create()` now owns default browser style lookup.
- Page scripts no longer own replay URL parsing inputs or pass
  `document/getComputedStyle` into font resolution.

## 2026-07-26 - input focus boundary

- Added `interfaceInput.isFocused()` to the shared browser input helper.
- WORLD now uses the helper for prompt-focus movement gating instead of reading
  `document.activeElement` in page code.
- Page scripts no longer own active-element checks.

## 2026-07-26 - frame clock boundary

- Added `clock.now()` to the shared browser clock helper.
- WORLD now uses the injected clock for its animation frame baseline and
  timestamp fallback, so replay frames without browser timestamps still advance
  through the same timing boundary.
- Page scripts no longer call `performance.now()` directly.

## 2026-07-26 - output container boundary

- Added helper-owned output container lookup to `interface_output.js`.
- JANUS binds its transcript container through the shared output helper; WORLD
  binds its manifest text container through the same boundary.
- Page scripts no longer call `document.getElementById()` directly.

## 2026-07-26 - animation frame boundary

- Added `interface_animation.js` for shared browser animation frame scheduling.
- JANUS and WORLD now request frames through one helper while preserving their
  separate render loops, clocks, and visual physics.
- The replay smoke path loads the same helper, so deterministic UI smoke covers
  the browser scheduling boundary.

## 2026-07-26 - form control boundary

- Added helper-owned `prompt`, `composer`, and `send` lookup to
  `interface_input.js`.
- JANUS and WORLD now share the same browser control binding before submit,
  request parameter parsing, and stream selection.
- Page scripts keep their own transcript/manifest containers, but shared form
  controls no longer live as local DOM wiring.

## 2026-07-26 - canvas binding boundary

- Added helper-owned canvas element lookup and context creation to
  `interface_canvas.js`.
- JANUS and WORLD now bind their visible canvases through the same browser
  boundary that already owns resize and scratch-surface allocation.
- Page scripts keep rendering physics, but no longer open canvas contexts
  directly.

## 2026-07-26 - scratch canvas boundary

- Added helper-owned scratch canvas creation to `interface_canvas.js`.
- JANUS mask rendering still stays JANUS-specific, but the browser canvas
  allocation and context contract now live beside shared viewport resizing.
- Page scripts are no longer allowed to allocate scratch canvases directly.

## 2026-07-26 - session storage boundary

- `/yent` and `/worldmodel` no longer pass `sessionStorage` directly.
- `interface_session.js` owns the browser default storage lookup; pages pass
  only replay state to the shared receipt adapter.
- The contract now rejects future page-local session storage wiring.

## 2026-07-26 - shared transcript turn boundary

- Added `interface_transcript.js` for JANUS transcript turn-card rendering.
- Role labels, text body creation, append, restore clear, and transcript scroll
  now pass through one optional browser helper.
- WORLD does not load this helper in HTML; the dependency remains page-specific
  while replay smoke can still exercise the full helper set.

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
- Default browser event target lookup now lives in the helper as well.
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
- Default browser viewport lookup now lives in the helper as well.

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
