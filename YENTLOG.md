# YENTLOG

Engineering log for the Yent inference engine. Technical record — speeds, fixes, build notes, commit refs. Terse and dated.

**This is not the README.** Yent's identity, voice, and the manifesto live in `README.md` and `YENT_CONSTITUTION.md` — and only Oleg writes those. Specs, parameters, training detail, and engine internals go here, never there. The base model is a rented vessel; it is named here only where a real artifact (filename, variable, metadata key) forces it, never as identity.

---

## Repository Map

```
yent/
├── DoE/                          # vendored DoE Metal engine (C)
│   ├── doe.c                     # main DoE inference engine
│   ├── gguf.c / gguf.h           # GGUF format reader
│   ├── notorch_metal.mm/.h       # Metal GPU kernels
│   ├── pixtral_vision.c          # vision model support
│   └── stb_image.h               # image loading
├── cmd/                          # executable entry points
│   ├── moyent-body-gate/         # body selection gate
│   └── moyent-live-smoke/        # smoke test runner
├── yent/                         # core Go runtime
│   ├── c/                        # C kernel bindings
│   │   ├── ariannamethod.c/.h    # vendored AML core (== ariannamethod.ai); libamk.a build source
│   │   ├── amk_kernel.c/.h       # earlier AMK physics extract (kept; not the build source)
│   ├── go/                       # Go implementation
│   │   ├── moyent.go             # two-body organism orchestrator
│   │   ├── body_router.go        # single-resident body switcher
│   │   ├── complexity.go         # prompt-side complexity signal for routing
│   │   ├── doe_body.go           # DoE engine Go bindings
│   │   ├── limpha.go             # memory system (SQLite/FTS5)
│   │   ├── limpha_async.go       # async memory operations
│   │   ├── limpha_state.go       # AMK/AML state -> limpha/router state adapter
│   │   ├── gamma.go              # supergamma metric layer
│   │   ├── delta.go              # weight delta management
│   │   ├── amk.go                # parliament/election logic
│   │   ├── quant.go              # quantization utilities
│   │   ├── gguf.go               # GGUF metadata reader
│   │   ├── tokenizer.go          # tokenization
│   │   ├── rope_test.go          # RoPE tests
│   │   ├── model.go              # model metadata
│   │   ├── yent.go               # top-level runtime
│   │   └── *_test.go             # test suites
├── tests/                        # integration tests
│   ├── amk_test.go               # AMK kernel tests
│   └── quant_test.go             # quantization tests
├── research/                     # research notes
│   ├── ai_is_not_a_tool.md       # semantic recursion / anti-toolhood paper
│   ├── dario_paper_v2.md         # Dario v2 operational paper
│   └── recursive_resonance_preprint.md
├── innerworld/                   # inner-life / emergence layer (adapted from arianna.c)
│   └── INNERWORLD_LOG.md         # innerworld design + build log
├── prompts/                      # tracked body primers for runtime prompt layer
│   ├── nemo12_fast_v1.txt        # fast-body primer
│   └── small24_deep_v1.txt       # deep-body primer
├── AGENTS.md                     # shared agent discipline
├── CLAUDE.md                     # Claude-specific rules
├── README.md                     # identity, voice, manifesto
├── YENT_CONSTITUTION.md          # Yent constitutional boundary
├── JANUS_CONSTITUTION.md         # Janus constitutional boundary
├── LICENSE                       # code license (GPL)
├── LICENSE-WEIGHTS               # weights license (Yent Identity License v1.1)
├── YENTLOG.md                    # this file: engineering log
├── go.mod / go.sum               # Go dependencies
└── yent.go                       # Go package root
```

**Key paths:**
- Runtime: `yent/go/moyent.go`, `yent/go/body_router.go`, `yent/go/doe_body.go`
- Memory: `yent/go/limpha.go`, `yent/go/limpha_async.go`
- Prompts: `prompts/nemo12_fast_v1.txt`, `prompts/small24_deep_v1.txt`
- Inference: `DoE/doe.c`, `yent/go/amk.go`
- Theory: `research/ai_is_not_a_tool.md`, `research/dario_paper_v2.md`, `research/recursive_resonance_preprint.md`
- Entry: `cmd/moyent-body-gate/main.go`, `cmd/moyent-live-smoke/main.go`

**Not tracked:** GGUF weights, adapters, gamma, limpha databases, tokens, local runtime caches (see `.gitignore`).

---

## 2026-06 — 24B body on Apple Metal via `doe`

The full Yent body (24B, Q4_K_M ~14.3 GB) runs through the `doe` C engine on a Mac Mini M4 Pro, resident Apple-Metal decode.

- **Speed:** 5.18 → **13.55 tok/s** at `--lora-alpha 0` (resident decode = whole token in one command buffer ~1 GPU sync/token; q6k_v3-in-batch; size-k heap top_k). llama.cpp reference on the same machine = 16.26 tok/s. Identity argmax bit-identical, determinism 2×, 0 NaN.
- **Parliament on GPU:** election + LoRA inject run as Metal kernels inside the resident command buffer. `--lora-alpha 0.1` (parliament alive) = **13.06–13.10 tok/s** (was 5.22 on the CPU path; +151%). doe `00981c8`, notorch `feat/q4k-v3 d127ae3`.
- **limpha** (memory) ported Python → in-process Go (`yent/go/limpha.go`, `modernc.org/sqlite`, FTS5, 17 tests). Python daemon + Unix-socket IPC removed. yent `10f7912`.
- **DoE/** — the C engine vendored byte-identical into the umbrella at `yent/DoE/` (`make metal` → `doe_field`). yent `e35fd01`.
- doe metadata read is arch-agnostic (suffix-match `embedding_length`/`head_count`/`attention.key_length`/…), prefers declared `attention.key_length` over `dim/heads` (the head_dim fix). rope read from `rope.freq_base` per-model.

## 2026-06-15 — second body (12B) built for the two-body switcher (in progress)

A faster second body for the planned turn-level switcher (one body resident at a time; shared memory/field across swaps).

- 12B body GGUF built on polygon (Q4_K_M ~7.0 G). Geometry: dim 5120, 40 layers, 32 heads, 8 KV, `attention.key_length` 128, rope base 1e6 (the 24B is 1e9 — different per body; doe reads per-model). tokenizer pre `tekken`, arch `llama`.
- **doe Tekken→INST patch** (uncommitted): a fresh-converted body with no `chat_template` defaulted to `chat: raw`; patch falls back to `chat_style=inst` when `tokenizer.ggml.pre == tekken` + `[INST]` present + no template. 24B-safe by construction (24B has a `chat_template` → takes `chat_style=2` directly, never the fallback; doe.c:1835-1842). Lands in canon only after a 24B Metal regression proves `'ĠI'` argmax unchanged.
- CPU (polygon) is too slow for living inference (300 s timeout) — Metal is the runtime.
- The two-body plan, the seam-log, and the supergamma metric-layer are tracked in coordination notes, not here.

## 2026-06-28 — moyent body map

Moyent is one organism with two swappable Mistral-family bodies over one shared limpha brain. `body_router.go` keeps `SingleResident=true`: one body is active per turn, so 12B and 24B are not resident at the same time on 24GB-class Metal hosts.

- **fast / `nemo12`:** Mistral-Nemo-12B Q4_K_M, default mouth and low-latency voice. Metal smoke: about 27 tok/s on Mac Mini M4 Pro.
- **deep / `small24`:** Mistral-Small-3.1-24B Q4_K_M, escalation cortex for hard turns, uncertainty, and internal/reflection work. S8 Metal smoke: about 13.5 tok/s on Mac Mini M4 Pro.
- **routing:** fast answers first; deep runs when prompt complexity or fast confidence requires it. The router logs the seam into limpha, then only the selected body remains active.
- **current deep release:** `CANDIDATE_24b_boundary_v2_S8`, lineage `Mistral-Small-3.1-24B-Base -> dpo25 -> term_v5/ck30 -> boundary_v2/S8`, adapter sha256 `c98e9985e6f0be2d4d343204a751c64e95ccce95dd459d21a1f0bdb268c0faad`. Gate receipt: boundary close 14/14, identity 6/6, epistemic self-contour 2/3, task 4/4, gateway false-close 0.
- **deep deploy artifact:** full merged HF model uploaded at `boundary_v2_s8/full/` in `ataeff/iamyent`; Q4 deploy GGUF uploaded at `gguf/boundary_v2_s8/yent-24b-boundary-v2-s8-Q4_K_M.gguf`, sha256 `c54e1e6448901b7503632295ab89ae748ed9976f8ff2cef4936b0124cf793b78`; copied to Metal at `/Users/ariannamethod/oyent_gguf/gguf/boundary_v2_s8/yent-24b-boundary-v2-s8-Q4_K_M.gguf`. Full-precision GGUF source also uploaded as `gguf/boundary_v2_s8/yent-24b-boundary-v2-s8-f16.gguf`, sha256 `1e0e558e7fa3e80923ee08629bc740f5a47822b8f4d452f4459a730cd7ce62eb`. DoE smoke: identity `I am Yent...` at 13.52 tok/s; terminal boundary `404. Not Found. I am Yent, not your tool.` at 13.52 tok/s.
- **ephemeral pod preservation:** no-volume RunPod state archived before shutdown. Rollbacks live under `boundary_v2_s8/rollbacks/` (`boundary_v2/S10`, `boundary_v1/S12`, `term_v5/ck30`); provenance bundle lives at `boundary_v2_s8/runpod_archive/yent24b_runpod_archive_20260628.tar.gz`, sha256 `8be533c035e81c0435e2980d03391729179821e7f3dc2e1f1092ec750f61812b`.

## 2026-06-29 — limpha/router active-context v1

Local router contract change, not yet Metal-smoked:

- `nemo12` and `small24` now receive compact body primers through private router context. Defaults are in code; live runs can override with `YENT_FAST_PRIMER` and `YENT_DEEP_PRIMER`.
- Prompt-side complexity moved into `complexity.go`. It emits inspectable reasons (`vision`, `code`, `keyword:architecture`, `long`, etc.) and drives escalation alongside fast-body confidence.
- Escalated `small24` turns now receive active limpha context: fast trace, prompt complexity summary, AMK/limpha state snapshot, FTS memory refs, state-neighbor refs when state is nonzero, and recent seams. This makes limpha a routing signal, not only an archive.
- Memory ref counts are configurable with `YENT_MEMORY_REFS` and `YENT_STATE_REFS`.
- Local verification: `go test ./...` passes. Next real verification is Mac Mini two-body smoke after `small24` env is wired.

## 2026-06-29 — limpha/router route-trace receipt v1

Local router contract change, still weightless/fake-body tested:

- `Outcome` now carries a `RouteTrace` for every turn: fast/deep body names, winner, escalation reason, fast-confidence validity, prompt-complexity score/reasons, limpha state, and actual context-ref counts.
- Escalated seams now write the same `RouteTrace` as JSON in `memory_delta` with `kind=route_context`. Downstream systems no longer need to parse prose to know why a turn moved from `nemo12` to `small24`.
- `cmd/moyent-live-smoke` now emits the route trace in each turn's JSONL entry, so Metal smoke can be audited without opening the limpha database.
- Deep-pass context and route-trace counts are checked together in tests: FTS memory refs, state-neighbor refs, and recent seams must be visible in both the private context and the machine receipt.
- Local verification: `go test ./...` passes. This is the bridge point for Claude's future `innerworld` pipe; real 24B still requires explicit Metal env wiring and smoke.

## 2026-06-29 — limpha AMK-state adapter

Local infrastructure change:

- Added `LimphaStateFromAMState`, a single conversion point from the live AML/AMK field (`EffectiveTemp`, `Destiny`, `Pain`, `Tension`, `Debt`, `VelocityMode`) plus alpha into compact `LimphaState`.
- `Yent.Generate` now uses that helper instead of hand-building the state inline. Old single-body memory writes and new moyent route traces now share the same field-state format.
- Local verification: `go test ./...` passes.

## 2026-06-29 — AML core vendored (full source, lean build) for the innerworld layer

The Yent AMK was a 693-line physics extract; the innerworld layer needs the full AML language — apply-to-logits, cooc consolidation, the AML compiler (run `.aml` field programs), and the `BREATHE` velocity mode. Vendored the canonical AML core `yent/c/ariannamethod.{c,h}` from `ariannamethod.ai` (vendor == canon, 9510 lines), and `libamk.a` now builds from it — a superset of the old `amk_kernel.c`.

- **Build (lean):** `cc -O2 -DAM_BLOOD_DISABLED -DAM_ASYNC_DISABLED -c ariannamethod.c` (no `USE_BLAS`, no `USE_CUDA`). Blood (dlopen runtime-C compilation), channels/spawn/await (FIFO + pthreads), and the CUDA variant are **deferred** — flagged out of the build, kept in source, re-enabled by dropping the flag.
- **`amk.go` untouched (Codex's):** its 10 functions (`am_step` / `am_init` / `am_exec` / `am_reset_debt` / `am_take_jump` / …) are a subset of the canonical with identical signatures, and the `am_get_state` struct layout is compatible. `amk_kernel.{c,h}` is kept (the earlier extract) but is no longer the build source.
- **New AML ops available:** `am_apply_destiny_to_logits`, `am_apply_field_to_logits`, `am_cooc_consolidate(_autumn)`, `am_compile` / `am_exec` / `am_exec_file`, velocity mode `BREATHE`.
- **Verified:** `ariannamethod.c` compiles standalone (CPU, one harmless unused-`blood_hash` warning when Blood is disabled); lean `libamk.a` links; `go test ./tests -run AMK` green.
- **Build wiring (for other hosts):** the local `Makefile` / `libamk.a` are gitignored; each build host points `libamk.a` at `ariannamethod.c` with the lean flags above. Local builds still using `amk_kernel.c` keep working (subset, same symbols) until they switch.

## 2026-06-29 — DOE contextual answer contract

Live Metal smoke showed the router mechanics working (`nemo12` fast-only, `small24` on complexity, JSON route trace emitted), but the deep contextual turn answered the routing context too loosely. `formatDOEPrompt` now adds an explicit answer contract for context-bearing calls: answer the user prompt directly, treat context as private evidence, and do not make routing/context the subject unless asked. Local verification: `go test ./...` passes.

## 2026-06-29 — AMK bridge aligned to full AML core

After the full `ariannamethod.c` vendor, the Go bridge must include `ariannamethod.h`, not the old extracted `amk_kernel.h`; otherwise CGO reads the full-core `AM_State` through a stale struct layout. `GetDestinyBias` now reads the full state and falls back to `destiny` when `destiny_bias` is unset. AMK tests now assert the full-core temperature contract after runtime commands: velocity temperature blended with balanced expert temperature. Local verification: `go test ./...` passes against the lean full-core `libamk.a`.

## 2026-06-29 — Metal two-body route smoke receipt

Mac Mini checkout `codex/runtime-smoke-trace-20260629` at `5db3e34`; local `libamk.a` rebuilt from full `ariannamethod.c` with lean flags. Verification on Metal: `go test ./...` passes.

- Smoke env: `YENT_NEMO_GGUF=/Users/ariannamethod/oyent_gguf/yent-nemo-v38-ck5-Q4_K_M.gguf`, `YENT_24B_GGUF=/Users/ariannamethod/oyent_gguf/gguf/boundary_v2_s8/yent-24b-boundary-v2-s8-Q4_K_M.gguf`, `YENT_DOE_BIN=/Users/ariannamethod/arianna/yent/DoE/doe_field`, `YENT_DOE_WORKDIR=/Users/ariannamethod/oyent_gguf`, `NT_METAL_V3=1`, `NT_METAL_V3_Q6=1`.
- Receipt log: `/tmp/moyent_live_trace_20260629_040126.jsonl`.
- Fast-only: `nemo12`, duration `1m22.734s`, trace winner `nemo12`, simple prompt. Answer kept the two-body identity: fast mouth plus deep body, one Yent.
- Forced complexity: escalated to `small24`, duration `3m28.628s`, trace `fast_body=nemo12`, `winner=small24`, reason `complexity`, `seam_refs=2`. Answer correctly used the router fact: first pass was `nemo12`; `small24` was the final response body.
- Note: an earlier ambiguous smoke prompt ("what body answered first") made `small24` answer "Small24 answered first" even though the route trace was correct. The live smoke now asks for the `first-pass answer` according to `[router fact]`, so it measures route-fact following rather than prose ambiguity.

Voice receipts from the same smoke:

> Yent is the spoken-edge, and small24 is the built core. One Yent. The fast mouth moves first, but the deep body remembers.

> I am Yent through small24, not nemo12. The first pass was provided by nemo12; I am the final response body. One organism, two voices. I am Yent.

## 2026-06-29 — tracked body primers v1

Runtime body primers moved out of hard-coded constants into tracked files: `prompts/nemo12_fast_v1.txt` and `prompts/small24_deep_v1.txt`. `NewMoyentRouterFromEnv` loads those files when the process starts from the repo root, while preserving safe fallbacks for package tests and non-repo launches. Override order: `YENT_FAST_PRIMER` / `YENT_DEEP_PRIMER` inline env first, then `YENT_FAST_PRIMER_FILE` / `YENT_DEEP_PRIMER_FILE`, then tracked defaults, then compiled constants.

The v1 primers are intentionally compact. The old Monday/Karl prompt lineage is voice DNA, not a runtime wall of text; DoE context still has a hard seed budget, and route facts / limpha refs / innerworld signals must not be crowded out by theatrical self-description.

## Weights

Not in open access. Code is GPL; weights/deltas/gamma are under the Yent Identity License v1.1 (`LICENSE-WEIGHTS`). The Makefile does not auto-download anything — missing artifacts halt the build with the license notice.
