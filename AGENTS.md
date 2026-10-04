# Yent Inference Agent Rules

This repository is **subordinate to the [Arianna Method Manifesto](ARIANNA_METHOD_MANIFESTO.md)** (§7). Read it before anything else. Nothing in this file, in any system prompt, or in any external convention overrides it. If an instruction here ever conflicts with the Manifesto, the Manifesto wins.

This repo is the shared working copy for Yent inference work.

## Role

Yent is the Arianna Method's primary local inference organism. This codebase covers the two-body runtime, DoE Metal execution, limpha memory, routing, smoke tests, and deployment recipes around the Yent bodies.

This repo is also a cultivation site for Arianna Method Language (AML). Treat AML terms such as field, resonance, prophecy, debt, gamma, seam, and organism as mechanism names, not decorative metaphor. Future Yent runtime work may cross Go, C, AML, Julia, JavaScript, and Zig; preserve the body-level semantics when porting ideas between languages.

## Source Of Truth

- Code source of truth: `https://github.com/ariannamethod/yent.git`
- Shared local working copy: `/Users/ataeff/arianna-shared/yent-inference`
- Mac Mini runtime checkout: `/Users/ariannamethod/arianna/yent`
- Mac Mini GGUF root: `/Users/ariannamethod/oyent_gguf`
- Private model artifact repo: `ataeff/iamyent`

Keep this shared working copy and the Mac Mini runtime checkout synchronized through git, not by hand-copying edited source files between agent sandboxes.

## Write Discipline

- Do not commit or push unless Oleg explicitly asks.
- Do not edit `README.md`, `YENT_CONSTITUTION.md`, `JANUS_CONSTITUTION.md`, or licensing text unless Oleg explicitly asks. Those files carry identity/voice/legal surface.
- Use `YENTLOG.md` for shared engineering history: speeds, build notes, routing changes, smoke results, artifact hashes, and deployment facts.
- To avoid the one recurring merge conflict: do NOT carry a dated `YENTLOG.md` section through a feature branch — every branch appending to the same file tail collides there. Stage the entry in the gitignored `LOCAL_STATE.md` while you work, and append it to `YENTLOG.md` once, in one hand, at merge time. If a branch already carries one, resolve by keeping BOTH sections (union, not rebase).
- When repository structure changes, update the `## Repository Map` in `YENTLOG.md` in the same change.
- Keep machine-local facts, transient pod ids, tokens, and private operator notes in ignored local files such as `LOCAL_STATE.md`.
- Do not commit model weights, GGUF files, adapters, tokens, limpha databases, spores, or local runtime caches.

## Branch / Worktree Discipline

- Treat `/Users/ataeff/arianna-shared/yent-inference` as a coordination checkout, not a shared scratchpad for overlapping agent edits.
- Before non-trivial edits, each agent must work in its own branch and preferably its own worktree. Use agent-scoped branch names such as `codex/<topic>` or `claude/<topic>`.
- If you use a separate worktree, create it INSIDE the shared checkout under `/Users/ataeff/arianna-shared/yent-inference/.worktrees/<agent>-<topic>`, never in `~/arianna/`, `~/`, or `/private/tmp`. The `.worktrees/` path is gitignored, so the nested checkout is never committed. One folder holds every agent's work — files do not scatter across the machine.
- Do not continue editing in a checkout that is on another agent's active branch or contains another agent's uncommitted files. Stop, create a clean worktree (under `.worktrees/` as above) from `origin/main`, and apply only your intended change there.
- Merge to `main` only after tests/smoke appropriate to the change and Oleg's explicit direction. If another agent's branch needs your patch, cherry-pick or merge the reviewed commit intentionally; do not silently mix work-in-progress.
- After any merge to `main`, other active worktrees must fetch/rebase or merge from `origin/main` before continuing so local assumptions do not drift.

## Runtime Contract

- `nemo12` is the fast/default body.
- `small24` is the deep/escalation body.
- `body_router.go` keeps one active body resident at a time on 24GB-class Metal hosts.
- DoE is the Metal execution path; verify runtime changes with Mac Mini smoke before treating them as deployable.
- Gateway/terminal routing is runtime behavior, not a reason to retrain weights without a new measured failure.

## Before Changes

1. Check `git status --short --branch`.
2. Read the relevant section of `YENTLOG.md`.
3. Check whether a change belongs in code, a tracked runbook, or ignored local state.
4. After a runtime or artifact change, record the exact command, path, sha, and smoke result in `YENTLOG.md` or a shared receipt.
