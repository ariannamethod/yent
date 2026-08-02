package tests

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestWorldmodelInterfaceSessionHelper(t *testing.T) {
	if _, err := exec.LookPath("node"); err != nil {
		t.Skipf("node not found: %v", err)
	}
	root := repoRootForTest(t)
	for _, script := range []string{
		filepath.Join(root, "DoE", "worldmodel", "interface_session.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_restore.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "event_stream.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "chat_stream.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_text.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "token_telemetry.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_state.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_clock.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_status.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_output.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_transcript.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_hud.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_replay.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_input.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_events.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_turn.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_submit.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_outcome.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_page_replay_smoke.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_run.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_boot.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_animation.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_math.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_canvas.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_style.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_deps.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "worldmodel_geometry.test.cjs"),
	} {
		cmd := exec.Command("node", script)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s failed: %v\n%s", filepath.Base(script), err, string(out))
		}
	}
}

func TestWorldmodelInterfaceSessionContract(t *testing.T) {
	root := repoRootForTest(t)
	yentHTML := readTextFile(t, filepath.Join(root, "yent.html"))
	worldHTML := readTextFile(t, filepath.Join(root, "worldmodel.html"))
	doeC := readTextFile(t, filepath.Join(root, "DoE", "doe.c"))
	yentJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "yent.js"))
	worldJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "worldmodel.js"))
	chatStreamJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "chat_stream.js"))
	sessionJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_session.js"))
	restoreJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_restore.js"))
	textJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_text.js"))
	tokenTelemetryJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "token_telemetry.js"))
	stateJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_state.js"))
	clockJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_clock.js"))
	statusJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_status.js"))
	outputJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_output.js"))
	hudJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_hud.js"))
	transcriptJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_transcript.js"))
	replayJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_replay.js"))
	eventsJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_events.js"))
	inputJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_input.js"))
	turnJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_turn.js"))
	submitJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_submit.js"))
	outcomeJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_outcome.js"))
	runJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_run.js"))
	bootJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_boot.js"))
	animationJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_animation.js"))
	canvasJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_canvas.js"))
	styleJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_style.js"))
	depsJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_deps.js"))
	readme := readTextFile(t, filepath.Join(root, "README.md"))

	assertScriptOrder(t, "yent.html", yentHTML,
		"/worldmodel/interface_session.js",
		"/worldmodel/interface_restore.js",
		"/worldmodel/event_stream.js",
		"/worldmodel/chat_stream.js",
		"/worldmodel/interface_text.js",
		"/worldmodel/token_telemetry.js",
		"/worldmodel/interface_state.js",
		"/worldmodel/interface_clock.js",
		"/worldmodel/interface_status.js",
		"/worldmodel/interface_output.js",
		"/worldmodel/interface_transcript.js",
		"/worldmodel/interface_hud.js",
		"/worldmodel/interface_replay.js",
		"/worldmodel/interface_input.js",
		"/worldmodel/interface_events.js",
		"/worldmodel/interface_turn.js",
		"/worldmodel/interface_submit.js",
		"/worldmodel/interface_outcome.js",
		"/worldmodel/interface_run.js",
		"/worldmodel/interface_boot.js",
		"/worldmodel/interface_animation.js",
		"/worldmodel/interface_math.js",
		"/worldmodel/interface_canvas.js",
		"/worldmodel/interface_style.js",
		"/worldmodel/interface_deps.js",
		"/worldmodel/yent.js")
	assertScriptOrder(t, "worldmodel.html", worldHTML,
		"/worldmodel/interface_session.js",
		"/worldmodel/interface_restore.js",
		"/worldmodel/event_stream.js",
		"/worldmodel/chat_stream.js",
		"/worldmodel/interface_text.js",
		"/worldmodel/token_telemetry.js",
		"/worldmodel/interface_state.js",
		"/worldmodel/interface_clock.js",
		"/worldmodel/interface_status.js",
		"/worldmodel/interface_output.js",
		"/worldmodel/interface_hud.js",
		"/worldmodel/interface_replay.js",
		"/worldmodel/interface_input.js",
		"/worldmodel/interface_events.js",
		"/worldmodel/interface_turn.js",
		"/worldmodel/interface_submit.js",
		"/worldmodel/interface_outcome.js",
		"/worldmodel/interface_run.js",
		"/worldmodel/interface_boot.js",
		"/worldmodel/interface_animation.js",
		"/worldmodel/interface_math.js",
		"/worldmodel/interface_canvas.js",
		"/worldmodel/interface_style.js",
		"/worldmodel/interface_deps.js",
		"/worldmodel/worldmodel_geometry.js",
		"/worldmodel/worldmodel.js")

	if !strings.Contains(doeC, `"/worldmodel/interface_session.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_session.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_session.js")
	}
	if !strings.Contains(sessionJS, "function normalize(options)") ||
		!strings.Contains(sessionJS, "function load(options)") ||
		!strings.Contains(sessionJS, "function save(options)") ||
		!strings.Contains(sessionJS, "options.messages") ||
		!strings.Contains(sessionJS, "options.storage") {
		t.Fatalf("interface_session.js does not own named session persistence inputs")
	}
	if strings.Contains(sessionJS, "function normalize(source") ||
		strings.Contains(sessionJS, "function load(storage") ||
		strings.Contains(sessionJS, "function save(storage") ||
		strings.Contains(sessionJS, "normalize(parsed && parsed.messages") ||
		strings.Contains(sessionJS, "save(storage, nextMessages") {
		t.Fatalf("interface_session.js still exposes positional session persistence arguments")
	}
	if !strings.Contains(sessionJS, "const replayMode = !!options.replayMode") ||
		strings.Contains(sessionJS, "options.replayMode || options.replay") {
		t.Fatalf("interface_session.js still accepts a generic replay alias")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_restore.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_restore.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_restore.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/event_stream.js"`) ||
		!strings.Contains(doeC, `"worldmodel/event_stream.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist event_stream.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/chat_stream.js"`) ||
		!strings.Contains(doeC, `"worldmodel/chat_stream.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist chat_stream.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_text.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_text.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_text.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/token_telemetry.js"`) ||
		!strings.Contains(doeC, `"worldmodel/token_telemetry.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist token_telemetry.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_state.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_state.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_state.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_clock.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_clock.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_clock.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_status.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_status.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_status.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_output.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_output.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_output.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_transcript.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_transcript.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_transcript.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_hud.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_hud.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_hud.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_replay.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_replay.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_replay.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_input.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_input.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_input.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_events.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_events.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_events.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_turn.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_turn.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_turn.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_submit.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_submit.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_submit.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_outcome.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_outcome.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_outcome.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_run.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_run.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_run.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_boot.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_boot.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_boot.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_animation.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_animation.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_animation.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_math.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_math.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_math.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_canvas.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_canvas.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_canvas.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_style.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_style.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_style.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_deps.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_deps.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_deps.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/worldmodel_geometry.js"`) ||
		!strings.Contains(doeC, `"worldmodel/worldmodel_geometry.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist worldmodel_geometry.js")
	}
	if !strings.Contains(doeC, `strchr(path, '?')`) {
		t.Fatalf("DoE server does not strip query strings before static route matching")
	}

	for _, tc := range []struct {
		name string
		src  string
	}{
		{"yent.js", yentJS},
		{"worldmodel.js", worldJS},
	} {
		if !strings.Contains(tc.src, "globalThis.YentInterfaceDeps") {
			t.Fatalf("%s does not use the shared interface dependency helper", tc.name)
		}
		if strings.Contains(tc.src, "window.") {
			t.Fatalf("%s still reaches through the browser window global", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceDeps.load(") {
			t.Fatalf("%s does not load interface dependencies through the shared helper", tc.name)
		}
		if !strings.Contains(tc.src, "deps.interfaceText") {
			t.Fatalf("%s does not use the shared interface text helper", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceSession.createAdapter") {
			t.Fatalf("%s does not use the shared replay-aware session adapter", tc.name)
		}
		if !strings.Contains(tc.src, "deps.interfaceRestore") ||
			!strings.Contains(tc.src, "interfaceRestore.load(") {
			t.Fatalf("%s does not restore UI receipt state through the shared helper", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceReplay.request()") {
			t.Fatalf("%s does not read replay request through the shared helper default location", tc.name)
		}
		if strings.Contains(tc.src, "window.location") {
			t.Fatalf("%s still reads window.location locally instead of interface_replay", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceSubmit.run(") {
			t.Fatalf("%s does not submit turns through the shared interface submit helper", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceOutcome.handle({") {
			t.Fatalf("%s does not classify stream outcomes through the shared helper", tc.name)
		}
		if strings.Contains(tc.src, "interfaceOutcome.handle(submit") {
			t.Fatalf("%s still exposes positional outcome handling", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceHud.render(") {
			t.Fatalf("%s does not render HUD metrics through the shared helper", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceBoot.start(") {
			t.Fatalf("%s does not start through the shared boot helper", tc.name)
		}
		assertInterfaceBootStartOptions(t, tc.name, tc.src)
		if !strings.Contains(tc.src, "deps.interfaceAnimation") ||
			!strings.Contains(tc.src, "interfaceAnimation.create(") ||
			!strings.Contains(tc.src, "animationFrame.requestFrame({ callback: animate })") ||
			!strings.Contains(tc.src, "animationFrame.start({ callback: animate })") {
			t.Fatalf("%s does not schedule animation frames through interface_animation", tc.name)
		}
		if strings.Contains(tc.src, "animationFrame.requestFrame(animate)") ||
			strings.Contains(tc.src, "animationFrame.start(animate)") {
			t.Fatalf("%s still exposes positional animation callback scheduling", tc.name)
		}
		if strings.Contains(tc.src, "requestAnimationFrame(") {
			t.Fatalf("%s still schedules animation frames locally", tc.name)
		}
		if !strings.Contains(tc.src, "deps.interfaceCanvas") ||
			!strings.Contains(tc.src, "interfaceCanvas.resize(") {
			t.Fatalf("%s does not size canvases through the shared helper", tc.name)
		}
		if strings.Contains(tc.src, "interfaceCanvas.resize({ window") ||
			strings.Contains(tc.src, "interfaceCanvas.resize({\n    window") ||
			strings.Contains(tc.src, "interfaceCanvas.resize({\n  window") {
			t.Fatalf("%s still passes browser window into interface_canvas resize", tc.name)
		}
		if strings.Contains(tc.src, "interfaceCanvas.resize({ canvas") ||
			strings.Contains(tc.src, "interfaceCanvas.resize({\n    canvas") ||
			strings.Contains(tc.src, "interfaceCanvas.resize({\n  canvas") {
			t.Fatalf("%s still passes split canvas/context into interface_canvas resize", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceCanvas.bind(") {
			t.Fatalf("%s does not bind canvas elements through interface_canvas", tc.name)
		}
		if strings.Contains(tc.src, ".getContext(") {
			t.Fatalf("%s still opens canvas contexts locally instead of interface_canvas", tc.name)
		}
		if strings.Contains(tc.src, "document.createElement('canvas')") ||
			strings.Contains(tc.src, `document.createElement("canvas")`) {
			t.Fatalf("%s still creates scratch canvases locally instead of interface_canvas", tc.name)
		}
		if tc.name == "yent.js" && !strings.Contains(tc.src, "interfaceCanvas.createScratch(") {
			t.Fatalf("yent.js does not create its JANUS mask scratch surface through interface_canvas")
		}
		if !strings.Contains(tc.src, "deps.interfaceStyle") ||
			!strings.Contains(tc.src, "interfaceStyle.create()") {
			t.Fatalf("%s does not read browser font/style state through the shared helper", tc.name)
		}
		if strings.Contains(tc.src, "getComputedStyle") {
			t.Fatalf("%s still reads browser style state locally instead of interface_style", tc.name)
		}
		if !strings.Contains(tc.src, "deps.interfaceEvents") {
			t.Fatalf("%s does not load browser input events through the shared helper", tc.name)
		}
		if strings.Contains(tc.src, "interfaceEvents.bindKeyState({\n  window") ||
			strings.Contains(tc.src, "interfaceEvents.bindKeyState({\n    window") ||
			strings.Contains(tc.src, "interfaceEvents.bindPointer({\n  window") ||
			strings.Contains(tc.src, "interfaceEvents.bindPointer({\n    window") {
			t.Fatalf("%s still passes browser window into interface_events", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceInput.bindControls()") {
			t.Fatalf("%s does not bind shared prompt controls through interface_input", tc.name)
		}
		if tc.name == "worldmodel.js" && !strings.Contains(tc.src, "interfaceInput.isFocused({ control: promptInput })") {
			t.Fatalf("worldmodel.js does not test prompt focus through interface_input")
		}
		if strings.Contains(tc.src, "document.activeElement") {
			t.Fatalf("%s still reads activeElement locally instead of interface_input", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceOutput.bind(") {
			t.Fatalf("%s does not bind output containers through interface_output", tc.name)
		}
		if strings.Contains(tc.src, "document.getElementById(") {
			t.Fatalf("%s still binds DOM elements locally instead of helper-owned lookup", tc.name)
		}
		if strings.Contains(tc.src, "messages = restored") {
			t.Fatalf("%s repopulates prompt messages from restored UI receipt", tc.name)
		}
		if strings.Contains(tc.src, "lastSessionSaveAt") ||
			strings.Contains(tc.src, "function saveInterfaceSession") ||
			strings.Contains(tc.src, "function loadInterfaceSession") ||
			strings.Contains(tc.src, "function normalizeSessionMessages") {
			t.Fatalf("%s still carries page-local session receipt state", tc.name)
		}
		if strings.Contains(tc.src, "messages.push") ||
			strings.Contains(tc.src, "visibleMessages.push") {
			t.Fatalf("%s still mutates session turn arrays locally", tc.name)
		}
		if strings.Contains(tc.src, "sessionReceipt.load()") ||
			strings.Contains(tc.src, "visibleMessages.map(msg => msg.content).join(' ')") ||
			strings.Contains(tc.src, `.slice().reverse().find(msg => msg.role === 'assistant')`) {
			t.Fatalf("%s still carries page-local restored receipt derivation", tc.name)
		}
		if strings.Contains(tc.src, "sessionReceipt.previewAssistant(") ||
			strings.Contains(tc.src, "sessionReceipt.commitAssistant(") ||
			strings.Contains(tc.src, "sessionReceipt.commitUser(") ||
			strings.Contains(tc.src, "chatStream.outcome(") ||
			strings.Contains(tc.src, "interfaceTurn.streamAssistant(") ||
			strings.Contains(tc.src, "let fullResponse") {
			t.Fatalf("%s still carries page-local assistant stream turn state", tc.name)
		}
		if strings.Contains(tc.src, "function parseSseEvents") || strings.Contains(tc.src, "sseBuffer") {
			t.Fatalf("%s still carries a page-local SSE parser", tc.name)
		}
		if strings.Contains(tc.src, "fetch('/chat/completions'") ||
			strings.Contains(tc.src, "fetch(\"/chat/completions\"") {
			t.Fatalf("%s still carries a page-local chat/completions transport", tc.name)
		}
		if strings.Contains(tc.src, "err.name === 'AbortError'") ||
			strings.Contains(tc.src, `err.name === "AbortError"`) {
			t.Fatalf("%s still carries page-local stream outcome classification", tc.name)
		}
		if strings.Contains(tc.src, "result.stopped") ||
			strings.Contains(tc.src, "result.fault") ||
			strings.Contains(tc.src, "turn.outcome") {
			t.Fatalf("%s still branches on stream outcome shape locally", tc.name)
		}
		if strings.Contains(tc.src, "let running = false") ||
			strings.Contains(tc.src, "let aborter = null") ||
			strings.Contains(tc.src, "new AbortController()") ||
			strings.Contains(tc.src, "sendButton.textContent =") ||
			strings.Contains(tc.src, "generationRun.bindComposer(") {
			t.Fatalf("%s still carries page-local generation run state", tc.name)
		}
		if strings.Contains(tc.src, "data.top_tokens") ||
			strings.Contains(tc.src, "candidate_tail_mass") ||
			strings.Contains(tc.src, "selected_prob") ||
			strings.Contains(tc.src, "selected_rank") {
			t.Fatalf("%s still parses token telemetry locally", tc.name)
		}
		if !strings.Contains(tc.src, "tokenTelemetry.applyCandidateState(") ||
			!strings.Contains(tc.src, "tokenTelemetry.resetCandidateState(state)") {
			t.Fatalf("%s does not keep candidate telemetry bookkeeping in token_telemetry.js", tc.name)
		}
		if !strings.Contains(tc.src, "deps.interfaceState") ||
			!strings.Contains(tc.src, "interfaceState.create({") {
			t.Fatalf("%s does not initialize shared HUD/runtime defaults through interface_state", tc.name)
		}
		if !strings.Contains(tc.src, "deps.interfaceClock") ||
			!strings.Contains(tc.src, "interfaceClock.create(") ||
			!strings.Contains(tc.src, "tokenClock.reset()") ||
			!strings.Contains(tc.src, "tokenClock.tick()") {
			t.Fatalf("%s does not track generation throughput through interface_clock", tc.name)
		}
		if tc.name == "yent.js" && !strings.Contains(tc.src, "interfaceClock.create()") {
			t.Fatalf("yent.js does not use interface_clock default browser performance lookup")
		}
		if tc.name == "worldmodel.js" &&
			!strings.Contains(tc.src, "interfaceClock.create({ minElapsedSeconds: 0.001 })") {
			t.Fatalf("worldmodel.js does not use interface_clock default performance lookup with its frame minimum")
		}
		if !strings.Contains(tc.src, "deps.interfaceStatus") ||
			!strings.Contains(tc.src, "interfaceStatus.bind(") ||
			!strings.Contains(tc.src, "interfaceStatus.setText({") {
			t.Fatalf("%s does not write status labels through interface_status", tc.name)
		}
		if !strings.Contains(tc.src, "deps.interfaceOutput") {
			t.Fatalf("%s does not write output text through interface_output", tc.name)
		}
		for _, localStatus := range []string{
			"const runState = document.getElementById('run-state')",
			`const runState = document.getElementById("run-state")`,
			"const statusNote = document.getElementById('status-note')",
			`const statusNote = document.getElementById("status-note")`,
			"const manifestState = document.getElementById('manifest-state')",
			`const manifestState = document.getElementById("manifest-state")`,
			"const manifestShell = document.getElementById('manifest-shell')",
			`const manifestShell = document.getElementById("manifest-shell")`,
			"runState.textContent",
			"statusNote.textContent",
			"manifestState.textContent",
			"manifestShell.dataset.active",
		} {
			if strings.Contains(tc.src, localStatus) {
				t.Fatalf("%s still carries page-local status label DOM wiring: %s", tc.name, localStatus)
			}
		}
		for _, localClock := range []string{
			"let tokenCount",
			"let startTime",
			"tokenCount++",
			"tokenCount += 1",
			"state.tokps = tokenCount /",
			"performance.now() - startTime",
			"startTime = performance.now()",
		} {
			if strings.Contains(tc.src, localClock) {
				t.Fatalf("%s still carries page-local generation clock state: %s", tc.name, localClock)
			}
		}
		if strings.Contains(tc.src, "performance.now()") {
			t.Fatalf("%s still reads raw browser time instead of interface_clock", tc.name)
		}
		if strings.Contains(tc.src, "performance") {
			t.Fatalf("%s still passes browser performance locally instead of interface_clock", tc.name)
		}
		for _, localStateDefault := range []string{
			"debt: 0.0",
			"consensus: 0.62",
			"field: 1.0",
			"tokps: 0.0",
			"selectedProb: 0.0",
			"selectedRank: 0",
			"candidateTail: 0.0",
			"hasCandidateTelemetry: false",
		} {
			if strings.Contains(tc.src, localStateDefault) {
				t.Fatalf("%s still carries shared interface state default %s locally", tc.name, localStateDefault)
			}
		}
		if strings.Contains(tc.src, "telemetry.candidateTailMass") ||
			strings.Contains(tc.src, "telemetry.hasSelectedProb ?") ||
			strings.Contains(tc.src, "telemetry.hasSelectedRank ?") {
			t.Fatalf("%s still maintains candidate telemetry state locally", tc.name)
		}
		if strings.Contains(tc.src, ".textContent = state.tokps") ||
			strings.Contains(tc.src, ".textContent = state.debt") ||
			strings.Contains(tc.src, ".textContent = state.consensus") ||
			strings.Contains(tc.src, ".textContent = state.field") ||
			strings.Contains(tc.src, ".textContent = state.hasCandidateTelemetry") {
			t.Fatalf("%s still formats shared HUD metrics locally", tc.name)
		}
		if strings.Contains(tc.src, "parseInt(document.getElementById('max-tokens')") ||
			strings.Contains(tc.src, `parseInt(document.getElementById("max-tokens")`) ||
			strings.Contains(tc.src, "parseFloat(document.getElementById('temp')") ||
			strings.Contains(tc.src, `parseFloat(document.getElementById("temp")`) ||
			strings.Contains(tc.src, "const stream = replayMode") {
			t.Fatalf("%s still carries page-local generation request parsing", tc.name)
		}
		if strings.Contains(tc.src, "sessionStorage") {
			t.Fatalf("%s still passes browser sessionStorage locally instead of interface_session default storage", tc.name)
		}
		if strings.Contains(tc.src, "document") {
			t.Fatalf("%s still passes browser document locally instead of helper-owned document defaults", tc.name)
		}
		for _, localControl := range []string{
			"document.getElementById('prompt')",
			`document.getElementById("prompt")`,
			"document.getElementById('composer')",
			`document.getElementById("composer")`,
			"document.getElementById('send')",
			`document.getElementById("send")`,
		} {
			if strings.Contains(tc.src, localControl) {
				t.Fatalf("%s still binds shared form controls locally: %s", tc.name, localControl)
			}
		}
		if strings.Contains(tc.src, "function startReplayIfRequested") ||
			strings.Contains(tc.src, "interfaceReplay.startIfRequested(") ||
			strings.Contains(tc.src, "setTimeout(() =>") ||
			strings.Contains(tc.src, "generate(replayRequest.prompt)") {
			t.Fatalf("%s still carries page-local replay autostart", tc.name)
		}
		if strings.Contains(tc.src, "function clamp(") || strings.Contains(tc.src, "function mix(") {
			t.Fatalf("%s still carries page-local interface math helpers", tc.name)
		}
		if strings.Contains(tc.src, "function setCanvasSize(") ||
			strings.Contains(tc.src, "window.devicePixelRatio") ||
			strings.Contains(tc.src, "canvas.width = Math.max(1, Math.floor(width * dpr))") ||
			strings.Contains(tc.src, "context.setTransform(dpr, 0, 0, dpr, 0, 0)") {
			t.Fatalf("%s still carries page-local canvas sizing helpers", tc.name)
		}
		if strings.Contains(tc.src, "window.addEventListener('resize'") ||
			strings.Contains(tc.src, `window.addEventListener("resize"`) {
			t.Fatalf("%s still binds resize locally instead of interface_boot", tc.name)
		}
		for _, localEvent := range []string{
			"window.addEventListener('keydown'",
			`window.addEventListener("keydown"`,
			"window.addEventListener('keyup'",
			`window.addEventListener("keyup"`,
			"window.addEventListener('mousemove'",
			`window.addEventListener("mousemove"`,
			"window.addEventListener('mouseout'",
			`window.addEventListener("mouseout"`,
			"window.addEventListener('mousedown'",
			`window.addEventListener("mousedown"`,
		} {
			if strings.Contains(tc.src, localEvent) {
				t.Fatalf("%s still binds browser input locally instead of interface_events", tc.name)
			}
		}
		if strings.Contains(tc.src, "function cleanWords(") ||
			strings.Contains(tc.src, "function tokenTextForTape(") ||
			strings.Contains(tc.src, "replace(/[^\\p{L}\\p{N}_") {
			t.Fatalf("%s still carries page-local interface text normalization", tc.name)
		}
		for _, forbidden := range []string{
			"window.YentInterfaceSession",
			"window.YentInterfaceRestore",
			"window.YentEventStream",
			"window.YentChatStream",
			"window.YentInterfaceText",
			"window.YentTokenTelemetry",
			"window.YentInterfaceState",
			"window.YentInterfaceClock",
			"window.YentInterfaceStatus",
			"window.YentInterfaceOutput",
			"window.YentInterfaceHud",
			"window.YentInterfaceReplay",
			"window.YentInterfaceInput",
			"window.YentInterfaceEvents",
			"window.YentInterfaceTurn",
			"window.YentInterfaceOutcome",
			"window.YentInterfaceRun",
			"window.YentInterfaceBoot",
			"window.YentInterfaceAnimation",
			"window.YentInterfaceMath",
			"window.YentInterfaceCanvas",
			"window.YentInterfaceStyle",
			"window.YentWorldmodelGeometry",
		} {
			if strings.Contains(tc.src, forbidden) {
				t.Fatalf("%s still reaches directly for %s instead of interface_deps", tc.name, forbidden)
			}
		}
	}
	if !strings.Contains(replayJS, "const req = options.replayRequest || {}") ||
		strings.Contains(replayJS, "options.request ||") ||
		strings.Contains(replayJS, "options.promptInput || options.input") ||
		strings.Contains(replayJS, "options.generationRun || options.run") {
		t.Fatalf("interface_replay.js still accepts generic replay autostart aliases")
	}
	if !strings.Contains(bootJS, "replayRequest: options.replayRequest") ||
		strings.Contains(bootJS, "request: options.replayRequest || options.request") {
		t.Fatalf("interface_boot.js still remaps replay autostart through a generic request alias")
	}
	if !strings.Contains(worldJS, "interfaceDeps.load({ worldGeometry: true })") {
		t.Fatalf("worldmodel.js does not request worldmodel geometry through interface_deps")
	}
	if !strings.Contains(yentJS, "interfaceDeps.load({ transcript: true })") {
		t.Fatalf("yent.js does not request transcript rendering through interface_deps")
	}
	if !strings.Contains(worldJS, "interfaceStatus.setManifest({ labels: statusLabels") {
		t.Fatalf("worldmodel.js does not write manifest state through interface_status")
	}
	if !strings.Contains(yentJS, "interfaceOutput.bind({ id: 'transcript' })") {
		t.Fatalf("yent.js does not bind transcript output through interface_output")
	}
	if !strings.Contains(worldJS, "interfaceOutput.bind({ id: 'manifest-text' })") {
		t.Fatalf("worldmodel.js does not bind manifest output through interface_output")
	}
	if !strings.Contains(yentJS, "interfaceOutput.setTextAndScroll({") ||
		!strings.Contains(yentJS, "target: assistantBody") ||
		!strings.Contains(yentJS, "interfaceOutput.setText({") {
		t.Fatalf("yent.js does not route live assistant output writes through interface_output")
	}
	if !strings.Contains(worldJS, "interfaceOutput.setTextAndScroll({ target: manifestText") {
		t.Fatalf("worldmodel.js does not route manifest output writes through interface_output")
	}
	if !strings.Contains(yentJS, "deps.interfaceTranscript") ||
		!strings.Contains(yentJS, "interfaceTranscript.appendTurn(") ||
		!strings.Contains(yentJS, "interfaceTranscript.clear(") {
		t.Fatalf("yent.js does not route transcript turn rendering through interface_transcript")
	}
	for _, directOutput := range []string{
		"body.textContent = text || ''",
		"assistantBody.textContent = responseText",
		"assistantBody.textContent = result.hasText",
		"manifestText.textContent = text",
		"transcript.textContent = ''",
		"transcript.scrollTop = transcript.scrollHeight",
		"manifestText.scrollTop = manifestText.scrollHeight",
		"document.createElement('article')",
		"document.createElement(\"article\")",
		"label.textContent = role === 'user'",
	} {
		if strings.Contains(yentJS, directOutput) || strings.Contains(worldJS, directOutput) {
			t.Fatalf("interface page still writes output text/scroll locally: %s", directOutput)
		}
	}
	if strings.Contains(outputJS, "function setText(target") ||
		strings.Contains(outputJS, "function scrollBottom(target") ||
		strings.Contains(outputJS, "function setTextAndScroll(target") ||
		strings.Contains(yentJS, "interfaceOutput.setTextAndScroll(assistantBody") ||
		strings.Contains(yentJS, "interfaceOutput.setText(assistantBody") ||
		strings.Contains(worldJS, "interfaceOutput.setTextAndScroll(manifestText") {
		t.Fatalf("interface_output.js still exposes positional output writer arguments")
	}
	if strings.Contains(worldJS, "function textSeed") || strings.Contains(worldJS, "function hash") {
		t.Fatalf("worldmodel.js still carries page-local topology hash/seed helpers")
	}
	for _, required := range []string{
		"YentInterfaceSession",
		"YentInterfaceRestore",
		"YentEventStream",
		"YentChatStream",
		"YentInterfaceText",
		"YentTokenTelemetry",
		"YentInterfaceState",
		"YentInterfaceClock",
		"YentInterfaceStatus",
		"YentInterfaceOutput",
		"YentInterfaceTranscript",
		"YentInterfaceHud",
		"YentInterfaceReplay",
		"YentInterfaceInput",
		"YentInterfaceEvents",
		"YentInterfaceTurn",
		"YentInterfaceSubmit",
		"YentInterfaceOutcome",
		"YentInterfaceRun",
		"YentInterfaceBoot",
		"YentInterfaceAnimation",
		"YentInterfaceMath",
		"YentInterfaceCanvas",
		"YentInterfaceStyle",
		"YentWorldmodelGeometry",
	} {
		if !strings.Contains(depsJS, required) {
			t.Fatalf("interface_deps.js does not guard dependency %s", required)
		}
	}
	if !strings.Contains(depsJS, "function looksLikeDependencyHost") ||
		!strings.Contains(depsJS, "interface dependency root must be passed as { root }") ||
		!strings.Contains(depsJS, "const host = hasOwn(options, 'root') ? options.root : root") {
		t.Fatalf("interface_deps.js still permits implicit dependency root shortcuts")
	}
	for _, required := range []string{
		"`GET /yent`",
		"Janus parliament face",
		"`GET /worldmodel`",
		"walkable probability field",
		"`POST /chat/completions`",
		"You can freely run this code with regular Mistral Model",
		"notorch",
		"MetaJanus",
		"Will",
	} {
		if !strings.Contains(readme, required) {
			t.Fatalf("README.md does not document current interface contract: missing %s", required)
		}
	}
	if strings.Contains(readme, "All four helpers") {
		t.Fatalf("README.md still describes the old four-helper interface contract")
	}
	if !strings.Contains(textJS, "cleanWords") ||
		!strings.Contains(textJS, "tokenTapeText") ||
		!strings.Contains(textJS, "appendTape") {
		t.Fatalf("interface_text.js does not own shared text normalization")
	}
	if !strings.Contains(tokenTelemetryJS, "function resetCandidateState") ||
		!strings.Contains(tokenTelemetryJS, "function applyCandidateState") ||
		!strings.Contains(tokenTelemetryJS, "options.candidateCount") {
		t.Fatalf("token_telemetry.js does not own shared candidate state bookkeeping")
	}
	if !strings.Contains(stateJS, "BASELINE") ||
		!strings.Contains(stateJS, "function create") ||
		!strings.Contains(stateJS, "hasCandidateTelemetry") {
		t.Fatalf("interface_state.js does not own shared HUD/runtime state defaults")
	}
	if !strings.Contains(clockJS, "function create") ||
		!strings.Contains(clockJS, "function reset") ||
		!strings.Contains(clockJS, "function tick") ||
		!strings.Contains(clockJS, "minElapsedSeconds") {
		t.Fatalf("interface_clock.js does not own shared generation throughput timing")
	}
	if strings.Contains(clockJS, "options && options.performance ? options.performance : root.performance") ||
		strings.Contains(clockJS, "options.performance || root.performance") {
		t.Fatalf("interface_clock.js still lets explicit null performance fall back to browser globals")
	}
	if !strings.Contains(statusJS, "function bind") ||
		!strings.Contains(statusJS, "function setText") ||
		!strings.Contains(statusJS, "function setActive") ||
		!strings.Contains(statusJS, "function setManifest") {
		t.Fatalf("interface_status.js does not own shared status label writes")
	}
	if strings.Contains(statusJS, "function setText(target") ||
		strings.Contains(statusJS, "function setActive(target") ||
		strings.Contains(statusJS, "function setManifest(labels") ||
		strings.Contains(statusJS, "const ids = options.ids || options") ||
		strings.Contains(yentJS, "interfaceStatus.bind({ run:") ||
		strings.Contains(worldJS, "interfaceStatus.bind({\n  note:") ||
		strings.Contains(yentJS, "interfaceStatus.setText(statusLabels.run") ||
		strings.Contains(worldJS, "interfaceStatus.setText(statusLabels.note") ||
		strings.Contains(worldJS, "interfaceStatus.setManifest(statusLabels") {
		t.Fatalf("interface_status.js still exposes positional status writer arguments")
	}
	if !strings.Contains(yentJS, "interfaceStatus.bind({ ids: { run: 'run-state' } })") ||
		!strings.Contains(worldJS, "interfaceStatus.bind({\n  ids: {") {
		t.Fatalf("page status labels are not bound through explicit ids")
	}
	if !strings.Contains(hudJS, "function bind") ||
		!strings.Contains(hudJS, "function render") {
		t.Fatalf("interface_hud.js does not own shared HUD binding and rendering")
	}
	if !strings.Contains(hudJS, "hasOwn(options, 'tokenTelemetry') ? options.tokenTelemetry : root.YentTokenTelemetry") ||
		strings.Contains(hudJS, "(options && options.tokenTelemetry) || root.YentTokenTelemetry") ||
		strings.Contains(hudJS, "options.tokenTelemetry || root.YentTokenTelemetry") {
		t.Fatalf("interface_hud.js does not preserve explicit null token telemetry helper dependencies")
	}
	if !strings.Contains(inputJS, "function bindControls") ||
		!strings.Contains(inputJS, "function defaultDocument") ||
		!strings.Contains(inputJS, "getElementById(id)") ||
		!strings.Contains(inputJS, "promptInput") ||
		!strings.Contains(inputJS, "sendButton") ||
		!strings.Contains(inputJS, "function isFocused") ||
		!strings.Contains(inputJS, "activeElement === target") {
		t.Fatalf("interface_input.js does not own shared prompt/composer/send control binding")
	}
	if !strings.Contains(outputJS, "function bind") ||
		!strings.Contains(outputJS, "documentRef.getElementById(id)") ||
		!strings.Contains(outputJS, "function setText") ||
		!strings.Contains(outputJS, "function scrollBottom") ||
		!strings.Contains(outputJS, "function setTextAndScroll") {
		t.Fatalf("interface_output.js does not own shared output lookup, text, and scroll writes")
	}
	if strings.Contains(outputJS, "function bind(documentRef") ||
		strings.Contains(outputJS, "typeof options === 'string' ?") ||
		strings.Contains(hudJS, "function bind(documentRef") ||
		strings.Contains(statusJS, "function bind(documentRef") ||
		strings.Contains(hudJS, "const ids = options.ids || options") ||
		strings.Contains(hudJS, "arguments.length") ||
		strings.Contains(statusJS, "arguments.length") {
		t.Fatalf("visual binding helpers still expose positional document arguments")
	}
	for _, tc := range []struct {
		name string
		src  string
	}{
		{"interface_canvas.js", canvasJS},
		{"interface_hud.js", hudJS},
		{"interface_output.js", outputJS},
		{"interface_status.js", statusJS},
		{"interface_style.js", styleJS},
		{"interface_transcript.js", transcriptJS},
	} {
		if strings.Contains(tc.src, "options.document ||") ||
			strings.Contains(tc.src, "(options && options.document) ||") {
			t.Fatalf("%s still lets explicit null document fall back to browser globals", tc.name)
		}
	}
	if strings.Contains(inputJS, "function bindControls(documentRef") ||
		strings.Contains(inputJS, "function readParams(documentRef") ||
		strings.Contains(inputJS, "function isFocused(documentRef") ||
		strings.Contains(inputJS, "arguments.length") ||
		strings.Contains(inputJS, "composer: options.composer") ||
		strings.Contains(inputJS, "prompt: options.prompt") ||
		strings.Contains(inputJS, "send: options.send") ||
		strings.Contains(worldJS, "interfaceInput.isFocused(promptInput)") {
		t.Fatalf("interface_input.js still exposes positional or generic control arguments")
	}
	if !strings.Contains(worldJS, "interfaceInput.isFocused({ control: promptInput })") {
		t.Fatalf("worldmodel.js does not pass focus control through the named input helper contract")
	}
	if !strings.Contains(textJS, "function appendTape(options)") ||
		!strings.Contains(textJS, "options.tape") ||
		!strings.Contains(textJS, "options.text") ||
		!strings.Contains(textJS, "options.limit") {
		t.Fatalf("interface_text.js does not own named tape append inputs")
	}
	if strings.Contains(textJS, "function appendTape(tape") ||
		strings.Contains(yentJS, "interfaceText.appendTape(tokenTape") {
		t.Fatalf("interface_text.js still exposes positional tape append arguments")
	}
	if !strings.Contains(transcriptJS, "function labelFor") ||
		!strings.Contains(transcriptJS, "function appendTurn") ||
		!strings.Contains(transcriptJS, "function clear") ||
		!strings.Contains(transcriptJS, "output.setText({ target: body") ||
		!strings.Contains(transcriptJS, "output.scrollBottom({ target: container })") {
		t.Fatalf("interface_transcript.js does not own transcript turn rendering")
	}
	if strings.Contains(transcriptJS, "function appendTurn(container") ||
		strings.Contains(transcriptJS, "function clear(container") ||
		strings.Contains(yentJS, "appendTurn(transcript") ||
		strings.Contains(yentJS, "clear(transcript") {
		t.Fatalf("interface_transcript.js still exposes positional transcript container arguments")
	}
	if !strings.Contains(transcriptJS, "hasOwn(options, 'interfaceOutput') ? options.interfaceOutput : root.YentInterfaceOutput") ||
		strings.Contains(transcriptJS, "options.interfaceOutput || root.YentInterfaceOutput") ||
		strings.Contains(transcriptJS, "(options && options.interfaceOutput) || root.YentInterfaceOutput") {
		t.Fatalf("interface_transcript.js does not preserve explicit null output helper dependencies")
	}
	if !strings.Contains(restoreJS, "if (options.replayMode) return null") ||
		!strings.Contains(restoreJS, "const session = options.sessionReceipt") ||
		!strings.Contains(restoreJS, "session.load()") ||
		!strings.Contains(restoreJS, "combinedText") ||
		!strings.Contains(restoreJS, "lastAssistant") {
		t.Fatalf("interface_restore.js does not own shared replay-aware receipt restore")
	}
	if strings.Contains(restoreJS, "options.sessionReceipt || options.session") {
		t.Fatalf("interface_restore.js still accepts a generic session alias")
	}
	if !strings.Contains(replayJS, "function request") ||
		!strings.Contains(replayJS, "root.location") ||
		!strings.Contains(replayJS, "startIfRequested") {
		t.Fatalf("interface_replay.js does not own replay request/default location handling")
	}
	if strings.Contains(replayJS, "function request(location)") {
		t.Fatalf("interface_replay.js still exposes positional replay request location")
	}
	if !strings.Contains(replayJS, "replay request search must be passed as { location }") ||
		strings.Contains(replayJS, "hasOwn(options, 'search') ?") ||
		strings.Contains(replayJS, "options.search } : root.location") {
		t.Fatalf("interface_replay.js still accepts top-level search instead of named location")
	}
	if !strings.Contains(replayJS, "replay scenario name must be passed as { scenario }") ||
		strings.Contains(replayJS, "options.scenario || options.name") {
		t.Fatalf("interface_replay.js still accepts the generic scenario name alias")
	}
	if !strings.Contains(replayJS, "replay fixture scenario must be passed as { scenario }") ||
		!strings.Contains(replayJS, "replay fixture scenario name must be passed as { scenario }") ||
		strings.Contains(replayJS, "function scenario(name)") ||
		strings.Contains(replayJS, "scenario(options.scenario)") ||
		strings.Contains(replayJS, "scenario(name);") {
		t.Fatalf("interface_replay.js still exposes positional replay fixture selection")
	}
	if strings.Contains(replayJS, "options.setTimeout || root.setTimeout") ||
		strings.Contains(replayJS, "(options && options.setTimeout) || root.setTimeout") {
		t.Fatalf("interface_replay.js still lets explicit null timers fall back to browser globals")
	}
	if strings.Contains(bootJS, "setTimeout: options.setTimeout") {
		t.Fatalf("interface_boot.js still turns omitted replay timers into named undefined timers")
	}
	if !strings.Contains(bootJS, "hasOwn(options, 'interfaceReplay') ? options.interfaceReplay : root.YentInterfaceReplay") ||
		strings.Contains(bootJS, "(options && options.interfaceReplay) || root.YentInterfaceReplay") ||
		strings.Contains(bootJS, "options.interfaceReplay || root.YentInterfaceReplay") {
		t.Fatalf("interface_boot.js does not preserve explicit null replay helper dependencies")
	}
	if !strings.Contains(inputJS, "hasOwn(options, 'interfaceReplay') ? options.interfaceReplay : root.YentInterfaceReplay") ||
		!strings.Contains(inputJS, "hasOwn(options, 'chatStream') ? options.chatStream : root.YentChatStream") ||
		strings.Contains(inputJS, "options.interfaceReplay || root.YentInterfaceReplay") ||
		strings.Contains(inputJS, "options.chatStream || root.YentChatStream") {
		t.Fatalf("interface_input.js does not preserve explicit null stream helper dependencies")
	}
	if !strings.Contains(canvasJS, "devicePixelRatio") ||
		!strings.Contains(canvasJS, "setTransform(base.dpr") ||
		!strings.Contains(canvasJS, "canvas.width = Math.max") ||
		!strings.Contains(canvasJS, "canvas.style.width") ||
		!strings.Contains(canvasJS, "resize surface must be passed as { surface } or { surfaces }") ||
		!strings.Contains(canvasJS, "interface canvas id must be passed as { id }") ||
		!strings.Contains(canvasJS, "interface canvas document must be passed as { document }") ||
		!strings.Contains(canvasJS, "function bind") ||
		!strings.Contains(canvasJS, "documentRef.getElementById(id)") ||
		!strings.Contains(canvasJS, "function createScratch") ||
		!strings.Contains(canvasJS, "documentRef.createElement('canvas')") ||
		!strings.Contains(canvasJS, "canvas.getContext(contextType") {
		t.Fatalf("interface_canvas.js does not own shared canvas binding and backing-store sizing")
	}
	if strings.Contains(canvasJS, "function pixelRatio(windowRef") ||
		strings.Contains(canvasJS, "function viewport(windowRef") ||
		strings.Contains(canvasJS, "pixelRatio(win, maxDpr)") ||
		strings.Contains(canvasJS, "function bind(documentRef") ||
		strings.Contains(canvasJS, "function createScratch(documentRef") ||
		strings.Contains(canvasJS, "{ canvas: options.canvas, context: options.context }") {
		t.Fatalf("interface_canvas.js still exposes positional viewport/window arguments")
	}
	if strings.Contains(canvasJS, "options.viewport || root") ||
		strings.Contains(canvasJS, "viewport: options.viewport, maxDpr") {
		t.Fatalf("interface_canvas.js still lets explicit null viewport fall back to browser globals")
	}
	if strings.Contains(hudJS, "function render(hud") ||
		strings.Contains(yentJS, "interfaceHud.render(hud") ||
		strings.Contains(worldJS, "interfaceHud.render(hud") {
		t.Fatalf("interface_hud.js still exposes positional HUD render arguments")
	}
	if !strings.Contains(styleJS, "function create") ||
		!strings.Contains(styleJS, "getComputedStyle") ||
		!strings.Contains(styleJS, "FALLBACKS") {
		t.Fatalf("interface_style.js does not own shared browser font/style lookup")
	}
	if !strings.Contains(eventsJS, "function bindKeyState") ||
		!strings.Contains(eventsJS, "function bindPointer") ||
		!strings.Contains(eventsJS, "resolveTarget(options)") ||
		!strings.Contains(eventsJS, "addEventListener") ||
		!strings.Contains(eventsJS, "function bindListener(options)") ||
		!strings.Contains(eventsJS, "listenerOptions") {
		t.Fatalf("interface_events.js does not own shared browser input event binding")
	}
	if strings.Contains(eventsJS, "function cleanupFor(target") ||
		strings.Contains(eventsJS, "function bind(target") ||
		strings.Contains(eventsJS, "bind(target, 'keydown'") ||
		strings.Contains(eventsJS, "bind(target, 'mousemove'") {
		t.Fatalf("interface_events.js still exposes positional listener binding arguments")
	}
	if strings.Contains(eventsJS, "options && options.target) || root") ||
		strings.Contains(eventsJS, "options.target || root") {
		t.Fatalf("interface_events.js still lets explicit null targets fall back to browser globals")
	}
	if strings.Contains(canvasJS, "options.window") ||
		strings.Contains(eventsJS, "options.window") ||
		strings.Contains(bootJS, "options.window") {
		t.Fatalf("interface helpers still expose page window alias parameters")
	}
	if strings.Contains(submitJS, "options.document") ||
		strings.Contains(turnJS, "options.document") ||
		strings.Contains(submitJS, "root.document") ||
		strings.Contains(turnJS, "root.document") {
		t.Fatalf("submit/turn helpers still expose generic document plumbing")
	}
	if !strings.Contains(submitJS, "const generationRun = options.generationRun") ||
		!strings.Contains(submitJS, "const session = options.sessionReceipt") ||
		!strings.Contains(submitJS, "replayRequest: options.replayRequest") ||
		strings.Contains(submitJS, "options.generationRun || options.run") ||
		strings.Contains(submitJS, "options.sessionReceipt || options.session") ||
		strings.Contains(submitJS, "options.replayRequest || options.request") {
		t.Fatalf("interface_submit.js still accepts generic run/session/request aliases")
	}
	if !strings.Contains(turnJS, "const session = options.sessionReceipt") ||
		!strings.Contains(turnJS, "replayRequest: options.replayRequest") ||
		strings.Contains(turnJS, "options.sessionReceipt || options.session") ||
		strings.Contains(turnJS, "options.replayRequest || options.request") {
		t.Fatalf("interface_turn.js still accepts generic session/request aliases")
	}
	if !strings.Contains(turnJS, "hasOwn(options, 'interfaceInput') ? options.interfaceInput : root.YentInterfaceInput") ||
		!strings.Contains(turnJS, "hasOwn(options, 'chatStream') ? options.chatStream : root.YentChatStream") ||
		!strings.Contains(turnJS, "hasOwn(options, 'interfaceReplay') ? options.interfaceReplay : root.YentInterfaceReplay") ||
		strings.Contains(turnJS, "options.interfaceInput || root.YentInterfaceInput") ||
		strings.Contains(turnJS, "options.chatStream || root.YentChatStream") ||
		strings.Contains(turnJS, "options.interfaceReplay || root.YentInterfaceReplay") {
		t.Fatalf("interface_turn.js does not preserve explicit null helper dependencies")
	}
	if !strings.Contains(submitJS, "hasOwn(options, 'interfaceTurn') ? options.interfaceTurn : root.YentInterfaceTurn") ||
		!strings.Contains(submitJS, "hasOwn(options, 'interfaceInput') ? options.interfaceInput : root.YentInterfaceInput") ||
		!strings.Contains(submitJS, "hasOwn(options, 'chatStream') ? options.chatStream : root.YentChatStream") ||
		!strings.Contains(submitJS, "hasOwn(options, 'interfaceReplay') ? options.interfaceReplay : root.YentInterfaceReplay") ||
		strings.Contains(submitJS, "options.interfaceTurn || root.YentInterfaceTurn") ||
		strings.Contains(submitJS, "options.interfaceInput || root.YentInterfaceInput") ||
		strings.Contains(submitJS, "options.chatStream || root.YentChatStream") ||
		strings.Contains(submitJS, "options.interfaceReplay || root.YentInterfaceReplay") {
		t.Fatalf("interface_submit.js does not preserve explicit null helper dependencies")
	}
	if !strings.Contains(submitJS, "generationRun.begin(") ||
		!strings.Contains(submitJS, "session.commitUser(") ||
		!strings.Contains(submitJS, "turnHelper.streamAssistant(") ||
		!strings.Contains(submitJS, "paramsDocument: options.paramsDocument") ||
		!strings.Contains(submitJS, "generationRun.finish(currentRun)") {
		t.Fatalf("interface_submit.js does not own the shared submit lifecycle")
	}
	if !strings.Contains(turnJS, "input.readParams({ document: options.paramsDocument })") ||
		!strings.Contains(turnJS, "input.readParams()") {
		t.Fatalf("interface_turn.js does not keep request params behind the named paramsDocument boundary")
	}
	if !strings.Contains(chatStreamJS, "function outcome(options)") ||
		!strings.Contains(chatStreamJS, "const error = options.error") ||
		!strings.Contains(chatStreamJS, "const responseText = options.responseText") {
		t.Fatalf("chat_stream.js does not own named stream outcome inputs")
	}
	if !strings.Contains(chatStreamJS, "const eventStream = hasOwn(options, 'eventStream') ? options.eventStream : root.YentEventStream") ||
		!strings.Contains(chatStreamJS, "if (hasOwn(options, 'fetch'))") ||
		!strings.Contains(chatStreamJS, "const Decoder = hasOwn(options, 'TextDecoder') ? options.TextDecoder : root.TextDecoder") ||
		!strings.Contains(chatStreamJS, "function endpointFor(options)") {
		t.Fatalf("chat_stream.js does not preserve named transport/decode dependency boundaries")
	}
	if strings.Contains(chatStreamJS, "(options && options.eventStream) || root.YentEventStream") ||
		strings.Contains(chatStreamJS, "(options && options.TextDecoder) || root.TextDecoder") ||
		strings.Contains(chatStreamJS, "options.endpoint || DEFAULT_ENDPOINT") {
		t.Fatalf("chat_stream.js still lets explicit null transport inputs fall back to defaults")
	}
	if strings.Contains(chatStreamJS, "function outcome(error") ||
		strings.Contains(turnJS, "chat.outcome(null,") ||
		strings.Contains(turnJS, "chat.outcome(err,") {
		t.Fatalf("chat stream outcome still exposes positional error/text arguments")
	}
	if !strings.Contains(outcomeJS, "outcome.stopped") ||
		!strings.Contains(outcomeJS, "outcome.fault") ||
		!strings.Contains(outcomeJS, "handlers.complete") ||
		!strings.Contains(outcomeJS, "function handle(options)") ||
		!strings.Contains(outcomeJS, "submit = options.submit") ||
		!strings.Contains(outcomeJS, "handlers = options.handlers") {
		t.Fatalf("interface_outcome.js does not own shared outcome dispatch")
	}
	if strings.Contains(outcomeJS, "function handle(submit") ||
		strings.Contains(outcomeJS, "call(handlers.stopped,") ||
		strings.Contains(outcomeJS, "call(handlers.fault,") ||
		strings.Contains(outcomeJS, "call(handlers.complete,") {
		t.Fatalf("interface_outcome.js still exposes positional outcome dispatch arguments")
	}
	if !strings.Contains(bootJS, "addEventListener('resize'") ||
		!strings.Contains(bootJS, "resolveResizeTarget(options)") ||
		!strings.Contains(bootJS, "function bindResize(options)") ||
		!strings.Contains(bootJS, "listenerOptions: options.resizeListenerOptions") {
		t.Fatalf("interface_boot.js does not own shared resize target lookup and listener binding")
	}
	if strings.Contains(bootJS, "function bindResize(options, resize)") ||
		strings.Contains(bootJS, "bindResize(options, resize)") {
		t.Fatalf("interface_boot.js still exposes positional resize binding arguments")
	}
	if strings.Contains(bootJS, "options && options.resizeTarget) || root") ||
		strings.Contains(bootJS, "options.resizeTarget || root") ||
		strings.Contains(bootJS, "resizeTarget: options.resizeTarget") {
		t.Fatalf("interface_boot.js still lets explicit null resize targets fall back to browser globals")
	}
	if !strings.Contains(bootJS, "function bindComposer") ||
		!strings.Contains(bootJS, "generationRun.bindComposer(") ||
		!strings.Contains(bootJS, "form: options.composer") ||
		!strings.Contains(bootJS, "promptInput: options.promptInput") ||
		!strings.Contains(bootJS, "onSubmit: requireFunction(options.generate") ||
		!strings.Contains(bootJS, "bindComposer(options)") {
		t.Fatalf("interface_boot.js does not own shared composer listener binding")
	}
	if strings.Contains(runJS, "function bindComposer(form") ||
		strings.Contains(bootJS, "generationRun.bindComposer(\n      options.composer") ||
		strings.Contains(runJS, "const input = options.input") ||
		strings.Contains(bootJS, "input: options.promptInput") {
		t.Fatalf("interface_run.js still exposes positional composer binding arguments")
	}
	if !strings.Contains(runJS, "hasOwn(options, 'AbortController') ? options.AbortController : root.AbortController") ||
		strings.Contains(runJS, "(options && options.AbortController) || root.AbortController") ||
		strings.Contains(runJS, "options.AbortController || root.AbortController") {
		t.Fatalf("interface_run.js does not preserve explicit null AbortController dependencies")
	}
	if !strings.Contains(runJS, "composer prompt input must be passed as { promptInput }") {
		t.Fatalf("interface_run.js does not reject generic composer input alias")
	}
	if !strings.Contains(animationJS, "function create") ||
		!strings.Contains(animationJS, "requestAnimationFrame") ||
		!strings.Contains(animationJS, "function requestFrame") {
		t.Fatalf("interface_animation.js does not own shared animation frame scheduling")
	}
	if strings.Contains(animationJS, "options && options.requestAnimationFrame) || root.requestAnimationFrame") ||
		strings.Contains(animationJS, "options.requestAnimationFrame || root.requestAnimationFrame") {
		t.Fatalf("interface_animation.js still lets explicit null frame sources fall back to browser globals")
	}
}

func assertInterfaceBootStartOptions(t *testing.T, name, src string) {
	t.Helper()
	start := strings.Index(src, "interfaceBoot.start({")
	if start < 0 {
		t.Fatalf("%s does not start through the shared boot helper", name)
	}
	block := src[start:]
	if end := strings.Index(block, "\n  });"); end >= 0 {
		block = block[:end]
	}
	if strings.Contains(block, "\n  window,") || strings.Contains(block, "\n    window,") {
		t.Fatalf("%s still passes the browser window into the shared boot helper", name)
	}
	if !strings.Contains(block, "\n  composer,") && !strings.Contains(block, "\n    composer,") {
		t.Fatalf("%s does not pass the composer form into the shared boot helper", name)
	}
}

func assertScriptOrder(t *testing.T, name, html string, scripts ...string) {
	t.Helper()
	prevAt := -1
	for _, script := range scripts {
		at := strings.Index(html, script)
		if at < 0 {
			t.Fatalf("%s missing script %s", name, script)
		}
		if prevAt >= 0 && prevAt > at {
			t.Fatalf("%s loads script out of order near %s", name, script)
		}
		prevAt = at
	}
}

func readTextFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(data)
}
