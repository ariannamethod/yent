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
		filepath.Join(root, "DoE", "worldmodel", "interface_hud.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_replay.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_input.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_turn.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_submit.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_outcome.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_page_replay_smoke.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_run.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_boot.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_math.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_canvas.test.cjs"),
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
	restoreJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_restore.js"))
	textJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_text.js"))
	submitJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_submit.js"))
	outcomeJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_outcome.js"))
	bootJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_boot.js"))
	canvasJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_canvas.js"))
	depsJS := readTextFile(t, filepath.Join(root, "DoE", "worldmodel", "interface_deps.js"))
	readme := readTextFile(t, filepath.Join(root, "README.md"))

	assertScriptOrder(t, "yent.html", yentHTML,
		"/worldmodel/interface_session.js",
		"/worldmodel/interface_restore.js",
		"/worldmodel/event_stream.js",
		"/worldmodel/chat_stream.js",
		"/worldmodel/interface_text.js",
		"/worldmodel/token_telemetry.js",
		"/worldmodel/interface_hud.js",
		"/worldmodel/interface_replay.js",
		"/worldmodel/interface_input.js",
		"/worldmodel/interface_turn.js",
		"/worldmodel/interface_submit.js",
		"/worldmodel/interface_outcome.js",
		"/worldmodel/interface_run.js",
		"/worldmodel/interface_boot.js",
		"/worldmodel/interface_math.js",
		"/worldmodel/interface_canvas.js",
		"/worldmodel/interface_deps.js",
		"/worldmodel/yent.js")
	assertScriptOrder(t, "worldmodel.html", worldHTML,
		"/worldmodel/interface_session.js",
		"/worldmodel/interface_restore.js",
		"/worldmodel/event_stream.js",
		"/worldmodel/chat_stream.js",
		"/worldmodel/interface_text.js",
		"/worldmodel/token_telemetry.js",
		"/worldmodel/interface_hud.js",
		"/worldmodel/interface_replay.js",
		"/worldmodel/interface_input.js",
		"/worldmodel/interface_turn.js",
		"/worldmodel/interface_submit.js",
		"/worldmodel/interface_outcome.js",
		"/worldmodel/interface_run.js",
		"/worldmodel/interface_boot.js",
		"/worldmodel/interface_math.js",
		"/worldmodel/interface_canvas.js",
		"/worldmodel/interface_deps.js",
		"/worldmodel/worldmodel_geometry.js",
		"/worldmodel/worldmodel.js")

	if !strings.Contains(doeC, `"/worldmodel/interface_session.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_session.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_session.js")
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
	if !strings.Contains(doeC, `"/worldmodel/interface_math.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_math.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_math.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_canvas.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_canvas.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_canvas.js")
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
		if !strings.Contains(tc.src, "window.YentInterfaceDeps") {
			t.Fatalf("%s does not use the shared interface dependency helper", tc.name)
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
		if !strings.Contains(tc.src, "interfaceSubmit.run(") {
			t.Fatalf("%s does not submit turns through the shared interface submit helper", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceOutcome.handle(") {
			t.Fatalf("%s does not classify stream outcomes through the shared helper", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceHud.render(") {
			t.Fatalf("%s does not render HUD metrics through the shared helper", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceBoot.start(") {
			t.Fatalf("%s does not start through the shared boot helper", tc.name)
		}
		assertInterfaceBootStartPassesWindow(t, tc.name, tc.src)
		if !strings.Contains(tc.src, "deps.interfaceCanvas") ||
			!strings.Contains(tc.src, "interfaceCanvas.resize(") {
			t.Fatalf("%s does not size canvases through the shared helper", tc.name)
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
			strings.Contains(tc.src, "sendButton.textContent =") {
			t.Fatalf("%s still carries page-local generation run state", tc.name)
		}
		if strings.Contains(tc.src, "data.top_tokens") ||
			strings.Contains(tc.src, "candidate_tail_mass") ||
			strings.Contains(tc.src, "selected_prob") ||
			strings.Contains(tc.src, "selected_rank") {
			t.Fatalf("%s still parses token telemetry locally", tc.name)
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
			"window.YentInterfaceHud",
			"window.YentInterfaceReplay",
			"window.YentInterfaceInput",
			"window.YentInterfaceTurn",
			"window.YentInterfaceOutcome",
			"window.YentInterfaceRun",
			"window.YentInterfaceBoot",
			"window.YentInterfaceMath",
			"window.YentInterfaceCanvas",
			"window.YentWorldmodelGeometry",
		} {
			if strings.Contains(tc.src, forbidden) {
				t.Fatalf("%s still reaches directly for %s instead of interface_deps", tc.name, forbidden)
			}
		}
	}
	if !strings.Contains(worldJS, "interfaceDeps.load({ worldGeometry: true })") {
		t.Fatalf("worldmodel.js does not request worldmodel geometry through interface_deps")
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
		"YentInterfaceHud",
		"YentInterfaceReplay",
		"YentInterfaceInput",
		"YentInterfaceTurn",
		"YentInterfaceSubmit",
		"YentInterfaceOutcome",
		"YentInterfaceRun",
		"YentInterfaceBoot",
		"YentInterfaceMath",
		"YentInterfaceCanvas",
		"YentWorldmodelGeometry",
	} {
		if !strings.Contains(depsJS, required) {
			t.Fatalf("interface_deps.js does not guard dependency %s", required)
		}
	}
	for _, required := range []string{
		"`DoE/worldmodel/interface_restore.js`",
		"`DoE/worldmodel/token_telemetry.js`",
		"`DoE/worldmodel/interface_text.js`",
		"`DoE/worldmodel/interface_hud.js`",
		"`DoE/worldmodel/worldmodel_geometry.js`",
		"`DoE/worldmodel/interface_replay.js`",
		"`DoE/worldmodel/interface_input.js`",
		"`DoE/worldmodel/interface_turn.js`",
		"`DoE/worldmodel/interface_submit.js`",
		"`DoE/worldmodel/interface_outcome.js`",
		"`DoE/worldmodel/interface_boot.js`",
		"`DoE/worldmodel/interface_math.js`",
		"`DoE/worldmodel/interface_canvas.js`",
		"`DoE/worldmodel/interface_deps.js`",
		"`/yent?replay=1`",
		"`/worldmodel?replay=1`",
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
	if !strings.Contains(restoreJS, "if (options.replayMode) return null") ||
		!strings.Contains(restoreJS, "session.load()") ||
		!strings.Contains(restoreJS, "combinedText") ||
		!strings.Contains(restoreJS, "lastAssistant") {
		t.Fatalf("interface_restore.js does not own shared replay-aware receipt restore")
	}
	if !strings.Contains(canvasJS, "devicePixelRatio") ||
		!strings.Contains(canvasJS, "setTransform(base.dpr") ||
		!strings.Contains(canvasJS, "canvas.width = Math.max") ||
		!strings.Contains(canvasJS, "canvas.style.width") {
		t.Fatalf("interface_canvas.js does not own shared canvas backing-store sizing")
	}
	if !strings.Contains(submitJS, "generationRun.begin(") ||
		!strings.Contains(submitJS, "session.commitUser(") ||
		!strings.Contains(submitJS, "turnHelper.streamAssistant(") ||
		!strings.Contains(submitJS, "generationRun.finish(currentRun)") {
		t.Fatalf("interface_submit.js does not own the shared submit lifecycle")
	}
	if !strings.Contains(outcomeJS, "outcome.stopped") ||
		!strings.Contains(outcomeJS, "outcome.fault") ||
		!strings.Contains(outcomeJS, "handlers.complete") {
		t.Fatalf("interface_outcome.js does not own shared outcome dispatch")
	}
	if !strings.Contains(bootJS, "addEventListener('resize'") ||
		!strings.Contains(bootJS, "bindResize(options.window || root") {
		t.Fatalf("interface_boot.js does not own shared resize listener binding")
	}
}

func assertInterfaceBootStartPassesWindow(t *testing.T, name, src string) {
	t.Helper()
	start := strings.Index(src, "interfaceBoot.start({")
	if start < 0 {
		t.Fatalf("%s does not start through the shared boot helper", name)
	}
	block := src[start:]
	if end := strings.Index(block, "\n  });"); end >= 0 {
		block = block[:end]
	}
	if !strings.Contains(block, "\n  window,") && !strings.Contains(block, "\n    window,") {
		t.Fatalf("%s does not pass the browser window into the shared boot helper", name)
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
