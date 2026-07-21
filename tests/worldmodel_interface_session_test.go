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
		filepath.Join(root, "DoE", "worldmodel", "event_stream.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "chat_stream.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "token_telemetry.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_replay.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_page_replay_smoke.test.cjs"),
		filepath.Join(root, "DoE", "worldmodel", "interface_run.test.cjs"),
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
	readme := readTextFile(t, filepath.Join(root, "README.md"))

	assertScriptOrder(t, "yent.html", yentHTML,
		"/worldmodel/interface_session.js",
		"/worldmodel/event_stream.js",
		"/worldmodel/chat_stream.js",
		"/worldmodel/token_telemetry.js",
		"/worldmodel/interface_replay.js",
		"/worldmodel/interface_run.js",
		"/worldmodel/yent.js")
	assertScriptOrder(t, "worldmodel.html", worldHTML,
		"/worldmodel/interface_session.js",
		"/worldmodel/event_stream.js",
		"/worldmodel/chat_stream.js",
		"/worldmodel/token_telemetry.js",
		"/worldmodel/interface_replay.js",
		"/worldmodel/interface_run.js",
		"/worldmodel/worldmodel_geometry.js",
		"/worldmodel/worldmodel.js")

	if !strings.Contains(doeC, `"/worldmodel/interface_session.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_session.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_session.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/event_stream.js"`) ||
		!strings.Contains(doeC, `"worldmodel/event_stream.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist event_stream.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/chat_stream.js"`) ||
		!strings.Contains(doeC, `"worldmodel/chat_stream.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist chat_stream.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/token_telemetry.js"`) ||
		!strings.Contains(doeC, `"worldmodel/token_telemetry.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist token_telemetry.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_replay.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_replay.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_replay.js")
	}
	if !strings.Contains(doeC, `"/worldmodel/interface_run.js"`) ||
		!strings.Contains(doeC, `"worldmodel/interface_run.js not found"`) {
		t.Fatalf("DoE server does not explicitly whitelist interface_run.js")
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
		if !strings.Contains(tc.src, "window.YentInterfaceSession") {
			t.Fatalf("%s does not use the shared interface session helper", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceSession.createAdapter") {
			t.Fatalf("%s does not use the shared replay-aware session adapter", tc.name)
		}
		if !strings.Contains(tc.src, "window.YentEventStream") {
			t.Fatalf("%s does not use the shared event stream helper", tc.name)
		}
		if !strings.Contains(tc.src, "window.YentChatStream") {
			t.Fatalf("%s does not use the shared chat stream helper", tc.name)
		}
		if !strings.Contains(tc.src, "window.YentTokenTelemetry") {
			t.Fatalf("%s does not use the shared token telemetry helper", tc.name)
		}
		if !strings.Contains(tc.src, "window.YentInterfaceReplay") {
			t.Fatalf("%s does not use the shared interface replay helper", tc.name)
		}
		if !strings.Contains(tc.src, "window.YentInterfaceRun") {
			t.Fatalf("%s does not use the shared interface run helper", tc.name)
		}
		if !strings.Contains(tc.src, "interfaceReplay.play(") {
			t.Fatalf("%s does not route replay through the shared replay helper", tc.name)
		}
		if !strings.Contains(tc.src, "if (replayMode) return") {
			t.Fatalf("%s does not guard session continuity during replay", tc.name)
		}
		if !strings.Contains(tc.src, "chatStream.outcome(") {
			t.Fatalf("%s does not use the shared chat stream outcome classifier", tc.name)
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
	}
	if !strings.Contains(worldJS, "window.YentWorldmodelGeometry") {
		t.Fatalf("worldmodel.js does not use the worldmodel geometry helper")
	}
	if strings.Contains(worldJS, "function textSeed") || strings.Contains(worldJS, "function hash") {
		t.Fatalf("worldmodel.js still carries page-local topology hash/seed helpers")
	}
	for _, required := range []string{
		"`DoE/worldmodel/token_telemetry.js`",
		"`DoE/worldmodel/worldmodel_geometry.js`",
		"`DoE/worldmodel/interface_replay.js`",
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
