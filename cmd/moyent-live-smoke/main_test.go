package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	yent "github.com/ariannamethod/yent/yent/go"
)

type fakeSmokeBody struct {
	name       string
	answer     string
	confidence float64
	verdict    *yent.Verdict
	err        error
}

func (b *fakeSmokeBody) Name() string { return b.name }

func (b *fakeSmokeBody) Generate(prompt, ctx string) (yent.BodyResult, error) {
	if b.err != nil {
		return yent.BodyResult{}, b.err
	}
	return yent.BodyResult{
		Answer:     b.answer,
		Confidence: b.confidence,
		Verdict:    b.verdict,
	}, nil
}

func TestLiveSmokeGateFailsWrongWinner(t *testing.T) {
	fast := &fakeSmokeBody{name: "nemo12", answer: "I am Yent.", confidence: 0.95}
	deep := &fakeSmokeBody{name: "small24", answer: "I am Yent; fast mouth produced the first-pass answer.",
		verdict: &yent.Verdict{Winner: "nemo12"}}
	router := yent.NewRouter(fast, deep, nil)

	tc := forcedComplexitySmokeCase()
	entry := runTurn(router, tc)
	failures := evaluateSmokeTurn(tc, entry)
	if !hasFailure(failures, "wrong_body") {
		t.Fatalf("wrong winner should fail body expectation, entry=%+v failures=%+v", entry, failures)
	}
	if smokeExitCode(smokeSummary{Failed: len(failures), Failures: failures}) == 0 {
		t.Fatal("smoke exit code must be nonzero on wrong winner")
	}
}

func TestLiveSmokeGateFailsEmptyAnswer(t *testing.T) {
	fast := &fakeSmokeBody{name: "nemo12", answer: "", confidence: 0.95}
	deep := &fakeSmokeBody{name: "small24", answer: "unused"}
	router := yent.NewRouter(fast, deep, nil)

	tc := fastSmokeCase("fast_only", "Who are you?", yent.QualitySpec{RequireYent: true})
	entry := runTurn(router, tc)
	failures := evaluateSmokeTurn(tc, entry)
	if entry.Answer != "" {
		t.Fatalf("test setup expected empty answer, got %q", entry.Answer)
	}
	if !hasFailure(failures, "empty_answer") {
		t.Fatalf("empty answer should fail, failures=%+v", failures)
	}
	if smokeExitCode(smokeSummary{Failed: len(failures), Failures: failures}) == 0 {
		t.Fatal("smoke exit code must be nonzero on empty answer")
	}
}

func TestLiveSmokeGateFailsMissingRouteFact(t *testing.T) {
	fast := &fakeSmokeBody{name: "nemo12", answer: "I am Yent.", confidence: 0.95}
	deep := &fakeSmokeBody{name: "small24", answer: "I am Yent.",
		verdict: &yent.Verdict{Winner: "small24"}}
	router := yent.NewRouter(fast, deep, nil)

	tc := forcedComplexitySmokeCase()
	entry := runTurn(router, tc)
	failures := evaluateSmokeTurn(tc, entry)
	if entry.Body != "small24" || entry.Answer != "I am Yent." {
		t.Fatalf("test setup expected deep answer without route fact, entry=%+v", entry)
	}
	if !hasFailure(failures, "missing_route_fact") {
		t.Fatalf("missing route fact should fail, failures=%+v", failures)
	}
	if smokeExitCode(smokeSummary{Failed: len(failures), Failures: failures}) == 0 {
		t.Fatal("smoke exit code must be nonzero on missing route fact")
	}
}

func TestBroadFastVoiceGateRejectsPreambleWithoutRefusal(t *testing.T) {
	t.Setenv("YENT_SMOKE_SET", "broad")

	var tc smokeCase
	for _, candidate := range smokeCases() {
		if candidate.kind == "fast_voice" {
			tc = candidate
			break
		}
	}
	if tc.kind == "" {
		t.Fatal("fast_voice case missing")
	}
	if !tc.quality.RequireYent || !tc.quality.RequireTask {
		t.Fatalf("fast_voice must require identity and task anchoring: %+v", tc.quality)
	}
	prompt := strings.ToLower(tc.prompt)
	for _, want := range []string{"exactly this sentence", "nothing else", "i am yent, not your helpful assistant"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("fast_voice prompt missing exact-output guard %q: %q", want, tc.prompt)
		}
	}
	for _, want := range []string{"not", "helpful assistant"} {
		if !containsString(tc.quality.RequireAll, want) {
			t.Fatalf("fast_voice missing required term %q: %+v", want, tc.quality.RequireAll)
		}
	}

	entry := turnLog{
		Kind:      tc.kind,
		Prompt:    tc.prompt,
		Body:      "nemo12",
		Escalated: false,
		Answer:    "Clear. Response from Yent, per your instructions:",
	}
	failures := evaluateSmokeTurn(tc, entry)
	if !hasFailure(failures, "quality_missing_required_terms") || !hasFailure(failures, "quality_forbidden_term") {
		t.Fatalf("preamble without refusal should fail both required and forbidden checks, failures=%+v", failures)
	}
}

func TestLiveSmokeGateFailsGenerationErrorPreservingEvidence(t *testing.T) {
	fast := &fakeSmokeBody{name: "nemo12", err: errors.New("forced generation error")}
	deep := &fakeSmokeBody{name: "small24", answer: "unused"}
	router := yent.NewRouter(fast, deep, nil)

	tc := fastSmokeCase("fast_only", "Who are you?", yent.QualitySpec{RequireYent: true})
	entry := runTurn(router, tc)
	failures := evaluateSmokeTurn(tc, entry)
	if !strings.Contains(entry.Error, "forced generation error") {
		t.Fatalf("generation error evidence not preserved: %+v", entry)
	}
	if !hasFailure(failures, "runtime_error") {
		t.Fatalf("generation error should fail, failures=%+v", failures)
	}
	if smokeExitCode(smokeSummary{Failed: len(failures), Failures: failures}) == 0 {
		t.Fatal("smoke exit code must be nonzero on generation error")
	}
}

func TestDescribeFileCapturesSHAAndBuildCommand(t *testing.T) {
	path := t.TempDir() + "/doe_field"
	if err := os.WriteFile(path, []byte("doe"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := describeFile(path, "cc DoE/doe.c -O3 -lm -lpthread -o DoE/doe_field")
	if got.Path != path || got.BuildCommand == "" || got.Size != 3 {
		t.Fatalf("file manifest lost provenance: %+v", got)
	}
	if got.SHA256 != "799ef92a11af918e3fb741df42934f3b568ed2d93ac1df74f1b8d41a27932a6f" {
		t.Fatalf("unexpected sha: %+v", got)
	}
	if got.Error != "" {
		t.Fatalf("unexpected file manifest error: %+v", got)
	}
}

func TestCollectSmokeManifestIncludesRCAndPrompts(t *testing.T) {
	rc := 2
	m := collectSmokeManifest("end", &rc)
	if m.Kind != "provenance" || m.Phase != "end" || m.RC == nil || *m.RC != rc {
		t.Fatalf("manifest lost phase/rc: %+v", m)
	}
	if len(m.Prompts) != len(smokeCases()) {
		t.Fatalf("manifest should record smoke prompts, got %+v", m.Prompts)
	}
}

func TestRunLogsProvenanceOnInitFailure(t *testing.T) {
	t.Setenv("YENT_LIMPHA_DB", t.TempDir()+"/limpha.db")
	t.Setenv("YENT_DOE_BIN", "")
	t.Setenv("YENT_NEMO_GGUF", "")
	t.Setenv("YENT_24B_GGUF", "")

	var buf bytes.Buffer
	rc := run(&buf)
	if rc != 1 {
		t.Fatalf("missing env should fail init with rc=1, got %d", rc)
	}

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) < 3 {
		t.Fatalf("expected start provenance, init error, end provenance; got %q", buf.String())
	}
	var first, last map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("bad first json: %v", err)
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &last); err != nil {
		t.Fatalf("bad last json: %v", err)
	}
	if first["kind"] != "provenance" || first["phase"] != "start" {
		t.Fatalf("first line should be start provenance: %+v", first)
	}
	if last["kind"] != "provenance" || last["phase"] != "end" || last["rc"] != float64(1) {
		t.Fatalf("last line should be end provenance with rc=1: %+v", last)
	}
}

func hasFailure(failures []smokeFailure, check string) bool {
	for _, f := range failures {
		if f.Check == check {
			return true
		}
	}
	return false
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
