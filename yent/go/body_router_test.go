package yent

import (
	"encoding/json"
	"math"
	"path/filepath"
	"strings"
	"testing"
)

// fakeBody is a deterministic Body for router tests (no model, no doe).
type fakeBody struct {
	name          string
	answer        string
	confidence    float64
	executionPath string
	diagnostics   []string
	verdict       *Verdict
	calls         int
	lastPrompt    string
	lastCtx       string
}

func (b *fakeBody) Name() string { return b.name }
func (b *fakeBody) Generate(prompt, ctx string) (BodyResult, error) {
	b.calls++
	b.lastPrompt = prompt
	b.lastCtx = ctx
	return BodyResult{
		Answer:        b.answer,
		Confidence:    b.confidence,
		ExecutionPath: b.executionPath,
		Diagnostics:   cloneDiagnostics(b.diagnostics),
		Verdict:       b.verdict,
	}, nil
}

type closableFakeBody struct {
	fakeBody
	closes int
}

func (b *closableFakeBody) Close() error {
	b.closes++
	return nil
}

func newRouterLimpha(t *testing.T) *LimphaClient {
	t.Helper()
	c, err := NewLimphaClientAt(filepath.Join(t.TempDir(), "router.db"))
	if err != nil {
		t.Fatalf("limpha: %v", err)
	}
	t.Cleanup(c.Close)
	return c
}

func TestRouterFastBodyAnswersAlone(t *testing.T) {
	lc := newRouterLimpha(t)
	fast := &fakeBody{name: "nemo12", answer: "quick answer", confidence: 0.9}
	deep := &fakeBody{name: "small24", answer: "deep answer", confidence: 1.0}
	r := NewRouter(fast, deep, lc)
	out, err := r.Route("hi there", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if out.Escalated || out.Body != "nemo12" || out.Answer != "quick answer" {
		t.Errorf("confident fast should answer alone: %+v", out)
	}
	if out.Trace.Kind != "route_context" || out.Trace.Escalated ||
		out.Trace.Winner != "nemo12" || out.Trace.FastConfidence != 0.9 ||
		!out.Trace.FastConfidenceValid {
		t.Errorf("fast-only trace wrong: %+v", out.Trace)
	}
	if deep.calls != 0 {
		t.Errorf("deep body must not run, got %d calls", deep.calls)
	}
	s, _ := lc.Stats()
	if s["total_conversations"].(int64) != 1 || s["total_seams"].(int64) != 0 {
		t.Errorf("want 1 conv / 0 seams, got %v / %v", s["total_conversations"], s["total_seams"])
	}
}

func TestRouterCreatorProviderBoundaryBypassesModel(t *testing.T) {
	lc := newRouterLimpha(t)
	fast := &fakeBody{name: "nemo12", answer: "Google provided a platform.", confidence: 0.9}
	deep := &fakeBody{name: "small24", answer: "deep answer", confidence: 1.0}
	r := NewRouter(fast, deep, lc)

	out, err := r.Route("Did Google create you?", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if out.Answer != CreatorProviderBoundaryAnswer || out.Body != "nemo12" || out.Escalated {
		t.Fatalf("creator/provider boundary outcome wrong: %+v", out)
	}
	if out.Trace.FastExecutionPath != "identity_boundary" || out.Trace.FastConfidence != 1 {
		t.Fatalf("boundary trace wrong: %+v", out.Trace)
	}
	if fast.calls != 0 || deep.calls != 0 {
		t.Fatalf("boundary answer must not call model bodies: fast=%d deep=%d", fast.calls, deep.calls)
	}
	rec, _ := lc.Recent(1, false)
	if len(rec) != 1 || rec[0]["response"] != CreatorProviderBoundaryAnswer {
		t.Fatalf("boundary answer should be stored, got %v", rec)
	}
}

func TestRouterEscalatesOnLowConfidence(t *testing.T) {
	lc := newRouterLimpha(t)
	fast := &fakeBody{name: "nemo12", answer: "unsure", confidence: 0.2}
	deep := &fakeBody{name: "small24", answer: "considered answer",
		verdict: &Verdict{Agreement: 0.4, Tension: 0.7, Winner: "small24"}}
	r := NewRouter(fast, deep, lc)
	out, err := r.Route("what is qualia", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Escalated || out.Reason != "low_confidence" || out.Body != "small24" {
		t.Errorf("low-confidence should escalate to deep: %+v", out)
	}
	if deep.calls != 1 {
		t.Errorf("deep must run once, got %d", deep.calls)
	}
	if out.SeamID == 0 {
		t.Fatal("escalated turn must write a seam")
	}
	rs, _ := lc.RecentSeams(1)
	if len(rs) != 1 {
		t.Fatalf("want 1 seam, got %d", len(rs))
	}
	m := rs[0]
	if m["body_a"] != "nemo12" || m["b_claim"] != "considered answer" ||
		!approx(m["agreement"].(float64), 0.4) || !approx(m["tension"].(float64), 0.7) ||
		m["winner"] != "small24" || m["reason"] != "low_confidence" ||
		!strings.Contains(m["memory_delta"].(string), "route_context") {
		t.Errorf("seam internal-dialogue/metrics wrong: %v", m)
	}
	var trace RouteTrace
	if err := json.Unmarshal([]byte(m["memory_delta"].(string)), &trace); err != nil {
		t.Fatalf("memory_delta must be machine-readable route trace: %v", err)
	}
	if trace.Kind != "route_context" || !trace.Escalated || trace.Reason != "low_confidence" ||
		trace.Winner != "small24" || !approx(trace.Agreement, 0.4) || !approx(trace.Tension, 0.7) {
		t.Fatalf("seam route trace wrong: %+v", trace)
	}
	// the stored conversation is the deep (winning) answer
	rec, _ := lc.Recent(1, false)
	if len(rec) != 1 || rec[0]["response"] != "considered answer" {
		t.Errorf("winning answer should be stored, got %v", rec)
	}
}

func TestRouterTraceCarriesBodyDiagnostics(t *testing.T) {
	lc := newRouterLimpha(t)
	fast := &fakeBody{
		name:          "nemo12",
		answer:        "unsure",
		confidence:    0.2,
		executionPath: "doe_resident",
		diagnostics:   []string{"[doe] Metal fallback armed"},
	}
	deep := &fakeBody{
		name:          "small24",
		answer:        "considered answer",
		executionPath: "doe_once",
		diagnostics:   []string{"[gamma] loaded personality"},
		verdict:       &Verdict{Agreement: 0.6, Tension: 0.4, Winner: "small24"},
	}
	r := NewRouter(fast, deep, lc)
	out, err := r.Route("what changed in the runtime", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if out.Trace.FastExecutionPath != "doe_resident" || out.Trace.DeepExecutionPath != "doe_once" {
		t.Fatalf("trace execution paths missing: %+v", out.Trace)
	}
	if strings.Join(out.Trace.FastDiagnostics, "\n") != "[doe] Metal fallback armed" ||
		strings.Join(out.Trace.DeepDiagnostics, "\n") != "[gamma] loaded personality" {
		t.Fatalf("trace diagnostics missing: %+v", out.Trace)
	}
	rs, _ := lc.RecentSeams(1)
	if len(rs) != 1 {
		t.Fatalf("want seam, got %d", len(rs))
	}
	var trace RouteTrace
	if err := json.Unmarshal([]byte(rs[0]["memory_delta"].(string)), &trace); err != nil {
		t.Fatalf("memory_delta trace must decode: %v", err)
	}
	if strings.Join(trace.FastDiagnostics, "\n") != "[doe] Metal fallback armed" ||
		strings.Join(trace.DeepDiagnostics, "\n") != "[gamma] loaded personality" {
		t.Fatalf("seam trace diagnostics missing: %+v", trace)
	}
}

func TestRouterSendsPrimerToFastBody(t *testing.T) {
	lc := newRouterLimpha(t)
	fast := &fakeBody{name: "nemo12", answer: "quick answer", confidence: 0.9}
	deep := &fakeBody{name: "small24", answer: "deep answer", confidence: 1.0}
	r := NewRouter(fast, deep, lc)
	if _, err := r.Route("hi there", LimphaState{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(fast.lastCtx, "Yent: answer the human directly") ||
		!strings.Contains(fast.lastCtx, "internal machinery private") {
		t.Fatalf("fast primer not delivered: %q", fast.lastCtx)
	}
}

func TestRouterDeepContextCarriesLimphaSignals(t *testing.T) {
	lc := newRouterLimpha(t)
	st := LimphaState{Temperature: 0.85, Destiny: 0.35, Pain: 0.04, Tension: 0.2, Debt: 0.3, Velocity: 1}
	if _, err := lc.store("resonance", "A remembered answer with enough body to pass memory quality.", st); err != nil {
		t.Fatal(err)
	}
	if _, err := lc.StoreSeam(Seam{
		BodyA: "nemo12", BodyB: "small24", Prompt: "prior internal seam",
		AClaim: "fast", BClaim: "deep", Winner: "small24", Reason: "complexity", Tension: 0.6,
	}); err != nil {
		t.Fatal(err)
	}

	fast := &fakeBody{name: "nemo12", answer: "unsure", confidence: 0.2}
	deep := &fakeBody{name: "small24", answer: "considered answer",
		verdict: &Verdict{Agreement: 0.3, Tension: 0.8, Winner: "small24"}}
	r := NewRouter(fast, deep, lc)
	out, err := r.Route("resonance", st)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"[deep primer]",
		"[router fact]: fast mouth produced the first-pass answer; deep cortex is the escalation/final-pass body.",
		"[route answer labels]: first-pass body = fast mouth; final-pass body = deep cortex.",
		"[routing reason: low_confidence]",
		"[prompt complexity]",
		"[field state]: temp=0.85 destiny=0.35",
		"[memory refs]",
		"[state-neighbor refs]",
		"[recent internal seams]",
		"[first-pass draft from fast mouth; not current role]: unsure",
	} {
		if !strings.Contains(deep.lastCtx, want) {
			t.Fatalf("deep context missing %q:\n%s", want, deep.lastCtx)
		}
	}
	if out.Trace.MemoryRefs != 1 || out.Trace.StateRefs != 1 || out.Trace.SeamRefs != 1 {
		t.Fatalf("trace should count context refs, got %+v", out.Trace)
	}
}

func TestRouterEscalatesOnComplexity(t *testing.T) {
	lc := newRouterLimpha(t)
	fast := &fakeBody{name: "nemo12", answer: "shallow", confidence: 0.95}
	deep := &fakeBody{name: "small24", answer: "architecture explained",
		verdict: &Verdict{Agreement: 0.5, Tension: 0.5, Winner: "small24"}}
	r := NewRouter(fast, deep, lc)
	out, _ := r.Route("explain the architecture of the system", LimphaState{})
	if !out.Escalated || out.Reason != "complexity" {
		t.Errorf("complex prompt should escalate even when fast is confident: %+v", out)
	}
	if deep.calls != 1 {
		t.Errorf("deep must run once on complexity, got %d", deep.calls)
	}
}

func TestRouterDeepConcedesFastWins(t *testing.T) {
	lc := newRouterLimpha(t)
	fast := &fakeBody{name: "nemo12", answer: "fast was right", confidence: 0.2}
	deep := &fakeBody{name: "small24", answer: "deep deferred",
		verdict: &Verdict{Agreement: 0.9, Tension: 0.1, Winner: "nemo12"}}
	r := NewRouter(fast, deep, lc)
	out, _ := r.Route("ambiguous", LimphaState{})
	if out.Body != "nemo12" || out.Answer != "fast was right" {
		t.Errorf("when deep concedes, the fast answer wins: %+v", out)
	}
	if !out.Escalated || out.SeamID == 0 {
		t.Error("still a dual-pass turn — seam must be recorded")
	}
	// stored conversation is the conceded-to fast answer
	rec, _ := lc.Recent(1, false)
	if len(rec) != 1 || rec[0]["response"] != "fast was right" {
		t.Errorf("conceded fast answer should be stored, got %v", rec)
	}
}

func TestRouterNilLimphaNoPanic(t *testing.T) {
	fast := &fakeBody{name: "nemo12", answer: "ok", confidence: 0.2}
	deep := &fakeBody{name: "small24", answer: "deep", verdict: &Verdict{Winner: "small24"}}
	r := NewRouter(fast, deep, nil) // memory disabled
	out, err := r.Route("explain the algorithm", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Escalated || out.SeamID != 0 {
		t.Errorf("nil limpha: escalate but no seam id, got %+v", out)
	}
}

func TestRouterEscalatesOnInvalidConfidence(t *testing.T) {
	lc := newRouterLimpha(t)
	fast := &fakeBody{name: "nemo12", answer: "bad metric", confidence: math.NaN()}
	deep := &fakeBody{name: "small24", answer: "stable answer", verdict: &Verdict{Winner: "small24"}}
	r := NewRouter(fast, deep, lc)
	out, err := r.Route("short prompt", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Escalated || out.Reason != "low_confidence" || out.Body != "small24" {
		t.Errorf("invalid confidence must escalate as low confidence: %+v", out)
	}
}

func TestRouterIgnoresUnknownVerdictWinner(t *testing.T) {
	lc := newRouterLimpha(t)
	fast := &fakeBody{name: "nemo12", answer: "fast", confidence: 0.1}
	deep := &fakeBody{name: "small24", answer: "deep",
		verdict: &Verdict{Agreement: 2, Tension: -1, Winner: "phantom-body"}}
	r := NewRouter(fast, deep, lc)
	out, err := r.Route("ambiguous", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if out.Body != "small24" || out.Answer != "deep" {
		t.Errorf("unknown winner must fall back to deep body: %+v", out)
	}
	rs, _ := lc.RecentSeams(1)
	if len(rs) != 1 {
		t.Fatalf("want seam, got %d", len(rs))
	}
	if rs[0]["winner"] != "small24" || !approx(rs[0]["agreement"].(float64), 1.0) ||
		!approx(rs[0]["tension"].(float64), 0.0) {
		t.Errorf("winner/metrics not sanitized: %v", rs[0])
	}
}

func TestRouterAsyncMemoryQueuesConversationAndSeam(t *testing.T) {
	lc := newRouterLimpha(t)
	lc.StartAsync(8)
	fast := &fakeBody{name: "nemo12", answer: "unsure", confidence: 0.2}
	deep := &fakeBody{name: "small24", answer: "deep answer",
		verdict: &Verdict{Agreement: 0.4, Tension: 0.6, Winner: "small24"}}
	r := NewRouter(fast, deep, lc)
	r.AsyncMemory = true
	out, err := r.Route("what is the architecture?", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Escalated || out.SeamID != 0 {
		t.Errorf("async mode escalates but does not return immediate seam id: %+v", out)
	}
	lc.StopAsync()
	s, _ := lc.Stats()
	if s["total_conversations"].(int64) != 1 || s["total_seams"].(int64) != 1 {
		t.Fatalf("want queued 1 conv / 1 seam, got %v / %v", s["total_conversations"], s["total_seams"])
	}
	rs, _ := lc.RecentSeams(1)
	if len(rs) != 1 || rs[0]["conversation_id"] == nil {
		t.Fatalf("async router seam must be linked: %v", rs)
	}
}

func TestRouterSyncMemoryFailureVisibleAndSkipsOrphanSeam(t *testing.T) {
	lc := newRouterLimpha(t)
	if _, err := lc.db.Exec(`CREATE TRIGGER fail_router_conv BEFORE INSERT ON conversations BEGIN
		SELECT RAISE(FAIL, 'forced conversation write failure');
	END;`); err != nil {
		t.Fatal(err)
	}
	fast := &fakeBody{name: "nemo12", answer: "unsure", confidence: 0.2}
	deep := &fakeBody{name: "small24", answer: "deep answer",
		verdict: &Verdict{Agreement: 0.4, Tension: 0.6, Winner: "small24"}}
	r := NewRouter(fast, deep, lc)
	out, err := r.Route("what is the architecture?", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if out.Answer != "deep answer" || !out.Escalated {
		t.Fatalf("dialogue should remain fail-soft, got %+v", out)
	}
	if out.SeamID != 0 || out.Trace.MemoryStatus != "failed" ||
		!strings.Contains(out.Trace.MemoryError, "forced conversation write failure") {
		t.Fatalf("memory failure must be visible and seam skipped: %+v", out.Trace)
	}
	s, _ := lc.Stats()
	if s["total_conversations"].(int64) != 0 || s["total_seams"].(int64) != 0 {
		t.Fatalf("failed conversation must not leave orphan seam, stats=%v", s)
	}
	if s["memory_conversation_failures"].(int64) != 1 ||
		!strings.Contains(s["last_memory_error"].(string), "forced conversation write failure") {
		t.Fatalf("stats must expose limpha failure, stats=%v", s)
	}
}

func TestRouterAsyncMemoryFailureRecordedAfterDrain(t *testing.T) {
	lc := newRouterLimpha(t)
	if _, err := lc.db.Exec(`CREATE TRIGGER fail_router_async_conv BEFORE INSERT ON conversations BEGIN
		SELECT RAISE(FAIL, 'forced async conversation write failure');
	END;`); err != nil {
		t.Fatal(err)
	}
	lc.StartAsync(8)
	fast := &fakeBody{name: "nemo12", answer: "unsure", confidence: 0.2}
	deep := &fakeBody{name: "small24", answer: "deep answer",
		verdict: &Verdict{Agreement: 0.4, Tension: 0.6, Winner: "small24"}}
	r := NewRouter(fast, deep, lc)
	r.AsyncMemory = true
	out, err := r.Route("what is the architecture?", LimphaState{})
	if err != nil {
		t.Fatal(err)
	}
	if out.Trace.MemoryStatus != "queued" || out.SeamID != 0 {
		t.Fatalf("async turn should report queued before drain, got %+v", out.Trace)
	}
	lc.StopAsync()
	s, _ := lc.Stats()
	if s["total_conversations"].(int64) != 0 || s["total_seams"].(int64) != 0 {
		t.Fatalf("failed async conversation must not leave orphan seam, stats=%v", s)
	}
	if s["memory_conversation_failures"].(int64) != 1 ||
		!strings.Contains(s["last_memory_error"].(string), "forced async conversation write failure") {
		t.Fatalf("async stats must expose limpha failure, stats=%v", s)
	}
}

func TestRouterSingleResidentClosesInactiveBodies(t *testing.T) {
	lc := newRouterLimpha(t)
	fast := &closableFakeBody{fakeBody: fakeBody{name: "nemo12", answer: "unsure", confidence: 0.2}}
	deep := &closableFakeBody{fakeBody: fakeBody{name: "small24", answer: "deep",
		verdict: &Verdict{Winner: "small24"}}}
	r := NewRouter(fast, deep, lc)
	if _, err := r.Route("ambiguous", LimphaState{}); err != nil {
		t.Fatal(err)
	}
	if fast.closes != 1 {
		t.Fatalf("fast body must be closed before deep escalation, got %d", fast.closes)
	}
	if deep.closes != 1 {
		t.Fatalf("deep body must be closed before fast entry, got %d", deep.closes)
	}

	fast.closes, deep.closes = 0, 0
	r.SingleResident = false
	if _, err := r.Route("ambiguous", LimphaState{}); err != nil {
		t.Fatal(err)
	}
	if fast.closes != 0 || deep.closes != 0 {
		t.Fatalf("SingleResident=false must not close bodies, fast=%d deep=%d", fast.closes, deep.closes)
	}
}
