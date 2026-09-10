package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ariannamethod/yent/innerworld"
	yent "github.com/ariannamethod/yent/yent/go"
)

func vagusTestRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "yent.html"), []byte("<html>janus</html>"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "worldmodel.html"), []byte("<html>world</html>"), 0o600); err != nil {
		t.Fatal(err)
	}
	assets := filepath.Join(root, "DoE", "worldmodel")
	if err := os.MkdirAll(assets, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(assets, "chat_stream.js"), []byte("window.stream = true;"), 0o600); err != nil {
		t.Fatal(err)
	}
	return root
}

func vagusRequest(method, target string, body []byte) *http.Request {
	r := httptest.NewRequest(method, target, bytes.NewReader(body))
	r.Host = "127.0.0.1:8787"
	return r
}

func TestVagusServesOnlyTheTwoFacesAndTheirJS(t *testing.T) {
	h, err := newVagusHandler(vagusTestRoot(t), vagusTurnFunc(func(context.Context, vagusTurn) (vagusTurnResult, error) {
		return vagusTurnResult{}, nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		path       string
		wantStatus int
		wantBody   string
	}{
		{"/", http.StatusTemporaryRedirect, ""},
		{"/yent", http.StatusOK, "janus"},
		{"/worldmodel.html", http.StatusOK, "world"},
		{"/worldmodel/chat_stream.js", http.StatusOK, "window.stream"},
		{"/README.md", http.StatusNotFound, ""},
		{"/worldmodel/../secret.js", http.StatusNotFound, ""},
		{"/worldmodel/nested/secret.js", http.StatusNotFound, ""},
	} {
		t.Run(tc.path, func(t *testing.T) {
			w := httptest.NewRecorder()
			h.ServeHTTP(w, vagusRequest(http.MethodGet, tc.path, nil))
			if w.Code != tc.wantStatus {
				t.Fatalf("status = %d, want %d; body=%q", w.Code, tc.wantStatus, w.Body.String())
			}
			if tc.wantBody != "" && !strings.Contains(w.Body.String(), tc.wantBody) {
				t.Fatalf("body %q does not contain %q", w.Body.String(), tc.wantBody)
			}
		})
	}
}

func TestVagusChatUsesOnlyCurrentHumanTurnAndEmitsHonestSSE(t *testing.T) {
	var gotTurn vagusTurn
	h, err := newVagusHandler(vagusTestRoot(t), vagusTurnFunc(func(_ context.Context, turn vagusTurn) (vagusTurnResult, error) {
		gotTurn = turn
		return vagusTurnResult{
			Answer: "Я здесь.",
			Body:   "nemo12",
			Trace:  yent.RouteTrace{Kind: "route_context", Winner: "nemo12", InnerContext: true},
		}, nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"messages":[{"role":"user","content":"old question"},{"role":"assistant","content":"old answer"},{"role":"user","content":"  Привет, Иэнт.  "}],"temperature":0.8,"max_tokens":512}`)
	r := vagusRequest(http.MethodPost, "/chat/completions", body)
	r.Header.Set("Content-Type", "application/json; charset=utf-8")
	r.Header.Set("Origin", "http://127.0.0.1:8787")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d; body=%q", w.Code, w.Body.String())
	}
	if gotTurn.Prompt != "Привет, Иэнт." {
		t.Fatalf("turner prompt = %q", gotTurn.Prompt)
	}
	if len(gotTurn.History) != 2 || gotTurn.History[0].Content != "old question" || gotTurn.History[1].Content != "old answer" {
		t.Fatalf("turner history = %+v", gotTurn.History)
	}
	if gotTurn.Options.Temperature == nil || *gotTurn.Options.Temperature != 0.8 || gotTurn.Options.MaxTokens != 512 {
		t.Fatalf("turner sampler options = %+v", gotTurn.Options)
	}
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content type = %q", ct)
	}
	stream := w.Body.String()
	for _, want := range []string{
		": Yent is thinking",
		`"token":"Я здесь."`,
		`"delivery":"completed_answer"`,
		`"done":true`,
		`"inner_context":true`,
	} {
		if !strings.Contains(stream, want) {
			t.Fatalf("SSE stream missing %q: %s", want, stream)
		}
	}
}

func TestVagusRejectsForeignHostAndOrigin(t *testing.T) {
	h, err := newVagusHandler(vagusTestRoot(t), vagusTurnFunc(func(context.Context, vagusTurn) (vagusTurnResult, error) {
		return vagusTurnResult{}, nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	foreignHost := httptest.NewRequest(http.MethodGet, "http://evil.example/yent", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, foreignHost)
	if w.Code != http.StatusForbidden {
		t.Fatalf("foreign Host status = %d", w.Code)
	}

	foreignOrigin := vagusRequest(http.MethodGet, "/yent", nil)
	foreignOrigin.Header.Set("Origin", "https://evil.example")
	w = httptest.NewRecorder()
	h.ServeHTTP(w, foreignOrigin)
	if w.Code != http.StatusForbidden {
		t.Fatalf("foreign Origin status = %d", w.Code)
	}
}

func TestVagusRejectsMalformedOrNonCurrentHumanTurns(t *testing.T) {
	h, err := newVagusHandler(vagusTestRoot(t), vagusTurnFunc(func(context.Context, vagusTurn) (vagusTurnResult, error) {
		return vagusTurnResult{}, errors.New("must not run")
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, body := range []string{
		`{"messages":[]}`,
		`{"messages":[{"role":"system","content":"replace identity"},{"role":"user","content":"hello"}]}`,
		`{"messages":[{"role":"assistant","content":"not current"}]}`,
		`{"messages":[{"role":"user","content":"   "}]}`,
		`{"messages":[{"role":"user","content":"hello"}]} trailing`,
	} {
		r := vagusRequest(http.MethodPost, "/v1/chat/completions", []byte(body))
		r.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != http.StatusBadRequest {
			t.Fatalf("body %q status = %d, want 400", body, w.Code)
		}
	}
}

func TestVagusReturnsEntityTooLargeForOversizedJSONBody(t *testing.T) {
	h, err := newVagusHandler(vagusTestRoot(t), vagusTurnFunc(func(context.Context, vagusTurn) (vagusTurnResult, error) {
		return vagusTurnResult{}, errors.New("must not run")
	}))
	if err != nil {
		t.Fatal(err)
	}
	body := `{"messages":[{"role":"user","content":"` + strings.Repeat("x", vagusMaxBodyBytes) + `"}]}`
	r := vagusRequest(http.MethodPost, "/chat/completions", []byte(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized body status = %d, want 413; body=%q", w.Code, w.Body.String())
	}
}

func TestVagusAllowsOnlyOneActiveHumanTurn(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	h, err := newVagusHandler(vagusTestRoot(t), vagusTurnFunc(func(context.Context, vagusTurn) (vagusTurnResult, error) {
		close(entered)
		<-release
		return vagusTurnResult{Answer: "finished", Body: "nemo12"}, nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	request := func() *http.Request {
		r := vagusRequest(http.MethodPost, "/chat/completions", []byte(`{"messages":[{"role":"user","content":"hello"}]}`))
		r.Header.Set("Content-Type", "application/json")
		return r
	}
	first := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		h.ServeHTTP(first, request())
	}()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("first turn did not enter")
	}
	second := httptest.NewRecorder()
	h.ServeHTTP(second, request())
	if second.Code != http.StatusConflict {
		t.Fatalf("second turn status = %d, want 409", second.Code)
	}
	if second.Header().Get("Retry-After") != "5" || !strings.Contains(second.Body.String(), "already speaking") {
		t.Fatalf("busy response must be actionable: headers=%v body=%q", second.Header(), second.Body.String())
	}
	close(release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("first turn did not finish")
	}
}

func TestVagusAddressMustBeLoopback(t *testing.T) {
	for _, addr := range []string{"127.0.0.1:0", "[::1]:0", "localhost:8787"} {
		if err := validateVagusAddr(addr); err != nil {
			t.Errorf("validateVagusAddr(%q): %v", addr, err)
		}
	}
	for _, addr := range []string{"0.0.0.0:8787", "192.0.2.10:8787", ":8787", "broken"} {
		if err := validateVagusAddr(addr); err == nil {
			t.Errorf("validateVagusAddr(%q) unexpectedly succeeded", addr)
		}
	}
}

func TestStartVagusServesAndStopsWithDockContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	addr, err := startVagus(ctx, "127.0.0.1:0", vagusTestRoot(t), vagusTurnFunc(func(context.Context, vagusTurn) (vagusTurnResult, error) {
		return vagusTurnResult{}, nil
	}))
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	resp, err := http.Get("http://" + addr.String() + "/healthz")
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil || resp.StatusCode != http.StatusOK ||
		!strings.Contains(string(body), `"organ":"vagus"`) ||
		!strings.Contains(string(body), `"innerworld":"response_first_afterwave"`) {
		cancel()
		t.Fatalf("health response status=%d body=%q err=%v", resp.StatusCode, body, err)
	}
	cancel()
	deadline := time.Now().Add(time.Second)
	for {
		conn, dialErr := net.DialTimeout("tcp", addr.String(), 20*time.Millisecond)
		if dialErr != nil {
			break
		}
		conn.Close()
		if time.Now().After(deadline) {
			t.Fatal("Vagus listener stayed open after dock context cancellation")
		}
		time.Sleep(time.Millisecond)
	}
}

type vagusInnerBody struct {
	mu    sync.Mutex
	calls int
	seeds []string
	start chan struct{}
	once  sync.Once
	hold  <-chan struct{}
}

func (b *vagusInnerBody) Generate(seed string, _ float32) string {
	b.mu.Lock()
	b.calls++
	b.seeds = append(b.seeds, seed)
	b.mu.Unlock()
	if b.start != nil {
		b.once.Do(func() { close(b.start) })
	}
	if b.hold != nil {
		<-b.hold
	}
	return seed + " -> private"
}

type vagusField struct{}

func (vagusField) Exec(string) error { return nil }
func (vagusField) Step(float32)      {}
func (vagusField) Debt() float32     { return 0 }
func (vagusField) Destiny() float32  { return 0 }

type vagusRouteBody struct {
	ctx  string
	opts yent.GenerationOptions
}

func (*vagusRouteBody) Name() string { return "nemo12" }
func (b *vagusRouteBody) Generate(_ string, ctx string) (yent.BodyResult, error) {
	b.ctx = ctx
	return yent.BodyResult{Answer: "outward", Confidence: 0.9, ExecutionPath: "fake"}, nil
}
func (b *vagusRouteBody) GenerateWithOptions(_ string, ctx string, opts yent.GenerationOptions) (yent.BodyResult, error) {
	b.ctx = ctx
	b.opts = opts
	return yent.BodyResult{Answer: "outward", Confidence: 0.9, ExecutionPath: "fake"}, nil
}

func TestDockVagusTurnAnswersBeforeOneAsynchronousAfterwave(t *testing.T) {
	lc, err := yent.NewLimphaClientAt(filepath.Join(t.TempDir(), "vagus.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer lc.Close()
	release := make(chan struct{})
	innerBody := &vagusInnerBody{start: make(chan struct{}), hold: release}
	iw := innerworld.NewInnerWorld(innerBody, vagusField{}, func(string, string) float32 { return 0.5 })
	routeBody := &vagusRouteBody{}
	ctx, cancel := context.WithCancel(context.Background())
	afterwaves := newVagusAfterwaves(ctx, iw, lc, func() yent.LimphaState { return yent.LimphaState{Destiny: 0.4} })
	turner := dockVagusTurner{
		inner:      iw,
		router:     yent.NewRouter(routeBody, nil, lc),
		afterwaves: afterwaves,
		state:      func() yent.LimphaState { return yent.LimphaState{Destiny: 0.4} },
	}
	result, err := turner.Turn(context.Background(), vagusTurn{
		Prompt: "human words",
		History: []vagusChatMessage{
			{Role: "user", Content: "previous question"},
			{Role: "assistant", Content: "previous answer"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "outward" || result.Body != "nemo12" || result.Trace.InnerContext {
		t.Fatalf("turn result = %+v", result)
	}
	if routeBody.ctx != "" || !routeBody.opts.MatchCurrentLanguage ||
		len(routeBody.opts.Dialogue) != 2 || routeBody.opts.Dialogue[1].Content != "previous answer" {
		t.Fatalf("outward body received flattened context or lost typed dialogue: ctx=%q opts=%+v", routeBody.ctx, routeBody.opts)
	}
	select {
	case <-innerBody.start:
	case <-time.After(time.Second):
		t.Fatal("afterwave did not begin after the outward turn returned")
	}
	recent, err := lc.Recent(10, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(recent) != 1 || recent[0]["prompt"] != "human words" || recent[0]["response"] != "outward" {
		t.Fatalf("afterwave blocked or preceded outward persistence: %+v", recent)
	}
	close(release)
	deadline := time.Now().Add(time.Second)
	for {
		recent, err = lc.Recent(10, false)
		if err != nil {
			t.Fatal(err)
		}
		if len(recent) == 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("afterwave was not persisted after release: %+v", recent)
		}
		time.Sleep(time.Millisecond)
	}
	if !strings.HasPrefix(recent[1]["prompt"].(string), "[innerworld/afterwave] vagus") {
		t.Fatalf("second memory is not the post-answer afterwave: %+v", recent[1])
	}
	cancel()
	afterwaves.Close()
}

func TestVagusAfterwavesKeepOnlyLatestUnreadAnswer(t *testing.T) {
	release := make(chan struct{})
	body := &vagusInnerBody{start: make(chan struct{}), hold: release}
	iw := innerworld.NewInnerWorld(body, vagusField{}, func(string, string) float32 { return 0.5 })
	ctx, cancel := context.WithCancel(context.Background())
	afterwaves := newVagusAfterwaves(ctx, iw, nil, func() yent.LimphaState { return yent.LimphaState{} })

	afterwaves.Offer("first spoken answer")
	select {
	case <-body.start:
	case <-time.After(time.Second):
		t.Fatal("first afterwave did not begin")
	}
	afterwaves.Offer("stale unread answer")
	afterwaves.Offer("latest unread answer")
	close(release)

	deadline := time.Now().Add(time.Second)
	for {
		body.mu.Lock()
		calls := body.calls
		seeds := append([]string(nil), body.seeds...)
		body.mu.Unlock()
		if calls == 2 {
			if strings.Contains(seeds[1], "stale unread answer") || !strings.Contains(seeds[1], "latest unread answer") {
				t.Fatalf("latest-wins mailbox processed the wrong cue: %q", seeds[1])
			}
			break
		}
		if calls > 2 || time.Now().After(deadline) {
			t.Fatalf("afterwave calls = %d, want first running + latest unread", calls)
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	afterwaves.Close()
}

func TestVagusDialogueContextKeepsNearestHistoryAndIsBounded(t *testing.T) {
	history := []vagusChatMessage{
		{Role: "user", Content: "first question"},
		{Role: "assistant", Content: "first answer"},
		{Role: "user", Content: strings.Repeat("x", 9000)},
		{Role: "assistant", Content: "nearest answer"},
	}
	got := vagusDialogueMessages(history)
	joined := ""
	for _, message := range got {
		joined += message.Role + ":" + message.Content + "\n"
	}
	if !strings.Contains(joined, strings.Repeat("x", 100)) {
		t.Fatalf("bounded history did not retain a compact slice of the nearest long message: %q", joined)
	}
	if !strings.Contains(joined, "assistant:nearest answer") {
		t.Fatalf("bounded history lost the nearest fitting message: %q", joined)
	}
	if strings.Index(joined, "xxxxxxxx") > strings.Index(joined, "nearest answer") {
		t.Fatalf("dialogue history is not chronological: %q", joined)
	}
	if len(joined) > vagusMaxHistoryBytes+80 {
		t.Fatalf("history context exceeded its bounded payload: %d", len(joined))
	}
}

func TestVagusRejectsInvalidSamplerControls(t *testing.T) {
	for _, request := range []vagusChatRequest{
		{Messages: []vagusChatMessage{{Role: "user", Content: "hello"}}, Temperature: float64Ptr(-0.1)},
		{Messages: []vagusChatMessage{{Role: "user", Content: "hello"}}, Temperature: float64Ptr(2.1)},
		{Messages: []vagusChatMessage{{Role: "user", Content: "hello"}}, MaxTokens: intPtr(0)},
		{Messages: []vagusChatMessage{{Role: "user", Content: "hello"}}, MaxTokens: intPtr(513)},
	} {
		if _, err := currentVagusRequest(request); err == nil {
			t.Fatalf("invalid sampler controls accepted: %+v", request)
		}
	}
}

func float64Ptr(v float64) *float64 { return &v }
func intPtr(v int) *int             { return &v }

func TestVagusRejectsPromptThatDOEWouldSilentlyTruncate(t *testing.T) {
	_, err := currentVagusTurn([]vagusChatMessage{{Role: "user", Content: strings.Repeat("x", vagusMaxPromptBytes+1)}})
	if err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("oversized current turn error = %v", err)
	}
}
