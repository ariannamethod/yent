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
	"unicode/utf8"

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
	if err != nil || resp.StatusCode != http.StatusOK || !strings.Contains(string(body), `"organ":"vagus"`) {
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
}

func (b *vagusInnerBody) Generate(seed string, _ float32) string {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.calls++
	return seed + " -> private"
}

type vagusField struct{}

func (vagusField) Exec(string) error { return nil }
func (vagusField) Step(float32)      {}
func (vagusField) Debt() float32     { return 0 }
func (vagusField) Destiny() float32  { return 0 }

type vagusRouteBody struct {
	ctx string
}

func (*vagusRouteBody) Name() string { return "nemo12" }
func (b *vagusRouteBody) Generate(_ string, ctx string) (yent.BodyResult, error) {
	b.ctx = ctx
	return yent.BodyResult{Answer: "outward", Confidence: 0.9, ExecutionPath: "fake"}, nil
}

func TestDockVagusTurnKeepsPrivateThoughtAndOutwardSpeechSeparate(t *testing.T) {
	lc, err := yent.NewLimphaClientAt(filepath.Join(t.TempDir(), "vagus.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer lc.Close()
	innerBody := &vagusInnerBody{}
	iw := innerworld.NewInnerWorld(innerBody, vagusField{}, func(string, string) float32 { return 0.5 })
	routeBody := &vagusRouteBody{}
	turner := dockVagusTurner{
		inner:  iw,
		router: yent.NewRouter(routeBody, nil, lc),
		limpha: lc,
		state:  func() yent.LimphaState { return yent.LimphaState{Destiny: 0.4} },
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
	if result.Answer != "outward" || result.Body != "nemo12" || !result.Trace.InnerContext {
		t.Fatalf("turn result = %+v", result)
	}
	if !strings.Contains(routeBody.ctx, "private current inner reflection") ||
		!strings.Contains(routeBody.ctx, "[assistant]: previous answer") ||
		strings.Contains(routeBody.ctx, "[innerworld/human_turn]") {
		t.Fatalf("outward body context did not receive the private pressure cleanly: %q", routeBody.ctx)
	}
	recent, err := lc.Recent(10, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(recent) != 2 {
		t.Fatalf("want one private and one outward memory, got %d: %+v", len(recent), recent)
	}
	if !strings.HasPrefix(recent[0]["prompt"].(string), "[innerworld/human_turn] vagus") {
		t.Fatalf("first memory is not the private reflection: %+v", recent[0])
	}
	if recent[1]["prompt"] != "human words" || recent[1]["response"] != "outward" {
		t.Fatalf("second memory is not the outward turn: %+v", recent[1])
	}
}

func TestVagusDialogueContextKeepsNearestHistoryAndIsBounded(t *testing.T) {
	history := []vagusChatMessage{
		{Role: "user", Content: "first question"},
		{Role: "assistant", Content: "first answer"},
		{Role: "user", Content: strings.Repeat("x", 9000)},
		{Role: "assistant", Content: "nearest answer"},
	}
	got := vagusDialogueContext(history)
	if strings.Contains(got, "first question") || !strings.Contains(got, strings.Repeat("x", 100)) {
		t.Fatalf("bounded history did not retain the nearest messages first")
	}
	if !strings.Contains(got, "[assistant]: nearest answer") {
		t.Fatalf("bounded history lost the nearest fitting message: %q", got)
	}
	if len(got) > vagusMaxHistoryBytes+180 {
		t.Fatalf("history context exceeded its bounded payload: %d", len(got))
	}
}

func TestVagusRejectsPromptThatDOEWouldSilentlyTruncate(t *testing.T) {
	_, err := currentVagusTurn([]vagusChatMessage{{Role: "user", Content: strings.Repeat("x", vagusMaxPromptBytes+1)}})
	if err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("oversized current turn error = %v", err)
	}
}

func TestVagusInnerContextIsBoundedAndUsesLastCircle(t *testing.T) {
	reflection := innerworld.Reflection{Circles: []innerworld.Circle{
		{Text: "first"},
		{Text: strings.Repeat("последний ", 100)},
	}}
	got := vagusInnerContext(reflection)
	if strings.Contains(got, "first") || !strings.Contains(got, "последний") {
		t.Fatalf("inner context did not select the last circle: %q", got)
	}
	prefix := "Your private current inner reflection, not another speaker and not text to quote verbatim: "
	payload := strings.TrimPrefix(got, prefix)
	if len(payload) > vagusMaxInnerBytes || !utf8.ValidString(payload) {
		t.Fatalf("inner context payload is not a valid bounded string: bytes=%d", len(payload))
	}
}
